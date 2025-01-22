import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the YOLO model
@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

yolo_model = load_yolo_model('F:/SOLUTION_O7/NUMBER_Plate/YOLO_V8_Model/best.pt')

# Load the CNN model
@st.cache_resource
def load_cnn_model(model_path):
    model = load_model(model_path)
    return model

cnn_model = load_cnn_model('CarNumberPlate_11.h5')

# Configure pytesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Streamlit UI
st.title('Car Number Plate Detection and Recognition')

option = st.selectbox("Choose input type", ("Image", "Webcam", "Video"))

# Initialize or clear extracted numbers list
if "extracted_numbers" not in st.session_state:
    st.session_state.extracted_numbers = []

def extract_text_with_tesseract(image, box):
    # Extract ROI based on the bounding box
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    roi = image[y1:y2, x1:x2]
    
    # Convert to grayscale as Tesseract works better on grayscale images
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to extract text from the ROI
    extracted_text = pytesseract.image_to_string(roi_gray, config='--psm 8').strip()  # psm 8 is for single word/line OCR
    return extracted_text

# Clear button to reset the extracted numbers
if st.button("Clear History"):
    st.session_state.extracted_numbers = []

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        results = yolo_model(image_cv2)
        
        for result in results:
            for box in result.boxes:
                text = extract_text_with_tesseract(image_cv2, box)
                if text:
                    st.session_state.extracted_numbers.insert(0, text)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    label = f'{text} ({box.conf[0].cpu().item():.2f})'
                    cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(image_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        result_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        st.image(result_image, caption='Detected Number Plates', use_column_width=True)
        
        st.sidebar.write("Extracted Number Plates:")
        for text in st.session_state.extracted_numbers[:5]:
            st.sidebar.write(text)

elif option == "Webcam":
    st.warning("Webcam support is experimental and may require additional permissions and setup.")
    run_webcam = st.button("Start Webcam")
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo_model(frame)
            for result in results:
                for box in result.boxes:
                    text = extract_text_with_tesseract(frame, box)
                    if text:
                        st.session_state.extracted_numbers.insert(0, text)
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        label = f'{text} ({box.conf[0].cpu().item():.2f})'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, text, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame, channels="BGR")
            st.sidebar.write("Extracted Number Plates:")
            for text in st.session_state.extracted_numbers[:5]:
                st.sidebar.write(f"Detected: {text}")
        cap.release()

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        video_path = uploaded_video.name
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo_model(frame)
            for result in results:
                for box in result.boxes:
                    text = extract_text_with_tesseract(frame, box)
                    if text:
                        st.session_state.extracted_numbers.insert(0, text)
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        label = f'{text} ({box.conf[0].cpu().item():.2f})'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, text, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame, channels="BGR")
            st.sidebar.write("Extracted Number Plates:")
            for text in st.session_state.extracted_numbers[:5]:
                st.sidebar.write(f"Detected: {text}")
        cap.release()
