# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Streamlit client for YuNet face detection with WebRTC support.

This client connects to a FastAPI server running YuNet inference
on Tenstorrent hardware for real-time face detection.

Usage:
    streamlit run yunet_streamlit.py --server.port 8501
"""

import io
import logging
import time

import av
import cv2
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Keypoint colors (left eye, right eye, nose, left mouth, right mouth)
KP_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


class FaceDetectionProcessor(VideoProcessorBase):
    """Video processor for real-time face detection."""

    def __init__(self):
        self.frame_count = 0
        self.api_url = None
        self.input_size = 640
        self.conf_thresh = 0.35

    def set_api_url(self, url):
        self.api_url = url

    def set_params(self, input_size, conf_thresh):
        self.input_size = input_size
        self.conf_thresh = conf_thresh

    def draw_detections(self, image, detections):
        """Draw face boxes and keypoints on image."""
        img = image.copy()
        height, width = img.shape[:2]

        scale = max(width, height) / 640
        thickness = max(1, int(2 * scale))
        font_scale = max(0.3, 0.4 * scale)
        kp_radius = max(2, int(3 * scale))

        for det in detections:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(det["box"][0] * width)
            y1 = int(det["box"][1] * height)
            x2 = int(det["box"][2] * width)
            y2 = int(det["box"][3] * height)
            conf = det["conf"]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

            # Draw confidence
            label = f"{conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

            # Draw keypoints
            for i, kp in enumerate(det["keypoints"]):
                kx = int(kp[0] * width)
                ky = int(kp[1] * height)
                cv2.circle(img, (kx, ky), kp_radius, KP_COLORS[i], -1)

        return img

    def recv(self, frame):
        """Process incoming video frame."""
        t0 = time.time()

        # Convert frame to PIL image and resize
        pil_image = frame.to_image()
        pil_image = pil_image.resize((self.input_size, self.input_size))
        t1 = time.time()

        # Save image as JPEG in-memory
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85, optimize=True)
        byte_im = buf.getvalue()
        file = {"file": byte_im}

        if self.api_url is None:
            # Return original frame if no API configured
            return frame

        url = f"{self.api_url}/facedetection"

        try:
            # Post request with timeout
            response = requests.post(
                url, files=file, params={"input_size": self.input_size, "conf_thresh": self.conf_thresh}, timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                detections = result.get("detections", [])
                inference_time = result.get("inference_time_ms", 0)
            else:
                logging.error(f"Request failed with status code {response.status_code}")
                return frame

        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return frame

        t2 = time.time()

        # Draw detections on original frame
        bgr_image = frame.to_ndarray(format="bgr24")
        result_image = self.draw_detections(bgr_image, detections)

        t3 = time.time()
        logging.info(
            f"Preprocess: {(t1-t0)*1000:.1f}ms | "
            f"Inference: {inference_time:.1f}ms | "
            f"Postprocess: {(t3-t2)*1000:.1f}ms | "
            f"Total: {(t3-t0)*1000:.1f}ms | "
            f"Faces: {len(detections)}"
        )

        return av.VideoFrame.from_ndarray(result_image, format="bgr24")


def main():
    st.set_page_config(
        page_title="YuNet Face Detection - Tenstorrent",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 YuNet Face Detection")
    st.markdown("**Real-time face detection powered by Tenstorrent AI Hardware**")
    st.markdown("---")

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")

    api_url = st.sidebar.text_input(
        "API Server URL", value="http://localhost:8000", help="URL of the YuNet FastAPI server"
    )

    input_size = st.sidebar.selectbox(
        "Input Size", options=[640, 320], index=0, help="Model input resolution. 640 is more accurate, 320 is faster."
    )

    conf_thresh = 0.35  # Fixed confidence threshold

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.markdown(f"- **Model:** YuNet")
    st.sidebar.markdown(f"- **Input Size:** {input_size}x{input_size}")
    st.sidebar.markdown(f"- **Output:** Face boxes + 5 keypoints")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Keypoint Legend")
    st.sidebar.markdown("- 🔵 Left Eye")
    st.sidebar.markdown("- 🟢 Right Eye")
    st.sidebar.markdown("- 🔴 Nose")
    st.sidebar.markdown("- 🟡 Left Mouth Corner")
    st.sidebar.markdown("- 🟣 Right Mouth Corner")

    # Main content
    st.subheader("📹 Live Camera Feed")

    # Create processor with settings
    def processor_factory():
        processor = FaceDetectionProcessor()
        processor.set_api_url(api_url)
        processor.set_params(input_size, conf_thresh)
        return processor

    # RTC configuration for remote access
    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
    }

    webrtc_streamer(
        key="yunet-face-detection",
        video_processor_factory=processor_factory,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {
                "width": {"min": 320, "ideal": 640, "max": 1280},
                "height": {"min": 320, "ideal": 480, "max": 720},
                "frameRate": {"min": 1, "ideal": 30, "max": 60},
            },
            "audio": False,
        },
        async_processing=True,
    )


if __name__ == "__main__":
    main()
