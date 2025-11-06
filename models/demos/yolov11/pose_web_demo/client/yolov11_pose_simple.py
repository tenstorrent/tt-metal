# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import logging
import time

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def plot_pose_cv2(self, bgr_img, detections):
        """Plot pose keypoints and skeleton on image"""
        img = np.copy(bgr_img)
        height, width = img.shape[:2]

        # Skeleton connections (COCO format)
        skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        # Colors for keypoints and skeleton
        keypoint_color = (0, 255, 0)  # Green
        skeleton_color = (255, 0, 0)  # Red

        for detection in detections:
            # Extract bbox and keypoints
            bbox = detection[:4]  # x, y, w, h
            keypoints = detection[5:]  # 51 values (17 keypoints x 3)

            # Draw bounding box
            x, y, w, h = bbox
            x1, y1 = int(x * width), int(y * height)
            x2, y2 = int((x + w) * width), int((y + h) * height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Reshape keypoints to (17, 3) - x, y, confidence
            keypoints = np.array(keypoints).reshape(17, 3)

            # Draw keypoints
            for i, (kx, ky, conf) in enumerate(keypoints):
                if conf > 0.5:  # Only draw confident keypoints
                    kx_pixel = int(kx * width)
                    ky_pixel = int(ky * height)
                    cv2.circle(img, (kx_pixel, ky_pixel), 3, keypoint_color, -1)

            # Draw skeleton
            for connection in skeleton:
                start_idx, end_idx = connection
                if keypoints[start_idx, 2] > 0.5 and keypoints[end_idx, 2] > 0.5:
                    start_x = int(keypoints[start_idx, 0] * width)
                    start_y = int(keypoints[start_idx, 1] * height)
                    end_x = int(keypoints[end_idx, 0] * width)
                    end_y = int(keypoints[end_idx, 1] * height)
                    cv2.line(img, (start_x, start_y), (end_x, end_y), skeleton_color, 2)

        return img

    def recv(self, frame):
        t0 = time.time()

        # Convert frame to PIL image and resize
        pil_image = frame.to_image()
        pil_image = pil_image.resize((640, 640))  # Resize to target dimensions
        t1 = time.time()

        # Save image as JPEG in-memory with optimized settings
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85, optimize=True)
        byte_im = buf.getvalue()
        file = {"file": byte_im}

        # Parse API URL once at the class level for efficiency
        if not hasattr(self, "api_url"):
            parser = argparse.ArgumentParser(description="YOLOv11 Pose script")
            parser.add_argument("--api-url", type=str, required=True, help="URL for the pose detection API")
            args = parser.parse_args()
            self.api_url = args.api_url

        url = f"{self.api_url}/pose_estimation_v2"

        try:
            # Use a persistent session for multiple requests
            with requests.Session() as session:
                # Post request with a timeout
                response = session.post(url, files=file, timeout=5)

                # Check if response is successful
                if response.status_code == 200:
                    # Parse JSON response
                    output = response.json()
                    print(f"DEBUG: Server returned: {output.keys() if isinstance(output, dict) else type(output)}")

                    if "raw_output" in output:
                        # Server returned raw model output for debugging
                        print(f"DEBUG: Raw output shape: {output['shape']}")
                        print(
                            f"DEBUG: Raw output sample: {output['raw_output'][:2] if output['raw_output'] else 'empty'}"
                        )
                        # For now, return empty detections to avoid client crash
                        output = {"detections": []}
                    elif isinstance(output, list):
                        # Server returned processed detections
                        output = {"detections": output}
                    elif isinstance(output, dict) and "detections" in output:
                        # Server returned properly formatted response: {"detections": [...]}
                        pass  # output is already in the correct format
                    else:
                        print(f"DEBUG: Unexpected response format: {output}")
                        return None
                else:
                    print(f"Request failed with status code {response.status_code}")
                    print(f"Response text: {response.text}")
                    return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

        t3 = time.time()
        # Convert frame to ndarray and perform post-processing
        bgr_image = frame.to_ndarray(format="bgr24")

        # Process detections
        detections = output.get("detections", [])
        image_final = self.plot_pose_cv2(bgr_image, detections)

        t4 = time.time()
        logging.info(
            f" IMG-IN | WH | Post | Total time: {(t1-t0):.3f} | {(t3-t1):.3f} | {(t4-t3):.3f} || {(t4-t0):.3f} "
        )

        return av.VideoFrame.from_ndarray(image_final, format="bgr24")


st.title("YOLOv11 Pose Detection Demo")

webrtc_streamer(
    key="pose-detection",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"min": 320, "ideal": 640, "max": 1280},
            "height": {"min": 320, "ideal": 640, "max": 1280},
            "frameRate": {"min": 1, "ideal": 30, "max": 60},
        }
    },
)
