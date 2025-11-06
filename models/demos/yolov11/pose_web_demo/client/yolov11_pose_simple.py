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

        # Skeleton connections (COCO format) - 17 keypoints (0-16)
        skeleton = [
            [15, 13],  # left ankle -> left knee
            [13, 11],  # left knee -> left hip
            [16, 14],  # right ankle -> right knee
            [14, 12],  # right knee -> right hip
            [11, 12],  # left hip -> right hip
            [5, 11],  # left shoulder -> left hip
            [6, 12],  # right shoulder -> right hip
            [5, 6],  # left shoulder -> right shoulder
            [5, 7],  # left shoulder -> left elbow
            [6, 8],  # right shoulder -> right elbow
            [7, 9],  # left elbow -> left wrist
            [8, 10],  # right elbow -> right wrist
            [1, 2],  # left eye -> right eye
            [0, 1],  # nose -> left eye
            [0, 2],  # nose -> right eye
            [1, 3],  # left eye -> left ear
            [2, 4],  # right eye -> right ear
            [3, 5],  # left ear -> left shoulder
            [4, 6],  # right ear -> right shoulder
        ]

        # Colors for keypoints and skeleton
        keypoint_color = (0, 255, 0)  # Green
        skeleton_color = (255, 0, 0)  # Red

        for detection in detections:
            # Extract bbox and keypoints
            bbox = detection[:4]  # x, y, w, h (normalized 0-1)
            confidence = detection[4]
            keypoints = detection[5:]  # 51 values (17 keypoints x 3)

            # Debug: print received coordinates
            if detections.index(detection) == 0:  # Only print first detection
                print(f"DEBUG: Client received bbox: {bbox} conf={confidence:.3f}")

            # Draw bounding box
            x, y, w, h = bbox
            x1, y1 = int(x * width), int(y * height)
            x2, y2 = int((x + w) * width), int((y + h) * height)

            # Debug: show computed coordinates
            if detections.index(detection) == 0:
                print(f"DEBUG: Client computed bbox pixels: ({x1}, {y1}) to ({x2}, {y2}) on {width}x{height} frame")

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Reshape keypoints to (17, 3) - x, y, confidence
            keypoints = np.array(keypoints).reshape(17, 3)

            # Draw keypoints
            for i, (kx, ky, conf) in enumerate(keypoints):
                if conf > 0.3:  # Lower threshold for keypoint visibility
                    kx_pixel = int(kx * width)
                    ky_pixel = int(ky * height)
                    cv2.circle(img, (kx_pixel, ky_pixel), 3, keypoint_color, -1)

            # Draw skeleton
            for connection in skeleton:
                start_idx, end_idx = connection
                if keypoints[start_idx, 2] > 0.3 and keypoints[end_idx, 2] > 0.3:
                    start_x = int(keypoints[start_idx, 0] * width)
                    start_y = int(keypoints[start_idx, 1] * height)
                    end_x = int(keypoints[end_idx, 0] * width)
                    end_y = int(keypoints[end_idx, 1] * height)
                    cv2.line(img, (start_x, start_y), (end_x, end_y), skeleton_color, 2)

        return img

    def recv(self, frame):
        t0 = time.time()

        # Convert frame to PIL image and resize to 640x640 for processing
        pil_image = frame.to_image()
        pil_image_resized = pil_image.resize((640, 640))  # Resize to target dimensions
        t1 = time.time()

        # Save resized image as JPEG for server processing
        buf = io.BytesIO()
        pil_image_resized.save(buf, format="JPEG", quality=85, optimize=True)
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

        # Use original frame size for display (like working YOLOv11 client)
        bgr_display = frame.to_ndarray(format="bgr24")

        image_final = self.plot_pose_cv2(bgr_display, detections)

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
