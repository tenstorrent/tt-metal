# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Streamlit demo for RF-DETR Medium.
Runs detection on Tenstorrent hardware; draws COCO detections on image or video.
Run: streamlit run models/experimental/rfdetr_medium/demo/streamlit_demo.py --server.port 8501
"""

import io
import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, ".")

RESOLUTION = 576

COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

COLORS = np.random.RandomState(42).randint(60, 255, size=(91, 3)).tolist()


def preprocess_image(pil_image):
    img = pil_image.convert("RGB")
    img_resized = img.resize((RESOLUTION, RESOLUTION), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def draw_detections(image_bgr, detections, score_thr=0.3):
    img = image_bgr.copy()
    h, w = img.shape[:2]
    scale_x = w / RESOLUTION
    scale_y = h / RESOLUTION

    if not detections or len(detections[0]["boxes"]) == 0:
        return img

    det = detections[0]
    boxes = det["boxes"]
    scores = det["scores"]
    labels = det["labels"]

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
        label_idx = int(labels[i])
        class_name = COCO_ID_TO_NAME.get(label_idx, f"class_{label_idx}")
        color = COLORS[label_idx % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def extract_unique_frames(video_path, max_frames=50):
    """Extract unique frames from video (skip near-duplicates)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    prev_hash = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (16, 16))
        curr_hash = small.tobytes()
        if curr_hash != prev_hash:
            frames.append(frame)
            prev_hash = curr_hash
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


@st.cache_resource
def load_torch_model_cached():
    """Cache the PyTorch model (CPU) across Streamlit reruns."""
    from models.experimental.rfdetr_medium.common import load_torch_model

    return load_torch_model()


def run_inference_fresh_device(image_tensor):
    """
    Run a single inference with a fresh device open/close cycle.
    This avoids the TTNN device state corruption that causes 0 detections
    on subsequent forward passes without device reset.
    """
    import ttnn
    from models.experimental.rfdetr_medium.common import RFDETR_MEDIUM_L1_SMALL_SIZE
    from models.experimental.rfdetr_medium.tt.tt_rfdetr import TtRFDETR
    from models.experimental.rfdetr_medium.tt.model_preprocessing import (
        load_backbone_weights,
        load_projector_weights,
        load_decoder_weights,
        load_detection_head_weights,
    )

    torch_model = load_torch_model_cached()
    device = ttnn.open_device(device_id=0, l1_small_size=RFDETR_MEDIUM_L1_SMALL_SIZE)
    try:
        bp = load_backbone_weights(torch_model, device)
        pp = load_projector_weights(torch_model, device)
        dp = load_decoder_weights(torch_model, device)
        hp = load_detection_head_weights(torch_model, device)
        tt_model = TtRFDETR(
            device=device,
            torch_model=torch_model,
            backbone_params=bp,
            projector_params=pp,
            decoder_params=dp,
            head_params=hp,
        )
        result = tt_model.forward(image_tensor)
        detections = result["detections"]
    finally:
        ttnn.close_device(device)
    return detections


def main():
    st.set_page_config(page_title="RF-DETR Medium — Object Detection on Tenstorrent", layout="wide")
    st.title("RF-DETR Medium — Object Detection on Tenstorrent")
    st.caption("Upload an image/video or use a sample. Runs on Tenstorrent hardware; draws COCO detections.")

    score_thr = st.slider("Score threshold", 0.1, 0.8, 0.3, 0.05)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        source = st.radio("Source", ["Upload image", "Sample image", "Input video"], horizontal=True)

        pil_image = None
        video_path = None

        if source == "Upload image":
            file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img")
            if file is None:
                st.info("Upload an image to run detection.")
                return
            pil_image = Image.open(file).convert("RGB")
            st.image(np.array(pil_image), use_container_width=True, channels="RGB")

        elif source == "Sample image":
            sample = st.selectbox(
                "Sample",
                ["Cats & remotes", "Living room", "Kitchen"],
                index=0,
            )
            urls = {
                "Cats & remotes": "http://images.cocodataset.org/val2017/000000039769.jpg",
                "Living room": "http://images.cocodataset.org/val2017/000000000139.jpg",
                "Kitchen": "http://images.cocodataset.org/val2017/000000397133.jpg",
            }
            import requests

            r = requests.get(urls[sample], timeout=10)
            r.raise_for_status()
            pil_image = Image.open(io.BytesIO(r.content)).convert("RGB")
            st.image(np.array(pil_image), use_container_width=True, channels="RGB")

        else:
            demo_dir = os.path.dirname(os.path.abspath(__file__))
            builtin_video = os.path.join(demo_dir, "rfdetr_demo_input.mp4")
            use_builtin = st.checkbox("Use built-in demo video", value=True, key="use_builtin")
            if use_builtin and os.path.exists(builtin_video):
                video_path = builtin_video
                st.caption("Built-in video: rfdetr_demo_input.mp4")
            elif use_builtin:
                st.warning("Built-in video not found. Upload a video below.")
            vid_file = st.file_uploader("Or upload a video", type=["mp4", "avi", "mov"], key="vid")
            if vid_file:
                tfile = os.path.join("/tmp", vid_file.name)
                with open(tfile, "wb") as f:
                    f.write(vid_file.getbuffer())
                video_path = tfile
                st.caption(f"Uploaded: {vid_file.name}")
            if video_path is None:
                st.info("Select built-in demo video or upload a video file.")
                return

    with col2:
        st.subheader("Detections (Tenstorrent)")

        # --- Video mode ---
        if source == "Input video" and video_path:
            if st.button("Run detection on video", type="primary"):
                unique_frames = extract_unique_frames(video_path, max_frames=30)
                if not unique_frames:
                    st.warning("No frames could be read from the video.")
                    return

                n_frames = len(unique_frames)
                progress = st.progress(0)
                status_text = st.empty()
                result_placeholder = st.empty()
                det_frames_rgb = []

                for idx, frame_bgr in enumerate(unique_frames):
                    status_text.text(f"Processing frame {idx+1}/{n_frames}...")
                    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    tensor = preprocess_image(pil)
                    t0 = time.perf_counter()
                    dets = run_inference_fresh_device(tensor)
                    t1 = time.perf_counter()
                    out_bgr = draw_detections(frame_bgr, dets, score_thr=score_thr)

                    n_det = len(dets[0]["boxes"]) if dets and len(dets[0]["boxes"]) > 0 else 0
                    cv2.putText(
                        out_bgr,
                        f"RF-DETR Medium on Tenstorrent | {(t1-t0)*1000:.0f}ms | {n_det} det",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                    det_frames_rgb.append(out_rgb)

                    result_placeholder.image(out_rgb, use_container_width=True, channels="RGB")
                    progress.progress((idx + 1) / n_frames)

                progress.progress(1.0)
                status_text.text(f"Done! Processed {n_frames} frames.")

                st.session_state["det_frames_rgb"] = det_frames_rgb

            if "det_frames_rgb" in st.session_state and st.session_state["det_frames_rgb"]:
                det_frames_rgb = st.session_state["det_frames_rgb"]
                frame_idx = st.slider(
                    "Browse frames", 0, len(det_frames_rgb) - 1, len(det_frames_rgb) - 1, key="frame_slider"
                )
                st.image(
                    det_frames_rgb[frame_idx],
                    use_container_width=True,
                    channels="RGB",
                    caption=f"Frame {frame_idx + 1}/{len(det_frames_rgb)}",
                )
            return

        # --- Image mode ---
        if pil_image is not None and st.button("Run detection", type="primary"):
            image_tensor = preprocess_image(pil_image)
            with st.spinner("Running inference on Tenstorrent..."):
                t0 = time.perf_counter()
                detections = run_inference_fresh_device(image_tensor)
                t1 = time.perf_counter()

            orig_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            out_bgr = draw_detections(orig_bgr, detections, score_thr=score_thr)

            n = len(detections[0]["boxes"]) if detections else 0
            cv2.putText(
                out_bgr,
                f"RF-DETR Medium on Tenstorrent | {(t1-t0)*1000:.0f}ms | {n} det",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            st.caption(f"Inference: {(t1-t0)*1000:.0f} ms  |  Detections: {n}")
            st.image(out_rgb, use_container_width=True, channels="RGB")


if __name__ == "__main__":
    main()
