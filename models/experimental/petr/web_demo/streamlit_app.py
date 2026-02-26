# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PETR 3D Object Detection Demo - Streamlit App
Interactive demo showcasing 3D object detection on Tenstorrent hardware.

Usage:
    cd /local/ttuser/teja/tt-metal
    source python_env/bin/activate
    python3 -m streamlit run models/experimental/petr/demo/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
"""

import sys
import time
import numpy as np
import cv2
import torch
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# Add tt-metal to path
TT_METAL_ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(TT_METAL_ROOT))

# PETR imports
from models.experimental.petr.reference.utils import LiDARInstance3DBoxes
from models.experimental.petr.reference.petr import PETR
from models.experimental.petr.tt.ttnn_petr import ttnn_PETR
from models.experimental.petr.tt.common import get_parameters, generate_petr_inputs

NUSCENES_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

CLASS_COLORS = {
    "car": "#FF6B6B",
    "truck": "#4ECDC4",
    "construction_vehicle": "#FFE66D",
    "bus": "#95E1D3",
    "trailer": "#F38181",
    "barrier": "#AA96DA",
    "motorcycle": "#FCBAD3",
    "bicycle": "#A8D8EA",
    "pedestrian": "#FF9F43",
    "traffic_cone": "#EE5A24",
}

CAMERA_NAMES = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

CAMERA_LAYOUT = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]


def load_sample_data():
    """Load sample nuScenes images and calibration."""
    data_root = Path(__file__).parent.parent / "resources" / "sample_input"

    _, meta_data = generate_petr_inputs()
    meta = meta_data[0] if isinstance(meta_data, list) else meta_data

    # Compute lidar2img
    if "lidar2img" not in meta:
        lidar2img_list = []
        for i in range(6):
            cam2img = meta["cam2img"][i]
            lidar2cam = meta["lidar2cam"][i]
            if isinstance(cam2img, torch.Tensor):
                cam2img = cam2img.to(torch.float32).cpu().numpy()
            if isinstance(lidar2cam, torch.Tensor):
                lidar2cam = lidar2cam.to(torch.float32).cpu().numpy()
            lidar2img = cam2img @ lidar2cam
            lidar2img_list.append(lidar2img)
        meta["lidar2img"] = lidar2img_list

    FILENAMES = [
        "n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402927604844.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402927647423.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402927627893.jpg",
    ]

    imgs_list = []
    camera_images = {}

    for cam, filename in zip(CAMERA_NAMES, FILENAMES):
        img_path = data_root / filename
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)

        img_resized = cv2.resize(img, (800, 320))
        camera_images[cam] = img_resized.copy()

        # Normalize for model
        img_norm = img_resized.astype(np.float32)
        mean = np.array([103.530, 116.280, 123.675])
        std = np.array([57.375, 57.120, 58.395])
        img_norm = (img_norm - mean) / std
        img_norm = img_norm.transpose(2, 0, 1)
        imgs_list.append(img_norm)

    imgs = np.stack(imgs_list).astype(np.float32)
    imgs = torch.from_numpy(imgs).unsqueeze(0)

    img_metas = [
        {
            "filename": CAMERA_NAMES,
            "ori_shape": [(900, 1600)] * 6,
            "img_shape": (320, 800),
            "pad_shape": (320, 800),
            "lidar2img": meta["lidar2img"],
            "cam2img": meta.get("cam2img", [np.eye(4, dtype=np.float32) for _ in range(6)]),
            "lidar2cam": meta.get("lidar2cam", [np.eye(4, dtype=np.float32) for _ in range(6)]),
            "box_type_3d": LiDARInstance3DBoxes,
        }
    ]

    return imgs, img_metas, camera_images


@st.cache_resource
def load_models():
    """Load and cache TTNN model (runs once)."""
    import ttnn

    # Download weights if needed
    import urllib.request

    weights_url = (
        "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/"
        "petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    )
    resources_dir = Path(__file__).parent.parent / "resources"
    weights_path = resources_dir / "petr_vovnet_gridmask_p4_800x320-e2191752.pth"

    if not weights_path.exists():
        urllib.request.urlretrieve(weights_url, str(weights_path))

    weights_state_dict = torch.load(str(weights_path), weights_only=False)["state_dict"]

    # Load PyTorch model for parameter extraction
    torch_model = PETR(use_grid_mask=True)
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    # Open TT device
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    # Create TTNN model
    parameters, query_embedding_input = get_parameters(torch_model, device)
    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    return ttnn_model, torch_model, device


def get_box_corners_3d(box):
    """Convert box parameters to 8 corner coordinates."""
    x, y, z, w, l, h, yaw = box[:7]
    hw, hl, hh = w / 2, l / 2, h / 2

    local_corners = np.array(
        [
            [-hl, -hw, -hh],
            [hl, -hw, -hh],
            [hl, hw, -hh],
            [-hl, hw, -hh],
            [-hl, -hw, hh],
            [hl, -hw, hh],
            [hl, hw, hh],
            [-hl, hw, hh],
        ],
        dtype=np.float64,
    )

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]], dtype=np.float64)

    corners = local_corners @ rotation_matrix.T + np.array([x, y, z])
    return corners


def create_3d_plotly(boxes, scores, labels, threshold, selected_classes):
    """Create interactive 3D plot with Plotly."""
    fig = go.Figure()

    # Add ego vehicle
    ego_x = [-2, 2, 2, -2, -2, -2, 2, 2, -2, -2, 2, 2, 2, 2, -2, -2]
    ego_y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    ego_z = [0, 0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0, 0, 1.5, 1.5, 0]

    fig.add_trace(
        go.Scatter3d(x=ego_x, y=ego_y, z=ego_z, mode="lines", line=dict(color="red", width=4), name="Ego Vehicle")
    )

    # Add detected boxes
    detection_count = 0
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        class_name = NUSCENES_CLASSES[int(label)]
        if class_name not in selected_classes:
            continue

        detection_count += 1
        corners = get_box_corners_3d(box)
        color = CLASS_COLORS.get(class_name, "#FF00FF")

        edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[corners[edge[0], 0], corners[edge[1], 0]],
                    y=[corners[edge[0], 1], corners[edge[1], 1]],
                    z=[corners[edge[0], 2], corners[edge[1], 2]],
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"{class_name} ({score:.2f})",
                    showlegend=(edge == [0, 1]),
                    hoverinfo="name",
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="X (forward) [m]",
            yaxis_title="Y (left) [m]",
            zaxis_title="Z (up) [m]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        title=dict(text=f"3D Bird's Eye View - {detection_count} Detections", x=0.5),
        height=500,
        showlegend=True,
        legend=dict(x=1.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig, detection_count


def draw_boxes_on_image(img, boxes, scores, labels, lidar2img, threshold, selected_classes):
    """Draw 3D boxes projected onto 2D image."""
    img_out = img.copy()
    h, w = img_out.shape[:2]

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        class_name = NUSCENES_CLASSES[int(label)]
        if class_name not in selected_classes:
            continue

        corners_3d = get_box_corners_3d(box)
        color_hex = CLASS_COLORS.get(class_name, "#FF00FF")
        color_rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))

        corners_2d = []
        valid_indices = []

        for idx, corner in enumerate(corners_3d):
            point_homo = np.array([corner[0], corner[1], corner[2], 1.0])
            lidar2img_np = lidar2img
            if isinstance(lidar2img, torch.Tensor):
                lidar2img_np = lidar2img.to(torch.float32).cpu().numpy()
            proj = lidar2img_np @ point_homo

            if proj[2] > 0:
                u, v = proj[0] / proj[2], proj[1] / proj[2]
                if 0 <= u < w and 0 <= v < h:
                    corners_2d.append([int(u), int(v)])
                    valid_indices.append(idx)

        if len(valid_indices) < 2:
            continue

        corners_2d = np.array(corners_2d)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

        for start_idx, end_idx in edges:
            if start_idx in valid_indices and end_idx in valid_indices:
                start_pos = valid_indices.index(start_idx)
                end_pos = valid_indices.index(end_idx)
                cv2.line(img_out, tuple(corners_2d[start_pos]), tuple(corners_2d[end_pos]), color_rgb, 2, cv2.LINE_AA)

        if len(corners_2d) > 0:
            label_pt = tuple(corners_2d[0])
            label_text = f"{class_name} {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                img_out, (label_pt[0], label_pt[1] - text_h - 4), (label_pt[0] + text_w + 4, label_pt[1]), (0, 0, 0), -1
            )
            cv2.putText(
                img_out,
                label_text,
                (label_pt[0] + 2, label_pt[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_rgb,
                1,
                cv2.LINE_AA,
            )

    return img_out


def run_inference(ttnn_model, device, imgs, img_metas):
    """Run TTNN inference and measure time."""
    import ttnn

    ttnn_inputs = {"imgs": ttnn.from_torch(imgs, device=device)}

    start_time = time.time()
    predictions = ttnn_model.predict(ttnn_inputs, img_metas)
    inference_time = (time.time() - start_time) * 1000

    return predictions, inference_time


def main():
    st.set_page_config(page_title="PETR 3D Detection - Tenstorrent", page_icon="🚗", layout="wide")

    st.title("🚗 PETR 3D Object Detection")
    st.markdown("**Multi-camera 3D object detection powered by Tenstorrent Hardware**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.2, 0.05)

        st.markdown("---")
        st.subheader("🏷️ Class Filter")
        selected_classes = []
        cols = st.columns(2)
        for i, cls in enumerate(NUSCENES_CLASSES):
            with cols[i % 2]:
                if st.checkbox(cls, value=True, key=f"cls_{cls}"):
                    selected_classes.append(cls)

        st.markdown("---")
        st.subheader("📊 Model Info")
        st.markdown(
            """
        - **Model:** PETR (VoVNet)
        - **Input:** 6 cameras, 800×320
        - **Hardware:** Tenstorrent
        - **Blackhole:** ~3 FPS
        """
        )

    # Session state
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
        st.session_state.inference_time = None

    # Load data
    imgs, img_metas, camera_images = load_sample_data()

    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 Run TTNN Inference", type="primary", use_container_width=True)

    if run_button:
        with st.spinner("Loading model and running inference..."):
            try:
                ttnn_model, torch_model, device = load_models()
                predictions, inference_time = run_inference(ttnn_model, device, imgs, img_metas)
                st.session_state.predictions = predictions
                st.session_state.inference_time = inference_time
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display results
    if st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        inference_time = st.session_state.inference_time

        pred = predictions[0]["pts_bbox"]
        boxes = pred["bboxes_3d"].tensor.to(torch.float32).cpu().numpy()
        scores = pred["scores_3d"].to(torch.float32).cpu().numpy()
        labels = pred["labels_3d"].to(torch.float32).cpu().numpy()

        st.markdown("---")

        # 3D View
        st.markdown("### 🌐 Interactive 3D Bird's Eye View")
        st.markdown("*Drag to rotate, scroll to zoom*")
        fig_3d, _ = create_3d_plotly(boxes, scores, labels, threshold, selected_classes)
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")

        # Camera views
        st.markdown("### 📷 Multi-Camera Views")
        for row_cams in CAMERA_LAYOUT:
            cols = st.columns(3)
            for col_idx, cam_name in enumerate(row_cams):
                with cols[col_idx]:
                    cam_idx = CAMERA_NAMES.index(cam_name)
                    lidar2img = img_metas[0]["lidar2img"][cam_idx]
                    img_with_boxes = draw_boxes_on_image(
                        camera_images[cam_name], boxes, scores, labels, lidar2img, threshold, selected_classes
                    )
                    st.image(img_with_boxes, caption=cam_name, use_container_width=True)
    else:
        st.markdown("### 📷 Sample Input Images")
        st.info("👆 Click 'Run TTNN Inference' to detect 3D objects")
        for row_cams in CAMERA_LAYOUT:
            cols = st.columns(3)
            for col_idx, cam_name in enumerate(row_cams):
                with cols[col_idx]:
                    st.image(camera_images[cam_name], caption=cam_name, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>PETR 3D Detection | Tenstorrent Hardware</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
