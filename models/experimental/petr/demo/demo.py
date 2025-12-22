# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import os
import torch
import ttnn
import numpy as np
import cv2
import matplotlib
import urllib.request

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import the TT-specific modules
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

CAMERA_NAMES = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def load_calibration(data_root="models/experimental/petr/resources/sample_input"):
    """Load the proven working calibration"""
    _, meta_data = generate_petr_inputs()
    meta = meta_data[0] if isinstance(meta_data, list) else meta_data
    cam2img_orig = meta["cam2img"][0]
    if isinstance(cam2img_orig, torch.Tensor):
        cam2img_orig = cam2img_orig.to(torch.float32).cpu().numpy()

    # Compute lidar2img
    if "lidar2img" not in meta:
        lidar2img_list = []

        for i in range(6):
            cam2img = meta["cam2img"][i]
            lidar2cam = meta["lidar2cam"][i]

            # Convert to numpy
            if isinstance(cam2img, torch.Tensor):
                cam2img = cam2img.to(torch.float32).cpu().numpy()
            if isinstance(lidar2cam, torch.Tensor):
                lidar2cam = lidar2cam.to(torch.float32).cpu().numpy()
            lidar2img = cam2img @ lidar2cam
            lidar2img_list.append(lidar2img)

        meta["lidar2img"] = lidar2img_list
    # Test the calibration
    test_point = np.array([10.0, 0.0, 0.0, 1.0])
    lidar2img = meta["lidar2img"][0]  # CAM_FRONT
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.to(torch.float32).cpu().numpy()

    proj = lidar2img @ test_point
    u, v = proj[0] / proj[2], proj[1] / proj[2]

    return meta


def load_images_with_calibration(data_root="models/experimental/petr/resources/sample_input"):
    """Load the images that match the calibration"""

    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    # Load calibration
    meta = load_calibration(data_root)

    FILENAMES = [
        "n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402927604844.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402927647423.jpg",
        "n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402927627893.jpg",
    ]

    imgs_list = []
    camera_images = []

    for cam, filename in zip(cam_names, FILENAMES):
        img_path = Path(data_root) / filename

        if img_path.exists():
            img = cv2.imread(str(img_path))
        else:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)

        # Resize to model input size
        img = cv2.resize(img, (800, 320))
        camera_images.append(img.copy())  # Store for visualization

        # Normalize
        img = img.astype(np.float32)
        mean = np.array([103.530, 116.280, 123.675])
        std = np.array([57.375, 57.120, 58.395])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        imgs_list.append(img)

    # Stack images
    imgs = np.stack(imgs_list).astype(np.float32)
    imgs = torch.from_numpy(imgs).unsqueeze(0)  # [1, 6, 3, H, W]

    # Create img_metas with calibration
    img_metas = [
        {
            "filename": cam_names,
            "ori_shape": [(900, 1600)] * 6,
            "img_shape": (320, 800),
            "pad_shape": (320, 800),
            "lidar2img": meta["lidar2img"],
            "cam2img": meta.get("cam2img", [np.eye(4, dtype=np.float32) for _ in range(6)]),
            "cam_intrinsic": meta.get("cam_intrinsic", [np.eye(3, dtype=np.float32) for _ in range(6)]),
            "lidar2cam": meta.get("lidar2cam", [np.eye(4, dtype=np.float32) for _ in range(6)]),
        }
    ]

    return {"imgs": imgs, "img_metas": img_metas}, camera_images


class Det3DDataPreprocessor:
    """Simple Implementation of Det3DDataPreprocessor"""

    def __init__(
        self, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False, pad_size_divisor=32
    ):
        self.mean = torch.tensor(mean).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 1, 3, 1, 1)
        self.bgr_to_rgb = bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor

    def __call__(self, data, training=False):
        """Process the input data"""
        if isinstance(data, str):
            data = torch.load(data)

        # Extract images
        if "imgs" in data:
            imgs = data["imgs"]
        elif "inputs" in data and "imgs" in data["inputs"]:
            imgs = data["inputs"]["imgs"]
        else:
            logger.warning(f"Warning: No images found, creating dummy data")
            imgs = torch.zeros(1, 6, 3, 320, 800)

        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs).float()

        if len(imgs.shape) == 4:
            imgs = imgs.unsqueeze(0)

        # Normalize if needed
        if imgs.max() > 10.0:
            imgs = (imgs - self.mean) / self.std

        # Pad to divisible size
        _, _, _, h, w = imgs.shape
        pad_h = (self.pad_size_divisor - h % self.pad_size_divisor) % self.pad_size_divisor
        pad_w = (self.pad_size_divisor - w % self.pad_size_divisor) % self.pad_size_divisor

        if pad_h > 0 or pad_w > 0:
            imgs = torch.nn.functional.pad(imgs, (0, pad_w, 0, pad_h))

        output = {"inputs": {"imgs": imgs}, "data_samples": []}

        batch_size = imgs.shape[0]
        for i in range(batch_size):
            sample = type("DataSample", (), {})()

            if "img_metas" in data:
                if isinstance(data["img_metas"], list):
                    metainfo = data["img_metas"][i] if i < len(data["img_metas"]) else data["img_metas"][0]
                else:
                    metainfo = data["img_metas"]
            else:
                metainfo = {}

            # Ensure all required fields exist
            metainfo.setdefault("img_shape", (h, w))
            metainfo.setdefault("pad_shape", (h + pad_h, w + pad_w))
            metainfo.setdefault("ori_shape", [(h, w)] * 6)

            metainfo["box_type_3d"] = LiDARInstance3DBoxes
            metainfo.setdefault("sample_idx", i)

            sample.metainfo = metainfo
            output["data_samples"].append(sample)

        return output


def get_box_corners_3d(box):
    """Convert box to 8 corner coordinates."""
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

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]], dtype=np.float64)

    corners = local_corners @ rotation_matrix.T + np.array([x, y, z])
    return corners


def project_point_simple(point_3d, lidar2img, image_shape):
    """Simple projection using lidar2img directly."""
    h, w = image_shape[:2]

    # Convert to homogeneous coordinates
    point_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])

    # Project using lidar2img matrix
    proj = lidar2img @ point_homo

    # Check depth
    if proj[2] <= 0:
        return None, None, False

    # Normalize
    u = proj[0] / proj[2]
    v = proj[1] / proj[2]

    # Check bounds
    in_image = (0 <= u < w) and (0 <= v < h)

    return u, v, in_image


def draw_box_simple(img, box, lidar2img, color=(255, 0, 255), thickness=2, label_text=""):
    """Draw 3D box on image."""
    corners_3d = get_box_corners_3d(box)
    h, w = img.shape[:2]

    corners_2d = []
    valid_indices = []

    for idx, corner in enumerate(corners_3d):
        u, v, in_img = project_point_simple(corner, lidar2img, img.shape)
        if in_img:
            corners_2d.append([u, v])
            valid_indices.append(idx)

    if len(valid_indices) < 2:
        return img

    corners_2d = np.array(corners_2d).astype(np.int32)

    # Draw edges
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for start_idx, end_idx in edges:
        if start_idx in valid_indices and end_idx in valid_indices:
            start_pos = valid_indices.index(start_idx)
            end_pos = valid_indices.index(end_idx)

            start_pt = tuple(corners_2d[start_pos])
            end_pt = tuple(corners_2d[end_pos])
            cv2.line(img, start_pt, end_pt, color, thickness, cv2.LINE_AA)

    # Draw center
    if len(corners_2d) > 0:
        center = corners_2d.mean(axis=0).astype(np.int32)
        if 0 <= center[0] < w and 0 <= center[1] < h:
            cv2.circle(img, tuple(center), 5, color, -1)

    # Draw label
    if label_text and len(corners_2d) > 0:
        label_pt = tuple(corners_2d[0])
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(
            img, (label_pt[0], label_pt[1] - text_h - 3), (label_pt[0] + text_w, label_pt[1] + 2), (0, 0, 0), -1
        )
        cv2.putText(
            img, label_text, (label_pt[0], label_pt[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    return img


def create_3d_plot(predictions, output_path, title="3D", threshold=0.2):
    """Create visualization."""
    logger.info(f"  {title}...")

    pred = predictions[0]["pts_bbox"]
    boxes = pred["bboxes_3d"].tensor.to(torch.float32).cpu().numpy()
    scores = pred["scores_3d"].to(torch.float32).cpu().numpy()

    keep = scores >= threshold
    boxes = boxes[keep]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X (m)", fontsize=11, weight="bold")
    ax.set_ylabel("Y (m)", fontsize=11, weight="bold")
    ax.set_zlabel("Z (m)", fontsize=11, weight="bold")
    ax.set_title(title, fontsize=13, weight="bold")

    # Ego vehicle
    ego_corners = np.array(
        [
            [-2, -1, 0],
            [2, -1, 0],
            [2, 1, 0],
            [-2, 1, 0],
            [-2, -1, -0.5],
            [2, -1, -0.5],
            [2, 1, -0.5],
            [-2, 1, -0.5],
        ]
    )
    ego_faces = [
        [ego_corners[0], ego_corners[1], ego_corners[5], ego_corners[4]],
        [ego_corners[2], ego_corners[3], ego_corners[7], ego_corners[6]],
    ]
    ego_collection = Poly3DCollection(ego_faces, alpha=0.5, facecolor="red", edgecolors="red", linewidths=2)
    ax.add_collection3d(ego_collection)

    # Draw boxes
    for box in boxes:
        corners = get_box_corners_3d(box)
        faces = [
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
        ]
        face_collection = Poly3DCollection(faces, alpha=0.2, facecolor="magenta", edgecolors="magenta", linewidths=1.5)
        ax.add_collection3d(face_collection)

    if len(boxes) > 0:
        all_corners = np.vstack([get_box_corners_3d(box) for box in boxes])
        x_min, x_max = all_corners[:, 0].min() - 5, all_corners[:, 0].max() + 5
        y_min, y_max = all_corners[:, 1].min() - 5, all_corners[:, 1].max() + 5
        z_min, z_max = all_corners[:, 2].min() - 5, all_corners[:, 2].max() + 5
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = -50, 50, -50, 50, -5, 5

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_on_images(predictions, camera_images, img_metas, output_dir, threshold=0.2, prefix=""):
    """Visualize boxes on camera images."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"  Camera projections ({prefix})...")

    pred = predictions[0]["pts_bbox"]
    boxes = pred["bboxes_3d"].tensor.to(torch.float32).cpu().numpy()
    scores = pred["scores_3d"].to(torch.float32).cpu().numpy()
    labels = pred["labels_3d"].to(torch.float32).cpu().numpy()

    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    img_meta = img_metas[0]
    lidar2img_list = img_meta["lidar2img"]

    if isinstance(camera_images, list):
        camera_items = [
            (CAMERA_NAMES[i] if i < len(CAMERA_NAMES) else f"CAM_{i}", img) for i, img in enumerate(camera_images)
        ]
    else:
        camera_items = list(camera_images.items())

    for cam_idx, (cam_name, cam_image) in enumerate(camera_items):
        if cam_idx >= len(lidar2img_list):
            continue

        lidar2img = lidar2img_list[cam_idx]
        if isinstance(lidar2img, torch.Tensor):
            lidar2img = lidar2img.to(torch.float32).cpu().numpy()

        vis_img = cam_image.copy()
        if vis_img.dtype in [np.float32, np.float64]:
            if vis_img.max() <= 1.0:
                vis_img = (vis_img * 255).astype(np.uint8)
            else:
                vis_img = vis_img.astype(np.uint8)
        if len(vis_img.shape) == 3 and vis_img.shape[2] == 3:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        for box, score, label in zip(boxes, scores, labels):
            class_name = NUSCENES_CLASSES[int(label)]
            label_text = f"{class_name} {score:.2f}"
            vis_img = draw_box_simple(vis_img, box, lidar2img, color=(255, 0, 255), thickness=2, label_text=label_text)

        output_path = os.path.join(output_dir, f"{prefix}{cam_name}.jpg")
        cv2.imwrite(output_path, vis_img)
        logger.info(f"    {cam_name}")


def test_demo(device, reset_seeds):
    """Main demo."""
    logger.info("Loading data...")
    input_data, camera_images = load_images_with_calibration("models/experimental/petr/resources/sample_input")

    data_preprocessor = Det3DDataPreprocessor()
    output_after_preprocess = data_preprocessor(input_data, False)
    batch_img_metas = [ds.metainfo for ds in output_after_preprocess["data_samples"]]

    weights_url = (
        "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/"
        "petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    )
    resources_dir = os.path.join(os.path.dirname(__file__), "..", "resources")
    weights_path = os.path.abspath(os.path.join(resources_dir, "petr_vovnet_gridmask_p4_800x320-e2191752.pth"))

    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    if not os.path.exists(weights_path):
        logger.info("Downloading weights...")
        urllib.request.urlretrieve(weights_url, weights_path)

    weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]

    logger.info("Running PyTorch inference...")
    torch_model = PETR(use_grid_mask=True)
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    with torch.no_grad():
        torch_output = torch_model.predict(output_after_preprocess["inputs"], batch_img_metas)

    logger.info("Running TTNN inference...")
    ttnn_inputs = dict()
    imgs_tensor = output_after_preprocess["inputs"]["imgs"]
    if len(imgs_tensor.shape) == 4:
        imgs_tensor = imgs_tensor.unsqueeze(0)
    ttnn_inputs["imgs"] = ttnn.from_torch(imgs_tensor, device=device)

    parameters, query_embedding_input = get_parameters(torch_model, device)
    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    ttnn_output = ttnn_model.predict(ttnn_inputs, batch_img_metas)

    output_dir = "models/experimental/petr/resources/sample_output"
    os.makedirs(output_dir, exist_ok=True)

    create_3d_plot(ttnn_output, os.path.join(output_dir, "ttnn_3d.png"), "TTNN 3D Predictions", 0.2)

    visualize_on_images(ttnn_output, camera_images, batch_img_metas, output_dir, 0.2, "ttnn_")

    logger.info("\n" + "=" * 80)
    logger.info(f"Files saved to: {output_dir}/\n")
    logger.info("Demo COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        test_demo(device, None)
    finally:
        ttnn.close_device(device)
