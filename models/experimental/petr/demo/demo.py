# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import os
import torch
import ttnn
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from loguru import logger

# Import the TT-specific modules
from models.experimental.petr.reference.utils import LiDARInstance3DBoxes
from models.experimental.petr.reference.petr import PETR
from models.experimental.petr.tt.ttnn_petr import ttnn_PETR

from models.experimental.petr.tt.common import get_parameters, generate_petr_inputs


class Det3DDataPreprocessor:
    """Mock Implementation of Det3DDataPreprocessor"""

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


def create_lidar_bev_image(boxes, scores, labels, img_size=800):
    """Create bird's eye view image"""

    # Create black canvas
    bev_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Coordinate system:
    # X: -50 to +50 meters (left to right)
    # Y: -50 to +50 meters (back to front)
    meters_per_pixel = 100 / img_size  # 100m range, img_size pixels

    def world_to_pixel(x, y):
        """Convert world coordinates (meters) to pixel coordinates"""
        px = int((x + 50) / meters_per_pixel)
        py = int((50 - y) / meters_per_pixel)  # Flip Y so front is up
        return px, py

    # Class colors
    class_colors = [
        (255, 0, 0),  # Red - Car
        (0, 255, 0),  # Green - Truck
        (0, 0, 255),  # Blue - Bus
        (255, 255, 0),  # Yellow - Trailer
        (255, 0, 255),  # Magenta - Pedestrian
        (0, 255, 255),  # Cyan - Motorcycle
        (128, 128, 0),  # Olive - Bicycle
        (128, 0, 128),  # Purple - Traffic cone
        (0, 128, 128),  # Teal - Barrier
        (255, 128, 0),  # Orange - Construction vehicle
    ]

    # Draw ego vehicle (center)
    ego_px, ego_py = world_to_pixel(0, 0)
    cv2.circle(bev_img, (ego_px, ego_py), 5, (255, 255, 255), -1)
    cv2.putText(bev_img, "EGO", (ego_px - 15, ego_py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw detected boxes
    for i, box in enumerate(boxes):
        if scores[i] < 0.2:  # Skip low confidence
            continue

        x, y, z, l, w, h, yaw = box[:7]
        label = int(labels[i])
        color = class_colors[label % len(class_colors)]

        # Get 4 corners of box (top-down view)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        corners_x = np.array([-l / 2, l / 2, l / 2, -l / 2])
        corners_y = np.array([-w / 2, -w / 2, w / 2, w / 2])

        # Rotate corners
        rot_x = corners_x * cos_yaw - corners_y * sin_yaw + x
        rot_y = corners_x * sin_yaw + corners_y * cos_yaw + y

        # Convert to pixels
        pts = []
        for cx, cy in zip(rot_x, rot_y):
            px, py = world_to_pixel(cx, cy)
            pts.append([px, py])

        pts = np.array(pts, dtype=np.int32)

        # Draw box
        cv2.polylines(bev_img, [pts], True, color, 2)

        # Draw heading direction (front of box)
        front_x = x + (l / 2) * cos_yaw
        front_y = y + (l / 2) * sin_yaw
        fx, fy = world_to_pixel(front_x, front_y)
        cx, cy = world_to_pixel(x, y)
        cv2.arrowedLine(bev_img, (cx, cy), (fx, fy), color, 2, tipLength=0.3)

        # Add score text
        cv2.putText(bev_img, f"{scores[i]:.2f}", (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Add grid lines
    for meter in range(-50, 51, 10):
        px, py_top = world_to_pixel(meter, 50)
        px, py_bot = world_to_pixel(meter, -50)
        cv2.line(bev_img, (px, py_top), (px, py_bot), (50, 50, 50), 1)

        px_left, py = world_to_pixel(-50, meter)
        px_right, py = world_to_pixel(50, meter)
        cv2.line(bev_img, (px_left, py), (px_right, py), (50, 50, 50), 1)

    # Add axis labels
    cv2.putText(bev_img, "FRONT", (img_size // 2 - 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(bev_img, "BACK", (img_size // 2 - 25, img_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return bev_img


def visualizations(ttnn_output, camera_images, img_metas, output_dir="./"):
    """Save visualizations comparing PyTorch and TTNN outputs side-by-side"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process TTNN output
    ttnn_dict = ttnn_output[0]["pts_bbox"]
    ttnn_boxes = ttnn_dict["bboxes_3d"].tensor.cpu().numpy()
    ttnn_scores = ttnn_dict["scores_3d"].to(torch.float32).cpu().numpy()
    ttnn_labels = ttnn_dict["labels_3d"].to(torch.float32).cpu().numpy()

    threshold = 0.2

    # Filter TTNN results
    ttnn_mask = ttnn_scores > threshold
    ttnn_filtered_boxes = ttnn_boxes[ttnn_mask]
    ttnn_filtered_scores = ttnn_scores[ttnn_mask]
    ttnn_filtered_labels = ttnn_labels[ttnn_mask]

    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    logger.info(f"Saved visualization to {output_dir}")
    for cam_id, cam_name in enumerate(cam_names):
        ttnn_img = camera_images[cam_id].copy()

        if "lidar2img" in img_metas[0] and len(img_metas[0]["lidar2img"]) > cam_id:
            # Now cam_id directly corresponds to the correct calibration
            lidar2img = img_metas[0]["lidar2img"][cam_id]

            if isinstance(lidar2img, torch.Tensor):
                lidar2img = lidar2img.cpu().numpy()

            # Draw boxes...
            ttnn_img = draw_lidar_bbox3d_on_img(
                ttnn_filtered_boxes[:50], ttnn_img, lidar2img, ttnn_img.shape[:2], color=(255, 0, 0), thickness=2
            )
            # Create ttnn image
            cv2.imwrite(f"{output_dir}/{cam_name}.jpg", ttnn_img)
            logger.info(f"{cam_name}.jpg")

    # TTNN BEV visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Single plot, not (ax1, ax2)

    class_colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "brown", "pink"]

    # TTNN BEV
    for i, box in enumerate(ttnn_filtered_boxes[:100]):
        x, y, z, l, w, h, yaw = box[:7]
        label = int(ttnn_filtered_labels[i])
        color = class_colors[label % len(class_colors)]

        rect = patches.Rectangle(
            (x - l / 2, y - w / 2), l, w, angle=np.degrees(yaw), linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x, y, f"{ttnn_filtered_scores[i]:.2f}", fontsize=8, ha="center")

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(f"TTNN BEV - {len(ttnn_filtered_boxes)} objects", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bev_ttnn.jpg", dpi=150, bbox_inches="tight")
    plt.close()

    bev_img = create_lidar_bev_image(ttnn_filtered_boxes, ttnn_filtered_scores, ttnn_filtered_labels)
    cv2.imwrite(f"{output_dir}/bev_lidar_view.jpg", bev_img)

    return ttnn_filtered_boxes


def draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, img_shape, color=(0, 255, 0), thickness=2):
    """Draws lidar bboxes on image, handles invalid projections and checks depth."""
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()

    if len(bboxes_3d) == 0:
        return img

    boxes_drawn = 0
    boxes_skipped_behind = 0
    boxes_skipped_outside = 0
    boxes_skipped_invalid = 0

    for bbox in bboxes_3d:
        l, w, h = bbox[3:6]

        # Create 8 corners
        x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
        z_corners = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
        corners = np.vstack([x_corners, y_corners, z_corners])

        # Rotate
        yaw = bbox[6]
        rot_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        corners = rot_mat @ corners

        # Translate
        corners[0, :] += bbox[0]
        corners[1, :] += bbox[1]
        corners[2, :] += bbox[2]
        corners = corners.T  # 8x3

        # Project to image
        pts_4d = np.concatenate([corners, np.ones((8, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img.T  # 8x4

        # Check depth
        depths = pts_2d[:, 2]
        valid_depth_mask = depths > 1.0
        if valid_depth_mask.sum() < 2:
            boxes_skipped_behind += 1
            continue

        pts_2d_normalized = pts_2d.copy()

        safe_depths = np.where(depths > 0.1, depths, 1e10)
        pts_2d_normalized[:, 0] = pts_2d[:, 0] / safe_depths
        pts_2d_normalized[:, 1] = pts_2d[:, 1] / safe_depths

        if np.any(np.isnan(pts_2d_normalized[:, :2])) or np.any(np.isinf(pts_2d_normalized[:, :2])):
            boxes_skipped_invalid += 1
            continue

        corners_2d = pts_2d_normalized[:, :2].astype(np.int32)

        # Draw only edges where BOTH corners have valid depth
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical
        ]

        lines_drawn = 0
        for i, j in edges:
            # Both corners must be in front of camera
            if not (valid_depth_mask[i] and valid_depth_mask[j]):
                continue

            pt1 = corners_2d[i]
            pt2 = corners_2d[j]

            # Check if line is reasonably within extended image bounds
            margin = 200
            if (
                min(pt1[0], pt2[0]) < img_shape[1] + margin
                and max(pt1[0], pt2[0]) > -margin
                and min(pt1[1], pt2[1]) < img_shape[0] + margin
                and max(pt1[1], pt2[1]) > -margin
            ):
                # Clip to image bounds to avoid OpenCV errors
                pt1_clipped = (
                    max(-1000, min(img_shape[1] + 1000, pt1[0])),
                    max(-1000, min(img_shape[0] + 1000, pt1[1])),
                )
                pt2_clipped = (
                    max(-1000, min(img_shape[1] + 1000, pt2[0])),
                    max(-1000, min(img_shape[0] + 1000, pt2[1])),
                )

                cv2.line(img, pt1_clipped, pt2_clipped, color, thickness)
                lines_drawn += 1

        if lines_drawn > 0:
            boxes_drawn += 1
        else:
            boxes_skipped_outside += 1

    return img


def load_calibration(data_root="models/experimental/functional_petr/resources/sample_input"):
    """Load the proven working calibration"""
    _, meta_data = generate_petr_inputs()
    meta = meta_data[0] if isinstance(meta_data, list) else meta_data
    cam2img_orig = meta["cam2img"][0]
    if isinstance(cam2img_orig, torch.Tensor):
        cam2img_orig = cam2img_orig.cpu().numpy()

    # Compute lidar2img
    if "lidar2img" not in meta:
        lidar2img_list = []

        for i in range(6):
            cam2img = meta["cam2img"][i]
            lidar2cam = meta["lidar2cam"][i]

            # Convert to numpy
            if isinstance(cam2img, torch.Tensor):
                cam2img = cam2img.cpu().numpy()
            if isinstance(lidar2cam, torch.Tensor):
                lidar2cam = lidar2cam.cpu().numpy()

            # NO SCALING NEEDED! cam2img is already for 320x800 images
            lidar2img = cam2img @ lidar2cam
            lidar2img_list.append(lidar2img)

        meta["lidar2img"] = lidar2img_list
    # Test the calibration
    test_point = np.array([10.0, 0.0, 0.0, 1.0])
    lidar2img = meta["lidar2img"][0]  # CAM_FRONT
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()

    proj = lidar2img @ test_point
    u, v = proj[0] / proj[2], proj[1] / proj[2]

    return meta


# Approximate calibration for the sample input
def load_images_with_calibration(data_root="models/experimental/functional_petr/resources/sample_input"):
    """
    Load the images that match the golden calibration
    """

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


def test_demo(device, reset_seeds):
    """
    Updated test_demo that uses calibration
    """
    input_data, camera_images = load_images_with_calibration("models/experimental/petr/resources/sample_input")

    # Initialize preprocessor
    data_preprocessor = Det3DDataPreprocessor()

    output_after_preprocess = data_preprocessor(input_data, False)
    batch_img_metas = [ds.metainfo for ds in output_after_preprocess["data_samples"]]

    # Verify calibration is correct
    test_point = np.array([10.0, 0.0, 0.0, 1.0])
    lidar2img = batch_img_metas[0]["lidar2img"][0]
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()
    proj = lidar2img @ test_point
    u, v = proj[0] / proj[2], proj[1] / proj[2]

    # Load model
    weights_url = (
        "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/petr/petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    )
    resources_dir = os.path.join(os.path.dirname(__file__), "..", "resources")
    weights_path = os.path.abspath(os.path.join(resources_dir, "petr_vovnet_gridmask_p4_800x320-e2191752.pth"))

    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    if not os.path.exists(weights_path):
        import urllib.request

        logger.info(f"Downloading PETR weights from {weights_url} ...")
        urllib.request.urlretrieve(weights_url, weights_path)
        logger.info(f"Weights downloaded to {weights_path}")
    weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]

    torch_model = PETR(use_grid_mask=True)
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    # Run PyTorch inference
    with torch.no_grad():
        torch_output = torch_model.predict(output_after_preprocess["inputs"], batch_img_metas)

    # Check results
    boxes = torch_output[0]["pts_bbox"]["bboxes_3d"].tensor.cpu().numpy()
    scores = torch_output[0]["pts_bbox"]["scores_3d"].cpu().numpy()

    # Convert to TTNN and run inference
    ttnn_inputs = dict()
    imgs_tensor = output_after_preprocess["inputs"]["imgs"]
    if len(imgs_tensor.shape) == 4:
        imgs_tensor = imgs_tensor.unsqueeze(0)
    ttnn_inputs["imgs"] = ttnn.from_torch(imgs_tensor, device=device)

    # Preprocess parameters
    parameters, query_embedding_input = get_parameters(torch_model, device)

    # Initialize TTNN model
    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    # Run TTNN inference
    ttnn_output = ttnn_model.predict(ttnn_inputs, batch_img_metas)

    # Save visualizations with approximate calibration
    ttnn_filtered = visualizations(
        ttnn_output, camera_images, batch_img_metas, output_dir="models/experimental/petr/resources/sample_output"
    )

    logger.info("Demo completed!")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        test_demo(device, None)
    finally:
        ttnn.close_device(device)
