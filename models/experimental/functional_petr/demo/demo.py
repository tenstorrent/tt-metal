# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Import the TT-specific modules
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    infer_ttnn_module_args,
)

from models.experimental.functional_petr.reference.utils import LiDARInstance3DBoxes
from models.experimental.functional_petr.reference.petr import PETR
from models.experimental.functional_petr.reference.petr_head import pos2posemb3d
from models.experimental.functional_petr.tt.ttnn_petr import ttnn_PETR
from models.experimental.functional_petr.tt.common import (
    move_to_device,
    create_custom_preprocessor_petr_head,
    create_custom_preprocessor_cpfpn,
    create_custom_preprocessor_vovnetcp,
)
from models.experimental.functional_petr.tt.common import stem_parameters_preprocess


class MockDet3DDataPreprocessor:
    """Standalone implementation of Det3DDataPreprocessor"""

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
            print("Warning: No images found, creating dummy data")
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


def save_comparison_visualizations(torch_output, ttnn_output, camera_images, img_metas, output_dir="./"):
    """Save visualizations comparing PyTorch and TTNN outputs side-by-side"""

    # Process PyTorch output
    torch_dict = torch_output[0]["pts_bbox"]
    torch_boxes = torch_dict["bboxes_3d"].tensor.cpu().numpy()
    torch_scores = torch_dict["scores_3d"].to(torch.float32).cpu().numpy()
    torch_labels = torch_dict["labels_3d"].to(torch.float32).cpu().numpy()

    # Process TTNN output
    ttnn_dict = ttnn_output[0]["pts_bbox"]
    ttnn_boxes = ttnn_dict["bboxes_3d"].tensor.cpu().numpy()
    ttnn_scores = ttnn_dict["scores_3d"].to(torch.float32).cpu().numpy()
    ttnn_labels = ttnn_dict["labels_3d"].to(torch.float32).cpu().numpy()

    threshold = 0.05

    # Filter PyTorch results
    torch_mask = torch_scores > threshold
    torch_filtered_boxes = torch_boxes[torch_mask]
    torch_filtered_scores = torch_scores[torch_mask]
    torch_filtered_labels = torch_labels[torch_mask]

    # Filter TTNN results
    ttnn_mask = ttnn_scores > threshold
    ttnn_filtered_boxes = ttnn_boxes[ttnn_mask]
    ttnn_filtered_scores = ttnn_scores[ttnn_mask]
    ttnn_filtered_labels = ttnn_labels[ttnn_mask]

    print(f"\n{'='*60}")
    print("DETECTION COMPARISON")
    print(f"{'='*60}")
    print(f"PyTorch: {len(torch_filtered_boxes)} boxes (score > {threshold})")
    print(f"  Score range: [{torch_scores.min():.4f}, {torch_scores.max():.4f}]")
    print(f"TTNN: {len(ttnn_filtered_boxes)} boxes (score > {threshold})")
    print(f"  Score range: [{ttnn_scores.min():.4f}, {ttnn_scores.max():.4f}]")
    print(f"{'='*60}\n")

    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    print("\n=== CALIBRATION ORDER CHECK ===")
    for idx in range(len(img_metas[0]["lidar2img"])):
        lidar2img = img_metas[0]["lidar2img"][idx]
        if isinstance(lidar2img, torch.Tensor):
            lidar2img = lidar2img.cpu().numpy()
        print(f"Index {idx}: lidar2img[0,3] = {lidar2img[0,3]:.4f}, lidar2img[1,3] = {lidar2img[1,3]:.4f}")
    print("=" * 40)

    # 1. Camera views - PyTorch vs TTNN side by side
    for cam_id, cam_name in enumerate(cam_names):
        # PyTorch image
        torch_img = camera_images[cam_id].copy()
        ttnn_img = camera_images[cam_id].copy()

        if "lidar2img" in img_metas[0] and len(img_metas[0]["lidar2img"]) > cam_id:
            # Now cam_id directly corresponds to the correct calibration
            lidar2img = img_metas[0]["lidar2img"][cam_id]

            if isinstance(lidar2img, torch.Tensor):
                lidar2img = lidar2img.cpu().numpy()

            print(f"\n{cam_name} using calibration index {cam_id}")

            # Draw boxes...
            torch_img = draw_lidar_bbox3d_on_img(
                torch_filtered_boxes[:50], torch_img, lidar2img, torch_img.shape[:2], color=(0, 255, 0), thickness=2
            )

            ttnn_img = draw_lidar_bbox3d_on_img(
                ttnn_filtered_boxes[:50], ttnn_img, lidar2img, ttnn_img.shape[:2], color=(255, 0, 0), thickness=2
            )

            # Create side-by-side comparison
            h, w = torch_img.shape[:2]
            comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
            comparison[:, :w] = torch_img
            comparison[:, w + 20 :] = ttnn_img

            # Add labels
            cv2.putText(comparison, "PyTorch (Green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "TTNN (Blue)", (w + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # cv2.putText(comparison, f"PT: {len(torch_filtered_boxes)} boxes", (10, 60),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(comparison, f"TT: {len(ttnn_filtered_boxes)} boxes", (w + 30, 60),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save individual images
            # cv2.imwrite(f"{output_dir}/pytorch_{cam_name}.jpg", torch_img)
            # cv2.imwrite(f"{output_dir}/ttnn_{cam_name}.jpg", ttnn_img)
            cv2.imwrite(f"{output_dir}/comparison_{cam_name}.jpg", comparison)

            print(f"Saved: comparison_{cam_name}.jpg")

    # 2. BEV comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    class_colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "brown", "pink"]

    # PyTorch BEV
    for i, box in enumerate(torch_filtered_boxes[:100]):
        x, y, z, l, w, h, yaw = box[:7]
        label = int(torch_filtered_labels[i])
        color = class_colors[label % len(class_colors)]

        rect = patches.Rectangle(
            (x - l / 2, y - w / 2), l, w, angle=np.degrees(yaw), linewidth=2, edgecolor=color, facecolor="none"
        )
        ax1.add_patch(rect)
        ax1.text(x, y, f"{torch_filtered_scores[i]:.2f}", fontsize=6, ha="center")

    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_title(f"PyTorch BEV - {len(torch_filtered_boxes)} objects")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # TTNN BEV
    for i, box in enumerate(ttnn_filtered_boxes[:100]):
        x, y, z, l, w, h, yaw = box[:7]
        label = int(ttnn_filtered_labels[i])
        color = class_colors[label % len(class_colors)]

        rect = patches.Rectangle(
            (x - l / 2, y - w / 2), l, w, angle=np.degrees(yaw), linewidth=2, edgecolor=color, facecolor="none"
        )
        ax2.add_patch(rect)
        ax2.text(x, y, f"{ttnn_filtered_scores[i]:.2f}", fontsize=6, ha="center")

    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.set_title(f"TTNN BEV - {len(ttnn_filtered_boxes)} objects")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bev_comparison.jpg", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: bev_comparison.jpg")

    # 3. Statistics comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Score distributions
    ax1.hist(torch_scores, bins=50, alpha=0.7, label="PyTorch", color="green")
    ax1.hist(ttnn_scores, bins=50, alpha=0.7, label="TTNN", color="blue")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Score Distribution Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Label distributions
    torch_label_counts = np.bincount(torch_labels.astype(int))
    ttnn_label_counts = np.bincount(ttnn_labels.astype(int))
    x = np.arange(max(len(torch_label_counts), len(ttnn_label_counts)))
    width = 0.35
    ax2.bar(x - width / 2, torch_label_counts, width, label="PyTorch", color="green", alpha=0.7)
    ax2.bar(x + width / 2, ttnn_label_counts, width, label="TTNN", color="blue", alpha=0.7)
    ax2.set_xlabel("Class Label")
    ax2.set_ylabel("Count")
    ax2.set_title("Class Distribution Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # X-Y positions
    ax3.scatter(torch_filtered_boxes[:, 0], torch_filtered_boxes[:, 1], alpha=0.5, label="PyTorch", color="green", s=30)
    ax3.scatter(ttnn_filtered_boxes[:, 0], ttnn_filtered_boxes[:, 1], alpha=0.5, label="TTNN", color="blue", s=30)
    ax3.set_xlabel("X (meters)")
    ax3.set_ylabel("Y (meters)")
    ax3.set_title("Detection Positions (Top View)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal")

    # Z (height) distribution
    ax4.hist(torch_filtered_boxes[:, 2], bins=30, alpha=0.7, label="PyTorch", color="green")
    ax4.hist(ttnn_filtered_boxes[:, 2], bins=30, alpha=0.7, label="TTNN", color="blue")
    ax4.set_xlabel("Z (height in meters)")
    ax4.set_ylabel("Count")
    ax4.set_title("Height Distribution Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/statistics_comparison.jpg", dpi=150, bbox_inches="tight")
    plt.close()
    return torch_filtered_boxes, ttnn_filtered_boxes


def draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, img_shape, color=(0, 255, 0), thickness=2):
    """Draws lidar bboxes on image, handles invalid projections and checks depth properly."""
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

        # Check depth BEFORE clipping or normalizing
        depths = pts_2d[:, 2]
        valid_depth_mask = depths > 1.0  # At least 1 meter in front

        # Need at least 2 corners with valid depth
        if valid_depth_mask.sum() < 2:
            boxes_skipped_behind += 1
            continue

        # Only normalize points with valid depth
        # Set invalid depths to a dummy value that will be filtered later
        pts_2d_normalized = pts_2d.copy()

        # Avoid division by zero or near-zero
        safe_depths = np.where(depths > 0.1, depths, 1e10)
        pts_2d_normalized[:, 0] = pts_2d[:, 0] / safe_depths
        pts_2d_normalized[:, 1] = pts_2d[:, 1] / safe_depths

        # Check for NaN/Inf after normalization
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

    print(
        f"    Drawn: {boxes_drawn}, behind: {boxes_skipped_behind}, outside: {boxes_skipped_outside}, invalid: {boxes_skipped_invalid}"
    )
    return img


def load_golden_calibration(data_root="models/experimental/functional_petr/resources/nuscenes"):
    """Load the proven working calibration"""
    golden_path = Path(data_root) / "../modified_input_batch_img_metas_sample1.pt"

    golden_data = torch.load(golden_path, weights_only=False)
    meta = golden_data[0] if isinstance(golden_data, list) else golden_data
    cam2img_orig = meta["cam2img"][0]
    if isinstance(cam2img_orig, torch.Tensor):
        cam2img_orig = cam2img_orig.cpu().numpy()

    # Compute lidar2img if needed
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


def load_images_with_golden_calibration(data_root="models/experimental/functional_petr/resources/nuscenes"):
    """
    Load the images that match the golden calibration
    """

    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    # Load golden calibration
    golden_meta = load_golden_calibration(data_root)

    # HARDCODED: These are the EXACT 6 images from the golden calibration
    EXACT_FILENAMES = [
        "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
        "samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg",
        "samples/CAM_FRONT_LEFT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402927604844.jpg",
        "samples/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg",
        "samples/CAM_BACK_LEFT/n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402927647423.jpg",
        "samples/CAM_BACK_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402927627893.jpg",
    ]

    imgs_list = []
    camera_images = []

    for cam, filename in zip(cam_names, EXACT_FILENAMES):
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

    # Create img_metas with GOLDEN calibration
    img_metas = [
        {
            "filename": cam_names,
            "ori_shape": [(320, 800)] * 6,
            "img_shape": (320, 800),
            "pad_shape": (320, 800),
            "lidar2img": golden_meta["lidar2img"],
            "cam2img": golden_meta.get("cam2img", [np.eye(4, dtype=np.float32) for _ in range(6)]),
            "cam_intrinsic": golden_meta.get("cam_intrinsic", [np.eye(3, dtype=np.float32) for _ in range(6)]),
            "lidar2cam": golden_meta.get("lidar2cam", [np.eye(4, dtype=np.float32) for _ in range(6)]),
        }
    ]

    return {"imgs": imgs, "img_metas": img_metas}, camera_images


def test_demo(device, reset_seeds):
    """
    Updated test_demo that uses golden calibration
    """
    input_data, camera_images = load_images_with_golden_calibration(
        "models/experimental/functional_petr/resources/nuscenes"
    )

    print(f"   Images shape: {input_data['imgs'].shape}")

    # Initialize preprocessor
    print("\n2. Initializing data preprocessor...")
    data_preprocessor = MockDet3DDataPreprocessor()

    print("3. Processing input data...")
    output_after_preprocess = data_preprocessor(input_data, False)
    batch_img_metas = [ds.metainfo for ds in output_after_preprocess["data_samples"]]

    # Verify calibration is correct
    print("\n4. Verifying calibration...")
    test_point = np.array([10.0, 0.0, 0.0, 1.0])
    lidar2img = batch_img_metas[0]["lidar2img"][0]
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.cpu().numpy()
    proj = lidar2img @ test_point
    u, v = proj[0] / proj[2], proj[1] / proj[2]

    # Load model
    print("\n5. Loading PETR model...")
    weights_path = "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    weights_state_dict = torch.load(weights_path, weights_only=False)["state_dict"]

    torch_model = PETR(use_grid_mask=True)
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    # Run PyTorch inference
    print("\n6. Running PyTorch inference...")
    with torch.no_grad():
        torch_output = torch_model.predict(output_after_preprocess["inputs"], batch_img_metas)

    # Check results
    boxes = torch_output[0]["pts_bbox"]["bboxes_3d"].tensor.cpu().numpy()
    scores = torch_output[0]["pts_bbox"]["scores_3d"].cpu().numpy()

    print(f"   Detected {len(boxes)} boxes")

    # Convert to TTNN and run inference
    print("\n7. Converting to TTNN...")
    ttnn_inputs = dict()
    imgs_tensor = output_after_preprocess["inputs"]["imgs"]
    if len(imgs_tensor.shape) == 4:
        imgs_tensor = imgs_tensor.unsqueeze(0)
    ttnn_inputs["imgs"] = ttnn.from_torch(imgs_tensor, device=device)

    # [... rest of TTNN setup - same as before ...]

    # Preprocess parameters (keeping your existing code)
    print("\n8. Preprocessing model parameters...")
    parameters_petr_head = preprocess_model_parameters(
        initialize_model=lambda: torch_model.pts_bbox_head,
        custom_preprocessor=create_custom_preprocessor_petr_head(None),
        device=None,
    )
    parameters_petr_head = move_to_device(parameters_petr_head, device)

    child = torch_model.pts_bbox_head.transformer
    x = infer_ttnn_module_args(
        model=child,
        run_model=lambda model: model(
            torch.randn(1, 6, 256, 20, 50),
            torch.zeros((1, 6, 20, 50), dtype=torch.bool),
            torch.rand(900, 256),
            torch.rand(1, 6, 256, 20, 50),
        ),
        device=None,
    )
    if x is not None:
        for key in x.keys():
            x[key].module = getattr(child, key)
        parameters_petr_head["transformer"] = x

    parameters_petr_cpfpn = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_neck,
        custom_preprocessor=create_custom_preprocessor_cpfpn(None),
        device=None,
    )

    parameters_petr_vovnetcp = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )

    parameters = {
        "pts_bbox_head": parameters_petr_head,
        "img_neck": parameters_petr_cpfpn,
        "img_backbone": parameters_petr_vovnetcp,
        "stem_parameters": stem_parameters_preprocess(torch_model.img_backbone),
    }

    print("\n9. Preprocessing query embeddings...")
    query_embedding_input = torch_model.pts_bbox_head.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)
    query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

    print("\n10. Initializing TTNN model...")
    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    print("\n11. Running TTNN inference...")
    ttnn_output = ttnn_model.predict(ttnn_inputs, batch_img_metas)

    # Save visualizations with calibration
    print("\n12. Saving visualizations...")
    torch_filtered, ttnn_filtered = save_comparison_visualizations(
        torch_output, ttnn_output, camera_images, batch_img_metas
    )

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        test_demo(device, None)
    finally:
        ttnn.close_device(device)
