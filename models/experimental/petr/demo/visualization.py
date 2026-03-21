# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from models.experimental.petr.reference.utils import LiDARInstance3DBoxes
from models.experimental.petr.tt.model_preprocessing import generate_petr_inputs


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
    _, meta_data = generate_petr_inputs()
    meta = meta_data[0] if isinstance(meta_data, list) else meta_data
    cam2img_orig = meta["cam2img"][0]
    if isinstance(cam2img_orig, torch.Tensor):
        cam2img_orig = cam2img_orig.to(torch.float32).cpu().numpy()

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

    test_point = np.array([10.0, 0.0, 0.0, 1.0])
    lidar2img = meta["lidar2img"][0]
    if isinstance(lidar2img, torch.Tensor):
        lidar2img = lidar2img.to(torch.float32).cpu().numpy()

    proj = lidar2img @ test_point
    u, v = proj[0] / proj[2], proj[1] / proj[2]

    return meta


def load_images_with_calibration(data_root="models/experimental/petr/resources/sample_input"):
    cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
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

        img = cv2.resize(img, (800, 320))
        camera_images.append(img.copy())

        img = img.astype(np.float32)
        mean = np.array([103.530, 116.280, 123.675])
        std = np.array([57.375, 57.120, 58.395])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        imgs_list.append(img)

    imgs = np.stack(imgs_list).astype(np.float32)
    imgs = torch.from_numpy(imgs).unsqueeze(0)

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
    def __init__(
        self, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False, pad_size_divisor=32
    ):
        self.mean = torch.tensor(mean).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 1, 3, 1, 1)
        self.bgr_to_rgb = bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor

    def __call__(self, data, training=False):
        if isinstance(data, str):
            data = torch.load(data)

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

        if imgs.max() > 10.0:
            imgs = (imgs - self.mean) / self.std

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

            metainfo.setdefault("img_shape", (h, w))
            metainfo.setdefault("pad_shape", (h + pad_h, w + pad_w))
            metainfo.setdefault("ori_shape", [(h, w)] * 6)

            metainfo["box_type_3d"] = LiDARInstance3DBoxes
            metainfo.setdefault("sample_idx", i)

            sample.metainfo = metainfo
            output["data_samples"].append(sample)

        return output


def get_box_corners_3d(box):
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
    h, w = image_shape[:2]

    point_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])

    proj = lidar2img @ point_homo

    if proj[2] <= 0:
        return None, None, False

    u = proj[0] / proj[2]
    v = proj[1] / proj[2]

    in_image = (0 <= u < w) and (0 <= v < h)

    return u, v, in_image


def draw_box_simple(img, box, lidar2img, color=(0, 255, 0), thickness=1, label_text=""):
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

    if label_text and len(corners_2d) > 0:
        label_pt = corners_2d.min(axis=0).astype(np.int32)
        label_pt[1] = max(10, label_pt[1] - 5)

        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        cv2.rectangle(
            img,
            (label_pt[0] - 2, label_pt[1] - text_h - 2),
            (label_pt[0] + text_w + 2, label_pt[1] + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            img,
            (label_pt[0] - 2, label_pt[1] - text_h - 2),
            (label_pt[0] + text_w + 2, label_pt[1] + baseline),
            color,
            1,
        )
        cv2.putText(img, label_text, tuple(label_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return img


def create_bev_plot(predictions, threshold=0.4, title=""):
    pred = predictions[0]["pts_bbox"]
    boxes = pred["bboxes_3d"].tensor.to(torch.float32).cpu().numpy()
    scores = pred["scores_3d"].to(torch.float32).cpu().numpy()
    labels = pred["labels_3d"].to(torch.float32).cpu().numpy()

    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    ax.set_xlabel("X (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=12, fontweight="bold")
    ax.set_title(f"BEV - {title} ({len(boxes)} detections)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_facecolor("#f0f0f0")

    ego_rect = plt.Rectangle((-2, -1), 4, 2, fill=True, facecolor="red", edgecolor="darkred", linewidth=2, alpha=0.7)
    ax.add_patch(ego_rect)
    ax.text(0, 0, "EGO", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    class_colors = {
        0: "#FF1493",
        1: "#FF4500",
        2: "#FFD700",
        3: "#00CED1",
        4: "#9370DB",
        8: "#00FF00",
    }

    for box, score, label in zip(boxes, scores, labels):
        x, y, z, w, l, h, yaw = box[:7]
        corners_2d = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]])
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_2d = corners_2d @ rotation.T + np.array([x, y])
        corners_2d = np.vstack([corners_2d, corners_2d[0]])

        color = class_colors.get(int(label), "#FF00FF")
        ax.plot(corners_2d[:, 0], corners_2d[:, 1], color=color, linewidth=2.5)
        ax.fill(corners_2d[:, 0], corners_2d[:, 1], facecolor=color, alpha=0.15)

        direction = np.array([l / 2 * cos_yaw, l / 2 * sin_yaw])
        ax.arrow(
            x,
            y,
            direction[0],
            direction[1],
            head_width=0.5,
            head_length=0.3,
            fc=color,
            ec=color,
            linewidth=2,
            alpha=0.8,
        )

        class_name = NUSCENES_CLASSES[int(label)]
        ax.text(
            x,
            y,
            f"{class_name}\n{score:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=color, linewidth=1.5),
        )

    x_range = 60
    y_range = 60
    ax.set_xlim([-10, x_range])
    ax.set_ylim([-y_range / 2, y_range / 2])

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    return cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)


def visualize_on_images(predictions, camera_images, img_metas, threshold=0.2):
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

    vis_images = []
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

        class_colors = {
            0: (0, 255, 0),
            1: (255, 165, 0),
            2: (255, 255, 0),
            3: (0, 191, 255),
            4: (147, 112, 219),
            5: (255, 0, 0),
            6: (255, 20, 147),
            7: (0, 255, 255),
            8: (173, 255, 47),
            9: (255, 105, 180),
        }

        for box, score, label in zip(boxes, scores, labels):
            class_name = NUSCENES_CLASSES[int(label)]
            label_text = f"{class_name} {score:.2f}"
            color = class_colors.get(int(label), (0, 255, 0))
            vis_img = draw_box_simple(vis_img, box, lidar2img, color=color, thickness=2, label_text=label_text)

        vis_images.append(vis_img)

    return vis_images


def create_combined_visualization(
    torch_predictions, ttnn_predictions, camera_images, img_metas, output_path, threshold=0.4, include_torch=False
):
    logger.info("Creating visualization...")

    ttnn_vis_imgs = visualize_on_images(ttnn_predictions, camera_images, img_metas, threshold)
    ttnn_bev = create_bev_plot(ttnn_predictions, threshold, "TTNN")

    cam_h, cam_w = 400, 600
    for i in range(len(ttnn_vis_imgs)):
        ttnn_vis_imgs[i] = cv2.resize(ttnn_vis_imgs[i], (cam_w, cam_h))

    bev_w = 900
    bev_h = cam_h * 2
    ttnn_bev = cv2.resize(ttnn_bev, (bev_w, bev_h))

    row1_ttnn = np.hstack([ttnn_vis_imgs[2], ttnn_vis_imgs[0], ttnn_vis_imgs[1]])
    row2_ttnn = np.hstack([ttnn_vis_imgs[4], ttnn_vis_imgs[3], ttnn_vis_imgs[5]])

    cam_grid_ttnn = np.vstack([row1_ttnn, row2_ttnn])
    ttnn_section = np.hstack([cam_grid_ttnn, ttnn_bev])

    label_h = 50
    ttnn_pred = ttnn_predictions[0]["pts_bbox"]
    ttnn_count = (ttnn_pred["scores_3d"].to(torch.float32).cpu().numpy() >= threshold).sum()

    if include_torch:
        torch_vis_imgs = visualize_on_images(torch_predictions, camera_images, img_metas, threshold)
        torch_bev = create_bev_plot(torch_predictions, threshold, "PyTorch")

        for i in range(len(torch_vis_imgs)):
            torch_vis_imgs[i] = cv2.resize(torch_vis_imgs[i], (cam_w, cam_h))

        torch_bev = cv2.resize(torch_bev, (bev_w, bev_h))

        row1_torch = np.hstack([torch_vis_imgs[2], torch_vis_imgs[0], torch_vis_imgs[1]])
        row2_torch = np.hstack([torch_vis_imgs[4], torch_vis_imgs[3], torch_vis_imgs[5]])

        cam_grid_torch = np.vstack([row1_torch, row2_torch])
        torch_section = np.hstack([cam_grid_torch, torch_bev])

        label_torch = np.ones((label_h, torch_section.shape[1], 3), dtype=np.uint8) * 240
        label_ttnn = np.ones((label_h, ttnn_section.shape[1], 3), dtype=np.uint8) * 240

        torch_pred = torch_predictions[0]["pts_bbox"]
        torch_count = (torch_pred["scores_3d"].to(torch.float32).cpu().numpy() >= threshold).sum()

        cv2.putText(
            label_torch, f"PyTorch ({torch_count} detections)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
        )
        cv2.putText(
            label_ttnn, f"TTNN ({ttnn_count} detections)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
        )

        final_image = np.vstack([label_torch, torch_section, label_ttnn, ttnn_section])
        logger.info(f"PyTorch detections: {torch_count}, TTNN detections: {ttnn_count}")
    else:
        label_ttnn = np.ones((label_h, ttnn_section.shape[1], 3), dtype=np.uint8) * 240
        cv2.putText(
            label_ttnn, f"TTNN ({ttnn_count} detections)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
        )
        final_image = np.vstack([label_ttnn, ttnn_section])
        logger.info(f"TTNN detections: {ttnn_count}")

    cv2.imwrite(output_path, final_image)
    logger.info(f"Visualization saved to: {output_path}")
