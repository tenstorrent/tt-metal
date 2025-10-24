#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Run YOLO11 Pose estimation on all images in demo/images directory
Processes each image individually and saves results
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from yolov11_pose_correct import YoloV11Pose

# COCO Keypoint skeleton
SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [6, 7],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [5, 6],
    [5, 7],
]

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def preprocess_image(image_path, target_size=(640, 640)):
    """Load and preprocess image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_img = img.copy()
    h, w = img.shape[:2]

    # Resize maintaining aspect ratio
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Pad to target size
    canvas = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = img_resized

    # Convert to tensor
    img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, original_img, (scale, top, left)


def postprocess_predictions(output, original_shape, preprocess_info, conf_threshold=0.3):
    """Extract pose predictions"""
    scale, top, left = preprocess_info
    orig_h, orig_w = original_shape[:2]

    bbox = output[0, 0:4, :].cpu().numpy()
    conf = output[0, 4, :].cpu().numpy()
    keypoints = output[0, 5:56, :].cpu().numpy()

    detections = []
    for i in range(output.shape[2]):
        if conf[i] > conf_threshold:
            x_center, y_center, width, height = bbox[:, i]

            # Transform to original image space
            x_center = (x_center - left) / scale
            y_center = (y_center - top) / scale
            width = width / scale
            height = height / scale

            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(orig_w, int(x_center + width / 2))
            y2 = min(orig_h, int(y_center + height / 2))

            if x2 <= x1 or y2 <= y1:
                continue

            # Transform keypoints
            kpts = keypoints[:, i].reshape(17, 3)
            kpts_scaled = []

            for kpt in kpts:
                kx, ky, kv = kpt
                kx = (kx - left) / scale
                ky = (ky - top) / scale
                kx = max(0, min(orig_w, kx))
                ky = max(0, min(orig_h, ky))
                kpts_scaled.append([kx, ky, kv])

            detections.append(
                {"bbox": [x1, y1, x2, y2], "confidence": float(conf[i]), "keypoints": np.array(kpts_scaled)}
            )

    # Simple NMS
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    keep = []
    for det in detections:
        if len(keep) == 0:
            keep.append(det)
        else:
            # Check IoU with existing detections
            overlaps = False
            for existing in keep:
                iou = compute_iou(det["bbox"], existing["bbox"])
                if iou > 0.45:
                    overlaps = True
                    break
            if not overlaps:
                keep.append(det)

    return keep


def compute_iou(box1, box2):
    """Compute IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def visualize_pose(image, detections, output_path):
    """Draw pose"""
    img_vis = image.copy()

    colors = [
        (255, 0, 0),
        (255, 85, 0),
        (255, 170, 0),
        (255, 255, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 255, 255),
        (0, 170, 255),
        (0, 85, 255),
        (0, 0, 255),
        (85, 0, 255),
        (170, 0, 255),
        (255, 0, 255),
        (255, 0, 170),
    ]

    for det in detections:
        bbox = det["bbox"]
        keypoints = det["keypoints"]
        conf = det["confidence"]

        # Draw bbox
        cv2.rectangle(img_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            img_vis, f"Person: {conf:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Draw keypoints
        for idx, (x, y, v) in enumerate(keypoints):
            if v > 0.3:
                color = colors[idx % len(colors)]
                cv2.circle(img_vis, (int(x), int(y)), 5, color, -1)
                cv2.circle(img_vis, (int(x), int(y)), 6, (255, 255, 255), 1)

        # Draw skeleton
        for connection in SKELETON:
            kpt1_idx = connection[0] - 1
            kpt2_idx = connection[1] - 1

            if kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints):
                kpt1 = keypoints[kpt1_idx]
                kpt2 = keypoints[kpt2_idx]

                if kpt1[2] > 0.3 and kpt2[2] > 0.3:
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(img_vis, pt1, pt2, (255, 255, 0), 2)

    cv2.imwrite(output_path, img_vis)
    return img_vis


def main():
    print("=" * 70)
    print("YOLO11 Pose Estimation - Process All Images")
    print("=" * 70)

    demo_dir = Path(__file__).parent
    images_dir = demo_dir / "images"
    output_dir = demo_dir / "runs" / "pose_all"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1/3] Loading model...")
    model = YoloV11Pose()
    weights_path = demo_dir.parent / "reference" / "yolov11_pose_pretrained_correct.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    print("✓ Model loaded")

    # Find all images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"\n[2/3] Found {len(image_files)} images")

    # Process each image
    print("\n[3/3] Processing images...")
    total_people = 0

    for img_path in sorted(image_files):
        print(f"\n  Processing: {img_path.name}")

        # Preprocess
        img_tensor, original_img, preprocess_info = preprocess_image(str(img_path))

        # Inference
        with torch.no_grad():
            output = model(img_tensor)

        # Postprocess
        detections = postprocess_predictions(output, original_img.shape, preprocess_info)

        num_people = len(detections)
        total_people += num_people
        print(f"    Detected {num_people} person(s)")

        # Visualize
        output_path = output_dir / f"pose_{img_path.name}"
        visualize_pose(original_img, detections, str(output_path))
        print(f"    ✓ Saved: {output_path.name}")

        # Print keypoint details
        for i, det in enumerate(detections):
            visible_kpts = sum(1 for kpt in det["keypoints"] if kpt[2] > 0.3)
            print(f"      Person {i+1}: {visible_kpts}/17 visible keypoints (conf: {det['confidence']:.3f})")

    print("\n" + "=" * 70)
    print(f"✓ Complete! Detected {total_people} people in {len(image_files)} images")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
