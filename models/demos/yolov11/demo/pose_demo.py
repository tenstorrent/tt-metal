#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO11 Pose Estimation Demo

This demo runs pose estimation on images and visualizes the detected keypoints.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add reference directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "reference"))
from yolov11_pose_correct import YoloV11Pose

# COCO Keypoint connections (skeleton)
SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],  # Legs
    [6, 12],
    [7, 13],  # Torso
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],  # Arms
    [6, 7],  # Shoulders
    [1, 2],
    [1, 3],
    [2, 3],
    [3, 5],
    [2, 4],  # Face (adjusted for 1-based)
    [1, 6],
    [1, 7],  # Neck to shoulders
]

# Keypoint names for visualization
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
    """Load and preprocess image for YOLO inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_img = img.copy()
    h, w = img.shape[:2]

    # Resize while maintaining aspect ratio
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


def nms(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping detections."""
    if len(detections) == 0:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    keep = []
    while len(detections) > 0:
        # Take the detection with highest confidence
        best = detections.pop(0)
        keep.append(best)

        # Filter out overlapping detections
        filtered = []
        for det in detections:
            iou = compute_iou(best["bbox"], det["bbox"])
            if iou < iou_threshold:
                filtered.append(det)
        detections = filtered

    return keep


def compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def postprocess_predictions(output, original_shape, preprocess_info, conf_threshold=0.5, nms_threshold=0.5):
    """Extract pose predictions from model output."""
    scale, top, left = preprocess_info
    orig_h, orig_w = original_shape[:2]

    # Output shape: [1, 56, num_anchors]
    bbox = output[0, 0:4, :].cpu().numpy()  # [4, num_anchors]
    conf = output[0, 4, :].cpu().numpy()  # [num_anchors]
    keypoints = output[0, 5:56, :].cpu().numpy()  # [51, num_anchors]

    detections = []
    for i in range(output.shape[2]):
        if conf[i] > conf_threshold:
            # Unscale bounding box (these are already in absolute pixel coords on 640x640)
            x_center, y_center, width, height = bbox[:, i]

            # Adjust for preprocessing to get original image coords
            x_center = (x_center - left) / scale
            y_center = (y_center - top) / scale
            width = width / scale
            height = height / scale

            # Convert to corner format
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(orig_w, int(x_center + width / 2))
            y2 = min(orig_h, int(y_center + height / 2))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract and decode keypoints
            # Keypoints are now in absolute pixel coordinates (same as bbox)
            # Just need to apply the same transformation as bbox
            kpts = keypoints[:, i].reshape(17, 3)
            kpts_scaled = []

            for kpt_idx, kpt in enumerate(kpts):
                kx, ky, kv = kpt
                # Apply same transformation as bbox (adjust for padding and scaling)
                kx = (kx - left) / scale
                ky = (ky - top) / scale
                # Ensure within bounds
                kx = max(0, min(orig_w, kx))
                ky = max(0, min(orig_h, ky))
                kpts_scaled.append([kx, ky, kv])

            detections.append(
                {"bbox": [x1, y1, x2, y2], "confidence": float(conf[i]), "keypoints": np.array(kpts_scaled)}
            )

    # Apply NMS to remove overlapping detections
    detections = nms(detections, nms_threshold)

    return detections


def visualize_pose(image, detections, output_path):
    """Draw pose keypoints and skeleton on image."""
    img_vis = image.copy()

    # Colors for visualization
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

        # Draw bounding box
        cv2.rectangle(img_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            img_vis, f"Person: {conf:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Draw keypoints (lower threshold to see if any keypoints exist)
        visible_count = 0
        for idx, (x, y, v) in enumerate(keypoints):
            if v > 0.3:  # Lower threshold to catch more keypoints
                visible_count += 1
                color = colors[idx % len(colors)]
                cv2.circle(img_vis, (int(x), int(y)), 5, color, -1)
                cv2.circle(img_vis, (int(x), int(y)), 6, (255, 255, 255), 1)
                # Add keypoint number for debugging
                cv2.putText(img_vis, str(idx), (int(x) + 8, int(y) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if visible_count == 0:
            print(f"    WARNING: No visible keypoints (all confidence < 0.3)")
            print(f"    Keypoint confidences: {[f'{v:.2f}' for x,y,v in keypoints]}")

        # Draw skeleton
        for connection in SKELETON:
            kpt1_idx = connection[0] - 1  # Convert to 0-based
            kpt2_idx = connection[1] - 1

            if kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints):
                kpt1 = keypoints[kpt1_idx]
                kpt2 = keypoints[kpt2_idx]

                if kpt1[2] > 0.3 and kpt2[2] > 0.3:  # Both keypoints visible (lower threshold)
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(img_vis, pt1, pt2, (255, 255, 0), 2)

    cv2.imwrite(output_path, img_vis)
    print(f"✓ Saved visualization to: {output_path}")
    return img_vis


def main():
    print("=" * 70)
    print("YOLO11 Pose Estimation Demo")
    print("=" * 70)

    # Setup paths
    demo_dir = Path(__file__).parent
    images_dir = demo_dir / "images"
    output_dir = demo_dir / "runs" / "pose"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1/4] Loading YOLO11 Pose model...")
    model = YoloV11Pose()

    # Try to load pretrained weights
    weights_path = demo_dir.parent / "reference" / "yolov11_pose_pretrained_correct.pth"
    if weights_path.exists():
        print(f"  Loading pretrained weights from: {weights_path.name}")
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print("  ✓ Pretrained weights loaded successfully!")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load pretrained weights: {e}")
            print("  ⚠ Using random weights (detections will be meaningless)")
    else:
        print("  ⚠ No pretrained weights found (using random initialization)")
        print(f"  ⚠ Run: cd {demo_dir.parent / 'reference'} && python3 load_weights_correct.py")
        print("  ⚠ Detections will be random/meaningless without pretrained weights!")

    model.eval()
    print("✓ Model loaded successfully")

    # Find images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(ext)))

    if not image_files:
        print(f"\n❌ No images found in {images_dir}")
        print("Please add some images to the demo/images directory")
        return

    print(f"\n[2/4] Found {len(image_files)} image(s) to process")

    # Process each image
    print("\n[3/4] Running pose estimation...")
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")

        # Preprocess
        img_tensor, original_img, preprocess_info = preprocess_image(str(image_path))

        # Inference
        with torch.no_grad():
            output = model(img_tensor)

        # Postprocess with NMS
        detections = postprocess_predictions(
            output,
            original_img.shape,
            preprocess_info,
            conf_threshold=0.3,  # Confidence threshold (lowered to catch more detections)
            nms_threshold=0.45,  # NMS IoU threshold (lower = fewer overlapping detections)
        )

        print(f"  Detected {len(detections)} person(s)")

        # Debug: Show keypoint info
        for i, det in enumerate(detections):
            visible_kpts = sum(1 for kpt in det["keypoints"] if kpt[2] > 0.5)
            print(f"    Person {i+1}: {visible_kpts}/17 visible keypoints (conf: {det['confidence']:.3f})")

        # Visualize
        output_path = output_dir / f"pose_{image_path.name}"
        visualize_pose(original_img, detections, str(output_path))

        # Print keypoint details
        for i, det in enumerate(detections):
            print(f"\n  Person {i+1} (confidence: {det['confidence']:.3f}):")
            for kpt_idx, kpt in enumerate(det["keypoints"]):
                if kpt[2] > 0.5:
                    print(f"    {KEYPOINT_NAMES[kpt_idx]:15s}: ({kpt[0]:6.1f}, {kpt[1]:6.1f}) vis={kpt[2]:.3f}")

    print("\n" + "=" * 70)
    print(f"[4/4] ✓ Pose estimation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
