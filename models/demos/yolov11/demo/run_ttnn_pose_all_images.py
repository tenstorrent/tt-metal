#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Run YOLO11 Pose with TTNN on all images
Processes each image individually with TT-Metal hardware acceleration
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.pose_postprocessing import decode_pose_keypoints_cpu
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose

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


def postprocess_and_visualize(output, original_img, preprocess_info, output_path, conf_threshold=0.3):
    """Postprocess predictions and visualize"""
    scale, top, left = preprocess_info
    orig_h, orig_w = original_img.shape[:2]

    bbox = output[0, 0:4, :].cpu().numpy()
    conf = output[0, 4, :].cpu().numpy()
    keypoints = output[0, 5:56, :].cpu().numpy()

    # Extract detections
    detections = []
    for i in range(output.shape[2]):
        if conf[i] > conf_threshold:
            x_center, y_center, width, height = bbox[:, i]

            # Transform to original space
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
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:10]  # Top 10

    # Visualize
    img_vis = original_img.copy()
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
            kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1
            if kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints):
                kpt1, kpt2 = keypoints[kpt1_idx], keypoints[kpt2_idx]
                if kpt1[2] > 0.3 and kpt2[2] > 0.3:
                    cv2.line(img_vis, (int(kpt1[0]), int(kpt1[1])), (int(kpt2[0]), int(kpt2[1])), (255, 255, 0), 2)

    cv2.imwrite(output_path, img_vis)
    return len(detections)


def main():
    print("=" * 70)
    print("YOLO11 Pose Estimation - TTNN on All Images")
    print("=" * 70)

    # Open TT device
    logger.info("Opening TT device...")
    device = ttnn.open_device(device_id=0)

    demo_dir = Path(__file__).parent
    images_dir = demo_dir / "images"
    output_dir = demo_dir / "runs" / "pose_ttnn_all"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    logger.info("Loading PyTorch model...")
    torch_model = YoloV11Pose()
    weights_path = demo_dir.parent / "reference" / "yolov11_pose_pretrained_correct.pth"
    torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    torch_model.eval()

    # Create TTNN model
    logger.info("Creating TTNN model...")
    dummy_input = torch.randn(1, 3, 640, 640)
    parameters = create_yolov11_pose_model_parameters(torch_model, dummy_input, device=device)
    ttnn_model = TtnnYoloV11Pose(device, parameters)
    logger.info("✓ TTNN model ready")

    # Get anchors/strides for decoding
    anchors = ttnn_model.pose_head.anchors
    strides = ttnn_model.pose_head.strides
    anchors_torch = ttnn.to_torch(anchors)
    strides_torch = ttnn.to_torch(strides)

    # Remove extra dimensions
    while anchors_torch.dim() > 2:
        anchors_torch = anchors_torch.squeeze(0)
    while strides_torch.dim() > 2:
        strides_torch = strides_torch.squeeze(0)
    if anchors_torch.shape[0] != 2 and anchors_torch.shape[1] == 2:
        anchors_torch = anchors_torch.transpose(0, 1)
    if strides_torch.dim() == 1:
        strides_torch = strides_torch.unsqueeze(0)

    # Find images
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    logger.info(f"Found {len(image_files)} images")

    # Process each image
    total_people = 0
    for img_path in image_files:
        logger.info(f"\nProcessing: {img_path.name}")

        # Preprocess
        img_tensor, original_img, preprocess_info = preprocess_image(str(img_path))

        # Convert to TTNN
        ttnn_input = ttnn.from_torch(
            img_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Configure input
        input_mem_config = ttnn.create_sharded_memory_config(
            [1, 16, 640, 640],  # Will be padded from 3 to 16
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        ttnn_input = ttnn_input.to(device, input_mem_config)

        # Run TTNN inference
        logger.info("  Running TTNN inference...")
        ttnn_output = ttnn_model(ttnn_input)

        # Convert to torch
        output_torch = ttnn.to_torch(ttnn_output, dtype=torch.float32)

        # Decode keypoints on CPU
        logger.info("  Decoding keypoints on CPU...")
        output_decoded = decode_pose_keypoints_cpu(output_torch, anchors_torch, strides_torch)

        # Postprocess and visualize
        output_path = str(output_dir / f"pose_{img_path.name}")
        num_people = postprocess_and_visualize(output_decoded, original_img, preprocess_info, output_path)

        total_people += num_people
        logger.info(f"  ✓ Detected {num_people} people, saved to: pose_{img_path.name}")

    # Close device
    ttnn.close_device(device)

    print("\n" + "=" * 70)
    print(f"✓ TTNN Pose Demo Complete!")
    print(f"  Processed: {len(image_files)} images")
    print(f"  Detected: {total_people} people total")
    print(f"  Results: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
