#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import sys
import os
import json

# Add paths
sys.path.insert(0, "/home/ubuntu/pose/tt-metal")

from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose
import ttnn
from loguru import logger


def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for pose estimation"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    orig_h, orig_w = img.shape[:2]

    # Calculate scaling and padding
    scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    padded_img[:new_h, :new_w] = resized_img

    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(padded_img).float().permute(2, 0, 1) / 255.0

    return img_tensor, orig_w, orig_h, scale, (target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2


def run_simple_pose_demo(image_path):
    """Run pose estimation on a single image"""

    logger.info(f"Running pose estimation on: {image_path}")

    # Load and preprocess image
    img_tensor, orig_w, orig_h, scale, pad_left, pad_top = preprocess_image(image_path)

    # Load PyTorch model for reference
    logger.info("Loading PyTorch model...")
    torch_model = YoloV11Pose()
    weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"
    if os.path.exists(weights_path):
        torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        logger.warning(f"Pretrained weights not found at {weights_path}")
    torch_model.eval()

    # Get PyTorch reference output
    logger.info("Running PyTorch inference...")
    with torch.no_grad():
        torch_input = img_tensor.unsqueeze(0)  # Add batch dimension
        torch_output = torch_model(torch_input)

    logger.info(f"PyTorch output shape: {torch_output.shape}")

    # Now try TTNN inference
    logger.info("Setting up TTNN model...")
    device = ttnn.open_device(device_id=0)

    try:
        # Create TTNN model
        parameters = create_yolov11_pose_model_parameters(torch_model, torch_input, device=device)
        ttnn_model = TtnnYoloV11Pose(device, parameters)

        # Convert input to TTNN format and move to device
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_input = ttnn.to_device(tt_input, device)

        # Run TTNN inference
        logger.info("Running TTNN inference...")
        tt_output = ttnn_model(tt_input)

        # Convert back to torch
        output_tensor = ttnn.to_torch(tt_output)

        logger.info(f"TTNN output shape: {output_tensor.shape}")

        # Postprocess and visualize
        visualize_pose_results(image_path, output_tensor, orig_w, orig_h, scale, pad_left, pad_top)

    finally:
        ttnn.close_device(device)


def visualize_pose_results(image_path, predictions, orig_w, orig_h, scale, pad_left, pad_top):
    """Visualize pose estimation results"""

    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image for visualization: {image_path}")
        return

    # Process predictions (similar to demo code)
    # predictions shape: [1, 56, num_anchors]
    # Format: [bbox(4) + conf(1) + keypoints(51)] = 56

    batch_size, num_features, num_anchors = predictions.shape
    logger.info(f"Processing {num_anchors} anchor predictions")

    # Extract components
    bbox = predictions[0, :4, :]  # [4, num_anchors]
    conf = predictions[0, 4:5, :]  # [1, num_anchors]
    keypoints = predictions[0, 5:56, :]  # [51, num_anchors] - 17 keypoints * 3 values each

    # Find detections above confidence threshold
    conf_threshold = 0.3
    valid_indices = torch.where(conf[0] > conf_threshold)[0]

    logger.info(f"Found {len(valid_indices)} detections above confidence threshold {conf_threshold}")

    # Colors for keypoints (COCO format)
    colors = [
        (255, 0, 0),  # nose - red
        (0, 255, 0),  # eyes - green
        (0, 255, 0),
        (0, 0, 255),  # ears - blue
        (0, 0, 255),
        (255, 255, 0),  # shoulders - cyan
        (255, 255, 0),
        (255, 0, 255),  # elbows - magenta
        (255, 0, 255),
        (0, 255, 255),  # wrists - yellow
        (0, 255, 255),
        (128, 0, 128),  # hips - purple
        (128, 0, 128),
        (0, 128, 128),  # knees - teal
        (0, 128, 128),
        (128, 128, 0),  # ankles - olive
        (128, 128, 0),
    ]

    # COCO pose skeleton connections
    skeleton = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ]

    detections_count = 0

    for j in valid_indices[:5]:  # Limit to first 5 detections
        detections_count += 1

        # Get bounding box (already in 640x640 space)
        x, y, w, h = bbox[:, j]

        # Transform bbox to original image space
        x = (x - pad_left) / scale
        y = (y - pad_top) / scale
        w = w / scale
        h = h / scale

        x1 = int(max(0, x - w / 2))
        y1 = int(max(0, y - h / 2))
        x2 = int(min(orig_w, x + w / 2))
        y2 = int(min(orig_h, y + h / 2))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Person {detections_count}: {conf[0, j]:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Process keypoints
        kpts = keypoints[:, j].reshape(17, 3)
        kpts_transformed = []

        visible_count = 0
        for kpt_idx, kpt in enumerate(kpts):
            kx, ky, kv = kpt

            # Decode keypoints (same as demo)
            kv_decoded = 1.0 / (1.0 + np.exp(-kv))  # sigmoid
            kx_decoded = (kx * 2.0 - 0.5) * 100.0
            ky_decoded = (ky * 2.0 - 0.5) * 100.0

            # Transform to original image coordinates
            kx_final = (kx_decoded - pad_left) / scale
            ky_final = (ky_decoded - pad_top) / scale

            # Ensure within bounds
            kx_final = max(0, min(orig_w, kx_final))
            ky_final = max(0, min(orig_h, ky_final))

            if kv_decoded > 0.3:  # visible
                visible_count += 1
                color = colors[kpt_idx % len(colors)]
                cv2.circle(img, (int(kx_final), int(ky_final)), 6, color, -1)  # filled circle
                cv2.circle(img, (int(kx_final), int(ky_final)), 8, (255, 255, 255), 2)  # white outline

            kpts_transformed.append([kx_final, ky_final, kv_decoded])

        # Draw skeleton connections
        for connection in skeleton:
            kpt1_idx = connection[0]
            kpt2_idx = connection[1]

            if (
                kpt1_idx < len(kpts_transformed)
                and kpt2_idx < len(kpts_transformed)
                and kpts_transformed[kpt1_idx][2] > 0.3
                and kpts_transformed[kpt2_idx][2] > 0.3
            ):
                pt1 = (int(kpts_transformed[kpt1_idx][0]), int(kpts_transformed[kpt1_idx][1]))
                pt2 = (int(kpts_transformed[kpt2_idx][0]), int(kpts_transformed[kpt2_idx][1]))
                cv2.line(img, pt1, pt2, (255, 255, 0), 3)  # yellow lines

        logger.info(f"Person {detections_count}: {visible_count}/17 keypoints visible")

    # Save result
    os.makedirs("models/demos/yolov11/demo/runs/pose", exist_ok=True)
    base_name = os.path.basename(image_path).rsplit(".", 1)[0]
    output_path = f"models/demos/yolov11/demo/runs/pose/{base_name}_pose.jpg"
    cv2.imwrite(output_path, img)
    logger.info(f"Pose estimation result saved to: {output_path}")


if __name__ == "__main__":
    # Test on dog.jpg
    image_path = "models/demos/yolov11/demo/images/dog.jpg"
    if os.path.exists(image_path):
        run_simple_pose_demo(image_path)
    else:
        logger.error(f"Image not found: {image_path}")
