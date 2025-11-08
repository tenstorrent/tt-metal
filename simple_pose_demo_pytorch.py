#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, "/home/ubuntu/pose/tt-metal")

import cv2
import numpy as np
import torch
from loguru import logger

from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.utils.common_demo_utils import LoadImages, preprocess


def run_pytorch_pose_demo(image_path):
    """Run pose estimation using PyTorch model only (no TTNN)"""

    logger.info(f"Running PyTorch pose estimation on: {image_path}")

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    scale = min(640 / orig_w, 640 / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize and pad
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.zeros((640, 640, 3), dtype=np.uint8)
    padded_img[:new_h, :new_w] = resized_img

    # Convert to tensor
    img_tensor = torch.from_numpy(padded_img).float().permute(2, 0, 1) / 255.0

    # Load PyTorch model (working version)
    logger.info("Loading PyTorch pose model...")
    torch_model = YoloV11Pose()
    weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"
    if os.path.exists(weights_path):
        torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info("Loaded pretrained weights")
    else:
        logger.warning(f"Pretrained weights not found at {weights_path}")
    torch_model.eval()

    # Run inference
    logger.info("Running PyTorch inference...")
    with torch.no_grad():
        torch_input = img_tensor.unsqueeze(0)  # Add batch dimension
        torch_output = torch_model(torch_input)

        logger.info(f"PyTorch output shape: {torch_output.shape}")
        logger.info(f"PyTorch output range: [{torch_output.min():.3f}, {torch_output.max():.3f}]")

        # Debug: Check raw PyTorch output values
        print(f"Raw PyTorch output shape: {torch_output.shape}")
        print(f"Raw PyTorch output sample bbox: {torch_output[0, :5, 0]}")  # First 5 bbox values
        print(f"Raw PyTorch output sample kpts: {torch_output[0, 64:69, 0]}")  # First 5 keypoint values

    # Postprocess and visualize
    visualize_pose_results_pytorch(
        image_path, torch_output, orig_w, orig_h, scale, (640 - new_w) // 2, (640 - new_h) // 2
    )


def visualize_pose_results_pytorch(image_path, predictions, orig_w, orig_h, scale, pad_left, pad_top):
    """Visualize pose estimation results from PyTorch"""

    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image for visualization: {image_path}")
        return

    # Process predictions (PyTorch format)
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

        # Process keypoints (PyTorch keypoints are already decoded)
        kpts = keypoints[:, j].reshape(17, 3)
        kpts_transformed = []

        visible_count = 0
        for kpt_idx, kpt in enumerate(kpts):
            kx, ky, kv = kpt

            # Debug: print first few keypoints before transformation
            if detections_count == 1 and kpt_idx < 5:
                print(f"PyTorch Keypoint {kpt_idx}: decoded=({kx:.1f}, {ky:.1f}), visibility={kv:.3f}")

            # Transform to original image coordinates (no decoding needed)
            kx_final = (kx - pad_left) / scale
            ky_final = (ky - pad_top) / scale

            # Debug: print transformed coordinates
            if detections_count == 1 and kpt_idx < 5:
                print(f"  -> PyTorch final=({kx_final:.1f}, {ky_final:.1f})")

            # Ensure within bounds
            kx_final = max(0, min(orig_w, kx_final))
            ky_final = max(0, min(orig_h, ky_final))

            if kv > 0.3:  # visible
                visible_count += 1
                color = colors[kpt_idx % len(colors)]
                cv2.circle(img, (int(kx_final), int(ky_final)), 6, color, -1)  # filled circle
                cv2.circle(img, (int(kx_final), int(ky_final)), 8, (255, 255, 255), 2)  # white outline

            kpts_transformed.append([kx_final, ky_final, kv])

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
    output_path = f"models/demos/yolov11/demo/runs/pose/{base_name}_pose_pytorch.jpg"
    cv2.imwrite(output_path, img)
    logger.info(f"PyTorch pose estimation result saved to: {output_path}")


if __name__ == "__main__":
    # Test on dog.jpg
    image_path = "models/demos/yolov11/demo/images/dog.jpg"
    if os.path.exists(image_path):
        run_pytorch_pose_demo(image_path)
        logger.info("PyTorch pose estimation demo completed successfully!")
    else:
        logger.error(f"Image not found: {image_path}")
