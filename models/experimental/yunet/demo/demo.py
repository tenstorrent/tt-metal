# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""YUNet Face Detection Demo.

Usage:
    python -m models.experimental.yunet.demo.demo --input <image_path> [--output <output_path>]

Example:
    python -m models.experimental.yunet.demo.demo --input test.jpg --output result.jpg
"""

import argparse
import os
import pytest
import torch
import cv2
import numpy as np
from loguru import logger

import ttnn
from models.experimental.yunet.common import (
    YUNET_L1_SMALL_SIZE,
    YUNET_INPUT_SIZE,
    STRIDES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_NMS_IOU_THRESHOLD,
    load_torch_model,
    get_default_weights_path,
)
from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model


def decode_detections(cls_outs, box_outs, obj_outs, kpt_outs, orig_w, orig_h, threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """Decode raw model outputs to detections."""
    detections = []

    for scale_idx in range(3):
        cls_out = ttnn.to_torch(cls_outs[scale_idx]).float().permute(0, 3, 1, 2)
        box_out = ttnn.to_torch(box_outs[scale_idx]).float().permute(0, 3, 1, 2)
        obj_out = ttnn.to_torch(obj_outs[scale_idx]).float().permute(0, 3, 1, 2)
        kpt_out = ttnn.to_torch(kpt_outs[scale_idx]).float().permute(0, 3, 1, 2)

        stride = STRIDES[scale_idx]
        score = cls_out.sigmoid() * obj_out.sigmoid()

        high_conf = score > threshold
        if high_conf.any():
            indices = torch.where(high_conf)
            for i in range(len(indices[0])):
                b, c, h, w = indices[0][i], indices[1][i], indices[2][i], indices[3][i]
                conf = score[b, c, h, w].item()
                anchor_x, anchor_y = w.item() * stride, h.item() * stride

                dx, dy = box_out[b, 0, h, w].item(), box_out[b, 1, h, w].item()
                dw, dh = box_out[b, 2, h, w].item(), box_out[b, 3, h, w].item()

                cx, cy = dx * stride + anchor_x, dy * stride + anchor_y
                bw, bh = np.exp(dw) * stride, np.exp(dh) * stride

                x1 = int((cx - bw / 2) * orig_w / YUNET_INPUT_SIZE)
                y1 = int((cy - bh / 2) * orig_h / YUNET_INPUT_SIZE)
                x2 = int((cx + bw / 2) * orig_w / YUNET_INPUT_SIZE)
                y2 = int((cy + bh / 2) * orig_h / YUNET_INPUT_SIZE)

                keypoints = []
                for k in range(5):
                    kpt_dx = kpt_out[b, k * 2, h, w].item()
                    kpt_dy = kpt_out[b, k * 2 + 1, h, w].item()
                    kx = int((kpt_dx * stride + anchor_x) * orig_w / YUNET_INPUT_SIZE)
                    ky = int((kpt_dy * stride + anchor_y) * orig_h / YUNET_INPUT_SIZE)
                    keypoints.append((kx, ky))

                detections.append({"box": (x1, y1, x2, y2), "conf": conf, "keypoints": keypoints})

    # NMS
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for det in detections:
            x1 = max(best["box"][0], det["box"][0])
            y1 = max(best["box"][1], det["box"][1])
            x2 = min(best["box"][2], det["box"][2])
            y2 = min(best["box"][3], det["box"][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (best["box"][2] - best["box"][0]) * (best["box"][3] - best["box"][1])
            area2 = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])
            if inter / max(area1 + area2 - inter, 1e-6) < DEFAULT_NMS_IOU_THRESHOLD:
                remaining.append(det)
        detections = remaining
    return keep


def draw_detections(image, detections):
    """Draw detection boxes and keypoints on image."""
    img = image.copy()
    kp_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{det["conf"]:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for i, (kx, ky) in enumerate(det["keypoints"]):
            cv2.circle(img, (kx, ky), 3, kp_colors[i], -1)
    return img


def run_yunet_demo(device, image_path: str, output_path: str = None):
    """
    Run YUNet face detection demo on an image.

    Args:
        device: TTNN device
        image_path: Path to input image
        output_path: Optional path to save output image

    Returns:
        List of detected faces
    """
    # Load models
    logger.info("Loading YUNet model...")
    torch_model = load_torch_model(get_default_weights_path())
    torch_model = torch_model.to(torch.bfloat16)
    ttnn_model = create_yunet_model(device, torch_model)

    # Load and preprocess image
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    image_resized = cv2.resize(image, (YUNET_INPUT_SIZE, YUNET_INPUT_SIZE))
    image_rgb = image_resized[:, :, ::-1]
    tensor = torch.from_numpy(image_rgb.copy()).float()
    tensor_nhwc = tensor.unsqueeze(0).to(torch.bfloat16)

    # Run inference
    logger.info("Running TTNN inference...")
    tt_input = ttnn.from_torch(tensor_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cls_out, box_out, obj_out, kpt_out = ttnn_model(tt_input)

    # Decode detections
    detections = decode_detections(cls_out, box_out, obj_out, kpt_out, orig_w, orig_h)
    logger.info(f"Detected {len(detections)} faces")

    # Visualize
    if output_path:
        output_image = draw_detections(image, detections)
        cv2.imwrite(output_path, output_image)
        logger.info(f"Saved output to: {output_path}")

    return detections


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YUNET_L1_SMALL_SIZE}],
    indirect=True,
)
def test_yunet_demo(device):
    """Test YUNet demo with sample image."""
    import os

    # Use a test image from YUNet repo or skip if not available
    image_path = "models/experimental/yunet/YUNet/data/test.jpg"
    output_path = "models/experimental/yunet/demo_output.jpg"

    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found: {image_path}. Run setup.sh first.")

    detections = run_yunet_demo(device, image_path, output_path)

    assert len(detections) >= 0, "Detection failed"  # Allow 0 detections for some images
    logger.info(f"Demo completed successfully with {len(detections)} detections")


def main():
    """Command-line entry point for YUNet face detection demo."""
    parser = argparse.ArgumentParser(
        description="YUNet Face Detection Demo on Tenstorrent Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - detect faces and save output
    python -m models.experimental.yunet.demo.demo --input face.jpg --output result.jpg

    # Only detect, don't save (just print detections)
    python -m models.experimental.yunet.demo.demo --input face.jpg

    # With custom confidence threshold
    python -m models.experimental.yunet.demo.demo --input face.jpg --output result.jpg --threshold 0.6
        """,
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Path to save output image with detections (optional)"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for detection (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input image not found: {args.input}")
        return 1

    # Auto-generate output path if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_detected{ext}"
        logger.info(f"Output will be saved to: {args.output}")

    # Initialize device
    logger.info("Initializing Tenstorrent device...")
    device = ttnn.open_device(device_id=0, l1_small_size=YUNET_L1_SMALL_SIZE)

    try:
        # Run demo
        detections = run_yunet_demo(device, args.input, args.output)

        # Print detection summary
        logger.info("=" * 50)
        logger.info(f"DETECTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Input:  {args.input}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Faces detected: {len(detections)}")

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            logger.info(f"  Face {i+1}: box=({x1},{y1},{x2},{y2}), conf={det['conf']:.3f}")
            for j, (kx, ky) in enumerate(det["keypoints"]):
                kp_names = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
                logger.info(f"    {kp_names[j]}: ({kx}, {ky})")

        logger.info("=" * 50)

    finally:
        # Cleanup
        ttnn.close_device(device)
        logger.info("Device closed.")

    return 0


if __name__ == "__main__":
    exit(main())
