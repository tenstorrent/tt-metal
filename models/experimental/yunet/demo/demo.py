# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""YUNet Face Detection Demo.

Usage:
    python models/experimental/yunet/demo/demo.py --input <image_path> [--output <output_path>]

Example:
    python models/experimental/yunet/demo/demo.py --input test.jpg --output result.jpg
    python models/experimental/yunet/demo/demo.py --input test.jpg --input-size 320
"""

import argparse
import os
import time
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


def decode_detections(
    cls_outs,
    box_outs,
    obj_outs,
    kpt_outs,
    orig_w,
    orig_h,
    input_size=YUNET_INPUT_SIZE,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD,
):
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

                x1 = int((cx - bw / 2) * orig_w / input_size)
                y1 = int((cy - bh / 2) * orig_h / input_size)
                x2 = int((cx + bw / 2) * orig_w / input_size)
                y2 = int((cy + bh / 2) * orig_h / input_size)

                keypoints = []
                for k in range(5):
                    kpt_dx = kpt_out[b, k * 2, h, w].item()
                    kpt_dy = kpt_out[b, k * 2 + 1, h, w].item()
                    kx = int((kpt_dx * stride + anchor_x) * orig_w / input_size)
                    ky = int((kpt_dy * stride + anchor_y) * orig_h / input_size)
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

    img_h, img_w = img.shape[:2]
    scale = max(img_w, img_h) / 640
    thickness = max(1, int(2 * scale))
    font_scale = max(0.3, 0.4 * scale)
    kp_radius = max(2, int(3 * scale))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(
            img, f'{det["conf"]:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness
        )
        for i, (kx, ky) in enumerate(det["keypoints"]):
            cv2.circle(img, (kx, ky), kp_radius, kp_colors[i], -1)

    return img


def run_yunet_demo(
    device, image_path: str, output_path: str = None, input_size: int = YUNET_INPUT_SIZE, num_iterations: int = 1
):
    """
    Run YUNet face detection demo on an image.

    Args:
        device: TTNN device
        image_path: Path to input image
        output_path: Optional path to save output image
        input_size: Model input size (320 or 640, default: 640)
        num_iterations: Number of iterations for timing (default: 1)

    Returns:
        Tuple of (detections, timing_info)
    """
    logger.info(f"Loading YUNet model (input size: {input_size}x{input_size})...")

    weights_path = get_default_weights_path()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. " f"Please run: cd models/experimental/yunet && ./setup.sh"
        )
    torch_model = load_torch_model(weights_path)
    torch_model = torch_model.to(torch.bfloat16)
    ttnn_model = create_yunet_model(device, torch_model)

    # Load and preprocess image
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    orig_h, orig_w = image.shape[:2]
    logger.info(f"Original image size: {orig_w}x{orig_h}")

    # Resize to model input size
    image_resized = cv2.resize(image, (input_size, input_size))
    image_rgb = image_resized[:, :, ::-1]  # BGR -> RGB
    tensor = torch.from_numpy(image_rgb.copy()).float()
    tensor_nhwc = tensor.unsqueeze(0).to(torch.bfloat16)

    # Warmup run
    logger.info("Warmup inference...")
    tt_input = ttnn.from_torch(tensor_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cls_out, box_out, obj_out, kpt_out = ttnn_model(tt_input)
    ttnn.synchronize_device(device)

    # Timed inference runs
    logger.info(f"Running {num_iterations} timed inference(s)...")
    times = []

    for i in range(num_iterations):
        tt_input = ttnn.from_torch(tensor_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        ttnn.synchronize_device(device)
        start_time = time.perf_counter()

        cls_out, box_out, obj_out, kpt_out = ttnn_model(tt_input)

        ttnn.synchronize_device(device)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000
        times.append(inference_time_ms)

    # Calculate timing stats
    avg_time_ms = sum(times) / len(times)
    min_time_ms = min(times)
    max_time_ms = max(times)
    fps = 1000.0 / avg_time_ms

    timing_info = {
        "avg_ms": avg_time_ms,
        "min_ms": min_time_ms,
        "max_ms": max_time_ms,
        "fps": fps,
        "iterations": num_iterations,
    }

    logger.info(f"")
    logger.info(f"=" * 50)
    logger.info(f"INFERENCE TIMING")
    logger.info(f"=" * 50)
    logger.info(f"  Iterations: {num_iterations}")
    logger.info(f"  Avg time:   {avg_time_ms:.2f} ms")
    logger.info(f"  Min time:   {min_time_ms:.2f} ms")
    logger.info(f"  Max time:   {max_time_ms:.2f} ms")
    logger.info(f"  FPS:        {fps:.1f}")
    logger.info(f"=" * 50)
    logger.info(f"")

    # Decode detections
    detections = decode_detections(cls_out, box_out, obj_out, kpt_out, orig_w, orig_h, input_size=input_size)
    logger.info(f"Detected {len(detections)} faces")

    # Visualize
    if output_path:
        output_image = draw_detections(image, detections)
        cv2.imwrite(output_path, output_image)
        logger.info(f"Saved output to: {output_path}")

    return detections, timing_info


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YUNET_L1_SMALL_SIZE}],
    indirect=True,
)
def test_yunet_demo(device, input_size):
    """Test YUNet demo with sample image."""
    if isinstance(input_size, tuple):
        input_size = input_size[0]

    image_path = "models/experimental/yunet/test_images/group1.jpg"
    output_path = f"models/experimental/yunet/demo_output_{input_size}.jpg"

    detections, timing = run_yunet_demo(device, image_path, output_path, input_size=input_size)

    assert len(detections) > 0, "No faces detected"
    logger.info(f"Demo completed successfully with {len(detections)} detections")


def main():
    """Command-line entry point for YUNet face detection demo."""
    parser = argparse.ArgumentParser(
        description="YUNet Face Detection Demo on Tenstorrent Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (640x640)
    python models/experimental/yunet/demo/demo.py --input face.jpg --output result.jpg

    # With 320x320 input size
    python models/experimental/yunet/demo/demo.py --input face.jpg --output result.jpg --input-size 320

    # Benchmark with multiple iterations
    python models/experimental/yunet/demo/demo.py --input face.jpg --iterations 100
        """,
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save output image")
    parser.add_argument(
        "--input-size",
        "-s",
        type=int,
        default=YUNET_INPUT_SIZE,
        choices=[320, 640],
        help=f"Model input size (default: {YUNET_INPUT_SIZE})",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=10, help="Number of iterations for timing (default: 10)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input image not found: {args.input}")
        return 1

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_detected_{args.input_size}{ext}"
        logger.info(f"Output will be saved to: {args.output}")

    logger.info("Initializing Tenstorrent device...")
    device = ttnn.open_device(device_id=0, l1_small_size=YUNET_L1_SMALL_SIZE)

    try:
        detections, timing = run_yunet_demo(
            device, args.input, args.output, input_size=args.input_size, num_iterations=args.iterations
        )

        logger.info("=" * 50)
        logger.info("DETECTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Input:  {args.input}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Input size: {args.input_size}x{args.input_size}")
        logger.info(f"Faces detected: {len(detections)}")
        logger.info(f"Inference: {timing['avg_ms']:.2f} ms ({timing['fps']:.1f} FPS)")

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            logger.info(f"  Face {i+1}: box=({x1},{y1},{x2},{y2}), conf={det['conf']:.3f}")

        logger.info("=" * 50)

    finally:
        ttnn.close_device(device)
        logger.info("Device closed.")

    return 0


if __name__ == "__main__":
    exit(main())
