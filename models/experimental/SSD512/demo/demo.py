# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import torch
import ttnn
from loguru import logger

from models.experimental.SSD512.common import SSD512_L1_SMALL_SIZE, SSD512_NUM_CLASSES, load_torch_model
from models.experimental.SSD512.demo.processing import (
    draw_detections,
    filter_top_detections,
    load_image,
    run_ssd_inference,
)
from models.experimental.SSD512.reference.config import voc
from models.experimental.SSD512.reference.prior_box import PriorBox
from models.experimental.SSD512.tt.tt_ssd import TtSSD


def main():
    parser = argparse.ArgumentParser(description="SSD512 Demo")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/experimental/SSD512/resources/",
        help="Path to save output image (default: input_image with _ttnn suffix)",
    )
    parser.add_argument("--conf_thresh", type=float, default=0.02)
    parser.add_argument("--nms_thresh", type=float, default=0.45)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_detections", type=int, default=1)
    args = parser.parse_args()

    # Random seed (same as conftest.py reset_seeds fixture)
    torch.manual_seed(213919)

    # Initialize model
    torch_model = load_torch_model(phase="test", size=512, num_classes=SSD512_NUM_CLASSES)
    device = ttnn.open_device(device_id=0, l1_small_size=SSD512_L1_SMALL_SIZE)
    torch_input = torch.randn(1, 3, 512, 512)
    ttnn_model = TtSSD(torch_model, torch_input, device, batch_size=1)

    # Generate prior boxes
    prior_box = PriorBox(voc["SSD512"])
    priors_torch = prior_box.forward()

    # Load and process image
    img_path = Path(args.input_image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")

    image_tensor, original_img = load_image(str(img_path))

    # Run inference
    output_tensor = run_ssd_inference(
        ttnn_model,
        image_tensor,
        priors_torch,
        device,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        top_k=args.top_k,
    )

    # Filter and convert to Dict format for drawing
    detections = filter_top_detections(output_tensor, max_detections=args.max_detections, min_score=args.conf_thresh)

    # Save results
    output_path = args.output_path + f"{img_path.stem}_ttnn.jpg"
    draw_detections(original_img.copy(), detections, str(output_path), "SSD512")
    logger.info(f"Demo completed! Results saved to: {output_path}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
