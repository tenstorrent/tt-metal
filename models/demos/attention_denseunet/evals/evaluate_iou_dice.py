# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate Attention DenseUNet with IoU/Dice metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.attention_denseunet.reference.model import create_attention_denseunet
from models.demos.attention_denseunet.tt.common import (
    ATTENTION_DENSEUNET_L1_SMALL_SIZE,
    ATTENTION_DENSEUNET_TRACE_SIZE,
    create_preprocessor,
)
from models.demos.attention_denseunet.tt.config import OptimizationLevel, create_configs_from_parameters
from models.demos.attention_denseunet.tt.model import create_model_from_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Attention DenseUNet IoU/Dice metrics")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory with input images")
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Optional directory with GT masks (matching file stem with image-dir)",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint for reference model")
    parser.add_argument("--height", type=int, default=256, help="Resize height")
    parser.add_argument("--width", type=int, default=256, help="Resize width")
    parser.add_argument(
        "--optimization-level",
        choices=[level.value for level in OptimizationLevel],
        default=OptimizationLevel.STAGE2.value,
        help="Optimization level for TTNN path",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold on sigmoid output")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON metrics file")
    return parser.parse_args()


def load_rgb_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((width, height))
    image_np = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def load_mask(path: Path, height: int, width: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((width, height))
    mask_np = np.asarray(mask).astype(np.float32) / 255.0
    return (mask_np > 0.5).astype(np.uint8)


def to_binary_mask(logits_or_probs: torch.Tensor, threshold: float, assume_probs: bool = False) -> np.ndarray:
    if assume_probs:
        probs = logits_or_probs
    else:
        probs = torch.sigmoid(logits_or_probs)
    return (probs[0, 0].detach().cpu().numpy() > threshold).astype(np.uint8)


def iou_and_dice(pred: np.ndarray, target: np.ndarray):
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    intersection = np.logical_and(pred_bool, target_bool).sum(dtype=np.float64)
    pred_sum = pred_bool.sum(dtype=np.float64)
    target_sum = target_bool.sum(dtype=np.float64)
    union = np.logical_or(pred_bool, target_bool).sum(dtype=np.float64)

    iou = intersection / (union + 1e-8)
    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
    return float(iou), float(dice)


def collect_images(image_dir: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(image_dir.glob(ext))
    return sorted(files)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    optimization_level = OptimizationLevel(args.optimization_level)

    if not image_dir.exists():
        raise FileNotFoundError(f"image-dir does not exist: {image_dir}")
    if mask_dir is not None and not mask_dir.exists():
        raise FileNotFoundError(f"mask-dir does not exist: {mask_dir}")

    image_paths = collect_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    logger.info(f"Found {len(image_paths)} images in {image_dir}")

    reference_model = create_attention_denseunet()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        reference_model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning("No checkpoint provided; model uses random weights.")
    reference_model.eval()

    device = ttnn.open_device(
        device_id=0,
        l1_small_size=ATTENTION_DENSEUNET_L1_SMALL_SIZE,
        trace_region_size=ATTENTION_DENSEUNET_TRACE_SIZE,
        num_command_queues=2,
    )

    try:
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_preprocessor(device),
            device=None,
        )
        configs = create_configs_from_parameters(
            parameters=parameters,
            in_channels=3,
            out_channels=1,
            input_height=args.height,
            input_width=args.width,
            batch_size=1,
            optimization_level=optimization_level,
        )
        ttnn_model = create_model_from_configs(configs, device)

        per_image = []
        for image_path in image_paths:
            x = load_rgb_tensor(image_path, args.height, args.width)

            with torch.no_grad():
                pytorch_logits = reference_model(x).float()

            tt_input = x.reshape(1, 1, 3, args.height * args.width)
            tt_input = ttnn.from_torch(
                tt_input,
                dtype=ttnn.bfloat16,
                device=device,
                memory_config=configs.l1_input_memory_config,
            )
            ttnn_logits = ttnn.to_torch(ttnn_model(tt_input)).reshape(1, 1, args.height, args.width).float()

            pytorch_mask = to_binary_mask(pytorch_logits, args.threshold, assume_probs=False)
            ttnn_mask = to_binary_mask(ttnn_logits, args.threshold, assume_probs=False)

            ref_iou, ref_dice = iou_and_dice(ttnn_mask, pytorch_mask)
            entry = {
                "image": image_path.name,
                "ttnn_vs_pytorch_iou": ref_iou,
                "ttnn_vs_pytorch_dice": ref_dice,
            }

            if mask_dir is not None:
                mask_path = mask_dir / image_path.name
                if not mask_path.exists():
                    # Allow same stem with different extension.
                    mask_candidates = list(mask_dir.glob(f"{image_path.stem}.*"))
                    if not mask_candidates:
                        raise FileNotFoundError(f"No GT mask found for {image_path.name} in {mask_dir}")
                    mask_path = mask_candidates[0]

                gt_mask = load_mask(mask_path, args.height, args.width)
                ttnn_iou, ttnn_dice = iou_and_dice(ttnn_mask, gt_mask)
                pt_iou, pt_dice = iou_and_dice(pytorch_mask, gt_mask)
                entry.update(
                    {
                        "ttnn_vs_gt_iou": ttnn_iou,
                        "ttnn_vs_gt_dice": ttnn_dice,
                        "pytorch_vs_gt_iou": pt_iou,
                        "pytorch_vs_gt_dice": pt_dice,
                    }
                )

            per_image.append(entry)

        # Aggregate summary.
        summary = {
            "num_images": len(per_image),
            "mean_ttnn_vs_pytorch_iou": float(np.mean([x["ttnn_vs_pytorch_iou"] for x in per_image])),
            "mean_ttnn_vs_pytorch_dice": float(np.mean([x["ttnn_vs_pytorch_dice"] for x in per_image])),
        }
        if mask_dir is not None:
            summary.update(
                {
                    "mean_ttnn_vs_gt_iou": float(np.mean([x["ttnn_vs_gt_iou"] for x in per_image])),
                    "mean_ttnn_vs_gt_dice": float(np.mean([x["ttnn_vs_gt_dice"] for x in per_image])),
                    "mean_pytorch_vs_gt_iou": float(np.mean([x["pytorch_vs_gt_iou"] for x in per_image])),
                    "mean_pytorch_vs_gt_dice": float(np.mean([x["pytorch_vs_gt_dice"] for x in per_image])),
                }
            )

        logger.info("==== Evaluation Summary ====")
        for k, v in summary.items():
            logger.info(f"{k}: {v}")

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"summary": summary, "per_image": per_image}
            output_path.write_text(json.dumps(payload, indent=2))
            logger.info(f"Wrote metrics JSON to {output_path}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

