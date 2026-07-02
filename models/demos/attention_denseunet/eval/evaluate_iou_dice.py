# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""IoU/Dice for Attention DenseUNet: TTNN vs PyTorch; optional vs GT masks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
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
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=str, required=True)
    p.add_argument("--mask-dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--optimization-level", choices=[x.value for x in OptimizationLevel], default="stage2")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def load_rgb_tensor(path: Path, height: int, width: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((width, height))
    np_img = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)


def load_mask(path: Path, height: int, width: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((width, height))
    m = np.asarray(mask).astype(np.float32) / 255.0
    return (m > 0.5).astype(np.uint8)


def to_bin(logits: torch.Tensor, threshold: float) -> np.ndarray:
    p = torch.sigmoid(logits)
    return (p[0, 0].detach().cpu().numpy() > threshold).astype(np.uint8)


def iou_dice(a: np.ndarray, b: np.ndarray):
    ab = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(ab, bb).sum(dtype=np.float64)
    union = np.logical_or(ab, bb).sum(dtype=np.float64)
    iou = inter / (union + 1e-8)
    dice = (2.0 * inter) / (ab.sum() + bb.sum() + 1e-8)
    return float(iou), float(dice)


def collect_images(d: Path):
    out = []
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        out.extend(d.glob(pat))
    return sorted(out)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    opt = OptimizationLevel(args.optimization_level)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if mask_dir and not mask_dir.is_dir():
        raise FileNotFoundError(mask_dir)

    paths = collect_images(image_dir)
    if not paths:
        raise RuntimeError(f"No images in {image_dir}")

    ref = create_attention_denseunet()
    if args.checkpoint:
        ref.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
        logger.info(f"Loaded {args.checkpoint}")
    else:
        logger.warning("No checkpoint — random weights; vs-GT metrics are not meaningful.")
    ref.eval()

    device = ttnn.open_device(
        device_id=0,
        l1_small_size=ATTENTION_DENSEUNET_L1_SMALL_SIZE,
        trace_region_size=ATTENTION_DENSEUNET_TRACE_SIZE,
        num_command_queues=2,
    )
    try:
        params = preprocess_model_parameters(
            initialize_model=lambda: ref,
            custom_preprocessor=create_preprocessor(device),
            device=None,
        )
        configs = create_configs_from_parameters(
            parameters=params,
            in_channels=3,
            out_channels=1,
            input_height=args.height,
            input_width=args.width,
            batch_size=1,
            optimization_level=opt,
        )
        tt_model = create_model_from_configs(configs, device)
        rows = []
        for p in paths:
            x = load_rgb_tensor(p, args.height, args.width)
            with torch.no_grad():
                pt_logits = ref(x).float()
            tt_in = x.reshape(1, 1, 3, args.height * args.width)
            tt_in = ttnn.from_torch(
                tt_in, dtype=ttnn.bfloat16, device=device, memory_config=configs.l1_input_memory_config
            )
            tt_logits = ttnn.to_torch(tt_model(tt_in)).reshape(1, 1, args.height, args.width).float()
            pm = to_bin(pt_logits, args.threshold)
            tm = to_bin(tt_logits, args.threshold)
            iou_pt, dice_pt = iou_dice(tm, pm)
            entry = {"image": p.name, "ttnn_vs_pytorch_iou": iou_pt, "ttnn_vs_pytorch_dice": dice_pt}
            if mask_dir:
                mp = mask_dir / p.name
                if not mp.exists():
                    cand = list(mask_dir.glob(f"{p.stem}.*"))
                    if not cand:
                        raise FileNotFoundError(mask_dir / p.stem)
                    mp = cand[0]
                gt = load_mask(mp, args.height, args.width)
                ti, td = iou_dice(tm, gt)
                pi, pd = iou_dice(pm, gt)
                entry.update(
                    ttnn_vs_gt_iou=ti,
                    ttnn_vs_gt_dice=td,
                    pytorch_vs_gt_iou=pi,
                    pytorch_vs_gt_dice=pd,
                )
            rows.append(entry)

        summary = {
            "num_images": len(rows),
            "mean_ttnn_vs_pytorch_iou": float(np.mean([r["ttnn_vs_pytorch_iou"] for r in rows])),
            "mean_ttnn_vs_pytorch_dice": float(np.mean([r["ttnn_vs_pytorch_dice"] for r in rows])),
        }
        if mask_dir:
            summary.update(
                mean_ttnn_vs_gt_iou=float(np.mean([r["ttnn_vs_gt_iou"] for r in rows])),
                mean_ttnn_vs_gt_dice=float(np.mean([r["ttnn_vs_gt_dice"] for r in rows])),
                mean_pytorch_vs_gt_iou=float(np.mean([r["pytorch_vs_gt_iou"] for r in rows])),
                mean_pytorch_vs_gt_dice=float(np.mean([r["pytorch_vs_gt_dice"] for r in rows])),
            )
        logger.info("Summary: {}", summary)
        if args.output_json:
            outp = Path(args.output_json)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2))
            logger.info(f"Wrote {outp}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
