#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ATSS Swin-L DyHead inference demo.

Runs object detection on an input image using:
  1. PyTorch reference model (CPU)
  2. TTNN hybrid model (device + CPU DyHead fallback)

Compares detection outputs side-by-side and optionally saves visualizations.

Usage:
  cd $TT_METAL_HOME
  source python_env/bin/activate
  export ARCH_NAME=wormhole_b0
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  export TT_METAL_HOME=$(pwd)
  export PYTHONPATH=$(pwd):$HOME/.local/lib/python3.10/site-packages

  python3 models/experimental/atss_swin_l_dyhead/demo/demo_inference.py \
      --image path/to/your/image.jpg \
      --output-dir path/to/output
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_image(image_path: str):
    """Load image as BGR numpy array and as float tensor for model input."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img_bgr.shape[:2]
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0).float()
    return img_bgr, img_tensor, (h, w)


def draw_detections(image_bgr, results, title="", score_thr=0.3):
    """Draw bounding boxes on image."""
    vis = image_bgr.copy()
    bboxes = results["bboxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()

    colors = {}
    for i in range(len(scores)):
        if scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bboxes[i]]
        label_id = int(labels[i])
        label_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else str(label_id)

        if label_id not in colors:
            rng = np.random.RandomState(label_id + 1)
            colors[label_id] = tuple(int(c) for c in rng.randint(60, 255, 3))
        color = colors[label_id]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label_name} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), color, -1)
        cv2.putText(vis, text, (x1, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if title:
        cv2.putText(vis, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def run_pytorch_inference(checkpoint_path, img_tensor, img_shape, score_thr=0.3):
    """Run inference with standalone PyTorch reference model."""
    from models.experimental.atss_swin_l_dyhead.reference.model import (
        build_atss_model,
        load_mmdet_checkpoint,
    )

    print("[PyTorch] Building model...")
    model = build_atss_model()
    load_mmdet_checkpoint(model, checkpoint_path)
    model.eval()

    print("[PyTorch] Warming up...")
    with torch.no_grad():
        _ = model.predict(img_tensor, img_shape, score_thr=score_thr)

    print("[PyTorch] Running inference...")
    t0 = time.perf_counter()
    with torch.no_grad():
        results = model.predict(img_tensor, img_shape, score_thr=score_thr)
    t1 = time.perf_counter()

    elapsed = (t1 - t0) * 1000
    n_det = results["bboxes"].shape[0]
    print(f"[PyTorch] {n_det} detections in {elapsed:.1f} ms")
    return results, elapsed, model


def run_ttnn_inference(checkpoint_path, img_tensor, img_shape, device, score_thr=0.3):
    """Run inference with TTNN hybrid model."""
    from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel
    from models.experimental.atss_swin_l_dyhead.common import ATSS_PAD_SIZE_DIVISOR

    _, _, h, w = img_tensor.shape
    pad_h = (ATSS_PAD_SIZE_DIVISOR - h % ATSS_PAD_SIZE_DIVISOR) % ATSS_PAD_SIZE_DIVISOR
    pad_w = (ATSS_PAD_SIZE_DIVISOR - w % ATSS_PAD_SIZE_DIVISOR) % ATSS_PAD_SIZE_DIVISOR
    padded_h, padded_w = h + pad_h, w + pad_w

    print(f"[TTNN] Building model (input={padded_h}x{padded_w})...")
    ttnn_model = TtATSSModel.from_checkpoint(checkpoint_path, device, input_h=padded_h, input_w=padded_w)

    print("[TTNN] Warming up...")
    with torch.no_grad():
        _ = ttnn_model.predict(img_tensor, img_shape, score_thr=score_thr)

    print("[TTNN] Running inference...")
    t0 = time.perf_counter()
    with torch.no_grad():
        results = ttnn_model.predict(img_tensor, img_shape, score_thr=score_thr)
    t1 = time.perf_counter()

    elapsed = (t1 - t0) * 1000
    n_det = results["bboxes"].shape[0]
    print(f"[TTNN] {n_det} detections in {elapsed:.1f} ms")
    return results, elapsed, ttnn_model


def compare_results(ref_results, ttnn_results):
    """Compare detection results between reference and TTNN."""
    ref_n = ref_results["bboxes"].shape[0]
    ttnn_n = ttnn_results["bboxes"].shape[0]
    print(f"\n{'='*60}")
    print(f"Detection count: PyTorch={ref_n}, TTNN={ttnn_n}")

    if ref_n > 0 and ttnn_n > 0:
        n = min(ref_n, ttnn_n, 20)
        print(f"\nTop {n} detections comparison:")
        print(f"{'#':>3} | {'PyTorch':^40} | {'TTNN':^40}")
        print(f"{'-'*3}-+-{'-'*40}-+-{'-'*40}")

        for i in range(n):
            ref_label = int(ref_results["labels"][i])
            ref_score = float(ref_results["scores"][i])
            ref_bbox = ref_results["bboxes"][i].tolist()
            ref_name = COCO_CLASSES[ref_label] if ref_label < len(COCO_CLASSES) else str(ref_label)

            ttnn_label = int(ttnn_results["labels"][i])
            ttnn_score = float(ttnn_results["scores"][i])
            ttnn_bbox = ttnn_results["bboxes"][i].tolist()
            ttnn_name = COCO_CLASSES[ttnn_label] if ttnn_label < len(COCO_CLASSES) else str(ttnn_label)

            ref_str = f"{ref_name:>12} {ref_score:.3f}"
            ttnn_str = f"{ttnn_name:>12} {ttnn_score:.3f}"
            print(f"{i+1:>3} | {ref_str:^40} | {ttnn_str:^40}")

        if ref_n > 0 and ttnn_n > 0:
            n_common = min(ref_n, ttnn_n)
            bbox_diff = (ref_results["bboxes"][:n_common] - ttnn_results["bboxes"][:n_common]).abs()
            score_diff = (ref_results["scores"][:n_common] - ttnn_results["scores"][:n_common]).abs()
            print(f"\nBBox mean abs diff (top {n_common}): {bbox_diff.mean():.4f}")
            print(f"Score mean abs diff (top {n_common}): {score_diff.mean():.6f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="ATSS Swin-L DyHead inference demo")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--checkpoint", default=None, help="mmdet checkpoint path")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    _default_out = str(Path(__file__).resolve().parent.parent / "results" / "demo")
    parser.add_argument("--output-dir", default=_default_out, help="Output directory")
    parser.add_argument("--skip-ttnn", action="store_true", help="Skip TTNN inference")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch inference")
    args = parser.parse_args()

    from models.experimental.atss_swin_l_dyhead.common import ATSS_CHECKPOINT

    checkpoint = args.checkpoint or ATSS_CHECKPOINT
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    img_bgr, img_tensor, img_shape = load_image(args.image)
    print(f"Image: {args.image} ({img_shape[1]}x{img_shape[0]})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_results = None
    ttnn_results = None

    if not args.skip_pytorch:
        ref_results, ref_ms, _ = run_pytorch_inference(checkpoint, img_tensor, img_shape, args.score_thr)
        vis_ref = draw_detections(img_bgr, ref_results, f"PyTorch ({ref_ms:.0f}ms)", args.score_thr)
        ref_path = output_dir / "pytorch_detections.jpg"
        cv2.imwrite(str(ref_path), vis_ref)
        print(f"Saved: {ref_path}")

    if not args.skip_ttnn:
        import ttnn

        device = ttnn.open_device(device_id=0, l1_small_size=32768)

        try:
            ttnn_results, ttnn_ms, _ = run_ttnn_inference(checkpoint, img_tensor, img_shape, device, args.score_thr)
            vis_ttnn = draw_detections(img_bgr, ttnn_results, f"TTNN ({ttnn_ms:.0f}ms)", args.score_thr)
            ttnn_path = output_dir / "ttnn_detections.jpg"
            cv2.imwrite(str(ttnn_path), vis_ttnn)
            print(f"Saved: {ttnn_path}")
        finally:
            ttnn.close_device(device)

    if ref_results is not None and ttnn_results is not None:
        compare_results(ref_results, ttnn_results)

    summary = {
        "image": args.image,
        "image_shape": list(img_shape),
        "score_thr": args.score_thr,
    }
    if ref_results is not None:
        summary["pytorch"] = {
            "detections": int(ref_results["bboxes"].shape[0]),
            "inference_ms": round(ref_ms, 1),
        }
    if ttnn_results is not None:
        summary["ttnn"] = {
            "detections": int(ttnn_results["bboxes"].shape[0]),
            "inference_ms": round(ttnn_ms, 1),
        }

    summary_path = output_dir / "demo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
