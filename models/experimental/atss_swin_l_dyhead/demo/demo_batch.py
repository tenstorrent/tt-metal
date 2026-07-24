#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Batch inference on multiple images: PyTorch vs TTNN side-by-side.
Saves comparison visualizations and a summary JSON.
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

PALETTE = {}


def get_color(label_id):
    if label_id not in PALETTE:
        rng = np.random.RandomState(label_id * 7 + 3)
        PALETTE[label_id] = tuple(int(c) for c in rng.randint(80, 255, 3))
    return PALETTE[label_id]


def draw_detections(image_bgr, results, title="", score_thr=0.25):
    vis = image_bgr.copy()
    bboxes = results["bboxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()

    for i in range(len(scores)):
        if scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bboxes[i]]
        lid = int(labels[i])
        name = COCO_CLASSES[lid] if lid < len(COCO_CLASSES) else str(lid)
        color = get_color(lid)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            vis, text, (x1 + 1, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    if title:
        cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--score-thr", type=float, default=0.25)
    _default_out = str(Path(__file__).resolve().parent.parent / "results" / "batch")
    parser.add_argument("--output-dir", default=_default_out)
    args = parser.parse_args()

    from models.experimental.atss_swin_l_dyhead.common import ATSS_CHECKPOINT, ATSS_PAD_SIZE_DIVISOR

    checkpoint = args.checkpoint or ATSS_CHECKPOINT

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir)
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file()]
    )
    print(f"Found {len(image_files)} images in {image_dir}")

    # ── Build PyTorch model ──
    from models.experimental.atss_swin_l_dyhead.reference.model import build_atss_model, load_mmdet_checkpoint

    print("[PyTorch] Building model...")
    pt_model = build_atss_model()
    load_mmdet_checkpoint(pt_model, checkpoint)
    pt_model.eval()

    # ── Build TTNN model ──
    import ttnn
    from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel

    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    all_results = []

    try:
        for idx, img_path in enumerate(image_files):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[skip] Cannot read: {img_path.name}")
                continue

            h, w = img_bgr.shape[:2]
            img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).unsqueeze(0).float()
            img_shape = (h, w)

            pad_h = (ATSS_PAD_SIZE_DIVISOR - h % ATSS_PAD_SIZE_DIVISOR) % ATSS_PAD_SIZE_DIVISOR
            pad_w = (ATSS_PAD_SIZE_DIVISOR - w % ATSS_PAD_SIZE_DIVISOR) % ATSS_PAD_SIZE_DIVISOR
            padded_h, padded_w = h + pad_h, w + pad_w

            print(f"\n[{idx+1}/{len(image_files)}] {img_path.name} ({w}x{h}, padded {padded_w}x{padded_h})")

            # PyTorch inference
            t0 = time.perf_counter()
            with torch.no_grad():
                pt_results = pt_model.predict(img_tensor, img_shape, score_thr=args.score_thr)
            pt_ms = (time.perf_counter() - t0) * 1000
            pt_n = pt_results["bboxes"].shape[0]
            print(f"  [PyTorch] {pt_n} detections, {pt_ms:.0f} ms")

            # TTNN inference (rebuild model per resolution if needed)
            ttnn_model = TtATSSModel.from_checkpoint(checkpoint, device, input_h=padded_h, input_w=padded_w)
            t0 = time.perf_counter()
            with torch.no_grad():
                tt_results = ttnn_model.predict(img_tensor, img_shape, score_thr=args.score_thr)
            tt_ms = (time.perf_counter() - t0) * 1000
            tt_n = tt_results["bboxes"].shape[0]
            print(f"  [TTNN]    {tt_n} detections, {tt_ms:.0f} ms")

            # Draw side-by-side
            vis_pt = draw_detections(img_bgr, pt_results, f"PyTorch ({pt_n} det)", args.score_thr)
            vis_tt = draw_detections(img_bgr, tt_results, f"TTNN ({tt_n} det)", args.score_thr)

            combined = np.concatenate([vis_pt, vis_tt], axis=1)
            out_path = output_dir / f"{img_path.stem}_comparison.jpg"
            cv2.imwrite(str(out_path), combined)
            print(f"  Saved: {out_path}")

            all_results.append(
                {
                    "image": img_path.name,
                    "size": f"{w}x{h}",
                    "pytorch": {"detections": pt_n, "ms": round(pt_ms, 1)},
                    "ttnn": {"detections": tt_n, "ms": round(tt_ms, 1)},
                }
            )

    finally:
        ttnn.close_device(device)

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    print(f"\n{'='*60}")
    print(f"{'Image':<25} {'PyTorch':>12} {'TTNN':>12} {'Match':>8}")
    print(f"{'─'*25} {'─'*12} {'─'*12} {'─'*8}")
    for r in all_results:
        match = "Yes" if r["pytorch"]["detections"] == r["ttnn"]["detections"] else "Close"
        print(f"{r['image']:<25} {r['pytorch']['detections']:>5} det    {r['ttnn']['detections']:>5} det    {match:>6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
