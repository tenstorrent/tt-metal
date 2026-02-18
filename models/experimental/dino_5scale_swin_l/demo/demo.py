#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DINO-5scale Swin-L detection demo: TTNN vs PyTorch side-by-side comparison.

Runs both the full TTNN pipeline and PyTorch reference on the same images,
compares detections (IoU matching, score correlation), and saves side-by-side
visualizations.

Usage:
    export PYTHONPATH=/home/ubuntu/tt-metal:$HOME/.local/lib/python3.10/site-packages
    python models/experimental/dino_5scale_swin_l/demo/demo.py
    python models/experimental/dino_5scale_swin_l/demo/demo.py --score-thr 0.3
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger

from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H,
    DINO_INPUT_W,
    SWIN_L_EMBED_DIM,
    SWIN_L_DEPTHS,
    SWIN_L_NUM_HEADS,
    SWIN_L_WINDOW_SIZE,
    NECK_IN_CHANNELS,
    NUM_QUERIES,
    NUM_CLASSES,
    NUM_LEVELS,
    ENCODER_EMBED_DIMS,
    ENCODER_NUM_HEADS,
    ENCODER_NUM_POINTS,
    ENCODER_NUM_LAYERS,
    DECODER_NUM_LAYERS,
)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]

SAMPLE_IMAGES = {
    "cats_remotes": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "baseball": "http://images.cocodataset.org/val2017/000000037777.jpg",
    "living_room": "http://images.cocodataset.org/val2017/000000087038.jpg",
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> Tuple[torch.Tensor, Tuple[int, int], float]:
    """
    Load and preprocess image for DINO. Returns:
        tensor: [1, 3, 800, 1333] normalized
        orig_shape: (H, W)
        resize_scale: uniform scale factor
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    target_h, target_w = DINO_INPUT_H, DINO_INPUT_W
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    img_np = np.array(img_resized, dtype=np.float32)
    padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = img_np

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    padded = (padded - mean) / std

    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
    return tensor, (orig_h, orig_w), scale


# ---------------------------------------------------------------------------
# PyTorch reference inference
# ---------------------------------------------------------------------------

def run_pytorch_reference(ref_model, image_tensor, orig_shape, resize_scale, score_thr, nms_thr):
    """Run PyTorch mmdet DINO and return postprocessed detections."""
    from torchvision.ops import batched_nms

    with torch.no_grad():
        backbone_feats = ref_model.forward_backbone(image_tensor)
        neck_feats = list(ref_model.model.neck(backbone_feats))

        enc_out = ref_model.forward_encoder(neck_feats)
        det = ref_model.model
        output_memory, output_proposals = det.gen_encoder_output_proposals(
            enc_out["memory"], enc_out.get("memory_mask"), enc_out["spatial_shapes"],
        )
        enc_cls = det.bbox_head.cls_branches[det.decoder.num_layers](output_memory)
        ref_topk_indices = torch.topk(enc_cls.max(-1)[0], k=NUM_QUERIES, dim=1)[1]

        dec_out = ref_model.forward_decoder(neck_feats)
        hidden_states = dec_out["hidden_states"]
        references = dec_out["references"]
        ref_cls, ref_coords = ref_model.model.bbox_head(hidden_states, references)

    cls_scores = ref_cls[-1]  # [B, 900, 80]
    bbox_preds = ref_coords[-1]  # [B, 900, 4]

    scores = cls_scores[0].sigmoid()
    bboxes = bbox_preds[0]

    cx, cy, bw, bh = bboxes.unbind(-1)
    x1 = (cx - bw / 2) * DINO_INPUT_W
    y1 = (cy - bh / 2) * DINO_INPUT_H
    x2 = (cx + bw / 2) * DINO_INPUT_W
    y2 = (cy + bh / 2) * DINO_INPUT_H
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    max_scores, max_labels = scores.max(dim=-1)
    keep = max_scores > score_thr
    boxes = boxes[keep]
    max_scores = max_scores[keep]
    max_labels = max_labels[keep]

    if boxes.numel() > 0:
        nms_keep = batched_nms(boxes, max_scores, max_labels, nms_thr)
        nms_keep = nms_keep[:300]
        boxes = boxes[nms_keep]
        max_scores = max_scores[nms_keep]
        max_labels = max_labels[nms_keep]
        boxes /= resize_scale
        boxes[:, [0, 2]].clamp_(0, orig_shape[1])
        boxes[:, [1, 3]].clamp_(0, orig_shape[0])

    return {
        "boxes": boxes,
        "scores": max_scores,
        "labels": max_labels,
        "raw_cls": cls_scores,
        "raw_bbox": bbox_preds,
        "topk_indices": ref_topk_indices,
    }


# ---------------------------------------------------------------------------
# TTNN inference
# ---------------------------------------------------------------------------

def run_ttnn_inference(tt_model, image_tensor, orig_shape, resize_scale, score_thr, nms_thr):
    """Run full TTNN DINO pipeline and return postprocessed detections."""
    from models.experimental.dino_5scale_swin_l.tt.tt_dino import TtDINO

    result = tt_model.forward_image(image_tensor, return_intermediates=True)
    cls_scores = result["all_cls_scores"][-1]
    bbox_preds = result["all_bbox_preds"][-1]

    detections = TtDINO.postprocess(
        cls_scores, bbox_preds,
        img_shape=(DINO_INPUT_H, DINO_INPUT_W),
        score_thr=score_thr,
        nms_thr=nms_thr,
    )

    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    if boxes.numel() > 0:
        boxes /= resize_scale
        boxes[:, [0, 2]].clamp_(0, orig_shape[1])
        boxes[:, [1, 3]].clamp_(0, orig_shape[0])

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "raw_cls": cls_scores,
        "raw_bbox": bbox_preds,
        "topk_indices": result.get("topk_indices"),
    }


# ---------------------------------------------------------------------------
# Detection comparison
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """IoU between two [4] boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compare_detections(ref_dets, tt_dets, iou_thr=0.5):
    """
    Compare two sets of detections. Returns a summary dict with:
        matched: number of IoU-matched detections with same class
        ref_only: detections only in reference
        tt_only: detections only in TTNN
        score_diffs: list of (ref_score, tt_score) for matched pairs
    """
    ref_boxes = ref_dets["boxes"]
    tt_boxes = tt_dets["boxes"]
    ref_scores = ref_dets["scores"]
    tt_scores = tt_dets["scores"]
    ref_labels = ref_dets["labels"]
    tt_labels = tt_dets["labels"]

    matched_pairs = []
    ref_matched = set()
    tt_matched = set()

    for i in range(len(ref_boxes)):
        best_iou = 0
        best_j = -1
        for j in range(len(tt_boxes)):
            if j in tt_matched:
                continue
            if ref_labels[i] != tt_labels[j]:
                continue
            iou = compute_iou(ref_boxes[i].tolist(), tt_boxes[j].tolist())
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            matched_pairs.append({
                "class": COCO_CLASSES[ref_labels[i].item()],
                "ref_score": ref_scores[i].item(),
                "tt_score": tt_scores[best_j].item(),
                "iou": best_iou,
                "ref_box": ref_boxes[i].tolist(),
                "tt_box": tt_boxes[best_j].tolist(),
            })
            ref_matched.add(i)
            tt_matched.add(best_j)

    ref_only = []
    for i in range(len(ref_boxes)):
        if i not in ref_matched:
            ref_only.append({
                "class": COCO_CLASSES[ref_labels[i].item()],
                "score": ref_scores[i].item(),
                "box": ref_boxes[i].tolist(),
            })

    tt_only = []
    for j in range(len(tt_boxes)):
        if j not in tt_matched:
            tt_only.append({
                "class": COCO_CLASSES[tt_labels[j].item()],
                "score": tt_scores[j].item(),
                "box": tt_boxes[j].tolist(),
            })

    return {
        "matched": matched_pairs,
        "ref_only": ref_only,
        "tt_only": tt_only,
        "num_ref": len(ref_boxes),
        "num_tt": len(tt_boxes),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_detections_on_image(img, boxes, scores, labels, color_offset=0):
    """Draw boxes on a PIL Image (in-place). Returns the image."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    colors = {}
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        lid = label.item()
        cls_name = COCO_CLASSES[lid] if lid < len(COCO_CLASSES) else f"cls_{lid}"
        if lid not in colors:
            torch.manual_seed(lid + 42 + color_offset)
            colors[lid] = tuple(torch.randint(60, 255, (3,)).tolist())
        color = colors[lid]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{cls_name} {score.item():.2f}"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1], fill=color)
        draw.text((x1, y1), text, fill="white", font=font)
    return img


def save_side_by_side(image_path, ref_dets, tt_dets, output_path, comparison):
    """Save side-by-side visualization: PyTorch (left) vs TTNN (right) + comparison text."""
    from PIL import Image, ImageDraw, ImageFont

    img_ref = Image.open(image_path).convert("RGB")
    img_tt = img_ref.copy()

    draw_detections_on_image(img_ref, ref_dets["boxes"], ref_dets["scores"], ref_dets["labels"])
    draw_detections_on_image(img_tt, tt_dets["boxes"], tt_dets["scores"], tt_dets["labels"])

    w, h = img_ref.size
    header_h = 30
    gap = 4
    combined = Image.new("RGB", (w * 2 + gap, h + header_h), (255, 255, 255))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(combined)
    n_matched = len(comparison["matched"])
    draw.text((10, 5), f"PyTorch Reference ({comparison['num_ref']} dets)", fill="blue", font=font)
    draw.text((w + gap + 10, 5), f"TTNN ({comparison['num_tt']} dets) — {n_matched} matched", fill="green", font=font)

    combined.paste(img_ref, (0, header_h))
    combined.paste(img_tt, (w + gap, header_h))

    combined.save(output_path)
    logger.info(f"Saved side-by-side: {output_path}")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_image(url: str, dest_dir: str, name: str) -> str:
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)
    if not os.path.isfile(dest):
        logger.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest)
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DINO-5scale Swin-L: TTNN vs PyTorch Comparison Demo")
    parser.add_argument("--image", type=str, nargs="*", default=None, help="Input image path(s)")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    parser.add_argument("--nms-thr", type=float, default=0.8, help="NMS IoU threshold")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    args = parser.parse_args()

    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    demo_dir = str(base / "models/experimental/dino_5scale_swin_l/demo")
    config_path = str(base / "models/experimental/dino_5scale_swin_l/reference/dino_5scale_swin_l.py")

    ckpt_path = str(
        base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
        / "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
    )
    if not Path(ckpt_path).is_file():
        alt = str(base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l/dino_5scale_swin_l.pth")
        if Path(alt).is_file():
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Resolve images
    if args.image:
        image_paths = args.image
    else:
        image_paths = [
            download_image(url, demo_dir, f"{name}.jpg")
            for name, url in SAMPLE_IMAGES.items()
        ]

    # ---- Load PyTorch reference ----
    logger.info("Loading PyTorch reference model...")
    from models.experimental.dino_5scale_swin_l.reference.dino_staged_forward import DINOStagedForward
    ref_model = DINOStagedForward(config_path, ckpt_path)

    # ---- Load TTNN model ----
    import ttnn
    logger.info(f"Opening TT device {args.device_id}...")
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768)

    try:
        from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import (
            load_backbone_weights, load_neck_weights,
            load_encoder_weights, load_decoder_weights, compute_attn_masks,
        )
        from models.experimental.dino_5scale_swin_l.tt.tt_dino import TtDINO

        logger.info("Loading TTNN weights...")
        t0 = time.time()
        backbone_params = load_backbone_weights(
            ckpt_path, device, embed_dim=SWIN_L_EMBED_DIM,
            depths=tuple(SWIN_L_DEPTHS), num_heads=tuple(SWIN_L_NUM_HEADS),
            window_size=SWIN_L_WINDOW_SIZE,
        )
        neck_params = load_neck_weights(ckpt_path, device)
        encoder_params = load_encoder_weights(ckpt_path, device)
        decoder_params = load_decoder_weights(ckpt_path, device)
        attn_masks = compute_attn_masks(DINO_INPUT_H, DINO_INPUT_W, 4, SWIN_L_WINDOW_SIZE, device)
        logger.info(f"Weights loaded in {time.time() - t0:.1f}s")

        tt_model = TtDINO(
            encoder_params=encoder_params, decoder_params=decoder_params,
            device=device, backbone_params=backbone_params,
            neck_params=neck_params, attn_masks=attn_masks,
            num_queries=NUM_QUERIES, num_classes=NUM_CLASSES,
            num_levels=NUM_LEVELS, embed_dims=ENCODER_EMBED_DIMS,
            num_heads=ENCODER_NUM_HEADS, num_points=ENCODER_NUM_POINTS,
            encoder_num_layers=ENCODER_NUM_LAYERS, decoder_num_layers=DECODER_NUM_LAYERS,
            pe_temperature=20, embed_dim=SWIN_L_EMBED_DIM,
            depths=tuple(SWIN_L_DEPTHS), backbone_num_heads=tuple(SWIN_L_NUM_HEADS),
            window_size=SWIN_L_WINDOW_SIZE, in_channels=tuple(NECK_IN_CHANNELS),
        )

        # ---- Process each image ----
        all_summaries = []

        for img_path in image_paths:
            img_name = Path(img_path).stem
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {img_path}")
            logger.info(f"{'='*60}")

            image_tensor, orig_shape, resize_scale = preprocess_image(img_path)

            # PyTorch reference
            logger.info("Running PyTorch reference...")
            t0 = time.time()
            ref_dets = run_pytorch_reference(
                ref_model, image_tensor, orig_shape, resize_scale,
                args.score_thr, args.nms_thr,
            )
            pt_time = time.time() - t0
            logger.info(f"PyTorch: {len(ref_dets['boxes'])} detections in {pt_time:.2f}s")

            # TTNN
            logger.info("Running TTNN inference...")
            t0 = time.time()
            tt_dets = run_ttnn_inference(
                tt_model, image_tensor, orig_shape, resize_scale,
                args.score_thr, args.nms_thr,
            )
            tt_time = time.time() - t0
            logger.info(f"TTNN: {len(tt_dets['boxes'])} detections in {tt_time:.2f}s")

            # Top-K overlap on this real image
            ref_tk = ref_dets.get("topk_indices")
            tt_tk = tt_dets.get("topk_indices")
            if ref_tk is not None and tt_tk is not None:
                ref_set = set(ref_tk[0].tolist())
                tt_set = set(tt_tk[0].tolist())
                topk_overlap = len(ref_set & tt_set)
                logger.info(f"  Top-K overlap (real image): {topk_overlap}/{NUM_QUERIES} "
                            f"({topk_overlap / NUM_QUERIES * 100:.1f}%)")

            # Compare
            comparison = compare_detections(ref_dets, tt_dets, iou_thr=0.5)

            logger.info(f"\n--- Comparison for {img_name} ---")
            logger.info(f"  PyTorch detections: {comparison['num_ref']}")
            logger.info(f"  TTNN detections:    {comparison['num_tt']}")
            logger.info(f"  Matched (IoU≥0.5, same class): {len(comparison['matched'])}")

            if comparison["matched"]:
                logger.info("  Matched detections:")
                for m in comparison["matched"]:
                    logger.info(
                        f"    {m['class']:15s}  ref={m['ref_score']:.3f}  "
                        f"tt={m['tt_score']:.3f}  IoU={m['iou']:.3f}"
                    )

            if comparison["ref_only"]:
                logger.info("  PyTorch only (missed by TTNN):")
                for d in comparison["ref_only"]:
                    logger.info(f"    {d['class']:15s}  score={d['score']:.3f}")

            if comparison["tt_only"]:
                logger.info("  TTNN only (not in PyTorch):")
                for d in comparison["tt_only"]:
                    logger.info(f"    {d['class']:15s}  score={d['score']:.3f}")

            # Score correlation for matched pairs
            if comparison["matched"]:
                ref_s = torch.tensor([m["ref_score"] for m in comparison["matched"]])
                tt_s = torch.tensor([m["tt_score"] for m in comparison["matched"]])
                avg_iou = np.mean([m["iou"] for m in comparison["matched"]])
                score_diff = (ref_s - tt_s).abs().mean().item()
                logger.info(f"  Avg matched IoU: {avg_iou:.3f}")
                logger.info(f"  Avg score diff:  {score_diff:.4f}")

            # Save side-by-side
            out_path = os.path.join(demo_dir, f"{img_name}_comparison.jpg")
            save_side_by_side(img_path, ref_dets, tt_dets, out_path, comparison)

            all_summaries.append({
                "image": img_name,
                "ref_count": comparison["num_ref"],
                "tt_count": comparison["num_tt"],
                "matched": len(comparison["matched"]),
                "ref_only": len(comparison["ref_only"]),
                "tt_only": len(comparison["tt_only"]),
            })

        # ---- Overall summary ----
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"{'Image':20s} {'Ref':>4s} {'TTNN':>5s} {'Match':>6s} {'RefOnly':>8s} {'TTOnly':>7s}")
        logger.info("-" * 60)
        total_ref = total_tt = total_matched = 0
        for s in all_summaries:
            logger.info(
                f"{s['image']:20s} {s['ref_count']:4d} {s['tt_count']:5d} "
                f"{s['matched']:6d} {s['ref_only']:8d} {s['tt_only']:7d}"
            )
            total_ref += s["ref_count"]
            total_tt += s["tt_count"]
            total_matched += s["matched"]
        logger.info("-" * 60)
        match_rate = total_matched / total_ref * 100 if total_ref > 0 else 0
        logger.info(
            f"{'TOTAL':20s} {total_ref:4d} {total_tt:5d} "
            f"{total_matched:6d} ({match_rate:.1f}% match rate)"
        )

    finally:
        import ttnn
        ttnn.close_device(device)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
