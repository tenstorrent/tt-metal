# SPDX-License-Identifier: Apache-2.0
"""RF-DETR benchmark harness (STABLE — do not modify per experiment protocol).

Measures the current `TtRfDetr` model end-to-end on device #1 and reports the three
keys the experiment greps for:

    inference_speed: <frames/sec>   (median over N warm iterations, device-synchronized)
    accuracy:        <percent>      (100 * min PCC of logits & pred_boxes vs fp32 reference)
    peak_dram:       <MiB>          (best-effort device DRAM peak)

The harness always imports and runs `models.experimental.rf_detr.tt.ttnn_rf_detr.TtRfDetr`,
so model optimizations are picked up without touching this file. Accuracy is the hard
gate (must stay >= 99.0); inference_speed is the optimization objective.

Usage: python3 models/experimental/rf_detr/tests/perf/benchmark.py [--device-id N] [--iters N]
"""
import argparse
import statistics
import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.rf_detr.reference.weights import get_preprocessor, load_rf_detr_base
from models.experimental.rf_detr.tt.ttnn_rf_detr import TtRfDetr

IMAGE = "/home/ttuser/experiments/rf-detr/assets/cats_000000039769.jpg"


def _pcc_value(golden, calc):
    _, msg = comp_pcc(golden, calc, 0.99)
    # comp_pcc returns (bool, "PCC: x" or float-like); extract a float
    try:
        return float(str(msg).split("PCC:")[-1].strip().rstrip(","))
    except Exception:
        try:
            return float(msg)
        except Exception:
            return float("nan")


def _cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def _iou(a, b):
    """a: [N,4] xyxy, b: [M,4] xyxy -> [N,M] IoU."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def detection_accuracy(ref_logits, ref_boxes, tt_logits, tt_boxes, score_thresh=0.25, min_k=5):
    """Order-invariant detection agreement vs the fp32 reference (the ground truth).

    For each *confident* reference detection (score > thresh), find the best same-label
    IoU among the ttnn detections; accuracy = 100 * mean(best IoU). 100 == identical
    detections; a label miss or box drift lowers it. Robust to the discrete query
    permutation. Low-confidence (~noise) predictions are excluded — they are not
    reported detections and must not gate the optimization loop.
    """
    ref_s, ref_l = ref_logits.sigmoid()[0].max(-1)
    keep = ref_s > score_thresh
    if int(keep.sum()) < min_k:  # only pad when almost nothing is confident
        keep = torch.zeros_like(ref_s, dtype=torch.bool)
        keep[ref_s.topk(min(min_k, ref_s.numel())).indices] = True
    ref_b = _cxcywh_to_xyxy(ref_boxes[0][keep])
    ref_lab = ref_l[keep]
    tt_s, tt_l = tt_logits.sigmoid()[0].max(-1)
    tt_b = _cxcywh_to_xyxy(tt_boxes[0])
    ious = []
    for i in range(ref_b.shape[0]):
        same = tt_l == ref_lab[i]
        if bool(same.any()):
            ious.append(float(_iou(ref_b[i : i + 1], tt_b[same]).max()))
        else:
            ious.append(0.0)
    return 100.0 * (sum(ious) / max(len(ious), 1))


def _peak_dram_mib(device):
    try:
        mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return mv.total_bytes_allocated_per_bank * mv.num_banks / (1024 * 1024)
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0)
    ref, cfg = load_rf_detr_base()
    ref = ref.eval()
    pre = get_preprocessor(cfg)

    try:
        from PIL import Image

        pixel_values = pre(Image.open(IMAGE).convert("RGB"))
    except Exception:
        pixel_values = torch.randn(1, 3, 560, 560)
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.float()

    with torch.no_grad():
        golden = ref(pixel_values, collect_intermediates=False)

    device = ttnn.open_device(device_id=args.device_id)
    try:
        model = TtRfDetr(ref, device)

        # ---- accuracy (detection-level, order-invariant vs fp32 reference) ----
        out = model(pixel_values)
        pcc_logits = _pcc_value(golden.logits, out.logits)
        pcc_boxes = _pcc_value(golden.pred_boxes, out.pred_boxes)
        accuracy = detection_accuracy(golden.logits, golden.pred_boxes, out.logits, out.pred_boxes)

        # ---- speed (warm, device-synchronized, median) ----
        for _ in range(3):
            model(pixel_values)
            ttnn.synchronize_device(device)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(pixel_values)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        fps = 1.0 / med if med > 0 else 0.0

        peak_dram = _peak_dram_mib(device)

        print(f"pcc_logits: {pcc_logits:.6f}")
        print(f"pcc_pred_boxes: {pcc_boxes:.6f}")
        print(f"median_latency_ms: {med * 1000:.3f}")
        print(f"inference_speed: {fps:.4f}")
        print(f"accuracy: {accuracy:.4f}")
        print(f"peak_dram: {peak_dram:.2f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
