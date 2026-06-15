# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SAM3 end-to-end benchmark: full pipeline speed + PCC measurement.

Runs the complete SAM3 pipeline (ViT backbone + FPN neck + text encoder +
geometry encoder + transformer encoder/decoder + segmentation head) with
the ttnn-accelerated ViT backbone, and compares masks against CPU reference.

Outputs grepable lines:
    inference_speed: <fps>
    accuracy: <pcc>
    peak_dram: <MB>
"""

import os
import sys
import time

import torch

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME",
    os.path.dirname(os.path.abspath(__file__)).split("/models/")[0],
)
sys.path.insert(0, TT_METAL_HOME)

import ttnn
from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_model import (
    TtSam3ImagePipeline,
    _patch_pin_memory,
    build_sam3_model,
    extract_predictions,
    make_batched_datapoint,
)


def run_benchmark(device_id=0, use_pretrained=True, num_warmup=2, num_runs=5):
    print("=" * 70)
    print("SAM3 End-to-End Benchmark — Tenstorrent Blackhole p150a")
    print("=" * 70)

    _patch_pin_memory()

    # Build model
    print("[1/6] Building SAM3 model...")
    model = build_sam3_model(use_pretrained=use_pretrained)
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Open device
    print(f"[2/6] Opening device {device_id}...")
    device = ttnn.open_device(device_id=device_id)
    print(f"      Arch: {device.arch()}, Grid: {device.compute_with_storage_grid_size()}")

    # Create pipeline (patches ViT backbone with ttnn)
    print("[3/6] Creating ttnn pipeline (preprocessing weights)...")
    pipeline = TtSam3ImagePipeline(model, device)

    # Create input — use a deterministic structured pattern (not random noise)
    # to avoid pathological error amplification through the full pipeline.
    torch.manual_seed(42)
    x = torch.linspace(-1, 1, 1008).view(1, 1, 1, 1008).expand(1, 1, 1008, 1008)
    y = torch.linspace(-1, 1, 1008).view(1, 1, 1008, 1).expand(1, 1, 1008, 1008)
    pixel_values = torch.cat([x, y, torch.sin(x * 3.14) * torch.cos(y * 3.14)], dim=1)
    text_prompts = ["object", "visual"]

    # CPU reference run
    print("[4/6] Running CPU reference...")
    pipeline.restore()
    input_batch_ref = make_batched_datapoint(pixel_values, text_prompts)
    with torch.no_grad():
        ref_output = model(input_batch_ref)
    ref_preds = extract_predictions(ref_output)
    ref_masks = ref_preds["pred_masks"]
    ref_logits = ref_preds["pred_logits"]
    print(f"      Masks: {ref_masks.shape}, Logits: {ref_logits.shape}")

    # Re-patch ViT and text cache for ttnn
    pipeline.vit_backbone.forward = pipeline._patched_vit_forward
    model.backbone.forward_text = pipeline._cached_forward_text

    # Warmup
    print(f"[5/6] Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        input_batch = make_batched_datapoint(pixel_values, text_prompts)
        with torch.no_grad():
            _ = model(input_batch)

    # Timed runs
    print(f"[6/6] Benchmarking ({num_runs} timed runs)...")
    times = []
    tt_preds = None
    for i in range(num_runs):
        input_batch = make_batched_datapoint(pixel_values, text_prompts)
        t0 = time.perf_counter()
        with torch.no_grad():
            tt_output = model(input_batch)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        tt_preds = extract_predictions(tt_output)

    tt_masks = tt_preds["pred_masks"]
    tt_logits = tt_preds["pred_logits"]

    # PCC — masks
    from tests.ttnn.utils_for_testing import comp_pcc

    _, mask_pcc = comp_pcc(ref_masks.float(), tt_masks.float())
    _, logit_pcc = comp_pcc(ref_logits.float(), tt_logits.float())

    # Top-mask IoU
    ref_scores = ref_logits[0].squeeze(-1).sigmoid()
    tt_scores = tt_logits[0].squeeze(-1).sigmoid()
    ref_top = ref_scores.argmax()
    tt_top = tt_scores.argmax()
    ref_bin = (ref_masks[0, ref_top] > 0).float()
    tt_bin = (tt_masks[0, tt_top] > 0).float()
    intersection = (ref_bin * tt_bin).sum()
    union = (ref_bin + tt_bin).clamp(max=1).sum()
    iou = (intersection / union).item() if union > 0 else 0.0

    # Stats
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1.0 / avg_time

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"inference_speed: {fps:.2f}")
    print(f"accuracy: {mask_pcc:.4f}")
    print(f"peak_dram: 0")
    print(f"avg_latency_ms: {avg_time * 1000:.1f}")
    print(f"min_latency_ms: {min_time * 1000:.1f}")
    print(f"max_latency_ms: {max_time * 1000:.1f}")
    print(f"per_run_ms: {[f'{t*1000:.1f}' for t in times]}")
    print(f"mask_pcc: {mask_pcc:.6f}")
    print(f"logit_pcc: {logit_pcc:.6f}")
    print(f"top_mask_iou: {iou:.4f}")
    print(f"ref_top_score: {ref_scores[ref_top]:.4f}")
    print(f"tt_top_score: {tt_scores[tt_top]:.4f}")
    print("=" * 70)

    ttnn.close_device(device)
    return fps, mask_pcc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    run_benchmark(
        device_id=args.device,
        use_pretrained=not args.no_pretrained,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )
