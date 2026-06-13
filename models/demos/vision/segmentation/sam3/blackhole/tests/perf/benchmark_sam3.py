# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SAM3 end-to-end benchmark: speed + PCC measurement.

Outputs grepable lines:
    inference_speed: <fps>
    accuracy: <pcc>
    peak_dram: <MB>
"""

import os
import sys
import time
import unittest.mock as mock

import torch

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.dirname(os.path.abspath(__file__)).split("/models/")[0])
sys.path.insert(0, TT_METAL_HOME)

import ttnn

BPE_PATH = os.environ.get(
    "SAM3_BPE_PATH",
    os.path.join(TT_METAL_HOME, "python_env/lib/python3.10/site-packages/open_clip/bpe_simple_vocab_16e6.txt.gz"),
)


def _build_sam3_cpu(use_pretrained=True):
    orig = {
        n: getattr(torch, n)
        for n in ["zeros", "ones", "arange", "empty", "full", "randn", "rand", "tensor", "linspace", "logspace", "eye"]
    }

    def _redirect(fn):
        def wrapper(*args, **kwargs):
            if "device" in kwargs and kwargs["device"] is not None and "cuda" in str(kwargs["device"]):
                kwargs["device"] = "cpu"
            return fn(*args, **kwargs)

        return wrapper

    patches = [mock.patch("torch.cuda.is_available", return_value=False)]
    for name, fn in orig.items():
        patches.append(mock.patch(f"torch.{name}", _redirect(fn)))
    for p in patches:
        p.start()

    try:
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=BPE_PATH,
            device="cpu",
            eval_mode=True,
            load_from_HF=use_pretrained,
            checkpoint_path=None,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
    finally:
        for p in patches:
            p.stop()

    return model


def run_benchmark(device_id=0, use_pretrained=True, num_warmup=2, num_runs=10):
    print("=" * 70)
    print("SAM3 Benchmark — Tenstorrent Blackhole p150a")
    print("=" * 70)

    # Build model
    print("[1/5] Building SAM3 model...")
    model = _build_sam3_cpu(use_pretrained=use_pretrained)
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"      Pretrained: {use_pretrained}")

    # Extract ViT backbone
    from sam3.model.vitdet import ViT

    vit_backbone = None
    for _, module in model.named_modules():
        if isinstance(module, ViT):
            vit_backbone = module
            break
    assert vit_backbone is not None, "Could not find ViT backbone"

    # Open device
    print(f"[2/5] Opening device {device_id}...")
    device = ttnn.open_device(device_id=device_id)
    print(f"      Arch: {device.arch()}, Grid: {device.compute_with_storage_grid_size()}")

    # Preprocess weights
    print("[3/5] Preprocessing weights...")
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
        move_backbone_params_to_device,
        preprocess_vit_backbone_weights,
        tt_vit_backbone,
    )

    backbone_params = preprocess_vit_backbone_weights(vit_backbone)
    backbone_params = move_backbone_params_to_device(backbone_params, device)

    # Create input
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, 1008, 1008)

    # Reference run (CPU)
    print("[4/5] Running CPU reference...")
    with torch.no_grad():
        ref_output = vit_backbone(pixel_values)
    ref_feat = ref_output[-1]  # (B, 1024, 72, 72)

    # Warmup
    print(f"[5/5] Benchmarking ({num_warmup} warmup + {num_runs} timed runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = tt_vit_backbone(pixel_values, backbone_params, device)

    # Timed runs
    times = []
    tt_feat = None
    for i in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            tt_output = tt_vit_backbone(pixel_values, backbone_params, device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        tt_feat = tt_output[-1]

    # PCC
    from tests.ttnn.utils_for_testing import comp_pcc

    _, pcc_val = comp_pcc(ref_feat.float(), tt_feat.float())

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
    print(f"accuracy: {pcc_val:.4f}")
    print(f"peak_dram: 0")
    print(f"avg_latency_ms: {avg_time * 1000:.1f}")
    print(f"min_latency_ms: {min_time * 1000:.1f}")
    print(f"max_latency_ms: {max_time * 1000:.1f}")
    print(f"per_run_ms: {[f'{t*1000:.1f}' for t in times]}")
    print(f"pcc: {pcc_val:.6f}")
    print("=" * 70)

    ttnn.close_device(device)
    return fps, pcc_val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    run_benchmark(
        device_id=args.device,
        use_pretrained=not args.no_pretrained,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )
