#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Forward-pass performance harness for the DiffusionDrive TTNN model.

Measures absolute forward latency and derived FPS for the full on-device stack at
production resolution (camera 256x1024, LiDAR 256x256), for both the eager
``__call__`` path and the trace-replay ``execute_compiled`` path, and probes the
conv core-grid utilisation the auto-sharded ``ttnn.conv2d`` selects at each
ResNet-34 stage resolution.

This is the committed source for the numbers in ``PERFORMANCE.md`` (the bounty's
"performance report" deliverable). Run it on your own hardware to reproduce.

Usage::

    source python_env/bin/activate
    export PYTHONPATH="${TT_METAL_HOME:-$PWD}"
    python models/demos/diffusion_drive/scripts/profile_forward.py \
        --checkpoint "$DD_CHECKPOINT_PATH" --anchors "$DD_ANCHOR_PATH" --iters 20

``--checkpoint`` / ``--anchors`` default to ``$DD_CHECKPOINT_PATH`` /
``$DD_ANCHOR_PATH`` (falling back to the ``$DD_DATA_ROOT`` reference layout). If no
checkpoint is found the harness falls back to random weights with ``latent=True``
(op timings are representative; accuracy is not). Latency is hardware- and
build-dependent, so numbers are not pinned in-repo — this script prints them.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import Optional

import torch

import ttnn


def _resolve(path: Optional[str], env: str, *fallbacks: str) -> Optional[str]:
    for cand in (path, os.environ.get(env), *fallbacks):
        if cand and Path(cand).exists():
            return cand
    return None


def _make_features() -> dict:
    return {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.randn(1, 1, 256, 256),
        "status_feature": torch.randn(1, 8),
    }


def _time_calls(fn, features, iters: int, seed: int = 1234) -> list:
    """Return per-call wall time in ms. Output conversion to torch forces a
    device->host sync, so each timed call includes the full forward."""
    times = []
    for _ in range(iters):
        torch.manual_seed(seed)  # pin DDIM noise so every call is equal work (DD-5)
        t0 = time.perf_counter()
        fn(features)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _report(label: str, times: list) -> None:
    lo, med, avg = min(times), statistics.median(times), sum(times) / len(times)
    print(
        f"  {label:<26} min={lo:8.2f} ms  median={med:8.2f} ms  avg={avg:8.2f} ms"
        f"   |  {1000.0 / med:6.1f} FPS (median)"
    )


def _probe_conv_core_grids(device) -> None:
    """Report how many cores the auto-sharded ttnn.conv2d picks at each ResNet-34
    stage resolution (a proxy for core utilisation — the Stage 3 'max core counts'
    question). Defensive: any failure is reported, not fatal."""
    from models.demos.diffusion_drive.tt.ttnn_resnet34 import _make_conv_config

    # (name, B, H, W, C_in, C_out, k, stride, pad) — one representative conv per stage.
    cases = [
        ("layer1 3x3  64ch  64x256", 1, 64, 256, 64, 64, 3, 1, 1),
        ("layer2 3x3 128ch  32x128", 1, 32, 128, 128, 128, 3, 1, 1),
        ("layer3 3x3 256ch  16x64 ", 1, 16, 64, 256, 256, 3, 1, 1),
        ("layer4 3x3 512ch   8x32 ", 1, 8, 32, 512, 512, 3, 1, 1),
    ]
    print("\n[conv core-grid utilisation]  (auto shard_layout=None; output shard grid)")
    for name, B, H, W, Cin, Cout, k, s, p in cases:
        try:
            x = torch.randn(1, 1, B * H * W, Cin, dtype=torch.bfloat16)
            w = torch.randn(Cout, Cin, k, k, dtype=torch.bfloat16)
            b = torch.randn(1, 1, 1, Cout, dtype=torch.bfloat16)
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            wt = ttnn.from_torch(w, dtype=ttnn.bfloat16)
            bt = ttnn.from_torch(b, dtype=ttnn.bfloat16)
            [out, _, _] = ttnn.conv2d(
                input_tensor=xt,
                weight_tensor=wt,
                bias_tensor=bt,
                in_channels=Cin,
                out_channels=Cout,
                device=device,
                kernel_size=[k, k],
                stride=[s, s],
                padding=[p, p, p, p],
                dilation=[1, 1],
                batch_size=B,
                input_height=H,
                input_width=W,
                conv_config=_make_conv_config(),
                return_weights_and_bias=True,
                return_output_dim=True,
            )
            if out.is_sharded():
                grid = out.memory_config().shard_spec.grid
                ncores = grid.num_cores()
                print(f"  {name}:  sharded, {ncores:2d} cores  ({out.memory_config().memory_layout})")
            else:
                print(f"  {name}:  interleaved output (not sharded)")
            ttnn.deallocate(out)
        except Exception as exc:  # noqa: BLE001 - probe must not abort the harness
            print(f"  {name}:  probe failed: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DiffusionDrive TTNN forward-pass profiler")
    ap.add_argument("--checkpoint", default=None, help="default: $DD_CHECKPOINT_PATH / $DD_DATA_ROOT layout")
    ap.add_argument("--anchors", default=None, help="default: $DD_ANCHOR_PATH / $DD_DATA_ROOT layout")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-trace", action="store_true", help="skip the traced execute_compiled measurement")
    args = ap.parse_args()

    data_root = os.environ.get("DD_DATA_ROOT", "/mnt/diffusion-drive")
    ckpt = _resolve(args.checkpoint, "DD_CHECKPOINT_PATH", f"{data_root}/weights/diffusiondrive_navsim_88p1_PDMS.pth")
    anchors = _resolve(args.anchors, "DD_ANCHOR_PATH", f"{data_root}/resnet34/kmeans_navsim_traj_20.npy")

    from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, DiffusionDriveModel, load_model
    from models.demos.diffusion_drive.tt.config import ModelConfig
    from models.demos.diffusion_drive.tt.ttnn_diffusion_drive import TtnnDiffusionDriveModel

    latent = ckpt is None
    print(f"[profile] torch {torch.__version__}  checkpoint={'<random,latent>' if latent else ckpt}")
    print(f"[profile] anchors={anchors}")

    # Reference model (source of TTNN weights): real checkpoint when present, else random.
    ref_cfg = DiffusionDriveConfig(plan_anchor_path=anchors, latent=latent)
    if latent:
        ref_model = DiffusionDriveModel(ref_cfg).eval()
    else:
        ref_model = load_model(ckpt, ref_cfg, device=torch.device("cpu")).eval()

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=256 * 1024 * 1024)
    try:
        model_config = ModelConfig(plan_anchor_path=anchors)
        ttnn_model = TtnnDiffusionDriveModel(ref_model, model_config, device)
        (
            ttnn_model.build_stage2(device)
            .build_stage3(device)
            .build_stage3_4(device)
            .build_stage3_5(device)
            .build_stage3_6(device)
            .build_stage3_7(device)
            .build_stage4(device)
        )

        features = _make_features()

        print(f"\n[warmup] {args.warmup} eager passes (JIT compile at full res — first pass is minutes)...")
        _time_calls(ttnn_model, features, args.warmup)

        print(f"\n[latency]  batch=1, production resolution, {args.iters} iters")
        _report("eager __call__", _time_calls(ttnn_model, features, args.iters))

        if not args.no_trace:
            print("\n[trace] capturing backbone-loop trace (compile)...")
            ttnn_model.compile(features)
            _time_calls(ttnn_model.execute_compiled, features, args.warmup)  # warm the traced path
            _report("traced execute_compiled", _time_calls(ttnn_model.execute_compiled, features, args.iters))
            ttnn_model.release_compiled()

        _probe_conv_core_grids(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
