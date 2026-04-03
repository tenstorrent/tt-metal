#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Conv3d device profiling for Wan 2.2 VAE decoder bottleneck layers.

Runs the top time-consuming conv3d shapes on a 1x1 mesh with 12x10 grid,
using the optimized blocking table from conv3d-blocking-sweep-v3.

Timing modes
------------
Default (wall-clock, batched):
    Dispatches all timed iterations in one batch, syncs once at the end.
    Amortises the single-sync PCIe roundtrip across N runs.
    Reported time ≈ pure device execution time (matches Tracy device kernel duration).

    python models/tt_dit/tests/models/wan2_2/profile_conv3d.py

Device profiler (exact device cycles):
    TT_METAL_DEVICE_PROFILER=1 python models/tt_dit/tests/models/wan2_2/profile_conv3d.py
    Reads generated/profiler/.logs/profile_log_device.csv after each layer
    and reports the per-dispatch device cycle time directly.

Tracy capture (host + device trace):
    ./build/tools/profiler/bin/capture-release -o conv3d_profile.tracy
    python models/tt_dit/tests/models/wan2_2/profile_conv3d.py

Specific layer:
    python models/tt_dit/tests/models/wan2_2/profile_conv3d.py --layer up2_res
"""

import argparse
import math
import os
import time
from collections import namedtuple

import torch

import ttnn
from models.tt_dit.utils.conv3d import aligned_channels, get_conv3d_config

Blocking = namedtuple("Blocking", ["cin", "cout", "t", "h", "w"])

# bh_4x32 720p uncached shapes — the production bottleneck config.
# These are per-device shapes after H/W fracturing across the 4x32 mesh.
# Shapes include external padding (+2 for 3x3x3, +2 for 1x3x3).
NUM_FRAMES = 81
LATENT_T = (NUM_FRAMES - 1) // 4 + 1  # 21
H_FACTOR, W_FACTOR = 4, 32

H_OUT, W_OUT = 720, 1280
VAE_SCALE = 8
H0 = math.ceil((H_OUT // VAE_SCALE) / H_FACTOR) * VAE_SCALE  # 184
W0 = ((W_OUT // VAE_SCALE) // W_FACTOR) * VAE_SCALE  # 32

# Temporal dims
T_LAT = LATENT_T + 2  # 23
T_TC0 = (LATENT_T - 1) + 2  # 22
T_MID = 2 * (LATENT_T - 1) + 1 + 2  # 43
T_TC1 = 2 * (LATENT_T - 1) + 1 - 1 + 2  # 42
T_HI = 2 * (2 * (LATENT_T - 1) + 1 - 1) + 1 + 2  # 83
T_133_MID = 2 * (LATENT_T - 1) + 1  # 41
T_133_HI = 2 * (2 * (LATENT_T - 1) + 1 - 1) + 1  # 81

# Spatial dims per stage
H_LAT, W_LAT = H0 // 8, W0 // 8  # 23, 4
H_MID, W_MID = H0 // 4, W0 // 4  # 46, 8 (note: 184/4=46)
H_HI, W_HI = H0 // 2, W0 // 2  # 92, 16
H_FULL, W_FULL = H0, W0  # 184, 32

# Layer configs: (name, T, H, W, C_in, kernel, C_out, repeat_count_in_decoder)
# Sorted by total wall time in decoder (repeat_count * per-call time).
LAYERS = {
    # 70% of decoder time in these two — already at optimal blocking
    "up3_res": ("96x96 k333", T_HI, H_FULL + 2, W_FULL + 2, 96, (3, 3, 3), 96, 6),
    "up2_res": ("192x192 k333", T_HI, H_HI + 2, W_HI + 2, 192, (3, 3, 3), 192, 6),
    # Next biggest
    "up1_res": ("384x384 k333 mid", T_MID, H_MID + 2, W_MID + 2, 384, (3, 3, 3), 384, 5),
    "conv_out": ("96x3 k333", T_HI, H_FULL + 2, W_FULL + 2, 96, (3, 3, 3), 3, 1),
    "up2_spatial": ("192x96 k133", T_133_HI, H_FULL + 2, W_FULL + 2, 192, (1, 3, 3), 96, 1),
    "up1_spatial": ("384x192 k133 hi", T_133_HI, H_HI + 2, W_HI + 2, 384, (1, 3, 3), 192, 1),
    # Latent + mid
    "up1_res0": ("192x384 k333", T_MID, H_MID + 2, W_MID + 2, 192, (3, 3, 3), 384, 1),
    "lat_res": ("384x384 k333 lat", T_LAT, H_LAT + 2, W_LAT + 2, 384, (3, 3, 3), 384, 10),
    # Time convs (huge speedup from blocking, but small absolute time now)
    "up1_tconv": ("384x768 k311 up", T_TC1, H_MID, W_MID, 384, (3, 1, 1), 768, 1),
    "up0_tconv": ("384x768 k311", T_TC0, H_LAT, W_LAT, 384, (3, 1, 1), 768, 1),
    "up0_spatial": ("384x192 k133", T_133_MID, H_MID + 2, W_MID + 2, 384, (1, 3, 3), 192, 1),
    "conv_in": ("32x384 k333", T_LAT, H_LAT + 2, W_LAT + 2, 32, (3, 3, 3), 384, 1),
}

WARMUP = 2
TIMED = 4


def make_tensors(mesh_device, T, H, W, C_in, C_out, kernel_size, C_in_block):
    padded_C_in = aligned_channels(C_in)
    tt_input = ttnn.from_torch(
        torch.randn(1, T, H, W, padded_C_in, dtype=torch.float32),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Weights in PyTorch format (C_out, C_in, kD, kH, kW), padded and prepared via
    # prepare_conv3d_weights for correct multi-C_in_block layout.
    w = torch.randn(C_out, padded_C_in, *kernel_size, dtype=torch.float32)
    tt_weight = ttnn.from_torch(
        w, device=mesh_device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    tt_weight = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=tt_weight, C_in_block=C_in_block, device=mesh_device
    )
    tt_bias = ttnn.from_torch(
        torch.randn(1, C_out, dtype=torch.float32),
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )
    return tt_input, tt_weight, tt_bias


def run_conv3d(mesh_device, tt_input, tt_weight, tt_bias, conv_config, C_out, kernel_size, ckc, sync=True):
    out = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        config=conv_config,
        output_channels=C_out,
        kernel_size=kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        padding_mode="zeros",
        dtype=ttnn.bfloat16,
        compute_kernel_config=ckc,
    )
    if sync:
        ttnn.synchronize_device(mesh_device)
    return out


def _read_device_kernel_us(freq_mhz=1350):
    """Parse profile_log_device.csv and return list of per-dispatch device durations (us)."""
    csv_path = "generated/profiler/.logs/profile_log_device.csv"
    if not os.path.exists(csv_path):
        return []
    starts, ends = {}, {}
    with open(csv_path) as f:
        next(f)  # arch line
        next(f)  # header
        for line in f:
            p = line.strip().split(",")
            if len(p) < 12:
                continue
            zone, ztype = p[10].strip(), p[11].strip()
            if "KERNEL" not in zone:
                continue
            try:
                cycles = int(p[5].strip())
                run_id = int(p[7].strip())
            except ValueError:
                continue
            starts[run_id] = min(starts.get(run_id, cycles), cycles)
            ends[run_id] = max(ends.get(run_id, cycles), cycles)
    return sorted(
        [(rid, (ends[rid] - starts[rid]) / freq_mhz) for rid in starts if rid in ends],
        key=lambda x: x[0],
    )


def profile_layer(mesh_device, grid_size, ckc, name, layer_info, warmup=WARMUP, timed=TIMED):
    desc, T, H, W, C_in, kernel_size, C_out, repeat = layer_info
    padded_C_in = aligned_channels(C_in)

    kT, kH, kW = kernel_size
    H_out = H - (kH - 1)
    W_out = W - (kW - 1)
    conv_config = get_conv3d_config(
        C_in,
        C_out,
        kernel_size,
        ttnn.bfloat16,
        grid_size,
        h_factor=H_FACTOR,
        w_factor=W_FACTOR,
        H_out=H_out,
        W_out=W_out,
    )

    print(f"\n{'='*80}")
    print(f"  {name}: {desc}")
    print(f"  Shape: T={T} H={H} W={W} C_in={C_in}(pad={padded_C_in}) C_out={C_out} kernel={kernel_size}")
    print(
        f"  Blocking: Cin={conv_config.C_in_block} Cout={conv_config.C_out_block} "
        f"T={conv_config.T_out_block} H={conv_config.H_out_block} W={conv_config.W_out_block}"
    )
    print(f"  Decoder repeats: {repeat}x")
    print(f"{'='*80}", flush=True)

    torch.manual_seed(42)
    tt_input, tt_weight, tt_bias = make_tensors(mesh_device, T, H, W, C_in, C_out, kernel_size, conv_config.C_in_block)

    # Warmup (sync after each to ensure clean state)
    for _ in range(warmup):
        out = run_conv3d(mesh_device, tt_input, tt_weight, tt_bias, conv_config, C_out, kernel_size, ckc, sync=True)
        ttnn.deallocate(out)

    use_device_profiler = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
    dispatches_before = len(_read_device_kernel_us()) if use_device_profiler else 0

    # Timed: dispatch all N runs in one batch, single sync at end.
    # This amortises the sync PCIe roundtrip and matches Tracy device kernel duration.
    t0 = time.perf_counter()
    outs = []
    for _ in range(timed):
        outs.append(
            run_conv3d(mesh_device, tt_input, tt_weight, tt_bias, conv_config, C_out, kernel_size, ckc, sync=False)
        )
    ttnn.synchronize_device(mesh_device)
    batch_us = (time.perf_counter() - t0) * 1e6
    for out in outs:
        ttnn.deallocate(out)

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    avg = batch_us / timed

    # Device profiler: read exact on-device dispatch times if available
    if use_device_profiler:
        dispatches_after = _read_device_kernel_us()
        new_dispatches = dispatches_after[dispatches_before:]
        # Last `timed` dispatches are the timed runs
        timed_dispatches = [d for _, d in new_dispatches[-timed:]]
        if timed_dispatches:
            avg_device = sum(timed_dispatches) / len(timed_dispatches)
            print(f"  Per-call (device profiler): {avg_device:.0f} us")
            avg = avg_device

    total_in_decoder = avg * repeat
    print(f"  Per-call (batched wall): {batch_us/timed:.0f} us  (batch={batch_us:.0f} us / {timed} runs)")
    print(f"  Decoder total ({repeat}x): {total_in_decoder:.0f} us")
    return avg, total_in_decoder


def main():
    parser = argparse.ArgumentParser(description="Profile conv3d bottleneck layers with Tracy")
    parser.add_argument("--layer", type=str, default=None, help="Profile specific layer only (e.g. up2_res)")
    parser.add_argument("--no-tracy", action="store_true", help="(No-op, kept for compatibility)")
    parser.add_argument("--timed", type=int, default=TIMED, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Number of warmup iterations")
    args = parser.parse_args()

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    grid_size = ttnn.CoreCoord(12, 10)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    layers_to_run = LAYERS
    if args.layer:
        if args.layer not in LAYERS:
            print(f"Unknown layer '{args.layer}'. Available: {', '.join(LAYERS.keys())}")
            return
        layers_to_run = {args.layer: LAYERS[args.layer]}

    results = {}
    for name, info in layers_to_run.items():
        avg, total = profile_layer(mesh_device, grid_size, ckc, name, info, warmup=args.warmup, timed=args.timed)
        results[name] = (avg, total, info[7])  # avg, total, repeat

    # Summary
    mode = "device profiler" if os.environ.get("TT_METAL_DEVICE_PROFILER") else "batched wall-clock"
    print(f"\n{'='*80}")
    print(f"  SUMMARY — bh_4x32 720p uncached (1x1 mesh, 12x10 grid)  [{mode}]")
    print(f"{'='*80}")
    print(f"  {'Layer':<15} {'Per-call (us)':>14} {'Repeats':>8} {'Total (us)':>12} {'% of total':>10}")
    print(f"  {'-'*15} {'-'*14} {'-'*8} {'-'*12} {'-'*10}")
    grand_total = sum(t for _, t, _ in results.values())
    for name, (avg, total, repeat) in results.items():
        pct = total / grand_total * 100 if grand_total else 0
        print(f"  {name:<15} {avg:>14.0f} {repeat:>8d} {total:>12.0f} {pct:>9.1f}%")
    print(f"  {'-'*15} {'-'*14} {'-'*8} {'-'*12} {'-'*10}")
    print(f"  {'TOTAL':<15} {'':>14} {'':>8} {grand_total:>12.0f}")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
