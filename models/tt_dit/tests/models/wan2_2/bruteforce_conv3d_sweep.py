#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
In-process brute-force conv3d blocking sweep. No subprocess isolation needed
since invalid configs throw exceptions (L1 OOM, matmul divisibility) that are
caught gracefully.

Runs all (C_in_block, C_out_block, H_out_block, W_out_block) combinations
through the conv3d op on a single device, measures wall-clock time (2 warmup +
4 timed runs), and reports the top blockings.

Usage:
    source python_env/bin/activate && export PYTHONPATH=$(pwd)
    python models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
        --C_in 96 --C_out 96 --kernel 3,3,3 --T 83 --H 186 --W 42 \
        --grid-size 12x10 --output sweep_results_v2/up3_96x96.json
"""

import argparse
import json
import math
import statistics
import time

import torch

import ttnn
from models.tt_dit.utils.conv3d import aligned_channels

WARMUP = 2
RUNS = 4


def valid_cin(c, padded_C_in, kernel_size):
    kT, kH, kW = kernel_size
    kernel_vol = kT * kH * kW
    return c >= 32 and c <= padded_C_in and padded_C_in % c == 0 and (kernel_vol * c) % 32 == 0


def valid_cout(c, C_out):
    padded = aligned_channels(C_out)
    return c >= 32 and c <= padded and padded % c == 0 and c % 32 == 0


def valid_matmul_subblock(C_out_block, fp32_dest_acc_en=True):
    """Check matmul_N_t % out_subblock_w == 0 (conv3d requirement)."""
    dst_size = 4 if fp32_dest_acc_en else 8
    matmul_N_t = math.ceil(C_out_block / 32)
    out_subblock_w = min(matmul_N_t, dst_size)
    return matmul_N_t % out_subblock_w == 0


def build_all_blockings(C_in, C_out, kernel_size, H, W):
    padded_cin = aligned_channels(C_in)
    cins = [c for c in range(32, padded_cin + 1, 32) if valid_cin(c, padded_cin, kernel_size)]
    couts = [c for c in range(32, aligned_channels(C_out) + 1, 32) if valid_cout(c, C_out)]

    # Power-of-2 spatial candidates (non-pow2 may hang on some shapes)
    hw = [(h, w) for h in [1, 2, 4, 8, 16, 32] for w in [1, 2, 4, 8, 16, 32] if h * w <= 256 and h <= H and w <= W]

    combos = []
    for cin in cins:
        for cout in couts:
            if not valid_matmul_subblock(cout):
                continue
            for h, w in hw:
                combos.append((cin, cout, 1, h, w))
    return combos


def main():
    parser = argparse.ArgumentParser(description="In-process brute-force conv3d blocking sweep")
    parser.add_argument("--C_in", type=int, required=True)
    parser.add_argument("--C_out", type=int, required=True)
    parser.add_argument("--kernel", type=str, required=True, help="e.g. 3,3,3")
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--W", type=int, required=True)
    parser.add_argument("--grid-size", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    kernel_size = tuple(int(x) for x in args.kernel.split(","))
    C_in, C_out, T, H, W = args.C_in, args.C_out, args.T, args.H, args.W
    padded_cin = aligned_channels(C_in)

    combos = build_all_blockings(C_in, C_out, kernel_size, H, W)
    print(f"Shape: C_in={C_in} C_out={C_out} kernel={kernel_size} T={T} H={H} W={W}")
    print(f"Grid: {args.grid_size or 'auto'}")
    print(f"Total valid combos: {len(combos)}")

    # Open device
    device = ttnn.open_device(device_id=0)
    if args.grid_size:
        gx, gy = args.grid_size.split("x")
        grid_size = ttnn.CoreCoord(int(gx), int(gy))
    else:
        grid_size = device.compute_with_storage_grid_size()

    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # Create tensors once
    torch.manual_seed(42)
    tt_input = ttnn.from_torch(
        torch.randn(1, T, H, W, padded_cin, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = torch.randn(C_out, C_in, *kernel_size, dtype=torch.float32).permute(2, 3, 4, 1, 0)
    if padded_cin != C_in:
        w = torch.nn.functional.pad(w, (0, 0, 0, padded_cin - C_in))
    tt_weight = ttnn.from_torch(
        w.reshape(-1, C_out), device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    tt_bias = ttnn.from_torch(
        torch.randn(1, C_out, dtype=torch.float32),
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )

    # Sweep
    results = []
    best_us = float("inf")
    best_blk = None
    t_start = time.time()
    ok_count = 0
    fail_count = 0

    for i, (cin, cout, t_blk, h_blk, w_blk) in enumerate(combos):
        try:
            cfg = ttnn.Conv3dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                T_out_block=t_blk,
                W_out_block=w_blk,
                H_out_block=h_blk,
                C_out_block=cout,
                C_in_block=cin,
                compute_with_storage_grid_size=grid_size,
            )
            # Warmup
            for _ in range(WARMUP):
                out = ttnn.experimental.conv3d(
                    input_tensor=tt_input,
                    weight_tensor=tt_weight,
                    bias_tensor=tt_bias,
                    config=cfg,
                    output_channels=C_out,
                    kernel_size=kernel_size,
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                    padding_mode="zeros",
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ckc,
                )
                ttnn.synchronize_device(device)
                ttnn.deallocate(out)

            # Timed
            times = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                out = ttnn.experimental.conv3d(
                    input_tensor=tt_input,
                    weight_tensor=tt_weight,
                    bias_tensor=tt_bias,
                    config=cfg,
                    output_channels=C_out,
                    kernel_size=kernel_size,
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                    padding_mode="zeros",
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ckc,
                )
                ttnn.synchronize_device(device)
                times.append((time.perf_counter() - t0) * 1e6)
                ttnn.deallocate(out)

            us = statistics.mean(times)
            ok_count += 1
            tag = ""
            if us < best_us:
                best_us = us
                best_blk = (cin, cout, t_blk, h_blk, w_blk)
                tag = " ** BEST"

            results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": us, "status": "ok"})
            if (i + 1) % 10 == 0 or tag:
                print(
                    f"  [{i+1:3d}/{len(combos)}] ({cin:3d},{cout:3d},{t_blk},{h_blk:2d},{w_blk:2d}) "
                    f"{us:8.0f}us{tag}",
                    flush=True,
                )
        except Exception:
            fail_count += 1
            results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": None, "status": "fail"})

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min). {ok_count} ok, {fail_count} failed.")

    # Top 10
    ok_results = sorted([r for r in results if r["us"]], key=lambda r: r["us"])
    print(f"\nTop 10:")
    for r in ok_results[:10]:
        b = r["blocking"]
        print(f"  ({b[0]:3d},{b[1]:3d},{b[2]},{b[3]:2d},{b[4]:2d}) = {r['us']:.0f} us")

    # Save
    from pathlib import Path

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final = {
        "C_in": C_in,
        "C_out": C_out,
        "kernel": list(kernel_size),
        "T": T,
        "H": H,
        "W": W,
        "grid_size": args.grid_size,
        "num_combos": len(combos),
        "num_ok": ok_count,
        "num_fail": fail_count,
        "elapsed_s": elapsed,
        "best_blocking": list(best_blk) if best_blk else None,
        "best_us": best_us if best_us < float("inf") else None,
        "top_20": ok_results[:20],
        "all_results": results,
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"Saved to {out_path}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
