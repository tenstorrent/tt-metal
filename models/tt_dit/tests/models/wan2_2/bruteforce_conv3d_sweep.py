#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
In-process brute-force conv3d blocking sweep with T_out_block support.

Sweeps (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block) combinations
with accurate L1 pre-filtering based on the actual CB sizing from conv3d_program_factory.

Usage:
    source python_env/bin/activate && export PYTHONPATH=$(pwd)
    python models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
        --C_in 96 --C_out 96 --kernel 3,3,3 --T 83 --H 186 --W 42 \
        --mesh 2x4 --output sweep_results_v3/up3_96x96.json
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
TILE_SIZE = 2048  # bf16 tile = 32*32*2
FP32_TILE_SIZE = 4096  # fp32 tile = 32*32*4
TILE_HEIGHT = 32


def valid_cin(c, padded_C_in, kernel_size):
    kT, kH, kW = kernel_size
    kernel_vol = kT * kH * kW
    return c >= 32 and c <= padded_C_in and padded_C_in % c == 0 and (kernel_vol * c) % 32 == 0


def valid_cout(c, C_out):
    padded = aligned_channels(C_out)
    return c >= 32 and c <= padded and padded % c == 0 and c % 32 == 0


def valid_matmul_subblock(C_out_block, fp32_dest_acc_en=True):
    dst_size = 4 if fp32_dest_acc_en else 8
    matmul_N_t = math.ceil(C_out_block / 32)
    out_subblock_w = min(matmul_N_t, dst_size)
    return matmul_N_t % out_subblock_w == 0


def estimate_l1_bytes(cin_block, cout_block, t_blk, h_blk, w_blk, kernel_size, C_in, dtype_bytes=2):
    """Accurate L1 estimate matching conv3d_program_factory CB sizing."""
    kT, kH, kW = kernel_size
    C_in_num_blocks = aligned_channels(C_in) // cin_block

    num_patches = t_blk * h_blk * w_blk
    patch_size = kT * kH * kW * cin_block
    padded_patch_size = math.ceil(patch_size / 32) * 32
    padded_patch_bytes = padded_patch_size * dtype_bytes

    matmul_M_t = math.ceil(num_patches / TILE_HEIGHT)
    matmul_K_t = math.ceil(patch_size / 32)
    matmul_N_t = math.ceil(cout_block / 32)

    # cb_vol2col_rm: TILE_HEIGHT pages (or 2x if not aligned)
    vol2col_rm_pages = TILE_HEIGHT if num_patches % TILE_HEIGHT == 0 else min(num_patches, 2 * TILE_HEIGHT)
    cb_vol2col_rm = padded_patch_bytes * vol2col_rm_pages

    # cb_vol2col_tiled: K_t tiles (fused tilize+matmul, one tile-row)
    cb_vol2col_tiled = TILE_SIZE * matmul_K_t

    # cb_weight_tiled: K_t * N_t tiles
    cb_weight = TILE_SIZE * matmul_K_t * matmul_N_t

    # cb_matmul_interm: M_t * N_t tiles (fp32 if multi C_in block)
    use_fp32_partials = C_in_num_blocks > 1
    partial_tile = FP32_TILE_SIZE if use_fp32_partials else TILE_SIZE
    cb_interm = partial_tile * matmul_M_t * matmul_N_t

    # cb_matmul_result_rm: M_t * N_t tiles
    cb_result = TILE_SIZE * matmul_M_t * matmul_N_t

    # cb_reduction + cb_worker_ack (only if multi C_in blocks)
    cb_reduction = 0
    if C_in_num_blocks > 1:
        cb_reduction = partial_tile * matmul_M_t * matmul_N_t + TILE_SIZE  # reduction + ack

    # cb_bias: N_t tiles
    cb_bias = TILE_SIZE * matmul_N_t

    # cb_zero (fp32 reduction)
    cb_zero = TILE_SIZE if use_fp32_partials else 0

    # L1 prefetch shard: T_shard * H_shard * W_shard * C_in_block_bytes
    T_shard = (t_blk - 1) + kT
    H_shard = (h_blk - 1) + kH
    W_shard = (w_blk - 1) + kW
    cin_block_bytes = cin_block * dtype_bytes
    cb_shard = T_shard * H_shard * W_shard * cin_block_bytes

    total = (
        cb_vol2col_rm
        + cb_vol2col_tiled
        + cb_weight
        + cb_interm
        + cb_result
        + cb_reduction
        + cb_bias
        + cb_zero
        + cb_shard
    )
    return total


# BH p150b L1 size = 1,572,864 bytes. Reserve 200KB for kernel code/stack.
L1_BUDGET = 1_572_864 - 200 * 1024  # ~1,372 KB


def build_all_blockings(C_in, C_out, kernel_size, H, W, T):
    padded_cin = aligned_channels(C_in)
    kT = kernel_size[0]

    cins = [c for c in range(32, padded_cin + 1, 32) if valid_cin(c, padded_cin, kernel_size)]
    couts = [c for c in range(32, aligned_channels(C_out) + 1, 32) if valid_cout(c, C_out)]

    # T_out for this shape (no padding in sweep — raw conv)
    T_out = T - (kT - 1) if kT > 1 else T

    # T_block candidates: 1 + multiples of 3 (for tile-row alignment with H=8,W=4)
    t_blocks = [1]
    if kT > 1:
        for t in [3, 6, 9]:
            if t <= T_out:
                t_blocks.append(t)

    # Spatial candidates
    tensor_vol = H * W
    if tensor_vol > 2000:
        hw = [(h, w) for h in [2, 4, 8, 16, 32] for w in [2, 4, 8, 16, 32] if h * w <= 256 and h <= H and w <= W]
    else:
        hw = [(h, w) for h in [1, 2, 4, 8, 16, 32] for w in [1, 2, 4, 8, 16, 32] if h * w <= 256 and h <= H and w <= W]

    # Filter small C_out_block for large tensors
    padded_cout = aligned_channels(C_out)
    if tensor_vol > 2000 and padded_cout > 64:
        couts = [c for c in couts if c >= 64]
    if tensor_vol > 10000 and padded_cout > 96:
        couts = [c for c in couts if c >= 96]

    combos = []
    skipped_l1 = 0
    for cin in cins:
        for cout in couts:
            if not valid_matmul_subblock(cout):
                continue
            for t_blk in t_blocks:
                for h, w in hw:
                    est = estimate_l1_bytes(cin, cout, t_blk, h, w, kernel_size, C_in)
                    if est > L1_BUDGET:
                        skipped_l1 += 1
                        continue
                    combos.append((cin, cout, t_blk, h, w))
    if skipped_l1:
        print(f"Skipped {skipped_l1} combos for estimated L1 OOM")

    # Sort: put likely-fast combos first (larger C_in_block, moderate spatial)
    # This ensures the probe cutoff has a good baseline early
    def combo_priority(c):
        cin, cout, t, h, w = c
        # Prefer: high cin (fewer C_in blocks), h*w near 32, T=3
        cin_score = -cin  # larger cin = better (fewer blocks)
        spatial_score = abs(h * w - 32)  # prefer ~32 patches
        t_score = 0 if t == 3 else (1 if t == 1 else 2)
        return (cin_score, t_score, spatial_score)

    combos.sort(key=combo_priority)
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
    parser.add_argument("--mesh", type=str, default=None, help="Mesh shape e.g. 2x4")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    kernel_size = tuple(int(x) for x in args.kernel.split(","))
    C_in, C_out, T, H, W = args.C_in, args.C_out, args.T, args.H, args.W
    padded_cin = aligned_channels(C_in)

    combos = build_all_blockings(C_in, C_out, kernel_size, H, W, T)
    print(f"Shape: C_in={C_in} C_out={C_out} kernel={kernel_size} T={T} H={H} W={W}")
    print(f"Grid: {args.grid_size or 'auto'}")
    print(f"Mesh: {args.mesh or 'single device'}")
    print(f"Total valid combos: {len(combos)}")

    # Open device
    mesh_device = None
    if args.mesh:
        mr, mc = (int(x) for x in args.mesh.split("x"))
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT)
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mr, mc))
        device = mesh_device
    else:
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
    w = torch.randn(C_out, padded_cin, *kernel_size, dtype=torch.float32)
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

    # Pre-prepare weights per C_in_block (cache to avoid re-preparing)
    weight_cache = {}

    for i, (cin, cout, t_blk, h_blk, w_blk) in enumerate(combos):
        try:
            # Get or prepare weights for this C_in_block
            if cin not in weight_cache:
                tt_w = ttnn.from_torch(
                    w, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0
                )
                tt_w = ttnn.experimental.prepare_conv3d_weights(weight_tensor=tt_w, C_in_block=cin, device=device)
                weight_cache[cin] = tt_w
            tt_weight = weight_cache[cin]

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
            # Probe run: single call, skip if > 10× current best (avoids wasting time on slow combos)
            probe_t0 = time.perf_counter()
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
            probe_us = (time.perf_counter() - probe_t0) * 1e6
            ttnn.deallocate(out)
            if best_us < float("inf") and probe_us > best_us * 10:
                fail_count += 1
                results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": probe_us, "status": "slow"})
                continue

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
        "mesh": args.mesh,
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

    if mesh_device:
        ttnn.close_mesh_device(mesh_device)
    else:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
