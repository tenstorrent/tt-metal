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
import pathlib
import statistics
import time

import torch

import ttnn
from models.tt_dit.utils.conv3d import _BLOCKINGS, _DEFAULT_BLOCKINGS, aligned_channels

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


def compute_parallelism(cin_block, cout_block, t_blk, h_blk, w_blk, kernel_size, H, W, T, C_in, C_out, num_cores=120):
    """Return the number of cores used by this blocking configuration."""
    padded_cin = aligned_channels(C_in)
    padded_cout = aligned_channels(C_out)
    kT = kernel_size[0]
    T_out = T - (kT - 1) if kT > 1 else T

    c_in_blocks = padded_cin // cin_block
    c_out_blocks = padded_cout // cout_block

    c_in_par = min(c_in_blocks, num_cores)
    cores_per_out = max(1, num_cores // c_in_par)
    c_out_par = min(c_out_blocks, cores_per_out)
    remaining = max(1, cores_per_out // c_out_par)

    T_out_blocks = max(1, math.ceil(T_out / t_blk))
    H_out_blocks = max(1, math.ceil(H / h_blk))
    W_out_blocks = max(1, math.ceil(W / w_blk))
    spatial_par = min(T_out_blocks * H_out_blocks * W_out_blocks, remaining)

    return c_in_par * c_out_par * spatial_par


def t_needed_to_hide_weight_read(cin_block, cout_block, kernel_size):
    """Estimate minimum T_out_block for tilize time to exceed weight read time.
    Weight read: K_t * N_t tiles * 2KB / 20 GB/s effective.
    Per M-row tilize ≈ 4 µs (measured). Need M_t * 4 ≥ weight_us.
    Returns the minimum T_out_block (with H=8, W=4 as reference spatial).
    """
    kT, kH, kW = kernel_size
    matmul_K_t = math.ceil(kT * kH * kW * cin_block / 32)
    matmul_N_t = math.ceil(cout_block / 32)
    weight_kb = matmul_K_t * matmul_N_t * 2  # KB (bf16 tiles)
    weight_us = weight_kb / 20  # ~20 GB/s effective with contention
    tilize_per_mrow_us = 4.0
    # M_t * 4µs ≥ weight_us → M_t ≥ weight_us/4
    # num_patches = M_t * 32, with H=8, W=4: num_patches = T * 8 * 4 = 32*T → T = M_t
    return max(1, math.ceil(weight_us / tilize_per_mrow_us))


def build_all_blockings(C_in, C_out, kernel_size, H, W, T, num_cores=120):
    padded_cin = aligned_channels(C_in)
    kT = kernel_size[0]

    cins = [c for c in range(32, padded_cin + 1, 32) if valid_cin(c, padded_cin, kernel_size)]
    couts = [c for c in range(32, aligned_channels(C_out) + 1, 32) if valid_cout(c, C_out)]

    # T_out for this shape (no padding in sweep — raw conv)
    T_out = T - (kT - 1) if kT > 1 else T

    # T_block candidates: data-driven based on T_out and weight read time estimate.
    # Goal: find T where M_t × per_row_tilize_us > weight_read_us, so the weight
    # stall is hidden. Per-row tilize ≈ 4µs; weight_KB / 20 GB/s = weight_us.
    # Also include all divisors of T_out up to a cap, since these avoid partial blocks.
    t_blocks_set = {1}
    if kT > 1 and T_out > 0:
        # All exact divisors of T_out up to T_out (avoid partial last block overhead)
        for t in range(2, min(T_out + 1, 32)):
            if T_out % t == 0:
                t_blocks_set.add(t)
        # Also try values that don't divide evenly but are useful (e.g. T_out - 1 pad)
        for t in [3, 5, 6, 7, 9, 11, 13, 15, 21]:
            if 1 < t <= T_out:
                t_blocks_set.add(t)
        # Cap: beyond T=21 the shard grows very large; rarely better
        t_blocks_set = {t for t in t_blocks_set if t <= min(T_out, 21)}
    t_blocks = sorted(t_blocks_set)

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
    skipped_par = 0
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
                    # Parallelism filter: skip combos using < 25% of cores
                    cores_used = compute_parallelism(
                        cin, cout, t_blk, h, w, kernel_size, H, W, T, C_in, C_out, num_cores
                    )
                    if cores_used < num_cores * 0.25:
                        skipped_par += 1
                        continue
                    combos.append((cin, cout, t_blk, h, w))
    if skipped_l1:
        print(f"Skipped {skipped_l1} combos for estimated L1 OOM")
    if skipped_par:
        print(f"Skipped {skipped_par} combos for low parallelism (<25% cores)")

    # Sort: T=1 runs first within each (cin, cout, H, W) group so the baseline
    # is established before evaluating T>1 variants. Within same T, prefer
    # larger cin (fewer blocks) and spatial near the t_needed target.
    def combo_priority(c):
        cin, cout, t, h, w = c
        t_first = 0 if t == 1 else 1  # T=1 always first to establish baseline
        cin_score = -cin  # larger cin = better
        # Prefer T near the estimated optimal for this (cin, cout)
        t_opt = t_needed_to_hide_weight_read(cin, cout, kernel_size)
        t_dist = abs(t - t_opt) if t > 1 else 0
        spatial_score = abs(h * w - 32)
        return (t_first, cin_score, t_dist, spatial_score)

    combos.sort(key=combo_priority)
    return combos


def _run_once(device, tt_input, tt_weight, tt_bias, cfg, C_out, kernel_size, ckc):
    t0 = time.perf_counter()
    o = ttnn.experimental.conv3d(
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
    us = (time.perf_counter() - t0) * 1e6
    ttnn.deallocate(o)
    return us


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
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Previous sweep JSON. Combos > 2× that best_us are skipped immediately.",
    )
    args = parser.parse_args()

    kernel_size = tuple(int(x) for x in args.kernel.split(","))
    C_in, C_out, T, H, W = args.C_in, args.C_out, args.T, args.H, args.W
    padded_cin = aligned_channels(C_in)

    # Determine num_cores from grid-size arg before opening device (for parallelism filter)
    if args.grid_size:
        gx, gy = args.grid_size.split("x")
        _num_cores = int(gx) * int(gy)
    else:
        _num_cores = 120  # BH default (12×10)

    combos = build_all_blockings(C_in, C_out, kernel_size, H, W, T, num_cores=_num_cores)
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

    # Load baseline from previous sweep if provided.
    # Seeds best_us and t1_baseline so early cutoffs kick in faster.
    best_us = float("inf")
    t1_baseline: dict = {}
    if args.baseline:
        bl_path = pathlib.Path(args.baseline)
        if bl_path.exists():
            with open(bl_path) as f:
                bl = json.load(f)
            best_us = bl.get("best_us") or float("inf")
            for r in bl.get("all_results", []):
                if r.get("status") == "ok" and r.get("us") and r["blocking"][2] == 1:
                    bl_cin, bl_cout, _, bl_h, bl_w = r["blocking"]
                    key = (bl_cin, bl_cout, bl_h, bl_w)
                    t1_baseline[key] = min(t1_baseline.get(key, float("inf")), r["us"])
            print(f"Baseline: best={best_us:.0f}us from {bl_path.name}, " f"{len(t1_baseline)} T=1 entries pre-loaded")
        else:
            print(f"Warning: baseline file {bl_path} not found, ignoring --baseline")

    # Pre-prepare weights per C_in_block (cache to avoid re-preparing)
    weight_cache = {}

    def get_weight(cin):
        if cin not in weight_cache:
            tt_w = ttnn.from_torch(w, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0)
            weight_cache[cin] = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=tt_w, C_in_block=cin, device=device
            )
        return weight_cache[cin]

    # Seed best_us from the conv3d blocking table (known-good production blockings).
    # This gives the cutoff logic a head start without needing a prior sweep JSON.
    if args.mesh:
        h_factor, w_factor = (int(x) for x in args.mesh.split("x"))
        table_blk = _BLOCKINGS.get((h_factor, w_factor, C_in, C_out, kernel_size, H, W)) or _DEFAULT_BLOCKINGS.get(
            (C_in, C_out, kernel_size)
        )
        if table_blk is not None:
            cin, cout, t_blk, h_blk, w_blk = table_blk
            cfg_tbl = ttnn.Conv3dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                T_out_block=t_blk,
                W_out_block=w_blk,
                H_out_block=h_blk,
                C_out_block=cout,
                C_in_block=cin,
                compute_with_storage_grid_size=grid_size,
            )
            tbl_args = (device, tt_input, get_weight(cin), tt_bias, cfg_tbl, C_out, kernel_size, ckc)
            try:
                for _ in range(WARMUP):
                    _run_once(*tbl_args)
                tbl_us = statistics.mean(_run_once(*tbl_args) for _ in range(RUNS))
                best_us = min(best_us, tbl_us)
                if t_blk == 1:
                    t1_baseline[(cin, cout, h_blk, w_blk)] = tbl_us
                print(f"Table blocking ({cin},{cout},{t_blk},{h_blk},{w_blk}) = {tbl_us:.0f}us (seeding best)")
            except Exception as e:
                print(f"Table blocking failed: {e}")

    # Sweep
    results = []
    best_blk = None
    t_start = time.time()
    ok_count = 0
    fail_count = 0

    for i, (cin, cout, t_blk, h_blk, w_blk) in enumerate(combos):
        try:
            tt_weight = get_weight(cin)

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
            # Progressive cutoff: after 20% of combos explored, tighten probe
            # threshold from 2.0× to 1.4×. By then we have a good best estimate.
            probe_threshold = 2.0 if i < len(combos) * 0.20 else 1.4

            # Baseline-guided T skip: if T>1 and T=1 baseline for this
            # (cin, cout, H, W) was > 1.5× global best, skip — the spatial
            # blocking itself is bad and larger T won't rescue it.
            spatial_key = (cin, cout, h_blk, w_blk)
            if t_blk > 1 and best_us < float("inf"):
                t1_us = t1_baseline.get(spatial_key)
                if t1_us is not None and t1_us > best_us * 1.5:
                    fail_count += 1
                    results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": t1_us, "status": "skip_t1_bad"})
                    continue

            # Two-tier probe: skip if clearly worse than best.
            # Tier 1: single probe run — if > 2× best, skip immediately.
            # Tier 2: second probe run for confirmation when close to cutoff (1.5-2×).
            probe_args = (device, tt_input, tt_weight, tt_bias, cfg, C_out, kernel_size, ckc)
            probe_us = _run_once(*probe_args)
            if best_us < float("inf"):
                if probe_us > best_us * probe_threshold:
                    # Clearly worse: skip immediately
                    fail_count += 1
                    results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": probe_us, "status": "slow"})
                    continue
                elif probe_us > best_us * (probe_threshold * 0.75):
                    # Borderline: one more probe to confirm before committing to full run
                    probe2 = _run_once(*probe_args)
                    if min(probe_us, probe2) > best_us * (probe_threshold * 0.9):
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
            # Record T=1 baseline for this spatial key
            if t_blk == 1:
                t1_baseline[(cin, cout, h_blk, w_blk)] = us
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
    ok_results = sorted([r for r in results if r["status"] == "ok"], key=lambda r: r["us"])
    print(f"\nTop 10:")
    for r in ok_results[:10]:
        b = r["blocking"]
        print(f"  ({b[0]:3d},{b[1]:3d},{b[2]},{b[3]:2d},{b[4]:2d}) = {r['us']:.0f} us")

    # Save
    out_path = pathlib.Path(args.output)
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
