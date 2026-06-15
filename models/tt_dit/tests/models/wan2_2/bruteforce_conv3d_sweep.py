#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Brute-force conv3d blocking sweep.

Enumerates all valid (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)
combinations for a given conv3d layer shape, filters out configs that would OOM L1
or underutilize the device, then benchmarks the survivors on hardware to find the
fastest blocking.

Run via pytest:

    pytest models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
        -k "bh_2x4 and up3_res" -s --timeout=0

    pytest models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
        -k "bh_2x4" -s --timeout=0   # all layers sequentially
"""

import csv
import json
import math
import os
import pathlib
import time

import pytest
import torch

import ttnn
from models.tt_dit.utils.conv3d import _BLOCKINGS, _DEFAULT_BLOCKINGS, aligned_channels

from ....utils.test import line_params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Timing is trace-based: each blocking is compiled once, then TRACE_ITERS op
# invocations are captured into a single trace. The trace is executed
# TRACE_EXECUTES times and the fastest per-op time is kept. Capturing N ops in
# one trace amortizes host dispatch to ~zero, so the reported µs is on-device
# execution time (much less noisy than per-call perf_counter timing).
TRACE_ITERS = 10  # op invocations captured per trace for the full measurement
TRACE_PROBE_ITERS = 3  # cheaper trace used to gate slow combos before full measure
TRACE_EXECUTES = 2  # timed trace executions; fastest per-op time is kept

# DRAM reserved for captured trace commands. 64 MB is ample for ~10 conv3d
# dispatches and tiny next to the activations these shapes allocate.
TRACE_REGION_SIZE = 64 * 1024 * 1024

# Compute fidelity used for every conv3d measurement in this file (sweep +
# baseline). HiFi2 trades some matmul precision for throughput vs HiFi4.
MATH_FIDELITY = ttnn.MathFidelity.HiFi2


TILE_SIZE = 2048  # bf16 tile = 32 * 32 * 2
FP32_TILE_SIZE = 4096  # fp32 tile = 32 * 32 * 4
TILE_HEIGHT = 32

# BH p150b L1 = 1,572,864 B.  Reserve 200 KB for kernel code/stack.
L1_BUDGET = 1_572_864 - 200 * 1024  # ~1,372 KB


# ---------------------------------------------------------------------------
# Blocking validity helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# L1 estimation — mirrors conv3d_program_factory CB sizing
# ---------------------------------------------------------------------------


def estimate_l1_bytes(cin_block, cout_block, t_blk, h_blk, w_blk, kernel_size, C_in, dtype_bytes=2):
    kT, kH, kW = kernel_size
    C_in_num_blocks = aligned_channels(C_in) // cin_block

    num_patches = t_blk * h_blk * w_blk
    patch_size = kT * kH * kW * cin_block
    padded_patch_size = math.ceil(patch_size / 32) * 32
    padded_patch_bytes = padded_patch_size * dtype_bytes

    matmul_M_t = math.ceil(num_patches / TILE_HEIGHT)
    matmul_K_t = math.ceil(patch_size / 32)
    matmul_N_t = math.ceil(cout_block / 32)

    vol2col_rm_pages = TILE_HEIGHT if num_patches % TILE_HEIGHT == 0 else min(num_patches, 2 * TILE_HEIGHT)
    cb_vol2col_rm = padded_patch_bytes * vol2col_rm_pages
    cb_vol2col_tiled = TILE_SIZE * matmul_K_t
    cb_weight = TILE_SIZE * matmul_K_t * matmul_N_t

    use_fp32_partials = C_in_num_blocks > 1
    partial_tile = FP32_TILE_SIZE if use_fp32_partials else TILE_SIZE
    cb_interm = partial_tile * matmul_M_t * matmul_N_t
    cb_result = TILE_SIZE * matmul_M_t * matmul_N_t

    cb_reduction = 0
    if C_in_num_blocks > 1:
        cb_reduction = partial_tile * matmul_M_t * matmul_N_t + TILE_SIZE

    cb_bias = TILE_SIZE * matmul_N_t
    cb_zero = TILE_SIZE if use_fp32_partials else 0

    T_shard = (t_blk - 1) + kT
    H_shard = (h_blk - 1) + kH
    W_shard = (w_blk - 1) + kW
    cb_shard = T_shard * H_shard * W_shard * cin_block * dtype_bytes

    return (
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


# ---------------------------------------------------------------------------
# Parallelism estimation
# ---------------------------------------------------------------------------


def compute_parallelism(cin_block, cout_block, t_blk, h_blk, w_blk, kernel_size, H, W, T, C_in, C_out, num_cores=120):
    padded_cin = aligned_channels(C_in)
    padded_cout = aligned_channels(C_out)
    kT = kernel_size[0]
    T_out = T - (kT - 1) if kT > 1 else T

    c_in_par = min(padded_cin // cin_block, num_cores)
    cores_per_out = max(1, num_cores // c_in_par)
    c_out_par = min(padded_cout // cout_block, cores_per_out)
    remaining = max(1, cores_per_out // c_out_par)

    T_out_blocks = max(1, math.ceil(T_out / t_blk))
    H_out_blocks = max(1, math.ceil(H / h_blk))
    W_out_blocks = max(1, math.ceil(W / w_blk))
    spatial_par = min(T_out_blocks * H_out_blocks * W_out_blocks, remaining)

    return c_in_par * c_out_par * spatial_par


# ---------------------------------------------------------------------------
# Combo generation — enumerate, filter, sort
# ---------------------------------------------------------------------------


def _t_needed_to_hide_weight_read(cin_block, cout_block, kernel_size):
    """Min T_out_block so tilize time hides the weight DRAM read."""
    kT, kH, kW = kernel_size
    matmul_K_t = math.ceil(kT * kH * kW * cin_block / 32)
    matmul_N_t = math.ceil(cout_block / 32)
    weight_us = (matmul_K_t * matmul_N_t * 2) / 20  # KB / ~20 GB/s
    return max(1, math.ceil(weight_us / 4.0))  # 4 µs per M-row tilize


def build_all_blockings(C_in, C_out, kernel_size, H, W, T, num_cores=120, max_t_block=None, hw_product=None):
    """Build sorted list of (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block) combos.

    max_t_block: if set, caps T_block candidates at this value.
                 BH 2x4 480p: use 7 (T=9+ causes device hangs on large spatial).
    hw_product:  if set, only test combos where H_block*W_block == hw_product.
                 BH 2x4 480p: use 32 (h*w=16/64 trigger hangs; 32 consistently wins).
    """
    padded_cin = aligned_channels(C_in)
    padded_cout = aligned_channels(C_out)
    kT = kernel_size[0]
    T_out = T - (kT - 1) if kT > 1 else T

    cins = [c for c in range(32, padded_cin + 1, 32) if valid_cin(c, padded_cin, kernel_size)]
    couts = [c for c in range(32, padded_cout + 1, 32) if valid_cout(c, C_out)]

    # T_block candidates: divisors of T_out plus a range of useful non-divisors.
    # Cap at 32 to cover larger T (e.g. T_out=64 for 4x32 720p cached t=16).
    t_max = min(T_out, 32)
    t_blocks_set = {1}
    if kT > 1 and T_out > 0:
        for t in range(2, min(T_out + 1, t_max + 1)):
            if T_out % t == 0:
                t_blocks_set.add(t)
        for t in [3, 5, 6, 7, 9, 11, 13, 15, 16, 21, 28, 32]:
            if 1 < t <= T_out and t <= t_max:
                t_blocks_set.add(t)
        t_blocks_set = {t for t in t_blocks_set if t <= t_max}
    if max_t_block is not None:
        t_blocks_set = {t for t in t_blocks_set if t <= max_t_block}
    t_blocks = sorted(t_blocks_set)

    # Spatial (H_block, W_block) candidates.
    tensor_vol = H * W
    min_hw = 2 if tensor_vol > 2000 else 1
    hw = [
        (h, w)
        for h in [min_hw, 2, 4, 8, 16, 32]
        if h <= H
        for w in [min_hw, 2, 4, 8, 16, 32]
        if w <= W and h * w <= 256
    ]
    hw = sorted(set(hw))

    # Drop small C_out_block on large tensors — never wins.
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
                    if hw_product is not None and h * w != hw_product:
                        continue
                    if estimate_l1_bytes(cin, cout, t_blk, h, w, kernel_size, C_in) > L1_BUDGET:
                        skipped_l1 += 1
                        continue
                    cores = compute_parallelism(cin, cout, t_blk, h, w, kernel_size, H, W, T, C_in, C_out, num_cores)
                    if cores < num_cores * 0.25:
                        skipped_par += 1
                        continue
                    combos.append((cin, cout, t_blk, h, w))

    if skipped_l1:
        print(f"Skipped {skipped_l1} combos for estimated L1 OOM")
    if skipped_par:
        print(f"Skipped {skipped_par} combos for low parallelism (<25% cores)")

    # T=1 first (establishes baseline), then larger cin, then T near optimal.
    def priority(c):
        cin, cout, t, h, w = c
        t_first = 0 if t == 1 else 1
        t_opt = _t_needed_to_hide_weight_read(cin, cout, kernel_size)
        return (t_first, -cin, abs(t - t_opt) if t > 1 else 0, abs(h * w - 32))

    combos.sort(key=priority)
    return combos


# ---------------------------------------------------------------------------
# Conv3d invocation + timing
# ---------------------------------------------------------------------------


def _invoke(args):
    """Run conv3d once and return the (undeallocated) output tensor.

    args = (device, tt_input, tt_weight, tt_bias, cfg, C_out, kernel_size, stride, padding, ckc)
    """
    _device, tt_input, tt_weight, tt_bias, cfg, C_out, kernel_size, stride, padding, ckc = args
    return ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        config=cfg,
        output_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode="zeros",
        dtype=ttnn.bfloat16,
        compute_kernel_config=ckc,
    )


def _trace_us(args, n_iters=TRACE_ITERS, executes=TRACE_EXECUTES):
    """Precise on-device timing via trace capture.

    Compiles once, captures n_iters conv3d invocations into a single trace
    (deallocating each output so peak memory stays at one output), then executes
    the trace `executes` times. Returns the fastest per-op time in µs.
    Trace execution carries no host dispatch, so the result is on-device time.
    """
    device = args[0]

    # Warm the program cache so capture records dispatch only (no JIT).
    o = _invoke(args)
    ttnn.deallocate(o)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(n_iters):
        o = _invoke(args)
        ttnn.deallocate(o)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    try:
        best_per_op = float("inf")
        for _ in range(executes):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            best_per_op = min(best_per_op, (time.perf_counter() - t0) * 1e6 / n_iters)
    finally:
        ttnn.release_trace(device, tid)
    return best_per_op


# ---------------------------------------------------------------------------
# Main sweep
#
# For each blocking combo the loop does:
#   1. Skip T>1 combos whose T=1 sibling was already way off the best.
#   2. Run 1 untimed call (JIT compile) + 1 host-timed probe.
#   3. If the probe is > 1.5× best, skip immediately ("slow").
#   4. Otherwise capture a trace of trace_iters ops and record the per-op time.
# ---------------------------------------------------------------------------


def run_sweep(
    device,
    C_in,
    C_out,
    kernel_size,
    T,
    H,
    W,
    output,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    h_factor=1,
    w_factor=1,
    grid_size=None,
    max_combos=500,
    max_t_block=None,
    hw_product=None,
    trace_iters=TRACE_ITERS,
):
    padded_cin = aligned_channels(C_in)
    _num_cores = grid_size.x * grid_size.y if grid_size else 120
    if grid_size is None:
        grid_size = device.compute_with_storage_grid_size()

    combos = build_all_blockings(
        C_in, C_out, kernel_size, H, W, T, num_cores=_num_cores, max_t_block=max_t_block, hw_product=hw_product
    )
    if max_combos and len(combos) > max_combos:
        print(f"Capping combos from {len(combos)} to {max_combos}")
        combos = combos[:max_combos]
    print(f"Shape: C_in={C_in} C_out={C_out} kernel={kernel_size} T={T} H={H} W={W}")
    print(f"stride={stride} padding={padding} h_factor={h_factor} w_factor={w_factor}")
    print(f"Total valid combos: {len(combos)}")

    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    is_mesh = hasattr(device, "get_num_devices") and device.get_num_devices() > 1
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

    torch.manual_seed(42)
    tt_input = ttnn.from_torch(
        torch.randn(1, T, H, W, padded_cin, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    w = torch.randn(C_out, padded_cin, *kernel_size, dtype=torch.float32)
    tt_bias = ttnn.from_torch(
        torch.randn(1, C_out, dtype=torch.float32),
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
        mesh_mapper=mesh_mapper,
    )

    # -- Weight cache (keyed by C_in_block) --
    weight_cache = {}

    def get_weight(cin):
        if cin not in weight_cache:
            tt_w = ttnn.from_torch(
                w,
                device=device,
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.TILE_LAYOUT,
                pad_value=0,
                mesh_mapper=mesh_mapper,
            )
            weight_cache[cin] = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=tt_w,
                C_in_block=cin,
                device=device,
            )
        return weight_cache[cin]

    def make_cfg(cin, cout, t_blk, h_blk, w_blk):
        return ttnn.Conv3dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            T_out_block=t_blk,
            W_out_block=w_blk,
            H_out_block=h_blk,
            C_out_block=cout,
            C_in_block=cin,
            compute_with_storage_grid_size=grid_size,
        )

    # -- Seed best_us from the production blocking table --
    best_us = float("inf")
    t1_baseline: dict = {}  # (cin, cout, h, w) -> best T=1 time

    kT, kH, kW = kernel_size
    H_out_key = H - (kH - 1)
    W_out_key = W - (kW - 1)
    table_blk = _BLOCKINGS.get(
        (h_factor, w_factor, C_in, C_out, kernel_size, T, H_out_key, W_out_key)
    ) or _DEFAULT_BLOCKINGS.get((C_in, C_out, kernel_size))
    table_us = None  # HiFi2 time of the original (table) blocking — the per-shape seed
    if table_blk is not None:
        cin, cout, t_blk, h_blk, w_blk = table_blk
        cfg_tbl = make_cfg(cin, cout, t_blk, h_blk, w_blk)
        tbl_args = (device, tt_input, get_weight(cin), tt_bias, cfg_tbl, C_out, kernel_size, stride, padding, ckc)
        try:
            tbl_us = _trace_us(tbl_args, trace_iters)
            table_us = tbl_us
            best_us = min(best_us, tbl_us)
            if t_blk == 1:
                t1_baseline[(cin, cout, h_blk, w_blk)] = tbl_us
            print(f"Table blocking ({cin},{cout},{t_blk},{h_blk},{w_blk}) = {tbl_us:.0f}us (seeding best)")
        except Exception as e:
            print(f"Table blocking failed: {e}")

    # -- Sweep --
    results = []
    best_blk = None
    t_start = time.time()
    ok_count = 0
    fail_count = 0
    PROBE_THRESHOLD = 1.5

    for i, (cin, cout, t_blk, h_blk, w_blk) in enumerate(combos):
        try:
            tt_weight = get_weight(cin)
            cfg = make_cfg(cin, cout, t_blk, h_blk, w_blk)
            args = (device, tt_input, tt_weight, tt_bias, cfg, C_out, kernel_size, stride, padding, ckc)

            # Skip T>1 if the same spatial with T=1 was already way off best.
            if t_blk > 1 and best_us < float("inf"):
                t1_us = t1_baseline.get((cin, cout, h_blk, w_blk))
                if t1_us is not None and t1_us > best_us * PROBE_THRESHOLD:
                    fail_count += 1
                    results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": t1_us, "status": "skip_t1_bad"})
                    continue

            # Cheap trace probe (few iters) to gate slow combos before the full
            # measurement. Trace-based so it is directly comparable to best_us.
            probe_us = _trace_us(args, TRACE_PROBE_ITERS, executes=1)

            if best_us < float("inf") and probe_us > best_us * PROBE_THRESHOLD:
                fail_count += 1
                results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": probe_us, "status": "slow"})
                continue

            # Precise trace-based measurement.
            us = _trace_us(args, trace_iters)
            ok_count += 1

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
                    f"  [{i+1:3d}/{len(combos)}] ({cin:3d},{cout:3d},{t_blk},{h_blk:2d},{w_blk:2d}) {us:8.0f}us{tag}",
                    flush=True,
                )

        except Exception as e:
            fail_count += 1
            print(f"  FAIL ({cin},{cout},{t_blk},{h_blk},{w_blk}): {e}", flush=True)
            results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": None, "status": "fail"})

    # -- Report --
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min). {ok_count} ok, {fail_count} failed.")
    ok_results = sorted([r for r in results if r["status"] == "ok"], key=lambda r: r["us"])
    print(f"\nTop 10:")
    for r in ok_results[:10]:
        b = r["blocking"]
        print(f"  ({b[0]:3d},{b[1]:3d},{b[2]},{b[3]:2d},{b[4]:2d}) = {r['us']:.0f} us")

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "C_in": C_in,
                "C_out": C_out,
                "kernel": list(kernel_size),
                "T": T,
                "H": H,
                "W": W,
                "h_factor": h_factor,
                "w_factor": w_factor,
                "num_combos": len(combos),
                "num_ok": ok_count,
                "num_fail": fail_count,
                "elapsed_s": elapsed,
                "table_blocking": list(table_blk) if table_blk else None,
                "table_us": table_us,
                "best_blocking": list(best_blk) if best_blk else None,
                "best_us": best_us if best_us < float("inf") else None,
                "top_20": ok_results[:20],
                "all_results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# Pytest — uses same fixture chain as test_pipeline_wan.py
# ---------------------------------------------------------------------------

# BH Loud Box 2x4, 480p, cached t_chunk_size=7 (vae_t_chunk_size=7)
# Same spatial dims as t_chunk=1, but larger T values.
#
# Cached T per stage (cur_T grows: 7 → 14 → 28):
#   stage 0: T_res=9, T_tconv=9, T_spatial=14   (cur_T=7)
#   stage 1: T_res=16, T_tconv=16, T_spatial=28  (cur_T=14)
#   stage 2: T_res=30, T_spatial=28              (cur_T=28, no temporal up)
#   stage 3: T_res=30                            (cur_T=28, no temporal up)
_SWEEP_LAYERS_H2W4_480P_T7 = [
    # (name,            C_in, C_out, kernel,   stride,   padding,   T,   H,   W, h, w)
    # --- stage 0 (cur_T=7) ---
    ("conv_in", 32, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 9, 32, 28, 2, 4),
    ("lat_mid_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 9, 32, 28, 2, 4),
    ("up0_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 9, 30, 26, 2, 4),
    ("up0_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 14, 62, 54, 2, 4),
    # --- stage 1 (cur_T=14) ---
    ("up1_res0", 192, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 16, 62, 54, 2, 4),
    ("up1_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 16, 62, 54, 2, 4),
    ("up1_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 16, 60, 52, 2, 4),
    ("up1_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 28, 122, 106, 2, 4),
    # --- stage 2 (cur_T=28, no temporal upsample) ---
    ("up2_res", 192, 192, (3, 3, 3), (1, 1, 1), (0, 0, 0), 30, 122, 106, 2, 4),
    ("up2_spatial", 192, 96, (1, 3, 3), (1, 1, 1), (0, 0, 0), 28, 242, 210, 2, 4),
    # --- stage 3 (cur_T=28, no temporal upsample) ---
    ("up3_res", 96, 96, (3, 3, 3), (1, 1, 1), (0, 0, 0), 30, 242, 210, 2, 4),
    ("conv_out", 96, 3, (3, 3, 3), (1, 1, 1), (0, 0, 0), 30, 242, 210, 2, 4),
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(2, 4), (2, 4), {**line_params, "trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_2x4"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_H2W4_480P_T7,
    ids=[l[0] for l in _SWEEP_LAYERS_H2W4_480P_T7],
)
def test_bruteforce_sweep_h2w4_480p_t7(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_h2w4_480p_t7/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        # BH 2x4 480p hang mitigations: T=9+ and h*w≠32 trigger device hangs
        # on large-spatial temporal layers. Keep these as fixture-specific params
        # so other sweeps (e.g. 4x32) can use the full search space.
        max_t_block=7,
        hw_product=32,
    )


# ---------------------------------------------------------------------------
# BH Galaxy 4x8, 720p, 81 frames full-T (latent T=21)
# ---------------------------------------------------------------------------
# Mesh: h_factor=4, w_factor=8.  Per-device spatial (unpadded):
#   lat(23,20)  mid(46,40)  hi(92,80)  full(184,160)
#
# Padded (int_pad=(0,1,1)): lat(25,22) mid(48,42) hi(94,82) full(186,162)
# (3,1,1): lat(23,20) mid(46,40)
#
# Same T flow as all 81-frame configs:
#   T_res=23, T_tconv=22, T_sp=41, T_res1=43, T_tconv1=42, T_sp1=81, T_res2/3=83
# ---------------------------------------------------------------------------
_SWEEP_LAYERS_H4W8_720P_FULL_T = [
    # (name, C_in, C_out, kernel, stride, padding, T, H, W, h, w)
    # Most-to-least compute (T × H × W)
    ("up3_res", 96, 96, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 186, 162, 4, 8),
    ("conv_out", 96, 3, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 186, 162, 4, 8),
    ("up2_spatial", 192, 96, (1, 3, 3), (1, 1, 1), (0, 0, 0), 81, 186, 162, 4, 8),
    ("up2_res", 192, 192, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 94, 82, 4, 8),
    ("up1_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 81, 94, 82, 4, 8),
    ("up1_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 43, 48, 42, 4, 8),
    ("up1_res0", 192, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 43, 48, 42, 4, 8),
    ("up0_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 41, 48, 42, 4, 8),
    ("up1_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 42, 46, 40, 4, 8),
    ("lat_mid_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 23, 25, 22, 4, 8),
    ("conv_in", 32, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 23, 25, 22, 4, 8),
    ("up0_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 22, 23, 20, 4, 8),
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_4x8_720p_full_t_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_H4W8_720P_FULL_T,
    ids=[l[0] for l in _SWEEP_LAYERS_H4W8_720P_FULL_T],
)
def test_bruteforce_sweep_h4w8_720p_full_t(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_h4w8_720p_full_t/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        max_t_block=8,
        # hw_product=32: avoids BH hang (second T=7 combo with non-32 hw hangs).
        # Explore T=1 with larger hw separately if needed.
        hw_product=32,
    )


# ---------------------------------------------------------------------------
# BH Galaxy 4x8, 720p, cached t_chunk_size=16 (vae_t_chunk_size=16)
# ---------------------------------------------------------------------------
# Mesh: h_factor=4, w_factor=8.  Per-device spatial (unpadded, same as full-T):
#   lat(23,20)  mid(46,40)  hi(92,80)  full(184,160)
#
# Padded dims fed to conv3d (add int_pad=(0,1,1) for (3,3,3)/(1,3,3)):
#   lat(25,22)  mid(48,42)  hi(94,82)  full(186,162)
# (3,1,1) kernels use no spatial padding: lat(23,20) mid(46,40)
#
# Cached T from compute_decoder_dims(720, 1280, 4, 8, 16, cached=True):
#   stage 0 (cur_T=16): T_res=18, T_tconv=18, T_spatial=32
#   stage 1 (cur_T=32): T_res=34, T_tconv=34, T_spatial=64
#   stage 2 (cur_T=64): T_res=66, T_spatial=64  (no temporal upsample)
#   stage 3 (cur_T=64): T_res=66
#
# Layers ordered most-to-least compute (T × H × W volume).
# ---------------------------------------------------------------------------
_SWEEP_LAYERS_H4W8_720P_T16 = [
    # (name,           C_in, C_out, kernel,   stride,   padding,   T,   H,   W, h, w)
    # --- stage 3 / 2 (largest spatial, cur_T=64) ---
    ("up3_res", 96, 96, (3, 3, 3), (1, 1, 1), (0, 0, 0), 66, 186, 162, 4, 8),  # T=66
    ("conv_out", 96, 3, (3, 3, 3), (1, 1, 1), (0, 0, 0), 66, 186, 162, 4, 8),  # T=66
    ("up2_spatial", 192, 96, (1, 3, 3), (1, 1, 1), (0, 0, 0), 64, 186, 162, 4, 8),  # T=64
    ("up2_res", 192, 192, (3, 3, 3), (1, 1, 1), (0, 0, 0), 66, 94, 82, 4, 8),  # T=66
    # --- stage 1 (cur_T=32) ---
    ("up1_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 64, 94, 82, 4, 8),  # T=64
    ("up1_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 34, 48, 42, 4, 8),  # T=34
    ("up1_res0", 192, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 34, 48, 42, 4, 8),  # T=34
    ("up1_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 34, 46, 40, 4, 8),  # T=34
    # --- stage 0 (cur_T=16) ---
    ("up0_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 32, 48, 42, 4, 8),  # T=32
    ("lat_mid_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 18, 25, 22, 4, 8),  # T=18
    ("conv_in", 32, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 18, 25, 22, 4, 8),  # T=18
    ("up0_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 18, 23, 20, 4, 8),  # T=18
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_4x8_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_H4W8_720P_T16,
    ids=[l[0] for l in _SWEEP_LAYERS_H4W8_720P_T16],
)
def test_bruteforce_sweep_h4w8_720p_t16(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_h4w8_720p_t16/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        # BH hardware hang mitigations: h*w≠32 causes device hangs.
        # T=8 confirmed safe (t_chunk/2); T=9+ hangs for C_in≥192.
        # Covers t_chunk divisors {1,2,4,8} naturally.
        max_t_block=8,
        hw_product=32,
    )


# ---------------------------------------------------------------------------
# BH Galaxy 4x8, 720p, cached t_chunk_size=1 (first/anchor frame)
# All stages see T = t_chunk + 2 = 3. T_out=1 for (3,3,3) kernels so only
# T_block=1 is valid — very fast sweep with few combos.
# ---------------------------------------------------------------------------
_SWEEP_LAYERS_H4W8_720P_T1 = [
    # (name, C_in, C_out, kernel, stride, padding, T, H, W, h, w)
    # Ordered most-to-least compute. All T=3.
    ("up3_res", 96, 96, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 186, 162, 4, 8),
    ("conv_out", 96, 3, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 186, 162, 4, 8),
    ("up2_spatial", 192, 96, (1, 3, 3), (1, 1, 1), (0, 0, 0), 3, 186, 162, 4, 8),
    ("up2_res", 192, 192, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 94, 82, 4, 8),
    ("up1_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 3, 94, 82, 4, 8),
    ("up1_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 48, 42, 4, 8),
    ("up1_res0", 192, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 48, 42, 4, 8),
    ("up1_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 3, 46, 40, 4, 8),
    ("up0_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 3, 48, 42, 4, 8),
    ("lat_mid_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 25, 22, 4, 8),
    ("conv_in", 32, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 25, 22, 4, 8),
    ("up0_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 3, 23, 20, 4, 8),
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_4x8_t1_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_H4W8_720P_T1,
    ids=[l[0] for l in _SWEEP_LAYERS_H4W8_720P_T1],
)
def test_bruteforce_sweep_h4w8_720p_t1(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_h4w8_720p_t1/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        hw_product=32,
        # T_out=1 for (3,3,3) kernels → only T_block=1; no hang risk.
    )


# ---------------------------------------------------------------------------
# BH Galaxy 4x8, 720p, cached t_chunk_size=15 (last partial chunk)
# Stage 0: T_res=17. Stage 1: T_res=32, T_sp_in=30. Stage 2/3: T_res=62.
# ---------------------------------------------------------------------------
_SWEEP_LAYERS_H4W8_720P_T15 = [
    # (name, C_in, C_out, kernel, stride, padding, T, H, W, h, w)
    # Ordered most-to-least compute.
    ("up3_res", 96, 96, (3, 3, 3), (1, 1, 1), (0, 0, 0), 62, 186, 162, 4, 8),
    ("conv_out", 96, 3, (3, 3, 3), (1, 1, 1), (0, 0, 0), 62, 186, 162, 4, 8),
    ("up2_spatial", 192, 96, (1, 3, 3), (1, 1, 1), (0, 0, 0), 60, 186, 162, 4, 8),
    ("up2_res", 192, 192, (3, 3, 3), (1, 1, 1), (0, 0, 0), 62, 94, 82, 4, 8),
    ("up1_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 60, 94, 82, 4, 8),
    ("up1_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 32, 48, 42, 4, 8),
    ("up1_res0", 192, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 32, 48, 42, 4, 8),
    ("up0_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 30, 48, 42, 4, 8),
    ("up1_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 32, 46, 40, 4, 8),
    ("lat_mid_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 17, 25, 22, 4, 8),
    ("conv_in", 32, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 17, 25, 22, 4, 8),
    ("up0_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 17, 23, 20, 4, 8),
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_4x8_t15_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_H4W8_720P_T15,
    ids=[l[0] for l in _SWEEP_LAYERS_H4W8_720P_T15],
)
def test_bruteforce_sweep_h4w8_720p_t15(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_h4w8_720p_t15/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        max_t_block=8,
        hw_product=32,
    )


# ---------------------------------------------------------------------------
# BH Galaxy 4x8, 480p, full-T (81 frames, latent T=21)
# ---------------------------------------------------------------------------
# Mesh: h_factor=4, w_factor=8.  Per-device spatial (unpadded):
#   lat(15,13)  mid(30,26)  hi(60,52)  full(120,104)
#
# Padded dims fed to conv3d (add int_pad=(0,1,1) for (3,3,3)/(1,3,3)):
#   lat(17,15)  mid(32,28)  hi(62,54)  full(122,106)
# (3,1,1) kernels use no spatial padding: lat(15,13) mid(30,26)
#
# Temporal: same as all 81-frame configs — T_res=23, T_tconv=22,
# T_spatial=41, T_res1=43, T_tconv1=42, T_spatial1=81, T_res2/3=83.
#
# Layers ordered most-to-least compute (T × H_sweep × W_sweep).
# ---------------------------------------------------------------------------
_SWEEP_LAYERS_H4W8_480P_FULL_T = [
    # (name,           C_in, C_out, kernel,   stride,   padding,   T,   H,   W, h, w)
    # --- stage 3 / 2 (largest spatial) ---
    ("up3_res", 96, 96, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 122, 106, 4, 8),
    ("conv_out", 96, 3, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 122, 106, 4, 8),
    ("up2_spatial", 192, 96, (1, 3, 3), (1, 1, 1), (0, 0, 0), 81, 122, 106, 4, 8),
    ("up2_res", 192, 192, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 62, 54, 4, 8),
    # --- stage 1 ---
    ("up1_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 81, 62, 54, 4, 8),
    ("up1_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 43, 32, 28, 4, 8),
    ("up1_res0", 192, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 43, 32, 28, 4, 8),
    ("up0_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 41, 32, 28, 4, 8),
    ("up1_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 42, 30, 26, 4, 8),
    # --- stage 0 (small spatial) ---
    ("lat_mid_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 23, 17, 15, 4, 8),
    ("conv_in", 32, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 23, 17, 15, 4, 8),
    ("up0_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 22, 15, 13, 4, 8),
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_4x8_480p_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_H4W8_480P_FULL_T,
    ids=[l[0] for l in _SWEEP_LAYERS_H4W8_480P_FULL_T],
)
def test_bruteforce_sweep_h4w8_480p_full_t(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_h4w8_480p_full_t/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        max_t_block=8,
        # hw_product=None: 4x8 480p full-T large stages (W=52-104) benefit from
        # larger spatial blocks (hw=64-256) — same as 720p full-T which uses
        # hw=64-256 without hangs. hw_product=32 was causing 9× VAE perf gap.
        hw_product=None,
    )


# ---------------------------------------------------------------------------
# BH Galaxy 6U 4x32, 480p, full-T (81 frames, latent T=21)
# ---------------------------------------------------------------------------
# Mesh: h_factor=4, w_factor=32.  Per-device spatial (unpadded):
#   lat(15,3)  mid(30,6)  hi(60,12)  full(120,24)
#
# Padded dims fed to conv3d (add int_pad=(0,1,1) for (3,3,3)/(1,3,3)):
#   lat(17,5)  mid(32,8)  hi(62,14)  full(122,26)
# (3,1,1) kernels use no spatial padding: lat(15,3) mid(30,6)
#
# Same temporal dims as all 81-frame configs (T_res=23, T_tconv=22, etc.).
# Very narrow W per device (W=3-26). Layers ordered most-to-least compute.
# ---------------------------------------------------------------------------
_SWEEP_LAYERS_H4W32_480P_FULL_T = [
    # (name,           C_in, C_out, kernel,   stride,   padding,   T,   H,   W, h, w)
    # --- stage 3 / 2 (largest spatial) ---
    ("up3_res", 96, 96, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 122, 26, 4, 32),
    ("conv_out", 96, 3, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 122, 26, 4, 32),
    ("up2_spatial", 192, 96, (1, 3, 3), (1, 1, 1), (0, 0, 0), 81, 122, 26, 4, 32),
    ("up2_res", 192, 192, (3, 3, 3), (1, 1, 1), (0, 0, 0), 83, 62, 14, 4, 32),
    # --- stage 1 ---
    ("up1_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 81, 62, 14, 4, 32),
    ("up1_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 43, 32, 8, 4, 32),
    ("up1_res0", 192, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 43, 32, 8, 4, 32),
    ("up0_spatial", 384, 192, (1, 3, 3), (1, 1, 1), (0, 0, 0), 41, 32, 8, 4, 32),
    ("up1_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 42, 30, 6, 4, 32),
    # --- stage 0 (small spatial) ---
    ("lat_mid_res", 384, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 23, 17, 5, 4, 32),
    ("conv_in", 32, 384, (3, 3, 3), (1, 1, 1), (0, 0, 0), 23, 17, 5, 4, 32),
    ("up0_tconv", 384, 768, (3, 1, 1), (1, 1, 1), (0, 0, 0), 22, 15, 3, 4, 32),
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_4x32_480p_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_H4W32_480P_FULL_T,
    ids=[l[0] for l in _SWEEP_LAYERS_H4W32_480P_FULL_T],
)
def test_bruteforce_sweep_h4w32_480p_full_t(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_h4w32_480p_full_t/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        max_t_block=8,
        # hw_product=32: large stages (W=14-26) support h*w=32 pairs; small
        # stages (W=3-8) get fewer combos but still avoid hang-triggering shapes.
        hw_product=32,
    )


# ---------------------------------------------------------------------------
# Table-driven generic sweep — every shape in models/tt_dit/utils/conv3d.py
# ---------------------------------------------------------------------------
# Each _BLOCKINGS key is (h_factor, w_factor, C_in, C_out, kernel, T, H_out, W_out)
# where H_out/W_out are the *output* spatial dims (the conv runs valid, padding=0,
# so the producing pipeline pre-pads the input). The swept *input* shape is
# recovered exactly:
#
#     T_in = T          (kT does not reduce T in the table convention)
#     H_in = H_out + (kH - 1)
#     W_in = W_out + (kW - 1)
#     stride = (1, 1, 1), padding = (0, 0, 0)
#
# This reproduces every per-chip conv3d shape any production mesh would see, all
# on a single chip (1x1 submesh). The (h_factor, w_factor) tag is carried through
# only so the result lands in the right bucket and run_sweep can seed best_us from
# the matching table entry.
#
# Run on a single chip:
#
#     pytest models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
#         -k "sweep_all" -s --timeout=0                 # every table shape
#
#     pytest models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
#         -k "sweep_all and h4w8 and 96x96" -s --timeout=0    # subset by id
#
# Search-space limits are configurable via env vars (safe BH defaults shown):
#     CONV3D_SWEEP_MAX_T_BLOCK=8     # cap T_out_block; "none" to disable
#     CONV3D_SWEEP_HW_PRODUCT=32     # require H_block*W_block==N; "none" for full
#     CONV3D_SWEEP_MAX_COMBOS=500    # cap combos benchmarked per shape
#     CONV3D_SWEEP_TRACE_ITERS=10    # conv3d invocations captured per trace
#
# e.g. widen the spatial search for a layer whose true winner uses hw=64:
#     CONV3D_SWEEP_HW_PRODUCT=none pytest ... -k "sweep_all and h4w8_96x96_k333_T83_H186"
# ---------------------------------------------------------------------------


def _sweep_params_from_blockings():
    """Derive (param tuple, pytest id) for every entry in the _BLOCKINGS table."""
    params = []
    ids = []
    for h_factor, w_factor, C_in, C_out, kernel, T, H_out, W_out in _BLOCKINGS:
        kT, kH, kW = kernel
        H = H_out + (kH - 1)
        W = W_out + (kW - 1)
        params.append((C_in, C_out, kernel, (1, 1, 1), (0, 0, 0), T, H, W, h_factor, w_factor))
        ids.append(f"h{h_factor}w{w_factor}_{C_in}x{C_out}_k{kT}{kH}{kW}_T{T}_H{H}_W{W}")
    return params, ids


_SWEEP_PARAMS_ALL, _SWEEP_IDS_ALL = _sweep_params_from_blockings()


def _env_opt(name, default):
    """Parse an int sweep-limit env var. 'none'/'off'/'' -> None; unset -> default."""
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("none", "off", ""):
        return None
    return int(v)


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_PARAMS_ALL,
    ids=_SWEEP_IDS_ALL,
)
def test_bruteforce_sweep_all(
    mesh_device, mesh_shape, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    kT, kH, kW = kernel
    output = f"sweep_results_all/h{h_factor}w{w_factor}/{C_in}x{C_out}_k{kT}{kH}{kW}_T{T}_H{H}_W{W}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=_env_opt("CONV3D_SWEEP_MAX_COMBOS", 500),
        max_t_block=_env_opt("CONV3D_SWEEP_MAX_T_BLOCK", 8),
        hw_product=_env_opt("CONV3D_SWEEP_HW_PRODUCT", 32),
        trace_iters=_env_opt("CONV3D_SWEEP_TRACE_ITERS", TRACE_ITERS),
    )


# ---------------------------------------------------------------------------
# Baseline collection — measure every table shape with ITS OWN table blocking
# ---------------------------------------------------------------------------
# This does NOT sweep. For each entry in _BLOCKINGS it runs the conv3d once with
# exactly the blocking the production table prescribes and records the trace-based
# on-device time. The result is the "current production" baseline to compare any
# re-swept blockings against. All shapes run on a single chip (1x1 submesh).
#
#     pytest models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
#         -k "collect_baseline" -s --timeout=0
#
# Output CSV path defaults to conv3d_baseline_perf.csv (override with
# CONV3D_BASELINE_CSV=<path>). Rows are written + flushed as each shape completes,
# so a device hang mid-run still preserves everything measured so far.
# ---------------------------------------------------------------------------

BASELINE_CSV = "conv3d_baseline_perf.csv"

_BASELINE_FIELDS = [
    "h_factor",
    "w_factor",
    "C_in",
    "C_out",
    "kernel",
    "T",
    "H_out",
    "W_out",
    "H_in",
    "W_in",
    "Cin_block",
    "Cout_block",
    "T_block",
    "H_block",
    "W_block",
    "baseline_us",
    "status",
]


def _measure_table_blocking(device, C_in, C_out, kernel_size, T, H, W, blocking, grid_size, ckc, trace_iters):
    """Build fresh tensors for one shape, time the given blocking via trace, clean up."""
    padded_cin = aligned_channels(C_in)
    cin_blk, cout_blk, t_blk, h_blk, w_blk = blocking

    torch.manual_seed(42)
    tt_input = ttnn.from_torch(
        torch.randn(1, T, H, W, padded_cin, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_w = ttnn.from_torch(
        torch.randn(C_out, padded_cin, *kernel_size, dtype=torch.float32),
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )
    tt_weight = ttnn.experimental.prepare_conv3d_weights(weight_tensor=tt_w, C_in_block=cin_blk, device=device)
    tt_bias = ttnn.from_torch(
        torch.randn(1, C_out, dtype=torch.float32),
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t_blk,
        W_out_block=w_blk,
        H_out_block=h_blk,
        C_out_block=cout_blk,
        C_in_block=cin_blk,
        compute_with_storage_grid_size=grid_size,
    )
    args = (device, tt_input, tt_weight, tt_bias, cfg, C_out, kernel_size, (1, 1, 1), (0, 0, 0), ckc)
    try:
        return _trace_us(args, trace_iters)
    finally:
        for t in (tt_input, tt_w, tt_weight, tt_bias):
            ttnn.deallocate(t)


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_collect_baseline(mesh_device, mesh_shape):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    grid_size = device.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    trace_iters = _env_opt("CONV3D_SWEEP_TRACE_ITERS", TRACE_ITERS)

    out_path = pathlib.Path(os.environ.get("CONV3D_BASELINE_CSV", BASELINE_CSV))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(_BLOCKINGS)
    n_ok = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BASELINE_FIELDS)
        writer.writeheader()
        f.flush()
        for i, (key, blk) in enumerate(_BLOCKINGS.items()):
            h_factor, w_factor, C_in, C_out, kernel, T, H_out, W_out = key
            kT, kH, kW = kernel
            H = H_out + (kH - 1)
            W = W_out + (kW - 1)
            cin_blk, cout_blk, t_blk, h_blk, w_blk = blk
            row = {
                "h_factor": h_factor,
                "w_factor": w_factor,
                "C_in": C_in,
                "C_out": C_out,
                "kernel": f"{kT}x{kH}x{kW}",
                "T": T,
                "H_out": H_out,
                "W_out": W_out,
                "H_in": H,
                "W_in": W,
                "Cin_block": cin_blk,
                "Cout_block": cout_blk,
                "T_block": t_blk,
                "H_block": h_blk,
                "W_block": w_blk,
                "baseline_us": "",
                "status": "",
            }
            tag = f"[{i+1:3d}/{n}] h{h_factor}w{w_factor} {C_in}x{C_out} k{kT}{kH}{kW} T{T} H{H} W{W} blk={blk}"
            try:
                us = _measure_table_blocking(device, C_in, C_out, kernel, T, H, W, blk, grid_size, ckc, trace_iters)
                row["baseline_us"] = f"{us:.1f}"
                row["status"] = "ok"
                n_ok += 1
                print(f"{tag} -> {us:.1f}us", flush=True)
            except Exception as e:
                row["status"] = f"fail: {e}"
                print(f"{tag} -> FAIL: {e}", flush=True)
            writer.writerow(row)
            f.flush()

    print(f"\nBaseline: {n_ok}/{n} shapes measured. Saved to {out_path}")
