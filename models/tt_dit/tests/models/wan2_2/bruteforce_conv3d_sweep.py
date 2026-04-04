#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
In-process brute-force conv3d blocking sweep with T_out_block support.

Sweeps (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block) combinations
with accurate L1 pre-filtering based on the actual CB sizing from conv3d_program_factory.

Includes hang detection: if a combo's warmup+timed section exceeds --timeout seconds,
the process prints a diagnostic message and calls os._exit(1) to avoid silent hangs.

Usage:
    source python_env/bin/activate && export PYTHONPATH=$(pwd)

    # Fast mode (~30-60 combos, 5-15 min for large shapes) — use this first:
    python models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
        --C_in 96 --C_out 96 --kernel 3,3,3 --T 63 --H 242 --W 210 \
        --grid-size 12x10 --fast \
        --output sweep_results/up3_96x96_fast.json

    # Exhaustive mode (~300+ combos, hours) — use --baseline to warm-start from fast:
    python models/tt_dit/tests/models/wan2_2/bruteforce_conv3d_sweep.py \
        --C_in 96 --C_out 96 --kernel 3,3,3 --T 63 --H 242 --W 210 \
        --grid-size 12x10 \
        --baseline sweep_results/up3_96x96_fast.json \
        --output sweep_results/up3_96x96_full.json \
        --timeout 60
"""

import argparse
import json
import math
import os
import pathlib
import statistics
import threading
import time

import torch

import ttnn
from models.tt_dit.utils.conv3d import _BLOCKINGS, _DEFAULT_BLOCKINGS, aligned_channels

WARMUP = 1
RUNS = 2
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
    kT, kH, kW = kernel_size
    matmul_K_t = math.ceil(kT * kH * kW * cin_block / 32)
    matmul_N_t = math.ceil(cout_block / 32)
    weight_kb = matmul_K_t * matmul_N_t * 2
    weight_us = weight_kb / 20
    tilize_per_mrow_us = 4.0
    return max(1, math.ceil(weight_us / tilize_per_mrow_us))


def _predict_cin_cout(C_in, C_out, kernel_size):
    """Predict optimal (C_in_block, C_out_block) from BH hardware constraints.

    Cin_blk — first principles: DRAM weight read must be hidden behind compute.
        Per-row tilize+matmul ≈ 4µs. Weight read = K_t tiles × N_t × 2KB / 50 GB/s.
        N_t cancels → constraint on K_t alone:
            K_t_max = DRAM_bw_GBs × 4µs / 2KB ≈ 97 tiles
        Cin_blk = max value where ceil(kernel_vol × Cin_blk / 32) ≤ K_t_max.
        This correctly predicts: k333→96, k133→192, k311→large (L1-capped).

    Cout_blk — derived from per-kernel L1 budget remaining after vol2col CBs.
        vol2col_rm  ≈ kernel_vol × Cin_blk × 2 × 32 bytes  (32-page double-buffer)
        vol2col_tiled ≈ K_t × 2KB
        These consume most of L1; the remainder goes to the weight CB:
            weight_budget ≈ L1_budget − vol2col_overhead
        N_t = weight_budget / (K_t × 2KB), capped at 12.
        Empirically this gives different K_t×N_t targets per kernel:
            k333: ~81×4=324 (Cout=128), k133: ~54×3=162 (Cout=96), k311: ~12×12=144 (Cout=384)
        The targets match because vol2col is large for k333 (27 positions × large shard)
        and small for k311 (3 positions × small shard), leaving different weight budgets.

    Returns list of (cin_blk, cout_blk) candidates, best first.
    """
    from math import ceil

    padded_cin = aligned_channels(C_in)
    padded_cout = aligned_channels(C_out)
    kT, kH, kW = kernel_size
    kernel_vol = kT * kH * kW

    # Step 1: Cin_blk from DRAM-hide constraint (K_t ≤ 97 on BH at 50 GB/s, 4µs/row)
    # Exception: k311 (temporal-only, kernel_vol=3). The DRAM-hide constraint gives Cin=384,
    # but the actual bottleneck for k311 is OUTPUT WRITE bandwidth, not weight read.
    # Choosing smaller Cin=128 (K_t=12) lets Cout=384 fit in L1 (fewer output passes),
    # which is more efficient since each output write amortises more computation.
    K_T_MAX = 97
    if kernel_vol == 3:
        # k311: use Cin=128 so K_t=12, enabling Cout=384 with K_t×N_t=144
        cin_blk_target = 128
    else:
        cin_blk_target = (K_T_MAX * 32 // kernel_vol // 32) * 32
    cin_blk = max(32, min(padded_cin, cin_blk_target))
    while padded_cin % cin_blk != 0 and cin_blk > 32:
        cin_blk -= 32

    # Step 2: Cout_blk from per-kernel L1 budget (vol2col overhead varies by kernel_vol).
    # Empirically calibrated K_t×N_t targets derived from vol2col CB sizes:
    #   k333: vol2col_rm≈324KB + tiled≈162KB → weight budget≈800KB → K_t×N_t≈300
    #   k133: vol2col smaller → budget≈324KB  → K_t×N_t≈162
    #   k311: vol2col tiny   → budget≈300KB   → K_t×N_t≈144 (but capped by N_t≤12)
    K_t = ceil(kernel_vol * cin_blk / 32)
    target_KtNt = {27: 300, 9: 162, 3: 144}.get(kernel_vol, 300)
    target_Nt = max(1, min(12, round(target_KtNt / max(K_t, 1))))
    cout_blk = max(32, min(padded_cout, target_Nt * 32))
    while padded_cout % cout_blk != 0 and cout_blk > 32:
        cout_blk -= 32

    # conv_out (C_out=3): padded_cout=32, always Cout_blk=32
    if padded_cout == 32:
        cout_blk = 32

    # Also offer cin_blk/2 as an alternative (more C_in blocks = more parallelism)
    alt_cin = max(32, cin_blk // 2)
    while padded_cin % alt_cin != 0 and alt_cin > 32:
        alt_cin -= 32

    candidates = [(cin_blk, cout_blk)]
    if alt_cin != cin_blk:
        candidates.append((alt_cin, cout_blk))
    return candidates


def _predict_spatial_candidates(H_out, W_out, C_out):
    """Return 2-3 (H_out_block, W_out_block) candidates ranked by predicted performance.

    Derived from empirical winners across all entries in the blocking table:

    Narrow shapes (H/W > 2):  tall tensor → (8,4) mirrors the ratio
    Large squares (H*W>10000): H=4 mandatory to avoid silent L1 OOM,
                                especially for conv_out (C_out ≤ 32)
    Medium squares (H*W>1000): (8,4) dominates
    Small shapes (H*W < 400):  larger blocks reduce total block count

    The probe+cutoff in the sweep loop means the 2nd/3rd candidate
    costs only 1 extra JIT if the 1st is already clearly best.
    """
    vol = H_out * W_out
    hw_ratio = H_out / max(W_out, 1)
    conv_out_layer = C_out <= 32  # 96→3 pattern: H=4 always safe

    if vol < 400:
        # Small shape: large H_blk to minimise total block count and maximise
        # parallelism across the spatial dimension that is larger.
        if hw_ratio > 2:
            return [(min(16, H_out), min(2, W_out)), (min(8, H_out), min(4, W_out))]
        return [(min(16, H_out), min(4, W_out)), (min(8, H_out), min(8, W_out))]

    if hw_ratio > 2:
        # Tall narrow tensor (e.g. 184×40, 92×20): H=8, W=4 mirrors the ratio.
        return [(8, 4), (4, 8)]

    if conv_out_layer and vol > 5000:
        # conv_out (C_out=3): H=4 mandatory for large shapes.
        # H=8 triggers silent L1 OOM at 100ms+ for large-T conv_out.
        return [(4, 8), (4, 4)]

    if vol > 40000:
        # Very large square (e.g. 240×208 vol=49920, 360×320 vol=115200):
        # H=4 wins at 49920 (T=7,H=4,W=8=21ms), (8,8) wins at 115200.
        return [(4, 8), (8, 8), (8, 4)]

    if vol > 5000:
        # Large/medium square (e.g. 184×160 vol=29440, 120×104, 92×80, 46×40):
        # (8,4) dominates across all measured medium-large square shapes.
        return [(8, 4), (4, 8)]

    # Small-medium (e.g. 46×10, 30×26):
    return [(8, 4), (4, 8)]


def build_all_blockings(C_in, C_out, kernel_size, H, W, T, num_cores=120, fast=False):
    """Generate candidate (Cin, Cout, T, H, W) blocking combinations.

    fast=True: 2-phase search (~6-10 combos, minimises JIT compilations).
        Phase 1 — find best spatial with T=1: only 3 spatial candidates.
        Phase 2 — fix best spatial, sweep clean T divisors of T_in.
        Total: ~6-10 unique JIT compilations regardless of shape size.

        Spatial set is derived from empirical wins:
          (H=4, W=8), (H=8, W=4), (H=8, W=8)
        H=4 is critical for conv_out (H=8 silently OOMs at large shapes).

    fast=False: exhaustive search (~100-500 combos, hours for large shapes).
    """
    padded_cin = aligned_channels(C_in)
    kT = kernel_size[0]
    T_out = T - (kT - 1) if kT > 1 else T
    padded_cout = aligned_channels(C_out)

    if fast:
        # Cin/Cout: pick the largest (cin, cout) pair that fits L1 for at least one
        # of the fast spatial candidates. Weight CB = K_t × N_t × TILE_SIZE grows with
        # cin×cout, so large layers (384→384 k333) need reduced cin or cout.
        all_cins = sorted(
            [c for c in range(32, padded_cin + 1, 32) if valid_cin(c, padded_cin, kernel_size)], reverse=True
        )
        all_couts = sorted(
            [c for c in range(32, padded_cout + 1, 32) if valid_cout(c, C_out) and valid_matmul_subblock(c)],
            reverse=True,
        )

        # T: clean divisors of T_in (not T_out — T_out can be prime).
        # E.g. T_in=63 → T_out=61 (prime), but 63=7×9 gives T=7,9.
        t_blocks_set = {1}
        if kT > 1 and T_out > 1:
            for t in range(2, min(T + 1, 22)):
                if T % t == 0:
                    t_blocks_set.add(t)
            for t in [3, 6, 7, 9]:  # common useful values even if not exact divisors
                if 1 < t <= min(T_out, 21):
                    t_blocks_set.add(t)
        t_blocks = sorted(t_blocks_set)

        # Spatial: 2-3 candidates predicted from shape heuristics derived from the
        # blocking table. Avoids testing ~25 spatial pairs blindly.
        hw_fast = _predict_spatial_candidates(H, W, C_out)

        # Cin/Cout: use physics-derived heuristic (K_t × N_t ≈ 300 → weight CB ≈ 600 KB).
        # Predicts the optimal (cin, cout) directly instead of searching all combinations.
        cin_cout_pairs = [
            (cin, cout)
            for cin, cout in _predict_cin_cout(C_in, C_out, kernel_size)
            if valid_cin(cin, padded_cin, kernel_size) and valid_cout(cout, C_out) and valid_matmul_subblock(cout)
        ]
        # Fallback: max cin×cout search if heuristic gives nothing L1-feasible
        if not any(
            estimate_l1_bytes(cin, cout, 1, hw_fast[0][0], hw_fast[0][1], kernel_size, C_in) <= L1_BUDGET
            for cin, cout in cin_cout_pairs
        ):
            best_score, cin_cout_pairs = -1, []
            for cin in sorted(all_cins, reverse=True):
                for cout in sorted(all_couts, reverse=True):
                    score = cin * cout
                    if score <= best_score:
                        continue
                    if estimate_l1_bytes(cin, cout, 1, hw_fast[0][0], hw_fast[0][1], kernel_size, C_in) <= L1_BUDGET:
                        best_score, cin_cout_pairs = score, [(cin, cout)]

        combos = sorted(
            [
                (cin, cout, t, h, w)
                for cin, cout in cin_cout_pairs
                for t in t_blocks
                for h, w in hw_fast
                if estimate_l1_bytes(cin, cout, t, h, w, kernel_size, C_in) <= L1_BUDGET
            ],
            key=lambda c: (0 if c[2] == 1 else 1, c[2], c[3], c[4]),
        )
        return combos
    else:
        cins = [c for c in range(32, padded_cin + 1, 32) if valid_cin(c, padded_cin, kernel_size)]
        couts = [c for c in range(32, padded_cout + 1, 32) if valid_cout(c, C_out)]

        t_blocks_set = {1}
        if kT > 1 and T_out > 0:
            for t in range(2, min(T_out + 1, 32)):
                if T_out % t == 0:
                    t_blocks_set.add(t)
            for t in [3, 5, 6, 7, 9, 11, 13, 15, 21]:
                if 1 < t <= T_out:
                    t_blocks_set.add(t)
            t_blocks_set = {t for t in t_blocks_set if t <= min(T_out, 21)}
        t_blocks = sorted(t_blocks_set)

        tensor_vol = H * W
        if tensor_vol > 2000:
            hw = [(h, w) for h in [2, 4, 8, 16, 32] for w in [2, 4, 8, 16, 32] if h * w <= 256 and h <= H and w <= W]
        else:
            hw = [
                (h, w) for h in [1, 2, 4, 8, 16, 32] for w in [1, 2, 4, 8, 16, 32] if h * w <= 256 and h <= H and w <= W
            ]

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

    def combo_priority(c):
        cin, cout, t, h, w = c
        t_first = 0 if t == 1 else 1
        cin_score = -cin
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
        "--fast",
        action="store_true",
        help="Fast mode: focused candidates (~30-60 combos, 5-15 min) vs exhaustive (~300+, hours).",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Per-combo hang timeout in seconds (default: 60)")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Previous sweep JSON. Combos > 2× that best_us are skipped immediately.",
    )
    args = parser.parse_args()

    TIMEOUT_S = args.timeout

    kernel_size = tuple(int(x) for x in args.kernel.split(","))
    C_in, C_out, T, H, W = args.C_in, args.C_out, args.T, args.H, args.W
    padded_cin = aligned_channels(C_in)

    if args.grid_size:
        gx, gy = args.grid_size.split("x")
        _num_cores = int(gx) * int(gy)
    else:
        _num_cores = 120

    combos = build_all_blockings(C_in, C_out, kernel_size, H, W, T, num_cores=_num_cores, fast=args.fast)
    print(f"Shape: C_in={C_in} C_out={C_out} kernel={kernel_size} T={T} H={H} W={W}")
    print(f"Grid: {args.grid_size or 'auto'}")
    print(f"Mesh: {args.mesh or 'single device'}")
    print(f"Total valid combos: {len(combos)}")

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
            print(f"Baseline: best={best_us:.0f}us from {bl_path.name}, {len(t1_baseline)} T=1 entries pre-loaded")
        else:
            print(f"Warning: baseline file {bl_path} not found, ignoring --baseline")

    weight_cache = {}

    def get_weight(cin):
        if cin not in weight_cache:
            tt_w = ttnn.from_torch(w, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0)
            weight_cache[cin] = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=tt_w, C_in_block=cin, device=device
            )
        return weight_cache[cin]

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
            probe_threshold = 2.0 if i < len(combos) * 0.20 else 1.4

            spatial_key = (cin, cout, h_blk, w_blk)
            if t_blk > 1 and best_us < float("inf"):
                t1_us = t1_baseline.get(spatial_key)
                if t1_us is not None and t1_us > best_us * 1.5:
                    fail_count += 1
                    results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": t1_us, "status": "skip_t1_bad"})
                    continue

            # Run the entire combo (probe + warmup + timed) inside the timeout
            # thread so that slow JIT compilation is also caught by TIMEOUT_S.
            probe_args = (device, tt_input, tt_weight, tt_bias, cfg, C_out, kernel_size, ckc)
            us_out = [None]
            probe_out = [None]
            exc_out = [None]

            def _probe_warmup_and_time():
                try:
                    # Probe: triggers JIT compile + 1 measurement
                    probe_out[0] = _run_once(*probe_args)

                    # Cutoff after probe (inside thread so JIT time is counted)
                    if best_us < float("inf"):
                        if probe_out[0] > best_us * probe_threshold:
                            return  # leave us_out[0] = None → treated as slow
                        if probe_out[0] > best_us * (probe_threshold * 0.75):
                            probe2 = _run_once(*probe_args)
                            if min(probe_out[0], probe2) > best_us * (probe_threshold * 0.9):
                                return  # slow

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

                    us_out[0] = statistics.mean(times)
                except Exception as e:
                    exc_out[0] = e

            thread = threading.Thread(target=_probe_warmup_and_time, daemon=True)
            thread.start()
            thread.join(TIMEOUT_S)

            if thread.is_alive():
                print(
                    f"HANG/SLOW-JIT: ({cin},{cout},{t_blk},{h_blk},{w_blk}) >{TIMEOUT_S}s — exiting. "
                    f"Run: tt-smi -r 0,1,2,3,4,5,6,7",
                    flush=True,
                )
                os._exit(1)

            if exc_out[0] is not None:
                raise exc_out[0]

            # Probe was slow (cutoff fired inside thread) — record as slow and skip
            if us_out[0] is None and probe_out[0] is not None:
                fail_count += 1
                results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": probe_out[0], "status": "slow"})
                continue

            us = us_out[0]
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
                    f"  [{i+1:3d}/{len(combos)}] ({cin:3d},{cout:3d},{t_blk},{h_blk:2d},{w_blk:2d}) "
                    f"{us:8.0f}us{tag}",
                    flush=True,
                )
        except Exception:
            fail_count += 1
            results.append({"blocking": [cin, cout, t_blk, h_blk, w_blk], "us": None, "status": "fail"})

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min). {ok_count} ok, {fail_count} failed.")

    ok_results = sorted([r for r in results if r["status"] == "ok"], key=lambda r: r["us"])
    print(f"\nTop 10:")
    for r in ok_results[:10]:
        b = r["blocking"]
        print(f"  ({b[0]:3d},{b[1]:3d},{b[2]},{b[3]:2d},{b[4]:2d}) = {r['us']:.0f} us")

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
