# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep 1D mcast_in0 matmul configs for small-M shapes on WH Galaxy.

Usage:
    pytest models/tt_dit/utils/sweep_1d_matmul.py -x -s --timeout=600

Or run a single shape:
    pytest models/tt_dit/utils/sweep_1d_matmul.py -k "32_6144_6144" -x -s
"""

import csv
import itertools
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

CSV_FILE = os.environ.get("SWEEP_1D_CSV", "sweep_results_1d.csv")
CSV_COLUMNS = [
    "device_config",
    "M",
    "K",
    "N",
    "grid_x",
    "grid_y",
    "num_cores",
    "in0_block_w",
    "per_core_N",
    "out_subblock_w",
    "device_kernel_duration_ns",
    "status",
]

SHAPES = [
    (32, 6144, 6144),  # 71.7 ms total — biggest
    (32, 6144, 4608),  # 56.4 ms total
    (32, 6144, 2304),  # 13.6 ms total
    (32, 6144, 1536),  # 10.4 ms total
    (32, 256, 6144),  # 5.2 ms total
]

SHAPE_IDS = [f"{M}_{K}_{N}" for M, K, N in SHAPES]

WARMUP_ITERS = 2
MEASURE_ITERS = 8


def write_csv_header():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def append_csv_row(row):
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def generate_combos(M, K, N, num_cores):
    M_tiles = M // 32
    K_tiles = K // 32
    N_tiles = N // 32

    in0_block_w_candidates = [w for w in range(1, K_tiles + 1) if K_tiles % w == 0]
    # Cap at reasonable values
    in0_block_w_candidates = [w for w in in0_block_w_candidates if w <= 32]

    per_core_N_candidates = []
    base = max(1, math.ceil(N_tiles / num_cores))
    for n in range(base, N_tiles + 1):
        if N_tiles % n == 0 or n == base:
            per_core_N_candidates.append(n)
        if len(per_core_N_candidates) >= 12:
            break
    # Also try some smaller values
    for n in [1, 2, 3, 4, 5, 6, 8]:
        if n not in per_core_N_candidates and N_tiles % n == 0:
            per_core_N_candidates.append(n)
    per_core_N_candidates = sorted(set(per_core_N_candidates))

    combos = []
    for in0_bw, pcN in itertools.product(in0_block_w_candidates, per_core_N_candidates):
        # N_blocks = ceil(N_tiles / per_core_N) — must not exceed num_cores
        n_blocks = math.ceil(N_tiles / pcN)
        if n_blocks > num_cores:
            continue
        for sbw in range(1, min(5, pcN + 1)):  # out_subblock_w * out_subblock_h <= 4 on WH
            if pcN % sbw == 0:
                combos.append((in0_bw, pcN, sbw))
    return combos


@pytest.fixture(scope="module")
def mesh_device():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_1d_sweep(mesh_device, shape):
    M, K, N = shape
    core_grid = mesh_device.compute_with_storage_grid_size()
    grid_x, grid_y = core_grid.x, core_grid.y
    num_cores = grid_x * grid_y

    logger.info(f"Sweeping 1D matmul ({M}, {K}, {N}) on {grid_x}x{grid_y} grid ({num_cores} cores)")

    combos = generate_combos(M, K, N, num_cores)
    logger.info(f"Generated {len(combos)} combos")

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    tt_input = ttnn.from_torch(
        torch.randn((1, 1, M, K), dtype=torch.float32),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_weight = ttnn.from_torch(
        torch.randn((K, N), dtype=torch.float32),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )

    write_csv_header()
    results = []

    for idx, (in0_bw, pcN, sbw) in enumerate(combos):
        M_tiles = M // 32
        per_core_M = max(1, M_tiles)

        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=core_grid,
            in0_block_w=in0_bw,
            out_subblock_h=1,
            out_subblock_w=sbw,
            per_core_M=per_core_M,
            per_core_N=pcN,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        try:
            # Warmup
            for _ in range(WARMUP_ITERS):
                ttnn.linear(
                    tt_input,
                    tt_weight,
                    compute_kernel_config=compute_config,
                    program_config=config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.synchronize_device(mesh_device)

            # Measure
            start = time.perf_counter_ns()
            for _ in range(MEASURE_ITERS):
                ttnn.linear(
                    tt_input,
                    tt_weight,
                    compute_kernel_config=compute_config,
                    program_config=config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.synchronize_device(mesh_device)
            elapsed_ns = (time.perf_counter_ns() - start) // MEASURE_ITERS

            row = ["wh_4x8", M, K, N, grid_x, grid_y, num_cores, in0_bw, pcN, sbw, elapsed_ns, "OK"]
            results.append((elapsed_ns, in0_bw, pcN, sbw))
            status = "OK"

        except Exception as e:
            elapsed_ns = 0
            row = ["wh_4x8", M, K, N, grid_x, grid_y, num_cores, in0_bw, pcN, sbw, 0, f"FAIL:{e}"]
            status = "FAIL"

        append_csv_row(row)

        if (idx + 1) % 50 == 0:
            logger.info(f"  Progress: {idx+1}/{len(combos)}")

    # Print best results
    if results:
        results.sort()
        logger.info(f"\n=== Best configs for ({M}, {K}, {N}) on {grid_x}x{grid_y} ===")
        for i, (ns, bw, pn, sw) in enumerate(results[:5]):
            logger.info(f"  {i+1}. in0_block_w={bw}, per_core_N={pn}, out_subblock_w={sw} -> {ns/1000:.1f} us")

    ok_count = sum(1 for r in results if r[0] > 0)
    logger.info(f"Done: {ok_count} valid, {len(combos)-ok_count} failed")
