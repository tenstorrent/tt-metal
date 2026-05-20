# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Perf comparison sweep for attention output projection: M=32, K=1536, N=1536
in0=BF16, in1=BFP4, out=BFP8, LoFi math. Wormhole B0 only.

Weight (BFP4): ~1.33 MB → roofline ~4.4 us at 300 GB/s.
At this tiny N, configs are likely overhead/dispatch-bound rather than bandwidth-bound.

Configs:
  DRAM-sharded (in0 L1 WIDTH_SHARDED, in1 DRAM WIDTH_SHARDED, out L1 WIDTH_SHARDED)
    dram_sharded_12banks_48cores  — 48 cores, grid 8x6, pcn=1,  k/core=1, ibw=1
    dram_sharded_12banks_24cores  — 24 cores, grid 8x3, pcn=2,  k/core=2, ibw=2
    dram_sharded_12banks_16cores  — 16 cores, grid 8x2, pcn=3,  k/core=3, ibw=3
    dram_sharded_12banks_8cores   — 8 cores,  grid 8x1, pcn=6,  k/core=6, ibw=6

  mcast1d (in0 interleaved L1, in1 interleaved DRAM, out interleaved L1)
    mcast1d_8x1_pcn6_osw6        — 8  cores, pcn=6, osw=6, ibw=8
    mcast1d_8x2_pcn3_osw3        — 16 cores, pcn=3, osw=3, ibw=8
    mcast1d_8x3_pcn2_osw2        — 24 cores, pcn=2, osw=2, ibw=8
    mcast1d_8x6_pcn1_osw1        — 48 cores, pcn=1, osw=1, ibw=8

Increase NUM_ITERS (e.g. to 100) for stable wall-clock measurements.

Run:
    pytest matmul_tests/test_attn_o_proj_matmul_configs.py -s -v
"""

from __future__ import annotations

import math
import time
from typing import Callable

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


M, K, N = 32, 1536, 1536
TILE = 32
K_TILES = K // TILE  # 48
N_TILES = N // TILE  # 48

NUM_WARMUP = 1
NUM_ITERS = 1
TARGET_US = 70.0
ROOFLINE_US = 4.4  # ~1.33 MB BFP4 weight at 300 GB/s


def _compute_kernel():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for d in range(min(value, limit), 0, -1):
        if value % d == 0:
            return d
    return 1


# ---------------------------------------------------------------------------
# DRAM-sharded config builder
# ---------------------------------------------------------------------------


def _cfg_dram_sharded(device, num_compute_cores: int, in0_block_w: int):
    dram_grid = device.dram_grid_size()
    num_banks = int(dram_grid.x) * int(dram_grid.y)

    # N=1536 is exactly divisible by 12 banks × 32: n_padded = 1536 (no padding)
    n_padded = math.ceil(N / (TILE * num_banks)) * TILE * num_banks
    per_core_n = N_TILES // num_compute_cores

    grid_x = 8
    grid_y = num_compute_cores // grid_x
    compute_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)

    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, M, K),
        core_grid=compute_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in1_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(int(dram_grid.x) - 1, int(dram_grid.y) - 1),
            )
        }
    )
    in1_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(in1_grid, [K, n_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    prog = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=M // TILE,
        per_core_N=per_core_n,
        fused_activation=None,
    )
    return in0_mem, in1_mem, out_mem, prog


# ---------------------------------------------------------------------------
# mcast1d config builder
# ---------------------------------------------------------------------------


def _cfg_mcast1d(grid_x: int, grid_y: int, per_core_n: int, out_subblock_w: int, in0_block_w: int):
    num_cores = grid_x * grid_y
    assert N_TILES % num_cores == 0, f"N_TILES={N_TILES} not divisible by {num_cores}"
    assert per_core_n == N_TILES // num_cores
    assert in0_block_w * per_core_n * 1024 <= 256 * 1024, "in1 L1 buffer exceeds 256 KB"

    prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    return ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, prog


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------
# N_TILES=K_TILES=48; gcd=48 → DRAM-sharded valid 8-wide cores: {8, 16, 24, 48}
# mcast1d 8-wide grids: 8x1=8, 8x2=16, 8x3=24, 8x6=48
#
CONFIGS: list[tuple[str, Callable]] = [
    # ── DRAM sharded (largest→smallest pcn, i.e. most→least efficient per-core) ──
    (
        "dram_sharded_12banks_8cores",
        lambda dev: _cfg_dram_sharded(dev, 8, 6),  # pcn=6, k/core=6, ibw=6
    ),
    (
        "dram_sharded_12banks_16cores",
        lambda dev: _cfg_dram_sharded(dev, 16, 3),  # pcn=3, k/core=3, ibw=3
    ),
    (
        "dram_sharded_12banks_24cores",
        lambda dev: _cfg_dram_sharded(dev, 24, 2),  # pcn=2, k/core=2, ibw=2
    ),
    (
        "dram_sharded_12banks_48cores",
        lambda dev: _cfg_dram_sharded(dev, 48, 1),  # pcn=1, k/core=1, ibw=1 (thin)
    ),
    (
        "mixed_in0_l1_in1_dram_outl1_8cores",
        lambda dev: _cfg_dram_sharded(dev, 8, 6),
    ),
    (
        "mixed_in0_l1_in1_dram_outl1_16cores",
        lambda dev: _cfg_dram_sharded(dev, 16, 3),
    ),
    (
        "mixed_in0_l1_in1_dram_outl1_24cores",
        lambda dev: _cfg_dram_sharded(dev, 24, 2),
    ),
    (
        "mixed_in0_l1_in1_dram_outl1_48cores",
        lambda dev: _cfg_dram_sharded(dev, 48, 1),
    ),
    # ── mcast1d ──────────────────────────────────────────────────────────────
    (
        "mcast1d_8x1_pcn6_osw6",
        lambda dev: _cfg_mcast1d(8, 1, 6, 6, 8),  # L1=48KB
    ),
    (
        "mcast1d_8x2_pcn3_osw3",
        lambda dev: _cfg_mcast1d(8, 2, 3, 3, 8),  # L1=24KB
    ),
    (
        "mcast1d_8x3_pcn2_osw2",
        lambda dev: _cfg_mcast1d(8, 3, 2, 2, 8),  # L1=16KB
    ),
    (
        "mcast1d_8x6_pcn1_osw1",
        lambda dev: _cfg_mcast1d(8, 6, 1, 1, 8),  # L1=8KB, no output reuse
    ),
]

CONFIG_IDS = [name for name, _ in CONFIGS]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_attn_o_proj_matmul_configs(device, config_name: str, cfg_builder: Callable):
    torch.manual_seed(0)

    torch_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16) * 0.1
    torch_b = torch.randn((1, 1, K, N), dtype=torch.bfloat16) * 0.1
    torch_ref = torch.matmul(torch_a, torch_b)

    in0_mem, in1_mem, out_mem, prog = cfg_builder(device)
    compute_cfg = _compute_kernel()

    input_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in0_mem,
        device=device,
    )
    input_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in1_mem,
        device=device,
    )

    def _run():
        return ttnn.matmul(
            input_a,
            input_b,
            program_config=prog,
            memory_config=out_mem,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=compute_cfg,
        )

    for _ in range(NUM_WARMUP):
        out = _run()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    last_out = None
    start = time.perf_counter()
    for _ in range(NUM_ITERS):
        if last_out is not None:
            ttnn.deallocate(last_out)
        last_out = _run()
    ttnn.synchronize_device(device)
    elapsed_us = (time.perf_counter() - start) * 1e6 / NUM_ITERS

    result = ttnn.to_torch(last_out)
    ttnn.deallocate(last_out)
    ttnn.deallocate(input_a)
    ttnn.deallocate(input_b)

    status = "PASS" if elapsed_us < TARGET_US else "SLOW"
    print(
        f"\n  [{status}] {config_name:<40}"
        f"  avg = {elapsed_us:6.1f} us"
        f"  ({elapsed_us / ROOFLINE_US:.2f}× roofline ~{ROOFLINE_US:.0f} us)"
    )

    assert result.shape == torch_ref.shape
    assert_with_pcc(torch_ref, result, pcc=0.99)
