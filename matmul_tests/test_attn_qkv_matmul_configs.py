# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Perf comparison sweep for attention QKV matmul: M=32, K=1536, N=2048
in0=BF16, in1=BFP8, out=BF16, HiFi2 math. Wormhole B0 only.

Weight (BFP8): ~3.34 MB → roofline ~11 us at 300 GB/s.
At this small N, dispatch overhead and compute efficiency matter more than bandwidth.

Configs:
  DRAM-sharded (in0 L1 WIDTH_SHARDED, in1 DRAM WIDTH_SHARDED, out L1 WIDTH_SHARDED)
    dram_sharded_12banks_16cores  — 16 cores, grid 8x2, pcn=4,  k/core=3, ibw=3
    dram_sharded_12banks_8cores   — 8 cores,  grid 8x1, pcn=8,  k/core=6, ibw=6

  mcast1d (in0 interleaved L1, in1 interleaved DRAM, out interleaved L1)
    mcast1d_8x1_pcn8_osw8        — 8  cores, pcn=8,  osw=8, ibw=8
    mcast1d_8x2_pcn4_osw4        — 16 cores, pcn=4,  osw=4, ibw=8
    mcast1d_8x4_pcn2_osw2        — 32 cores, pcn=2,  osw=2, ibw=8
    mcast1d_8x8_pcn1_osw1        — 64 cores, pcn=1,  osw=1, ibw=8

Increase NUM_ITERS (e.g. to 100) for stable wall-clock measurements.

Run:
    pytest matmul_tests/test_attn_qkv_matmul_configs.py -s -v
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


M, K, N = 32, 1536, 2048
TILE = 32
K_TILES = K // TILE  # 48
N_TILES = N // TILE  # 64

NUM_WARMUP = 1
NUM_ITERS = 1
TARGET_US = 70.0
ROOFLINE_US = 11.1  # ~3.34 MB BFP8 weight at 300 GB/s


def _compute_kernel():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
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
# N_TILES=64, valid mcast1d 8-wide grids: 8x1=8, 8x2=16, 8x4=32, 8x8=64
# DRAM-sharded: gcd(K_TILES=48, N_TILES=64)=16 → valid cores {8, 16}
#
CONFIGS: list[tuple[str, Callable]] = [
    (
        "dram_sharded_12banks_16cores",
        lambda dev: _cfg_dram_sharded(dev, 16, 3),  # pcn=4, k/core=3
    ),
    (
        "dram_sharded_12banks_8cores",
        lambda dev: _cfg_dram_sharded(dev, 8, 6),  # pcn=8, k/core=6
    ),
    (
        "mixed_in0_l1_in1_dram_outl1_16cores",
        lambda dev: _cfg_dram_sharded(dev, 16, 3),
    ),
    (
        "mixed_in0_l1_in1_dram_outl1_8cores",
        lambda dev: _cfg_dram_sharded(dev, 8, 6),
    ),
    (
        "mcast1d_8x1_pcn8_osw8",
        lambda dev: _cfg_mcast1d(8, 1, 8, 8, 8),  # L1=64KB
    ),
    (
        "mcast1d_8x2_pcn4_osw4",
        lambda dev: _cfg_mcast1d(8, 2, 4, 4, 8),  # L1=32KB
    ),
    (
        "mcast1d_8x4_pcn2_osw2",
        lambda dev: _cfg_mcast1d(8, 4, 2, 2, 8),  # L1=16KB
    ),
    (
        "mcast1d_8x8_pcn1_osw1",
        lambda dev: _cfg_mcast1d(8, 8, 1, 1, 8),  # L1=8KB, pcn=1 (no output reuse)
    ),
]

CONFIG_IDS = [name for name, _ in CONFIGS]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_attn_qkv_matmul_configs(device, config_name: str, cfg_builder: Callable):
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
        dtype=ttnn.bfloat8_b,
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
            dtype=ttnn.bfloat16,
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
