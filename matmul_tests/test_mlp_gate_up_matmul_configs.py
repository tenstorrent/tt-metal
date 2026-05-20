# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Perf comparison sweep for MLP gate/up matmul: M=32, K=1536, N=17920
in0=BF16, in1=BFP4, out=BFP8, LoFi math. Wormhole B0 only.

Roofline: ~52 us at 300 GB/s (weight tensor ≈ 15.5 MB at BFP4).
Target: < 70 us.

Configs:
  DRAM-sharded (in0 L1 WIDTH_SHARDED, in1 DRAM WIDTH_SHARDED, out L1 WIDTH_SHARDED)
    baseline_6banks_16cores       — current production config (6 of 12 DRAM banks)
    dram_sharded_12banks_16cores  — all 12 DRAM banks, 16 compute cores, in0_block_w=3
    dram_sharded_12banks_8cores   — all 12 DRAM banks, 8 compute cores, in0_block_w=6
                                    (k/core=6 allows larger K block; 8↔12 fan-in handled by kernel)

  mcast1d (in0 interleaved L1, in1 interleaved DRAM, out interleaved L1)
    mcast1d_8x7_pcn10_osw5       — 56 cores, 10 N-tiles/core, out_subblock_w=5
    mcast1d_8x5_pcn14_osw7       — 40 cores, 14 N-tiles/core, out_subblock_w=7
    mcast1d_8x2_pcn35_osw7       — 16 cores, 35 N-tiles/core, out_subblock_w=7
                                    (baseline mcast1d code gives osw=1 for pcn=35 due to cap=4 bug)
    mcast1d_8x1_pcn70_osw7       — 8 cores,  70 N-tiles/core, out_subblock_w=7, in0_block_w=3

Run:
    pytest matmul_tests/test_mlp_gate_up_matmul_configs.py -s -v
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


M, K, N = 32, 1536, 17920
TILE = 32
K_TILES = K // TILE  # 48
N_TILES = N // TILE  # 560

NUM_WARMUP = 1
NUM_ITERS = 1
TARGET_US = 70.0
ROOFLINE_US = 52.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


def _cfg_dram_sharded(
    device,
    num_compute_cores: int,
    in0_block_w: int,
    num_banks_override: Optional[int] = None,
):
    """
    in0: L1 WIDTH_SHARDED across num_compute_cores
    in1: DRAM WIDTH_SHARDED across num_dram_banks
    out: L1 WIDTH_SHARDED (spec filled by the op)

    Kernel handles non-equal core/bank counts internally (e.g. 8 cores : 12 banks).
    """
    dram_grid = device.dram_grid_size()
    avail_banks = int(dram_grid.x) * int(dram_grid.y)
    num_banks = num_banks_override if num_banks_override is not None else avail_banks

    if num_banks > avail_banks:
        pytest.skip(f"requested {num_banks} DRAM banks, device has {avail_banks}")

    n_padded = math.ceil(N / (TILE * num_banks)) * TILE * num_banks
    per_core_n = N_TILES // num_compute_cores

    # Prefer wide grids (more DRAM bank coverage per row).
    # Valid for num_compute_cores in {8, 16} on the 8-wide WH worker grid.
    grid_x = 8
    grid_y = num_compute_cores // grid_x
    compute_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)

    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, M, K),
        core_grid=compute_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    if num_banks < avail_banks:
        # Partial DRAM bank set: linear range along x.
        in1_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    else:
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


def _cfg_mcast1d(
    grid_x: int,
    grid_y: int,
    per_core_n: int,
    out_subblock_w: int,
    in0_block_w: int,
):
    """
    in0: interleaved L1 — activation broadcast to all cores
    in1: interleaved DRAM — each core independently fetches its N-slice
    out: interleaved L1

    Requires N_TILES % (grid_x * grid_y) == 0 for clean tiling.
    L1 budget per core for in1 double-buffer:
        in0_block_w × per_core_n × ~576 bytes/tile (BFP4)
    """
    num_cores = grid_x * grid_y
    assert N_TILES % num_cores == 0, f"N_TILES={N_TILES} not divisible by {grid_x}×{grid_y}={num_cores} cores"
    assert (
        per_core_n == N_TILES // num_cores
    ), f"per_core_n={per_core_n} does not match N_TILES/num_cores={N_TILES // num_cores}"

    # L1 budget check (conservative: use BF16 tile size = 1024 B as the kernel does internally)
    l1_for_in1 = in0_block_w * per_core_n * 1024
    assert l1_for_in1 <= 256 * 1024, (
        f"in1 L1 buffer {l1_for_in1 // 1024} KB exceeds 256 KB " f"(in0_block_w={in0_block_w}, per_core_n={per_core_n})"
    )

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
#
# N_TILES=560 divisors that fit an 8-wide grid (for mcast1d):
#   8×7=56 cores → pcn=10, osw=5
#   8×5=40 cores → pcn=14, osw=7
#   8×2=16 cores → pcn=35, osw=7  (note: existing code gives osw=1 for pcn=35 at cap=4)
#   8×1= 8 cores → pcn=70, osw=7
#
# DRAM-sharded valid core counts (need gcd(K_TILES=48, N_TILES=560)=16):
#   valid: {4, 8, 16}; use 8 and 16.
#
CONFIGS: list[tuple[str, Callable]] = [
    # ── DRAM sharded ─────────────────────────────────────────────────────────
    (
        "baseline_6banks_16cores",
        lambda dev: _cfg_dram_sharded(dev, 16, 3, num_banks_override=6),
    ),
    (
        "dram_sharded_12banks_16cores",
        lambda dev: _cfg_dram_sharded(dev, 16, 3),
    ),
    (
        "dram_sharded_12banks_8cores_ibw6",
        lambda dev: _cfg_dram_sharded(dev, 8, 6),
    ),
    (
        "dram_sharded_12banks_8cores_ibw3",
        lambda dev: _cfg_dram_sharded(dev, 8, 3),
    ),
    (
        "mixed_in0_l1_in1_dram6_outl1_16cores",
        lambda dev: _cfg_dram_sharded(dev, 16, 3, num_banks_override=6),
    ),
    (
        "mixed_in0_l1_in1_dram12_outl1_16cores",
        lambda dev: _cfg_dram_sharded(dev, 16, 3),
    ),
    (
        "mixed_in0_l1_in1_dram12_outl1_8cores_ibw6",
        lambda dev: _cfg_dram_sharded(dev, 8, 6),
    ),
    (
        "mixed_in0_l1_in1_dram12_outl1_8cores_ibw3",
        lambda dev: _cfg_dram_sharded(dev, 8, 3),
    ),
    # ── mcast1d ──────────────────────────────────────────────────────────────
    (
        "mcast1d_8x7_pcn10_osw5",
        lambda dev: _cfg_mcast1d(8, 7, 10, 5, 4),
    ),
    (
        "mcast1d_8x5_pcn14_osw7",
        lambda dev: _cfg_mcast1d(8, 5, 14, 7, 4),
    ),
    (
        "mcast1d_8x2_pcn35_osw7",
        lambda dev: _cfg_mcast1d(8, 2, 35, 7, 4),
    ),
    (
        "mcast1d_8x1_pcn70_osw7",
        lambda dev: _cfg_mcast1d(8, 1, 70, 7, 3),
    ),
]

CONFIG_IDS = [name for name, _ in CONFIGS]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name,cfg_builder", CONFIGS, ids=CONFIG_IDS)
def test_mlp_gate_up_matmul_configs(device, config_name: str, cfg_builder: Callable):
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

    # warmup (compile + 1 extra dispatch)
    for _ in range(NUM_WARMUP):
        out = _run()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    # timed loop: keep last output alive for PCC check
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

    vs_roofline = elapsed_us / ROOFLINE_US
    status = "PASS" if elapsed_us < TARGET_US else "SLOW"
    print(
        f"\n  [{status}] {config_name:<40}"
        f"  avg = {elapsed_us:6.1f} us"
        f"  ({vs_roofline:.2f}× roofline)"
        f"  target < {TARGET_US:.0f} us"
    )

    assert result.shape == torch_ref.shape
    assert_with_pcc(torch_ref, result, pcc=0.99)
