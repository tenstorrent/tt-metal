# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

# MatmulDeviceOperation dimensions: M x K x N
M = 32
K = 1536
N = 17920

# Shape in tiles:  Mt = 1,  Kt = 48,  Nt = 560
#
# 16-core DRAM-sharded variant of test_matmul_32x1536x17920_dram_sharded_bfp.py.
# Same layout strategy; the only delta is the compute grid: (8, 1) → (8, 2),
# doubling compute parallelism from 8 → 16 cores.
#
# Why 16 cores is the upper bound for DRAM-sharded on this shape:
#   compute_cores must simultaneously divide Kt=48 and Nt=560 (for clean
#   width-sharding of in0 along K and clean per_core_N along N).
#   Common divisors of 48 and 560 that fit in the WH 8×8 worker grid:
#     1, 2, 4, 8, 16.
#   16 is the max.  24 cores would give Kt/24=2 (clean) but Nt/24=23.3 (broken).
#
# Per-core work vs the 8-core variant:
#   8 cores : per_core_N=70, Kt_per_core=6 → 70 × 6 = 420 tile-MAC inner blocks
#  16 cores : per_core_N=35, Kt_per_core=3 → 35 × 3 = 105 tile-MAC inner blocks
#  → 4× less work per core; expect roughly 2× speed-up if compute-bound.
NUM_DRAM_BANKS = 12  # Wormhole_b0
TILE = 32
N_PADDED = math.ceil(N / (TILE * NUM_DRAM_BANKS)) * (TILE * NUM_DRAM_BANKS)  # 18048
assert N_PADDED == 18048, f"unexpected N_PADDED={N_PADDED}"

COMPUTE_GRID_X = 8  # cols (must be ≤ 8 on WH worker grid)
COMPUTE_GRID_Y = 2  # rows
NUM_COMPUTE_CORES = COMPUTE_GRID_X * COMPUTE_GRID_Y  # 16
PER_CORE_M = M // TILE  # 1
PER_CORE_N = (N // TILE) // NUM_COMPUTE_CORES  # 35
# Kt per compute core = K / 16 / 32 = 3.  in0_block_w must divide 3 → {1, 3}.
# Pick 1 (3 inner iters of 1 K-tile each — smallest L1 footprint per iter).
IN0_BLOCK_W = 1


@pytest.mark.parametrize(
    "m_size, k_size, n_size",
    [(M, K, N)],
    ids=[f"MatmulDeviceOperation_{M}x{K}x{N}"],
)
def test_matmul_device_operation_32x1536x17920_dram_sharded_bfp_16c(device, m_size, k_size, n_size):
    """16-core DRAM-sharded matmul-1D PCC test for M=32, K=1536, N=17920 on
    Wormhole_b0 with bfloat8_b activations and bfloat4_b weights.

    Scales the 8-core variant up to the max valid compute parallelism for this
    shape under the DRAM-sharded program config.
    """
    torch.manual_seed(0)

    torch_input_a = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((1, 1, k_size, n_size), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    # in0: L1 width-sharded across a 8×2 compute grid.  Per-core shard = [32, 96].
    in0_mem_cfg = ttnn.create_sharded_memory_config(
        (1, 1, m_size, k_size),
        core_grid=ttnn.CoreGrid(y=COMPUTE_GRID_Y, x=COMPUTE_GRID_X),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # in1: DRAM width-sharded across all 12 banks.  Per-bank shard = [1536, 1504].
    dram_grid_size = device.dram_grid_size()
    in1_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
    )
    in1_mem_cfg = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(
            in1_shard_grid,
            [k_size, N_PADDED // NUM_DRAM_BANKS],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Output: L1 width-sharded; shard spec computed by the op from per_core_N.
    out_mem_cfg = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=IN0_BLOCK_W,
        per_core_M=PER_CORE_M,
        per_core_N=PER_CORE_N,
        fused_activation=None,
    )

    # LoFi + math_approx is the standard pairing for bfloat4_b weights
    # (matches the bfp4 DRAM-sharded paths in test_matmul_deepseek.py).
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    input_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mem_cfg,
    )
    input_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_mem_cfg,
    )

    output = ttnn.matmul(
        input_a,
        input_b,
        program_config=program_config,
        memory_config=out_mem_cfg,
        compute_kernel_config=compute_kernel_config,
    )
    output = ttnn.to_torch(output)

    assert output.shape == torch_output.shape
    # bfloat4_b weights at K=1536 → PCC settles around 0.99 (not 0.999).
    # Tolerances mirror the bfp4 branch in test_matmul_deepseek.py.
    assert_numeric_metrics(
        torch_output,
        output,
        atol=0.0347 * k_size,
        rtol=24.625 * k_size,
        frobenius_threshold=0.0005 * k_size,
        pcc_threshold=0.99,
        check_ulp=False,
        check_allclose=False,
    )
