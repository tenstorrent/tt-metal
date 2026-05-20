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
# Optimization summary (vs. the un-tuned baseline that picks default heuristics):
#   1. bfloat8_b activations + bfloat4_b weights
#      → in1 (the dominant tensor) drops from ~55 MB to ~14 MB,
#        directly cutting DRAM read cost ~4× for a memory-bound shape.
#   2. DRAM-sharded matmul-1D across all 12 Wormhole_b0 DRAM banks
#      → reads stream from every bank in parallel instead of contending
#        on interleaved DRAM addresses.
#   3. L1-sharded in0 + L1-sharded output
#      → no DRAM round-trip for activations or for whatever consumes the result.
#
# Layout details (mirrors the canonical WH DRAM-sharded pattern in test_matmul.py):
#   - in1 (weights) is width-sharded across `device.dram_grid_size()` = 12 banks.
#     N must be a multiple of (32 * 12) = 384 for the DRAM shards to be tile-
#     aligned; Nt=560 is not, so N is padded to 18048 (= 47 * 384).  Padding
#     columns live on the last bank and are masked off by the op.
#   - in0 (activations) is width-sharded in L1 across a (8, 1) compute grid
#     (8 cores in one TENSIX row).  Note the compute grid is INTENTIONALLY
#     smaller than the DRAM grid: 12 worker cores in a row would exceed the
#     8-wide WH worker grid.  The op bridges the 8↔12 fan-in internally.
#   - per_core_N is computed from the *un-padded* N (17920/8/32 = 70 tiles)
#     so each compute core owns the same number of output tiles.
NUM_DRAM_BANKS = 12  # Wormhole_b0
TILE = 32
N_PADDED = math.ceil(N / (TILE * NUM_DRAM_BANKS)) * (TILE * NUM_DRAM_BANKS)  # 18048
assert N_PADDED == 18048, f"unexpected N_PADDED={N_PADDED}"

COMPUTE_GRID_X = 8  # cols (must be ≤ 8 on WH worker grid)
COMPUTE_GRID_Y = 1  # rows
NUM_COMPUTE_CORES = COMPUTE_GRID_X * COMPUTE_GRID_Y  # 8
PER_CORE_M = M // TILE  # 1
PER_CORE_N = (N // TILE) // NUM_COMPUTE_CORES  # 70
# Kt per compute core = K / 8 / 32 = 6.  in0_block_w must divide 6 → {1, 2, 3, 6}.
IN0_BLOCK_W = 2


@pytest.mark.parametrize(
    "m_size, k_size, n_size",
    [(M, K, N)],
    ids=[f"MatmulDeviceOperation_{M}x{K}x{N}"],
)
def test_matmul_device_operation_32x1536x17920_dram_sharded_bfp(device, m_size, k_size, n_size):
    """DRAM-sharded matmul-1D PCC test for M=32, K=1536, N=17920 on Wormhole_b0
    with bfloat8_b activations and bfloat4_b weights.

    Combines the layout win from `test_matmul_32x1536x17920_shard.py` with the
    dtype win from the un-tuned baseline (which used bfp8/bfp4 but no explicit
    program config).  Expected to be the lowest-latency path for this shape.
    """
    torch.manual_seed(0)

    torch_input_a = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((1, 1, k_size, n_size), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    # in0: L1 width-sharded across an 8×1 compute row.  Per-core shard = [32, 192].
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
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
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
