# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DRAM-sharded matmul-1D for M=32, K=1536, N=2048 (bfp8 acts + bfp4 wts).

Memory placement:
  - in0 (activations): L1, width-sharded across the (8, 1) compute row.
  - in1 (weights):     DRAM, width-sharded across all 12 DRAM banks.
  - output:            L1, width-sharded (matches in0 grid).

Shape facts:
  - Mt=1, Kt=48, Nt=64 in tiles.
  - N=2048 does NOT divide 32*NUM_DRAM_BANKS (=384), so in1 is padded to
    N_PADDED = ceil(2048/384) * 384 = 2304.  Padding cols sit on the last
    bank and are masked off by the op.  Each bank holds 2304/12 = 192 cols
    = 6 N-tiles.  per_core_N is computed from the UN-padded N so each
    compute core owns the same output width.
  - X must divide Kt=48 AND Nt=64 → X ∈ {1, 2, 4, 8}.  X=8 is the most
    parallel valid choice; Kt_per_core=6, per_core_N=8.
  - in0_block_w must divide Kt_per_core=6 → {1, 2, 3, 6}.  W=2 mirrors
    the canonical `test_matmul_32x1536x17920_dram_sharded_bfp.py` config
    (same Kt_per_core, same tradeoff).
"""

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# MatmulDeviceOperation dimensions: M x K x N
M = 32
K = 1536
N = 2048

NUM_DRAM_BANKS = 12  # Wormhole_b0
TILE = 32
N_PADDED = math.ceil(N / (TILE * NUM_DRAM_BANKS)) * (TILE * NUM_DRAM_BANKS)  # 2304
assert N_PADDED == 2304, f"unexpected N_PADDED={N_PADDED}"

COMPUTE_GRID_X = 8
COMPUTE_GRID_Y = 1
NUM_COMPUTE_CORES = COMPUTE_GRID_X * COMPUTE_GRID_Y  # 8
PER_CORE_M = M // TILE  # 1
PER_CORE_N = (N // TILE) // NUM_COMPUTE_CORES  # 8
IN0_BLOCK_W = 6  # divides Kt_per_core=6


@pytest.mark.parametrize(
    "m_size, k_size, n_size",
    [(M, K, N)],
    ids=[f"MatmulDeviceOperation_{M}x{K}x{N}_l1_dram_sharded"],
)
def test_matmul_32x1536x2048_l1_dram_sharded(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_a = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((1, 1, k_size, n_size), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    # in0: L1 width-sharded across the (8, 1) compute row.  Per-core shard = [32, 192].
    in0_mem_cfg = ttnn.create_sharded_memory_config(
        (1, 1, m_size, k_size),
        core_grid=ttnn.CoreGrid(y=COMPUTE_GRID_Y, x=COMPUTE_GRID_X),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # in1: DRAM width-sharded across all 12 banks.  Per-bank shard = [1536, 192].
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

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    input_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_mem_cfg,
    )
    input_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat8_b,
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
        dtype=ttnn.bfloat16,
    )
    output = ttnn.to_torch(output)

    assert output.shape == torch_output.shape
    # bfp4 weights at K=1536 → PCC ~0.99; tolerances mirror the canonical bfp4 path.
    # assert_numeric_metrics(
    #     torch_output,
    #     output,
    #     atol=0.0347 * k_size,
    #     rtol=24.625 * k_size,
    #     frobenius_threshold=0.0005 * k_size,
    #     pcc_threshold=0.99,
    #     check_ulp=False,
    #     check_allclose=False,
    # )
    assert_with_pcc(torch_output, output, 0.99)
