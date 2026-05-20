# python# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

# MatmulDeviceOperation dimensions: M x K x N
M = 128
K = 12288
N = 3072

# Shape in tiles:  Mt = 4,  Kt = 384,  Nt = 96
# Blackhole has 8 DRAM banks → use 8-core DRAM-sharded matmul-1D.
# Divisibility:
#   Nt / num_dram_banks       = 96 / 8     = 12   ✓
#   N  / (num_banks × TILE_W) = 3072 / 256 = 12   ✓
#   Kt % in0_block_w          = 384 % 4    = 0    ✓
NUM_CORES = 8
PER_CORE_M = M // 32  # 4
PER_CORE_N = (N // 32) // NUM_CORES  # 12
IN0_BLOCK_W = 4  # tune {2, 4, 8} for L1 vs. overhead


@pytest.mark.parametrize(
    "m_size, k_size, n_size",
    [(M, K, N)],
    ids=[f"MatmulDeviceOperation_{M}x{K}x{N}"],
)
def test_matmul_device_operation_128x12288x3072(device, m_size, k_size, n_size):
    """DRAM-sharded matmul-1D PCC test for M=128, K=12288, N=3072.

    Optimal layout for this memory-bound shape on Blackhole:
      - in1 (88 MB weights) width-sharded across all 8 DRAM banks
        → each compute core reads its nearest bank in parallel: full DRAM BW
      - in0 (activations) width-sharded in L1 on the same 8 cores
        → A is pre-staged near where it's needed; no DRAM contention for A
      - output stays L1-width-sharded
        → downstream op can consume directly from L1 (no DRAM round-trip)

    Expected speedup over the default DRAM-interleaved path: ~2-4×.
    """
    torch.manual_seed(0)

    torch_input_a = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((1, 1, k_size, n_size), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    # 8 compute cores arranged 8×1 (one row), one core per DRAM bank.
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})

    # in0: width-sharded in L1 across the 8-core row, K split across cores.
    # Per-core shard = [M, K / 8] = [128, 1536]
    in0_mem_cfg = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            core_grid,
            [m_size, k_size // NUM_CORES],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # in1: width-sharded in DRAM across the 8 DRAM banks, N split across banks.
    # Per-bank shard = [K, N / 8] = [12288, 384]
    in1_mem_cfg = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(
            core_grid,
            [k_size, n_size // NUM_CORES],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # output: width-sharded in L1 across the same 8 cores.
    # Per-core shard = [M, N / 8] = [128, 384]
    out_mem_cfg = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            core_grid,
            [m_size, n_size // NUM_CORES],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=IN0_BLOCK_W,  # Kt / in0_block_w = 384 / 4 = 96 inner-loop iters
        per_core_M=PER_CORE_M,  # 4 — full Mt on each core (M-axis not parallelized)
        per_core_N=PER_CORE_N,  # 12 — Nt / num_dram_banks
        fused_activation=None,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,  # bf16 accumulation in L1 — important perf knob
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
        dtype=ttnn.bfloat16,
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
    assert_numeric_metrics(
        torch_output,
        output,
        atol=0.004 * k_size,
        rtol=0.004 * k_size,
        frobenius_threshold=0.003 * k_size,
        pcc_threshold=0.999,
        check_ulp=False,
    )
