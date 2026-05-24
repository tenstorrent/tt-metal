# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

# MatmulDeviceOperation dimensions: M x K x N
M = 32
K = 1536
N = 17920

# Shape in tiles:  Mt = 1,  Kt = 48,  Nt = 560
# Blackhole has 8 DRAM banks → use 8-core DRAM-sharded matmul-1D.
# Divisibility:
#   Nt / num_dram_banks            = 560 / 8     = 70   ✓
#   N  / (num_banks × TILE_W)      = 17920 / 256 = 70   ✓
#   Kt_per_core = Kt / num_cores   = 48 / 8      = 6    ✓
#   Kt_per_core % in0_block_w      = 6 % 2       = 0    ✓
# Note: validation is on PER-CORE Kt (= 6 tiles), not total Kt.
# Valid in0_block_w divisors of 6: {1, 2, 3, 6}.
NUM_CORES = 8
PER_CORE_M = M // 32  # 1
PER_CORE_N = (N // 32) // NUM_CORES  # 70
IN0_BLOCK_W = 2  # tune {1, 2, 3, 6} for L1 vs. overhead


@pytest.mark.parametrize(
    "m_size, k_size, n_size",
    [(M, K, N)],
    ids=[f"MatmulDeviceOperation_{M}x{K}x{N}"],
)
def test_matmul_device_operation_32x1536x17920(device, m_size, k_size, n_size):
    """DRAM-sharded matmul-1D PCC test for M=32, K=1536, N=17920.

    Optimal layout for this thin-M / wide-N memory-bound shape on Blackhole:
      - in1 (~55 MB weights) width-sharded across all 8 DRAM banks
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
    # Per-core shard = [M, K / 8] = [32, 192]
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
    # Per-bank shard = [K, N / 8] = [1536, 2240]
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
    # Per-core shard = [M, N / 8] = [32, 2240]
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
        in0_block_w=IN0_BLOCK_W,  # Kt_per_core / in0_block_w = 6 / 2 = 3 inner-loop iters
        per_core_M=PER_CORE_M,  # 1 — full Mt on each core (M-axis not parallelized)
        per_core_N=PER_CORE_N,  # 70 — Nt / num_dram_banks
        fused_activation=None,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,  # bf16 accumulation in L1 — important perf knob
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
