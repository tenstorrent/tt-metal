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
#
# Why mcast-1D (not DRAM-sharded) here?
#   DRAM-sharded matmul-1D is LOCKED to num_dram_banks cores (8 on Blackhole)
#   because each compute core is paired one-to-one with a DRAM bank.
#   mcast-1D, by contrast, lets us use ANY core grid that divides Nt.
#
# Divisors of Nt = 560 that fit on a 8x8 Blackhole worker grid:
#   1, 2, 4, 5, 7, 8, 10, 14, 16, 20, 28, 35, 40, 56
# → 56 cores (8x7) is the highest usable count: 7x more parallelism than DRAM-sharded.
GRID_X = 8  # cols
GRID_Y = 7  # rows
NUM_CORES = GRID_X * GRID_Y  # 56
PER_CORE_M = M // 32  # 1
PER_CORE_N = (N // 32) // NUM_CORES  # 560 / 56 = 10
IN0_BLOCK_W = 4  # divides Kt=48 → 12 inner-loop iters
OUT_SUBBLOCK_H = 1  # ≤ per_core_M
OUT_SUBBLOCK_W = 5  # divides per_core_N=10, h*w=5 ≤ 8 dest regs


@pytest.mark.parametrize(
    "m_size, k_size, n_size",
    [(M, K, N)],
    ids=[f"MatmulDeviceOperation_{M}x{K}x{N}"],
)
def test_matmul_device_operation_32x1536x17920_mcast1d(device, m_size, k_size, n_size):
    """mcast-1D matmul PCC test for M=32, K=1536, N=17920 on 56 cores.

    Parallelization strategy:
      - mcast_in0=True → activations (in0) are broadcast across the core grid
      - work is split along N: each of 56 cores computes a 1x10-tile output slice
      - in1 (weights) read from interleaved DRAM, distributing reads across
        all 8 DRAM banks naturally via the interleaved layout
      - in0 fits trivially (32x1536 = 96 KB) so multicast cost is negligible

    Compare against the 8-core DRAM-sharded variant
    (`test_matmul_32x1536x17920_shard.py`). For this memory-bound shape, the
    DRAM-sharded path is usually still slightly better because it gives
    perfect bank locality, but mcast-1D gets more compute parallelism and
    is the right choice when num_dram_banks-locked layout is not viable
    (e.g. shape not divisible by 8, or fused with downstream ops that need
    a different sharding).
    """
    torch.manual_seed(0)

    torch_input_a = torch.randn((1, 1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_b = torch.randn((1, 1, k_size, n_size), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    input_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(GRID_X, GRID_Y),
        in0_block_w=IN0_BLOCK_W,
        out_subblock_h=OUT_SUBBLOCK_H,
        out_subblock_w=OUT_SUBBLOCK_W,
        per_core_M=PER_CORE_M,
        per_core_N=PER_CORE_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    output = ttnn.matmul(
        input_a,
        input_b,
        program_config=program_config,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
