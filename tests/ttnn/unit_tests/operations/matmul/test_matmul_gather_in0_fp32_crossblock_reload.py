# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for fp32 accumulation precision of the gather_in0 matmul cross-block reload.

The gather_in0 1D matmul (bmm_large_block_zm_fused_bias_activation_gathered.cpp) reduces K across
a ring of cores; num_blocks == num_cores and, with fp32_dest_acc_en=True and packer_l1_acc=False,
the Float32 K-partials are reloaded into DEST between ring steps by copy_block_matmul_partials.
Unless that CB is marked UnpackToDestFp32 the reload is routed through SrcA and truncated to TF32,
so fp32 accumulation degrades.

Large-offset inputs with in1 columns summing to zero over K make the offset cancel out of the true
result while driving the mid-accumulation partials large, so a TF32-truncated reload corrupts the
(small) true result. fp32 output makes the partials Float32 so the fp32-reload path is exercised.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


def test_matmul_gather_in0_fp32_crossblock_reload_precision(device):
    grid_x, grid_y = 8, 1
    num_cores = grid_x * grid_y
    M, K, N = 32, 2048, 2048  # K,N split across the 8-core ring -> num_blocks = num_cores = 8
    torch.manual_seed(0)

    a = torch.randn(1, 1, M, K) + 1000.0
    b = torch.randn(1, 1, K, N)
    b = b - b.mean(dim=2, keepdim=True)  # each output column sums to 0 over K
    a = a.bfloat16()
    b = b.bfloat16()
    ref = (a.double() @ b.double()).reshape(M, N).float()

    k_per_shard = K // num_cores
    n_per_shard = N // num_cores
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})

    def width_sharded(shard_hw):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_range_set, shard_hw, ttnn.ShardOrientation.ROW_MAJOR),
        )

    a_t = ttnn.from_torch(
        a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=width_sharded([M, k_per_shard])
    )
    b_t = ttnn.from_torch(
        b, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=width_sharded([K, n_per_shard])
    )

    out_block_w = N // num_cores // ttnn.TILE_SIZE  # 8 tiles; out_subblock_w=4 -> 2 subblocks -> spill
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=K // num_cores // ttnn.TILE_SIZE,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=M // ttnn.TILE_SIZE,
        per_core_N=out_block_w,
        fuse_batch=True,
        mcast_in0=False,
        gather_in0=True,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=True,
    )

    out = ttnn.matmul(
        a_t,
        b_t,
        program_config=program_config,
        memory_config=width_sharded([M, n_per_shard]),
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.float32,
    )
    out = ttnn.to_torch(out).reshape(M, N).float()

    assert_numeric_metrics(
        ref,
        out,
        pcc_threshold=0.99,
        frobenius_threshold=0.5,
        check_allclose=False,
        check_pcc=True,
        check_frobenius=True,
        check_ulp=False,
    )
