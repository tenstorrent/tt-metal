# SPDX-FileCopyrightText: � 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_matmul_block_sharded_multicast(device):
    """
    Test matrix multiplication with:
    - input0: [1, 1, 1024, 18432] bfloat16_b block-sharded on 64 cores
    - input1: [1, 1, 18432, 4608] bfp8_b dram-interleaved
    - output: block-sharded bfloat16
    """
    torch.manual_seed(0)

    # Input shapes
    seq_len = 1024
    inner_dim = 18432
    outer_dim = 4608
    input0_shape = [1, 1, seq_len, inner_dim]
    input1_shape = [1, 1, inner_dim, outer_dim]

    # Create random torch inputs
    torch_input0 = torch.randn(input0_shape, dtype=torch.bfloat16)
    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)

    # Compute torch reference
    torch_output = torch.matmul(torch_input0, torch_input1)

    # Setup block-sharded memory config for input0 (64 cores = 8x8 grid)
    grid_size = (8, 8)
    shard_height = seq_len // 8  # 256
    shard_width = inner_dim // 8  # 2304

    input0_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    input0_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=input0_shard_spec,
    )

    # Convert to ttnn tensors
    in0_dtype = ttnn.bfloat16  # baseline
    # in0_dtype = ttnn.bfloat4_b
    ttnn_input0 = ttnn.from_torch(
        torch_input0, dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input0_memory_config
    )

    # baseline
    in1_dtype = ttnn.bfloat8_b
    # in1_dtype = ttnn.bfloat4_b
    ttnn_input1 = ttnn.from_torch(
        torch_input1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Setup output block-sharded memory config
    output_shard_height = seq_len // 8  # 128
    output_shard_width = outer_dim // 8  # 576

    output_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    # initial setup
    subblock_h = 1
    subblock_w = 1

    # immediate
    # subblock_h = 1
    # subblock_w = 6

    # Setup program config
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=8,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        per_core_M=4,
        per_core_N=18,
        transpose_mcast=False,
        fused_activation=None,
    )

    math_fidelity = ttnn.MathFidelity.HiFi2
    # math_fidelity = ttnn.MathFidelity.LoFi
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    # Run ttnn matmul
    ttnn_output = ttnn.matmul(
        ttnn_input0,
        ttnn_input1,
        program_config=program_config,
        memory_config=output_memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config,
    )

    # Convert back to torch and compare
    output_torch = ttnn.to_torch(ttnn_output)

    # Check PCC
    assert_with_pcc(torch_output, output_torch, 0.99)
