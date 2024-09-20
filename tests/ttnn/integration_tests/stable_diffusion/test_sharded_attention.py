# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    comp_pcc,
    tt2torch_tensor,
    torch2tt_tensor,
    is_wormhole_b0,
    skip_for_grayskull,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    determine_largest_subblock_size,
    determine_blocking,
)


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize("seq_len", [4096, 1024])
@pytest.mark.parametrize("num_slices", [16])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttnn.bfloat8_b])
def test_time_sharded_attnention_hwb(
    device,
    seq_len,
    num_slices,
    num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
    pytest.skip()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    M = seq_len
    K = 64
    N = seq_len

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, seq_len]
    value_layer_shape = [1, num_heads, seq_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG

    height_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1
    )
    block_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    attn_weights_qkt = torch_query_layer @ torch_key_layer_transposed
    attn_weights_torch_sm = torch.nn.functional.softmax(attn_weights_qkt, dim=-1)
    attn_weights_torch = attn_weights_torch_sm @ torch_value_layer

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    mm_out = torch2tt_tensor(
        torch_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    tiles_per_shard = math.ceil((((num_heads * seq_len) / num_cores) / num_slices) / 32)
    mm_output_block_shard_spec = [seq_len // 8, seq_len // 8]
    tiles_per_shard = math.ceil((((num_heads * seq_len) / num_cores) / num_slices) / 32)
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    heads_per_slice = num_heads // num_slices
    for i in range(num_slices):
        q_slice = ttnn.interleaved_to_sharded_partial(
            reference_query_layer,
            ttnn.CoreCoord(1, grid_size[0]),
            [M // grid_size[0], K],
            num_slices,
            i,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        k_slice = ttnn.interleaved_to_sharded_partial(
            reference_key_layer_transposed,
            ttnn.CoreCoord(grid_size[1], 1),
            [K, N // grid_size[1]],
            num_slices,
            i,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=M // (32 * grid_size[0]),
            per_core_N=N // (32 * grid_size[1]),
            transpose_mcast=False,
            fused_activation=None,
        )

        mm_slice = ttnn.matmul(
            q_slice,
            k_slice,
            program_config=program_config,
            memory_config=block_sharded_mem_config,
            dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        # mmt = tt2torch_tensor(mm_slice)
        # passed, message = comp_pcc(mmt, attn_weights_qkt[:, i * heads_per_slice : (i + 1) * heads_per_slice, :, :])
        # print(message)
        # assert passed
        k_slice.deallocate()
        q_slice.deallocate()

        height_per_core = seq_len // 64
        output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        output_shard_spec = ttnn.ShardSpec(
            output_shard_grid, [height_per_core, seq_len], ttnn.ShardOrientation.ROW_MAJOR, False
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )
        mm_slice = ttnn.reshard(
            mm_slice,
            output_mem_config,
        )
        mm_slice = ttnn.move(mm_slice)

        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )
        # print(program_config)

        mm_slice = ttnn.softmax_in_place(mm_slice, program_config=softmax_program_config)
        # mmt = tt2torch_tensor(mm_slice)
        # passed, message = comp_pcc(mmt, attn_weights_torch_sm[:, i * heads_per_slice : (i + 1) * heads_per_slice, :, :])
        # print(message)
        # assert passed

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        v_slice = ttnn.slice(
            reference_value_layer,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), seq_len - 1, 63),
            memory_config=dram_interleaved_memory_config,
        )

        mm_slice = ttnn.matmul(
            mm_slice,
            v_slice,
            program_config=program_config,
            memory_config=height_sharded_mem_config,
            dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        v_slice.deallocate()

        ttnn.sharded_to_interleaved_partial(
            mm_slice,
            mm_out,
            num_slices,
            i,
            memory_config=dram_interleaved_memory_config,
        )

        mm_slice.deallocate()

    mm_out_torch = tt2torch_tensor(mm_out)

    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize("seq_len", [4096, 1024])
@pytest.mark.parametrize("num_slices", [16])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttnn.bfloat8_b])
def test_time_sharded_attnention(
    device,
    seq_len,
    num_slices,
    num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
    pytest.skip()  # ND hang on CI
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, seq_len]
    value_layer_shape = [1, num_heads, seq_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG
    l1_interleaved_memory_config = ttnn.L1_MEMORY_CONFIG

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    passing = True
    output = None

    mm_out = torch2tt_tensor(
        torch_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    tiles_per_shard = math.ceil((((num_heads * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    heads_per_slice = num_heads // num_slices
    for i in range(num_slices):
        slice = ttnn.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,
            i,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=seq_len // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        k_slice = ttnn.slice(
            reference_key_layer_transposed,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), 63, seq_len - 1),
            memory_config=l1_interleaved_memory_config,
        )
        mm_slice = ttnn.matmul(
            slice,
            k_slice,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        k_slice.deallocate()
        slice.deallocate()

        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttnn.softmax_in_place(mm_slice, program_config=softmax_program_config)

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        v_slice = ttnn.slice(
            reference_value_layer,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), seq_len - 1, 63),
            memory_config=l1_interleaved_memory_config,
        )
        mm_slice = ttnn.matmul(
            mm_slice,
            v_slice,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        v_slice.deallocate()

        ttnn.sharded_to_interleaved_partial(
            mm_slice,
            mm_out,
            num_slices,
            i,
            memory_config=dram_interleaved_memory_config,
        )

        mm_slice.deallocate()

        return

    mm_out_torch = tt2torch_tensor(mm_out)

    attn_weights = ttnn.matmul(
        reference_query_layer, reference_key_layer_transposed, memory_config=dram_interleaved_memory_config
    )
    attn_weights = ttnn.softmax_in_place(attn_weights)
    attn_weights = ttnn.matmul(attn_weights, reference_value_layer, memory_config=dram_interleaved_memory_config)

    attn_weights_torch = tt2torch_tensor(attn_weights)
    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize("seq_len", [4096, 1024, 256, 64])
@pytest.mark.parametrize("kv_len", [96])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttnn.bfloat8_b])
@pytest.mark.parametrize("reshard_for_softmax", [True, False])
def test_cross_attnention(
    device,
    seq_len,
    kv_len,
    num_heads,
    data_format,
    reshard_for_softmax,
    function_level_defaults,
):
    if seq_len == 64 and reshard_for_softmax:
        pytest.skip()
    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = (8, 2)
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, kv_len]
    value_layer_shape = [1, num_heads, kv_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG
    l1_interleaved_memory_config = ttnn.L1_MEMORY_CONFIG

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    q_sharded = ttnn.interleaved_to_sharded(
        reference_query_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=kv_len // 32,
    )
    print(program_config)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mm_slice = ttnn.matmul(
        q_sharded,
        reference_key_layer_transposed,
        program_config=program_config,
        memory_config=height_sharded_memory_config,
        dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    q_sharded.deallocate()

    if reshard_for_softmax:
        height_per_core = num_heads * seq_len // 64
        orig_mem_config = mm_slice.memory_config()
        if seq_len == 1024:
            mm_slice = ttnn.sharded_to_interleaved(mm_slice, dram_interleaved_memory_config)
            mm_slice = ttnn.interleaved_to_sharded(
                mm_slice,
                (8, 8),
                [height_per_core, kv_len],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )
        else:
            output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid, [height_per_core, kv_len], ttnn.ShardOrientation.COL_MAJOR, False
            )
            output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
            )
            mm_slice = ttnn.reshard(
                mm_slice,
                output_mem_config,
            )
        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=1,
            block_h=32,
            block_w=3,
        )
        mm_slice = ttnn.softmax_in_place(mm_slice, program_config=softmax_program_config)
        mm_slice = ttnn.reshard(mm_slice, orig_mem_config)

    else:
        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=seq_len // 32,
            block_w=kv_len // 32,
        )
        mm_slice = ttnn.softmax_in_place(mm_slice, program_config=softmax_program_config)

    v_sharded = ttnn.interleaved_to_sharded(
        reference_value_layer,
        grid_size,
        [num_heads * kv_len // num_cores, 64],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=kv_len // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=2,
    )
    mm_slice = ttnn.matmul(
        mm_slice,
        v_sharded,
        program_config=program_config,
        memory_config=height_sharded_memory_config,
        dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    v_sharded.deallocate()

    mm_out_torch = tt2torch_tensor(mm_slice)

    attn_weights_torch = torch_query_layer @ torch_key_layer_transposed
    attn_weights_torch = torch.nn.functional.softmax(attn_weights_torch, dim=-1)
    attn_weights_torch = attn_weights_torch @ torch_value_layer

    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize("seq_len", [1024, 256, 64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttnn.bfloat8_b])
@pytest.mark.parametrize("reshard_for_softmax", [True, False])
def test_attention(
    device,
    seq_len,
    num_heads,
    data_format,
    reshard_for_softmax,
    function_level_defaults,
):
    if (seq_len == 64 or seq_len == 1024) and reshard_for_softmax:
        pytest.skip()
    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = (2, 8)
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, seq_len]
    value_layer_shape = [1, num_heads, seq_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG
    l1_interleaved_memory_config = ttnn.L1_MEMORY_CONFIG

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    q_sharded = ttnn.interleaved_to_sharded(
        reference_query_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    M = num_heads * seq_len
    K = 64
    N = seq_len
    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M // num_cores // 32,
        per_core_N=N // 32,
    )
    print(program_config)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mm_slice = ttnn.matmul(
        q_sharded,
        reference_key_layer_transposed,
        program_config=program_config,
        memory_config=height_sharded_memory_config,
        dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    q_sharded.deallocate()

    if reshard_for_softmax:
        height_per_core = num_heads * seq_len // 64
        orig_mem_config = mm_slice.memory_config()
        if seq_len == 1024:
            mm_slice = ttnn.sharded_to_interleaved(mm_slice, l1_interleaved_memory_config)
            mm_slice = ttnn.interleaved_to_sharded(
                mm_slice,
                (8, 8),
                [height_per_core, seq_len],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                subblock_w=1,
                block_h=height_per_core // 32,
                block_w=seq_len // 32,
            )
            mm_slice = ttnn.softmax_in_place(mm_slice, program_config=softmax_program_config)
            mm_slice = ttnn.sharded_to_interleaved(mm_slice, l1_interleaved_memory_config)
            mm_slice = ttnn.interleaved_to_sharded(
                mm_slice,
                (8, 2),
                [num_heads * seq_len // 16, seq_len],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR,
            )

        else:
            output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid, [height_per_core, seq_len], ttnn.ShardOrientation.COL_MAJOR, False
            )
            output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
            )
            mm_slice = ttnn.reshard(
                mm_slice,
                output_mem_config,
            )
            softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                subblock_w=1,
                block_h=height_per_core // 32,
                block_w=seq_len // 32,
            )
            mm_slice = ttnn.softmax_in_place(mm_slice, program_config=softmax_program_config)
            mm_slice = ttnn.reshard(mm_slice, orig_mem_config)
    else:
        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=seq_len // 32,
            block_w=seq_len // 32,
        )
        print(softmax_program_config)
        mm_slice = ttnn.softmax_in_place(mm_slice, program_config=softmax_program_config)

    v_sharded = ttnn.interleaved_to_sharded(
        reference_value_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=seq_len // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=2,
    )
    print(program_config)
    mm_slice = ttnn.matmul(
        mm_slice,
        v_sharded,
        program_config=program_config,
        memory_config=height_sharded_memory_config,
        dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    v_sharded.deallocate()

    mm_out_torch = tt2torch_tensor(mm_slice)

    attn_weights_torch = torch_query_layer @ torch_key_layer_transposed
    attn_weights_torch = torch.nn.functional.softmax(attn_weights_torch, dim=-1)
    attn_weights_torch = attn_weights_torch @ torch_value_layer

    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


@skip_for_grayskull()
@pytest.mark.parametrize("size", [4096, 1024, 256, 64])
@pytest.mark.parametrize("is_qkv", [1, 2, 3])
@pytest.mark.parametrize("data_format", [ttnn.bfloat8_b])
def test_q_and_kv(
    device,
    size,
    data_format,
    is_qkv,
    function_level_defaults,
):
    # Test matmul attention sequence with InterleavedToShardedPartialOp
    sizes = {4096: [1, 8192, 320, 512], 1024: [1, 2048, 640, 768], 256: [1, 512, 1280, 1280], 64: [1, 128, 1280, 1280]}
    grid_sizes = {4096: (5, 8), 1024: (5, 8), 256: (8, 8), 64: (8, 4)}
    B, M, K, N = sizes[size]
    N = N * is_qkv
    grid_size = grid_sizes[size]
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    in_0_shape = [1, B, M, K]
    in_1_shape = [1, B, K, N]
    in_2_shape = [1, B, 192, K]
    in_3_shape = [1, B, K, 2 * N]

    in_0_torch = torch.randn(in_0_shape).bfloat16().float()
    in_1_torch = torch.randn(in_1_shape).bfloat16().float()
    in_2_torch = torch.randn(in_2_shape).bfloat16().float()
    in_3_torch = torch.randn(in_3_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG
    l1_interleaved_memory_config = ttnn.L1_MEMORY_CONFIG

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1
    )

    block_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1
    )

    # compare output to regular case
    in_0 = torch2tt_tensor(
        in_0_torch,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_1 = torch2tt_tensor(
        in_1_torch,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_2 = torch2tt_tensor(
        in_2_torch,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_3 = torch2tt_tensor(
        in_3_torch,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    in_0_sharded = ttnn.interleaved_to_sharded(
        in_0,
        grid_size,
        [M // grid_size[1], K // grid_size[0]],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    M, K = in_0.shape[-2], in_0.shape[-1]
    N = in_1.shape[-1]
    in0_block_h, in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w = determine_blocking(
        M, K, N, grid_size
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=None,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    mm = ttnn.matmul(
        in_0_sharded if size != 4096 else in_0,
        in_1,
        program_config=program_config,
        memory_config=block_sharded_memory_config,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=compute_kernel_config,
    )
    in_0_sharded.deallocate()

    M, K, N = in_2.shape[-2], in_2.shape[-1], in_3.shape[-1]
    in0_block_h = M // grid_size[1] // 32
    in0_block_w = K // grid_size[0] // 32
    out_block_h = math.ceil(M / grid_size[1] / 32)
    out_block_w = math.ceil(N / grid_size[0] / 32)
    out_subblock_h, out_subblock_w = determine_largest_subblock_size(out_block_h, out_block_w)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=None,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mm_out_torch = tt2torch_tensor(mm)

    out_torch = in_0_torch @ in_1_torch

    passing, output = comp_pcc(mm_out_torch, out_torch)

    print(output)
    assert passing
