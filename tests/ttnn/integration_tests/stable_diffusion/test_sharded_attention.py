# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import pytest

import tt_lib as ttl
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0


# Test matmul attention sequence with InterleavedToShardedPartialOp
@pytest.mark.parametrize("seq_len", [4096, 1024])
@pytest.mark.parametrize("num_slices", [16])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT16])
def test_time_sharded_attnention_hwb(
    device,
    seq_len,
    num_slices,
    num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
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

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )
    block_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
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

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
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
        q_slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            ttl.tensor.CoreCoord(1, grid_size[0]),
            [M // grid_size[0], K],
            num_slices,
            i,
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        k_slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_key_layer_transposed,
            ttl.tensor.CoreCoord(grid_size[1], 1),
            [K, N // grid_size[1]],
            num_slices,
            i,
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=M // (32 * grid_size[0]),
            per_core_N=N // (32 * grid_size[1]),
            transpose_mcast=False,
            fused_activation=None,
        )

        mm_slice = ttl.operations.primary.matmul(
            q_slice,
            k_slice,
            program_config=program_config,
            output_mem_config=block_sharded_mem_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        # mmt = tt2torch_tensor(mm_slice)
        # passed, message = comp_pcc(mmt, attn_weights_qkt[:, i * heads_per_slice : (i + 1) * heads_per_slice, :, :])
        # print(message)
        # assert passed
        k_slice.deallocate()
        q_slice.deallocate()

        height_per_core = seq_len // 64
        output_shard_grid = ttl.tensor.CoreRangeSet(
            {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(7, 7))}
        )
        output_shard_spec = ttl.tensor.ShardSpec(
            output_shard_grid, [height_per_core, seq_len], ttl.tensor.ShardOrientation.ROW_MAJOR, False
        )
        output_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, output_shard_spec
        )
        mm_slice = ttl.tensor.reshard(
            mm_slice,
            output_mem_config,
        )
        mm_slice = ttl.tensor.move_sharded(mm_slice)

        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )
        # print(program_config)

        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)
        # mmt = tt2torch_tensor(mm_slice)
        # passed, message = comp_pcc(mmt, attn_weights_torch_sm[:, i * heads_per_slice : (i + 1) * heads_per_slice, :, :])
        # print(message)
        # assert passed

        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
        v_slice = ttl.tensor.unpad(
            reference_value_layer,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), seq_len - 1, 63),
            output_mem_config=dram_interleaved_memory_config,
        )

        mm_slice = ttl.operations.primary.matmul(
            mm_slice,
            v_slice,
            program_config=program_config,
            output_mem_config=height_sharded_mem_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        v_slice.deallocate()

        ttl.tensor.sharded_to_interleaved_partial(
            mm_slice,
            mm_out,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        mm_slice.deallocate()

    mm_out_torch = tt2torch_tensor(mm_out)

    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@pytest.mark.parametrize("seq_len", [4096, 1024])
@pytest.mark.parametrize("num_slices", [16])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
def test_time_sharded_attnention(
    device,
    seq_len,
    num_slices,
    num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
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

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
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

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
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
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]
    heads_per_slice = num_heads // num_slices
    for i in range(num_slices):
        slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,
            i,
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=seq_len // 32,
            out_subblock_h=1,
            out_subblock_w=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        k_slice = ttl.tensor.unpad(
            reference_key_layer_transposed,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), 63, seq_len - 1),
            output_mem_config=dram_interleaved_memory_config,
        )
        mm_slice = ttl.operations.primary.matmul(
            slice,
            k_slice,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        k_slice.deallocate()
        slice.deallocate()

        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=1,
            out_subblock_w=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        v_slice = ttl.tensor.unpad(
            reference_value_layer,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), seq_len - 1, 63),
            output_mem_config=dram_interleaved_memory_config,
        )
        mm_slice = ttl.operations.primary.matmul(
            mm_slice,
            v_slice,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        v_slice.deallocate()

        ttl.tensor.sharded_to_interleaved_partial(
            mm_slice,
            mm_out,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        mm_slice.deallocate()

    mm_out_torch = tt2torch_tensor(mm_out)

    attn_weights = ttl.tensor.bmm(
        reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
    )
    attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)
    attn_weights = ttl.tensor.bmm(attn_weights, reference_value_layer, output_mem_config=dram_interleaved_memory_config)

    attn_weights_torch = tt2torch_tensor(attn_weights)
    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@pytest.mark.parametrize("seq_len", [4096, 1024, 256, 64])
@pytest.mark.parametrize("kv_len", [96])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
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

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
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

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    q_sharded = ttl.tensor.interleaved_to_sharded(
        reference_query_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=kv_len // 32,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mm_slice = ttl.operations.primary.matmul(
        q_sharded,
        reference_key_layer_transposed,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    q_sharded.deallocate()

    if reshard_for_softmax:
        height_per_core = num_heads * seq_len // 64
        orig_mem_config = mm_slice.memory_config()
        output_shard_grid = ttl.tensor.CoreRangeSet(
            {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(7, 7))}
        )
        output_shard_spec = ttl.tensor.ShardSpec(
            output_shard_grid, [height_per_core, kv_len], ttl.tensor.ShardOrientation.COL_MAJOR, False
        )
        output_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, output_shard_spec
        )
        mm_slice = ttl.tensor.reshard(
            mm_slice,
            output_mem_config,
        )
        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=1,
            block_h=32,
            block_w=3,
        )
        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)
        mm_slice = ttl.tensor.reshard(mm_slice, orig_mem_config)

    else:
        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=seq_len // 32,
            block_w=kv_len // 32,
        )
        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

    v_sharded = ttl.tensor.interleaved_to_sharded(
        reference_value_layer,
        grid_size,
        [num_heads * kv_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=kv_len // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=2,
    )
    mm_slice = ttl.operations.primary.matmul(
        mm_slice,
        v_sharded,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
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
@pytest.mark.parametrize("seq_len", [1024, 256, 64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("reshard_for_softmax", [True, False])
def test_attnention(
    device,
    seq_len,
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
    key_layer_transposed_shape = [1, num_heads, 64, seq_len]
    value_layer_shape = [1, num_heads, seq_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
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

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    q_sharded = ttl.tensor.interleaved_to_sharded(
        reference_query_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=seq_len // 32,
    )
    print(program_config)

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mm_slice = ttl.operations.primary.matmul(
        q_sharded,
        reference_key_layer_transposed,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    q_sharded.deallocate()

    if reshard_for_softmax:
        mm_slice = ttl.tensor.move_sharded(mm_slice)
        height_per_core = num_heads * seq_len // 64
        orig_mem_config = mm_slice.memory_config()
        output_shard_grid = ttl.tensor.CoreRangeSet(
            {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(7, 7))}
        )
        output_shard_spec = ttl.tensor.ShardSpec(
            output_shard_grid, [height_per_core, seq_len], ttl.tensor.ShardOrientation.COL_MAJOR, False
        )
        output_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, output_shard_spec
        )
        mm_slice = ttl.tensor.reshard(
            mm_slice,
            output_mem_config,
        )
        mm_slice = ttl.tensor.move_sharded(mm_slice)
        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=1,
            block_h=height_per_core // 32,
            block_w=seq_len // 32,
        )
        print(softmax_program_config)
        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)
        mm_slice = ttl.tensor.reshard(mm_slice, orig_mem_config)

    else:
        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=seq_len // 32,
            block_w=seq_len // 32,
        )
        print(softmax_program_config)
        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

    v_sharded = ttl.tensor.interleaved_to_sharded(
        reference_value_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=seq_len // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=2,
    )
    print(program_config)
    mm_slice = ttl.operations.primary.matmul(
        mm_slice,
        v_sharded,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
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


@pytest.mark.parametrize("size", [4096, 1024, 256, 64])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT16])
@pytest.mark.parametrize("interleaved_output", [True, False])
def test_qkv(
    device,
    size,
    data_format,
    interleaved_output,
    function_level_defaults,
):
    sizes = {
        4096: [1, 8192, 320, 1536],
        1024: [1, 2048, 640, 2304],
        256: [1, 512, 1280, 3840],
        64: [1, 128, 1280, 3840],
    }
    grid_sizes = {4096: (8, 5), 1024: (8, 5), 256: (8, 8), 64: (4, 8)}
    out_subblock_hs = {4096: 8, 1024: 8, 256: 2, 64: 1}
    B, M, K, N = sizes[size]
    grid_size = grid_sizes[size]
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    in_0_shape = [1, B, M, K]
    in_1_shape = [1, B, K, N]

    in_0_torch = torch.randn(in_0_shape).bfloat16().float()
    in_1_torch = torch.randn(in_1_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    block_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    in_0 = torch2tt_tensor(
        in_0_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_1 = torch2tt_tensor(
        in_1_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    in_0_sharded = ttl.tensor.interleaved_to_sharded(
        in_0,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    Nt = N // 32
    G = grid_size[1]
    per_core_N = (Nt - 1) // (G - 1) if Nt != 16 else 4
    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // grid_size[1] // 32,
        out_subblock_h=out_subblock_hs[size] if interleaved_output else 1,
        out_subblock_w=1,
        per_core_M=M // grid_size[0] // 32,
        per_core_N=per_core_N,
        fused_activation=None,
        transpose_mcast=True,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    mm = ttl.operations.primary.matmul(
        in_0_sharded,
        in_1,
        program_config=program_config,
        output_mem_config=dram_interleaved_memory_config if interleaved_output else block_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    # mm = ttl.tensor.bmm(
    #     in_0,
    #     in_1,
    #     l1_interleaved_memory_config,
    #     compute_kernel_config,
    # )

    mm_out_torch = tt2torch_tensor(mm)

    out_torch = in_0_torch @ in_1_torch

    passing, output = comp_pcc(mm_out_torch, out_torch)

    print(output)
    assert passing


@pytest.mark.parametrize("size", [4096, 1024, 256, 64])
@pytest.mark.parametrize("is_kv", [True, False])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT16])
@pytest.mark.parametrize("interleaved_output", [True, False])
def test_q_and_kv(
    device,
    size,
    is_kv,
    data_format,
    interleaved_output,
    function_level_defaults,
):
    sizes = {4096: [1, 8192, 320, 512], 1024: [1, 2048, 640, 768], 256: [1, 512, 1280, 1280], 64: [1, 128, 1280, 1280]}
    grid_sizes = {4096: (8, 5), 1024: (8, 5), 256: (8, 8), 64: (4, 8)}
    out_subblock_hs = {4096: 8, 1024: 8, 256: 2, 64: 1}

    # if size == 4096 and not is_kv:
    #     pytest.skip()

    B, M, K, N = sizes[size]
    if is_kv:
        N *= 2
    grid_size = grid_sizes[size]
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    in_0_shape = [1, B, M, K]
    in_1_shape = [1, B, K, N]

    in_0_torch = torch.randn(in_0_shape).bfloat16().float()
    in_1_torch = torch.randn(in_1_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    block_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    in_0 = torch2tt_tensor(
        in_0_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_1 = torch2tt_tensor(
        in_1_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    in_0_sharded = ttl.tensor.interleaved_to_sharded(
        in_0,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    Nt = N // 32
    G = grid_size[1]
    per_core_N = (Nt - 1) // (G - 1) if Nt != 16 else 4
    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // grid_size[1] // 32,
        out_subblock_h=out_subblock_hs[size] if interleaved_output else 1,
        out_subblock_w=1,
        per_core_M=M // grid_size[0] // 32,
        per_core_N=per_core_N,
        fused_activation=None,
        transpose_mcast=True,
    )
    print(f"Nt: {Nt}, G: {grid_size[1]}, Nt-1: {(Nt-1)}, G-1: {(grid_size[1]-1)} pcn: {(Nt-1)/(grid_size[1]-1)}")
    print(f"Nt/G: {Nt/grid_size[1]}, Nt/(G-1) = {Nt/(grid_size[1]-1)}")

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    mm = ttl.operations.primary.matmul(
        in_0 if size == 4096 and not is_kv else in_0_sharded,
        in_1,
        program_config=program_config,
        output_mem_config=dram_interleaved_memory_config if interleaved_output else block_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    # mm = ttl.tensor.bmm(
    #     in_0,
    #     in_1,
    #     l1_interleaved_memory_config,
    #     compute_kernel_config,
    # )

    mm_out_torch = tt2torch_tensor(mm)

    out_torch = in_0_torch @ in_1_torch

    passing, output = comp_pcc(mm_out_torch, out_torch)

    print(output)
    assert passing
