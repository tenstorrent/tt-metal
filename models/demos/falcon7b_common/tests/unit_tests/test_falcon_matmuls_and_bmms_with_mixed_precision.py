# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from enum import Enum, auto
from loguru import logger

import ttnn
from models.demos.falcon7b_common.tt.falcon_causallm import falcon_lm_head_matmul
from models.demos.falcon7b_common.tt.falcon_mlp import falcon_dense_4h_to_h_matmul, falcon_dense_h_to_4h_matmul
from models.demos.falcon7b_common.tt.model_utils import get_falcon_default_core_grid
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0
import torch
import math


class MatmulOpEnum(Enum):
    FALCON_FUSED_QKV_MATMUL = auto()
    FALCON_SELFOUT_MATMUL = auto()
    FALCON_DENSE_4H_TO_H_MATMUL = auto()
    FALCON_DENSE_H_TO_4H_MATMUL = auto()
    FALCON_LM_HEAD_MATMUL = auto()


def run_falcon_matmul_test(
    falcon_op,
    seq_len,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    device,
):
    pcc = 0.99
    if out_dtype == ttnn.bfloat8_b:
        pcc = 0.98

    if falcon_op == MatmulOpEnum.FALCON_FUSED_QKV_MATMUL:
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 4672]
        expected_output_shape = [1, 1, seq_len, 4672]
    elif falcon_op == MatmulOpEnum.FALCON_SELFOUT_MATMUL:
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 4544]
        expected_output_shape = [1, 1, seq_len, 4544]
    elif falcon_op == MatmulOpEnum.FALCON_DENSE_4H_TO_H_MATMUL:
        a_shape = [1, 1, seq_len, 18176]
        b_shape = [1, 1, 18176, 4544]
        expected_output_shape = [1, 1, seq_len, 4544]

        if (seq_len == 1024 and in0_dtype == in1_dtype == out_dtype == ttnn.bfloat16) or (
            seq_len == 2048 and (in0_dtype == ttnn.bfloat16 or in1_dtype == ttnn.bfloat16 or out_dtype == ttnn.bfloat16)
        ):
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttnn.DRAM_MEMORY_CONFIG
            in1_mem_config = ttnn.DRAM_MEMORY_CONFIG
            out_mem_config = ttnn.DRAM_MEMORY_CONFIG
    elif falcon_op == MatmulOpEnum.FALCON_DENSE_H_TO_4H_MATMUL:
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 18176]
        expected_output_shape = [1, 1, seq_len, 18176]

        if seq_len == 2048 and out_dtype == ttnn.bfloat16:
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttnn.DRAM_MEMORY_CONFIG
            in1_mem_config = ttnn.DRAM_MEMORY_CONFIG
            out_mem_config = ttnn.DRAM_MEMORY_CONFIG
    elif falcon_op == MatmulOpEnum.FALCON_LM_HEAD_MATMUL:
        a_shape = [1, 1, seq_len, 4544]
        b_shape = [1, 1, 4544, 65024]
        expected_output_shape = [1, 1, seq_len, 65024]

        if (
            seq_len == 512
            and (in0_dtype == ttnn.bfloat16 or in1_dtype == ttnn.bfloat16 or out_dtype == ttnn.bfloat16)
            or seq_len == 1024
            or seq_len == 2048
        ):
            logger.warning(
                f"For seq_len: {seq_len}, in0_dtype: {in0_dtype}, in1_dtype: {in1_dtype}, and out_dtype: {out_dtype}, L1 space is not enough. Running with in0, in1, and out on DRAM instead!"
            )
            in0_mem_config = ttnn.DRAM_MEMORY_CONFIG
            in1_mem_config = ttnn.DRAM_MEMORY_CONFIG
            out_mem_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        raise NotImplementedError(f"falcon matmul op is undefined!")

    torch.manual_seed(1234)

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = ttnn.Tensor(A, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)
    b_t = ttnn.Tensor(B, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, in1_mem_config)

    default_core_grid = get_falcon_default_core_grid(device)
    if falcon_op in (MatmulOpEnum.FALCON_FUSED_QKV_MATMUL, MatmulOpEnum.FALCON_SELFOUT_MATMUL):
        out = ttnn.matmul(
            a_t,
            b_t,
            memory_config=out_mem_config,
            dtype=out_dtype,
            core_grid=default_core_grid,
        )
    elif falcon_op == MatmulOpEnum.FALCON_DENSE_4H_TO_H_MATMUL:
        out = falcon_dense_4h_to_h_matmul(
            a_t,
            b_t,
            core_grid=default_core_grid,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
        )
    elif falcon_op == MatmulOpEnum.FALCON_DENSE_H_TO_4H_MATMUL:
        out = falcon_dense_h_to_4h_matmul(
            a_t,
            b_t,
            core_grid=default_core_grid,
            fused_activation=None,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
        )
    elif falcon_op == MatmulOpEnum.FALCON_LM_HEAD_MATMUL:
        out = falcon_lm_head_matmul(
            a_t, b_t, core_grid=default_core_grid, output_mem_config=out_mem_config, output_dtype=out_dtype
        )
    else:
        raise NotImplementedError(f"falcon matmul op is undefined!")

    # Check memory and dtype of inputs and outputs
    assert a_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert a_t.get_dtype() == in0_dtype
    assert b_t.memory_config().buffer_type == in1_mem_config.buffer_type
    assert b_t.get_dtype() == in1_dtype
    assert out.memory_config().buffer_type == out_mem_config.buffer_type
    assert out.get_dtype() == out_dtype
    logger.debug(f"in0 ({a_shape}): {a_t.memory_config().buffer_type} and {a_t.get_dtype()}")
    logger.debug(f"in1 ({b_shape}): {b_t.memory_config().buffer_type} and {b_t.get_dtype()}")
    logger.debug(f"out ({expected_output_shape}): {out.memory_config().buffer_type} and {out.get_dtype()}")

    assert out.get_legacy_shape() == expected_output_shape
    pyt_got_back_rm = tt2torch_tensor(out)

    ref_bmm = torch.matmul(A, B)

    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, pcc)
    logger.debug(f"Passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


# TODO: We could parametrize these separately for comprehensive testing
@skip_for_wormhole_b0("non-determinstic hang, see issue #5882")
@pytest.mark.parametrize(
    "in0_mem_config, in1_mem_config, out_mem_config",
    (
        (
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ),
    ids=["weights_DRAM"],
)
@pytest.mark.parametrize(
    "out_dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16),
    ids=["out_BFLOAT8_B", "out_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in1_dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16),
    ids=["in1_BFLOAT8_B", "in1_BFLOAT16"],
)
@pytest.mark.parametrize(
    "in0_dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16),
    ids=["in0_BFLOAT8_B", "in0_BFLOAT16"],
)
@pytest.mark.parametrize(
    "falcon_op",
    (
        MatmulOpEnum.FALCON_FUSED_QKV_MATMUL,
        MatmulOpEnum.FALCON_SELFOUT_MATMUL,
        MatmulOpEnum.FALCON_DENSE_4H_TO_H_MATMUL,
        MatmulOpEnum.FALCON_DENSE_H_TO_4H_MATMUL,
        MatmulOpEnum.FALCON_LM_HEAD_MATMUL,
    ),
    ids=["fused_qkv", "selfout", "dense_4h_to_h", "dense_h_to_4h", "lm_head"],
)
@pytest.mark.parametrize(
    "seq_len",
    (128, 256, 512, 1024, 2048),
    ids=["seq_len_128", "seq_len_256", "seq_len_512", "seq_len_1024", "seq_len_2048"],
)
def test_falcon_matmul(
    falcon_op,
    seq_len,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    request,
    device,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    is_e75_grid_size = (compute_grid_size.x * compute_grid_size.y) == 88
    if is_e75_grid_size and (seq_len == 512) and (falcon_op == MatmulOpEnum.FALCON_LM_HEAD_MATMUL):
        pytest.skip(f"LM Head does not work on E75 grid size {compute_grid_size}")

    run_falcon_matmul_test(
        falcon_op,
        seq_len,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        device,
    )


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_wormhole_b0("non-determinstic hang, see issue #5882")
@pytest.mark.parametrize("seq_len", [128, 1024, 2048], ids=["seq_len_128", "seq_len_1024", "seq_len_2048"])
@pytest.mark.parametrize("num_cores", [64])
def test_falcon7b_attnention_sliced(
    device,
    seq_len,
    num_cores,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    num_heads = 64

    if seq_len == 128:
        num_slices = 1
    elif seq_len == 1024:
        num_slices = 4
    elif seq_len == 2048:
        num_slices = 16

    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    attention_mask_shape = [1, 71, seq_len, seq_len]
    scalar_shape = [1, 1, 32, 32]
    value_layer_shape = [1, 1, seq_len, 64]
    attention_output_shape = [1, 71, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_attention_mask = torch.randn(attention_mask_shape).bfloat16().float()
    torch_scalar = (torch.ones(scalar_shape) * (1 / math.sqrt(num_heads))).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_attention_output = torch.randn(attention_output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )
    attention_mask = torch2tt_tensor(
        torch_attention_mask,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )
    reference_scalar = torch2tt_tensor(
        torch_scalar,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    passing = True
    output = None

    attention_output_concatenated = torch2tt_tensor(
        torch_attention_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )
    tiles_per_shard = math.ceil((((71 * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    for i in range(num_slices):
        slice = ttnn.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        subblock_h = 1
        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8  # best option
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=seq_len // 32,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        mm_slice = ttnn.matmul(
            slice,
            reference_key_layer_transposed,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
        )

        # This should be done in-place.  This will be done by using an optional output tensor passed to multiply
        # but this feature is being done on a separate effort.
        mm_slice = ttnn.multiply(mm_slice, reference_scalar, memory_config=height_sharded_memory_config)

        # Deallocating here causes pcc to drop - issue #6638
        # So we have to move it after the entire sequence is finished
        # slice.deallocate()

        attn_mask_slice = ttnn.interleaved_to_sharded_partial(
            attention_mask,
            grid_size,
            mm_output_height_shard_spec,
            num_slices,
            i,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        mm_slice = ttnn.add(
            mm_slice,
            attn_mask_slice,
            fused_activations=None,
            memory_config=height_sharded_memory_config,
            output_dtype=ttnn.bfloat16,
            output_tensor=mm_slice,
        )

        attn_mask_slice.deallocate()

        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8
        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=subblock_w,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttnn.softmax_in_place(
            mm_slice, program_config=softmax_program_config, compute_kernel_config=compute_kernel_config
        )

        subblock_w = 2
        subblock_h = 1
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        attn_out_slice = ttnn.matmul(
            mm_slice,
            reference_value_layer,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
        )

        ttnn.sharded_to_interleaved_partial(
            attn_out_slice,
            attention_output_concatenated,
            num_slices,
            i,
            memory_config=dram_interleaved_memory_config,
        )

        slice.deallocate()
        mm_slice.deallocate()
        attn_out_slice.deallocate()

    attention_output_concatenated_torch = tt2torch_tensor(attention_output_concatenated)

    attn_weights = ttnn.matmul(
        reference_query_layer, reference_key_layer_transposed, memory_config=dram_interleaved_memory_config
    )

    attn_weights = ttnn.multiply(
        attn_weights,
        reference_scalar,
        memory_config=dram_interleaved_memory_config,
    )
    attn_weights = ttnn.add(attn_weights, attention_mask, memory_config=dram_interleaved_memory_config)
    attn_weights = ttnn.softmax_in_place(attn_weights, compute_kernel_config=compute_kernel_config)
    attn_output = ttnn.matmul(attn_weights, reference_value_layer)
    attn_output_torch = tt2torch_tensor(attn_output)
    passing = True

    attn_output_torch_reshaped = attn_output_torch.view(1, 1, 71 * seq_len, 64)
    attention_output_concatenated_torch_reshaped = attention_output_concatenated_torch.view(1, 1, 71 * seq_len, 64)
    slice_length = (71 * seq_len) // num_slices
    for slice_index in range(num_slices):
        print("Comparing slice ", slice_index, "...")
        slice_passing = False
        slice_passing, output = comp_pcc(
            attn_output_torch_reshaped[:, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :],
            attention_output_concatenated_torch_reshaped[
                :, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :
            ],
        )
        passing = passing and slice_passing
        print("Slice PCC is: ", output)

    # Compare entire tensors as well
    entire_tensor_passing, output = comp_pcc(attn_output_torch, attention_output_concatenated_torch)
    passing = entire_tensor_passing and passing

    print(output)
    assert passing


@pytest.mark.parametrize("seq_len", [128, 1024, 2048], ids=["seq_len_128", "seq_len_1024", "seq_len_2048"])
@pytest.mark.parametrize("num_cores", [64])
@skip_for_wormhole_b0("non-determinstic hang, see issue #5882")
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_falcon7b_attention_softmax_sequence(
    device,
    seq_len,
    num_cores,
    async_mode,
    use_program_cache,
    function_level_defaults,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    device.enable_async(async_mode)
    grid_size = (8, 8)
    num_heads = 64

    if seq_len == 1024:
        num_slices = 4
    elif seq_len == 2048:
        num_slices = 16
    elif seq_len == 128:
        num_slices = 1

    query_layer_shape = [1, 71, seq_len, 64]
    key_layer_transposed_shape = [1, 1, 64, seq_len]
    attention_mask_shape = [1, 71, seq_len, seq_len]
    attention_mask_proper_dim_shape = [1, 1, seq_len, seq_len]
    scalar_shape = [1, 1, 32, 32]
    value_layer_shape = [1, 1, seq_len, 64]
    attention_output_shape = [1, 71, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_attention_mask_proper_dim = torch.randn(attention_mask_proper_dim_shape).bfloat16().float()
    torch_attention_mask = torch_attention_mask_proper_dim.repeat(1, attention_mask_shape[1], 1, 1)
    scalar_value = 1 / math.sqrt(num_heads)
    torch_scalar = (torch.ones(scalar_shape) * scalar_value).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_attention_output = torch.randn(attention_output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )

    attention_mask_proper_dim = torch2tt_tensor(
        torch_attention_mask_proper_dim,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat4_b,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # We need to create attention masks per slice
    attention_masks_per_slice = []
    attention_mask_starting_index_per_slice = 0
    slice_length = (71 * seq_len) // num_slices
    number_of_attention_mask_elements_used_per_slice = slice_length - seq_len * (slice_length // seq_len)
    # print("Slice length is: ", slice_length)
    # print("Number of attention mask elements per slice = ", number_of_attention_mask_elements_used_per_slice)
    for slice_index in range(num_slices):
        print("Slice attention mask starting index: ", attention_mask_starting_index_per_slice)
        torch_attention_mask_per_slice = torch.cat(
            [
                torch_attention_mask_proper_dim[:, :, attention_mask_starting_index_per_slice:, :],
                torch_attention_mask_proper_dim[:, :, :attention_mask_starting_index_per_slice, :],
            ],
            dim=2,
        )
        tt_attention_slice = torch2tt_tensor(
            torch_attention_mask_per_slice,
            device,
            tt_memory_config=dram_interleaved_memory_config,
            tt_dtype=ttnn.bfloat4_b,
        )
        attention_masks_per_slice.append(tt_attention_slice)
        attention_mask_starting_index_per_slice = (
            attention_mask_starting_index_per_slice + number_of_attention_mask_elements_used_per_slice
        ) % seq_len  # mod attention_mask.height

    reference_scalar = torch2tt_tensor(
        torch_scalar,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )

    passing = True
    output = None

    attention_output_concatenated = torch2tt_tensor(
        torch_attention_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )
    tiles_per_shard = math.ceil((((71 * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    for i in range(num_slices):
        slice = ttnn.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,  # num_slices
            i,  # slice_index
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        subblock_h = 1
        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8  # best option
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=seq_len // 32,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        mm_slice = ttnn.matmul(
            slice,
            reference_key_layer_transposed,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
        )

        # Deallocating here causes pcc to drop - issue #6638
        # So we have to move it after the entire sequence is finished
        # slice.deallocate()

        subblock_w = 1
        if seq_len == 2048:
            subblock_w = 8
        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=subblock_w,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
            mm_slice,
            scalar_value,
            attention_masks_per_slice[i],
            program_config=softmax_program_config,
            compute_kernel_config=compute_kernel_config,
        )

        subblock_w = 2
        subblock_h = 1
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        attn_out_slice = ttnn.matmul(
            mm_slice,
            reference_value_layer,
            program_config=program_config,
            memory_config=height_sharded_memory_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
        )

        ttnn.sharded_to_interleaved_partial(
            attn_out_slice,
            attention_output_concatenated,
            num_slices,
            i,
            memory_config=dram_interleaved_memory_config,
        )

        slice.deallocate()
        mm_slice.deallocate()
        attn_out_slice.deallocate()

    attention_output_concatenated_torch = tt2torch_tensor(attention_output_concatenated)

    attn_weights = ttnn.matmul(
        reference_query_layer, reference_key_layer_transposed, memory_config=dram_interleaved_memory_config
    )

    attn_weights = ttnn.scale_mask_softmax_in_place(
        attn_weights,
        scalar_value,
        attention_mask_proper_dim,
        program_config=ttnn.SoftmaxDefaultProgramConfig(),
        is_causal_mask=True,
        compute_kernel_config=compute_kernel_config,
    )

    attn_output = ttnn.matmul(attn_weights, reference_value_layer)
    attn_output_torch = tt2torch_tensor(attn_output)
    passing = True

    attn_output_torch_reshaped = attn_output_torch.view(1, 1, 71 * seq_len, 64)
    attention_output_concatenated_torch_reshaped = attention_output_concatenated_torch.view(1, 1, 71 * seq_len, 64)
    for slice_index in range(num_slices):
        print("Comparing slice ", slice_index, "...")
        slice_passing = False
        slice_passing, output = comp_pcc(
            attn_output_torch_reshaped[:, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :],
            attention_output_concatenated_torch_reshaped[
                :, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :
            ],
        )
        passing = passing and slice_passing
        print("Slice PCC is: ", output)

    # Compare entire tensors as well
    entire_tensor_passing, output = comp_pcc(attn_output_torch, attention_output_concatenated_torch)
    passing = entire_tensor_passing and passing

    print(output)
    assert passing


@pytest.mark.parametrize(
    "seq_len",
    (32, 64, 128, 1024, 2048),
    ids=["seq_len_32", "seq_len_64", "seq_len_128", "seq_len_1024", "seq_len_2048"],
)
@pytest.mark.parametrize("num_cores", [64])
def test_softmax(device, num_cores, seq_len):
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    head_dim = 71
    num_heads = 64
    torch.manual_seed(0)

    input_shape = [1, head_dim, seq_len, seq_len]
    attention_mask_shape = [1, 1, seq_len, seq_len]
    scalar_shape = [1, 1, 32, 32]

    scalar_value = 1 / math.sqrt(num_heads)
    torch_input = torch.randn(input_shape).bfloat16().float()
    torch_attention_mask = torch.randn(attention_mask_shape).bfloat16().float()
    torch_attention_mask_full = torch_attention_mask.repeat(1, head_dim, 1, 1)
    torch_scalar = (torch.ones(scalar_shape) * scalar_value).bfloat16().float()

    num_slices = 1
    if seq_len == 1024:
        num_slices = 4
    elif seq_len == 2048:
        num_slices = 16

    slice_length = (head_dim * seq_len) // num_slices

    torch_outputs = torch.zeros(input_shape).bfloat16().float()

    dram_interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )

    tt_attention_mask_full = torch2tt_tensor(
        torch_attention_mask_full,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )

    tt_scalar = torch2tt_tensor(
        torch_scalar,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )

    tt_attention_masks_per_slice = []
    attention_mask_starting_index_per_slice = 0
    number_of_attention_mask_elements_used_per_slice = slice_length - seq_len * (slice_length // seq_len)
    # print("Slice length is: ", slice_length)
    # print("Number of attention mask elements per slice = ", number_of_attention_mask_elements_used_per_slice)
    for slice_index in range(num_slices):
        # print("Slice attention mask starting index: ", attention_mask_starting_index_per_slice)
        torch_attention_mask_per_slice = torch.cat(
            [
                torch_attention_mask[:, :, attention_mask_starting_index_per_slice:, :],
                torch_attention_mask[:, :, :attention_mask_starting_index_per_slice, :],
            ],
            dim=2,
        )
        tt_attention_slice = torch2tt_tensor(
            torch_attention_mask_per_slice,
            device,
            tt_memory_config=dram_interleaved_memory_config,
            tt_dtype=ttnn.bfloat16,
        )
        tt_attention_masks_per_slice.append(tt_attention_slice)
        attention_mask_starting_index_per_slice = (
            attention_mask_starting_index_per_slice + number_of_attention_mask_elements_used_per_slice
        ) % seq_len  # mod attention_mask.height

    tt_output_sharded_softmax = torch2tt_tensor(
        torch_outputs,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=ttnn.bfloat16,
    )

    # Sharded softmax
    tiles_per_shard = math.ceil((((head_dim * seq_len) / num_cores) / num_slices) / 32)
    height_shard_spec = [tiles_per_shard * 32, seq_len]

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    for i in range(num_slices):
        input_slice = ttnn.interleaved_to_sharded_partial(
            tt_input,
            grid_size,
            height_shard_spec,
            num_slices,
            i,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        softmax_program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=height_shard_spec[0] // 32,
            block_w=height_shard_spec[1] // 32,
        )

        input_slice = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
            input_slice,
            scalar_value,
            tt_attention_masks_per_slice[i],
            program_config=softmax_program_config,
            compute_kernel_config=compute_kernel_config,
        )

        ttnn.sharded_to_interleaved_partial(
            input_slice, tt_output_sharded_softmax, num_slices, i, memory_config=dram_interleaved_memory_config
        )
        input_slice.deallocate()

    out = ttnn.multiply(tt_input, tt_scalar, memory_config=dram_interleaved_memory_config)

    out = ttnn.add(
        out,
        tt_attention_mask_full,
        memory_config=dram_interleaved_memory_config,
    )

    out = ttnn.softmax_in_place(out, compute_kernel_config=compute_kernel_config)

    out_torch = tt2torch_tensor(out)
    out_torch_view = out_torch.view(1, 1, out_torch.shape[1] * out_torch.shape[2], out_torch.shape[3])

    out_torch_softmax = tt2torch_tensor(tt_output_sharded_softmax)
    out_torch_softmax_view = out_torch_softmax.view(
        1, 1, out_torch_softmax.shape[1] * out_torch_softmax.shape[2], out_torch_softmax.shape[3]
    )

    # Compare slice pcc
    passing = True
    for slice_index in range(num_slices):
        print("Comparing slice ", slice_index, "...")
        slice_passing = False
        slice_passing, output = comp_pcc(
            out_torch_view[:, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :],
            out_torch_softmax_view[:, :, (slice_length) * slice_index : (slice_length) * (slice_index + 1), :],
        )
        passing = passing and slice_passing
        print("Slice PCC is: ", output)

    # Compare entire tensors as well
    entire_tensor_passing, output = comp_pcc(out_torch_view, out_torch_softmax_view)
    passing = entire_tensor_passing and passing

    print(output)
    assert passing
