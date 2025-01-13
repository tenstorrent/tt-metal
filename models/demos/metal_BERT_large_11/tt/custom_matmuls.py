# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn


def bert_large_fused_qkv_matmul(
    input_tensor_a, input_tensor_b, bias=None, output_mem_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=None
):
    batch_size = input_tensor_a.shape.with_tile_padding()[0]

    assert input_tensor_a.shape.with_tile_padding() == [batch_size, 1, 384, 1024], "Unsupported input shape"
    assert input_tensor_b.shape.with_tile_padding() == [1, 1, 1024, 3072], "Unsupported input shape"

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(12, batch_size),
        in0_block_w=4,
        out_subblock_h=4,
        out_subblock_w=2,
        per_core_M=12,
        per_core_N=8,
        transpose_mcast=False,
        fused_activation=None,
    )
    return ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )


def bert_large_ff1_matmul(
    input_tensor_a,
    input_tensor_b,
    bias=None,
    fused_activation=None,
    output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
    output_dtype=None,
):
    batch_size = input_tensor_a.shape.with_tile_padding()[0]

    assert (
        (
            input_tensor_a.dtype != ttnn.bfloat16
            or input_tensor_b.dtype != ttnn.bfloat16
            or output_dtype != ttnn.bfloat16
        )
        or (output_mem_config.buffer_type == ttnn.BufferType.DRAM)
        or (
            input_tensor_a.memory_config().buffer_type == ttnn.BufferType.DRAM
            and input_tensor_b.memory_config().buffer_type == tttn.BufferType.DRAM
        )
    ), "For BFLOAT16, if output is on L1, one of in0 or in1 must be on DRAM!"
    assert input_tensor_a.shape.with_tile_padding() == [batch_size, 1, 384, 1024], "Unsupported input shape"
    assert input_tensor_b.shape.with_tile_padding() == [1, 1, 1024, 4096], "Unsupported input shape"

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(12, batch_size),
        in0_block_w=4,
        out_subblock_h=6,
        out_subblock_w=1,
        per_core_M=12,
        per_core_N=11,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )
    return ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )


def bert_large_ff2_matmul(
    input_tensor_a, input_tensor_b, bias=None, output_mem_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=None
):
    batch_size = input_tensor_a.shape.with_tile_padding()[0]

    assert input_tensor_a.shape.with_tile_padding() == [batch_size, 1, 384, 4096], "Unsupported input shape"
    assert input_tensor_b.shape.with_tile_padding() == [1, 1, 4096, 1024], "Unsupported input shape"

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(12, batch_size),
        in0_block_w=4,
        out_subblock_h=6,
        out_subblock_w=1,
        per_core_M=12,
        per_core_N=3,
        transpose_mcast=False,
        fused_activation=None,
    )
    return ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )


def bert_large_selfout_matmul(
    input_tensor_a, input_tensor_b, bias=None, output_mem_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=None
):
    batch_size = input_tensor_a.shape.with_tile_padding()[0]

    assert input_tensor_a.shape.with_tile_padding() == [batch_size, 1, 384, 1024], "Unsupported input shape"
    assert input_tensor_b.shape.with_tile_padding() == [1, 1, 1024, 1024], "Unsupported input shape"

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(12, batch_size),
        in0_block_w=4,
        out_subblock_h=6,
        out_subblock_w=1,
        per_core_M=12,
        per_core_N=3,
        transpose_mcast=False,
        fused_activation=None,
    )
    return ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )


def bert_large_pre_softmax_bmm(
    input_tensor_a, input_tensor_b, output_mem_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=None
):
    batch_size = input_tensor_a.shape.with_tile_padding()[0]

    assert input_tensor_a.shape.with_tile_padding() == [batch_size, 16, 384, 64], "Unsupported input shape"
    assert input_tensor_b.shape.with_tile_padding() == [batch_size, 16, 64, 384], "Unsupported input shape"

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(12, batch_size),
        in0_block_w=1,
        out_subblock_h=4,
        out_subblock_w=2,
        per_core_M=12,
        per_core_N=12,
    )
    return ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )


def bert_large_post_softmax_bmm(
    input_tensor_a, input_tensor_b, output_mem_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=None
):
    batch_size = input_tensor_a.shape.with_tile_padding()[0]

    assert input_tensor_a.shape.with_tile_padding() == [batch_size, 16, 384, 384], "Unsupported input shape"
    assert input_tensor_b.shape.with_tile_padding() == [batch_size, 16, 384, 64], "Unsupported input shape"

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(12, batch_size),
        in0_block_w=2,
        out_subblock_h=4,
        out_subblock_w=2,
        per_core_M=12,
        per_core_N=2,
    )
    return ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )
