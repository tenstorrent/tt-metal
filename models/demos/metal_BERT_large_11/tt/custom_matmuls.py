# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import math

import ttnn
from models.demos.metal_BERT_large_11.tt import grid


def _fits_matmul_configs(input_tensor_a):
    """Whether the program configs below fit the device's grid; otherwise every matmul here runs
    with the program config ttnn chooses."""
    device_grid = input_tensor_a.device().compute_with_storage_grid_size()
    return grid.fits_matmul_configs(input_tensor_a.padded_shape[0], device_grid)


def _matmul_columns(input_tensor_a):
    return grid.matmul_columns(input_tensor_a.device().compute_with_storage_grid_size())


def _multicast_config(input_tensor_a, n, per_core_N, out_subblock_h, out_subblock_w, **kwargs):
    """A multicast matmul config on one core row per image and grid.matmul_columns columns. ``per_core_N``
    and the subblock width are the 12-column values; with fewer columns per_core_N grows until the
    columns cover the ``n`` output tiles, and the subblock width shrinks to a divisor of it."""
    if not _fits_matmul_configs(input_tensor_a):
        return None
    batch_size = input_tensor_a.padded_shape[0]
    columns = _matmul_columns(input_tensor_a)
    per_core_N = max(per_core_N, math.ceil(n / columns))
    while per_core_N % out_subblock_w:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(columns, batch_size),
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_N=per_core_N,
        **kwargs,
    )


def bert_large_fused_qkv_matmul(
    input_tensor_a, input_tensor_b, bias=None, output_mem_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=None
):
    batch_size = input_tensor_a.padded_shape[0]

    assert input_tensor_a.padded_shape == [batch_size, 1, 384, 1024], "Unsupported input shape"
    assert input_tensor_b.padded_shape == [1, 1, 1024, 3072], "Unsupported input shape"

    program_config = _multicast_config(
        input_tensor_a,
        n=3072 // ttnn.TILE_SIZE,
        per_core_N=8,
        out_subblock_h=4,
        out_subblock_w=2,
        in0_block_w=4,
        per_core_M=12,
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
    batch_size = input_tensor_a.padded_shape[0]

    assert (
        (
            input_tensor_a.dtype != ttnn.bfloat16
            or input_tensor_b.dtype != ttnn.bfloat16
            or output_dtype != ttnn.bfloat16
        )
        or (output_mem_config.buffer_type == ttnn.BufferType.DRAM)
        or (
            input_tensor_a.memory_config().buffer_type == ttnn.BufferType.DRAM
            and input_tensor_b.memory_config().buffer_type == ttnn.BufferType.DRAM
        )
    ), "For BFLOAT16, if output is on L1, one of in0 or in1 must be on DRAM!"
    assert input_tensor_a.padded_shape == [batch_size, 1, 384, 1024], "Unsupported input shape"
    assert input_tensor_b.padded_shape == [1, 1, 1024, 4096], "Unsupported input shape"

    program_config = _multicast_config(
        input_tensor_a,
        n=4096 // ttnn.TILE_SIZE,
        per_core_N=11,
        out_subblock_h=6,
        out_subblock_w=1,
        in0_block_w=4,
        per_core_M=12,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )
    output = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )
    if program_config is None and fused_activation is not None:
        # The activation is fused through the program config; without one it runs after the matmul.
        op, approximate = fused_activation
        assert op == ttnn.UnaryOpType.GELU, f"Unsupported activation {op}"
        output = ttnn.gelu(output, fast_and_approximate_mode=approximate, memory_config=output_mem_config)
    return output


def bert_large_ff2_matmul(
    input_tensor_a, input_tensor_b, bias=None, output_mem_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=None
):
    batch_size = input_tensor_a.padded_shape[0]

    assert input_tensor_a.padded_shape == [batch_size, 1, 384, 4096], "Unsupported input shape"
    assert input_tensor_b.padded_shape == [1, 1, 4096, 1024], "Unsupported input shape"

    program_config = _multicast_config(
        input_tensor_a,
        n=1024 // ttnn.TILE_SIZE,
        per_core_N=3,
        out_subblock_h=6,
        out_subblock_w=1,
        in0_block_w=4,
        per_core_M=12,
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
    batch_size = input_tensor_a.padded_shape[0]

    assert input_tensor_a.padded_shape == [batch_size, 1, 384, 1024], "Unsupported input shape"
    assert input_tensor_b.padded_shape == [1, 1, 1024, 1024], "Unsupported input shape"

    program_config = _multicast_config(
        input_tensor_a,
        n=1024 // ttnn.TILE_SIZE,
        per_core_N=3,
        out_subblock_h=6,
        out_subblock_w=1,
        in0_block_w=4,
        per_core_M=12,
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
    batch_size = input_tensor_a.padded_shape[0]

    assert input_tensor_a.padded_shape == [batch_size, 16, 384, 64], "Unsupported input shape"
    assert input_tensor_b.padded_shape == [batch_size, 16, 64, 384], "Unsupported input shape"

    program_config = None
    if _fits_matmul_configs(input_tensor_a):
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(_matmul_columns(input_tensor_a), batch_size),
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
    batch_size = input_tensor_a.padded_shape[0]

    assert input_tensor_a.padded_shape == [batch_size, 16, 384, 384], "Unsupported input shape"
    assert input_tensor_b.padded_shape == [batch_size, 16, 384, 64], "Unsupported input shape"

    program_config = None
    if _fits_matmul_configs(input_tensor_a):
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(_matmul_columns(input_tensor_a), batch_size),
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
