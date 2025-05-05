# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS
import ttnn
import torch


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (353, 384, 8, 8, WS, None),
        (128, 128, 32, 32, BS, None),
        (16, 16, 256, 256, HS, {"act_block_h": 32}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [True, False],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [False],
)
@pytest.mark.parametrize(
    "filter, padding",
    [
        [3, (1, 2, 2, 3)],
        [1, 0],
        [5, (2, 4, 3, 5)],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_features(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    fp32_accum,
    packer_l1_acc,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and shard_layout == WS:
        pytest.skip("Bug in Width Sharded Row Major Tensor Creation when height%32!=0. #19408")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat16 and packer_l1_acc and fp32_accum:
        pytest.skip("skipping due to pack_untilize_dst issue!")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=True,
        run_twice=True,
    )


SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, slice_type, num_slices, weights_dtype, activations_dtype, kernel, stride, padding, dilation, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        (2, 512,  512,  128,   128,   SliceWidth,    4,  ttnn.bfloat8_b, ttnn.bfloat16, (3, 3), (1, 1), (1, 1), (1, 1), 32 * 8,  ttnn.MathFidelity.LoFi  ),
        (2, 64,   64,   384,   64,    SliceHeight,   6,  ttnn.bfloat8_b, ttnn.bfloat16, (4, 4), (2, 2), (1, 1), (1, 1), 0,       ttnn.MathFidelity.LoFi  ),
        (1, 4,    32,   1024,  1024,  SliceWidth,    4,  ttnn.bfloat8_b, ttnn.bfloat16, (5, 5), (1, 1), (0, 0), (1, 1), 32,      ttnn.MathFidelity.LoFi  ),
        (1, 64,   128,  992,   992,   SliceWidth,   64,  ttnn.bfloat8_b, ttnn.bfloat16, (2, 2), (1, 1), (0, 0), (1, 1), 32 * 4,  ttnn.MathFidelity.LoFi  ),
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, False, False]],
)
def test_conv_dram(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    activations_dtype,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=False,
        transpose_shards=True,
        run_twice=False,
        fast_compare=False,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )


# OFT


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config",
    (
        (256, 256, 159, 159, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, {"act_block_h": 32}),
        (256, 256, 159, 159, ttnn.TensorMemoryLayout.WIDTH_SHARDED, None),
        (256, 256, 159, 159, ttnn.TensorMemoryLayout.BLOCK_SHARDED, None),
        (256, 256, 159, 159, None, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [False],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [False],
)
@pytest.mark.parametrize(
    "filter, stride, padding",
    [
        [3, 1, 1],
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_oft1(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    fp32_accum,
    packer_l1_acc,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and shard_layout == WS:
        pytest.skip("Bug in Width Sharded Row Major Tensor Creation when height%32!=0. #19408")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat16 and packer_l1_acc and fp32_accum:
        pytest.skip("skipping due to pack_untilize_dst issue!")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=True,
        run_twice=True,
    )


# Update Configs
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config, weights_dtype, activations_dtype, fp32_accum, packer_l1_acc, filter, stride, padding",
    (
        (64, 3, 370, 1224, None, None, ttnn.bfloat8_b, ttnn.bfloat16, False, False, 7, 2, 3),  # passed
        (
            64,
            64,
            93,
            306,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            3,
            1,
            1,
        ),  # passed
        (
            128,
            64,
            93,
            306,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            3,
            2,
            1,
        ),  # passed
        (
            128,
            128,
            47,
            153,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            3,
            1,
            1,
        ),  # passed
        (
            128,
            64,
            93,
            306,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            1,
            2,
            0,
        ),  # passed
        (
            256,
            128,
            47,
            153,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            3,
            2,
            1,
        ),  # passed
        (
            256,
            256,
            24,
            77,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            3,
            1,
            1,
        ),  # passed
        (
            256,
            128,
            47,
            153,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            1,
            2,
            0,
        ),  # passed
        (
            512,
            256,
            24,
            77,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            3,
            2,
            1,
        ),  # passed
        (
            512,
            512,
            12,
            39,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            3,
            1,
            1,
        ),  # passed
        (
            512,
            256,
            24,
            77,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            1,
            2,
            0,
        ),  # passed
        (
            256,
            128,
            47,
            153,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            1,
            1,
            0,
        ),  # passed
        (
            256,
            256,
            24,
            77,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            1,
            1,
            0,
        ),  # passed
        (
            256,
            512,
            12,
            39,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            1,
            1,
            0,
        ),  # passed
        (256, 256, 159, 159, None, None, ttnn.bfloat8_b, ttnn.bfloat16, False, False, 3, 1, 1),  # passed
        (
            9,
            256,
            159,
            159,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            {"act_block_h": 32},
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            False,
            False,
            3,
            1,
            1,
        ),  # passed
    ),
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_oft(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    shard_layout,
    config,
    filter,
    stride,
    padding,
    output_layout,
    fp32_accum,
    packer_l1_acc,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and shard_layout == WS:
        pytest.skip("Bug in Width Sharded Row Major Tensor Creation when height%32!=0. #19408")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat16 and packer_l1_acc and fp32_accum:
        pytest.skip("skipping due to pack_untilize_dst issue!")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter,
        filter,
        stride,
        stride,
        padding,
        config,
        shard_layout=shard_layout,
        output_layout=output_layout,
        has_bias=True,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        preprocess_weights_on_device=True,
        run_twice=True,
    )
