# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Original shapes and parameter combinations of conv unit tests that the unit tests run smaller or in a subset.
Each test delegates to the unit test so the assertions can't drift apart.
"""

import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import torch_tensor_map, HS, WS, BS
from tests.ttnn.unit_tests.operations.conv.test_conv2d import (
    test_conv_features as run_conv_features,
    test_conv_dram as run_conv_dram,
)
from tests.ttnn.unit_tests.operations.conv.test_conv_transpose2d import test_convt2d_dram as run_convt2d_dram
from tests.ttnn.unit_tests.operations.conv.test_prepare_conv_weights import (
    test_conv_dram as run_prepare_weights_conv_dram,
)

SliceWidth = ttnn.Conv2dDRAMSliceWidth


# The unit test pairs fp32_accum with packer_l1_acc and runs the height-sharded case at 128x128:
# here are the mixed pairs and the original 256x256 case.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, shard_layout, config, fp32_accum, packer_l1_acc",
    (
        (353, 384, 8, 8, WS, None, True, False),
        (353, 384, 8, 8, WS, None, False, True),
        (128, 128, 32, 32, BS, None, True, False),
        (128, 128, 32, 32, BS, None, False, True),
        (16, 16, 256, 256, HS, {"act_block_h": 32}, True, True),
        (16, 16, 256, 256, HS, {"act_block_h": 32}, True, False),
        (16, 16, 256, 256, HS, {"act_block_h": 32}, False, True),
        (16, 16, 256, 256, HS, {"act_block_h": 32}, False, False),
    ),
)
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("filter, padding", [[3, (1, 2, 2, 3)], [5, (2, 4, 3, 5)]])
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv_features_large(
    device,
    torch_tensor_map,
    math_fidelity,
    output_dtype,
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
    input_dtype,
):
    run_conv_features(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
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
        input_dtype,
    )


@pytest.mark.parametrize(
    "input_layout, dtype",
    [[ttnn.TILE_LAYOUT, ttnn.bfloat8_b], [ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, weights_dtype, kernel, stride, padding, dilation, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        (1, 400,  528,  192,   192,    ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1),      0,  ttnn.MathFidelity.HiFi4 ),
        (2, 512,  512,  128,   128,    ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 32 * 8,  ttnn.MathFidelity.LoFi  ),
        (1, 4,    32,   1024,  1024,   ttnn.bfloat8_b, (5, 5), (1, 1), (0, 0), (1, 1),     32,  ttnn.MathFidelity.LoFi  ),
        (1, 2904, 2904,   48,    48,   ttnn.bfloat8_b, (3, 3), (1, 1), (0, 0), (1, 1),     32,  ttnn.MathFidelity.HiFi4 ),
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, True, False]],
)
def test_conv_dram_large(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    dtype,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    input_layout,
    packer_l1_acc,
):
    run_conv_dram(
        device,
        torch_tensor_map,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        has_bias,
        weights_dtype,
        dtype,
        kernel,
        stride,
        padding,
        dilation,
        act_block_h_override,
        math_fidelity,
        fp32_accum,
        input_layout,
        packer_l1_acc,
        shard_layout=None,
        dram_slice=None,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, config, shard_layout, num_slices, slice_type",
    ((1, 512, 512, 64, 64, 3, 3, 1, 1, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 4, SliceWidth),),
)
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "activations_dtype, layout",
    [
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize("preprocess_weights", [False])
@pytest.mark.parametrize("mirror_kernel", [False])
def test_convt2d_dram_large(
    device,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    out_pad_h,
    out_pad_w,
    config,
    shard_layout,
    layout,
    mirror_kernel,
    preprocess_weights,
    num_slices,
    slice_type,
):
    run_convt2d_dram(
        device,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_pad_h,
        out_pad_w,
        config,
        shard_layout,
        layout,
        mirror_kernel,
        preprocess_weights,
        num_slices,
        slice_type,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, slice_type, num_slices, kernel, stride, padding, dilation, act_block_h_override",
    ((1, 32, 32, 1024, 1024, SliceWidth, 4, (5, 5), (1, 1), (0, 0), (1, 1), 32),),
)
def test_prepare_weights_conv_dram_large(
    device,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    slice_type,
    num_slices,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
):
    run_prepare_weights_conv_dram(
        device,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        slice_type,
        num_slices,
        kernel,
        stride,
        padding,
        dilation,
        act_block_h_override,
    )
