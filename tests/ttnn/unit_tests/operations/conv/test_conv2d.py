# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS
import ttnn
import torch
from models.utility_functions import skip_for_blackhole


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
    [None, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "input_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [True, False],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [True, False],
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
    math_fidelity,
    output_dtype,
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
    input_dtype,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and output_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        output_dtype,
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
        run_twice=True,
        input_layout=ttnn.TILE_LAYOUT if input_dtype == ttnn.bfloat8_b else None,
        input_dtype=input_dtype,
    )


SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


# @skip_for_blackhole("Not fully tested on Blackhole")
@pytest.mark.parametrize(
    "input_layout, dtype",
    [
        [ttnn.TILE_LAYOUT, ttnn.bfloat8_b],
    ],  # , [ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, slice_type, num_slices, weights_dtype, kernel, stride, padding, dilation, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        # (2,  13,   31,  313,    71,   SliceWidth,   16,  ttnn.bfloat8_b, (5, 5), (1, 1), (2, 2), (2, 2), 32 * 4,  ttnn.MathFidelity.LoFi  ),
        # (2,  63,  129,  981,    39,   SliceHeight,  16,  ttnn.bfloat8_b, (3, 3), (2, 2), (2, 2), (1, 1),      0,  ttnn.MathFidelity.LoFi  ),
        # (2, 512,  512,  128,   128,   SliceWidth,    4,  ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 32 * 8,  ttnn.MathFidelity.LoFi  ),
        # (2, 64,   64,   384,   64,    SliceHeight,   6,  ttnn.bfloat8_b, (4, 4), (2, 2), (1, 1), (1, 1), 0,       ttnn.MathFidelity.LoFi  ),
        # (1, 4,    32,   1024,  1024,  SliceWidth,    4,  ttnn.bfloat8_b, (5, 5), (1, 1), (0, 0), (1, 1), 32,      ttnn.MathFidelity.LoFi  ),
        # (1, 64,   128,  992,   992,   SliceWidth,   64,  ttnn.bfloat8_b, (2, 2), (1, 1), (0, 0), (1, 1), 32 * 4,  ttnn.MathFidelity.LoFi  ),
        # (1, 2904, 2904,  48,    48,   SliceWidth,   4,  ttnn.bfloat8_b, (3, 3), (1, 1), (0, 0), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ),
        # (1, 2944, 2944,  48,    48,   SliceWidth,   4,  ttnn.bfloat8_b,  (3, 3), (1, 1), (0, 0), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ),
        (1, 2048, 256,  32,    64,   SliceHeight,   2,  ttnn.bfloat8_b,  (3, 3), (1, 1), (12, 12), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ), #Panotpic
        # (1, 2048, 256,  32,    64,   SliceHeight,   2,  ttnn.bfloat8_b,  (3, 3), (1, 1), (18, 18), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ), #Panotpic OOM for SliceHeight(2,4,8,16,32 slices), hangs for SliceWidth
        (1, 2048, 256,  32,    64,   SliceWidth,   2,  ttnn.bfloat8_b,  (3, 3), (1, 1), (6, 6), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ), #Panotpic OOM for SliceHeight (2,4,8,16,32 slices)
        (1, 256, 256,  128,    256,   SliceHeight,   2,  ttnn.bfloat8_b,  (3, 3), (1, 1), (1, 1), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ), #Panotpic
        (1, 256, 512,  128,    256,   SliceHeight,   4,  ttnn.bfloat8_b,  (1, 1), (2, 2), (1, 1), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ), #Panotpic
        (1, 288, 256,  128,    256,   SliceHeight,   4,  ttnn.bfloat8_b,  (3, 3), (1, 1), (1, 1), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ), #Panotpic
        (1, 64, 128,  256,    512,   SliceHeight,   2,  ttnn.bfloat8_b,  (3, 3), (1, 1), (1, 1), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ), #Panotpic
        (1, 64, 64,  256,    512,   SliceHeight,   2,  ttnn.bfloat8_b,  (3, 3), (1, 1), (1, 1), (1, 1), 32,  ttnn.MathFidelity.HiFi4  ), #Panotpic
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, True, False]],
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
    dtype,
    slice_type,
    num_slices,
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
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        dtype,
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
        input_dtype=dtype,
        input_layout=input_layout,
        output_layout=input_layout,
        run_twice=True,
        fast_compare=True,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )


@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, has_bias",
    [
        (1, 1024, 2048, 32, 64, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 1024, 256, 32, 64, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 1024, 512, 32, 64, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 128, 128, 128, 256, 3, 3, 1, 1, 1, 1, 1, False),
        (1, 128, 128, 128, 256, 3, 3, 2, 2, 1, 1, 1, False),
        (1, 128, 256, 128, 256, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 128, 32, 128, 256, 3, 3, 1, 1, 1, 1, 1, False),
        (1, 128, 64, 128, 256, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 128, 128, 64, 128, 3, 3, 1, 1, 1, 1, 1, False),
        (1, 128, 512, 64, 128, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 1280, 256, 32, 64, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 160, 128, 128, 256, 3, 3, 1, 1, 1, 1, 1, False),
        (1, 2048, 256, 1, 1, 1, 1, 1, 1, 0, 0, 1, True),
        (1, 2048, 256, 32, 64, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 2048, 512, 32, 64, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 256, 128, 128, 256, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 256, 19, 128, 256, 1, 1, 1, 1, 0, 0, 1, True),
        (1, 256, 32, 128, 256, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 256, 64, 128, 256, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 256, 1024, 32, 64, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 256, 256, 32, 64, 3, 3, 1, 1, 1, 1, 1, False),
        (1, 256, 256, 64, 128, 3, 3, 2, 2, 1, 1, 1, False),
        (1, 256, 256, 64, 128, 3, 3, 1, 1, 1, 1, 1, False),
        (1, 3, 64, 512, 1024, 3, 3, 2, 2, 1, 1, 1, False),
        (1, 32, 1, 128, 256, 1, 1, 1, 1, 0, 0, 1, True),
        (1, 32, 2, 128, 256, 1, 1, 1, 1, 0, 0, 1, True),
        (1, 320, 128, 64, 128, 3, 3, 1, 1, 1, 1, 1, False),
        (1, 320, 256, 64, 128, 3, 3, 1, 1, 1, 1, 1, False),
        (1, 512, 2048, 32, 64, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 512, 512, 32, 64, 3, 3, 1, 1, 2, 2, 1, False),
        (1, 512, 512, 32, 64, 3, 3, 1, 1, 4, 4, 1, False),
        (1, 512, 512, 32, 64, 3, 3, 1, 1, 8, 8, 1, False),
        (1, 512, 1024, 64, 128, 1, 1, 2, 2, 0, 0, 1, False),
        (1, 512, 128, 64, 128, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 512, 256, 64, 128, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 512, 64, 64, 128, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 64, 256, 128, 256, 1, 1, 1, 1, 0, 0, 1, False),
        (1, 64, 64, 128, 256, 3, 3, 1, 1, 1, 1, 1, False),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_conv2d_panoptic(
    device,
    batch_size,
    input_channels,
    output_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    groups,
    torch_tensor_map,
    has_bias,
):
    compute_grid = device.compute_with_storage_grid_size()
    print(f"compute_grid: {compute_grid.x}x{compute_grid.y}")
    if compute_grid.x != 5 or compute_grid.y != 4:
        pytest.skip(f"Test requires compute grid size of 5x4, but got {compute_grid.x}x{compute_grid.y}")

    run_conv(
        device=device,
        config_override=None,
        torch_tensor_map=torch_tensor_map,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        output_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        padding=(pad_h, pad_w),
        groups=groups,
        has_bias=has_bias,
    )
