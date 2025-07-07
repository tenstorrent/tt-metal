# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn


def prepare_conv_weights_func(
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
    config_override,
    device,
    groups,
    is_owned,
    slice_config=None,
    weights_dtype=None,
    torch_weights_dtype=None,
    enable_kernel_stride_folding=False,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    if batch_size == 20 and (
        output_channels == 64 or (stride_h == 2 and (output_channels == 256 or output_channels == 128))
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    has_bias = False
    inp_shape = (batch_size, input_channels, input_height, input_width)
    conv_weight_shape = (output_channels, input_channels // groups, filter_height, filter_width)
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16)
    torch_input_tensor = torch.randn(inp_shape, dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn((1, 1, 1, output_channels), dtype=torch.bfloat16) if has_bias else None

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(1, 1),
        groups=groups,
    ).permute(0, 2, 3, 1)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor.transpose(-3, -2).transpose(-2, -1), ttnn.bfloat16)

    if is_owned:
        temp_tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
        temp_tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16) if has_bias else None
        tt_weight_tensor = ttnn.zeros(torch_weight_tensor.shape, ttnn.bfloat16)
        tt_bias_tensor = ttnn.zeros(torch_bias_tensor.shape, ttnn.bfloat16) if has_bias else None
        tt_weight_tensor = temp_tt_weight_tensor[:, :, :]
        tt_bias_tensor = temp_tt_bias_tensor[:, :, :] if has_bias else None
    else:
        tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
        tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16) if has_bias else None

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=weights_dtype,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        enable_kernel_stride_folding=enable_kernel_stride_folding,
    )
    compute_config = ttnn.init_device_compute_kernel_config(device.arch())
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    conv_kwargs = {
        "input_layout": ttnn.ROW_MAJOR_LAYOUT,
        "in_channels": input_channels,
        "out_channels": output_channels,
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": (filter_height, filter_width),
        "stride": (stride_h, stride_w),
        "padding": (pad_h, pad_w),
        "dilation": (1, 1),
        "groups": groups,
        "device": device,
        "conv_config": conv_config,
        "slice_config": slice_config,
    }

    tt_input_tensor = ttnn.to_device(tt_input_tensor, device)

    tt_weight_tensor_formatted = ttnn.prepare_conv_weights(
        weight_tensor=tt_weight_tensor,
        weights_format="OIHW",
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        has_bias=has_bias,
        **conv_kwargs,
        input_dtype=ttnn.bfloat16,
    )
    tt_bias_tensor_formatted = (
        ttnn.prepare_conv_bias(
            bias_tensor=tt_bias_tensor,
            input_memory_config=tt_input_tensor.memory_config(),
            **conv_kwargs,
            input_dtype=ttnn.bfloat16,
        )
        if has_bias
        else None
    )

    tt_weight_tensor_formatted = ttnn.to_device(tt_weight_tensor_formatted, device)
    tt_bias_tensor_formatted = ttnn.to_device(tt_bias_tensor_formatted, device) if has_bias else None
    (k := next(iter(conv_kwargs)), conv_kwargs.pop(k))  ##removing 1st element from dict
    tt_output_tensor_on_device = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor_formatted,
        bias_tensor=tt_bias_tensor_formatted,
        **conv_kwargs,
        compute_config=compute_config,
        dtype=ttnn.bfloat16,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]
    torch_output_tensor = torch_output_tensor.reshape(torch_out_golden_tensor.shape)

    pcc = 0.99
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w,  config_override, groups",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, None), HANGS!!
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, {"act_block_h": 256}, 1),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, {"act_block_h": 32}, 1),  Out of Memory!!
        # rn50 layer1
        (8, 64, 64, 56, 1, 3, 1, 1, 1, 1, 0, None, 64),
        (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None, 1),
        (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None, 1),
        # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, None, 1),
        (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, None, 1),
        (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, {"act_block_h": 32}, 1),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, None, 1),
        (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, None, 1),
        (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, None, 1),
        # rn50 layer3
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, None, 1),
        (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, None, 1),
        (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, None, 1),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, None, 1),
        (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, None, 1),
        (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, None, 1),
        # rn50 layer4
        (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, None, 1),
        (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, None, 1),
        (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, None, 1),
        (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, None, 1),
        (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, None, 1),
        (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, None, 1),
        ## small test
        (1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 2, "grid_size": (2, 2)}, 1),
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 4, "grid_size": (2, 4)}, 1),
        # (1, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, None, 1), sliding_window_op_infra/sliding_window.cpp:341: indices_length_last_core <= indices_length_per_core
        (8, 256, 256, 7, 7, 3, 3, 1, 1, 1, 1, None, 1),
        # r50 1x1s2 shapes
        # Fails with packer_l1_acc = True (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, None, 1),  # r50 first bottleneck downsample shape
        (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, None, 1),  # r50 first bottleneck downsample shape
        # Fails with packer_l1_acc = True (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, None, 1),  # r50 second bottleneck downsample shape
        # (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, True, None, 1), - doesnt fit
        (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, None, 1),  # r50 third bottleneck downsample shape
        # (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, True, None, 1), - doesnt fit
        (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, None, 1),  # r50 fourth bottleneck downsample shape
        # (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, True, None, 1), - doesnt fit
        # (20, 128, 256, 56, 56, 1, 1, 2, 2, 0, 0, True, None, 1),  ## L2M1 DS: doesn't fit
        # formerly failing test case in segformer when ntiles_channels not evenly divisible with num_cores_c
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, None, 1),
    ),
)
@pytest.mark.parametrize("is_owned", [True, False], ids=["owned_storage", "borrowed_storage"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2**15}], indirect=True)
def test_prepare_conv_weights(
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
    config_override,
    device,
    groups,
    is_owned,
):
    prepare_conv_weights_func(
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
        config_override,
        device,
        groups,
        is_owned,
    )


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w,  config_override, groups",
    (
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, {"act_block_h": 256}, 1),
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, None, 1),
    ),
)
@pytest.mark.parametrize("weights_dtype", [None, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("torch_weights_dtype", [ttnn.float32])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2**15}], indirect=True)
def test_conv_weights_dtype(
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
    config_override,
    device,
    groups,
    weights_dtype,
    torch_weights_dtype,
):
    prepare_conv_weights_func(
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
        config_override,
        device,
        groups,
        False,
        weights_dtype=weights_dtype,
        torch_weights_dtype=torch_weights_dtype,
    )


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override",
    (
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None),
        (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None),
        (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, None),
        # formerly failing test case in segformer when ntiles_channels not evenly divisible with num_cores_c
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, None),
    ),
)
@pytest.mark.parametrize("has_bias", [True], ids=["has_bias"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2**15}], indirect=True)
def test_prepare_bias(
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
    config_override,
    has_bias,
    device,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")

    if batch_size == 20 and (
        output_channels == 64 or (stride_h == 2 and (output_channels == 256 or output_channels == 128))
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    inp_shape = (batch_size, input_channels, input_height, input_width)
    conv_weight_shape = (output_channels, input_channels, filter_height, filter_width)
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16)
    torch_input_tensor = torch.randn(inp_shape, dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn((1, 1, 1, output_channels), dtype=torch.bfloat16) if has_bias else None

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(1, 1),
        groups=1,
    ).permute(0, 2, 3, 1)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor.transpose(-3, -2).transpose(-2, -1), ttnn.bfloat16)
    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
    tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16) if has_bias else None

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(device.arch())
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    conv_kwargs = {
        "input_layout": ttnn.ROW_MAJOR_LAYOUT,
        "in_channels": input_channels,
        "out_channels": output_channels,
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": (filter_height, filter_width),
        "stride": (stride_h, stride_w),
        "padding": (pad_h, pad_w),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    tt_input_tensor = ttnn.to_device(tt_input_tensor, device)

    tt_bias_tensor_formatted = (
        ttnn.prepare_conv_bias(
            bias_tensor=tt_bias_tensor,
            input_memory_config=tt_input_tensor.memory_config(),
            **conv_kwargs,
            input_dtype=ttnn.bfloat16,
        )
        if has_bias
        else None
    )

    tt_bias_tensor_formatted = ttnn.to_device(tt_bias_tensor_formatted, device) if has_bias else None
    (k := next(iter(conv_kwargs)), conv_kwargs.pop(k))  ##removing 1st element from dict
    tt_output_tensor_on_device = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        bias_tensor=tt_bias_tensor_formatted,
        **conv_kwargs,
        compute_config=compute_config,
        dtype=ttnn.bfloat16,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]
    torch_output_tensor = torch_output_tensor.reshape(torch_out_golden_tensor.shape)

    pcc = 0.99
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, slice_type, num_slices, kernel, stride, padding, dilation, act_block_h_override",
    # fmt: off
    (
        (2, 64,   64,   384,   64,    SliceHeight,   6, (4, 4), (2, 2), (1, 1), (1, 1),  0,       ),
        (1, 32,   32,   1024,  1024,  SliceWidth,    4, (5, 5), (1, 1), (0, 0), (1, 1),  32,      ),
        (1, 64,   128,  992,   992,   SliceWidth,   64, (2, 2), (1, 1), (0, 0), (1, 1),  32 * 4,  ),
    )
    # fmt: on
)
def test_conv_dram(
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
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }
    prepare_conv_weights_func(
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        config,
        device,
        groups=1,
        is_owned=False,
        slice_config=ttnn.Conv2dSliceConfig(
            slice_type=slice_type,
            num_slices=num_slices,
        ),
    )


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w",
    (
        (1, 1024, 3, 224, 224, 16, 16, 16, 16),
        (1, 1024, 3, 224, 224, 32, 32, 32, 32),
        (1, 192, 3, 512, 672, 16, 16, 16, 16),
        (1, 192, 3, 512, 672, 32, 32, 32, 32),
        (1, 768, 3, 384, 512, 32, 32, 32, 32),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 2**15}], indirect=True)
def test_prepare_conv_weights_with_fold(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    device,
):
    pad_h = 0
    pad_w = 0
    groups = 1

    prepare_conv_weights_func(
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
        None,
        device,
        groups,
        is_owned=False,
        enable_kernel_stride_folding=True,
    )
