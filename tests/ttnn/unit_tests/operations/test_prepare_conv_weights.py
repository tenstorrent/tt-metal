# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    is_wormhole_b0,
    skip_for_grayskull,
    is_grayskull,
    is_wormhole_b0,
    is_x2_harvested,
    is_blackhole,
    skip_for_blackhole,
    is_blackhole,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, None), HANGS!!
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, {"act_block_h": 256}),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, {"act_block_h": 32}),  Out of Memory!!
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, None),
        (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, None),
        (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, None),
        # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, None),
        (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, None),
        (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, {"act_block_h": 32}),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True, None),
        (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True, None),
        (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True, None),
        # rn50 layer3
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False, None),
        (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False, None),
        (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False, None),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False, None),
        (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False, None),
        (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False, None),
        # rn50 layer4
        (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False, None),
        (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False, None),
        (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False, None),
        (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False, None),
        (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False, None),
        (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False, None),
        ## small test
        (1, 64, 64, 8, 8, 3, 3, 1, 1, 1, 1, False, {"num_cores_nhw": 2, "grid_size": (2, 2)}),
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, False, {"num_cores_nhw": 4, "grid_size": (2, 4)}),
        # (1, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, False, None), sliding_window_op_infra/sliding_window.cpp:341: indices_length_last_core <= indices_length_per_core
        (8, 256, 256, 7, 7, 3, 3, 1, 1, 1, 1, False, None),
        # r50 1x1s2 shapes
        # Fails with packer_l1_acc = True (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, False, None),  # r50 first bottleneck downsample shape
        (20, 256, 64, 56, 56, 1, 1, 2, 2, 0, 0, True, None),  # r50 first bottleneck downsample shape
        # Fails with packer_l1_acc = True (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, False, None),  # r50 second bottleneck downsample shape
        # (20, 512, 256, 56, 56, 1, 1, 2, 2, 0, 0, True, None), - doesnt fit
        (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, False, None),  # r50 third bottleneck downsample shape
        # (20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, True, None), - doesnt fit
        (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, False, None),  # r50 fourth bottleneck downsample shape
        # (20, 2048, 1024, 14, 14, 1, 1, 2, 2, 0, 0, True, None), - doesnt fit
        # (20, 128, 256, 56, 56, 1, 1, 2, 2, 0, 0, True, None),  ## L2M1 DS: doesn't fit
    ),
)
@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize("has_bias", [True, False], ids=["has_bias", "no_bias"])
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
    use_1d_systolic_array,
    packer_l1_acc,
    config_override,
    has_bias,
    device,
):
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
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        input_channels_alignment=(16 if input_channels == 16 and input_height == 115 else 32),
        packer_l1_accum_enabled=packer_l1_acc,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
    )

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

    tt_weight_tensor_formatted = ttnn.prepare_conv_weights_for_ttnn(
        weight_tensor=tt_weight_tensor, weights_format="OIHW", **conv_kwargs
    )
    tt_bias_tensor_formatted = (
        ttnn.prepare_conv_bias_for_ttnn(bias_tensor=tt_bias_tensor, **conv_kwargs) if has_bias else None
    )

    tt_weight_tensor_formatted = ttnn.to_device(tt_weight_tensor_formatted, device)
    tt_bias_tensor_formatted = ttnn.to_device(tt_bias_tensor_formatted, device) if has_bias else None

    tt_output_tensor_on_device = ttnn.conv2d_device_weights(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor_formatted,
        bias_tensor=tt_bias_tensor_formatted,
        **conv_kwargs,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]
    torch_output_tensor = torch_output_tensor.reshape(torch_out_golden_tensor.shape)
    #
    pcc = 0.99
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing
