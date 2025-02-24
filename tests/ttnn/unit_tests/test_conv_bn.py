# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    skip_for_grayskull,
    is_grayskull,
    is_wormhole_b0,
    is_blackhole,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
import torch.nn as nn


def run_conv_l1(
    device,
    math_fidelity,
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
    use_1d_systolic_array,
    config_override,
    dilation=1,
    use_shallow_conv_variant=False,
    transpose_mcast=True,
    enable_auto_formatting=False,
    padded_input_channels=None,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    debug=False,
    groups=1,
    bias=True,
    has_bias=True,
    shard_layout=None,
    auto_shard=False,
    memory_config=None,
    input_mesh_mapper=None,
    weight_mesh_mapper=None,
    output_mesh_composer=None,
):
    has_bias = bias
    if isinstance(device, ttnn.MeshDevice):
        assert input_mesh_mapper is not None, "Expected mesh mapper for input tensor when using device mesh"
        assert weight_mesh_mapper is not None, "Expected mesh mapper for weight tensors when using device mesh"
        assert output_mesh_composer is not None, "Expected mesh composer for output tensor when using device mesh"
        num_devices = len(device.get_device_ids())
        total_batch_size = num_devices * batch_size  # Batch size across all devices
        logger.info(f"Using {num_devices} devices for this test")
    else:
        total_batch_size = batch_size
    torch.manual_seed(0)
    conv_input_shape = [total_batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]

    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    conv = nn.Conv2d(
        input_channels,
        output_channels,
        (filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        groups=groups,
    )
    conv.weight = nn.Parameter(torch.load("conv_weight.pt"))
    conv.bias = None
    bn = nn.BatchNorm2d(num_features=output_channels)
    # bn.weight = nn.Parameter(torch.load("scale.pt"))
    # bn.bias = nn.Parameter(torch.load("shift.pt"))

    torch_out_golden_tensor = conv(torch_input_tensor_nchw)
    torch_out_golden_tensor = bn(torch_out_golden_tensor)
    relu = nn.ReLU()
    torch_out_golden_tensor = relu(torch_out_golden_tensor)

    torch_weight_tensor, torch_bias_tensor = fold_batch_norm2d_into_conv2d(conv, bn)
    torch_bias_tensor = torch_bias_tensor.reshape(1, 1, 1, -1)

    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]
    reader_patterns_cache = {}
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=weight_mesh_mapper,
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor,
            weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
            mesh_mapper=weight_mesh_mapper,
        )
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        ttnn.bfloat16,
        mesh_mapper=input_mesh_mapper,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
    )
    if shard_layout is None and not auto_shard:
        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=shard_layout,
        input_channels_alignment=(
            16 if use_shallow_conv_variant or (input_channels == 16 and input_height == 115) else 32
        ),
        deallocate_activation=deallocate_activation,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=output_layout,
        activation="relu",
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override and not auto_shard:
        conv_config.act_block_h_override = config_override["act_block_h"]
    if config_override and "act_block_w_div" in config_override and not auto_shard:
        conv_config.act_block_w_div = config_override["act_block_w_div"]
    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        return_weights_and_bias=True,
        return_output_dim=True,
    )
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor, mesh_composer=output_mesh_composer)
    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(
        total_batch_size, out_height, out_width, torch_output_tensor.shape[-1]
    )
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()
    # if not fp32_accum:
    #     pcc = 0.985
    # elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
    #     pcc = 0.996
    # else:
    #     pcc = 0.997
    pcc = 0.99
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing
    if memory_config:
        output_memory_config = ttnn.get_memory_config(tt_output_tensor_on_device)
        logger.info(f"Output Memory Config : {output_memory_config}")
        assert output_memory_config == memory_config


"""
Conv Batch norm unit test works good with random input but fails with real input
The real inputs are attached in the issue https://github.com/tenstorrent/tt-metal/issues/18137

"""


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant, bias",
    (
        # (1, 64, 64, 112, 112, 1, 1, 1, 1, 0, 0, 1, 1, 1, True, None, False, False),
        (1, 64, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, 1, True, None, False, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    # [ttnn.bfloat8_b],
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    # [ttnn.bfloat8_b],
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
def test_vovnet_conv(
    device,
    use_program_cache,
    math_fidelity,
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
    dilation_h,
    dilation_w,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    bias,
):
    run_conv_l1(
        device,
        math_fidelity,
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
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        dilation=dilation_h,
        groups=groups,
        bias=bias,
    )
