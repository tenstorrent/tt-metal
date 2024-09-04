# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_grayskull,
    is_grayskull,
    is_wormhole_b0,
    is_x2_harvested,
    is_blackhole,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn


def _nearest_32(x):
    return math.ceil(x / 32) * 32


from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores

# def plot_diff(vals, fid, nsticks, stick_len):
#     import matplotlib.pyplot as plt

#     plt.clf()
#     plt.figure(figsize=(100, 50))
#     plt.xticks(torch.arange(0, stick_len) + 0.5, range(0, stick_len))
#     plt.yticks(torch.arange(0, nsticks) + 0.5, range(0, nsticks))
#     # plt.grid()
#     bool_vals = vals > 0
#     plt.imshow(bool_vals, interpolation="none", vmin=0, vmax=1, cmap="Blues")
#     plt.savefig(f"diff_core_{fid}.png", bbox_inches="tight", pad_inches=0.1)
#     plt.close()


def run_conv(
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
    has_bias=True,
    shard_layout=None,
):
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    # torch_input_tensor_nchw = torch.ones(conv_input_shape, dtype=torch.bfloat16).float()
    # torch_input_tensor_nchw = torch.tensor(range(input_height * input_width)).reshape([1,1,input_height,input_width]).float()
    # torch_input_tensor_nchw = torch_input_tensor_nchw.broadcast_to(conv_input_shape).float()

    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    # torch_weight_tensor = torch.ones(conv_weight_shape, dtype=torch.bfloat16).float()

    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation, dilation),
        groups=groups,
    )
    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]

    reader_patterns_cache = {}

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    if shard_layout is None:
        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        shard_layout=shard_layout,
        input_channels_alignment=(
            16 if use_shallow_conv_variant or (input_channels == 16 and input_height == 115) else 32
        ),
        deallocate_activation=deallocate_activation,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]
        print("Setting Act Block H to ", conv_config.act_block_h_override)

    if config_override and "act_block_w_div" in config_override:
        conv_config.act_block_w_div = config_override["act_block_w_div"]
        print("Setting Act Block H to ", conv_config.act_block_h_override)
    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
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
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()

    if not fp32_accum:
        pcc = 0.99
    elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    print(pcc, pcc_msg)
    assert passing


def run_conv_with_split(
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
    split_factor=2,
    fp32_accum=False,
    packer_l1_acc=False,
):
    torch.manual_seed(0)
    assert input_channels % split_factor == 0
    split_input_channels = input_channels // split_factor
    full_conv_input_shape = [batch_size, input_channels, input_height, input_width]
    full_conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    torch_input_tensor_nchw = torch.randn(full_conv_input_shape, dtype=torch.bfloat16).float()
    torch_weight_tensor = torch.randn(full_conv_weight_shape, dtype=torch.bfloat16).float()
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
    torch_bias_zeroes_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
    )

    split_input_tensors = torch.split(torch_input_tensor_nchw, split_input_channels, 1)
    split_weight_tensors = torch.split(torch_weight_tensor, split_input_channels, 1)
    # conv1_output_tensor = torch.nn.functional.conv2d(
    #                     split_input_tensors[0],
    #                     split_weight_tensors[0],
    #                     bias=torch_bias_tensor.reshape(-1),
    #                     stride=(stride_h, stride_w),
    #                     padding=(pad_h, pad_w),
    #                 )
    # conv2_output_tensor = torch.nn.functional.conv2d(
    #                     split_input_tensors[1],
    #                     split_weight_tensors[1],
    #                     stride=(stride_h, stride_w),
    #                     padding=(pad_h, pad_w),
    #                 )
    # torch_output_tensor = torch.add(conv1_output_tensor, conv2_output_tensor)

    torch_input1_tensor = torch.permute(split_input_tensors[0], (0, 2, 3, 1))
    torch_input2_tensor = torch.permute(split_input_tensors[1], (0, 2, 3, 1))
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if use_1d_systolic_array
        else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        # input_channels_alignment=(16 if use_shallow_conv_variant else 32),
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]
        print("Setting Act Block H to ", conv_config.act_block_h_override)
    torch_output_tensor = None
    for i in range(split_factor):
        tt_weight_tensor = ttnn.from_torch(
            split_weight_tensors[i], weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
        if i == 0:
            tt_bias_tensor = ttnn.from_torch(
                torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
            )
        else:
            tt_bias_tensor = ttnn.from_torch(
                torch_bias_zeroes_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
            )
        torch_input_tensor = torch.permute(split_input_tensors[i], (0, 2, 3, 1))
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        # tt_input_tensor_on_device = convs[i].copy_input_to_device(tt_input_tensor)
        # tt_output_tensor_on_device = convs[i](tt_input_tensor_on_device)
        [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor,
            in_channels=split_input_channels,
            out_channels=output_channels,
            device=device,
            bias_tensor=tt_bias_tensor,
            kernel_size=(filter_height, filter_width),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            conv_op_cache=reader_patterns_cache,
        )
        tt_conv_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
        torch_conv_output_tensor = ttnn.to_torch(tt_conv_output_tensor)
        print(f"Output shape : {batch_size} {out_height} {out_width} {output_channels}")
        torch_conv_output_tensor = torch_conv_output_tensor.reshape(batch_size, out_height, out_width, output_channels)

        # torch_output_tensor is in row major layout and NHWC shape
        # NHWC to NCHW
        torch_conv_output_tensor = torch.permute(torch_conv_output_tensor, (0, 3, 1, 2))
        if i == 0:
            torch_output_tensor = torch_conv_output_tensor
        else:
            torch_output_tensor = torch.add(torch_output_tensor, torch_conv_output_tensor)
        print("Split output shapes ", torch_output_tensor.shape, torch_conv_output_tensor.shape)

    if math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998
    assert_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, pad_h, pad_w, act_block_w_div",
    (
        (128, 256, 8, 8, 3, 3, 1, 1, 1),
        (256, 2048, 8, 8, 3, 3, 1, 1, 8),
        (512, 2048, 16, 16, 3, 3, 1, 1, 4),
        (768, 768, 8, 8, 3, 3, 1, 1, 1),
        (768, 768, 16, 16, 3, 3, 1, 1, 1),
        (1280, 1280, 16, 16, 3, 3, 1, 1, 1),
        (1280, 2560, 16, 16, 3, 3, 1, 1, 2),
    ),
)
@pytest.mark.parametrize(
    "has_bias",
    [True],
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
def test_conv_ws(
    device,
    use_program_cache,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    pad_h,
    pad_w,
    act_block_w_div,
    stride,
    has_bias,
    weights_dtype,
    activations_dtype,
):
    if is_grayskull():
        if input_channels >= 2048:
            pytest.skip("Skipping on grayskull due to insufficient L1")
        if input_channels >= 768 and input_height >= 10:
            pytest.skip("Skipping on grayskull due to insufficient L1")

    stride_h = stride
    stride_w = stride
    batch_size = 2
    fp32_accum = False
    packer_l1_acc = False
    deallocate_activation = False
    debug = False
    groups = 1

    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]

    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor_nchw = torch_input_tensor_nchw.broadcast_to(conv_input_shape).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()

    tt_bias_tensor = None
    torch_bias_tensor = None
    if has_bias:
        torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() * 50
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
        torch_bias_tensor = torch_bias_tensor.reshape(-1)
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        groups=groups,
    )
    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]

    reader_patterns_cache = {}
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.bfloat16)

    tt_input_tensor = ttnn.reshape(tt_input_tensor, [1, 1, input_height * input_width * batch_size, input_channels])

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        input_channels_alignment=32,
        deallocate_activation=deallocate_activation,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        reshard_if_not_optimal=True,
        act_block_w_div=act_block_w_div,
    )
    [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, output_channels)

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()

    pcc = 0.94
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    if not passing:
        logger.error("Fails with PCC ", pcc_msg)
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        # mlp sub_module
        (1, 3, 32, 512, 512, 7, 7, 4, 4, 3, 3, 1, True, {"act_block_h": 64}, False),  # ncrisc build failed
        # efficient selfattention sub_module
        (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, True, None, False),  # ncrisc build failed, Two times called in model
        (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, True, None, False),  # ncrisc build failed, Two times called in model
        (1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, True, None, False),  # pass , Two times called in model
        # dwconv sub_module
        (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 128, True, {"act_block_h": 64}, False),
        (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 256, True, None, False),  # pass , Two times called in model
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 640, False, {"act_block_h": 32}, False),
        # (1,1024, 1024, 16, 16, 3, 3, 1, 1, 1, 1, 1024, False, None, False), #Switch to Width Sharding
        # decode_head sub_module
        # (1,1024, 256, 128, 128, 1, 1, 1, 1, 0, 0, 1, False, {"act_block_h": 32}, False), #pass for activation_dtype=bf8 but fails for bf16
        (1, 256, 150, 128, 128, 1, 1, 1, 1, 0, 0, 1, True, None, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@skip_for_grayskull()
def test_conv_for_segformer_512x512(
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
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
):
    run_conv(
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
        groups=groups,
        output_layout=output_layout,
        has_bias=False,
    )


@skip_for_wormhole_b0("This is test is for Grayskull only. Skipping")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True), act_block_h_ntiles % 2 == 0
        # rn50 layer1
        (64, 64, 56, 56, 1, 1, 1, 1, 0, 0, True, None),
        (64, 64, 56, 56, 1, 1, 2, 2, 0, 0, True, None),
        (64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, None),
        # rn50 layer2
        (128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, None),
        (128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True, None),
        # rn50 layer3
        (256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False, None),
        (256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False, None),
        # rn50 layer4
        (512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False, None),
        (512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False, None),
        ## 1x1s2
        # (256, 256, 28, 28, 1, 1, 2, 2, 0, 0, True, {"num_cores_nhw": 98}),
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    [8, 16, 20],
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
def test_resnet50_conv_gs(
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
    use_1d_systolic_array,
    config_override,
):
    if is_blackhole():
        pytest.skip("This test is for Grayskull only")

    if batch_size > 8 and (activations_dtype != ttnn.bfloat8_b or weights_dtype != ttnn.bfloat8_b):
        pytest.skip("Batch > 8 must be run fully bfp8")
    if batch_size == 20 and input_channels >= 128 and filter_width > 1:
        pytest.skip("L1 Allocation error")
    if (
        activations_dtype == ttnn.bfloat16
        and batch_size == 20
        and (
            output_channels == 64
            or (
                stride_h == 2
                and (output_channels == 256 or (output_channels == 128 and weights_dtype == ttnn.bfloat16))
            )
        )
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    run_conv(
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
        config_override=config_override,
        use_shallow_conv_variant=input_channels == 16,
        padded_input_channels=16 if input_channels == 16 else None,
        debug=not (batch_size == 20 and input_height == 115),
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
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
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize("has_bias", [True, False], ids=["with_bias", "no_bias"])
def test_resnet50_conv_wh(
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
    use_1d_systolic_array,
    config_override,
    packer_l1_acc,
    has_bias,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6992: Statically allocated circular buffers in program clash with L1 buffers on core range")
    if batch_size > 8 and (activations_dtype != ttnn.bfloat8_b or weights_dtype != ttnn.bfloat8_b):
        pytest.skip("Batch > 8 must be run fully bfp8")

    if (
        (
            activations_dtype == ttnn.bfloat16
            and batch_size == 20
            and (
                output_channels == 64
                or (
                    stride_h == 2
                    and (output_channels == 256 or (output_channels == 128 and weights_dtype == ttnn.bfloat16))
                )
            )
        )
        # packer l1 acc has separate buffers when interm != output df, cannot fit into L1
        or (batch_size == 20 and activations_dtype == ttnn.bfloat8_b and packer_l1_acc and input_height >= 64)
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    use_shallow_conv_variant = (input_channels == 16) and device.arch() != ttnn.device.Arch.WORMHOLE_B0
    run_conv(
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
        config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        packer_l1_acc=packer_l1_acc,
        fp32_accum=False,
        has_bias=has_bias,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, None),
        # (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, {"act_block_h": 32}),
        # (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, {"act_block_h": 32}),
        # rn50 layer1
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, None),
        # (16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, None),
        # (20, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True, None),
        # # rn50 layer2
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, None),
        # (16, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, None),
        # (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True, {"act_block_h": 32}),
        (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True, None),
        # (16, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True, None),
        # (20, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True, None),
        # # rn50 layer3
        # (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False, None),
        # (16, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False, None),
        # (20, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False, None),
        # (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False, None),
        # (16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False, None),
        # (20, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False, None),
        # # rn50 layer4
        # (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False, None),
        # (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False, None),
        # (20, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False, None),
        # (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False, None),
        # (16, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False, None),
        # (20, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.float32, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
def test_resnet50_conv_wh_fp32(
    device,
    use_program_cache,
    math_fidelity,
    fp32_accum,
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
    packer_l1_acc,
):
    if batch_size > 8 and (activations_dtype != ttnn.bfloat8_b or weights_dtype != ttnn.bfloat8_b):
        pytest.skip("Batch > 8 must be run fully bfp8")

    if (
        activations_dtype == ttnn.bfloat16
        and batch_size == 20
        and (
            output_channels == 64
            or (
                stride_h == 2
                and (output_channels == 256 or (output_channels == 128 and weights_dtype == ttnn.bfloat16))
            )
        )
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    use_shallow_conv_variant = (input_channels == 16) and device.arch() != ttnn.device.Arch.WORMHOLE_B0
    run_conv(
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
        config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override",
    (
        # sd convs with HxW=32x32
        # (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, False, None),
        # (1, 320, 320, 32, 32, 3, 3, 2, 2, 1, 1, False, None),
        # (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, False, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, False, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, False, None), # bfloat16 activations doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, None), # slighlty low pcc with 0.99689. bfloat16 weights doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 2, 2, 1, 1, False, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 4, 4, 3, 3, 1, 1, 1, 1, False, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None), # slightly low pcc with 0.99698. bfloat16 weights doesnt fit
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, None), # doesnt fit at all.. for all data types
        # sd convs with HxW=64x64 with batch size = 1
        (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, True, None),
        (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, False, None),
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),  #
        (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 weights doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit.
        (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        # (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, False, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (1, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, False, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # sd convs with HxW=64x64 with batch size=2
        # (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, True, None), Hangs on WH
        (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
        (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, False, None),  # fits with bfloat8_b
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        # (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
        # (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, False, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, False, None), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}), IndexError: vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)
        # 1x1 conv
        (2, 320, 960, 64, 64, 1, 1, 1, 1, 0, 0, True, None),
        # Small conv
        # (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, True, None),  ## batch = 1 is currently not supported
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("enable_auto_formatting", [True, False])
def test_sd_conv(
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
    use_1d_systolic_array,
    config_override,
    enable_auto_formatting,
):
    if filter_height > 1 and (input_channels > 1280 or (input_channels > 640 and input_height > 16)):
        if enable_auto_formatting:
            pytest.skip("Not running split SD conv with auto formatting")
        run_conv_with_split(
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
            split_factor=3 if input_channels == 1920 else 2,
        )
    else:
        run_conv(
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
            use_shallow_conv_variant=(input_channels == 16),
            enable_auto_formatting=enable_auto_formatting,
            padded_input_channels=16 if input_channels == 16 else None,
        )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override",
    (
        # sd convs with HxW=32x32
        # (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, False, None),
        # (1, 320, 320, 32, 32, 3, 3, 2, 2, 1, 1, False, None),
        # (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, False, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, False, None),
        # (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, False, None), # bfloat16 activations doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, None), # slighlty low pcc with 0.99689. bfloat16 weights doesnt fit
        # (1, 1280, 1280, 8, 8, 3, 3, 2, 2, 1, 1, False, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 4, 4, 3, 3, 1, 1, 1, 1, False, None), #fails to parallelize with sharding
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None), # slightly low pcc with 0.99698. bfloat16 weights doesnt fit
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, None), # doesnt fit at all.. for all data types
        # sd convs with HxW=64x64 with batch size = 1
        # (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, True, None),
        # (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        # (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, False, None),
        # (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),  #
        # (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit
        # (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit.
        # (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 weights doesnt fit
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, None),
        # (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        # (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, False, None),
        # (1, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, False, None),
        # # sd convs with HxW=64x64 with batch size=2
        (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, True, None),
        (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
        (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, False, None),  # fits with bfloat8_b
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        # (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}), L1 Allocation Error
        (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, False, None),
        (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, False, None),
        (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        # 1x1 conv
        (2, 320, 960, 64, 64, 1, 1, 1, 1, 0, 0, True, None),
        # Small conv
        # (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, True, None), fails
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("enable_auto_formatting", [True, False])
def test_sd_conv_wh(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    fp32_accum,
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
    enable_auto_formatting,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")

    # Skip test cases raising OOM, but do not affect the SD e2e test
    if (
        (input_channels == 320 and config_override == None and activations_dtype == ttnn.bfloat16)
        or (input_channels == 960 and config_override == None and fp32_accum == True)
        or (
            output_channels == 1280
            and input_height == 32
            and activations_dtype == ttnn.bfloat16
            and weights_dtype == ttnn.bfloat16
            and enable_auto_formatting == False
        )
    ):
        pytest.skip("Skip the test cases raising OOM but not affecting e2e test")

    if filter_height > 1 and (input_channels > 1280 or (input_channels > 640 and input_height > 16)):
        if enable_auto_formatting:
            pytest.skip("Not running split SD conv with auto formatting")
        run_conv_with_split(
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
            split_factor=3 if input_channels == 1920 else 2,
            fp32_accum=fp32_accum,
            packer_l1_acc=True,
        )
    else:
        run_conv(
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
            use_shallow_conv_variant=(input_channels == 16),
            transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
            enable_auto_formatting=enable_auto_formatting,
            padded_input_channels=16 if input_channels == 16 else None,
            fp32_accum=fp32_accum,
            packer_l1_acc=True,
        )


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        # unet convs with batch size 2
        # unique convs in unet (complete list)
        (2, 16, 3, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 64}, True),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 64}, True),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, True, None, True),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, True, None, True),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 64, 64, 66, 10, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 64, 264, 40, 3, 3, 1, 1, 1, 1, True, None, True),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, True, None, True),
        (
            2,
            16,
            48,
            528,
            80,
            3,
            3,
            1,
            1,
            1,
            1,
            True,
            {"act_block_h": 32},
            False,
        ),  # fails. mismatch. It passes when input_channels=64. Probably an issue with padding when input_channels % 32 != 0.
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 22 * 32}, False),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 22 * 32}, False),
        (2, 1, 16, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 22 * 32}, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_unet_conv(
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
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if is_blackhole():
        pytest.skip("This test is for Grayskull only")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
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
        padded_input_channels=16 if input_channels == 3 else None,
        output_layout=output_layout,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        # unet convs with batch size 2
        # unique convs in unet (complete list)
        (2, 16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 5 * 32}, False),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 5 * 32}, False),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 64, 64, 66, 10, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 64, 264, 40, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, True, None, False),
        (
            2,
            16,
            48,
            528,
            80,
            3,
            3,
            1,
            1,
            1,
            1,
            True,
            {"act_block_h": 32},
            False,
        ),  # fails. mismatch. It passes when input_channels=64. Probably an issue with padding when input_channels % 32 != 0.
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, True, None, False),
        (2, 16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 5 * 32}, False),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 5 * 32}, False),
        # (2, 1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, True, {"act_block_h": 5 * 32}, False) # Enable when issue #11490 resolved
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
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_unet_conv_wh(
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
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
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
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override",
    (
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 1}),
        (1, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, {"act_reshard_num_cores_nhw": 1}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 4}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8}),
        (1, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8, "num_cores_nhw": 4}),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 8, "num_cores_nhw": 4}),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"act_reshard_num_cores_nhw": 4, "num_cores_nhw": 8}),
    ),
)
@pytest.mark.parametrize("use_1d_systolic_array", [False, True])
def test_halo_reshard_conv(
    device,
    use_program_cache,
    use_1d_systolic_array,
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
):
    math_fidelity = ttnn.MathFidelity.HiFi4
    activations_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b

    run_conv(
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
    )


@pytest.mark.skip("New API needs to be tested")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, config_override, xfail",
    (
        (1, 128, 128, 17, 17, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 4}, False),
        (1, 128, 128, 17, 17, 3, 3, 2, 2, 1, 1, {"num_cores_nhw": 2}, False),
        (2, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 3}, False),
        (2, 64, 64, 23, 23, 3, 3, 2, 2, 1, 1, {"num_cores_nhw": 3}, False),
        (1, 64, 64, 23, 23, 3, 3, 1, 1, 1, 1, {"num_cores_nhw": 10}, True),
    ),
)
@pytest.mark.parametrize("use_1d_systolic_array", [False, True])
def test_conv_core_nondivis(
    device,
    use_program_cache,
    use_1d_systolic_array,
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
    xfail,
):
    if xfail:
        pytest.xfail()

    math_fidelity = ttnn.MathFidelity.HiFi4
    activations_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b

    run_conv(
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
    )


# The following test takes various shape sizes from resnet50, unet and stable diffusion and tests for different number of groups - all the way to num_groups = num_in_channels (depthwise conv)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width,  act_block_w_div, shard_layout",
    (
        (768, 768, 16, 16, 1, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (1280, 1280, 16, 16, 1, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (1280, 1280, 8, 8, 1, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (1280, 2560, 8, 8, 2, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (128, 128, 8, 8, 1, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (128, 128, 16, 16, 1, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (128, 128, 32, 32, 1, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (32, 32, 64, 64, 1, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (32, 32, 128, 64, 1, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (16, 16, 528, 80, 1, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (32, 16, 264, 40, 1, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "filter, dilation, pad",
    [
        [3, 2, 2],
        [3, 3, 3],
    ],
)
def test_conv_dilation(
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
    act_block_w_div,
    shard_layout,
    filter,
    stride,
    pad,
    output_layout,
    dilation,
):
    config_override = {"act_block_w_div": act_block_w_div}
    run_conv(
        device,
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
        pad,
        pad,
        True,
        config_override,
        shard_layout=shard_layout,
        output_layout=output_layout,
        dilation=dilation,
        has_bias=False,
    )


# The following test takes various shape sizes from resnet50, unet and stable diffusion and tests for different number of groups - all the way to num_groups = num_in_channels (depthwise conv)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1, 1, 2, True, None, False),
        (1, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 64, True, None, False),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 1, True, None, False),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 2, True, None, False),
        (2, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 8, True, None, False),
        (1, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 1, True, None, False),
        (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 64, True, None, False),
        (4, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, 128, True, None, False),
        (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 128, True, None, False),
        # (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, 256, False, None, False), circular buffer error
        # (16, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 256, False, None, False), # doesn't fit with bfloat16 weights
        # (32, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, 512, False, None, False), # doesn't fit with bfloat16 weights
        (32, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, 40, False, None, False),
        (32, 160, 160, 7, 7, 3, 3, 1, 1, 1, 1, 10, False, None, False),
        (1, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 8, True, None, False),
        (1, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, 16, True, None, False),
        (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, 32, True, None, False),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 2, False, None, False),
        (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 4, False, None, False),
        (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, 2, False, None, False),
        (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, 320, False, None, False),
        # (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, 1, False, None, False), # doesn't fit with bfloat16 weights
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, 32, True, None, False),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, 2, True, None, False),
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
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_groups(
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
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
):
    run_conv(
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
        groups=groups,
        output_layout=output_layout,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override, use_shallow_conv_variant, groups",
    (
        # yolov4 convs with batch size 1
        # unique convs in yolov4 (complete list) # groups: number
        # (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, True, None, False, 32),  # groups: 32
        # (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, True, None, False, 32),  # groups: 32
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, True, None, False, 64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, True, None, False, 64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, True, None, False, 64),  # groups: 64
        # (1, 64, 64, 480, 640, 3, 3, 1, 1, 1, 1, True, None, False, 64),  # groups: 64
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 128, 128, 240, 320, 3, 3, 1, 1, 1, 1, True, None, False, 128),  # groups: 128
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 256, 256, 120, 160, 3, 3, 1, 1, 1, 1, True, None, False, 256),  # groups: 256
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, True, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, True, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, True, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, True, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, True, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, True, None, False, 512),  # groups: 512
        # (1, 512, 512, 60, 80, 3, 3, 1, 1, 1, 1, True, None, False, 512),  # groups: 512
        (1, 128, 128, 60, 80, 3, 3, 1, 1, 1, 1, True, None, False, 2),  # groups: 512
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    # [ttnn.bfloat8_b, ttnn.bfloat16],
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
# @pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_yolov4_conv_groups_larger_than_one(
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
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
):
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
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
        groups=groups,
        padded_input_channels=16 if input_channels == 3 else None,
        output_layout=output_layout,
    )
