# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import ttnn.torch_tracer
from models.utility_functions import (
    is_wormhole_b0,
    skip_for_grayskull,
)
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn

torch.set_printoptions(linewidth=400, profile="full", sci_mode=False)


def run_conv_transpose2d(
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
    out_pad_h,
    out_pad_w,
    config_override=None,
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
    auto_shard=False,
    mirror_kernel=True,
    enable_split_reader=False,
    enable_act_double_buffer=False,
):
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [input_channels, output_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()

    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv_transpose2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        output_padding=(out_pad_h, out_pad_w),
        dilation=(dilation, dilation),
        groups=groups,
    )

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    if not mirror_kernel:
        torch_flipped_weights = torch.flip(torch_weight_tensor, [2, 3])
        tt_weight_tensor = ttnn.from_torch(
            torch_flipped_weights, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

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
        enable_act_double_buffer=enable_act_double_buffer,
        enable_split_reader=enable_split_reader,
        enable_subblock_padding=False,
        output_layout=output_layout,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_accum,
        packer_l1_acc=packer_l1_acc,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    if config_override and "act_block_w_div" in config_override:
        conv_config.act_block_w_div = config_override["act_block_w_div"]

    [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv_transpose2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        output_padding=(out_pad_h, out_pad_w),
        dilation=(dilation, dilation),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=groups,
        mirror_kernel=mirror_kernel,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    logger.info(f"Conv2d Transpose Input = {(input_height, input_width)} Output = {out_height, out_width}")

    torch_output_tensor = ttnn.to_torch((tt_output_tensor_on_device).cpu())

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :output_channels]

    out = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    if not fp32_accum:
        pcc = 0.99
    elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998

    ref = torch_out_golden_tensor
    passing, pcc_msg = check_with_pcc_without_tensor_printout(out, ref, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, config, shard_layout",
    (
        # Stride = 1
        (1, 8, 8, 256, 256, 3, 3, 1, 1, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (1, 16, 16, 256, 256, 3, 3, 1, 1, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        (1, 256, 256, 32, 32, 3, 3, 1, 1, 1, 1, 0, 0, {"act_block_h": 64}, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 256, 256, 32, 32, 1, 1, 1, 1, 0, 0, 0, 0, {"act_block_h": 64}, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        # Stride = 2
        (1, 8, 8, 32, 64, 3, 3, 2, 2, 1, 1, 1, 1, None, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (1, 128, 128, 32, 64, 3, 3, 2, 2, 1, 1, 1, 1, {"act_block_h": 64}, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 16, 16, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, None, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        # # (1, 16, 16, 32, 32, 3, 3, 2, 2, 1, 1, 0, 0, None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED), # Issue with reading block sharded tensor
        # Vanilla Unet
        # Filter Size = 2 not supported in Block sharded
        # (1, 30, 40, 512, 256, 3, 3, 2, 2, 1, 1, 1, 1,  {"act_block_h": 64}, ttnn.TensorMemoryLayout.BLOCK_SHARDED), # Issue with reading block sharded tensor
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("mirror_kernel", [True, False])
def test_simple_conv_t2d(
    device,
    use_program_cache,
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
    mirror_kernel,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 Grid for Wormhole_b0")
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        out_pad_h=out_pad_h,
        out_pad_w=out_pad_w,
        config_override=config,
        shard_layout=shard_layout,
        auto_shard=True,
        mirror_kernel=mirror_kernel,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
# fmt: off
@pytest.mark.parametrize(
    "batch_size, input_height, input_width, input_channels, output_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w, config, enable_split_reader, enable_act_double_buffer, shard_layout",
    (
        (1, 64, 8, 64, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True, True, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 128, 16, 128, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True, True, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 256, 32, 128, 64, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True, True, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        (1, 512, 64, 128, 2, 4, 4, 2, 2, 1, 1, 0, 0, {"act_block_h": 32*2}, True, True, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ),
)
# fmt: on
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
def test_conv_transpose2d_model_fruit(
    device,
    use_program_cache,
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
    enable_split_reader,
    enable_act_double_buffer,
    shard_layout,
    mirror_kernel=False,
):
    if device.core_grid.y != 8 and is_wormhole_b0():
        pytest.skip("Needs 8x8 Grid for Wormhole_b0")
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.HiFi4,
# fmt: off
@pytest.mark.parametrize(
    "input_shape_nhwc, output_channels, filter, stride, padding, output_padding",
    (
        ((1, 512, 64, 128), 1, (4, 4), (2, 2), (1, 1), (0, 0)),
        ((1, 512, 64, 128), 2, (4, 4), (2, 2), (1, 1), (0, 0)),
    ),
    ids=("1_out_channels_after_knit", "2_out_channels_after_knit"),
)
# fmt: on
@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
def test_old_knit(device, input_shape_nhwc, output_channels, filter, stride, padding, output_padding):
    batch_size, input_height, input_width, input_channels = input_shape_nhwc
    print(f"bs={batch_size}, ih={input_height}, iw={input_width}, ic={input_channels}, oc={output_channels}, kernel={filter}, stride={stride}, padding={padding}, out_pad={output_padding}")

    groups = 1
    dilation = 1
    assert groups == 1, "Groups not supported in split knit"
    assert dilation == 1, "Dilation not supported in split knit"
    assert filter == (4, 4), "Only 4x4 filter supported in split knit"
    assert stride == (2, 2), "Only 2x2 stride supported in split knit"

    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [input_channels, output_channels, filter[0], filter[1]]
    conv_bias_shape = [1, 1, 1, output_channels]

    torch_input_tensor_nchw = torch.rand(conv_input_shape, dtype=torch.bfloat16).float()
    torch_weight_tensor = torch.rand(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.rand(conv_bias_shape, dtype=torch.bfloat16).float()

    # debug
    # torch_input_tensor_nchw = torch.ones(conv_input_shape, dtype=torch.bfloat16).float()*1
    # torch_input_tensor_nchw[:, :, ::2, 1::2] = 2
    # torch_input_tensor_nchw[:, :, 1::2, ::2] = 3
    # torch_input_tensor_nchw[:, :, 1::2, 1::2] = 4
    # torch_weight_tensor = torch.ones(conv_weight_shape, dtype=torch.bfloat16).float()

    for ch in range(torch_weight_tensor.shape[1]):
        torch_weight_tensor[:, ch, ::2, ::2] = (ch + 10)
        torch_weight_tensor[:, ch, ::2, 1::2] = (ch + 20) * 10
        torch_weight_tensor[:, ch, 1::2, ::2] = (ch + 30) / 10
        torch_weight_tensor[:, ch, 1::2, 1::2] = (ch + 40) / 100
    # torch_bias_tensor = torch.zeros(conv_bias_shape, dtype=torch.bfloat16).float()

    torch_out_golden_tensor = torch.nn.functional.conv_transpose2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=None, #torch_bias_tensor.reshape(-1), # if has_bias else None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=1,
        groups=1,
    )

    ##########
    # split_knit_original implementation in torch

    # decompose conv_weight_shape
    # Permute IOHW to OIHW to run base conv2d
    torch_weight_tensor_permute = torch.permute(torch_weight_tensor, (1, 0, 2, 3))
    # Flip kernel to run base conv2d
    torch_weight_tensor_permute = torch.flip(torch_weight_tensor_permute, [2, 3])

    # original implementation 4 convolutions [2x2]
    # 4 convs!
    def split_knit_original(torch_input_tensor_nchw, torch_weight_tensor_permute, torch_bias_tensor, padding):
        conv_weights_decomposed = [
            # reversed order
            torch_weight_tensor_permute[:, :, 1::2, 1::2], # w11, w13, w31, w33
            torch_weight_tensor_permute[:, :, 1::2, 0::2], # w10, w12, w30, w32
            torch_weight_tensor_permute[:, :, 0::2, 1::2], # w01, w03, w21, w23
            torch_weight_tensor_permute[:, :, 0::2, 0::2], # w00, w02, w20, w22
        ]

        torch_split_knit_out_partial = []
        for weight in conv_weights_decomposed:
            torch_split_knit_partial = torch.nn.functional.conv2d(
                torch_input_tensor_nchw,
                weight,
                bias=torch_bias_tensor.reshape(-1),
                stride=[x//2 for x in stride],
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            torch_split_knit_out_partial.append(torch_split_knit_partial)

        # interleave-back convs processed with decomposed filters
        torch_split_knit_out = torch.empty(torch_split_knit_partial.shape[0], torch_split_knit_partial.shape[1], torch_split_knit_partial.shape[2]* 2, torch_split_knit_partial.shape[3]*2, dtype=torch_split_knit_partial.dtype)
        torch_split_knit_out[:,:,0::2, 0::2] = torch_split_knit_out_partial[0]
        torch_split_knit_out[:,:,0::2, 1::2] = torch_split_knit_out_partial[1]
        torch_split_knit_out[:,:,1::2, 0::2] = torch_split_knit_out_partial[2]
        torch_split_knit_out[:,:,1::2, 1::2] = torch_split_knit_out_partial[3]

        # remove padding, ideally should be done in the conv2d function with control on top/bottom and left/right padding
        if (padding[0] != 0 or padding[1] != 0):
            torch_split_knit_out = torch_split_knit_out[:,:,padding[0]:-1, padding[1]:-1]

        return torch_split_knit_out

    torch_split_knit_out = split_knit_original(torch_input_tensor_nchw, torch_weight_tensor_permute, torch_bias_tensor, padding)

    assert torch_out_golden_tensor.shape == torch_split_knit_out.shape

    # Validate torch split-knit against golden torch
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, torch_split_knit_out, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

    ##########
    # modified implementation x4 output channels 1 convolution [2x2]
    def split_knit_modified(torch_input_tensor_nchw, torch_weight_tensor_permute, torch_bias_tensor, padding):
        conv_weights_decomposed_ = torch.zeros([output_channels * 4, input_channels, filter[0]//2, filter[1]//2])

        # reversed order
        for out_ch in range(output_channels):
            conv_weights_decomposed_[out_ch * 4 + 0,:,:,:] = torch_weight_tensor_permute[out_ch, :, 1::2, 1::2] # w11, w13, w31, w33
            conv_weights_decomposed_[out_ch * 4 + 1,:,:,:] = torch_weight_tensor_permute[out_ch, :, 1::2, 0::2] # w10, w12, w30, w32
            conv_weights_decomposed_[out_ch * 4 + 2,:,:,:] = torch_weight_tensor_permute[out_ch, :, 0::2, 1::2] # w01, w03, w21, w23
            conv_weights_decomposed_[out_ch * 4 + 3,:,:,:] = torch_weight_tensor_permute[out_ch, :, 0::2, 0::2] # w00, w02, w20, w22

        # update bias
        s = torch_bias_tensor.shape
        torch_bias_tensor_ = torch_bias_tensor.reshape(-1).repeat_interleave(4)
        torch_bias_tensor_ = torch_bias_tensor_.reshape(s[0], s[1], s[2], s[3] * 4)

        torch_split_knit_out1 = torch.nn.functional.conv2d(
            torch_input_tensor_nchw,
            conv_weights_decomposed_,
            bias=torch_bias_tensor_.reshape(-1),
            stride=[x//2 for x in stride],
            padding=(1,1),
            dilation=1,
            groups=1,
        )

        # interleave-back convs processed with decomposed filters
        # todo fix shape hardcoding
        torch_split_knit_out_ = torch.empty(torch_split_knit_out1.shape[0], torch_split_knit_out1.shape[1]//4, torch_split_knit_out1.shape[2]* 2, torch_split_knit_out1.shape[3]*2, dtype=torch_split_knit_out1.dtype)
        # is there permute/reshape to achieve this?
        torch_split_knit_out_[:,:,0::2, 0::2] = torch_split_knit_out1[:,0,:]
        torch_split_knit_out_[:,:,0::2, 1::2] = torch_split_knit_out1[:,1,:]
        torch_split_knit_out_[:,:,1::2, 0::2] = torch_split_knit_out1[:,2,:]
        torch_split_knit_out_[:,:,1::2, 1::2] = torch_split_knit_out1[:,3,:]

        # remove padding, ideally should be done in the conv2d function with control on top/bottom and left/right padding
        if (padding[0] != 0 or padding[1] != 0):
            torch_split_knit_out_ = torch_split_knit_out_[:,:,padding[0]:-1, padding[1]:-1]
        return [torch_split_knit_out_, conv_weights_decomposed_, torch_bias_tensor_]

    [torch_split_knit_mod_out, torch_conv_weights_decomposed_, torch_bias_tensor_] = split_knit_modified(torch_input_tensor_nchw, torch_weight_tensor_permute, torch_bias_tensor, padding)

    assert torch_out_golden_tensor.shape == torch_split_knit_mod_out.shape

    # Validate torch split-knit /w fused output against golden torch
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, torch_split_knit_mod_out, pcc=0.999)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg



    ##########
    # TTNN

    weights_dtype = ttnn.bfloat8_b
    activations_dtype = ttnn.bfloat16
    math_fidelity = ttnn.MathFidelity.HiFi4
    fp32_dest_acc_en = False
    packer_l1_acc = False

    # Original transposed conv2d, run for perf numbers
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter_height,
        filter_width=filter_width,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        out_pad_h=out_pad_h,
        out_pad_w=out_pad_w,
        config_override=config,
        shard_layout=shard_layout,
        auto_shard=True,
        enable_split_reader=enable_split_reader,
        enable_act_double_buffer=enable_act_double_buffer,
        mirror_kernel=mirror_kernel,
    )
        filter_height=filter[0],
        filter_width=filter[1],
        stride_h=stride[0],
        stride_w=stride[1],
        pad_h=padding[0],
        pad_w=padding[1],
        out_pad_h=output_padding[0],
        out_pad_w=output_padding[1],
        config_override=None,
        shard_layout=None,
        auto_shard=True,
        mirror_kernel=False,
    )

    # Split-knit implementation with fused original conv2d
    torch_input_tensor_nhwc = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor_nhwc,
        activations_dtype if activations_dtype == ttnn.float32 else ttnn.bfloat16,
        mesh_mapper=None,
    )


    tt_weight_tensor = ttnn.from_torch(
        torch_conv_weights_decomposed_,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None
    )

    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor_,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None,
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=None,
        input_channels_alignment=32,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        activation="",
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    kernel_size = [x//2 for x in filter]
    [tt_output_tensor_on_device, [out_h, out_w]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        device=device,
        in_channels=input_channels,
        out_channels = output_channels * 4,
        batch_size = batch_size,
        input_height = input_height,
        input_width = input_width,
        kernel_size=kernel_size,
        stride=[x//2 for x in stride],
        padding=padding,
        dilation=(dilation,dilation),
        groups=groups,
        bias_tensor=tt_bias_tensor,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=False,
    )

    ttnn.synchronize_device(device)

    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 7)), ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(7, 0))})
    shard_shape = (
                tt_output_tensor_on_device.padded_shape[2] // core_range_set.num_cores(),
                tt_output_tensor_on_device.padded_shape[-1],)

    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    resharded_conv_output = ttnn.reshard(tt_output_tensor_on_device, memory_config)
    ttnn.synchronize_device(device)
    num_knit_input_channels = output_channels * 4
    tt_output_tensor = ttnn.to_torch(tt_output_tensor_on_device, mesh_composer=None)

    # this fcks up some memory
    tt_knited_tensor = ttnn.conv_knit(resharded_conv_output, kernel_size[0], output_channels, out_w, num_knit_input_channels)
    resharded_conv_output.deallocate()

    # tt_knited_tensor_out = ttnn.to_torch(tt_knited_tensor, mesh_composer=None)
    # ttnn.synchronize_device(device)

    ttnn.synchronize_device(device)

    # ## HOST FALLBACK
    tt_output_tensor = tt_output_tensor.reshape([batch_size, out_h, out_w, output_channels*4])
    tt_output_tensor = tt_output_tensor.permute(0, 3, 1, 2)
    tt_split_knit_out_ = torch.empty(tt_output_tensor.shape[0], tt_output_tensor.shape[1]//4, tt_output_tensor.shape[2]* 2, tt_output_tensor.shape[3]*2, dtype=tt_output_tensor.dtype)
    # is there permute/reshape to achieve this?
    tt_split_knit_out_[:,:,0::2, 0::2] = tt_output_tensor[:,0,:]
    tt_split_knit_out_[:,:,0::2, 1::2] = tt_output_tensor[:,1,:]
    tt_split_knit_out_[:,:,1::2, 0::2] = tt_output_tensor[:,2,:]
    tt_split_knit_out_[:,:,1::2, 1::2] = tt_output_tensor[:,3,:]
    print("ref_out_shape_after knitting: ", tt_split_knit_out_.shape)
    # print("tt_knit out shape pre permute: ", tt_knited_tensor_out.shape)

    # tt_knited_tensor_out = tt_knited_tensor_out.permute(0, 3, 2, 1)
    # print("tt_knit out shape pre reshape: ", tt_knited_tensor_out.shape)

    # tt_knited_tensor_out = tt_knited_tensor_out.reshape(tt_split_knit_out_.shape)
    # print("tt_knit out shape pre reshape: ", tt_knited_tensor_out.shape)

    # row_id = 4
    # print("Shape check: ", tt_knited_tensor_out.shape, tt_split_knit_out_.shape)
    # print("Pre unpadding tt_knited_tensor_out  is:", tt_knited_tensor_out[:, :, row_id, :])
    # print("Pre unpadding reference split knit out is:", tt_split_knit_out_[:, :, row_id, :])

    # pcc = 0.999
    # passing, pcc_msg = check_with_pcc_without_tensor_printout(tt_knited_tensor_out, tt_split_knit_out_, pcc=pcc)
    # assert passing, pcc_msg

    # remove padding, ideally should be done in the conv2d function with control on top/bottom and left/right padding
    if (padding[0] != 0 or padding[1] != 0):
        tt_split_knit_out_ = tt_split_knit_out_[:,:,padding[0]:-1, padding[1]:-1]
        #tt_knited_tensor_out = tt_knited_tensor_out[:,:,padding[0]:-1, padding[1]:-1]
    # Validate ttnn split-knit /w fused output against golden torch
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, tt_split_knit_out_, pcc=pcc)
    # passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, tt_knited_tensor_out, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

        # # ((1, 64, 8, 64),     64, (4, 4), (2, 2), (1, 1), (0, 0)),  # transposed conv 1 - fails physical_height == physical_shard_height
        #  ((1, 128, 16, 128),  64, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv 2
        #  ((1, 256, 32, 128),  64, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv 3
        #  ((1, 512, 64, 128),   2, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv 4


# fmt: off
@pytest.mark.parametrize(
    "input_shape_nhwc, output_channels, filter, stride, padding, output_padding",
    (
        # ((1, 64, 8, 64),  64, (4, 4), (2, 2), (1, 1), (0, 0)),  # transposed conv 1 - fails physical_height == physical_shard_height
        ((1, 128, 16, 128), 64, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv 2
        ((1, 256, 32, 128), 64, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv 3
        ((1, 512, 64, 128),  1, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv4 - 1 out channels version
        ((1, 512, 64, 128),  2, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv4 - 2 out channels version (original one)
    ),
    ids=("conv_transpose2", "conv_transpose3", "conv_transpose4_1_out_channels", "conv_transpose4_2_out_channels"),
)
# fmt: on
@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
def test_conv_transpose2d_split_knit(request, device, input_shape_nhwc, output_channels, filter, stride, padding, output_padding):

    test_id = request.node.callspec.id
    is_conv4 = "conv_transpose4_1_out_channels" in test_id or "conv_transpose4_2_out_channels" in test_id
    print("Test ID: ", test_id)
    print("Is conv4: ", is_conv4)
    batch_size, input_height, input_width, input_channels = input_shape_nhwc
    print(f"{batch_size}, {input_height}, {input_width}, {input_channels}, {output_channels}, {filter}, {stride}, {padding}, {output_padding}")

    groups = 1
    dilation = 1
    assert groups == 1, "Groups not supported in split knit"
    assert dilation == 1, "Dilation not supported in split knit"
    assert filter == (4, 4), "Only 4x4 filter supported in split knit"
    assert stride == (2, 2), "Only 2x2 stride supported in split knit"

    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [input_channels, output_channels, filter[0], filter[1]]
    conv_bias_shape = [1, 1, 1, output_channels]

    torch_input_tensor_nchw = torch.rand(conv_input_shape, dtype=torch.bfloat16).float()
    torch_weight_tensor = torch.rand(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.rand(conv_bias_shape, dtype=torch.bfloat16).float()

    # debug
    torch_input_tensor_nchw = torch.ones(conv_input_shape, dtype=torch.bfloat16).float()*1
    torch_input_tensor_nchw[:, :, ::2, 1::2] = 2
    torch_input_tensor_nchw[:, :, 1::2, ::2] = 3
    torch_input_tensor_nchw[:, :, 1::2, 1::2] = 4
    torch_weight_tensor = torch.ones(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.zeros(conv_bias_shape, dtype=torch.bfloat16).float()

    for ch in range(torch_weight_tensor.shape[1]):
        torch_weight_tensor[:, ch, ::2, ::2] = (ch + 10)
        torch_weight_tensor[:, ch, ::2, 1::2] = (ch + 20) * 10
        torch_weight_tensor[:, ch, 1::2, ::2] = (ch + 30) / 10
        torch_weight_tensor[:, ch, 1::2, 1::2] = (ch + 40) / 100

    torch_out_golden_tensor = torch.nn.functional.conv_transpose2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1), # if has_bias else None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=1,
        groups=1,
    )

    ##########
    # split_knit_original implementation in torch

    # decompose conv_weight_shape
    # Permute IOHW to OIHW to run base conv2d
    torch_weight_tensor_oihw = torch.permute(torch_weight_tensor, (1, 0, 2, 3))
    # Flip kernel to run base conv2d
    torch_weight_tensor_oihw = torch.flip(torch_weight_tensor_oihw, [2, 3])

    # original implementation 4 convolutions [2x2]
    # 4 convs!
    def split_knit_original(torch_input_tensor_nchw, torch_weight_tensor_permute, torch_bias_tensor, padding):
        conv_weights_decomposed = [
            # reversed order
            torch_weight_tensor_permute[:, :, 1::2, 1::2], # 00 = w11, w13, w31, w33
            torch_weight_tensor_permute[:, :, 1::2, 0::2], # 01 = w10, w12, w30, w32
            torch_weight_tensor_permute[:, :, 0::2, 1::2], # 10 = w01, w03, w21, w23
            torch_weight_tensor_permute[:, :, 0::2, 0::2], # 11 = w00, w02, w20, w22
        ]

        torch_split_knit_out_partial = []
        for weight in conv_weights_decomposed:
            torch_split_knit_partial = torch.nn.functional.conv2d(
                torch_input_tensor_nchw,
                weight,
                bias=torch_bias_tensor.reshape(-1),
                stride=[x//2 for x in stride],
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            torch_split_knit_out_partial.append(torch_split_knit_partial)

        # interleave-back convs processed with decomposed filters
        torch_split_knit_out = torch.empty(torch_split_knit_partial.shape[0], torch_split_knit_partial.shape[1], torch_split_knit_partial.shape[2]* 2, torch_split_knit_partial.shape[3]*2, dtype=torch_split_knit_partial.dtype)
        torch_split_knit_out[:,:,0::2, 0::2] = torch_split_knit_out_partial[0]
        torch_split_knit_out[:,:,0::2, 1::2] = torch_split_knit_out_partial[1]
        torch_split_knit_out[:,:,1::2, 0::2] = torch_split_knit_out_partial[2]
        torch_split_knit_out[:,:,1::2, 1::2] = torch_split_knit_out_partial[3]

        # remove padding, ideally should be done in the conv2d function with control on top/bottom and left/right padding
        if (padding[0] != 0 or padding[1] != 0):
            torch_split_knit_out = torch_split_knit_out[:,:,padding[0]:-1, padding[1]:-1]

        return torch_split_knit_out

    torch_split_knit_out = split_knit_original(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, padding)

    assert torch_out_golden_tensor.shape == torch_split_knit_out.shape

    # Validate torch split-knit against golden torch
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, torch_split_knit_out, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

    ##########
    # modified implementation x4 output channels 1 convolution [2x2]
    def split_knit_modified(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, padding):
        conv_weights_decomposed_ = torch.zeros([output_channels * 4, input_channels, filter[0]//2, filter[1]//2])

        # reversed order
        for out_ch in range(output_channels):
            conv_weights_decomposed_[out_ch * 4 + 0,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 1::2, 1::2] # 00 = w11, w13, w31, w33
            conv_weights_decomposed_[out_ch * 4 + 1,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 1::2, 0::2] # 01 = w10, w12, w30, w32
            conv_weights_decomposed_[out_ch * 4 + 2,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 0::2, 1::2] # 10 = w01, w03, w21, w23
            conv_weights_decomposed_[out_ch * 4 + 3,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 0::2, 0::2] # 11 = w00, w02, w20, w22

        # update bias
        s = torch_bias_tensor.shape
        torch_bias_tensor_ = torch_bias_tensor.reshape(-1).repeat_interleave(4)
        torch_bias_tensor_ = torch_bias_tensor_.reshape(s[0], s[1], s[2], s[3] * 4)

        torch_split_knit_out1 = torch.nn.functional.conv2d(
            torch_input_tensor_nchw,
            conv_weights_decomposed_,
            bias=torch_bias_tensor_.reshape(-1),
            stride=[x//2 for x in stride],
            padding=(1,1),
            dilation=1,
            groups=1,
        )

        # interleave-back convs processed with decomposed filters
        # todo fix shape hardcoding
        torch_split_knit_out_ = torch.empty(torch_split_knit_out1.shape[0], torch_split_knit_out1.shape[1]//4, torch_split_knit_out1.shape[2]* 2, torch_split_knit_out1.shape[3]*2, dtype=torch_split_knit_out1.dtype)
        # is there permute/reshape to achieve this?
        torch_split_knit_out_[:,:,0::2, 0::2] = torch_split_knit_out1[:,0,:]
        torch_split_knit_out_[:,:,0::2, 1::2] = torch_split_knit_out1[:,1,:]
        torch_split_knit_out_[:,:,1::2, 0::2] = torch_split_knit_out1[:,2,:]
        torch_split_knit_out_[:,:,1::2, 1::2] = torch_split_knit_out1[:,3,:]

        # remove padding, ideally should be done in the conv2d function with control on top/bottom and left/right padding
        if (padding[0] != 0 or padding[1] != 0):
            torch_split_knit_out_ = torch_split_knit_out_[:,:,padding[0]:-1, padding[1]:-1]
        return [torch_split_knit_out_, conv_weights_decomposed_, torch_bias_tensor_]

    [torch_split_knit_mod_out, torch_conv_weights_decomposed_, torch_bias_tensor_] = split_knit_modified(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, padding)

    assert torch_out_golden_tensor.shape == torch_split_knit_mod_out.shape

    # Validate torch split-knit /w fused output against golden torch
    pcc = 0.991
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, torch_split_knit_mod_out, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

    ##########
    # DM friendly modification

    def split_knit_modified2(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, padding):
        conv_weights_decomposed_ = torch.zeros([output_channels * 4, input_channels, filter[0]//2, filter[1]//2])

        # NEW order - | CH0 00, CH1 00, CH0 01, CH1 01, CH0 10, CH1 10, CH0 11, CH1 11|
        for out_ch in range(output_channels):
            conv_weights_decomposed_[0*output_channels + out_ch,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 1::2, 1::2] # 00
            conv_weights_decomposed_[1*output_channels + out_ch,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 1::2, 0::2] # 01
            conv_weights_decomposed_[2*output_channels + out_ch,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 0::2, 1::2] # 10
            conv_weights_decomposed_[3*output_channels + out_ch,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 0::2, 0::2] # 11

        # update bias
        s = torch_bias_tensor.shape
        torch_bias_tensor_ = torch_bias_tensor.reshape(-1).repeat_interleave(4)
        torch_bias_tensor_ = torch_bias_tensor_.reshape(s[0], s[1], s[2], s[3] * 4)

        torch_split_knit_out1 = torch.nn.functional.conv2d(
            torch_input_tensor_nchw,
            conv_weights_decomposed_,
            bias=torch_bias_tensor_.reshape(-1),
            stride=[x//2 for x in stride],
            padding=(1,1),
            dilation=1,
            groups=1,
        )

        # interleave-back convs processed with decomposed filters
        # todo fix shape hardcoding
        torch_split_knit_out_ = torch.empty(torch_split_knit_out1.shape[0], torch_split_knit_out1.shape[2]* 2, torch_split_knit_out1.shape[3]*2, torch_split_knit_out1.shape[1]//4, dtype=torch_split_knit_out1.dtype) # ch last
        torch_split_knit_out1 = torch_split_knit_out1.permute([0,2,3,1]) # ch last

        # input layout:
        # CH0 00, CH1 00, CH0 01, CH1 01, CH0 10, CH1 10, CH0 11, CH1 11

        # output layout:
        # CH0 00, CH1 00, CH0 01, CH1 01, .. CH0 0W, CH1 0W
        # CH0 10, CH1 10, CH0 11, CH1 11, .. CH0 1W, CH1 1W
        # ...
        # CH0 H0, CH1 H0, CH0 H1, CH1 H1, .. CH0 HW, CH1 HW

        for h in range(torch_split_knit_out1.shape[1]):
            for w in range(torch_split_knit_out1.shape[2]):
                # even rows
                torch_split_knit_out_[0, h*2, w*2, :] = torch_split_knit_out1[0, h, w, output_channels*0:output_channels*0+output_channels]
                torch_split_knit_out_[0, h*2, w*2+1, :] = torch_split_knit_out1[0, h, w, output_channels*1:output_channels*1+output_channels]
                # odd rows
                torch_split_knit_out_[0, h*2+1, w*2, :] = torch_split_knit_out1[0, h, w, output_channels*2:output_channels*2+output_channels]
                torch_split_knit_out_[0, h*2+1, w*2+1, :] = torch_split_knit_out1[0, h, w, output_channels*3:output_channels*3+output_channels]

        # remove padding, ideally should be done in the conv2d function with control on top/bottom and left/right padding
        if (padding[0] != 0 or padding[1] != 0):
            torch_split_knit_out_ = torch_split_knit_out_[:,padding[0]:-1, padding[1]:-1,:]

        # return to channel first for torch comparison
        torch_split_knit_out_ = torch_split_knit_out_.permute([0,3,1,2])
        return [torch_split_knit_out_, conv_weights_decomposed_, torch_bias_tensor_]

    [torch_split_knit_mod_out2, torch_conv_weights_decomposed_2, torch_bias_tensor_2] = split_knit_modified2(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, padding)

    assert torch_out_golden_tensor.shape == torch_split_knit_mod_out2.shape

    # Validate torch split-knit /w fused output against golden torch
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, torch_split_knit_mod_out2, pcc=0.999)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg


    ##########
    # TTNN

    weights_dtype = ttnn.bfloat16
    activations_dtype = ttnn.bfloat16
    math_fidelity = ttnn.MathFidelity.HiFi4
    fp32_dest_acc_en = False
    packer_l1_acc = False

    # Original transposed conv2d, run for perf numbers, uses different input/outputs tensors - to cleanup
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter[0],
        filter_width=filter[1],
        stride_h=stride[0],
        stride_w=stride[1],
        pad_h=padding[0],
        pad_w=padding[1],
        out_pad_h=output_padding[0],
        out_pad_w=output_padding[1],
        config_override=None,
        shard_layout=None,
        auto_shard=True,
        mirror_kernel=False,
    )

    # Split-knit implementation with fused original conv2d
    torch_input_tensor_nhwc = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor_nhwc,
        activations_dtype if activations_dtype == ttnn.float32 else ttnn.bfloat16,
        mesh_mapper=None,
    )


    tt_weight_tensor = ttnn.from_torch(
        torch_conv_weights_decomposed_2,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None
    )

    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor_2,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None,
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=None,
        input_channels_alignment=32,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        activation="",
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    kernel_size = [x//2 for x in filter]
    conv2d_out_channels = output_channels * 4
    [tt_output_tensor_on_device, [out_h, out_w]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        device=device,
        in_channels=input_channels,
        out_channels = conv2d_out_channels,
        batch_size = batch_size,
        input_height = input_height,
        input_width = input_width,
        kernel_size=kernel_size,
        stride=[x//2 for x in stride],
        padding=padding,
        dilation=(dilation,dilation),
        groups=groups,
        bias_tensor=tt_bias_tensor,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=False,
    )

    print(f"conv2d output shape: h = {out_h}, w = {out_w} out_channels = {conv2d_out_channels}")
    print(f"conv2d out shard spec: {tt_output_tensor_on_device.memory_config.shard_spec}")
    ttnn.synchronize_device(device)

    if is_conv4:
        core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 7)), ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(7, 0))})
        shard_shape = (
                    tt_output_tensor_on_device.padded_shape[2] // core_range_set.num_cores(),
                    tt_output_tensor_on_device.padded_shape[-1],)

        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_set,
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        resharded_conv_output = ttnn.reshard(tt_output_tensor_on_device, memory_config)
        ttnn.synchronize_device(device)
    else:
        resharded_conv_output = tt_output_tensor_on_device
    # tt_output_tensor = ttnn.to_torch(tt_output_tensor_on_device, mesh_composer=None)

    tt_knited_tensor = ttnn.conv_knit(resharded_conv_output, kernel_size[0], output_channels, out_w, conv2d_out_channels)
    resharded_conv_output.deallocate()
    #tt_knited_tensor_out = ttnn.to_torch(tt_knited_tensor, mesh_composer=None)

    #print("tt knitted tensor pre doing anything: ", tt_knited_tensor_out.shape)
    #print("first row: ", tt_knited_tensor_out[0, 0, :130, :])

    tt_output_tensor = ttnn.to_torch(tt_output_tensor_on_device, mesh_composer=None)

    # ## HOST FALLBACK 2.0
    tt_split_knit_out_ = torch.empty(tt_output_tensor.shape[0], out_h*2, out_w*2, tt_output_tensor.shape[3]//4, dtype=tt_output_tensor.dtype)
    # slow version of knit!
    for h in range(out_h):
        for w in range(out_w):
            # for ch in range(output_channels):
            # even rows
            tt_split_knit_out_[0, h*2, w*2, :] = tt_output_tensor[0, 0, h*out_w + w, output_channels*0:output_channels*0+output_channels]
            tt_split_knit_out_[0, h*2, w*2+1, :] = tt_output_tensor[0, 0, h*out_w + w, output_channels*1:output_channels*1+output_channels]
            # odd rows
            tt_split_knit_out_[0, h*2+1, w*2, :] = tt_output_tensor[0, 0, h*out_w + w, output_channels*2:output_channels*2+output_channels]
            tt_split_knit_out_[0, h*2+1, w*2+1, :] = tt_output_tensor[0, 0, h*out_w + w, output_channels*3:output_channels*3+output_channels]
    #fast version of knit
    # reshaped = tt_output_tensor.view(1, 1, out_h, out_w, 4, output_channels)
    # # Assign values using slicing
    # tt_split_knit_out_[:, ::2, ::2, :]   = reshaped[:, :, :, :, 0, :]  # Top-left
    # tt_split_knit_out_[:, ::2, 1::2, :]  = reshaped[:, :, :, :, 1, :]  # Top-right
    # tt_split_knit_out_[:, 1::2, ::2, :]  = reshaped[:, :, :, :, 2, :]  # Bottom-left
    # tt_split_knit_out_[:, 1::2, 1::2, :] = reshaped[:, :, :, :, 3, :]  # Bottom-right

    #print("tt_knit out shape pre reshape: ", tt_knited_tensor_out.shape)

    # tt_knited_tensor_out = tt_knited_tensor_out.permute(0, 3, 2, 1)
    # print("tt_knit out shape pre reshape: ", tt_knited_tensor_out.shape)

    #print("tt_knit out shape post reshape: ", tt_knited_tensor_out.shape)

    row_id = 0
    # print("Shape check: ", tt_knited_tensor_out.shape, tt_split_knit_out_.shape)
    #print("Pre unpadding TT_OUT is:", tt_knited_tensor_out[:, row_id, :, :])
    #print("Pre unpadding REF out is:", tt_split_knit_out_[:, row_id, :, :])
    pcc = 0.999
    #print("Pre unpadding pcc check!")
    #passing, pcc_msg = check_with_pcc_without_tensor_printout(tt_knited_tensor_out, tt_split_knit_out_, pcc=pcc)
    #assert passing, pcc_msg

    # todo merge this with upper loop
    print(f"Padding is: 0={padding[0]} 1={padding[1]}")
    print(f"Pre unpadding shape: {tt_split_knit_out_.shape}")
    if (padding[0] != 0 or padding[1] != 0):
        tt_split_knit_out_ = tt_split_knit_out_[:,padding[0]:-1, padding[1]:-1,:]
        #tt_knited_tensor_out = tt_knited_tensor_out[:,padding[0]:-1, padding[1]:-1,:]
    print(f"Post unpadding shape: {tt_split_knit_out_.shape}")

    h_after_knit = out_h * 2
    w_after_knit = out_w * 2
    if is_conv4:
        out_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        print("Setting up core range for num_cores: ", out_core_range_set.num_cores())
        print(
            f"Setting up ttnn:conv_crop pre_crop_tensor_h: {h_after_knit}, pre_crop_tensor_w: {w_after_knit}, post_crop_tensor_h: {tt_split_knit_out_.shape[1]}, post_crop_tensor_w: {tt_split_knit_out_.shape[2]}"
        )
        out_shard_shape = (
            tt_split_knit_out_.shape[1] * tt_split_knit_out_.shape[2] // out_core_range_set.num_cores(),
            tt_knited_tensor.padded_shape[3],
        )

        post_crop_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                out_core_range_set,
                out_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        print(f"Setting up shard shape: {out_shard_shape}")
        tt_knited_tensor = ttnn.conv_crop(
            tt_knited_tensor, post_crop_mem_config, padding[0], padding[1], h_after_knit, w_after_knit
        )
        tt_knited_tensor_out = ttnn.to_torch(tt_knited_tensor, mesh_composer=None)
        tt_knited_tensor_out = tt_knited_tensor_out.reshape(tt_split_knit_out_.shape)


    ## end of HOST FALLBACK

    tt_split_knit_out_ = tt_split_knit_out_.permute(0, 3, 1, 2) # NCHW
    tt_knited_tensor_out = tt_knited_tensor_out.permute(0, 3, 1, 2) # NCHW
    # Validate ttnn split-knit /w fused output against golden torch
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, tt_split_knit_out_, pcc=pcc)
    print("Final pcc check!")
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, tt_knited_tensor_out, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg

# fmt: off
@pytest.mark.parametrize(
    "input_shape_nhwc, output_channels, filter, stride, padding, output_padding",
    (
        # ((1, 512, 64, 128), 2, (4, 4), (2, 2), (1, 1), (0, 0)),
        # ((1, 512, 64, 128), 32//4, (4, 4), (2, 2), (1, 1), (0, 0)),
        # ((1, 4, 4, 1), 1, (4, 4), (2, 2), (0, 0), (0, 0)),
        # ((1, 512, 64, 128), 2, (4, 4), (2, 2), (0, 0), (0, 0)),
        # ((1, 4, 4, 1), 2, (4, 4), (2, 2), (1, 1), (0, 0)),
        # ((1, 4, 4, 1), 3, (4, 4), (2, 2), (1, 1), (0, 0)),
        # ((1, 4, 4, 1), 1, (4, 4), (2, 2), (1, 1), (0, 0)),
        # ((1, 32, 32, 1), 2, (4, 4), (2, 2), (1, 1), (0, 0)),
        # ((1, 32, 32, 1), 4, (4, 4), (2, 2), (1, 1), (0, 0)),

        # ((1, 64, 8, 64),     64, (4, 4), (2, 2), (1, 1), (0, 0)),  # transposed conv 1 - fails physical_height == physical_shard_height
        ((1, 128, 16, 128),  64, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv 2
        ((1, 256, 32, 128),  64, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv 3
        ((1, 512, 64, 128),   2, (4, 4), (2, 2), (1, 1), (0, 0)), # transposed conv 4
    ),
    ids=("conv_transpose_2", "conv_transpose_3", "conv_transpose_4"),
)
# fmt: on
@pytest.mark.parametrize("device_params", [{"l1_small_size": 64 * 1024}], indirect=True)
def test_conv_transpose2d_split_knit_new(request, device, input_shape_nhwc, output_channels, filter, stride, padding, output_padding):
    batch_size, input_height, input_width, input_channels = input_shape_nhwc
    print(f"{batch_size}, {input_height}, {input_width}, {input_channels}, {output_channels}, {filter}, {stride}, {padding}, {output_padding}")

    test_id = request.node.callspec.id
    is_conv4 = "conv_transpose_4" in test_id
    is_conv3 = "conv_transpose_3" in test_id
    is_conv2 = "conv_transpose_2" in test_id

    groups = 1
    dilation = 1
    assert groups == 1, "Groups not supported in split knit"
    assert dilation == 1, "Dilation not supported in split knit"
    assert filter == (4, 4), "Only 4x4 filter supported in split knit"
    assert stride == (2, 2), "Only 2x2 stride supported in split knit"

    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [input_channels, output_channels, filter[0], filter[1]]
    conv_bias_shape = [1, 1, 1, output_channels]

    torch_input_tensor_nchw = torch.rand(conv_input_shape, dtype=torch.bfloat16).float()
    torch_weight_tensor = torch.rand(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.rand(conv_bias_shape, dtype=torch.bfloat16).float()

    # debug
    torch_input_tensor_nchw = torch.ones(conv_input_shape, dtype=torch.bfloat16).float()*1
    torch_input_tensor_nchw[:, :, ::2, 1::2] = 2
    torch_input_tensor_nchw[:, :, 1::2, ::2] = 3
    torch_input_tensor_nchw[:, :, 1::2, 1::2] = 4
    torch_weight_tensor = torch.ones(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.zeros(conv_bias_shape, dtype=torch.bfloat16).float()

    for ch in range(torch_weight_tensor.shape[1]):
        torch_weight_tensor[:, ch, ::2, ::2] = (ch + 10)
        torch_weight_tensor[:, ch, ::2, 1::2] = (ch + 20) * 10
        torch_weight_tensor[:, ch, 1::2, ::2] = (ch + 30) / 10
        torch_weight_tensor[:, ch, 1::2, 1::2] = (ch + 40) / 100

    torch_out_golden_tensor = torch.nn.functional.conv_transpose2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1), # if has_bias else None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=1,
        groups=1,
    )

    ##########
    # split_knit_original implementation in torch

    # decompose conv_weight_shape
    # Permute IOHW to OIHW to run base conv2d
    torch_weight_tensor_oihw = torch.permute(torch_weight_tensor, (1, 0, 2, 3))
    # Flip kernel to run base conv2d
    torch_weight_tensor_oihw = torch.flip(torch_weight_tensor_oihw, [2, 3])

    ##########
    # DM friendly modification

    def split_knit_modified2(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, padding):
        conv_weights_decomposed_ = torch.zeros([output_channels * 4, input_channels, filter[0]//2, filter[1]//2])

        # NEW order - | CH0 00, CH1 00, CH0 01, CH1 01, CH0 10, CH1 10, CH0 11, CH1 11|
        for out_ch in range(output_channels):
            conv_weights_decomposed_[0*output_channels + out_ch,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 1::2, 1::2] # 00
            conv_weights_decomposed_[1*output_channels + out_ch,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 1::2, 0::2] # 01
            conv_weights_decomposed_[2*output_channels + out_ch,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 0::2, 1::2] # 10
            conv_weights_decomposed_[3*output_channels + out_ch,:,:,:] = torch_weight_tensor_oihw[out_ch, :, 0::2, 0::2] # 11

        # update bias
        s = torch_bias_tensor.shape
        torch_bias_tensor_ = torch_bias_tensor.reshape(-1).repeat_interleave(4)
        torch_bias_tensor_ = torch_bias_tensor_.reshape(s[0], s[1], s[2], s[3] * 4)

        torch_split_knit_out1 = torch.nn.functional.conv2d(
            torch_input_tensor_nchw,
            conv_weights_decomposed_,
            bias=torch_bias_tensor_.reshape(-1),
            stride=[x//2 for x in stride],
            padding=(1,1),
            dilation=1,
            groups=1,
        )

        # interleave-back convs processed with decomposed filters
        # todo fix shape hardcoding
        torch_split_knit_out_ = torch.empty(torch_split_knit_out1.shape[0], torch_split_knit_out1.shape[2]* 2, torch_split_knit_out1.shape[3]*2, torch_split_knit_out1.shape[1]//4, dtype=torch_split_knit_out1.dtype) # ch last
        torch_split_knit_out1 = torch_split_knit_out1.permute([0,2,3,1]) # ch last

        # input layout:
        # CH0 00, CH1 00, CH0 01, CH1 01, CH0 10, CH1 10, CH0 11, CH1 11

        # output layout:
        # CH0 00, CH1 00, CH0 01, CH1 01, .. CH0 0W, CH1 0W
        # CH0 10, CH1 10, CH0 11, CH1 11, .. CH0 1W, CH1 1W
        # ...
        # CH0 H0, CH1 H0, CH0 H1, CH1 H1, .. CH0 HW, CH1 HW

        for h in range(torch_split_knit_out1.shape[1]):
            for w in range(torch_split_knit_out1.shape[2]):
                # even rows
                torch_split_knit_out_[0, h*2, w*2, :] = torch_split_knit_out1[0, h, w, output_channels*0:output_channels*0+output_channels]
                torch_split_knit_out_[0, h*2, w*2+1, :] = torch_split_knit_out1[0, h, w, output_channels*1:output_channels*1+output_channels]
                # odd rows
                torch_split_knit_out_[0, h*2+1, w*2, :] = torch_split_knit_out1[0, h, w, output_channels*2:output_channels*2+output_channels]
                torch_split_knit_out_[0, h*2+1, w*2+1, :] = torch_split_knit_out1[0, h, w, output_channels*3:output_channels*3+output_channels]

        # remove padding, ideally should be done in the conv2d function with control on top/bottom and left/right padding
        if (padding[0] != 0 or padding[1] != 0):
            torch_split_knit_out_ = torch_split_knit_out_[:,padding[0]:-1, padding[1]:-1,:]

        # return to channel first for torch comparison
        torch_split_knit_out_ = torch_split_knit_out_.permute([0,3,1,2])
        return [torch_split_knit_out_, conv_weights_decomposed_, torch_bias_tensor_]

    [torch_split_knit_mod_out2, torch_conv_weights_decomposed_2, torch_bias_tensor_2] = split_knit_modified2(torch_input_tensor_nchw, torch_weight_tensor_oihw, torch_bias_tensor, padding)

    assert torch_out_golden_tensor.shape == torch_split_knit_mod_out2.shape

    # Validate torch split-knit /w fused output against golden torch
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, torch_split_knit_mod_out2, pcc=0.999)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg


    ##########
    # TTNN

    weights_dtype = ttnn.bfloat16
    activations_dtype = ttnn.bfloat16
    math_fidelity = ttnn.MathFidelity.HiFi4
    fp32_dest_acc_en = False
    packer_l1_acc = False

    # Original transposed conv2d, run for perf numbers, uses different input/outputs tensors - to cleanup
    run_conv_transpose2d(
        device,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activations_dtype=activations_dtype,
        weights_dtype=weights_dtype,
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        filter_height=filter[0],
        filter_width=filter[1],
        stride_h=stride[0],
        stride_w=stride[1],
        pad_h=padding[0],
        pad_w=padding[1],
        out_pad_h=output_padding[0],
        out_pad_w=output_padding[1],
        config_override=None,
        shard_layout=None,
        auto_shard=True,
        mirror_kernel=False,
    )

    # Split-knit implementation with fused original conv2d
    torch_input_tensor_nhwc = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor_nhwc,
        activations_dtype if activations_dtype == ttnn.float32 else ttnn.bfloat16,
        mesh_mapper=None,
    )


    tt_weight_tensor = ttnn.from_torch(
        torch_conv_weights_decomposed_2,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None
    )

    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor_2,
        weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32,
        mesh_mapper=None,
    )

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=32,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        activation="",
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )

    conv2d_out_channels = output_channels * 4
    kernel_size = [x//2 for x in filter]

    if is_conv3:
        # Apply more padding if conv3, so we get a valid height shard spec for reshard that can be proplerly split on some number of cores
        padding = (2, 2)

    [tt_output_tensor_on_device, [out_h, out_w]] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        device=device,
        in_channels=input_channels,
        out_channels = conv2d_out_channels,
        batch_size = batch_size,
        input_height = input_height,
        input_width = input_width,
        kernel_size=kernel_size,
        stride=[x//2 for x in stride],
        padding=padding,
        dilation=(dilation,dilation),
        groups=groups,
        bias_tensor=tt_bias_tensor,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=False,
    )
    ttnn.synchronize_device(device)
    print(f"conv2d output shape: h = {out_h}, w = {out_w} out_channels = {conv2d_out_channels}")
    mem_cfg = ttnn.get_memory_config(tt_output_tensor_on_device)
    print(f"conv2d out shard spec: {mem_cfg.shard_spec}")
    print(f"conv2d out num cores: {mem_cfg.shard_spec.num_cores()}")
    print(f"conv2d memory_layout: {mem_cfg.memory_layout}")

    if is_conv4:
        core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 7)), ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(7, 0))})
        shard_shape = (
                    tt_output_tensor_on_device.padded_shape[2] // core_range_set.num_cores(),
                    tt_output_tensor_on_device.padded_shape[-1],)

        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_set,
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        resharded_conv_output = ttnn.reshard(tt_output_tensor_on_device, memory_config)
        ttnn.synchronize_device(device)
    elif is_conv2:
        core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 7)), ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 2))})
        shard_shape = (
            tt_output_tensor_on_device.padded_shape[2] // core_range_set.num_cores(),
            tt_output_tensor_on_device.padded_shape[-1],)
        print("Setting up core range for num_cores: ", core_range_set.num_cores())
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_set,
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        resharded_conv_output = ttnn.reshard(tt_output_tensor_on_device, memory_config)
        # tt_output_tensor = ttnn.to_torch(tt_output_tensor_on_device)
        #     #fast version of knit
        # reshaped = tt_output_tensor.view(1, 1, out_h, out_w, 4, output_channels)
        # # Assign values using slicing
        # tt_split_knit_out_ = torch.empty(tt_output_tensor.shape[0], out_h*2, out_w*2, tt_output_tensor.shape[3]//4, dtype=tt_output_tensor.dtype)
        # tt_split_knit_out_[:, ::2, ::2, :]   = reshaped[:, :, :, :, 0, :]  # Top-left
        # tt_split_knit_out_[:, ::2, 1::2, :]  = reshaped[:, :, :, :, 1, :]  # Top-right
        # tt_split_knit_out_[:, 1::2, ::2, :]  = reshaped[:, :, :, :, 2, :]  # Bottom-left
        # tt_split_knit_out_[:, 1::2, 1::2, :] = reshaped[:, :, :, :, 3, :]  # Bottom-right
        tt_output_tensor_on_device.deallocate()
        resharded_conv_output = ttnn.move(resharded_conv_output)
        ttnn.synchronize_device(device)
    elif is_conv3:
        core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7)), ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 4))})
        shard_shape = (
            tt_output_tensor_on_device.padded_shape[2] // core_range_set.num_cores(),
            tt_output_tensor_on_device.padded_shape[-1],)
        print("Setting up core range for num_cores: ", core_range_set.num_cores())
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_set,
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        resharded_conv_output = ttnn.reshard(tt_output_tensor_on_device, memory_config)
        tt_output_tensor_on_device.deallocate()
        resharded_conv_output = ttnn.move(resharded_conv_output)
        ttnn.synchronize_device(device)

    print("Start with conv knit")
    tt_knited_tensor = ttnn.conv_knit(resharded_conv_output, kernel_size[0], output_channels, out_w, conv2d_out_channels)
    ttnn.synchronize_device(device)
    print("Done with conv knit")
    resharded_conv_output.deallocate()
    # tt_knited_out = ttnn.to_torch(tt_knited_tensor, mesh_composer=None)
    # tt_knited_out = tt_knited_out.reshape(tt_split_knit_out_.shape)

    # pcc = 0.999
    # passing, pcc_msg = check_with_pcc_without_tensor_printout(tt_knited_out, tt_split_knit_out_, pcc=0.999)
    # logger.info(f"knit check: PCC = {pcc_msg}. Threshold = {pcc}")
    # assert passing, pcc_msg

    h_after_knit = out_h * 2
    w_after_knit = out_w * 2
    h_after_crop = h_after_knit - 2 * padding[0]
    w_after_crop = w_after_knit - 2 * padding[1]
    if is_conv3:
        # Remove additional 2 units of padding
        h_after_crop = h_after_crop - 2
        w_after_crop = w_after_crop - 2

    out_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    print("Setting up core range for num_cores: ", out_core_range_set.num_cores())
    print(
        f"Setting up ttnn:conv_crop pre_crop_tensor_h: {h_after_knit}, pre_crop_tensor_w: {w_after_knit}, post_crop_tensor_h: {h_after_crop}, post_crop_tensor_w: {w_after_crop}"
    )
    out_shard_shape = (
        h_after_crop * w_after_crop // out_core_range_set.num_cores(),
        tt_knited_tensor.padded_shape[3],
    )

    post_crop_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            out_core_range_set,
            out_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    print(f"Setting up shard shape: {out_shard_shape}")

    crop_h = padding[0]
    crop_w = padding[1]
    if is_conv3:
        crop_h += 1
        crop_w += 1

    tt_knited_tensor = ttnn.conv_crop(
        tt_knited_tensor, post_crop_mem_config, crop_h, crop_w, h_after_knit, w_after_knit
    )

    if is_conv3:
        padding=(3, 3)

    tt_split_knit_out = ttnn.to_torch(tt_knited_tensor, mesh_composer=None)
    tt_split_knit_out = tt_split_knit_out.reshape([1, h_after_knit - 2 * padding[0], w_after_knit - 2 * padding[1], output_channels])
    tt_split_knit_out = tt_split_knit_out.permute(0, 3, 1, 2) # NCHW
    # Validate ttnn split-knit /w fused output against golden torch
    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_out_golden_tensor, tt_split_knit_out, pcc=pcc)
    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing, pcc_msg
