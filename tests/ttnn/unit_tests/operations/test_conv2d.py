# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
import tt_lib
import math


def prepare_conv_input_and_copy_to_device_interleaved(
    device, torch_input_tensor_nhwc, input_tensor_shape, use_shallow_conv_variant
):
    # Pad for 16 byte alignnment
    # TODO: for bfp16, pad to 8 only
    padded_input_channels = math.ceil(torch_input_tensor_nhwc.shape[3] / 16) * 16
    torch_input_tensor_nhwc = torch.nn.functional.pad(
        torch_input_tensor_nhwc, (0, 0, 0, 0, 0, padded_input_channels - torch_input_tensor_nhwc.shape[3])
    )
    # Reshape 4d to 2d
    torch_input_tensor_nhwc = torch.reshape(
        torch_input_tensor_nhwc,
        (1, 1, input_tensor_shape[0] * input_tensor_shape[1] * input_tensor_shape[2], input_tensor_shape[3]),
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor_nhwc, ttnn.bfloat16)
    tt_input_tensor_on_device = ttnn.to_device(tt_input_tensor, device)

    if not use_shallow_conv_variant:
        tt_input_tensor_on_device = ttnn.to_layout(tt_input_tensor_on_device, ttnn.TILE_LAYOUT)
    return tt_input_tensor_on_device


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
    use_shallow_conv_variant=False,
    enable_auto_formatting=False,
):
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
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
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    conv = ttnn.Conv2d(
        input_channels,
        output_channels,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dtype=activations_dtype,
        device=device,
        use_1d_systolic_array=use_1d_systolic_array,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        reader_patterns_cache=reader_patterns_cache,
        weight=tt_weight_tensor,
        bias=tt_bias_tensor,
        math_fidelity=math_fidelity,
        weights_dtype=weights_dtype,
        conv_blocking_and_parallelization_config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        enable_auto_formatting=enable_auto_formatting,
        deallocate_activation=True,
    )

    assert "conv" in reader_patterns_cache and "halo" in reader_patterns_cache
    if enable_auto_formatting:
        tt_input_tensor_on_device = prepare_conv_input_and_copy_to_device_interleaved(
            device,
            torch_input_tensor,
            [batch_size, input_height, input_width, input_channels],
            use_shallow_conv_variant,
        )
    else:
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        tt_input_tensor_on_device = conv.copy_input_to_device(tt_input_tensor)
    tt_output_tensor_on_device = conv(tt_input_tensor_on_device)
    if enable_auto_formatting:
        tt_output_tensor_on_device = ttnn.to_layout(tt_output_tensor_on_device, ttnn.ROW_MAJOR_LAYOUT)
        tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
        torch_output_tensor = ttnn.to_torch(tt_output_tensor)
        torch_output_tensor = torch.split(torch_output_tensor, output_channels, 3)[0]
        torch_output_tensor = torch.reshape(torch_output_tensor, output_shape_nhwc)
    else:
        tt_output_tensor = conv.copy_output_from_device(tt_output_tensor_on_device)
        assert tt_output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()
    if math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998
    assert_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)


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

    convs = []
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
        convs.append(
            ttnn.Conv2d(
                split_input_channels,
                output_channels,
                kernel_size=(filter_height, filter_width),
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
                dtype=activations_dtype,
                device=device,
                use_1d_systolic_array=use_1d_systolic_array,
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                reader_patterns_cache=reader_patterns_cache,
                weight=tt_weight_tensor,
                bias=tt_bias_tensor,
                math_fidelity=math_fidelity,
                weights_dtype=weights_dtype,
                conv_blocking_and_parallelization_config_override=config_override,
            )
        )

    torch_output_tensor = None
    for i in range(split_factor):
        torch_input_tensor = torch.permute(split_input_tensors[i], (0, 2, 3, 1))
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        tt_input_tensor_on_device = convs[i].copy_input_to_device(tt_input_tensor)
        tt_output_tensor_on_device = convs[i](tt_input_tensor_on_device)
        tt_output_tensor = convs[i].copy_output_from_device(tt_output_tensor_on_device)
        torch_conv_output_tensor = ttnn.to_torch(tt_output_tensor)
        # torch_output_tensor is in row major layout and NHWC shape
        # NHWC to NCHW
        torch_conv_output_tensor = torch.permute(torch_conv_output_tensor, (0, 3, 1, 2))
        if i == 0:
            torch_output_tensor = torch_conv_output_tensor
        else:
            torch_output_tensor = torch.add(torch_output_tensor, torch_conv_output_tensor)

    if math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998
    assert_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        (64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True),
        # rn50 layer1
        (64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True),
        # rn50 layer2
        (128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True),
        (128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True),
        # rn50 layer3
        (256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False),
        (256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False),
        # rn50 layer4
        (512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False),
        (512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False),
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
    use_program_cache,
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
        config_override=None,
        use_shallow_conv_variant=input_channels == 16,
    )


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override",
    (
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, None),
        (16, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, {"act_block_h": 32}),
        (20, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True, {"act_block_h": 32}),
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
def test_resnet50_conv_wh(
    use_program_cache,
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

    use_shallow_conv_variant = (input_channels == 16) and device.arch() != tt_lib.device.Arch.WORMHOLE_B0
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
    )


@skip_for_wormhole_b0()
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
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
        (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, False, None),
        (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, False, None),
        (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
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
    use_program_cache,
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
    enable_auto_formatting,
):
    if input_channels > 1280 or (input_channels > 640 and input_height > 16):
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
        )


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override",
    (
        # unet convs with batch size 2
        # unique convs in unet (complete list)
        (2, 16, 3, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 64}),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 64}),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, True, None),
        (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, True, None),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, True, None),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, True, None),
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, True, None),
        (2, 64, 64, 66, 10, 3, 3, 1, 1, 1, 1, True, None),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, True, None),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, True, None),
        (2, 32, 64, 264, 40, 3, 3, 1, 1, 1, 1, True, None),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, True, None),
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
        ),  # fails. mismatch. It passes when input_channels=64. Probably an issue with padding when input_channels % 32 != 0.
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, True, None),
        (2, 16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 22 * 32}),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 22 * 32}),
        (2, 1, 16, 1056, 160, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 22 * 32}),
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
def test_unet_conv(
    use_program_cache,
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
):
    use_shallow_conv_variant = False
    if input_channels == 3:
        # use shallow conv variant for first conv only
        use_shallow_conv_variant = True
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
    )
