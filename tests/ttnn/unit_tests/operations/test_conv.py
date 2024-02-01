# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn
import math


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

    reader_patterns_cache = {}

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    conv = ttnn.Conv2D(
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
    )

    assert "conv" in reader_patterns_cache and "halo" in reader_patterns_cache

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
    tt_input_tensor_on_device = conv.copy_input_to_device(tt_input_tensor)
    tt_output_tensor_on_device = conv(tt_input_tensor_on_device)
    tt_output_tensor = conv.copy_output_from_device(tt_output_tensor_on_device)

    assert tt_output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    reader_patterns_cache.clear()

    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

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
            ttnn.Conv2D(
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
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi])
def test_resnet50_conv(
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
    if input_channels == 16:
        pytest.skip("These tests are hanging in interleaved_to_sharded after rebase. Issue: #4336")

    if math_fidelity != ttnn.MathFidelity.LoFi:
        pytest.skip(
            "By default, only run tests with LoFi math for pipelines. For local unit testing, enable the other variants by uncommenting the skip here!"
        )

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

    if (
        input_channels >= 320
        and (not input_channels == 512)
        and (activations_dtype == ttnn.bfloat16 or weights_dtype == ttnn.bfloat16)
    ):
        pytest.skip("Skipping tests with bfloat16 for sd convs")

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
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi])
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
):
    if math_fidelity != ttnn.MathFidelity.LoFi:
        pytest.skip(
            "By default, only run tests with LoFi math for pipelines. For local unit testing, enable the other variants by uncommenting the skip here!"
        )
    if input_channels > 1280 or (input_channels > 640 and input_height > 16):
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
        # (2, 16, 48, 528, 80, 3, 3, 1, 1, 1, 1, True, {"act_block_h": 32}), # fails. mismatch. It passes when input_channels=64. Probably an issue with padding when input_channels % 32 != 0.
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
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi])
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
    if math_fidelity != ttnn.MathFidelity.LoFi:
        pytest.skip(
            "By default, only run tests with LoFi math for pipelines. For local unit testing, enable the other variants by uncommenting the skip here!"
        )
    if input_channels == 3:
        # use shallow conv variant for first conv only
        # TODO: add automatic padding with 0s in the unit test
        input_channels = 16
    elif input_channels < 32:
        # this is an intermediate conv. The shape would already be padded to 32 (tile shape) by previous op
        # TODO: add automatic padding with 0s in the unit test
        input_channels = 32
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
