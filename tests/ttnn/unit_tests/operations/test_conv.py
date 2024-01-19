# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


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
    ids=["batch_size_8", "batch_size_16", "batch_size_20"],
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["weights_BFLOAT16", "weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["activations_BFLOAT16", "activations_BFLOAT8_B"],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi], ids=["HiFi4", "LoFi"])
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
        # (2, 32, 16, 64, 64, 3, 3, 1, 1, 1, 1, True, None), # not supported
        (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, False, None),
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),  #
        (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 weights doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit.
        (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 weights doesnt fit
        (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, None),
        (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        # sd convs with HxW=64x64 with batch size=2
        (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
        (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, False, None),  # fits with bfloat8_b
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, False, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, False, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, False, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 32}),
        (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),  # bfloat16 doesnt fit
        (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, False, {"act_block_h": 64}),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
    ids=["weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
    ids=["activations_BFLOAT8_B"],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi], ids=["HiFi4", "LoFi"])
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
