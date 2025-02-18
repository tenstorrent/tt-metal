# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from ..conv_test_utils import *


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, dilation, shard_layout",
    (
        (1, 48, 32, 252, 252, 3, 3, 1, 1, 0, 0, 2, HS),
        (1, 56, 48, 248, 248, 3, 3, 1, 1, 0, 0, 4, HS),
        (1, 64, 56, 240, 240, 3, 3, 1, 1, 0, 0, 8, HS),
        (1, 48, 32, 124, 124, 3, 3, 1, 1, 0, 0, 2, HS),
        (1, 56, 48, 120, 120, 3, 3, 1, 1, 0, 0, 4, HS),
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
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_model_k_256x256(
    device,
    torch_tensor_map,
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
    dilation,
    shard_layout,
    auto_shard,
):
    run_conv(
        device,
        torch_tensor_map,
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
        None,
        shard_layout=shard_layout,
        dilation=dilation,
        auto_shard=auto_shard,
    )
