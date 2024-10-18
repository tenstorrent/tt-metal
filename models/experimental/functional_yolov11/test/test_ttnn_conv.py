# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.experimental.functional_yolov11.reference.yolov11 import Conv as torch_conv
from models.experimental.functional_yolov11.tt.ttnn_yolov11 import Conv as ttnn_conv


# @pytest.mark.parametrize(
#     "in_channel, out_channel, kernel, stride, padding, dilation, groups, enable_act, fwd_input_shape",
#     [(3, 16, 3, 2, 1, 1, 1, True, (1, 3, 224, 224))]
# )


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups, enable_act, fwd_input_shape",
    (
        (3, 16, 3, 2, 1, 1, 1, True, (1, 3, 224, 224)),
        # (16, 32, 3, 2, 1, 1, 1, True, (1, 16, 112, 112)),
        # (32, 32, 1, 1, 0, 1, 1, True, (1, 32, 56, 56)),
        # (48, 64, 1, 1, 0, 1, 1, True, (1, 16, 56, 56)),
        # (16, 8, 3, 1, 1, 1, 1, True, (1, 8, 56, 56)),
        # (8, 16, 3, 1, 1, 1, 1, True, (1, 48, 56, 56)),
        # (64, 64, 3, 2, 1, 1, 1, True, (1, 64, 56, 56)),
        # (64, 64, 1, 1, 0, 1, 1, True, (1, 64, 28, 28)),
        # (96, 128, 1, 1, 0, 1, 1, True, (1, 32, 28, 28)),
        # (32, 16, 3, 1, 1, 1, 1, True, (1, 16, 28, 28)),
        # (16, 32, 3, 1, 1, 1, 1, True, (1, 96, 28, 28)),
        # (128, 128, 3, 2, 1, 1, 1, True, (1, 128, 28, 28)),
        # (128, 128, 1, 1, 0, 1, 1, True, (1, 128, 14, 14)),
        # (192, 128, 1, 1, 0, 1, 1, True, (1, 64, 14, 14)),
        # (64, 32, 1, 1, 0, 1, 1, True, (1, 64, 14, 14)),
        # (64, 32, 1, 1, 0, 1, 1, True, (1, 32, 14, 14)),
        # (64, 64, 1, 1, 0, 1, 1, True, (1, 32, 14, 14)),
        # (32, 32, 3, 1, 1, 1, 1, True, (1, 32, 14, 14)),
        # (32, 32, 3, 1, 1, 1, 1, True, (1, 64, 14, 14)),
        # (32, 32, 3, 1, 1, 1, 1, True, (1, 192, 14, 14)),
        # (128, 256, 3, 2, 1, 1, 1, True, (1, 128, 14, 14)),
        # (256, 256, 1, 1, 0, 1, 1, True, (1, 256, 7, 7)),
        # (384, 256, 1, 1, 0, 1, 1, True, (1, 128, 7, 7)),
        # (128, 64, 1, 1, 0, 1, 1, True, (1, 128, 7, 7)),
        # (128, 64, 1, 1, 0, 1, 1, True, (1, 64, 7, 7)),
        # (128, 128, 1, 1, 0, 1, 1, True, (1, 64, 7, 7)),
        # (64, 64, 3, 1, 1, 1, 1, True, (1, 64, 7, 7)),
        # (64, 64, 3, 1, 1, 1, 1, True, (1, 128, 7, 7)),
        # (64, 64, 3, 1, 1, 1, 1, True, (1, 384, 7, 7)),
        # (256, 128, 1, 1, 0, 1, 1, True, (1, 256, 7, 7)),
        # (512, 256, 1, 1, 0, 1, 1, True, (1, 512, 7, 7)),
        # (256, 256, 1, 1, 0, 1, 1, True, (1, 128, 7, 7)),
        # (128, 256, 1, 1, 0, 1, 1, False, (1, 128, 7, 7)),
        # (128, 128, 1, 1, 0, 1, 1, False, (1, 128, 7, 7)),
        # (128, 128, 3, 1, 1, 1, 128, False, (1, 128, 7, 7)),
        # (128, 256, 1, 1, 0, 1, 1, True, (1, 256, 7, 7)),
        # (256, 128, 1, 1, 0, 1, 1, False, (1, 256, 7, 7)),
        # (384, 128, 1, 1, 0, 1, 1, True, (1, 384, 14, 14)),
        # (64, 32, 3, 1, 1, 1, 1, True, (1, 32, 14, 14)),
        # (32, 64, 3, 1, 1, 1, 1, True, (1, 192, 14, 14)),
        # (256, 64, 1, 1, 0, 1, 1, True, (1, 256, 28, 28)),
        # (96, 64, 1, 1, 0, 1, 1, True, (1, 32, 28, 28)),
        # (64, 64, 3, 2, 1, 1, 1, True, (1, 64, 28, 28)),
        # (192, 128, 1, 1, 0, 1, 1, True, (1, 192, 14, 14)),
        # (128, 128, 3, 2, 1, 1, 1, True, (1, 128, 14, 14)),
        # (384, 256, 1, 1, 0, 1, 1, True, (1, 384, 7, 7)),
        # (64, 64, 3, 1, 1, 1, 1, True, (1, 64, 28, 28)),
        # (128, 64, 3, 1, 1, 1, 1, True, (1, 128, 14, 14)),
        # (64, 64, 3, 1, 1, 1, 1, True, (1, 64, 14, 14)),
        # (256, 64, 3, 1, 1, 1, 1, True, (1, 256, 7, 7)),
        # (64, 64, 3, 1, 1, 1, 64, True, (1, 64, 28, 28)),
        # (64, 80, 1, 1, 0, 1, 1, True, (1, 64, 28, 28)),
        # (80, 80, 3, 1, 1, 1, 80, True, (1, 80, 28, 28)),
        # (80, 80, 1, 1, 0, 1, 1, True, (1, 80, 28, 28)),
        # (128, 128, 3, 1, 1, 1, 128, True, (1, 128, 14, 14)),
        # (128, 80, 1, 1, 0, 1, 1, True, (1, 128, 14, 14)),
        # (80, 80, 3, 1, 1, 1, 80, True, (1, 80, 14, 14)),
        # (80, 80, 1, 1, 0, 1, 1, True, (1, 80, 14, 14)),
        # (256, 256, 3, 1, 1, 1, 256, True, (1, 256, 7, 7)),
        # (256, 80, 1, 1, 0, 1, 1, True, (1, 256, 7, 7)),
        # (80, 80, 3, 1, 1, 1, 80, True, (1, 80, 7, 7)),
        # (80, 80, 1, 1, 0, 1, 1, True, (1, 80, 7, 7)),
    ),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_conv(
    device,
    use_program_cache,
    reset_seeds,
    in_channel,
    out_channel,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    enable_act,
    fwd_input_shape,
):
    torch_module = torch_conv(in_channel, out_channel, kernel, stride, padding, dilation, groups, enable_act)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_conv(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99999)
