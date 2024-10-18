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
from models.experimental.functional_yolov11.reference.yolov11 import C3k as torch_c3k
from models.experimental.functional_yolov11.tt.ttnn_yolov11 import C3K as ttnn_c3k


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        (
            [64, 64, 64, 32, 32, 32, 32],
            [32, 32, 64, 32, 32, 32, 32],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 64, 14, 14],
        ),
        (
            [128, 128, 128, 64, 64, 64, 64],
            [64, 64, 128, 64, 64, 64, 64],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 128, 7, 7],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_c3k(
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
    fwd_input_shape,
):
    torch_module = torch_c3k(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    print(parameters.conv_args, parameters)
    ttnn_module = ttnn_c3k(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)  # ttnn.Shape([1, 1, 224, 64])
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99999)
