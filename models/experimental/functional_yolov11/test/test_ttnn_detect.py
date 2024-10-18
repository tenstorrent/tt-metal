# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters_detect,
    create_yolov11_model_parameters,
)
from models.experimental.functional_yolov11.reference.yolov11 import Detect as torch_detect
from models.experimental.functional_yolov11.tt.ttnn_yolov11 import Detect as ttnn_detect
from ttnn.model_preprocessing import preprocess_model_parameters
import math


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        (
            [64, 64, 64, 128, 64, 64, 256, 64, 64, 64, 64, 80, 80, 80, 128, 128, 80, 80, 80, 256, 256, 80, 80, 80, 16],
            [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 80, 80, 80, 80, 128, 80, 80, 80, 80, 256, 80, 80, 80, 80, 1],
            [3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 1, 80, 1, 1, 128, 1, 80, 1, 1, 256, 1, 80, 1, 1, 1],
            [[1, 64, 28, 28], [1, 128, 14, 14], [1, 256, 7, 7]],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_detect(
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
    torch_module = torch_detect(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input_1, ttnn_input_1 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[2][0],
        input_channels=fwd_input_shape[2][1],
        input_height=fwd_input_shape[2][2],
        input_width=fwd_input_shape[2][3],
    )
    ttnn_input_1 = ttnn.to_device(ttnn_input_1, device=device)
    ttnn_input_1 = ttnn.to_layout(ttnn_input_1, layout=ttnn.TILE_LAYOUT)
    ttnn_input_2 = ttnn.to_device(ttnn_input_2, device=device)
    ttnn_input_2 = ttnn.to_layout(ttnn_input_2, layout=ttnn.TILE_LAYOUT)
    ttnn_input_3 = ttnn.to_device(ttnn_input_3, device=device)
    ttnn_input_3 = ttnn.to_layout(ttnn_input_3, layout=ttnn.TILE_LAYOUT)
    torch_output = torch_module(torch_input_1, torch_input_2, torch_input_3)
    parameters = create_yolov11_model_parameters_detect(
        torch_module, torch_input_1, torch_input_2, torch_input_3, device=device
    )
    ttnn_module = ttnn_detect(device=device, parameter=parameters.model, conv_pt=parameters)

    ttnn_output = ttnn_module(y1=ttnn_input_1, y2=ttnn_input_2, y3=ttnn_input_3, device=device)
    print("shape outputs", ttnn_output.shape, torch_output.shape)
    ttnn_output = ttnn.to_torch(ttnn_output)
    # ttnn_output = ttnn_output.permute(0, 2, 1)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99999)
