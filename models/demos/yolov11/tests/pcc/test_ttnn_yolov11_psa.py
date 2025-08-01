# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE
from models.demos.yolov11.reference.yolov11 import PSABlock as torch_psa_block
from models.demos.yolov11.tt.model_preprocessing import create_yolov11_input_tensors, create_yolov11_model_parameters
from models.demos.yolov11.tt.ttnn_yolov11_psa import TtnnPSABlock as ttnn_psa_block
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        (
            [128, 128, 128, 128, 256],
            [256, 128, 128, 256, 128],
            [1, 1, 3, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 128, 1, 1],
            [1, 128, 7, 7],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11_L1_SMALL_SIZE}], indirect=True)
def test_yolo_v11_psa_block(
    device,
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
    torch_module = torch_psa_block(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_psa_block(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
