# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.yolov11s.common import YOLOV11S_L1_SMALL_SIZE
from models.demos.yolov11s.reference.yolov11s import Detect as torch_detect
from models.demos.yolov11s.tests.pcc.pcc_logging import log_assert_with_pcc
from models.demos.yolov11s.tt.model_preprocessing import (
    create_yolov11s_input_tensors,
    create_yolov11s_model_parameters_detect,
)
from models.demos.yolov11s.tt.ttnn_yolov11s_detect import TtnnDetect as ttnn_detect


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        (
            [
                128,
                64,
                64,
                256,
                64,
                64,
                512,
                64,
                64,
                128,
                128,
                128,
                128,
                128,
                256,
                256,
                128,
                128,
                128,
                512,
                512,
                128,
                128,
                128,
                16,
            ],
            [
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                64,
                128,
                128,
                128,
                128,
                80,
                256,
                128,
                128,
                128,
                80,
                512,
                128,
                128,
                128,
                80,
                1,
            ],
            [3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 128, 1, 128, 1, 1, 256, 1, 128, 1, 1, 512, 1, 128, 1, 1, 1],
            [[1, 128, 40, 40], [1, 256, 20, 20], [1, 512, 10, 10]],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV11S_L1_SMALL_SIZE}], indirect=True)
def test_yolo_v11_detect(
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
    torch_module = torch_detect(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input_1, ttnn_input_1 = create_yolov11s_input_tensors(
        device,
        batch=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov11s_input_tensors(
        device,
        batch=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov11s_input_tensors(
        device,
        batch=fwd_input_shape[2][0],
        input_channels=fwd_input_shape[2][1],
        input_height=fwd_input_shape[2][2],
        input_width=fwd_input_shape[2][3],
    )
    ttnn_input_1 = ttnn.to_device(ttnn_input_1, device=device)
    ttnn_input_1 = ttnn.to_layout(ttnn_input_1, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input_2 = ttnn.to_device(ttnn_input_2, device=device)
    ttnn_input_2 = ttnn.to_layout(ttnn_input_2, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input_3 = ttnn.to_device(ttnn_input_3, device=device)
    ttnn_input_3 = ttnn.to_layout(ttnn_input_3, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input_1, torch_input_2, torch_input_3)
    parameters = create_yolov11s_model_parameters_detect(
        torch_module, torch_input_1, torch_input_2, torch_input_3, device=device
    )
    ttnn_module = ttnn_detect(device=device, parameter=parameters.model, conv_pt=parameters)

    ttnn_output = ttnn_module(y1=ttnn_input_1, y2=ttnn_input_2, y3=ttnn_input_3, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    log_assert_with_pcc("YOLOv11s Detect", torch_output, ttnn_output, 0.99)
