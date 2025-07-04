# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.yolov12x.reference import yolov12x
from models.experimental.yolov12x.tt.detect import TtnnDetect
from models.experimental.yolov12x.tt.model_preprocessing import (
    create_yolov12x_input_tensors,
    create_yolov12x_model_parameters_detect,
)


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups, fwd_input_shape",
    [
        (
            [
                384,
                96,
                96,
                768,
                96,
                96,
                768,
                96,
                96,
                384,
                384,
                384,
                384,
                384,
                768,
                768,
                384,
                384,
                384,
                768,
                768,
                384,
                384,
                384,
                16,
            ],
            [
                96,
                96,
                64,
                96,
                96,
                64,
                96,
                96,
                64,
                384,
                384,
                384,
                384,
                80,
                768,
                384,
                384,
                384,
                80,
                768,
                384,
                384,
                384,
                80,
                1,
            ],
            [3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 384, 1, 384, 1, 1, 768, 1, 384, 1, 1, 768, 1, 384, 1, 1, 1],
            [[1, 384, 80, 80], [1, 768, 40, 40], [1, 768, 20, 20]],
        ),  # 1
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov12x_detect(
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
    torch_module = yolov12x.Detect(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input_1, ttnn_input_1 = create_yolov12x_input_tensors(
        device,
        batch_size=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov12x_input_tensors(
        device,
        batch_size=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov12x_input_tensors(
        device,
        batch_size=fwd_input_shape[2][0],
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
    parameters = create_yolov12x_model_parameters_detect(
        torch_module, torch_input_1, torch_input_2, torch_input_3, device=device
    )
    ttnn_module = TtnnDetect(device=device, parameter=parameters.model, conv_pt=parameters)
    ttnn_output = ttnn_module(y1=ttnn_input_1, y2=ttnn_input_2, y3=ttnn_input_3)
    ttnn_output[0] = ttnn.to_torch(ttnn_output[0])
    ttnn_output[0] = ttnn_output[0].reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output[0], 0.99)
