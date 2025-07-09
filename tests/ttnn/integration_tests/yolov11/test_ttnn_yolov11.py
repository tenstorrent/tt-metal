# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ultralytics import YOLO
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.demos.yolov11.reference import yolov11
from models.demos.yolov11.tt import ttnn_yolov11


@pytest.mark.parametrize(
    "resolution",
    [
        ([1, 3, 640, 640]),
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        # True,
        False
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov11(device, reset_seeds, resolution, use_weights_from_ultralytics, min_channels=8):
    weights = "yolo11n.pt"
    if use_weights_from_ultralytics:
        torch_model = YOLO(weights)
        state_dict = {k.replace("model.", "", 1): v for k, v in torch_model.state_dict().items()}
    torch_model = yolov11.YoloV11()
    torch_model.eval()
    if use_weights_from_ultralytics:
        torch_model.load_state_dict(state_dict)
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=resolution[0],
        input_channels=resolution[1],
        input_height=resolution[2],
        input_width=resolution[3],
        is_sub_module=False,
    )
    n, c, h, w = ttnn_input.shape
    if c == 3:  # for sharding config of padded input
        c = min_channels
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn_input.to(device, input_mem_config)
    torch_output = torch_model(torch_input)
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov11.TtnnYoloV11(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
