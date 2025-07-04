# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from ultralytics import YOLO

from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.yolov12x.reference import yolov12x
from models.experimental.yolov12x.tt.yolov12x import YoloV12x
from models.experimental.yolov12x.tt.model_preprocessing import (
    create_yolov12x_input_tensors,
    create_yolov12x_model_parameters,
)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768}],
    indirect=True,
    ids=["0"],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
def test_yolov12x(use_pretrained_weight, device, reset_seeds):
    torch_input, ttnn_input = create_yolov12x_input_tensors(device)
    state_dict = None

    if use_pretrained_weight:
        torch_model = YOLO("yolo12x.pt")
        state_dict = torch_model.state_dict()

    torch_model = yolov12x.YoloV12x()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)

    torch_model.eval()
    torch_output = torch_model(torch_input)
    parameters = create_yolov12x_model_parameters(torch_model, torch_input, device)
    ttnn_model = YoloV12x(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output[0])

    assert_with_pcc(torch_output[0], ttnn_output, 0.99)
