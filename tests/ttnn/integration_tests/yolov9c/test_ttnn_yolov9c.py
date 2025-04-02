# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import ttnn
import torch
import pickle
import pytest
import torch.nn as nn
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_yolov9c.tt.model_preprocessing import (
    create_yolov9c_input_tensors,
    create_yolov9c_model_parameters,
    create_yolov9c_model_parameters_detect,
)
from models.experimental.functional_yolov9c.tt import ttnn_yolov9c
from models.experimental.functional_yolov9c.reference import yolov9c
from models.experimental.functional_yolov9c.demo.demo_utils import attempt_load
from ultralytics import YOLO


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9c(use_pretrained_weight, device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_yolov9c_input_tensors(device, model=True)
    state_dict = None

    if use_pretrained_weight:
        torch_model = YOLO("yolov9c.pt")
        state_dict = torch_model.state_dict()

    torch_model = yolov9c.YoloV9()
    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    torch_output = torch_model(torch_input)
    parameters = create_yolov9c_model_parameters(torch_model, torch_input, device)
    ttnn_model = ttnn_yolov9c.YoloV9(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)
