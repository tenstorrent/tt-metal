# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.yolov9c.common import YOLOV9C_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov9c.reference import yolov9c
from models.demos.yolov9c.tt import ttnn_yolov9c
from models.demos.yolov9c.tt.model_preprocessing import create_yolov9c_input_tensors, create_yolov9c_model_parameters
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",  # To run the demo for instance segmentation
        "detect",  # Uncomment to run the demo for Object Detection
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE}], indirect=True)
def test_yolov9c(use_pretrained_weights, model_task, device, reset_seeds, model_location_generator):
    torch_input, ttnn_input = create_yolov9c_input_tensors(device, model=True)
    state_dict = None

    weights = "yolov9c-seg.pt" if model_task == "segment" else "yolov9c.pt"
    enable_segment = model_task == "segment"

    if use_pretrained_weights:
        torch_model = load_torch_model(model_task, model_location_generator)
    else:
        torch_model = yolov9c.YoloV9(enable_segment=enable_segment)
        state_dict = state_dict if state_dict else torch_model.state_dict()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()

    torch_output = torch_model(torch_input)
    parameters = create_yolov9c_model_parameters(torch_model, torch_input, device)
    ttnn_model = ttnn_yolov9c.YoloV9(device, parameters, enable_segment=enable_segment)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output_ = ttnn.to_torch(ttnn_output[0])
    assert_with_pcc(torch_output[0], ttnn_output_, 0.99)
