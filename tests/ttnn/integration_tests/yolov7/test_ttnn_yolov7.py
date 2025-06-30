# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.yolov7.reference import yolov7_model, yolov7_utils
from models.demos.yolov7.reference.model import Yolov7_model
from models.demos.yolov7.reference.yolov7_utils import download_yolov7_weights
from models.demos.yolov7.tt.ttnn_yolov7 import ttnn_yolov7
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.yolov7.ttnn_yolov7_utils import (
    create_custom_preprocessor,
    load_weights,
    create_yolov7_input_tensors,
)

sys.modules["models.common"] = yolov7_utils
sys.modules["models.yolo"] = yolov7_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov7(device, reset_seeds):
    torch_model = Yolov7_model()

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    weights_path = "tests/ttnn/integration_tests/yolov7/yolov7.pt"
    weights_path = download_yolov7_weights(weights_path)
    load_weights(torch_model, weights_path)

    torch_input, ttnn_input = create_yolov7_input_tensors(device, model=True)
    torch_output_tensor = torch_model(torch_input)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    nx_ny = [80, 40, 20]
    grid_tensors = []
    for i in range(3):
        yv, xv = torch.meshgrid([torch.arange(nx_ny[i]), torch.arange(nx_ny[i])])
        grid_tensors.append(torch.stack((xv, yv), 2).view((1, 1, nx_ny[i], nx_ny[i], 2)).float())

    ttnn_model = ttnn_yolov7(device, parameters, grid_tensors)
    output = ttnn_model(ttnn_input)[0]

    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output_tensor[0], output, pcc=0.999)
