# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.yolov6l.common import YOLOV6L_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters_detect
from models.demos.yolov6l.tt.ttnn_detect import TtDetect
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV6L_L1_SMALL_SIZE}], indirect=True)
def test_detect(device, reset_seeds, model_location_generator):
    model = load_torch_model(model_location_generator)

    model = model.detect

    torch_input_0 = torch.randn(1, 128, 80, 80)
    torch_input_1 = torch.randn(1, 256, 40, 40)
    torch_input_2 = torch.randn(1, 512, 20, 20)

    parameters = create_yolov6l_model_parameters_detect(model, [torch_input_0, torch_input_1, torch_input_2], device)

    ttnn_model = TtDetect(device, parameters, parameters.model_args)

    input_tensor_0 = torch.permute(torch_input_0, (0, 2, 3, 1))
    ttnn_input_0 = ttnn.from_torch(
        input_tensor_0,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensor_1 = torch.permute(torch_input_1, (0, 2, 3, 1))
    ttnn_input_1 = ttnn.from_torch(
        input_tensor_1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensor_2 = torch.permute(torch_input_2, (0, 2, 3, 1))
    ttnn_input_2 = ttnn.from_torch(
        input_tensor_2,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn_model([ttnn_input_0, ttnn_input_1, ttnn_input_2])

    torch_output = model([torch_input_0, torch_input_1, torch_input_2])

    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output, output, pcc=0.99)
