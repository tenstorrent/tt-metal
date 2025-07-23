# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.demos.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters, load_torch_model_yolov6l
from models.demos.yolov6l.tt.ttnn_bifusion import TtBiFusion
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l_bifusion(device, reset_seeds):
    model = load_torch_model_yolov6l()

    model = model.neck.Bifusion0

    torch_input_0 = torch.randn(1, 256, 20, 20)
    torch_input_1 = torch.randn(1, 512, 40, 40)
    torch_input_2 = torch.randn(1, 256, 80, 80)

    parameters = create_yolov6l_model_parameters(model, [torch_input_0, torch_input_1, torch_input_2], device)

    ttnn_model = TtBiFusion(device, parameters, parameters.model_args)

    input_tensor_0 = torch_input_0.reshape(1, 256, 1, 400)
    input_tensor_0 = torch.permute(input_tensor_0, (0, 2, 3, 1))
    ttnn_input_0 = ttnn.from_torch(
        input_tensor_0,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensor_1 = torch_input_1.reshape(1, 512, 1, 1600)
    input_tensor_1 = torch.permute(torch_input_1, (0, 2, 3, 1))
    ttnn_input_1 = ttnn.from_torch(
        input_tensor_1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensor_2 = torch_input_2.reshape(1, 256, 1, 6400)
    input_tensor_2 = torch.permute(input_tensor_2, (0, 2, 3, 1))
    ttnn_input_2 = ttnn.from_torch(
        input_tensor_2,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output, out_h, out_w = ttnn_model([ttnn_input_0, ttnn_input_1, ttnn_input_2])

    torch_output = model([torch_input_0, torch_input_1, torch_input_2])

    output = ttnn.to_torch(output)
    output = output.reshape(1, out_h, out_w, output.shape[-1])
    output = output.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output, output, pcc=0.99)
