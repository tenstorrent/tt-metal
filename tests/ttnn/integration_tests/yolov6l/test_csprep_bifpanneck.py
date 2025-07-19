# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.demos.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters, load_torch_model_yolov6l
from models.demos.yolov6l.tt.ttnn_csprep_bifpanneck import TtCSPRepBiFPANNeck
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l_csprep_bifpanneck(device, reset_seeds):
    model = load_torch_model_yolov6l()

    model = model.neck

    torch_input_0 = torch.randn(1, 128, 160, 160)
    torch_input_1 = torch.randn(1, 256, 80, 80)
    torch_input_2 = torch.randn(1, 512, 40, 40)
    torch_input_3 = torch.randn(1, 1024, 20, 20)

    parameters = create_yolov6l_model_parameters(
        model, [torch_input_0, torch_input_1, torch_input_2, torch_input_3], device
    )

    ttnn_model = TtCSPRepBiFPANNeck(device, parameters, parameters.model_args)

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
    input_tensor_3 = torch.permute(torch_input_3, (0, 2, 3, 1))
    ttnn_input_3 = ttnn.from_torch(
        input_tensor_3,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn_model([ttnn_input_0, ttnn_input_1, ttnn_input_2, ttnn_input_3])

    torch_output = model([torch_input_0, torch_input_1, torch_input_2, torch_input_3])

    output_0 = ttnn.to_torch(output[0])
    output_0 = output_0.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output[0], output_0, pcc=0.99)

    output_1 = ttnn.to_torch(output[1])
    output_1 = output_1.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output[1], output_1, pcc=0.99)

    output_2 = ttnn.to_torch(output[2])
    output_2 = output_2.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output[2], output_2, pcc=0.99)
