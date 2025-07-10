# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.demos.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters, load_torch_model_yolov6l
from models.demos.yolov6l.tt.ttnn_bepc3 import TtBepC3
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l_bepc3(device, reset_seeds):
    model = load_torch_model_yolov6l()

    model = model.backbone.ERBlock_3[1]

    torch_input = torch.randn(1, 256, 80, 80)
    torch_input_1 = torch_input.reshape(
        1, torch_input.shape[1], 1, torch_input.shape[0] * torch_input.shape[2] * torch_input.shape[3]
    )

    parameters = create_yolov6l_model_parameters(model, torch_input, device)

    ttnn_model = TtBepC3(device, parameters, parameters.model_args, n=12)

    input_tensor = torch.permute(torch_input_1, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output = ttnn_model(ttnn_input)

    torch_output = model(torch_input)

    output = ttnn.to_torch(output)
    output = output.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output, output, pcc=0.99)
