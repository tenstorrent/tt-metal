# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.demos.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters, load_torch_model_yolov6l
from models.demos.yolov6l.tt.ttnn_bottlerep import TtBottleRep
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l_bottlerep(device, reset_seeds):
    model = load_torch_model_yolov6l()

    model = model.backbone.ERBlock_2[1].m.conv1

    torch_input = torch.randn(1, 64, 160, 160)
    torch_input_1 = torch_input.reshape(
        1, torch_input.shape[1], 1, torch_input.shape[0] * torch_input.shape[2] * torch_input.shape[3]
    )

    parameters = create_yolov6l_model_parameters(model, torch_input, device)

    ttnn_model = TtBottleRep(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input_1, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output, out_h, out_w = ttnn_model(ttnn_input)

    torch_output = model(torch_input)

    output = ttnn.to_torch(output)
    output = output.reshape(1, out_h, out_w, output.shape[-1])
    output = output.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output, output, pcc=0.99)
