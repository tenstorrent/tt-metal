# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.experimental.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters, load_torch_model_yolov6l
from models.experimental.yolov6l.tt.ttnn_cspbep_backbone import TtCSPBepBackbone
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l_cspbep_backbone(device, reset_seeds):
    model = load_torch_model_yolov6l()

    model = model.backbone

    torch_input = torch.randn(1, 3, 640, 640)

    parameters = create_yolov6l_model_parameters(model, torch_input, device)

    ttnn_model = TtCSPBepBackbone(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output = ttnn_model(ttnn_input)

    torch_output = model(torch_input)

    output_0 = ttnn.to_torch(output[0])
    output_0 = output_0.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output[0], output_0, pcc=0.99)

    output_1 = ttnn.to_torch(output[1])
    output_1 = output_1.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output[1], output_1, pcc=0.99)

    output_2 = ttnn.to_torch(output[2])
    output_2 = output_2.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output[2], output_2, pcc=0.99)

    output_3 = ttnn.to_torch(output[3])
    output_3 = output_3.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output[3], output_3, pcc=0.99)
