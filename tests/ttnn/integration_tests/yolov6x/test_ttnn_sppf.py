# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_yolov6x.reference.yolov6x import SPPF
from models.experimental.functional_yolov6x.tt.ttnn_sppf import Ttnn_Sppf
from models.experimental.functional_yolov6x.tt.model_preprocessing import create_yolov6x_model_parameters_sppf


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov6x_sppf(device, reset_seeds):
    torch_model = SPPF(640, 640)
    torch_model.eval()
    torch_input = torch.randn(1, 640, 20, 20)

    parameters = create_yolov6x_model_parameters_sppf(torch_model, torch_input, device)

    ttnn_model = Ttnn_Sppf(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn_model(device, ttnn_input)

    torch_output = torch_model(torch_input)

    output = ttnn.to_torch(output)
    output = output.permute(0, 3, 1, 2)
    output = output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, output, pcc=0.999)  # PCC: 0.9998703105881518
