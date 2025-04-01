# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.functional_yolov6x.reference.yolov6x import Yolov6x_model
from models.experimental.functional_yolov6x.tt.ttnn_yolov6x import Ttnn_Yolov6x
from models.experimental.functional_yolov6x.tt.model_preprocessing import create_yolov6x_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov6x(device, reset_seeds):
    torch_model = Yolov6x_model()
    torch_model.eval()
    torch_input = torch.randn(1, 3, 640, 640)

    parameters = create_yolov6x_model_parameters(torch_model, torch_input, device)

    ttnn_model = Ttnn_Yolov6x(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn_model(device, ttnn_input)

    torch_output = torch_model(torch_input)

    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output[0], output, pcc=0.999)  # 0.9999890988262345
