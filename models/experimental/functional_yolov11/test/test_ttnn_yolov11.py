# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_yolov11.reference import yolov11

# from models.experimental.functional_yolov11.reference.yolov11 import C3k2
# from models.experimental.functional_yolov11.tt.ttnn_yolov11 import C3k2

from models.experimental.functional_yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.experimental.functional_yolov11.tt import ttnn_yolov11
import torch.nn as nn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov11(device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_yolov11_input_tensors(device)

    torch_model = yolov11.YoloV11()
    torch_model.eval()
    print(torch_model)
    torch_output = torch_model(torch_input)

    # parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)

    # ttnn_model = ttnn_yolov11.YoloV11(device, parameters)

    # ttnn_output = ttnn_model(ttnn_input)

    # ttnn_output = ttnn.to_torch(ttnn_output)
    # ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    # ttnn_output = ttnn_output.reshape(torch_output.shape)
    # assert_with_pcc(torch_output, ttnn_output, 0.99999)


# @pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
# def test_C3k2(device, use_program_cache, reset_seeds):
#     torch_input, ttnn_input = create_yolov11_input_tensors(device)

#     torch_model = yolov11.YoloV11()
#     torch_model.eval()

#     parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
#     parameters = parameters.c3k2_2

#     torch_model = yolov11.C3k2( [64, 96, 32, 16],
#                         [64, 128, 16, 32],
#                         [1, 1, 3, 3],
#                         [1, 1, 1, 1],
#                         [0, 0, 1, 1],
#                         [1, 1, 1, 1],
#                         [1, 1, 1, 1],
#                         is_bk_enabled=True,
#                     )

#     torch_input_tensor = torch.randn(1, 64, 28, 28)
#     ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
#     ttnn_input_tensor = ttnn_input_tensor.reshape(
#         1,
#         1,
#         ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
#         ttnn_input_tensor.shape[3],
#     )
#     ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)


#     # parameters = create_yolov11_model_parameters(torch_model, torch_input_tensor, device=device)

#     torch_output = torch_model(torch_input_tensor)

#     ttnn_model = ttnn_yolov11.C3k2(device, parameters, parameters.module, is_bk_enabled=True)
#     ttnn_output = ttnn_model(device, ttnn_input_tensor)


#     ttnn_output = ttnn.to_torch(ttnn_output)
#     ttnn_output = ttnn_output.permute(0, 3, 1, 2)
#     ttnn_output = ttnn_output.reshape(torch_output.shape)
#     assert_with_pcc(torch_output, ttnn_output, 0.99999)
