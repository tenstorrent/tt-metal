# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import torch.nn as nn
from models.experimental.functional_Ultralane_detection_V2.tt.ttnn_Ultralane_fast_detection_v2 import ttnn_Basic_Block
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d, infer_ttnn_module_args
from models.experimental.functional_Ultralane_detection_V2.reference.tu_simple_model import Tu_Simple, BasicBlock
from tests.ttnn.utils_for_testing import assert_with_pcc


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, BasicBlock):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        weight, bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
        parameters["conv2"] = {}
        parameters["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 64, 80, 200),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_tu_simple_res34_basic_block(device, batch_size, input_channels, height, width):
    torch_model = Tu_Simple(input_height=height, input_width=width).model.layer1[0]
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width), dtype=torch.bfloat16)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=device
    )
    ttnn_model = ttnn_Basic_Block(parameters.conv_args, parameters, device=device)
    torch_out = torch_model(torch_input_tensor)
    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)
