# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import torch.nn as nn
from models.experimental.functional_UFLD_v2.ttnn.ttnn_UFLD_v2 import ttnn_UFLD_V2_Conv2D
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.experimental.functional_UFLD_v2.reference.UFLD_v2_model import Tu_Simple
from tests.ttnn.utils_for_testing import assert_with_pcc


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        else:
            parameters["bias"] = None

    return parameters


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_UFLD_V2_conv(device, batch_size, input_channels, height, width):
    torch_model = Tu_Simple(input_height=height, input_width=width).res_model.conv1
    torch_model.to(torch.bfloat16)
    torch_model.eval()
    torch_input_tensor = torch.randn((batch_size, input_channels, height, width), dtype=torch.bfloat16)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda model: torch_model(torch_input_tensor), device=None
    )
    ttnn_model = ttnn_UFLD_V2_Conv2D(parameters.conv_args, parameters, activation="", device=device)
    torch_out = torch_model(torch_input_tensor)
    ttnn_output = ttnn_model(ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output[0])
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_out.shape)
    assert_with_pcc(ttnn_output, torch_out, 0.9999)
