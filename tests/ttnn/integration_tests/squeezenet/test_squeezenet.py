# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import pytest
import torch
import torch.nn as nn
from models.demos.squeezenet.tt.tt_squeezenet import tt_squeezenet
from torchvision import models
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = ttnn.from_torch(model.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)

    return parameters


@pytest.mark.parametrize(
    "batch_size,input_height,input_width,conv_1_params, conv_2_params",
    [
        (1, 224, 224, [3, 96], [512, 1000]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_model(batch_size, input_height, input_width, conv_1_params, conv_2_params, device):
    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    state_dict = torch_squeezenet.state_dict()
    torch_input = torch.randn(batch_size, input_height, input_width, conv_1_params[0])
    torch_input_for_premodel = torch.permute(torch_input, (0, 3, 1, 2))
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_squeezenet, custom_preprocessor=custom_preprocessor, device=None
    )
    tt_out = tt_squeezenet(device, parameters, tt_input)
    torch_out = torch_squeezenet(torch_input_for_premodel)
    tt_out_in_torch = ttnn.to_torch(tt_out)
    assert_with_pcc(torch_out, tt_out_in_torch, pcc=0.99)
