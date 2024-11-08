# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import pytest
import torch
from models.demos.squeezenet.tt.tt_squeezenet import tt_Fire
from torchvision import models
from tests.ttnn.utils_for_testing import assert_with_pcc
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = ttnn.from_torch(model.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)

    return parameters


@pytest.mark.parametrize(
    "batch_size, input_height, input_width, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, features_block",
    [
        (1, 54, 54, 96, 16, 64, 64, 3),
        (1, 54, 54, 128, 16, 64, 64, 4),
        (1, 54, 54, 128, 32, 128, 128, 5),
        (1, 27, 27, 256, 32, 128, 128, 7),
        (1, 27, 27, 256, 48, 192, 192, 8),
        (1, 27, 27, 384, 48, 192, 192, 9),
        (1, 27, 27, 384, 64, 256, 256, 10),
        (1, 13, 13, 512, 64, 256, 256, 12),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_fire(
    device,
    batch_size,
    input_height,
    input_width,
    inplanes,
    squeeze_planes,
    expand1x1_planes,
    expand3x3_planes,
    features_block,
):
    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    state_dict = torch_squeezenet.state_dict()
    torch_input = torch.randn([batch_size, input_height, input_width, inplanes])
    torch_input_for_premodel = torch.permute(torch_input, (0, 3, 1, 2))
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_squeezenet.features[features_block],
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    tt_out = tt_Fire(
        inplanes,
        squeeze_planes,
        expand1x1_planes,
        expand3x3_planes,
        input_tensor=tt_input,
        parameters=parameters,
        device=device,
    )
    tt_out_in_torch = ttnn.to_torch(tt_out)
    tt_out_in_torch = torch.permute(tt_out_in_torch, (0, 3, 1, 2))
    torch_model = torch_squeezenet.features[features_block]
    torch_out = torch_model(torch_input_for_premodel)
    assert_with_pcc(torch_out, tt_out_in_torch, pcc=0.99)
