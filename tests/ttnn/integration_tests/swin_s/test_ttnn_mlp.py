# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import pytest
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.swin_s.reference.mlp import MLP
from models.experimental.swin_s.tt.tt_mlp import TtMLP
from models.experimental.swin_s.tt.common import get_mesh_mappers
from torchvision import models
import ttnn


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return bias


def preprocess_layernorm_parameter(parameter, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return parameter


def custom_preprocessor(torch_model, name, mesh_mapper=None):
    parameters = {}
    if isinstance(torch_model, MLP):
        parameters[0] = {}
        parameters[3] = {}
        parameters[0]["weight"] = preprocess_linear_weight(
            torch_model[0].weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters[0]["bias"] = preprocess_linear_bias(
            torch_model[0].bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters[3]["weight"] = preprocess_linear_weight(
            torch_model[3].weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters[3]["bias"] = preprocess_linear_bias(
            torch_model[3].bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        if torch.layer_norm in torch_model:
            parameters[1] = {}
            parameters[1]["weight"] = preprocess_layernorm_parameter(
                torch_model[1].weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
            parameters[1]["bias"] = preprocess_layernorm_parameter(
                torch_model[1].bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            )
    return parameters


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "in_channels,hidden_channels,seq_len,i,j",
    [
        (96, [384, 96], 128, 1, 0),
        (192, [768, 192], 64, 3, 0),
        (384, [1536, 384], 32, 5, 0),
        (768, [3072, 768], 16, 7, 0),
    ],
)
def test_mlp(device, in_channels, hidden_channels, batch_size, seq_len, i, j, reset_seeds):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = state_dict = model.state_dict()
    mlp_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}.{j}.mlp."))}

    if not mlp_state_dict:
        raise ValueError("No parameters found in mlp_state_dict")

    torch_model = MLP(in_channels, hidden_channels, activation_layer=nn.GELU)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in mlp_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    # Input tensor for testing
    torch_input_tensor = torch.randn(batch_size, seq_len, seq_len, in_channels)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)

    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    # Convert the model to TTNN
    ttnn_model = TtMLP(hidden_channels, device, parameters, activation_layer=ttnn.gelu)

    # Convert input tensor to TTNN format
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Apply TTNN model
    output_tensor = ttnn_model(input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor
