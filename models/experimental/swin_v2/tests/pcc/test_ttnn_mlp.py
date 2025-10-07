# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import pytest
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
)
from models.common.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.swin_v2.reference.mlp import MLP
from models.experimental.swin_v2.tt.tt_mlp import TtMLP
import ttnn
from models.experimental.swin_v2.common import load_torch_model, SWIN_V2_L1_SMALL_SIZE


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, MLP):
            parameters[0] = {}
            parameters[3] = {}
            parameters[0]["weight"] = preprocess_linear_weight(torch_model[0].weight, dtype=ttnn.bfloat16)
            parameters[0]["bias"] = preprocess_linear_bias(torch_model[0].bias, dtype=ttnn.bfloat16)
            parameters[3]["weight"] = preprocess_linear_weight(torch_model[3].weight, dtype=ttnn.bfloat16)
            parameters[3]["bias"] = preprocess_linear_bias(torch_model[3].bias, dtype=ttnn.bfloat16)
            if torch.layer_norm in torch_model:
                parameters[1] = {}
                parameters[1]["weight"] = preprocess_layernorm_parameter(torch_model[1].weight, dtype=ttnn.bfloat16)
                parameters[1]["bias"] = preprocess_layernorm_parameter(torch_model[1].bias, dtype=ttnn.bfloat16)
        return parameters

    return custom_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": SWIN_V2_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "in_channels,hidden_channels,seq_len,i,j",
    [
        (96, [384, 96], 128, 1, 0),
        (192, [768, 192], 64, 3, 0),
        (384, [1536, 384], 32, 5, 0),
        (768, [3072, 768], 16, 7, 0),
    ],
)
def test_mlp(device, in_channels, hidden_channels, seq_len, i, j, reset_seeds, model_location_generator):
    torch_model = MLP(in_channels, hidden_channels, activation_layer=nn.GELU)

    torch_model = load_torch_model(
        torch_model, i=i, j=j, module="mlp", model_location_generator=model_location_generator
    )

    # Input tensor for testing
    torch_input_tensor = torch.randn(1, seq_len, seq_len, in_channels)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
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
