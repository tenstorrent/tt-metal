# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn, Tensor
import pytest
from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_swin_s.reference.mlp import MLP
from models.experimental.functional_swin_s.tt.tt_mlp import TtMLP
from torchvision import models
import ttnn


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, torch.nn.Linear):
            parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat16)
            parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat16)
        if isinstance(torch_model, torch.nn.LayerNorm):
            parameters["norm_weight"] = preprocess_layernorm_parameter(torch_model.weight, dtype=ttnn.bfloat16)
            parameters["norm_bias"] = preprocess_layernorm_parameter(torch_model.bias, dtype=ttnn.bfloat16)
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_mlp(device, reset_seeds):
    model = models.swin_s(weights="IMAGENET1K_V1")
    state_dict = state_dict = model.state_dict()
    mlp_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("features.1.0.mlp."))}

    if not mlp_state_dict:
        raise ValueError("No parameters found in mlp_state_dict")

    torch_model = MLP(96, [384, 96], activation_layer=nn.GELU)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in mlp_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    # Input tensor for testing
    torch_input_tensor = torch.randn(8, 128, 128, 96)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    # Convert the model to TTNN
    ttnn_model = TtMLP([384, 96], device, parameters, activation_layer=ttnn.gelu)

    # Convert input tensor to TTNN format
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Apply TTNN model
    output_tensor = ttnn_model(input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
