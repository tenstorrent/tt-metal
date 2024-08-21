# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
import pytest
from loguru import logger

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc

torch.manual_seed(0)


def torch_functional_falcon_linear(hidden_states, parameters):
    hidden_states = hidden_states @ parameters.weight
    if parameters.get("bias", None):
        hidden_states = hidden_states + parameters.bias
    return hidden_states


def torch_functional_falcon_mlp(hidden_states, *, parameters):
    hidden_states = torch_functional_falcon_linear(hidden_states, parameters.dense_h_to_4h)
    hidden_states = torch.nn.functional.gelu(hidden_states)
    hidden_states = torch_functional_falcon_linear(hidden_states, parameters.dense_4h_to_h)

    return hidden_states


def ttnn_functional_falcon_linear(hidden_states, parameters):
    hidden_states = hidden_states @ parameters.weight
    if parameters.get("bias", None):
        hidden_states = hidden_states + parameters.bias
    return hidden_states


def ttnn_functional_falcon_mlp(hidden_states, *, parameters):
    hidden_states = ttnn_functional_falcon_linear(hidden_states, parameters.dense_h_to_4h)
    hidden_states = ttnn.gelu(hidden_states)
    hidden_states = ttnn_functional_falcon_linear(hidden_states, parameters.dense_4h_to_h)

    return hidden_states


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b-instruct"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [128])
def test_torch_functional_falcon_mlp(model_name, batch_size, sequence_length):
    config = transformers.FalconConfig.from_pretrained(model_name)
    model = transformers.models.falcon.modeling_falcon.FalconMLP(config).eval()
    hidden_states = (torch.rand(batch_size, 1, sequence_length, config.hidden_size, dtype=torch.float32) * 2) - 1

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )
    torch_output = model.forward(hidden_states)
    output = torch_functional_falcon_mlp(hidden_states, parameters=parameters)

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.parametrize("model_name", ["tiiuae/falcon-7b-instruct"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_length", [128])
def test_ttnn_functional_falcon_mlp(device, model_name, batch_size, sequence_length):
    config = transformers.FalconConfig.from_pretrained(model_name)
    model = transformers.models.falcon.modeling_falcon.FalconMLP(config).eval()

    torch_hidden_states = (torch.rand(batch_size, 1, sequence_length, config.hidden_size) * 2) - 1
    torch_output = model(torch_hidden_states)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn_functional_falcon_mlp(
        hidden_states,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.985)
