# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from models.experimental.openpdn_mnist.reference.openpdn_mnist import (
    OpenPDNMnist,
)
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


def create_openpdn_mnist_model_input_tensors(batch=2, input_channels=5, input_height=78, input_width=78):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        ttnn_input_tensor.shape[0],
        1,
        ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)
    return torch_input_tensor, ttnn_input_tensor


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = preprocess_conv_parameter(model.weight, dtype=ttnn.float32)
        bias = model.bias.reshape((1, 1, 1, -1))
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.float32)
    if isinstance(model, nn.Linear):
        parameters["weight"] = preprocess_linear_weight(model.weight, dtype=ttnn.float32)
        parameters["bias"] = preprocess_linear_bias(model.bias, dtype=ttnn.float32)

    return parameters


def create_openpdn_mnist_model_model_parameters(model: OpenPDNMnist, input_tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)

    return parameters
