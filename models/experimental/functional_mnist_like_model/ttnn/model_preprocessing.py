# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.functional_mnist_like_model.reference.mnist_like_model import (
    Mnist_like_model,
)
from ttnn.model_preprocessing import infer_ttnn_module_args


def create_mnist_like_model_input_tensors(batch=2, input_channels=5, input_height=78, input_width=78):
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


def create_mnist_like_model_model_parameters(model: Mnist_like_model, input_tensor, device):
    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)
    return parameters
