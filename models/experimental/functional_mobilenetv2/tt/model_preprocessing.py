# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.functional_mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from ttnn.model_preprocessing import infer_ttnn_module_args


def create_mobilenetv2_input_tensors(batch=1, input_channels=3, input_height=128, input_width=128):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)

    return torch_input_tensor, ttnn_input_tensor


def create_mobilenetv2_model_parameters(model: Mobilenetv2, input_tensor, device):
    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)

    parameters["l1"] = {}
    parameters["l1"]["weight"] = model.l1.weight
    parameters["l1"]["bias"] = model.l1.bias

    return parameters
