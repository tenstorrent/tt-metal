# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.functional_mobilenetv2_new_conv_api.reference.mobilenetv2 import Mobilenetv2
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

    parameters.c1["conv_blocking_and_parallelization_config_override"] = None
    parameters.c2["conv_blocking_and_parallelization_config_override"] = None
    parameters.c3["conv_blocking_and_parallelization_config_override"] = None
    parameters.c4["conv_blocking_and_parallelization_config_override"] = None
    parameters.c5["conv_blocking_and_parallelization_config_override"] = None
    parameters.c6["conv_blocking_and_parallelization_config_override"] = None
    parameters.c7["conv_blocking_and_parallelization_config_override"] = None
    parameters.c8["conv_blocking_and_parallelization_config_override"] = None
    parameters.c9["conv_blocking_and_parallelization_config_override"] = None
    parameters.c10["conv_blocking_and_parallelization_config_override"] = None
    parameters.c11["conv_blocking_and_parallelization_config_override"] = None
    parameters.c12["conv_blocking_and_parallelization_config_override"] = None
    parameters.c13["conv_blocking_and_parallelization_config_override"] = None
    parameters.c14["conv_blocking_and_parallelization_config_override"] = None
    parameters.c15["conv_blocking_and_parallelization_config_override"] = None
    parameters.c16["conv_blocking_and_parallelization_config_override"] = None
    parameters.c17["conv_blocking_and_parallelization_config_override"] = None
    parameters.c18["conv_blocking_and_parallelization_config_override"] = None
    parameters.c19["conv_blocking_and_parallelization_config_override"] = None
    parameters.c20["conv_blocking_and_parallelization_config_override"] = None
    parameters.c21["conv_blocking_and_parallelization_config_override"] = None
    parameters.c22["conv_blocking_and_parallelization_config_override"] = None
    parameters.c23["conv_blocking_and_parallelization_config_override"] = None
    parameters.c24["conv_blocking_and_parallelization_config_override"] = None
    parameters.c25["conv_blocking_and_parallelization_config_override"] = None
    parameters.c26["conv_blocking_and_parallelization_config_override"] = None
    parameters.c27["conv_blocking_and_parallelization_config_override"] = None
    parameters.c28["conv_blocking_and_parallelization_config_override"] = None
    parameters.c29["conv_blocking_and_parallelization_config_override"] = None
    parameters.c30["conv_blocking_and_parallelization_config_override"] = None
    parameters.c31["conv_blocking_and_parallelization_config_override"] = None
    parameters.c32["conv_blocking_and_parallelization_config_override"] = None
    parameters.c33["conv_blocking_and_parallelization_config_override"] = None
    parameters.c34["conv_blocking_and_parallelization_config_override"] = None
    parameters.c35["conv_blocking_and_parallelization_config_override"] = None
    parameters.c36["conv_blocking_and_parallelization_config_override"] = None
    parameters.c37["conv_blocking_and_parallelization_config_override"] = None
    parameters.c38["conv_blocking_and_parallelization_config_override"] = None
    parameters.c39["conv_blocking_and_parallelization_config_override"] = None
    parameters.c40["conv_blocking_and_parallelization_config_override"] = None
    parameters.c41["conv_blocking_and_parallelization_config_override"] = None
    parameters.c42["conv_blocking_and_parallelization_config_override"] = None
    parameters.c43["conv_blocking_and_parallelization_config_override"] = None
    parameters.c44["conv_blocking_and_parallelization_config_override"] = None
    parameters.c45["conv_blocking_and_parallelization_config_override"] = None
    parameters.c46["conv_blocking_and_parallelization_config_override"] = None
    parameters.c47["conv_blocking_and_parallelization_config_override"] = None
    parameters.c48["conv_blocking_and_parallelization_config_override"] = None
    parameters.c49["conv_blocking_and_parallelization_config_override"] = None
    parameters.c50["conv_blocking_and_parallelization_config_override"] = None
    parameters.c51["conv_blocking_and_parallelization_config_override"] = None
    parameters.c52["conv_blocking_and_parallelization_config_override"] = None

    parameters["l1"] = {}
    parameters["l1"]["weight"] = model.l1.weight
    parameters["l1"]["bias"] = model.l1.bias

    return parameters
