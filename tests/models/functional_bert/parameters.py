# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import tt_lib as ttl
from tt_lib.tensor import MemoryConfig, BufferStorage

TILE_HEIGHT = 32
TILE_WIDTH = 32


@dataclass
class ParametersConfig:
    linear_weight_dtype: ttl.tensor.DataType = ttl.tensor.DataType.BFLOAT16
    linear_bias_dtype: ttl.tensor.DataType = ttl.tensor.DataType.BFLOAT16
    layernorm_parameter_dtype: ttl.tensor.DataType = ttl.tensor.DataType.BFLOAT16


def pad_tensor(tensor, height_multiple=TILE_HEIGHT, width_multiple=TILE_WIDTH):
    *_, height, width = tensor.shape
    padded_height = int(np.ceil(height / height_multiple)) * height_multiple
    padded_width = int(np.ceil(width / width_multiple)) * width_multiple
    return F.pad(tensor, (0, padded_width - width, 0, padded_height - height))


def preprocess_linear_weight(parameters_config, weight, **kwargs):
    weight = weight.T.unsqueeze(0).unsqueeze(0)
    weight = pad_tensor(weight)
    tensor = ttl.tensor.Tensor(weight, parameters_config.linear_weight_dtype).to(
        ttl.tensor.Layout.TILE
    )
    tensor = tensor.to(kwargs["device"])
    return tensor


def preprocess_linear_bias(parameters_config, bias, **kwargs):
    bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    bias = pad_tensor(bias)
    tensor = ttl.tensor.Tensor(bias, parameters_config.linear_bias_dtype).to(
        ttl.tensor.Layout.TILE
    )
    tensor = tensor.to(kwargs["device"])
    return tensor


def preprocess_layernorm_parameter(parameters_config, parameter, **kwargs):
    parameter = parameter.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    parameter = pad_tensor(parameter, height_multiple=1, width_multiple=TILE_WIDTH)
    parameter = parameter.reshape((1, 1, -1, TILE_WIDTH))
    tensor = ttl.tensor.Tensor(parameter, parameters_config.layernorm_parameter_dtype)
    tensor = tensor.to(kwargs["device"], MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, BufferStorage.L1))
    return tensor


def default_preprocessor(parameters_config, torch_model, full_name, **kwargs):
    parameters = {}
    if isinstance(torch_model, torch.nn.Linear):
        parameters[f"{full_name}weight"] = preprocess_linear_weight(
            parameters_config, torch_model.weight, **kwargs
        )
        parameters[f"{full_name}bias"] = preprocess_linear_bias(
            parameters_config, torch_model.bias, **kwargs
        )
    elif isinstance(torch_model, torch.nn.LayerNorm):
        parameters[f"{full_name}weight"] = preprocess_layernorm_parameter(
            parameters_config, torch_model.weight, **kwargs
        )
        parameters[f"{full_name}bias"] = preprocess_layernorm_parameter(
            parameters_config, torch_model.bias, **kwargs
        )
    return parameters


def preprocess_model_parameters(
    parameters_config,
    torch_model,
    *,
    prefix="",
    is_to_be_converted,
    custom_preprocessor=None,
    **kwargs,
):
    parameters = {}

    named_children = list(torch_model.named_children())

    if not named_children:
        for name, parameter in torch_model.named_parameters():
            full_name = f"{prefix}{name}"
            parameters[full_name] = parameter

    for name, child in named_children:
        full_name = f"{prefix}{name}."

        use_default_preprocessor = True
        if custom_preprocessor is not None:
            custom_preprocessor_parameters = custom_preprocessor(
                parameters_config=parameters_config,
                torch_model=child,
                full_name=full_name,
                **kwargs,
            )
            if custom_preprocessor_parameters:
                parameters.update(custom_preprocessor_parameters)
                # Custom preprocessor didn't handle this case, so, try using default preprocessor
                use_default_preprocessor = False

        if use_default_preprocessor:
            if not is_to_be_converted(child, full_name):
                child_parameters = preprocess_model_parameters(
                    parameters_config,
                    child,
                    prefix=full_name,
                    is_to_be_converted=is_to_be_converted,
                    custom_preprocessor=custom_preprocessor,
                    **kwargs,
                )
                parameters.update(child_parameters)
            else:
                default_preprocessor_parameters = default_preprocessor(
                    parameters_config, child, full_name, **kwargs
                )
                if default_preprocessor_parameters:
                    parameters.update(default_preprocessor_parameters)
                else:
                    child_parameters = preprocess_model_parameters(
                        parameters_config,
                        child,
                        prefix=full_name,
                        is_to_be_converted=is_to_be_converted,
                        custom_preprocessor=custom_preprocessor,
                        **kwargs,
                    )
                    parameters.update(child_parameters)

    return parameters
