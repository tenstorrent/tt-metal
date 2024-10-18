# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0]

import torch
import ttnn
from ttnn.model_preprocessing import infer_ttnn_module_args
from models.experimental.functional_yolov11.reference.yolov11 import YoloV11
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters


def create_yolov11_input_tensors(device, batch=1, input_channels=3, input_height=224, input_width=224):
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


def preprocess_params(d, parameters, path=None, depth=0, max_depth=6):
    if path is None:
        path = []  # Initialize the path for the first call

    if isinstance(d, dict):
        # If the dictionary has the 'conv' key, handle it
        if "conv" in d:
            weight_full_path = ".".join(path + ["conv", "weight"])
            bias_full_path = ".".join(path + ["conv", "bias"])
            weight, bias = preprocess(parameters, weight_full_path, bias_full_path)
            d.conv.bias = None
            d.conv.weight = weight
            if bias is not None:
                d.conv.bias = bias

        # Recurse deeper only if we haven't reached the max depth
        if depth < max_depth:
            for key, value in d.items():
                if isinstance(value, dict):  # If the value is a dictionary, continue recursion
                    if depth == 0:
                        d[key] = preprocess_params(value, parameters, path + [key, "module"], depth + 1, max_depth)
                    else:
                        d[key] = preprocess_params(value, parameters, path + [key], depth + 1, max_depth)

    return d


class DotDict(dict):
    """
    A dictionary subclass that allows attribute-style access to its keys.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


def preprocess(d: dict, weights_path: str, bias_path: str):
    """
    Accesses a tensor within a nested dictionary using a path string.

    Args:
        d: The dictionary containing the nested structure.
        path: A string representing the path to the tensor,
              e.g., "conv1.module.conv.weight".

    Returns:
        The tensor found at the specified path, or None if not found.
    """
    tt_bias = None
    weight_keys = weights_path.split(".")
    bias_keys = bias_path.split(".")
    w_current = DotDict(d)  # Convert the top-level dictionary to DotDict
    b_current = DotDict(d)

    for key in weight_keys:
        w_current = getattr(w_current, key)  # Use getattr for dot notation access
    for key in bias_keys:
        b_current = getattr(b_current, key)  # Use getattr for dot notation access

    tt_weight = ttnn.from_torch(w_current, dtype=ttnn.float32)
    if b_current is not None:
        b_current = torch.reshape(b_current, (1, 1, 1, -1))
        tt_bias = ttnn.from_torch(b_current, dtype=ttnn.float32)

    return tt_weight, tt_bias


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
    return parameters


def create_yolov11_model_parameters(model: YoloV11, input_tensor: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    return parameters
