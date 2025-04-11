# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.experimental.yolov6l.reference.yolov6l import Model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    # if isinstance(model, Conv):
    #     weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
    #     parameters["conv"] = {}
    #     parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
    #     bias = bias.reshape((1, 1, 1, -1))
    #     parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    # if isinstance(model, nn.ConvTranspose2d):
    #     weight = model.weight
    #     bias = model.bias
    #     parameters["conv_t"] = {}
    #     parameters["conv_t"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
    #     bias = bias.reshape((1, 1, 1, -1))
    #     parameters["conv_t"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


def create_yolov6l_model_parameters(model: Model, torch_input: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(torch_input), device=device
    )
    parameters["model_args"] = model
    return parameters


def create_yolov6l_model_parameters_bottlerep(model: Model, torch_input: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(torch_input), device=device
    )
    parameters["model_args"] = model
    return parameters


def create_yolov6l_model_parameters_repblock(model: Model, torch_input: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(torch_input), device=device
    )
    parameters["model_args"] = model
    return parameters


def create_yolov6l_model_parameters_bep_c3(model: Model, torch_input: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(torch_input), device=device
    )
    parameters["model_args"] = model
    return parameters


def create_yolov6l_model_parameters_sppf(model: Model, torch_input: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(torch_input), device=device
    )
    parameters["model_args"] = model
    return parameters
