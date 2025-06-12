# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv2.reference.resnet import ResNet
from models.experimental.vadv2.reference.fpn import FPN
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
)


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, FPN):
        parameters["fpn"] = {}
        parameters["fpn"]["lateral_convs"] = {}
        parameters["fpn"]["lateral_convs"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs.conv.weight, dtype=ttnn.float32
        )
        bias = model.lateral_convs.conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        parameters["fpn"]["fpn_convs"] = {}
        parameters["fpn"]["fpn_convs"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs.conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs.conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

        return parameters


def create_vadv2_model_parameters(model: ResNet, input_tensor, device=None):
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
