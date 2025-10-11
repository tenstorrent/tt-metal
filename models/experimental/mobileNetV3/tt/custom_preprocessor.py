# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import (
    fold_batch_norm2d_into_conv2d,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidual, SElayer, Conv2dNormActivation


# ------------------------
# Small helpers
# ------------------------
def conv_bn_to_params(conv, bn, mesh_mapper):
    """Fold BN into Conv and return as TTNN params."""
    if bn is None:
        # No BN: keep conv weight/bias
        weight = conv.weight
        bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
    else:
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)

    return {
        "weight": ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        "bias": ttnn.from_torch(torch.reshape(bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    }


def linear_to_params(weight, bias, mesh_mapper):
    """Preprocess Linear weight + bias to TTNN format."""
    return {
        "weight": preprocess_linear_weight(weight, dtype=ttnn.bfloat16),
        "bias": preprocess_linear_bias(bias, dtype=ttnn.bfloat16),
    }


def se_to_params(se, mesh_mapper):
    """Convert SElayer fc1/fc2 to TTNN params."""
    return {
        "fc1": {
            "weight": ttnn.from_torch(se.fc1.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(
                torch.reshape(se.fc1.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            ),
        },
        "fc2": {
            "weight": ttnn.from_torch(se.fc2.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
            "bias": ttnn.from_torch(
                torch.reshape(se.fc2.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
            ),
        },
    }


# ------------------------
# Unified Preprocessor
# ------------------------
def create_custom_preprocessor(mesh_mapper=None, debug=False):
    def custom_preprocessor(model, name="", ttnn_module_args=None):
        if debug:
            print(f"[Preprocess] {name}: {type(model).__name__}")

        parameters = {}

        # Case 1: Full MobileNetV3
        if isinstance(model, MobileNetV3):
            features = {}
            for idx, child in enumerate(model.features.children()):
                if isinstance(child, Conv2dNormActivation):
                    conv = child[0]
                    bn = child[1] if len(child) > 1 and hasattr(child[1], "weight") else None
                    features[idx] = {0: conv_bn_to_params(conv, bn, mesh_mapper)}
                elif isinstance(child, InvertedResidual):
                    features[idx] = {"block": custom_preprocessor(child, f"IR_{idx}", ttnn_module_args)}
            parameters["features"] = features

            # Classifier (Linear layers only)
            classifier = {}
            for idx, layer in enumerate(model.classifier):
                if isinstance(layer, torch.nn.Linear):
                    classifier[idx] = linear_to_params(layer.weight, layer.bias, mesh_mapper)
            parameters["classifier"] = classifier

        # Case 2: InvertedResidual
        elif isinstance(model, InvertedResidual):
            for idx, child in enumerate(model.block.children()):
                if isinstance(child, SElayer):
                    parameters[idx] = se_to_params(child, mesh_mapper)
                elif isinstance(child, Conv2dNormActivation):
                    conv = child[0]
                    bn = child[1] if len(child) > 1 and hasattr(child[1], "weight") else None
                    parameters[idx] = {0: conv_bn_to_params(conv, bn, mesh_mapper)}

        # Case 3: SElayer
        elif isinstance(model, SElayer):
            parameters = se_to_params(model, mesh_mapper)

        return parameters

    return custom_preprocessor
