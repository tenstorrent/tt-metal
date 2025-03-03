# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.experimental.functional_yolov8s_world.reference.yolov8s_world_utils import *


def fold_batch_norm2d_into_conv2d(conv, bn):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")

    weight = conv.weight
    bias = conv.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    return weight, bias


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, WorldModel):
            parameters["model"] = {}
            for index, child in enumerate(model.model):
                parameters["model"][index] = {}
                if isinstance(child, Conv):
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.conv, child.bn)
                    parameters["model"][index]["conv"] = {}
                    parameters["model"][index]["conv"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters["model"][index]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
                    )
                elif isinstance(child, C2f):
                    parameters["model"][index]["cv1"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.cv1.conv, child.cv1.bn)
                    parameters["model"][index]["cv1"]["conv"] = {}
                    parameters["model"][index]["cv1"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight, dtype=ttnn.bfloat16
                    )
                    parameters["model"][index]["cv1"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
                    )

                    parameters["model"][index]["cv2"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.cv2.conv, child.cv2.bn)
                    parameters["model"][index]["cv2"]["conv"] = {}
                    parameters["model"][index]["cv2"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight, dtype=ttnn.bfloat16
                    )
                    parameters["model"][index]["cv2"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
                    )

                    parameters["model"][index]["m"] = {}
                    for index_2, child_2 in enumerate(child.m):
                        parameters["model"][index]["m"][index_2] = {}

                        parameters["model"][index]["m"][index_2]["cv1"] = {}
                        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_2.cv1.conv, child_2.cv1.bn)
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"] = {}
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"]["weight"] = ttnn.from_torch(
                            conv_weight, dtype=ttnn.bfloat16
                        )
                        parameters["model"][index]["m"][index_2]["cv1"]["conv"]["bias"] = ttnn.from_torch(
                            conv_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
                        )

                        parameters["model"][index]["m"][index_2]["cv2"] = {}
                        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child_2.cv2.conv, child_2.cv2.bn)
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"] = {}
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"]["weight"] = ttnn.from_torch(
                            conv_weight, dtype=ttnn.bfloat16
                        )
                        parameters["model"][index]["m"][index_2]["cv2"]["conv"]["bias"] = ttnn.from_torch(
                            conv_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
                        )
                elif isinstance(child, SPPF):
                    parameters["model"][index]["cv1"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.cv1.conv, child.cv1.bn)
                    parameters["model"][index]["cv1"]["conv"] = {}
                    parameters["model"][index]["cv1"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight, dtype=ttnn.bfloat16
                    )
                    parameters["model"][index]["cv1"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
                    )

                    parameters["model"][index]["cv2"] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.cv2.conv, child.cv2.bn)
                    parameters["model"][index]["cv2"]["conv"] = {}
                    parameters["model"][index]["cv2"]["conv"]["weight"] = ttnn.from_torch(
                        conv_weight, dtype=ttnn.bfloat16
                    )
                    parameters["model"][index]["cv2"]["conv"]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16
                    )

        return parameters

    return custom_preprocessor
