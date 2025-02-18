# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d
from models.experimental.functional_vanilla_unet.reference.unet import UNet


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, UNet):
            for i in range(1, 5):
                parameters[f"encoder{i}"] = {}
                parameters[f"encoder{i}"][0] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                    getattr(model, f"encoder{i}")[0], getattr(model, f"encoder{i}")[1]
                )
                parameters[f"encoder{i}"][0]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[f"encoder{i}"][0]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

                parameters[f"encoder{i}"][1] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                    getattr(model, f"encoder{i}")[3], getattr(model, f"encoder{i}")[4]
                )
                parameters[f"encoder{i}"][1]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[f"encoder{i}"][1]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )

            parameters["bottleneck"] = {}
            parameters["bottleneck"][0] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.bottleneck[0], model.bottleneck[1])
            parameters["bottleneck"][0]["weight"] = ttnn.from_torch(
                conv_weight,
                dtype=ttnn.bfloat16,
            )
            parameters["bottleneck"][0]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

            parameters["bottleneck"][1] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.bottleneck[3], model.bottleneck[4])
            parameters["bottleneck"][1]["weight"] = ttnn.from_torch(
                conv_weight,
                dtype=ttnn.bfloat16,
            )
            parameters["bottleneck"][1]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

            for i in range(4, 1, -1):
                parameters[f"upconv{i}"] = {}
                parameters[f"upconv{i}"]["weight"] = ttnn.from_torch(
                    getattr(model, f"upconv{i}").weight, dtype=ttnn.bfloat16
                )
                parameters[f"upconv{i}"]["bias"] = ttnn.from_torch(
                    torch.reshape(getattr(model, f"upconv{i}").bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            for i in range(4, 0, -1):
                parameters[f"decoder{i}"] = {}
                parameters[f"decoder{i}"][0] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                    getattr(model, f"decoder{i}")[0], getattr(model, f"decoder{i}")[1]
                )
                parameters[f"decoder{i}"][0]["weight"] = ttnn.from_torch(
                    conv_weight,
                    dtype=ttnn.bfloat16,
                )
                parameters[f"decoder{i}"][0]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )

                parameters[f"decoder{i}"][1] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                    getattr(model, f"decoder{i}")[3], getattr(model, f"decoder{i}")[4]
                )
                parameters[f"decoder{i}"][1]["weight"] = ttnn.from_torch(
                    conv_weight,
                    dtype=ttnn.bfloat16,
                )
                parameters[f"decoder{i}"][1]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )

            parameters["conv"] = {}
            parameters["conv"]["weight"] = ttnn.from_torch(
                model.conv.weight,
                dtype=ttnn.bfloat16,
            )
            parameters["conv"]["bias"] = ttnn.from_torch(
                torch.reshape(model.conv.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

        return parameters

    return custom_preprocessor
