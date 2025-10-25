# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

import ttnn
from models.demos.vanilla_unet_new.reference.model import UNet

VANILLA_UNET_L1_SMALL_SIZE = 12 * 8192
VANILLA_UNET_PCC_WH = 0.97700


def load_reference_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/vanilla_unet/unet.pt"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/vanilla_unet/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/unet_vanilla", model_subdir="", download_if_ci_v2=True) / "unet.pt"
        )

    state_dict = torch.load(
        weights_path,
        map_location=torch.device("cpu"),
    )
    ds_state_dict = {k: v for k, v in state_dict.items()}

    reference_model = UNet()

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()
    return reference_model


def create_unet_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, UNet):
            for i in range(1, 5):
                parameters[f"encoder{i}"] = {}
                parameters[f"encoder{i}"][0] = {}
                from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

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

            for i in range(4, 0, -1):
                parameters[f"upconv{i}"] = {}
                parameters[f"upconv{i}"]["weight"] = ttnn.from_torch(
                    getattr(model, f"upconv{i}").weight, dtype=ttnn.bfloat16
                )
                parameters[f"upconv{i}"]["bias"] = ttnn.from_torch(
                    torch.reshape(getattr(model, f"upconv{i}").bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            for i in range(4, 1, -1):
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

            parameters[f"decoder1"] = {}
            parameters[f"decoder1"][0] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                getattr(model, f"decoder1")[0], getattr(model, f"decoder1")[1]
            )
            parameters[f"decoder1"][0]["weight"] = ttnn.from_torch(
                conv_weight,
                dtype=ttnn.bfloat16,
            )
            parameters[f"decoder1"][0]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

            parameters[f"decoder1"][1] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                getattr(model, f"decoder1")[3], getattr(model, f"decoder1")[4]
            )
            parameters[f"decoder1"][1]["weight"] = ttnn.from_torch(
                conv_weight,
                dtype=ttnn.bfloat16,
            )
            parameters[f"decoder1"][1]["bias"] = ttnn.from_torch(
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
