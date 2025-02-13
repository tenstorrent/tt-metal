# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d
from models.utility_functions import skip_for_grayskull
from models.experimental.functional_vanilla_unet.reference.unet import UNet
from models.experimental.functional_vanilla_unet.ttnn.ttnn_unet import TtUnet
from tests.ttnn.utils_for_testing import assert_with_pcc


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

            # for i in range(4,0,-1):
            #     parameters[f"upconv{i}"]={}
            #     parameters[f"upconv{i}"]["weight"] = ttnn.from_torch(getattr(model,f"upconv{i}").weight, dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT)
            #     parameters[f"upconv{i}"]["bias"] = ttnn.from_torch(
            #         torch.reshape(getattr(model,f"upconv{i}").bias, (1, 1, 1, -1)), dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT
            #         )

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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
def test_unet(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("vanilla_unet")

    state_dict = torch.load(
        model_path / "unet.pt",
        map_location=torch.device("cpu"),
    )
    ds_state_dict = {k: v for k, v in state_dict.items()}

    reference_model = UNet()

    for layer in reference_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_input_tensor = torch.randn(1, 3, 480, 640)
    torch_output_tensor = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    ttnn_model = TtUnet(device=device, parameters=parameters, model=reference_model)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor.permute(0, 2, 3, 1), device=device)

    ttnn_output = ttnn_model(device, ttnn_input_tensor)

    # ttnn_output = ttnn.from_device(ttnn_output)
    # ttnn_output = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)
    # ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output_tensor, ttnn_output, pcc=0.93)
