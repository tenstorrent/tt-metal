# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vanilla_unet.common import VANILLA_UNET_L1_SMALL_SIZE, load_torch_model
from models.demos.vanilla_unet.reference.unet import UNet
from models.demos.vanilla_unet.tt.ttnn_unet import create_unet_from_configs
from models.demos.vanilla_unet.tt.unet_config import create_unet_configs_from_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_preprocessor(device):
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE}], indirect=True)
def test_tt_cnn_unet_small_inference(device, reset_seeds, model_location_generator):
    reference_model = load_torch_model(model_location_generator)

    torch_input_tensor = torch.randn(1, 3, 480, 640)
    torch_output_tensor = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )

    configs = create_unet_configs_from_parameters(
        parameters=parameters, input_height=480, input_width=640, batch_size=1
    )
    model = create_unet_from_configs(configs, device)

    n, c, h, w = torch_input_tensor.shape
    ttnn_input_host = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1),  # NCHW -> NHWC
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,  # device,
        memory_config=None,  # ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = model(ttnn_input_host)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)  # NHWC -> NCHW
    ttnn_output = ttnn_output.reshape(torch_output_tensor.shape)

    assert ttnn_output.shape == torch_output_tensor.shape
    logger.info(f"Output shape verification passed: {ttnn_output.shape}")

    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output, pcc=0.80)
    logger.info(pcc_message)
