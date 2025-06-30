# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d
from models.utility_functions import skip_for_grayskull
from models.demos.vanilla_unet.reference.unet import UNet
from models.demos.vanilla_unet.ttnn.ttnn_unet import TtUnet
from tests.ttnn.utils_for_testing import assert_with_pcc
import os
import torch.nn.functional as F
from loguru import logger


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
            parameters[f"decoder1"][0]["weight"] = ttnn.from_torch(model.decoder1[0].weight, dtype=ttnn.bfloat16)
            parameters[f"decoder1"][0]["bias"] = None

            bn_layer = model.decoder1[1]  # BatchNorm2d layer
            channel_size = bn_layer.num_features

            # Extract PyTorch tensors
            weight_torch = bn_layer.weight if bn_layer.affine else None
            bias_torch = bn_layer.bias if bn_layer.affine else None
            batch_mean_torch = bn_layer.running_mean
            batch_var_torch = bn_layer.running_var

            # Reshape for broadcast compatibility (1, C, 1, 1)
            batch_mean_torch = batch_mean_torch.view(1, channel_size, 1, 1)
            batch_var_torch = batch_var_torch.view(1, channel_size, 1, 1)
            weight_torch = weight_torch.view(1, channel_size, 1, 1) if weight_torch is not None else None
            bias_torch = bias_torch.view(1, channel_size, 1, 1) if bias_torch is not None else None

            parameters["decoder1"]["bn"] = {}
            weight = (
                ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
                if weight_torch is not None
                else None
            )
            parameters["decoder1"]["bn"]["weight"] = ttnn.to_device(weight, device)

            bias = (
                ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
                if bias_torch is not None
                else None
            )
            parameters["decoder1"]["bn"]["bias"] = ttnn.to_device(bias, device)

            running_mean = ttnn.from_torch(
                batch_mean_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
            )
            parameters["decoder1"]["bn"]["running_mean"] = ttnn.to_device(running_mean, device)

            running_var = ttnn.from_torch(batch_var_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            parameters["decoder1"]["bn"]["running_var"] = ttnn.to_device(running_var, device)

            parameters["decoder1"]["bn"]["eps"] = bn_layer.eps  # scalar, used directly in ops

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


@pytest.mark.parametrize("device_params", [{"l1_small_size": (7 * 8192) + 1730}], indirect=True)
@skip_for_grayskull()
def test_unet(device, reset_seeds, model_location_generator):
    weights_path = "models/demos/vanilla_unet/unet.pt"
    if not os.path.exists(weights_path):
        os.system("bash models/demos/vanilla_unet/weights_download.sh")

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

    torch_input_tensor = torch.randn(1, 3, 480, 640)
    torch_output_tensor = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )

    ttnn_model = TtUnet(device=device, parameters=parameters, model=reference_model)

    n, c, h, w = torch_input_tensor.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, 640, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input_host = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    ttnn_output = ttnn_model(device, ttnn_input_host)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output_tensor.shape)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output, pcc=0.95)
    logger.info(pcc_message)
