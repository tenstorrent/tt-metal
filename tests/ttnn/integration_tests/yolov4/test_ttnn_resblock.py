# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_yolov4.reference.resblock import ResBlock
from models.experimental.functional_yolov4.tt.ttnn_resblock import TtResBlock
import ttnn


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, ResBlock):
            for i, block in enumerate(model.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]
                ttnn_module_args[f"resblock_{i}_conv1"] = ttnn_module_args["0"]
                ttnn_module_args[f"resblock_{i}_conv1"]["math_fidelity"] = ttnn.MathFidelity.LoFi
                ttnn_module_args[f"resblock_{i}_conv1"]["dtype"] = ttnn.bfloat8_b
                ttnn_module_args[f"resblock_{i}_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args[f"resblock_{i}_conv1"]["activation"] = "relu"  # Fuse relu with conv1
                ttnn_module_args[f"resblock_{i}_conv1"]["deallocate_activation"] = True
                ttnn_module_args[f"resblock_{i}_conv1"]["conv_blocking_and_parallelization_config_override"] = None

                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args[f"resblock_{i}_conv1"])
                parameters[f"resblock_{i}_conv1"] = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args[f"resblock_{i}_conv1"]
                )

                ttnn_module_args[f"resblock_{i}_conv2"] = ttnn_module_args["3"]
                ttnn_module_args[f"resblock_{i}_conv2"]["math_fidelity"] = ttnn.MathFidelity.LoFi
                ttnn_module_args[f"resblock_{i}_conv2"]["dtype"] = ttnn.bfloat8_b
                ttnn_module_args[f"resblock_{i}_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                ttnn_module_args[f"resblock_{i}_conv2"]["activation"] = "relu"  # Fuse relu with conv1
                ttnn_module_args[f"resblock_{i}_conv2"]["deallocate_activation"] = True
                ttnn_module_args[f"resblock_{i}_conv2"]["conv_blocking_and_parallelization_config_override"] = None
                # Preprocess the convolutional layer
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args[f"resblock_{i}_conv2"])
                parameters[f"resblock_{i}_conv2"] = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args[f"resblock_{i}_conv2"]
                )

        return parameters

    return custom_preprocessor


@pytest.mark.skip("Issue #8749")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_wormhole_b0()
def test_resblock(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolo")
    weights_pth = str(model_path / "yolov4.pth")
    state_dict = torch.load(weights_pth)
    resblock_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("down2.resblock."))}

    if not resblock_state_dict:
        raise ValueError("No parameters found in resblock_state_dict")

    torch_model = ResBlock(ch=64, nblocks=2, shortcut=True)
    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in resblock_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    # Input tensor for testing
    torch_input_tensor = torch.randn(1, 64, 32, 32)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)

    # Preprocess the model for TTNN
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    # Convert the model to TTNN
    ttnn_model = TtResBlock(parameters, nblocks=2, shortcut=True)

    # Convert input tensor to TTNN format
    input_shape = torch_input_tensor.shape
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )

    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Apply TTNN model
    output_tensor = ttnn_model(device, input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(1, 32, 32, 64)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)

    # Assertion
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
