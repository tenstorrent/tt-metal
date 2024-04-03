# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_yolov4.reference.downsample2 import DownSample2
from models.experimental.functional_yolov4.tt.ttnn_downsample2 import TtDownSample2

import ttnn
import tt_lib
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["activation"] = "relu"


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, DownSample2):
            ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
            update_ttnn_module_args(ttnn_module_args.c1)
            parameters["c1"], c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
            )

            ttnn_module_args.c2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
            update_ttnn_module_args(ttnn_module_args.c2)
            parameters["c2"], c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
            )

            ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
            update_ttnn_module_args(ttnn_module_args.c3)
            parameters["c3"], c3_parallel_config = preprocess_conv2d(
                conv3_weight, conv3_bias, ttnn_module_args.c3, return_parallel_config=True
            )

            parameters["res"] = {}
            for i, block in enumerate(model.res.module_list):
                conv1 = block[0]
                bn1 = block[1]
                conv2 = block[3]
                bn2 = block[4]

                ttnn_module_args["res"][f"resblock_{i}_conv1"] = ttnn_module_args["res"]["0"]
                ttnn_module_args["res"][f"resblock_{i}_conv1"]["weights_dtype"] = ttnn.bfloat8_b
                weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
                update_ttnn_module_args(ttnn_module_args["res"][f"resblock_{i}_conv1"])
                parameters["res"][f"resblock_{i}_conv1"], _ = preprocess_conv2d(
                    weight1, bias1, ttnn_module_args["res"][f"resblock_{i}_conv1"], return_parallel_config=True
                )

                ttnn_module_args["res"][f"resblock_{i}_conv2"] = ttnn_module_args["res"]["3"]
                ttnn_module_args["res"][f"resblock_{i}_conv2"]["weights_dtype"] = ttnn.bfloat8_b
                weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
                update_ttnn_module_args(ttnn_module_args["res"][f"resblock_{i}_conv2"])
                parameters["res"][f"resblock_{i}_conv2"], _ = preprocess_conv2d(
                    weight2, bias2, ttnn_module_args["res"][f"resblock_{i}_conv2"], return_parallel_config=True
                )

            ttnn_module_args.c4["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
            conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
            update_ttnn_module_args(ttnn_module_args.c4)
            parameters["c4"], c4_parallel_config = preprocess_conv2d(
                conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
            )

            ttnn_module_args.c5["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c5["weights_dtype"] = ttnn.bfloat8_b
            conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
            update_ttnn_module_args(ttnn_module_args.c5)
            parameters["c5"], c5_parallel_config = preprocess_conv2d(
                conv5_weight, conv5_bias, ttnn_module_args.c5, return_parallel_config=True
            )
            return parameters

    return custom_preprocessor


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_l1_small_size", [32768], indirect=True)
def test_downsample2(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolo")
    weights_pth = str(model_path / "yolov4.pth")
    state_dict = torch.load(weights_pth)
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("down2."))}

    torch_model = DownSample2()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 64, 160, 160)  # Batch size of 1, 64 input channels, 160x160 height and width
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = TtDownSample2(parameters)

    # Tensor Preprocessing
    #
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(1, 80, 80, 128)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
