# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_unet.reference.unet import UNet
from models.experimental.functional_unet.tt.tt_unet import TtUnet


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def update_ttnn_module_args(ttnn_module_args):
    # ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    # ttnn_module_args["use_shallow_conv_variant"] = False
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    # ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["activation"] = "relu"


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, UNet):
            ttnn_module_args["encoder1_c1"] = ttnn_module_args.encoder1["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder1[0], model.encoder1[1])
            update_ttnn_module_args(ttnn_module_args["encoder1_c1"])
            ttnn_module_args["encoder1_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder1_c1"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 16 * 32
            }
            ttnn_module_args["encoder1_c1"]["use_shallow_conv_variant"] = True
            # print("encoder1_c1", ttnn_module_args["encoder1_c1"])
            parameters["encoder1_c1"], encoder1_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder1_c1"], return_parallel_config=True
            )
            ttnn_module_args["encoder1_c2"] = ttnn_module_args.encoder1["3"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.encoder1[3], model.encoder1[4])
            update_ttnn_module_args(ttnn_module_args["encoder1_c2"])
            ttnn_module_args["encoder1_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder1_c2"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 16 * 32
            }
            ttnn_module_args["encoder1_c2"]["use_shallow_conv_variant"] = True
            # print("encoder1_c2", ttnn_module_args["encoder1_c2"])
            parameters["encoder1_c2"], encoder1_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["encoder1_c2"], return_parallel_config=True
            )
            ttnn_module_args["encoder2_c1"] = ttnn_module_args.encoder2["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder2[0], model.encoder2[1])
            update_ttnn_module_args(ttnn_module_args["encoder2_c1"])
            ttnn_module_args["encoder2_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder2_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder2_c1"]["use_shallow_conv_variant"] = True
            # print("encoder2_c1", ttnn_module_args["encoder2_c1"])
            parameters["encoder2_c1"], encoder2_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder2_c1"], return_parallel_config=True
            )
            ttnn_module_args["encoder2_c2"] = ttnn_module_args.encoder2["3"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.encoder2[3], model.encoder2[4])
            update_ttnn_module_args(ttnn_module_args["encoder2_c2"])
            ttnn_module_args["encoder2_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder2_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder2_c2"]["use_shallow_conv_variant"] = True
            # print("encoder2_c2", ttnn_module_args["encoder2_c2"])
            parameters["encoder2_c2"], encoder2_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["encoder2_c2"], return_parallel_config=True
            )
            ttnn_module_args["encoder3_c1"] = ttnn_module_args.encoder3["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder3[0], model.encoder3[1])
            update_ttnn_module_args(ttnn_module_args["encoder3_c1"])
            ttnn_module_args["encoder3_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder3_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder3_c1"]["use_shallow_conv_variant"] = False
            # print("encoder3_c1", ttnn_module_args["encoder3_c1"])
            parameters["encoder3_c1"], encoder3_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder3_c1"], return_parallel_config=True
            )
            ttnn_module_args["encoder3_c2"] = ttnn_module_args.encoder3["3"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.encoder3[3], model.encoder3[4])
            update_ttnn_module_args(ttnn_module_args["encoder3_c2"])
            ttnn_module_args["encoder3_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder3_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder3_c2"]["use_shallow_conv_variant"] = False
            # print("encoder3_c2", ttnn_module_args["encoder3_c2"])
            parameters["encoder3_c2"], encoder3_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["encoder3_c2"], return_parallel_config=True
            )
            ttnn_module_args["encoder4_c1"] = ttnn_module_args.encoder4["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.encoder4[0], model.encoder4[1])
            update_ttnn_module_args(ttnn_module_args["encoder4_c1"])
            ttnn_module_args["encoder4_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder4_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder4_c1"]["use_shallow_conv_variant"] = False
            # print("encoder4_c1", ttnn_module_args["encoder4_c1"])
            parameters["encoder4_c1"], encoder4_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["encoder4_c1"], return_parallel_config=True
            )
            ttnn_module_args["encoder4_c2"] = ttnn_module_args.encoder4["3"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.encoder4[3], model.encoder4[4])
            update_ttnn_module_args(ttnn_module_args["encoder4_c2"])
            ttnn_module_args["encoder4_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["encoder4_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["encoder4_c2"]["use_shallow_conv_variant"] = True
            # ttnn_module_args["encoder4_c2"]["use_1d_systolic_array"] = True
            # print("encoder4_c2", ttnn_module_args["encoder4_c2"])
            parameters["encoder4_c2"], encoder4_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["encoder4_c2"], return_parallel_config=True
            )
            ttnn_module_args["bottleneck_c1"] = ttnn_module_args.bottleneck["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.bottleneck[0], model.bottleneck[1])
            update_ttnn_module_args(ttnn_module_args["bottleneck_c1"])
            ttnn_module_args["bottleneck_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["bottleneck_c1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["bottleneck_c1"]["use_shallow_conv_variant"] = False
            ttnn_module_args["bottleneck_c1"]["use_1d_systolic_array"] = True
            # print("bottleneck_c1", ttnn_module_args["bottleneck_c1"])
            parameters["bottleneck_c1"], bottleneck_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["bottleneck_c1"], return_parallel_config=True
            )

            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.bottleneck[3], model.bottleneck[4])
            # print("conv2_weight",conv2_weight.shape)

            parameters["bottleneck_c2"] = {}
            parameters["bottleneck_c2"]["weight"] = conv2_weight
            parameters["bottleneck_c2"]["bias"] = conv2_bias

            # print("parameters",parameters["bottleneck_c2"]["weight"].shape)

            ttnn_module_args["decoder4_c1"] = ttnn_module_args.decoder4["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder4[0], model.decoder4[1])
            update_ttnn_module_args(ttnn_module_args["decoder4_c1"])
            ttnn_module_args["decoder4_c1"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder4_c1"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder4_c1"]["use_shallow_conv_variant"] = False
            parameters["decoder4_c1"], decoder4_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["decoder4_c1"], return_parallel_config=True
            )
            ttnn_module_args["decoder4_c2"] = ttnn_module_args.decoder4["3"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder4[3], model.decoder4[4])
            update_ttnn_module_args(ttnn_module_args["decoder4_c2"])
            ttnn_module_args["decoder4_c2"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder4_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["decoder4_c2"]["use_shallow_conv_variant"] = False
            parameters["decoder4_c2"], decoder4_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder4_c2"], return_parallel_config=True
            )
            ttnn_module_args["decoder3_c1"] = ttnn_module_args.decoder3["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder3[0], model.decoder3[1])
            update_ttnn_module_args(ttnn_module_args["decoder3_c1"])
            ttnn_module_args["decoder3_c1"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder3_c1"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder3_c1"]["use_shallow_conv_variant"] = False
            parameters["decoder3_c1"], decoder3_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["decoder3_c1"], return_parallel_config=True
            )
            ttnn_module_args["decoder3_c2"] = ttnn_module_args.decoder3["3"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder3[3], model.decoder3[4])
            update_ttnn_module_args(ttnn_module_args["decoder3_c2"])
            ttnn_module_args["decoder3_c2"]["use_1d_systolic_array"] = False
            ttnn_module_args["decoder3_c2"]["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
            ttnn_module_args["decoder3_c2"]["use_shallow_conv_variant"] = False
            parameters["decoder3_c2"], decoder3_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder3_c2"], return_parallel_config=True
            )
            ttnn_module_args["decoder2_c1"] = ttnn_module_args.decoder2["0"]
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder2[0], model.decoder2[1])
            update_ttnn_module_args(ttnn_module_args["decoder2_c1"])
            ttnn_module_args["decoder2_c1"]["use_1d_systolic_array"] = True
            ttnn_module_args["decoder2_c1"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 4 * 32
            }
            ttnn_module_args["decoder2_c1"]["use_shallow_conv_variant"] = False
            parameters["decoder2_c1"], decoder2_c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args["decoder2_c1"], return_parallel_config=True
            )
            ttnn_module_args["decoder2_c2"] = ttnn_module_args.decoder2["3"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder2[3], model.decoder2[4])
            update_ttnn_module_args(ttnn_module_args["decoder2_c2"])
            ttnn_module_args["decoder2_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["decoder2_c2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["decoder2_c2"]["use_shallow_conv_variant"] = True
            parameters["decoder2_c2"], decoder2_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder2_c2"], return_parallel_config=True
            )

            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.decoder1[0], model.decoder1[1])
            parameters["decoder1_c1"] = {}
            parameters["decoder1_c1"]["weight"] = conv1_weight
            parameters["decoder1_c1"]["bias"] = conv1_bias

            ttnn_module_args["decoder1_c2"] = ttnn_module_args.decoder1["3"]
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.decoder1[3], model.decoder1[4])
            update_ttnn_module_args(ttnn_module_args["decoder1_c2"])
            ttnn_module_args["decoder1_c2"]["use_1d_systolic_array"] = True
            ttnn_module_args["decoder1_c2"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 16 * 32
            }
            ttnn_module_args["decoder1_c2"]["use_shallow_conv_variant"] = True
            parameters["decoder1_c2"], decoder1_c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args["decoder1_c2"], return_parallel_config=True
            )

            parameters["conv"] = {}
            parameters["conv"]["weight"] = model.conv.weight
            parameters["conv"]["bias"] = model.conv.bias

            return parameters

    return custom_preprocessor


@skip_for_wormhole_b0()
def test_unet(reset_seeds):
    device = ttnn.open_device(device_id=0)
    state_dict = torch.load("tests/ttnn/integration_tests/unet/unet.pt", map_location=torch.device("cpu"))
    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = UNet()

    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
        # print(keys[i],values[i].shape)

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 3, 480, 640)  # (1, 3, 160, 160)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtUnet(device, parameters, new_state_dict)

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
    # print("Output shape:", torch_output_tensor.shape)
    output_tensor = ttnn.to_torch(output_tensor)
    # print("output_tensor",output_tensor.shape)
    output_tensor = output_tensor.reshape(1, 480, 640, 1)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    ttnn.close_device(device)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)  # pcc = 0.8045866608593448
