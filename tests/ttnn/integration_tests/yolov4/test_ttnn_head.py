# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_yolov4.reference.head import Head
from models.experimental.functional_yolov4.tt.ttnn_head import TtHead

import time
import tt_lib as ttl
import tt_lib.profiler as profiler

import ttnn
import tt_lib
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d
import ttnn


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["use_shallow_conv_variant"] = False


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        print("We do reach here!")
        parameters = {}
        if isinstance(model, Head):
            ttnn_module_args.c1["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c1["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c1["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c1["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c1["deallocate_activation"] = True
            ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = None
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
            update_ttnn_module_args(ttnn_module_args.c1)
            parameters["c1"], c1_parallel_config = preprocess_conv2d(
                conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
            )

            ttnn_module_args.c2["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c2["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c2["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c2["activation"] = "relu"  # Fuse relu with conv2
            ttnn_module_args.c2["deallocate_activation"] = True
            ttnn_module_args.c2["conv_blocking_and_parallelization_config_override"] = None

            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
            update_ttnn_module_args(ttnn_module_args.c2)
            parameters["c2"], c2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
            )

            ttnn_module_args.c3["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c3["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c3["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c3["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c3["deallocate_activation"] = True
            ttnn_module_args.c3["conv_blocking_and_parallelization_config_override"] = None

            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
            update_ttnn_module_args(ttnn_module_args.c3)
            parameters["c3"], c3_parallel_config = preprocess_conv2d(
                conv3_weight, conv3_bias, ttnn_module_args.c3, return_parallel_config=True
            )

            ttnn_module_args.c4["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c4["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c4["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c4["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c4["deallocate_activation"] = True
            ttnn_module_args.c4["conv_blocking_and_parallelization_config_override"] = None

            conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
            update_ttnn_module_args(ttnn_module_args.c4)
            parameters["c4"], c4_parallel_config = preprocess_conv2d(
                conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
            )

            ttnn_module_args.c5["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c5["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c5["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c5["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c5["deallocate_activation"] = True
            ttnn_module_args.c5["conv_blocking_and_parallelization_config_override"] = None

            conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
            update_ttnn_module_args(ttnn_module_args.c5)
            parameters["c5"], c5_parallel_config = preprocess_conv2d(
                conv5_weight, conv5_bias, ttnn_module_args.c5, return_parallel_config=True
            )

            ttnn_module_args.c6["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c6["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c6["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c6["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c6["deallocate_activation"] = True
            ttnn_module_args.c6["conv_blocking_and_parallelization_config_override"] = None

            conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
            update_ttnn_module_args(ttnn_module_args.c6)
            parameters["c6"], c6_parallel_config = preprocess_conv2d(
                conv6_weight, conv6_bias, ttnn_module_args.c6, return_parallel_config=True
            )

            ttnn_module_args.c7["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c7["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c7["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c7["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c7["deallocate_activation"] = True
            ttnn_module_args.c7["conv_blocking_and_parallelization_config_override"] = None

            conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
            update_ttnn_module_args(ttnn_module_args.c7)
            parameters["c7"], c7_parallel_config = preprocess_conv2d(
                conv7_weight, conv7_bias, ttnn_module_args.c7, return_parallel_config=True
            )
            print("parameters['c7'] type is: ", type(parameters["c7"]))
            print("parameters['c7'] is: ", parameters["c7"])
            ttnn_module_args.c8["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c8["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c8["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c8["deallocate_activation"] = True
            ttnn_module_args.c8["conv_blocking_and_parallelization_config_override"] = None
            # conv8_weight, conv8_bias = model.c8, model.b8
            conv8_weight = model.c8.weight
            print("conv8_weight: ", conv8_weight)
            conv8_bias = None
            update_ttnn_module_args(ttnn_module_args.c8)
            parameters["c8"] = {}
            parameters["c8"]["weight"] = conv8_weight
            #            parameters["c8"], c8_parallel_config = preprocess_conv2d(
            #                conv8_weight, conv8_bias, ttnn_module_args.c8, return_parallel_config=True
            #            )

            ttnn_module_args.c9["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c9["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c9["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c9["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c9["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c9["deallocate_activation"] = True
            ttnn_module_args.c9["conv_blocking_and_parallelization_config_override"] = None

            conv9_weight, conv9_bias = fold_batch_norm2d_into_conv2d(model.c9, model.b9)
            update_ttnn_module_args(ttnn_module_args.c9)
            parameters["c9"], c9_parallel_config = preprocess_conv2d(
                conv9_weight, conv9_bias, ttnn_module_args.c9, return_parallel_config=True
            )

            ttnn_module_args.c10["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c10["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c10["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c10["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c10["deallocate_activation"] = True
            ttnn_module_args.c10["conv_blocking_and_parallelization_config_override"] = None
            conv10_weight = model.c10.weight
            parameters["c10"] = {}
            parameters["c10"]["weight"] = conv10_weight
            # conv10_weight, conv10_bias = model.c10, model.b10
            update_ttnn_module_args(ttnn_module_args.c10)
            #            parameters["c10"], c10_parallel_config = preprocess_conv2d(
            #                conv10_weight, conv10_bias, ttnn_module_args.c10, return_parallel_config=True
            #            )

            ttnn_module_args.c11["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c11["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c11["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c11["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c11["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c11["deallocate_activation"] = True
            ttnn_module_args.c11["conv_blocking_and_parallelization_config_override"] = None

            conv11_weight, conv11_bias = fold_batch_norm2d_into_conv2d(model.c11, model.b11)
            update_ttnn_module_args(ttnn_module_args.c11)
            parameters["c11"], c11_parallel_config = preprocess_conv2d(
                conv11_weight, conv11_bias, ttnn_module_args.c11, return_parallel_config=True
            )

            ttnn_module_args.c12["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c12["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c12["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c12["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c12["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c12["deallocate_activation"] = True
            ttnn_module_args.c12["conv_blocking_and_parallelization_config_override"] = None

            conv12_weight, conv12_bias = fold_batch_norm2d_into_conv2d(model.c12, model.b12)
            update_ttnn_module_args(ttnn_module_args.c12)
            parameters["c12"], c12_parallel_config = preprocess_conv2d(
                conv12_weight, conv12_bias, ttnn_module_args.c12, return_parallel_config=True
            )

            ttnn_module_args.c13["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c13["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c13["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c13["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c13["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c13["deallocate_activation"] = True
            ttnn_module_args.c13["conv_blocking_and_parallelization_config_override"] = None

            conv13_weight, conv13_bias = fold_batch_norm2d_into_conv2d(model.c13, model.b13)
            update_ttnn_module_args(ttnn_module_args.c13)
            parameters["c13"], c13_parallel_config = preprocess_conv2d(
                conv13_weight, conv13_bias, ttnn_module_args.c13, return_parallel_config=True
            )

            ttnn_module_args.c14["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c14["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c14["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c14["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c14["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c14["deallocate_activation"] = True
            ttnn_module_args.c14["conv_blocking_and_parallelization_config_override"] = None

            conv14_weight, conv14_bias = fold_batch_norm2d_into_conv2d(model.c14, model.b14)
            update_ttnn_module_args(ttnn_module_args.c14)
            parameters["c14"], c14_parallel_config = preprocess_conv2d(
                conv14_weight, conv14_bias, ttnn_module_args.c14, return_parallel_config=True
            )

            ttnn_module_args.c15["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c15["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c15["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c15["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c15["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c15["deallocate_activation"] = True
            ttnn_module_args.c15["conv_blocking_and_parallelization_config_override"] = None

            conv15_weight, conv15_bias = fold_batch_norm2d_into_conv2d(model.c15, model.b15)
            update_ttnn_module_args(ttnn_module_args.c15)
            parameters["c15"], c15_parallel_config = preprocess_conv2d(
                conv15_weight, conv15_bias, ttnn_module_args.c15, return_parallel_config=True
            )

            ttnn_module_args.c16["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c16["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c16["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c16["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c16["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.c16["deallocate_activation"] = True
            ttnn_module_args.c16["conv_blocking_and_parallelization_config_override"] = None

            conv16_weight, conv16_bias = fold_batch_norm2d_into_conv2d(model.c16, model.b16)
            update_ttnn_module_args(ttnn_module_args.c16)
            parameters["c16"], c16_parallel_config = preprocess_conv2d(
                conv16_weight, conv16_bias, ttnn_module_args.c16, return_parallel_config=True
            )

            ttnn_module_args.c17["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c17["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c17["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c17["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c17["deallocate_activation"] = True
            ttnn_module_args.c17["conv_blocking_and_parallelization_config_override"] = None
            # conv17_weight, conv17_bias = model.c17, model.b17
            conv17_weight, conv17_bias = fold_batch_norm2d_into_conv2d(model.c17, model.b17)
            update_ttnn_module_args(ttnn_module_args.c17)
            parameters["c17"], c17_parallel_config = preprocess_conv2d(
                conv17_weight, conv17_bias, ttnn_module_args.c17, return_parallel_config=True
            )
            ttnn_module_args.c18["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args.c18["use_shallow_conv_variant"] = (
                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
            )
            ttnn_module_args.c18["dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c18["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.c18["deallocate_activation"] = True
            ttnn_module_args.c18["conv_blocking_and_parallelization_config_override"] = None
            conv18_weight = model.c18.weight
            parameters["c18"] = {}
            parameters["c18"]["weight"] = conv18_weight
            # conv10_weight, conv10_bias = model.c10, model.b10
            update_ttnn_module_args(ttnn_module_args.c18)
        return parameters

    return custom_preprocessor


@skip_for_wormhole_b0()
def test_head(device, reset_seeds):
    state_dict = torch.load("tests/ttnn/integration_tests/yolov4/yolov4.pth")
    ds_state_dict = {k: v for k, v in state_dict.items() if k.startswith("head.")}
    torch_model = Head()

    # Initialize a new state dictionary
    new_state_dict = torch_model.state_dict()
    old_state_dict = torch_model.state_dict().copy()  # or deepcopy()
    # Get the sizes of the parameters from the Head module
    sizes = {}
    for name, param in torch_model.named_parameters():
        sizes[name] = param.size()

    # Iterate over the parameters of the model
    for name, param in torch_model.named_parameters():
        print("name: ", name)
        found_match = False
        # Check if the parameter has a corresponding key in the loaded state dictionary
        for ds_name, ds_param in ds_state_dict.items():
            if param.size() == ds_param.size():
                new_state_dict[name] = ds_param
                found_match = True
                print("found_match: ", found_match)
                print("new_state_dict[name] == old_state_dict[name]: ", new_state_dict[name] == old_state_dict[name])
                break  # Found a match, move to the next parameter

    # Update the parameters in the model
    torch_model.load_state_dict(new_state_dict)

    #    # Update the parameters in the model
    #    torch_model.load_state_dict(new_state_dict)

    torch_model.eval()

    torch_input_tensor = torch.randn(1, 128, 40, 40)  # Batch size of 1, 128 input channels, 160x160 height and width
    torch_output_tensor1, torch_output_tensor2, torch_output_tensor3 = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtHead(device, parameters)

    # Tensor Preprocessing
    #
    input_shape = torch_input_tensor.shape
    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    input_tensor = input_tensor.reshape(
        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
    )
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    output_tensor1, output_tensor2, output_tensor3 = ttnn_model(device, input_tensor)

    #
    # Tensor Postprocessing
    #
    output_tensor1 = ttnn.to_torch(output_tensor1)
    output_tensor1 = output_tensor1.reshape(1, 20, 20, 255)
    output_tensor1 = torch.permute(output_tensor1, (0, 3, 1, 2))
    output_tensor1 = output_tensor1.to(torch_input_tensor.dtype)

    output_tensor2 = ttnn.to_torch(output_tensor2)
    output_tensor2 = output_tensor2.reshape(1, 40, 40, 255)
    output_tensor2 = torch.permute(output_tensor2, (0, 3, 1, 2))
    output_tensor2 = output_tensor2.to(torch_input_tensor.dtype)

    output_tensor3 = ttnn.to_torch(output_tensor3)
    output_tensor3 = output_tensor3.reshape(1, 40, 40, 255)
    output_tensor3 = torch.permute(output_tensor3, (0, 3, 1, 2))
    output_tensor3 = output_tensor3.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor1, output_tensor1, pcc=0.99)
    # assert_with_pcc(torch_output_tensor2, output_tensor2, pcc=0.99)
    # assert_with_pcc(torch_output_tensor3, output_tensor3, pcc=0.99)
