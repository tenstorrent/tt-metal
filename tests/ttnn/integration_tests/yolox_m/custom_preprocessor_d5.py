# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import tt_lib

from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d

from models.experimental.functional_yolox_m.reference.dark5 import Dark5


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["activation"] = None


def custom_preprocessor(device, model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, Dark5):
        ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
        update_ttnn_module_args(ttnn_module_args.c1)
        parameters["c1"], c1_parallel_config = preprocess_conv2d(
            conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
        )

        ttnn_module_args.c2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c2["use_shallow_conv_variant"] = False

        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
        update_ttnn_module_args(ttnn_module_args.c2)
        parameters["c2"], c2_parallel_config = preprocess_conv2d(
            conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
        )

        ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b

        ttnn_module_args.c3["use_shallow_conv_variant"] = False
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
        update_ttnn_module_args(ttnn_module_args.c3)
        parameters["c3"], c3_parallel_config = preprocess_conv2d(
            conv3_weight, conv3_bias, ttnn_module_args.c3, return_parallel_config=True
        )

        conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
        parameters["c4"] = {}
        parameters["c4"]["weight"] = conv4_weight
        parameters["c4"]["bias"] = conv4_bias

        conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
        parameters["c5"] = {}
        parameters["c5"]["weight"] = conv5_weight
        parameters["c5"]["bias"] = conv5_bias

        parameters["bblock"] = {}
        for i, block in enumerate(model.bblock.module_list):
            conv1 = block[0]
            bn1 = block[1]
            conv2 = block[3]
            bn2 = block[4]

            ttnn_module_args["bblock"][f"bblock_{i}_conv1"] = ttnn_module_args["bblock"]["0"]
            ttnn_module_args["bblock"][f"bblock_{i}_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
            update_ttnn_module_args(ttnn_module_args["bblock"][f"bblock_{i}_conv1"])
            ttnn_module_args["bblock"][f"bblock_{i}_conv1"]["use_1d_systolic_array"] = False
            ttnn_module_args["bblock"][f"bblock_{i}_conv1"]["use_shallow_conv_variant"] = False
            parameters["bblock"][f"bblock_{i}_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["bblock"][f"bblock_{i}_conv1"], return_parallel_config=True
            )

            ttnn_module_args["bblock"][f"bblock_{i}_conv2"] = ttnn_module_args["bblock"]["3"]
            ttnn_module_args["bblock"][f"bblock_{i}_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
            update_ttnn_module_args(ttnn_module_args["bblock"][f"bblock_{i}_conv2"])
            parameters["bblock"][f"bblock_{i}_conv2"], _ = preprocess_conv2d(
                weight2, bias2, ttnn_module_args["bblock"][f"bblock_{i}_conv2"], return_parallel_config=True
            )

        ttnn_module_args.c6["use_shallow_conv_variant"] = (
            False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        )
        ttnn_module_args.c6["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c6["use_shallow_conv_variant"] = False
        conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
        update_ttnn_module_args(ttnn_module_args.c6)
        parameters["c6"], c6_parallel_config = preprocess_conv2d(
            conv6_weight, conv6_bias, ttnn_module_args.c6, return_parallel_config=True
        )

        return parameters
