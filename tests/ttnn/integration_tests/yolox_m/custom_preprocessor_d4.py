# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn

from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d

from models.experimental.functional_yolox_m.reference.dark4 import Dark4


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
    if isinstance(model, Dark4):
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
        ttnn_module_args.c2["use_1d_systolic_array"] = True
        parameters["c2"], c2_parallel_config = preprocess_conv2d(
            conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
        )

        ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c3["use_shallow_conv_variant"] = False

        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
        update_ttnn_module_args(ttnn_module_args.c3)
        ttnn_module_args.c3["use_1d_systolic_array"] = True
        parameters["c3"], c3_parallel_config = preprocess_conv2d(
            conv3_weight, conv3_bias, ttnn_module_args.c3, return_parallel_config=True
        )

        parameters["bblock"] = {}
        for i, block in enumerate(model.bblock.module_list):
            conv1 = block[0]
            bn1 = block[1]
            conv2 = block[3]
            bn2 = block[4]

            ttnn_module_args["bblock"][f"bblock_{i}_conv1"] = ttnn_module_args["bblock"]["0"]
            ttnn_module_args["bblock"][f"bblock_{i}_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args["bblock"][f"bblock_{i}_conv1"]["use_shallow_conv_variant"] = False
            weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
            update_ttnn_module_args(ttnn_module_args["bblock"][f"bblock_{i}_conv1"])
            parameters["bblock"][f"bblock_{i}_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["bblock"][f"bblock_{i}_conv1"], return_parallel_config=True
            )

            ttnn_module_args["bblock"][f"bblock_{i}_conv2"] = ttnn_module_args["bblock"]["3"]
            ttnn_module_args["bblock"][f"bblock_{i}_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args["bblock"][f"bblock_{i}_conv2"]["use_shallow_conv_variant"] = False
            weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
            update_ttnn_module_args(ttnn_module_args["bblock"][f"bblock_{i}_conv2"])
            parameters["bblock"][f"bblock_{i}_conv2"], _ = preprocess_conv2d(
                weight2, bias2, ttnn_module_args["bblock"][f"bblock_{i}_conv2"], return_parallel_config=True
            )

        ttnn_module_args.c4["use_shallow_conv_variant"] = False

        ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
        conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
        update_ttnn_module_args(ttnn_module_args.c4)
        parameters["c4"], c4_parallel_config = preprocess_conv2d(
            conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
        )

        return parameters
