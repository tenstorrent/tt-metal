# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d
from models.experimental.functional_yolov4.reference.downsample5 import DownSample5
from models.experimental.functional_yolov4.tt.ttnn_downsample5 import TtDownSample5

import ttnn
import tt_lib


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = False  # ttnn_module_args.in_channels <= 256
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["activation"] = "relu"


def custom_preprocessor(device, model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, DownSample5):
        ttnn_module_args.c1["use_shallow_conv_variant"] = False
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
        update_ttnn_module_args(ttnn_module_args.c1)
        parameters["c1"], c1_parallel_config = preprocess_conv2d(
            conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
        )

        ttnn_module_args.c2["use_shallow_conv_variant"] = False
        ttnn_module_args.c2["weights_dtype"] = ttnn.bfloat8_b
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
        update_ttnn_module_args(ttnn_module_args.c2)
        parameters["c2"], c2_parallel_config = preprocess_conv2d(
            conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
        )

        ttnn_module_args.c3["use_shallow_conv_variant"] = False
        ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b
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

        ttnn_module_args.c4["use_shallow_conv_variant"] = False
        ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
        conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
        update_ttnn_module_args(ttnn_module_args.c4)
        parameters["c4"], c4_parallel_config = preprocess_conv2d(
            conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
        )

        ttnn_module_args.c5["use_shallow_conv_variant"] = False
        ttnn_module_args.c5["weights_dtype"] = ttnn.bfloat8_b
        conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
        update_ttnn_module_args(ttnn_module_args.c5)
        parameters["c5"], c5_parallel_config = preprocess_conv2d(
            conv5_weight, conv5_bias, ttnn_module_args.c5, return_parallel_config=True
        )

    return parameters
