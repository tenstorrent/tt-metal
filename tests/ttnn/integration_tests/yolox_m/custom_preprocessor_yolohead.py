# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn

from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d

from models.experimental.functional_yolox_m.reference.yolo_head import YOLOXHead


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
    if isinstance(model, YOLOXHead):
        ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1["use_shallow_conv_variant"] = False
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
        update_ttnn_module_args(ttnn_module_args.c1)
        parameters["c1"], c1_parallel_config = preprocess_conv2d(
            conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
        )

        ttnn_module_args.c2["weights_dtype"] = ttnn.bfloat8_b

        ttnn_module_args.c2["use_shallow_conv_variant"] = False
        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
        update_ttnn_module_args(ttnn_module_args.c2)
        ttnn_module_args.c4["use_1d_systolic_array"] = False
        parameters["c2"], c2_parallel_config = preprocess_conv2d(
            conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
        )

        ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b

        ttnn_module_args.c3["use_shallow_conv_variant"] = False
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
        update_ttnn_module_args(ttnn_module_args.c3)
        ttnn_module_args.c3["use_1d_systolic_array"] = False
        parameters["c3"], c3_parallel_config = preprocess_conv2d(
            conv3_weight, conv3_bias, ttnn_module_args.c3, return_parallel_config=True
        )

        ttnn_module_args.c4["use_shallow_conv_variant"] = False
        ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
        conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
        update_ttnn_module_args(ttnn_module_args.c4)
        ttnn_module_args.c4["use_1d_systolic_array"] = False
        parameters["c4"], c4_parallel_config = preprocess_conv2d(
            conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
        )

        ttnn_module_args.c5["use_shallow_conv_variant"] = False
        ttnn_module_args.c5["weights_dtype"] = ttnn.bfloat8_b
        conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
        update_ttnn_module_args(ttnn_module_args.c5)
        ttnn_module_args.c5["use_1d_systolic_array"] = False
        parameters["c5"], c5_parallel_config = preprocess_conv2d(
            conv5_weight, conv5_bias, ttnn_module_args.c5, return_parallel_config=True
        )

        ttnn_module_args.c6["use_shallow_conv_variant"] = False
        ttnn_module_args.c6["weights_dtype"] = ttnn.bfloat8_b
        conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
        update_ttnn_module_args(ttnn_module_args.c6)
        ttnn_module_args.c6["use_1d_systolic_array"] = False
        parameters["c6"], c6_parallel_config = preprocess_conv2d(
            conv6_weight, conv6_bias, ttnn_module_args.c6, return_parallel_config=True
        )

        ttnn_module_args.c7["use_shallow_conv_variant"] = False
        ttnn_module_args.c7["weights_dtype"] = ttnn.bfloat8_b
        conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
        update_ttnn_module_args(ttnn_module_args.c7)
        parameters["c7"], c7_parallel_config = preprocess_conv2d(
            conv7_weight, conv7_bias, ttnn_module_args.c7, return_parallel_config=True
        )

        ttnn_module_args.c8["use_shallow_conv_variant"] = False
        ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
        conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
        update_ttnn_module_args(ttnn_module_args.c8)
        parameters["c8"], c8_parallel_config = preprocess_conv2d(
            conv8_weight, conv8_bias, ttnn_module_args.c8, return_parallel_config=True
        )

        ttnn_module_args.c9["use_shallow_conv_variant"] = False
        ttnn_module_args.c9["weights_dtype"] = ttnn.bfloat8_b
        conv9_weight, conv9_bias = fold_batch_norm2d_into_conv2d(model.c9, model.b9)
        update_ttnn_module_args(ttnn_module_args.c9)
        ttnn_module_args.c9["use_1d_systolic_array"] = False
        parameters["c9"], c9_parallel_config = preprocess_conv2d(
            conv9_weight, conv9_bias, ttnn_module_args.c9, return_parallel_config=True
        )

        ttnn_module_args.c10["use_shallow_conv_variant"] = False
        ttnn_module_args.c10["weights_dtype"] = ttnn.bfloat8_b
        conv10_weight, conv10_bias = fold_batch_norm2d_into_conv2d(model.c10, model.b10)
        update_ttnn_module_args(ttnn_module_args.c10)
        ttnn_module_args.c10["use_1d_systolic_array"] = False
        parameters["c10"], c10_parallel_config = preprocess_conv2d(
            conv10_weight, conv10_bias, ttnn_module_args.c10, return_parallel_config=True
        )

        ttnn_module_args.c11["use_shallow_conv_variant"] = False
        ttnn_module_args.c11["weights_dtype"] = ttnn.bfloat8_b
        conv11_weight, conv11_bias = fold_batch_norm2d_into_conv2d(model.c11, model.b11)
        update_ttnn_module_args(ttnn_module_args.c11)
        ttnn_module_args.c11["use_1d_systolic_array"] = False
        parameters["c11"], c11_parallel_config = preprocess_conv2d(
            conv11_weight, conv11_bias, ttnn_module_args.c11, return_parallel_config=True
        )

        ttnn_module_args.c12["use_shallow_conv_variant"] = False
        ttnn_module_args.c12["weights_dtype"] = ttnn.bfloat8_b
        conv12_weight, conv12_bias = fold_batch_norm2d_into_conv2d(model.c12, model.b12)
        update_ttnn_module_args(ttnn_module_args.c12)
        ttnn_module_args.c12["use_1d_systolic_array"] = False
        parameters["c12"], c12_parallel_config = preprocess_conv2d(
            conv12_weight, conv12_bias, ttnn_module_args.c12, return_parallel_config=True
        )

        ttnn_module_args.c13["use_shallow_conv_variant"] = False
        ttnn_module_args.c13["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c13)
        parameters["c13"], c13_parallel_config = preprocess_conv2d(
            model.c13.weight, model.c13.bias, ttnn_module_args.c13, return_parallel_config=True
        )

        ttnn_module_args.c14["use_shallow_conv_variant"] = False
        ttnn_module_args.c14["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c14)
        parameters["c14"], c14_parallel_config = preprocess_conv2d(
            model.c14.weight, model.c14.bias, ttnn_module_args.c14, return_parallel_config=True
        )

        ttnn_module_args.c15["use_shallow_conv_variant"] = False
        ttnn_module_args.c15["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c15)
        parameters["c15"], c15_parallel_config = preprocess_conv2d(
            model.c15.weight, model.c15.bias, ttnn_module_args.c15, return_parallel_config=True
        )

        ttnn_module_args.c16["use_shallow_conv_variant"] = False
        ttnn_module_args.c16["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c16)
        parameters["c16"], c16_parallel_config = preprocess_conv2d(
            model.c16.weight, model.c16.bias, ttnn_module_args.c16, return_parallel_config=True
        )

        ttnn_module_args.c17["use_shallow_conv_variant"] = False
        ttnn_module_args.c17["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c17)
        parameters["c17"], c17_parallel_config = preprocess_conv2d(
            model.c17.weight, model.c17.bias, ttnn_module_args.c17, return_parallel_config=True
        )

        ttnn_module_args.c18["use_shallow_conv_variant"] = False
        ttnn_module_args.c18["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c18)
        parameters["c18"], c18_parallel_config = preprocess_conv2d(
            model.c18.weight, model.c18.bias, ttnn_module_args.c18, return_parallel_config=True
        )

        ttnn_module_args.c19["use_shallow_conv_variant"] = False
        ttnn_module_args.c19["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c19)
        parameters["c19"], c19_parallel_config = preprocess_conv2d(
            model.c19.weight, model.c19.bias, ttnn_module_args.c19, return_parallel_config=True
        )

        ttnn_module_args.c20["use_shallow_conv_variant"] = False
        ttnn_module_args.c20["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c20)
        parameters["c20"], c20_parallel_config = preprocess_conv2d(
            model.c20.weight, model.c20.bias, ttnn_module_args.c20, return_parallel_config=True
        )

        ttnn_module_args.c21["use_shallow_conv_variant"] = False
        ttnn_module_args.c21["weights_dtype"] = ttnn.bfloat8_b
        update_ttnn_module_args(ttnn_module_args.c21)
        parameters["c21"], c21_parallel_config = preprocess_conv2d(
            model.c21.weight, model.c21.bias, ttnn_module_args.c21, return_parallel_config=True
        )

        ttnn_module_args.c22["use_shallow_conv_variant"] = False
        ttnn_module_args.c22["weights_dtype"] = ttnn.bfloat8_b
        conv22_weight, conv22_bias = fold_batch_norm2d_into_conv2d(model.c22, model.b22)
        update_ttnn_module_args(ttnn_module_args.c22)
        parameters["c22"], c22_parallel_config = preprocess_conv2d(
            conv22_weight, conv22_bias, ttnn_module_args.c22, return_parallel_config=True
        )

        ttnn_module_args.c23["use_shallow_conv_variant"] = False
        ttnn_module_args.c23["weights_dtype"] = ttnn.bfloat8_b
        conv23_weight, conv23_bias = fold_batch_norm2d_into_conv2d(model.c23, model.b23)
        update_ttnn_module_args(ttnn_module_args.c23)
        parameters["c23"], c23_parallel_config = preprocess_conv2d(
            conv23_weight, conv23_bias, ttnn_module_args.c23, return_parallel_config=True
        )

        ttnn_module_args.c24["use_shallow_conv_variant"] = False
        ttnn_module_args.c24["weights_dtype"] = ttnn.bfloat8_b
        conv24_weight, conv24_bias = fold_batch_norm2d_into_conv2d(model.c24, model.b24)
        update_ttnn_module_args(ttnn_module_args.c24)
        parameters["c24"], c24_parallel_config = preprocess_conv2d(
            conv24_weight, conv24_bias, ttnn_module_args.c24, return_parallel_config=True
        )

        return parameters
