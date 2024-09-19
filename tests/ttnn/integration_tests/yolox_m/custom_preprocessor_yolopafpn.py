# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import tt_lib

from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d

from models.experimental.functional_yolox_m.reference.yolo_pafpn import YOLOPAFPN


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
    if isinstance(model, YOLOPAFPN):
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

        ttnn_module_args.c4["use_shallow_conv_variant"] = False
        ttnn_module_args.c4["weights_dtype"] = ttnn.bfloat8_b
        conv4_weight, conv4_bias = fold_batch_norm2d_into_conv2d(model.c4, model.b4)
        update_ttnn_module_args(ttnn_module_args.c4)
        parameters["c4"], c4_parallel_config = preprocess_conv2d(
            conv4_weight, conv4_bias, ttnn_module_args.c4, return_parallel_config=True
        )

        parameters["bblock1"] = {}
        for i, block in enumerate(model.bblock1.module_list):
            conv1 = block[0]
            bn1 = block[1]
            conv2 = block[3]
            bn2 = block[4]

            ttnn_module_args["bblock1"][f"bblock1_{i}_conv1"] = ttnn_module_args["bblock1"]["0"]
            ttnn_module_args["bblock1"][f"bblock1_{i}_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
            update_ttnn_module_args(ttnn_module_args["bblock1"][f"bblock1_{i}_conv1"])
            ttnn_module_args["bblock1"][f"bblock1_{i}_conv1"]["use_1d_systolic_array"]
            parameters["bblock1"][f"bblock1_{i}_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["bblock1"][f"bblock1_{i}_conv1"], return_parallel_config=True
            )

            ttnn_module_args["bblock1"][f"bblock1_{i}_conv2"] = ttnn_module_args["bblock1"]["3"]
            ttnn_module_args["bblock1"][f"bblock1_{i}_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
            update_ttnn_module_args(ttnn_module_args["bblock1"][f"bblock1_{i}_conv2"])
            parameters["bblock1"][f"bblock1_{i}_conv2"], _ = preprocess_conv2d(
                weight2, bias2, ttnn_module_args["bblock1"][f"bblock1_{i}_conv2"], return_parallel_config=True
            )

        ttnn_module_args.c5["use_shallow_conv_variant"] = False
        ttnn_module_args.c5["weights_dtype"] = ttnn.bfloat8_b
        conv5_weight, conv5_bias = fold_batch_norm2d_into_conv2d(model.c5, model.b5)
        update_ttnn_module_args(ttnn_module_args.c5)
        parameters["c5"], c5_parallel_config = preprocess_conv2d(
            conv5_weight, conv5_bias, ttnn_module_args.c5, return_parallel_config=True
        )

        ttnn_module_args.c6["use_shallow_conv_variant"] = False
        ttnn_module_args.c6["weights_dtype"] = ttnn.bfloat8_b
        conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
        update_ttnn_module_args(ttnn_module_args.c6)
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

        ttnn_module_args.c8["use_shallow_conv_variant"] = (
            False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        )
        ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
        conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
        update_ttnn_module_args(ttnn_module_args.c8)
        parameters["c8"], c8_parallel_config = preprocess_conv2d(
            conv8_weight, conv8_bias, ttnn_module_args.c8, return_parallel_config=True
        )

        parameters["bblock2"] = {}
        for i, block in enumerate(model.bblock2.module_list):
            conv1 = block[0]
            bn1 = block[1]
            conv2 = block[3]
            bn2 = block[4]

            ttnn_module_args["bblock2"][f"bblock2_{i}_conv1"] = ttnn_module_args["bblock2"]["0"]
            ttnn_module_args["bblock2"][f"bblock2_{i}_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
            update_ttnn_module_args(ttnn_module_args["bblock2"][f"bblock2_{i}_conv1"])
            parameters["bblock2"][f"bblock2_{i}_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["bblock2"][f"bblock2_{i}_conv1"], return_parallel_config=True
            )

            ttnn_module_args["bblock2"][f"bblock2_{i}_conv2"] = ttnn_module_args["bblock2"]["3"]
            ttnn_module_args["bblock2"][f"bblock2_{i}_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
            update_ttnn_module_args(ttnn_module_args["bblock2"][f"bblock2_{i}_conv2"])
            parameters["bblock2"][f"bblock2_{i}_conv2"], _ = preprocess_conv2d(
                weight2, bias2, ttnn_module_args["bblock2"][f"bblock2_{i}_conv2"], return_parallel_config=True
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
        parameters["c10"], c10_parallel_config = preprocess_conv2d(
            conv10_weight, conv10_bias, ttnn_module_args.c10, return_parallel_config=True
        )

        ttnn_module_args.c11["use_shallow_conv_variant"] = False
        ttnn_module_args.c11["weights_dtype"] = ttnn.bfloat8_b
        conv11_weight, conv11_bias = fold_batch_norm2d_into_conv2d(model.c11, model.b11)
        update_ttnn_module_args(ttnn_module_args.c11)
        parameters["c11"], c11_parallel_config = preprocess_conv2d(
            conv11_weight, conv11_bias, ttnn_module_args.c11, return_parallel_config=True
        )

        ttnn_module_args.c12["use_shallow_conv_variant"] = False
        ttnn_module_args.c12["weights_dtype"] = ttnn.bfloat8_b
        conv12_weight, conv12_bias = fold_batch_norm2d_into_conv2d(model.c12, model.b12)
        update_ttnn_module_args(ttnn_module_args.c12)
        parameters["c12"], c12_parallel_config = preprocess_conv2d(
            conv12_weight, conv12_bias, ttnn_module_args.c12, return_parallel_config=True
        )

        parameters["bblock3"] = {}
        for i, block in enumerate(model.bblock3.module_list):
            conv1 = block[0]
            bn1 = block[1]
            conv2 = block[3]
            bn2 = block[4]

            ttnn_module_args["bblock3"][f"bblock3_{i}_conv1"] = ttnn_module_args["bblock3"]["0"]
            ttnn_module_args["bblock3"][f"bblock3_{i}_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
            update_ttnn_module_args(ttnn_module_args["bblock3"][f"bblock3_{i}_conv1"])
            parameters["bblock3"][f"bblock3_{i}_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["bblock3"][f"bblock3_{i}_conv1"], return_parallel_config=True
            )

            ttnn_module_args["bblock3"][f"bblock3_{i}_conv2"] = ttnn_module_args["bblock3"]["3"]
            ttnn_module_args["bblock3"][f"bblock3_{i}_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
            update_ttnn_module_args(ttnn_module_args["bblock3"][f"bblock3_{i}_conv2"])
            parameters["bblock3"][f"bblock3_{i}_conv2"], _ = preprocess_conv2d(
                weight2, bias2, ttnn_module_args["bblock3"][f"bblock3_{i}_conv2"], return_parallel_config=True
            )

        ttnn_module_args.c13["use_shallow_conv_variant"] = False
        ttnn_module_args.c13["weights_dtype"] = ttnn.bfloat8_b
        conv13_weight, conv13_bias = fold_batch_norm2d_into_conv2d(model.c13, model.b13)
        update_ttnn_module_args(ttnn_module_args.c13)
        parameters["c13"], c13_parallel_config = preprocess_conv2d(
            conv13_weight, conv13_bias, ttnn_module_args.c13, return_parallel_config=True
        )

        ttnn_module_args.c14["use_shallow_conv_variant"] = False
        ttnn_module_args.c14["weights_dtype"] = ttnn.bfloat8_b
        conv14_weight, conv14_bias = fold_batch_norm2d_into_conv2d(model.c14, model.b14)
        update_ttnn_module_args(ttnn_module_args.c14)
        parameters["c14"], c14_parallel_config = preprocess_conv2d(
            conv14_weight, conv14_bias, ttnn_module_args.c14, return_parallel_config=True
        )

        ttnn_module_args.c15["use_shallow_conv_variant"] = False
        ttnn_module_args.c15["weights_dtype"] = ttnn.bfloat8_b
        conv15_weight, conv15_bias = fold_batch_norm2d_into_conv2d(model.c15, model.b15)
        update_ttnn_module_args(ttnn_module_args.c15)
        parameters["c15"], c15_parallel_config = preprocess_conv2d(
            conv15_weight, conv15_bias, ttnn_module_args.c15, return_parallel_config=True
        )

        ttnn_module_args.c16["use_shallow_conv_variant"] = False
        ttnn_module_args.c16["weights_dtype"] = ttnn.bfloat8_b
        conv16_weight, conv16_bias = fold_batch_norm2d_into_conv2d(model.c16, model.b16)
        update_ttnn_module_args(ttnn_module_args.c16)
        parameters["c16"], c16_parallel_config = preprocess_conv2d(
            conv16_weight, conv16_bias, ttnn_module_args.c16, return_parallel_config=True
        )

        parameters["bblock4"] = {}
        for i, block in enumerate(model.bblock4.module_list):
            conv1 = block[0]
            bn1 = block[1]
            conv2 = block[3]
            bn2 = block[4]

            ttnn_module_args["bblock4"][f"bblock4_{i}_conv1"] = ttnn_module_args["bblock4"]["0"]
            ttnn_module_args["bblock4"][f"bblock4_{i}_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(conv1, bn1)
            update_ttnn_module_args(ttnn_module_args["bblock4"][f"bblock4_{i}_conv1"])
            ttnn_module_args["bblock4"][f"bblock4_{i}_conv1"]["use_1d_systolic_array"] = True
            parameters["bblock4"][f"bblock4_{i}_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["bblock4"][f"bblock4_{i}_conv1"], return_parallel_config=True
            )

            ttnn_module_args["bblock4"][f"bblock4_{i}_conv2"] = ttnn_module_args["bblock4"]["3"]
            ttnn_module_args["bblock4"][f"bblock4_{i}_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight2, bias2 = fold_batch_norm2d_into_conv2d(conv2, bn2)
            update_ttnn_module_args(ttnn_module_args["bblock4"][f"bblock4_{i}_conv2"])
            ttnn_module_args["bblock4"][f"bblock4_{i}_conv2"]["use_1d_systolic_array"] = True
            parameters["bblock4"][f"bblock4_{i}_conv2"], _ = preprocess_conv2d(
                weight2, bias2, ttnn_module_args["bblock4"][f"bblock4_{i}_conv2"], return_parallel_config=True
            )

        return parameters
