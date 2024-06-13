# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from torch import nn, Tensor
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.experimental.functional_fadnetpp.reference.dispnetres import DispNetRes
from models.experimental.functional_fadnetpp.tt.tt_dispnetres import TtDispNetRes
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d
from loguru import logger


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["activation"] = "relu"
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi


def create_custom_preprocessor(device, resblock=True):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, DispNetRes):
            ttnn_module_args["conv1"] = ttnn_module_args.conv1["0"]
            conv1_weight, conv1_bias = torch_model.conv1[0].weight, torch_model.conv1[0].bias
            parameters["conv1"] = {}
            parameters["conv1"]["weight"] = conv1_weight
            parameters["conv1"]["bias"] = conv1_bias
            if resblock:
                parameters["conv2"] = {}

                ttnn_module_args["conv2"]["resblock_1_conv1"] = ttnn_module_args["conv2"]["resblock_1_conv1"]
                conv2_weight1, conv2_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv2.resblock_1_conv1, torch_model.conv2.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_1_conv1"])
                parameters["conv2"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv2_weight1,
                    conv2_bias1,
                    ttnn_module_args["conv2"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv2"]["resblock_2_conv2"] = ttnn_module_args["conv2"]["resblock_2_conv2"]
                conv2_weight2, conv2_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv2.resblock_2_conv2, torch_model.conv2.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_2_conv2"])
                parameters["conv2"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv2_weight2,
                    conv2_bias2,
                    ttnn_module_args["conv2"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv2"]["resblock_sc_conv"] = ttnn_module_args["conv2"]["shortcut_c"]
                conv2_weight3, conv2_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv2.shortcut_c, torch_model.conv2.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv2"]["resblock_sc_conv"])
                parameters["conv2"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv2_weight3,
                    conv2_bias3,
                    ttnn_module_args["conv2"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

                parameters["conv3"] = {}

                ttnn_module_args["conv3"]["resblock_1_conv1"] = ttnn_module_args["conv3"]["resblock_1_conv1"]
                conv3_weight1, conv3_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3.resblock_1_conv1, torch_model.conv3.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_1_conv1"])
                parameters["conv3"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv3_weight1,
                    conv3_bias1,
                    ttnn_module_args["conv3"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv3"]["resblock_2_conv2"] = ttnn_module_args["conv3"]["resblock_2_conv2"]
                conv3_weight2, conv3_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3.resblock_2_conv2, torch_model.conv3.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_2_conv2"])
                parameters["conv3"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv3_weight2,
                    conv3_bias2,
                    ttnn_module_args["conv3"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv3"]["resblock_sc_conv"] = ttnn_module_args["conv3"]["shortcut_c"]
                conv3_weight3, conv3_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3.shortcut_c, torch_model.conv3.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv3"]["resblock_sc_conv"])
                parameters["conv3"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv3_weight3,
                    conv3_bias3,
                    ttnn_module_args["conv3"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

                parameters["conv3_1"] = {}

                ttnn_module_args["conv3_1"]["resblock_1_conv1"] = ttnn_module_args["conv3_1"]["resblock_1_conv1"]
                conv3_1_weight1, conv3_1_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3_1.resblock_1_conv1, torch_model.conv3_1.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv3_1"]["resblock_1_conv1"])
                parameters["conv3_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv3_1_weight1,
                    conv3_1_bias1,
                    ttnn_module_args["conv3_1"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv3_1"]["resblock_2_conv2"] = ttnn_module_args["conv3_1"]["resblock_2_conv2"]
                conv3_1_weight2, conv3_1_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv3_1.resblock_2_conv2, torch_model.conv3_1.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv3_1"]["resblock_2_conv2"])
                parameters["conv3_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv3_1_weight2,
                    conv3_1_bias2,
                    ttnn_module_args["conv3_1"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                parameters["conv4"] = {}

                ttnn_module_args["conv4"]["resblock_1_conv1"] = ttnn_module_args["conv4"]["resblock_1_conv1"]
                conv4_weight1, conv4_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4.resblock_1_conv1, torch_model.conv4.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_1_conv1"])
                parameters["conv4"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv4_weight1,
                    conv4_bias1,
                    ttnn_module_args["conv4"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv4"]["resblock_2_conv2"] = ttnn_module_args["conv4"]["resblock_2_conv2"]
                conv4_weight2, conv4_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4.resblock_2_conv2, torch_model.conv4.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_2_conv2"])
                parameters["conv4"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv4_weight2,
                    conv4_bias2,
                    ttnn_module_args["conv4"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv4"]["resblock_sc_conv"] = ttnn_module_args["conv4"]["shortcut_c"]
                conv4_weight3, conv4_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4.shortcut_c, torch_model.conv4.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv4"]["resblock_sc_conv"])
                parameters["conv4"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv4_weight3,
                    conv4_bias3,
                    ttnn_module_args["conv4"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

                parameters["conv4_1"] = {}

                ttnn_module_args["conv4_1"]["resblock_1_conv1"] = ttnn_module_args["conv4_1"]["resblock_1_conv1"]
                conv4_1_weight1, conv4_1_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4_1.resblock_1_conv1, torch_model.conv4_1.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv4_1"]["resblock_1_conv1"])
                parameters["conv4_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv4_1_weight1,
                    conv4_1_bias1,
                    ttnn_module_args["conv4_1"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv4_1"]["resblock_2_conv2"] = ttnn_module_args["conv4_1"]["resblock_2_conv2"]
                conv4_1_weight2, conv4_1_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv4_1.resblock_2_conv2, torch_model.conv4_1.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv4_1"]["resblock_2_conv2"])
                parameters["conv4_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv4_1_weight2,
                    conv4_1_bias2,
                    ttnn_module_args["conv4_1"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                parameters["conv5"] = {}

                ttnn_module_args["conv5"]["resblock_1_conv1"] = ttnn_module_args["conv5"]["resblock_1_conv1"]
                conv5_weight1, conv5_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv5.resblock_1_conv1, torch_model.conv5.resblock_1_bn1
                )
                parameters["conv5"]["resblock_1_conv1"] = {}
                parameters["conv5"]["resblock_1_conv1"]["weight"] = conv5_weight1
                parameters["conv5"]["resblock_1_conv1"]["bias"] = conv5_bias1

                ttnn_module_args["conv5"]["resblock_2_conv2"] = ttnn_module_args["conv5"]["resblock_2_conv2"]
                conv5_weight2, conv5_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv5.resblock_2_conv2, torch_model.conv5.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv5"]["resblock_2_conv2"])
                parameters["conv5"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv5_weight2,
                    conv5_bias2,
                    ttnn_module_args["conv5"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv5"]["resblock_sc_conv"] = ttnn_module_args["conv5"]["shortcut_c"]
                conv5_weight3, conv5_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv5.shortcut_c, torch_model.conv5.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv5"]["resblock_sc_conv"])
                parameters["conv5"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv5_weight3,
                    conv5_bias3,
                    ttnn_module_args["conv5"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

                parameters["conv5_1"] = {}

                ttnn_module_args["conv5_1"]["resblock_1_conv1"] = ttnn_module_args["conv5_1"]["resblock_1_conv1"]
                conv5_1_weight1, conv5_1_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv5_1.resblock_1_conv1, torch_model.conv5_1.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv5_1"]["resblock_1_conv1"])
                parameters["conv5_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv5_1_weight1,
                    conv5_1_bias1,
                    ttnn_module_args["conv5_1"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv5_1"]["resblock_2_conv2"] = ttnn_module_args["conv5_1"]["resblock_2_conv2"]
                conv5_1_weight2, conv5_1_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv5_1.resblock_2_conv2, torch_model.conv5_1.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv5_1"]["resblock_2_conv2"])
                parameters["conv5_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv5_1_weight2,
                    conv5_1_bias2,
                    ttnn_module_args["conv5_1"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                parameters["conv6"] = {}

                ttnn_module_args["conv6"]["resblock_1_conv1"] = ttnn_module_args["conv6"]["resblock_1_conv1"]
                conv6_weight1, conv6_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv6.resblock_1_conv1, torch_model.conv6.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv6"]["resblock_1_conv1"])
                parameters["conv6"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv6_weight1,
                    conv6_bias1,
                    ttnn_module_args["conv6"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv6"]["resblock_2_conv2"] = ttnn_module_args["conv6"]["resblock_2_conv2"]
                conv6_weight2, conv6_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv6.resblock_2_conv2, torch_model.conv6.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv6"]["resblock_2_conv2"])
                parameters["conv6"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv6_weight2,
                    conv6_bias2,
                    ttnn_module_args["conv6"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv6"]["resblock_sc_conv"] = ttnn_module_args["conv6"]["shortcut_c"]
                conv6_weight3, conv6_bias3 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv6.shortcut_c, torch_model.conv6.shortcut_b
                )
                update_ttnn_module_args(ttnn_module_args["conv6"]["resblock_sc_conv"])
                parameters["conv6"]["resblock_sc_conv"], _ = preprocess_conv2d(
                    conv6_weight3,
                    conv6_bias3,
                    ttnn_module_args["conv6"]["resblock_sc_conv"],
                    return_parallel_config=True,
                )

                parameters["conv6_1"] = {}

                ttnn_module_args["conv6_1"]["resblock_1_conv1"] = ttnn_module_args["conv6_1"]["resblock_1_conv1"]
                conv6_1_weight1, conv6_1_bias1 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv6_1.resblock_1_conv1, torch_model.conv6_1.resblock_1_bn1
                )
                update_ttnn_module_args(ttnn_module_args["conv6_1"]["resblock_1_conv1"])
                parameters["conv6_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                    conv6_1_weight1,
                    conv6_1_bias1,
                    ttnn_module_args["conv6_1"]["resblock_1_conv1"],
                    return_parallel_config=True,
                )

                ttnn_module_args["conv6_1"]["resblock_2_conv2"] = ttnn_module_args["conv6_1"]["resblock_2_conv2"]
                conv6_1_weight2, conv6_1_bias2 = fold_batch_norm2d_into_conv2d(
                    torch_model.conv6_1.resblock_2_conv2, torch_model.conv6_1.resblock_2_bn2
                )
                update_ttnn_module_args(ttnn_module_args["conv6_1"]["resblock_2_conv2"])
                parameters["conv6_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                    conv6_1_weight2,
                    conv6_1_bias2,
                    ttnn_module_args["conv6_1"]["resblock_2_conv2"],
                    return_parallel_config=True,
                )

            else:
                conv2_weight, conv2_bias = torch_model.conv2.weight, torch_model.conv2.bias
                update_ttnn_module_args(ttnn_module_args.conv2)
                parameters["conv2"], conv2_parallel_config = preprocess_conv2d(
                    conv2_weight, conv2_bias, ttnn_module_args.conv2, return_parallel_config=True
                )

                conv3_weight, conv3_bias = torch_model.conv3.weight, torch_model.conv3.bias
                update_ttnn_module_args(ttnn_module_args.conv3)
                parameters["conv3"], conv2_parallel_config = preprocess_conv2d(
                    conv3_weight, conv3_bias, ttnn_module_args.conv3, return_parallel_config=True
                )

                conv3_1_weight, conv3_1_bias = torch_model.conv3_1.weight, torch_model.conv3_1.bias
                update_ttnn_module_args(ttnn_module_args.conv3_1)
                parameters["conv3_1"], conv2_parallel_config = preprocess_conv2d(
                    conv3_1_weight, conv3_1_bias, ttnn_module_args.conv3_1, return_parallel_config=True
                )

                conv4_weight, conv4_bias = torch_model.conv4.weight, torch_model.conv4.bias
                update_ttnn_module_args(ttnn_module_args.conv4)
                parameters["conv4"], conv2_parallel_config = preprocess_conv2d(
                    conv4_weight, conv4_bias, ttnn_module_args.conv4, return_parallel_config=True
                )

                conv4_1_weight, conv4_1_bias = torch_model.conv4_1.weight, torch_model.conv4_1.bias
                update_ttnn_module_args(ttnn_module_args.conv4_1)
                parameters["conv4_1"], conv2_parallel_config = preprocess_conv2d(
                    conv4_1_weight, conv4_1_bias, ttnn_module_args.conv4_1, return_parallel_config=True
                )

                conv5_weight, conv5_bias = torch_model.conv5.weight, torch_model.conv5.bias
                update_ttnn_module_args(ttnn_module_args.conv5)
                parameters["conv5"], conv2_parallel_config = preprocess_conv2d(
                    conv5_weight, conv5_bias, ttnn_module_args.conv5, return_parallel_config=True
                )

                conv5_1_weight, conv5_1_bias = torch_model.conv5_1.weight, torch_model.conv5_1.bias
                update_ttnn_module_args(ttnn_module_args.conv5_1)
                parameters["conv5_1"], conv2_parallel_config = preprocess_conv2d(
                    conv5_1_weight, conv5_1_bias, ttnn_module_args.conv5_1, return_parallel_config=True
                )

                conv6_weight, conv6_bias = torch_model.conv6.weight, torch_model.conv6.bias
                update_ttnn_module_args(ttnn_module_args.conv6)
                parameters["conv6"], conv2_parallel_config = preprocess_conv2d(
                    conv6_weight, conv6_bias, ttnn_module_args.conv6, return_parallel_config=True
                )

                conv6_1_weight, conv6_1_bias = torch_model.conv6_1.weight, torch_model.conv6_1.bias
                update_ttnn_module_args(ttnn_module_args.conv6_1)
                parameters["conv6_1"], conv2_parallel_config = preprocess_conv2d(
                    conv6_1_weight, conv6_1_bias, ttnn_module_args.conv6_1, return_parallel_config=True
                )

            ttnn_module_args["pred_res6"] = ttnn_module_args.pred_res6
            pred_res6_weight, pred_res6_bias = torch_model.pred_res6.weight, torch_model.pred_res6.bias
            ttnn_module_args["pred_res6"]["dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res6"]["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res6"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["pred_res6"]["math_fidelity"] = ttnn.MathFidelity.LoFi
            ttnn_module_args["pred_res6"]["use_1d_systolic_array"] = True
            parameters["pred_res6"], pred_res6_parallel_config = preprocess_conv2d(
                pred_res6_weight, pred_res6_bias, ttnn_module_args["pred_res6"], return_parallel_config=True
            )

            ttnn_module_args["pred_res5"] = ttnn_module_args.pred_res5
            pred_res5_weight, pred_res5_bias = torch_model.pred_res5.weight, torch_model.pred_res5.bias

            parameters["pred_res5"] = {}
            parameters["pred_res5"]["weight"] = pred_res5_weight
            parameters["pred_res5"]["bias"] = pred_res5_bias

            ttnn_module_args["pred_res4"] = ttnn_module_args.pred_res4
            pred_res4_weight, pred_res4_bias = torch_model.pred_res4.weight, torch_model.pred_res4.bias
            ttnn_module_args["pred_res4"]["use_1d_systolic_array"] = ttnn_module_args.pred_res4.in_channels <= 256
            ttnn_module_args["pred_res4"]["dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res4"]["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res4"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["pred_res4"]["math_fidelity"] = ttnn.MathFidelity.LoFi
            parameters["pred_res4"], pred_res4_parallel_config = preprocess_conv2d(
                pred_res4_weight, pred_res4_bias, ttnn_module_args["pred_res4"], return_parallel_config=True
            )

            ttnn_module_args["pred_res3"] = ttnn_module_args.pred_res3
            pred_res3_weight, pred_res3_bias = torch_model.pred_res3.weight, torch_model.pred_res3.bias
            ttnn_module_args["pred_res3"]["use_1d_systolic_array"] = ttnn_module_args.pred_res4.in_channels <= 256
            ttnn_module_args["pred_res3"]["dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res3"]["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res3"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["pred_res3"]["math_fidelity"] = ttnn.MathFidelity.LoFi
            parameters["pred_res3"], pred_res3_parallel_config = preprocess_conv2d(
                pred_res3_weight, pred_res3_bias, ttnn_module_args["pred_res3"], return_parallel_config=True
            )

            ttnn_module_args["pred_res2"] = ttnn_module_args.pred_res2
            pred_res2_weight, pred_res2_bias = torch_model.pred_res2.weight, torch_model.pred_res2.bias
            ttnn_module_args["pred_res2"]["use_1d_systolic_array"] = ttnn_module_args.pred_res4.in_channels <= 256
            ttnn_module_args["pred_res2"]["dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res2"]["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res2"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["pred_res2"]["math_fidelity"] = ttnn.MathFidelity.LoFi
            parameters["pred_res2"], pred_res2_parallel_config = preprocess_conv2d(
                pred_res2_weight, pred_res2_bias, ttnn_module_args["pred_res2"], return_parallel_config=True
            )

            ttnn_module_args["pred_res1"] = ttnn_module_args.pred_res1
            pred_res1_weight, pred_res1_bias = torch_model.pred_res1.weight, torch_model.pred_res1.bias
            ttnn_module_args["pred_res1"]["use_1d_systolic_array"] = ttnn_module_args.pred_res4.in_channels <= 256
            ttnn_module_args["pred_res1"]["dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res1"]["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args["pred_res1"]["conv_blocking_and_parallelization_config_override"] = None
            ttnn_module_args["pred_res1"]["math_fidelity"] = ttnn.MathFidelity.LoFi
            parameters["pred_res1"], pred_res1_parallel_config = preprocess_conv2d(
                pred_res1_weight, pred_res1_bias, ttnn_module_args["pred_res1"], return_parallel_config=True
            )

            ttnn_module_args["pred_res0"] = ttnn_module_args.pred_res0
            pred_res0_weight, pred_res0_bias = torch_model.pred_res0.weight, torch_model.pred_res0.bias

            parameters["pred_res0"] = {}
            parameters["pred_res0"]["weight"] = pred_res0_weight
            parameters["pred_res0"]["bias"] = pred_res0_bias

            iconv5_weight = torch_model.iconv5.weight
            iconv5_bias = torch_model.iconv5.bias
            parameters["iconv5"] = {}
            parameters["iconv5"]["weight"] = iconv5_weight
            parameters["iconv5"]["bias"] = iconv5_bias

            iconv4_weight = torch_model.iconv4.weight
            iconv4_bias = torch_model.iconv4.bias
            parameters["iconv4"] = {}
            parameters["iconv4"]["weight"] = iconv4_weight
            parameters["iconv4"]["bias"] = iconv4_bias

            iconv3_weight = torch_model.iconv3.weight
            iconv3_bias = torch_model.iconv3.bias
            parameters["iconv3"] = {}
            parameters["iconv3"]["weight"] = iconv3_weight
            parameters["iconv3"]["bias"] = iconv3_bias

            iconv2_weight = torch_model.iconv2.weight
            iconv2_bias = torch_model.iconv2.bias
            parameters["iconv2"] = {}
            parameters["iconv2"]["weight"] = iconv2_weight
            parameters["iconv2"]["bias"] = iconv2_bias

            iconv1_weight = torch_model.iconv1.weight
            iconv1_bias = torch_model.iconv1.bias
            parameters["iconv1"] = {}
            parameters["iconv1"]["weight"] = iconv1_weight
            parameters["iconv1"]["bias"] = iconv1_bias

            iconv0_weight = torch_model.iconv0.weight
            iconv0_bias = torch_model.iconv0.bias
            parameters["iconv0"] = {}
            parameters["iconv0"]["weight"] = iconv0_weight
            parameters["iconv0"]["bias"] = iconv0_bias

            upconv5_weight = torch_model.upconv5[0].weight
            parameters["upconv5"] = {}
            parameters["upconv5"]["weight"] = upconv5_weight

            upconv4_weight = torch_model.upconv4[0].weight
            parameters["upconv4"] = {}
            parameters["upconv4"]["weight"] = upconv4_weight

            upconv3_weight = torch_model.upconv3[0].weight
            parameters["upconv3"] = {}
            parameters["upconv3"]["weight"] = upconv3_weight

            upconv2_weight = torch_model.upconv2[0].weight
            parameters["upconv2"] = {}
            parameters["upconv2"]["weight"] = upconv2_weight

            upconv1_weight = torch_model.upconv1[0].weight
            parameters["upconv1"] = {}
            parameters["upconv1"]["weight"] = upconv1_weight

            upconv0_weight = torch_model.upconv0[0].weight
            parameters["upconv0"] = {}
            parameters["upconv0"]["weight"] = upconv0_weight

            upflow6to5_weight = torch_model.upflow6to5.weight
            parameters["upflow6to5"] = {}
            parameters["upflow6to5"]["weight"] = upflow6to5_weight

            upflow5to4_weight = torch_model.upflow5to4.weight
            parameters["upflow5to4"] = {}
            parameters["upflow5to4"]["weight"] = upflow5to4_weight

            upflow4to3_weight = torch_model.upflow4to3.weight
            parameters["upflow4to3"] = {}
            parameters["upflow4to3"]["weight"] = upflow4to3_weight

            upflow3to2_weight = torch_model.upflow3to2.weight
            parameters["upflow3to2"] = {}
            parameters["upflow3to2"]["weight"] = upflow3to2_weight

            upflow2to1_weight = torch_model.upflow2to1.weight
            parameters["upflow2to1"] = {}
            parameters["upflow2to1"]["weight"] = upflow2to1_weight

            upflow1to0_weight = torch_model.upflow1to0.weight
            parameters["upflow1to0"] = {}
            parameters["upflow1to0"]["weight"] = upflow1to0_weight

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dispnetres(device, reset_seeds):
    in_planes = 3 * 3 + 1 + 1
    torch_model = DispNetRes(in_planes)
    for layer in torch_model.children():
        print(layer)

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    state_dict = torch_model
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 11, 960, 576)
    pr0 = torch.randn(1, 1, 960, 576)
    pr1 = torch.randn(1, 1, 480, 288)
    pr2 = torch.randn(1, 1, 240, 144)
    pr3 = torch.randn(1, 1, 120, 72)
    pr4 = torch.randn(1, 1, 60, 36)
    pr5 = torch.randn(1, 1, 30, 18)
    pr6 = torch.randn(1, 1, 15, 9)
    dispnetc_flows = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)
    get_features = True
    torch_output_tensor = torch_model(torch_input_tensor, dispnetc_flows, get_features=get_features)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor, dispnetc_flows, get_features=get_features),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtDispNetRes(parameters, device, in_planes)
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    torch_input_tensor = torch_input_tensor.reshape(
        torch_input_tensor.shape[0],
        1,
        torch_input_tensor.shape[1] * torch_input_tensor.shape[2],
        torch_input_tensor.shape[3],
    )
    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    pr0 = torch.permute(pr0, (0, 2, 3, 1))
    pr0 = pr0.reshape(pr0.shape[0], 1, pr0.shape[1] * pr0.shape[2], pr0.shape[3])
    ttnn_pr0 = ttnn.from_torch(pr0, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    pr1 = torch.permute(pr1, (0, 2, 3, 1))
    pr1 = pr1.reshape(pr1.shape[0], 1, pr1.shape[1] * pr1.shape[2], pr1.shape[3])
    ttnn_pr1 = ttnn.from_torch(pr1, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    pr2 = torch.permute(pr2, (0, 2, 3, 1))
    pr2 = pr2.reshape(pr2.shape[0], 1, pr2.shape[1] * pr2.shape[2], pr2.shape[3])
    ttnn_pr2 = ttnn.from_torch(pr2, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    pr3 = torch.permute(pr3, (0, 2, 3, 1))
    pr3 = pr3.reshape(pr3.shape[0], 1, pr3.shape[1] * pr3.shape[2], pr3.shape[3])
    ttnn_pr3 = ttnn.from_torch(pr3, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    pr4 = torch.permute(pr4, (0, 2, 3, 1))
    pr4 = pr4.reshape(pr4.shape[0], 1, pr4.shape[1] * pr4.shape[2], pr4.shape[3])
    ttnn_pr4 = ttnn.from_torch(pr4, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    pr5 = torch.permute(pr5, (0, 2, 3, 1))
    pr5 = pr5.reshape(pr5.shape[0], 1, pr5.shape[1] * pr5.shape[2], pr5.shape[3])
    ttnn_pr5 = ttnn.from_torch(pr5, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    pr6 = torch.permute(pr6, (0, 2, 3, 1))
    pr6 = pr6.reshape(pr6.shape[0], 1, pr6.shape[1] * pr6.shape[2], pr6.shape[3])
    ttnn_pr6 = ttnn.from_torch(pr6, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    ttnn_dispnetc_flows = (ttnn_pr0, ttnn_pr1, ttnn_pr2, ttnn_pr3, ttnn_pr4, ttnn_pr5, ttnn_pr6)

    output_tensor = ttnn_model(device, input_tensor, ttnn_dispnetc_flows, get_features=get_features)

    output_tensor_0 = ttnn.to_torch(output_tensor[0])
    output_tensor_0 = output_tensor_0.reshape(1, 960, 576, 1)
    output_tensor_0 = torch.permute(output_tensor_0, (0, 3, 1, 2))

    output_tensor_1 = ttnn.to_torch(output_tensor[1])
    output_tensor_1 = output_tensor_1.reshape(1, 480, 288, 1)
    output_tensor_1 = torch.permute(output_tensor_1, (0, 3, 1, 2))

    output_tensor_2 = ttnn.to_torch(output_tensor[2])
    output_tensor_2 = output_tensor_2.reshape(1, 240, 144, 1)
    output_tensor_2 = torch.permute(output_tensor_2, (0, 3, 1, 2))

    output_tensor_3 = ttnn.to_torch(output_tensor[3])
    output_tensor_3 = output_tensor_3.reshape(1, 120, 72, 1)
    output_tensor_3 = torch.permute(output_tensor_3, (0, 3, 1, 2))

    output_tensor_4 = ttnn.to_torch(output_tensor[4])
    output_tensor_4 = output_tensor_4.reshape(1, 60, 36, 1)
    output_tensor_4 = torch.permute(output_tensor_4, (0, 3, 1, 2))

    output_tensor_5 = ttnn.to_torch(output_tensor[5])
    output_tensor_5 = output_tensor_5.reshape(1, 30, 18, 1)
    output_tensor_5 = torch.permute(output_tensor_5, (0, 3, 1, 2))

    output_tensor_6 = ttnn.to_torch(output_tensor[6])
    output_tensor_6 = output_tensor_6.reshape(1, 15, 9, 1)
    output_tensor_6 = torch.permute(output_tensor_6, (0, 3, 1, 2))

    passing, pcc_msg_0 = check_with_pcc_without_tensor_printout(torch_output_tensor[0], output_tensor_0, pcc=0.99)
    logger.info(pcc_msg_0)
    passing, pcc_msg_1 = check_with_pcc_without_tensor_printout(torch_output_tensor[1], output_tensor_1, pcc=0.99)
    logger.info(pcc_msg_1)
    passing, pcc_msg_2 = check_with_pcc_without_tensor_printout(torch_output_tensor[2], output_tensor_2, pcc=0.99)
    logger.info(pcc_msg_2)
    passing, pcc_msg_3 = check_with_pcc_without_tensor_printout(torch_output_tensor[3], output_tensor_3, pcc=0.99)
    logger.info(pcc_msg_3)
    passing, pcc_msg_4 = check_with_pcc_without_tensor_printout(torch_output_tensor[4], output_tensor_4, pcc=0.99)
    logger.info(pcc_msg_4)
    passing, pcc_msg_5 = check_with_pcc_without_tensor_printout(torch_output_tensor[5], output_tensor_5, pcc=0.99)
    logger.info(pcc_msg_5)
    passing, pcc_msg_6 = check_with_pcc_without_tensor_printout(torch_output_tensor[6], output_tensor_6, pcc=0.99)
    logger.info(pcc_msg_6)

    if get_features:
        output_tensor_7 = ttnn.to_torch(output_tensor[7])
        output_tensor_7 = output_tensor_7.reshape(1, 960, 576, 32)
        output_tensor_7 = torch.permute(output_tensor_7, (0, 3, 1, 2))
        passing, pcc_msg_7 = check_with_pcc_without_tensor_printout(torch_output_tensor[7], output_tensor_7, pcc=0.99)
        logger.info(pcc_msg_7)
