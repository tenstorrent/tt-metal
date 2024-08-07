# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d
from models.experimental.functional_fadnetpp.reference.dispnetc import DispNetC
import ttnn
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels <= 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["conv_blocking_and_parallelization_config_override"] = None
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi


def custom_preprocessor(device, model, name, ttnn_module_args, resblock=True):
    parameters = {}
    if isinstance(model, DispNetC):
        conv1b_weight, conv1b_bias = model.extractnet.conv1b.weight, model.extractnet.conv1b.bias
        parameters["extractnet"] = {}
        parameters["extractnet"]["conv1b"] = {}
        parameters["extractnet"]["conv1b"]["weight"] = conv1b_weight
        parameters["extractnet"]["conv1b"]["bias"] = conv1b_bias

        conv1a_weight, conv1a_bias = model.extractnet.conv1a.weight, model.extractnet.conv1a.bias
        parameters["extractnet"]["conv1a"] = {}
        parameters["extractnet"]["conv1a"]["weight"] = conv1a_weight
        parameters["extractnet"]["conv1a"]["bias"] = conv1a_bias

        if resblock:
            parameters["extractnet"]["conv2"] = {}
            ttnn_module_args["extractnet"]["conv2"]["resblock_1_conv1"] = ttnn_module_args["extractnet"]["conv2"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["extractnet"]["conv2"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.extractnet.conv2.resblock_1_conv1, model.extractnet.conv2.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["extractnet"]["conv2"]["resblock_1_conv1"])
            parameters["extractnet"]["conv2"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["extractnet"]["conv2"]["resblock_1_conv1"],
                return_parallel_config=True,
            )

            ttnn_module_args["extractnet"]["conv2"]["resblock_2_conv2"] = ttnn_module_args["extractnet"]["conv2"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["extractnet"]["conv2"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight2, bias2 = fold_batch_norm2d_into_conv2d(
                model.extractnet.conv2.resblock_2_conv2, model.extractnet.conv2.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["extractnet"]["conv2"]["resblock_2_conv2"])
            parameters["extractnet"]["conv2"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight2,
                bias2,
                ttnn_module_args["extractnet"]["conv2"]["resblock_2_conv2"],
                return_parallel_config=True,
            )

            ttnn_module_args["extractnet"]["conv2"]["resblock_sc_conv"] = ttnn_module_args["extractnet"]["conv2"][
                "shortcut_c"
            ]
            ttnn_module_args["extractnet"]["conv2"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
            weight3, bias3 = fold_batch_norm2d_into_conv2d(
                model.extractnet.conv2.shortcut_c, model.extractnet.conv2.shortcut_b
            )
            update_ttnn_module_args(ttnn_module_args["extractnet"]["conv2"]["resblock_sc_conv"])
            parameters["extractnet"]["conv2"]["resblock_sc_conv"], _ = preprocess_conv2d(
                weight3,
                bias3,
                ttnn_module_args["extractnet"]["conv2"]["resblock_sc_conv"],
                return_parallel_config=True,
            )

            parameters["extractnet"]["conv3"] = {}
            ttnn_module_args["extractnet"]["conv3"]["resblock_1_conv1"] = ttnn_module_args["extractnet"]["conv3"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["extractnet"]["conv3"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.extractnet.conv3.resblock_1_conv1, model.extractnet.conv3.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["extractnet"]["conv3"]["resblock_1_conv1"])
            parameters["extractnet"]["conv3"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["extractnet"]["conv3"]["resblock_1_conv1"],
                return_parallel_config=True,
            )

            ttnn_module_args["extractnet"]["conv3"]["resblock_2_conv2"] = ttnn_module_args["extractnet"]["conv3"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["extractnet"]["conv3"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight2, bias2 = fold_batch_norm2d_into_conv2d(
                model.extractnet.conv3.resblock_2_conv2, model.extractnet.conv3.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["extractnet"]["conv3"]["resblock_2_conv2"])
            parameters["extractnet"]["conv3"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight2,
                bias2,
                ttnn_module_args["extractnet"]["conv3"]["resblock_2_conv2"],
                return_parallel_config=True,
            )

            ttnn_module_args["extractnet"]["conv3"]["resblock_sc_conv"] = ttnn_module_args["extractnet"]["conv3"][
                "shortcut_c"
            ]
            ttnn_module_args["extractnet"]["conv3"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
            weight3, bias3 = fold_batch_norm2d_into_conv2d(
                model.extractnet.conv3.shortcut_c, model.extractnet.conv3.shortcut_b
            )
            update_ttnn_module_args(ttnn_module_args["extractnet"]["conv3"]["resblock_sc_conv"])
            parameters["extractnet"]["conv3"]["resblock_sc_conv"], _ = preprocess_conv2d(
                weight3,
                bias3,
                ttnn_module_args["extractnet"]["conv3"]["resblock_sc_conv"],
                return_parallel_config=True,
            )

        else:
            ttnn_module_args.extractnet.conv2["weights_dtype"] = ttnn.bfloat8_b
            conv2_weight, conv2_bias = model.conv2.weight, model.conv2.bias
            update_ttnn_module_args(ttnn_module_args.extractnet.conv2)
            parameters["extractnet"]["conv2"], conv2_parallel_config = preprocess_conv2d(
                conv2_weight, conv2_bias, ttnn_module_args.extractnet.conv2, return_parallel_config=True
            )

            ttnn_module_args.extractnet.conv3["weights_dtype"] = ttnn.bfloat8_b
            conv3_weight, conv3_bias = model.conv3.weight, model.conv3.bias
            update_ttnn_module_args(ttnn_module_args.extractnet.conv3)
            parameters["extractnet"]["conv3"], conv3_parallel_config = preprocess_conv2d(
                conv3_weight, conv3_bias, ttnn_module_args.extractnet.conv3, return_parallel_config=True
            )
        parameters["cunet"] = {}
        if resblock:
            parameters["cunet"]["conv_redir"] = {}
            ttnn_module_args["cunet"]["conv_redir"]["resblock_1_conv1"] = ttnn_module_args["cunet"]["conv_redir"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["cunet"]["conv_redir"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv_redir.resblock_1_conv1, model.cunet.conv_redir.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv_redir"]["resblock_1_conv1"])

            parameters["cunet"]["conv_redir"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv_redir"]["resblock_1_conv1"],
                return_parallel_config=True,
            )

            ttnn_module_args["cunet"]["conv_redir"]["resblock_2_conv2"] = ttnn_module_args["cunet"]["conv_redir"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["cunet"]["conv_redir"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv_redir.resblock_2_conv2, model.cunet.conv_redir.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv_redir"]["resblock_2_conv2"])
            parameters["cunet"]["conv_redir"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv_redir"]["resblock_2_conv2"],
                return_parallel_config=True,
            )

            ttnn_module_args["cunet"]["conv_redir"]["resblock_sc_conv"] = ttnn_module_args["cunet"]["conv_redir"][
                "shortcut_c"
            ]
            ttnn_module_args["cunet"]["conv_redir"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv_redir.shortcut_c, model.cunet.conv_redir.shortcut_b
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv_redir"]["resblock_sc_conv"])
            parameters["cunet"]["conv_redir"]["resblock_sc_conv"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv_redir"]["resblock_sc_conv"],
                return_parallel_config=True,
            )

            ttnn_module_args.cunet.conv_dy["weights_dtype"] = ttnn.bfloat8_b
            parameters["cunet"]["conv_dy"] = {}
            parameters["cunet"]["conv_dy"]["weight"] = model.cunet.conv_dy.weight
            update_ttnn_module_args(ttnn_module_args.cunet.conv_dy)

            ttnn_module_args.cunet.conv_d2["weights_dtype"] = ttnn.bfloat8_b
            conv_d2_weight, conv_d2_bias = fold_batch_norm2d_into_conv2d(model.cunet.conv_d2, model.cunet.bn_d2)
            update_ttnn_module_args(ttnn_module_args.cunet.conv_d2)
            parameters["cunet"]["conv_d2"], conv_d2_parallel_config = preprocess_conv2d(
                conv_d2_weight, conv_d2_bias, ttnn_module_args.cunet.conv_d2, return_parallel_config=True
            )

            ttnn_module_args.cunet.conv_dy_1["weights_dtype"] = ttnn.bfloat8_b
            parameters["cunet"]["conv_dy_1"] = {}
            parameters["cunet"]["conv_dy_1"]["weight"] = model.cunet.conv_dy_1.weight
            update_ttnn_module_args(ttnn_module_args.cunet.conv_dy_1)

            parameters["cunet"]["conv4"] = {}
            ttnn_module_args["cunet"]["conv4"]["resblock_1_conv1"] = ttnn_module_args["cunet"]["conv4"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["cunet"]["conv4"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv4.resblock_1_conv1, model.cunet.conv4.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv4"]["resblock_1_conv1"])
            parameters["cunet"]["conv4"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv4"]["resblock_1_conv1"], return_parallel_config=True
            )

            ttnn_module_args["cunet"]["conv4"]["resblock_2_conv2"] = ttnn_module_args["cunet"]["conv4"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["cunet"]["conv4"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv4.resblock_2_conv2, model.cunet.conv4.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv4"]["resblock_2_conv2"])
            parameters["cunet"]["conv4"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv4"]["resblock_2_conv2"], return_parallel_config=True
            )

            ttnn_module_args["cunet"]["conv4"]["resblock_sc_conv"] = ttnn_module_args["cunet"]["conv4"]["shortcut_c"]
            ttnn_module_args["cunet"]["conv4"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(model.cunet.conv4.shortcut_c, model.cunet.conv4.shortcut_b)
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv4"]["resblock_sc_conv"])
            parameters["cunet"]["conv4"]["resblock_sc_conv"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv4"]["resblock_sc_conv"], return_parallel_config=True
            )

            parameters["cunet"]["conv4_1"] = {}
            ttnn_module_args["cunet"]["conv4_1"]["resblock_1_conv1"] = ttnn_module_args["cunet"]["conv4_1"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["cunet"]["conv4_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv4_1.resblock_1_conv1, model.cunet.conv4_1.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv4_1"]["resblock_1_conv1"])
            parameters["cunet"]["conv4_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv4_1"]["resblock_1_conv1"],
                return_parallel_config=True,
            )

            ttnn_module_args["cunet"]["conv4_1"]["resblock_2_conv2"] = ttnn_module_args["cunet"]["conv4_1"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["cunet"]["conv4_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv4_1.resblock_2_conv2, model.cunet.conv4_1.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv4_1"]["resblock_2_conv2"])
            parameters["cunet"]["conv4_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv4_1"]["resblock_2_conv2"],
                return_parallel_config=True,
            )

            parameters["cunet"]["conv5"] = {}
            ttnn_module_args["cunet"]["conv5"]["resblock_1_conv1"] = ttnn_module_args["cunet"]["conv5"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["cunet"]["conv5"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv5.resblock_1_conv1, model.cunet.conv5.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv5"]["resblock_1_conv1"])
            parameters["cunet"]["conv5"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv5"]["resblock_1_conv1"], return_parallel_config=True
            )

            ttnn_module_args["cunet"]["conv5"]["resblock_2_conv2"] = ttnn_module_args["cunet"]["conv5"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["cunet"]["conv5"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv5.resblock_2_conv2, model.cunet.conv5.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv5"]["resblock_2_conv2"])
            parameters["cunet"]["conv5"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv5"]["resblock_2_conv2"], return_parallel_config=True
            )

            ttnn_module_args["cunet"]["conv5"]["resblock_sc_conv"] = ttnn_module_args["cunet"]["conv5"]["shortcut_c"]
            ttnn_module_args["cunet"]["conv5"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(model.cunet.conv5.shortcut_c, model.cunet.conv5.shortcut_b)
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv5"]["resblock_sc_conv"])
            parameters["cunet"]["conv5"]["resblock_sc_conv"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv5"]["resblock_sc_conv"], return_parallel_config=True
            )

            parameters["cunet"]["conv5_1"] = {}
            ttnn_module_args["cunet"]["conv5_1"]["resblock_1_conv1"] = ttnn_module_args["cunet"]["conv5_1"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["cunet"]["conv5_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv5_1.resblock_1_conv1, model.cunet.conv5_1.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv5_1"]["resblock_1_conv1"])
            parameters["cunet"]["conv5_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv5_1"]["resblock_1_conv1"],
                return_parallel_config=True,
            )

            ttnn_module_args["cunet"]["conv5_1"]["resblock_2_conv2"] = ttnn_module_args["cunet"]["conv5_1"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["cunet"]["conv5_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv5_1.resblock_2_conv2, model.cunet.conv5_1.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv5_1"]["resblock_2_conv2"])
            parameters["cunet"]["conv5_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv5_1"]["resblock_2_conv2"],
                return_parallel_config=True,
            )

            parameters["cunet"]["conv6"] = {}
            ttnn_module_args["cunet"]["conv6"]["resblock_1_conv1"] = ttnn_module_args["cunet"]["conv6"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["cunet"]["conv6"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv6.resblock_1_conv1, model.cunet.conv6.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv6"]["resblock_1_conv1"])
            parameters["cunet"]["conv6"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv6"]["resblock_1_conv1"], return_parallel_config=True
            )

            ttnn_module_args["cunet"]["conv6"]["resblock_2_conv2"] = ttnn_module_args["cunet"]["conv6"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["cunet"]["conv6"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv6.resblock_2_conv2, model.cunet.conv6.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv6"]["resblock_2_conv2"])
            parameters["cunet"]["conv6"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv6"]["resblock_2_conv2"], return_parallel_config=True
            )

            ttnn_module_args["cunet"]["conv6"]["resblock_sc_conv"] = ttnn_module_args["cunet"]["conv6"]["shortcut_c"]
            ttnn_module_args["cunet"]["conv6"]["resblock_sc_conv"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(model.cunet.conv6.shortcut_c, model.cunet.conv6.shortcut_b)
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv6"]["resblock_sc_conv"])
            parameters["cunet"]["conv6"]["resblock_sc_conv"], _ = preprocess_conv2d(
                weight1, bias1, ttnn_module_args["cunet"]["conv6"]["resblock_sc_conv"], return_parallel_config=True
            )

            parameters["cunet"]["conv6_1"] = {}
            ttnn_module_args["cunet"]["conv6_1"]["resblock_1_conv1"] = ttnn_module_args["cunet"]["conv6_1"][
                "resblock_1_conv1"
            ]
            ttnn_module_args["cunet"]["conv6_1"]["resblock_1_conv1"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv6_1.resblock_1_conv1, model.cunet.conv6_1.resblock_1_bn1
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv6_1"]["resblock_1_conv1"])
            parameters["cunet"]["conv6_1"]["resblock_1_conv1"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv6_1"]["resblock_1_conv1"],
                return_parallel_config=True,
            )

            ttnn_module_args["cunet"]["conv6_1"]["resblock_2_conv2"] = ttnn_module_args["cunet"]["conv6_1"][
                "resblock_2_conv2"
            ]
            ttnn_module_args["cunet"]["conv6_1"]["resblock_2_conv2"]["weights_dtype"] = ttnn.bfloat8_b
            weight1, bias1 = fold_batch_norm2d_into_conv2d(
                model.cunet.conv6_1.resblock_2_conv2, model.cunet.conv6_1.resblock_2_bn2
            )
            update_ttnn_module_args(ttnn_module_args["cunet"]["conv6_1"]["resblock_2_conv2"])
            parameters["cunet"]["conv6_1"]["resblock_2_conv2"], _ = preprocess_conv2d(
                weight1,
                bias1,
                ttnn_module_args["cunet"]["conv6_1"]["resblock_2_conv2"],
                return_parallel_config=True,
            )

        else:
            ttnn_module_args.cunet.conv_redir["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.cunet.conv_redir["activation"] = "relu"
            conv_redir_weight, conv_redir_bias = model.cunet.conv_redir, model.cunet.bn_redir
            update_ttnn_module_args(ttnn_module_args.cunet.conv_redir)
            parameters["cunet"]["conv_redir"], conv_redir_parallel_config = preprocess_conv2d(
                conv_redir_weight, conv_redir_bias, ttnn_module_args.cunet.conv_redir, return_parallel_config=True
            )

            ttnn_module_args.cunet.conv3_1["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.cunet.conv3_1["activation"] = "relu"
            conv3_1_weight, conv3_1_bias = model.cunet.conv3_1, model.cunet.bn_3_1
            update_ttnn_module_args(ttnn_module_args.cunet.conv3_1)
            parameters["cunet"]["conv3_1"], conv3_1_parallel_config = preprocess_conv2d(
                conv3_1_weight, conv3_1_bias, ttnn_module_args.cunet.conv3_1, return_parallel_config=True
            )

            ttnn_module_args.cunet.conv_4["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.cunet.conv_4["activation"] = "relu"
            conv_4_weight, conv_4_bias = model.cunet.conv_4, model.cunet.bn_4
            update_ttnn_module_args(ttnn_module_args.cunet.conv_4)
            parameters["cunet"]["conv_4"], conv_4_parallel_config = preprocess_conv2d(
                conv_4_weight, conv_4_bias, ttnn_module_args.cunet.conv_4, return_parallel_config=True
            )

            ttnn_module_args.cunet.conv_4_1["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.cunet.conv_4_1["activation"] = "relu"
            conv_4_1_weight, conv_4_1_bias = model.cunet.conv_4_1, model.cunet.bn_4_1
            update_ttnn_module_args(ttnn_module_args.cunet.conv_4_1)
            parameters["cunet"]["conv_4_1"], conv_4_1_parallel_config = preprocess_conv2d(
                conv_4_1_weight, conv_4_1_bias, ttnn_module_args.cunet.conv_4_1, return_parallel_config=True
            )

            ttnn_module_args.cunet.conv_5["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.cunet.conv_5["activation"] = "relu"
            conv_5_weight, conv_5_bias = model.cunet.conv_5, model.cunet.bn_5
            update_ttnn_module_args(ttnn_module_args.cunet.conv_5)
            parameters["cunet"]["conv_5"], conv_5_parallel_config = preprocess_conv2d(
                conv_5_weight, conv_5_bias, ttnn_module_args.cunet.conv_5, return_parallel_config=True
            )

            ttnn_module_args.cunet.conv_5_1["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.cunet.conv_5_1["activation"] = "relu"
            conv_5_1_weight, conv_5_1_bias = model.cunet.conv_5_1, model.cunet.bn_5_1
            update_ttnn_module_args(ttnn_module_args.cunet.conv_5_1)
            parameters["cunet"]["conv_5_1"], conv_5_1_parallel_config = preprocess_conv2d(
                conv_5_1_weight, conv_5_1_bias, ttnn_module_args.cunet.conv_5_1, return_parallel_config=True
            )

            ttnn_module_args.cunet.conv_6["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.cunet.conv_6["activation"] = "relu"
            conv_6_weight, conv_6_bias = model.cunet.conv_6, model.cunet.bn_6
            update_ttnn_module_args(ttnn_module_args.cunet.conv_6)
            parameters["cunet"]["conv_6"], conv_6_parallel_config = preprocess_conv2d(
                conv_6_weight, conv_6_bias, ttnn_module_args.cunet.conv_6, return_parallel_config=True
            )

            ttnn_module_args.cunet.conv_6_1["weights_dtype"] = ttnn.bfloat8_b
            ttnn_module_args.cunet.conv_6_1["activation"] = "relu"
            conv_6_1_weight, conv_6_1_bias = model.cunet.conv_6_1, model.cunet.bn_6_1
            update_ttnn_module_args(ttnn_module_args.cunet.conv_6_1)
            parameters["cunet"]["conv_6_1"], conv_6_1_parallel_config = preprocess_conv2d(
                conv_6_1_weight, conv_6_1_bias, ttnn_module_args.cunet.conv_6_1, return_parallel_config=True
            )

        ttnn_module_args.cunet.pred_flow6["weights_dtype"] = ttnn.bfloat8_b
        parameters["cunet"]["pred_flow6"] = {}
        pred_flow6_weight = model.cunet.pred_flow6.weight
        update_ttnn_module_args(ttnn_module_args.cunet.pred_flow6)
        ttnn_module_args.cunet.pred_flow6["use_1d_systolic_array"] = True
        parameters["cunet"]["pred_flow6"], pred_flow6_parallel_config = preprocess_conv2d(
            pred_flow6_weight, None, ttnn_module_args.cunet.pred_flow6, return_parallel_config=True
        )

        ttnn_module_args.cunet.pred_flow5["weights_dtype"] = ttnn.bfloat8_b
        parameters["cunet"]["pred_flow5"] = {}
        pred_flow5_weight = model.cunet.pred_flow5.weight
        update_ttnn_module_args(ttnn_module_args.cunet.pred_flow5)
        ttnn_module_args.cunet.pred_flow5["use_1d_systolic_array"] = True
        parameters["cunet"]["pred_flow5"], pred_flow5_parallel_config = preprocess_conv2d(
            pred_flow5_weight, None, ttnn_module_args.cunet.pred_flow5, return_parallel_config=True
        )

        ttnn_module_args.cunet.pred_flow4["weights_dtype"] = ttnn.bfloat8_b
        parameters["cunet"]["pred_flow4"] = {}
        pred_flow4_weight = model.cunet.pred_flow4.weight
        update_ttnn_module_args(ttnn_module_args.cunet.pred_flow4)
        parameters["cunet"]["pred_flow4"], pred_flow4_parallel_config = preprocess_conv2d(
            pred_flow4_weight, None, ttnn_module_args.cunet.pred_flow4, return_parallel_config=True
        )

        ttnn_module_args.cunet.pred_flow3["weights_dtype"] = ttnn.bfloat8_b
        parameters["cunet"]["pred_flow3"] = {}
        pred_flow3_weight = model.cunet.pred_flow3.weight
        update_ttnn_module_args(ttnn_module_args.cunet.pred_flow3)
        parameters["cunet"]["pred_flow3"], pred_flow3_parallel_config = preprocess_conv2d(
            pred_flow3_weight, None, ttnn_module_args.cunet.pred_flow3, return_parallel_config=True
        )

        ttnn_module_args.cunet.pred_flow2["weights_dtype"] = ttnn.bfloat8_b
        parameters["cunet"]["pred_flow2"] = {}
        pred_flow2_weight = model.cunet.pred_flow2.weight
        update_ttnn_module_args(ttnn_module_args.cunet.pred_flow2)
        parameters["cunet"]["pred_flow2"], pred_flow2_parallel_config = preprocess_conv2d(
            pred_flow2_weight, None, ttnn_module_args.cunet.pred_flow2, return_parallel_config=True
        )

        ttnn_module_args.cunet.pred_flow1["weights_dtype"] = ttnn.bfloat8_b
        parameters["cunet"]["pred_flow1"] = {}
        pred_flow1_weight = model.cunet.pred_flow1.weight
        update_ttnn_module_args(ttnn_module_args.cunet.pred_flow1)
        ttnn_module_args.cunet.pred_flow0["use_1d_systolic_array"] = True
        parameters["cunet"]["pred_flow1"], pred_flow1_parallel_config = preprocess_conv2d(
            pred_flow1_weight, None, ttnn_module_args.cunet.pred_flow1, return_parallel_config=True
        )

        parameters["cunet"]["pred_flow0"] = {}
        parameters["cunet"]["pred_flow0"]["weight"] = model.cunet.pred_flow0.weight

        upconv5_weight = model.cunet.upconv5.weight
        parameters["cunet"]["upconv5"] = {}
        parameters["cunet"]["upconv5"]["weight"] = upconv5_weight

        upconv4_weight = model.cunet.upconv4.weight
        parameters["cunet"]["upconv4"] = {}
        parameters["cunet"]["upconv4"]["weight"] = upconv4_weight

        upconv3_weight = model.cunet.upconv3.weight
        parameters["cunet"]["upconv3"] = {}
        parameters["cunet"]["upconv3"]["weight"] = upconv3_weight

        upconv2_weight = model.cunet.upconv2.weight
        parameters["cunet"]["upconv2"] = {}
        parameters["cunet"]["upconv2"]["weight"] = upconv2_weight

        upconv1_weight = model.cunet.upconv1.weight
        parameters["cunet"]["upconv1"] = {}
        parameters["cunet"]["upconv1"]["weight"] = upconv1_weight

        upconv0_weight = model.cunet.upconv0.weight
        parameters["cunet"]["upconv0"] = {}
        parameters["cunet"]["upconv0"]["weight"] = upconv0_weight

        iconv5_weight = model.cunet.iconv5.weight
        iconv5_bias = model.cunet.iconv5.bias
        parameters["cunet"]["iconv5"] = {}
        parameters["cunet"]["iconv5"]["weight"] = iconv5_weight
        parameters["cunet"]["iconv5"]["bias"] = iconv5_bias

        iconv4_weight = model.cunet.iconv4.weight
        iconv4_bias = model.cunet.iconv4.bias
        parameters["cunet"]["iconv4"] = {}
        parameters["cunet"]["iconv4"]["weight"] = iconv4_weight
        parameters["cunet"]["iconv4"]["bias"] = iconv4_bias

        iconv3_weight = model.cunet.iconv3.weight
        iconv3_bias = model.cunet.iconv3.bias
        parameters["cunet"]["iconv3"] = {}
        parameters["cunet"]["iconv3"]["weight"] = iconv3_weight
        parameters["cunet"]["iconv3"]["bias"] = iconv3_bias

        iconv2_weight = model.cunet.iconv2.weight
        iconv2_bias = model.cunet.iconv2.bias
        parameters["cunet"]["iconv2"] = {}
        parameters["cunet"]["iconv2"]["weight"] = iconv2_weight
        parameters["cunet"]["iconv2"]["bias"] = iconv2_bias

        iconv1_weight = model.cunet.iconv1.weight
        iconv1_bias = model.cunet.iconv1.bias
        parameters["cunet"]["iconv1"] = {}
        parameters["cunet"]["iconv1"]["weight"] = iconv1_weight
        parameters["cunet"]["iconv1"]["bias"] = iconv1_bias

        iconv0_weight = model.cunet.iconv0.weight
        iconv0_bias = model.cunet.iconv0.bias
        parameters["cunet"]["iconv0"] = {}
        parameters["cunet"]["iconv0"]["weight"] = iconv0_weight
        parameters["cunet"]["iconv0"]["bias"] = iconv0_bias

        upflow6to5_weight = model.cunet.upflow6to5.weight
        parameters["cunet"]["upflow6to5"] = {}
        parameters["cunet"]["upflow6to5"]["weight"] = upflow6to5_weight

        upflow5to4_weight = model.cunet.upflow5to4.weight
        parameters["cunet"]["upflow5to4"] = {}
        parameters["cunet"]["upflow5to4"]["weight"] = upflow5to4_weight

        upflow4to3_weight = model.cunet.upflow4to3.weight
        parameters["cunet"]["upflow4to3"] = {}
        parameters["cunet"]["upflow4to3"]["weight"] = upflow4to3_weight

        upflow3to2_weight = model.cunet.upflow3to2.weight
        parameters["cunet"]["upflow3to2"] = {}
        parameters["cunet"]["upflow3to2"]["weight"] = upflow3to2_weight

        upflow2to1_weight = model.cunet.upflow2to1.weight
        parameters["cunet"]["upflow2to1"] = {}
        parameters["cunet"]["upflow2to1"]["weight"] = upflow2to1_weight

        upflow1to0_weight = model.cunet.upflow1to0.weight
        parameters["cunet"]["upflow1to0"] = {}
        parameters["cunet"]["upflow1to0"]["weight"] = upflow1to0_weight
    return parameters
