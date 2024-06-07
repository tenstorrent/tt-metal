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
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi


def custom_preprocessor(device, model, name, ttnn_module_args):
    # print("We do reach here!")
    parameters = {}
    if isinstance(model, Head):
        ttnn_module_args.c1["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c1["use_shallow_conv_variant"] = (
            False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        )
        ttnn_module_args.c1["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c1["deallocate_activation"] = False
        ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = None
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
        update_ttnn_module_args(ttnn_module_args.c1)
        parameters["c1"], c1_parallel_config = preprocess_conv2d(
            conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
        )

        conv2_weight = model.c2.weight.detach()
        conv2_bias = model.c2.bias
        parameters["c2"] = {}
        parameters["c2"]["weight"] = conv2_weight
        parameters["c2"]["bias"] = conv2_bias

        ttnn_module_args.c3["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c3["use_shallow_conv_variant"] = (
            False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        )
        ttnn_module_args.c3["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c3["activation"] = None  # Fuse relu with conv1
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
        ttnn_module_args.c4["activation"] = None  # Fuse relu with conv1
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
        ttnn_module_args.c5["activation"] = None  # Fuse relu with conv1
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
        ttnn_module_args.c6["activation"] = None  # Fuse relu with conv1
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
        ttnn_module_args.c7["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c7["deallocate_activation"] = True
        ttnn_module_args.c7["conv_blocking_and_parallelization_config_override"] = None

        conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
        update_ttnn_module_args(ttnn_module_args.c7)
        parameters["c7"], c7_parallel_config = preprocess_conv2d(
            conv7_weight, conv7_bias, ttnn_module_args.c7, return_parallel_config=True
        )

        ttnn_module_args.c8["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c8["use_shallow_conv_variant"] = (
            False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        )
        ttnn_module_args.c8["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c8["deallocate_activation"] = False
        ttnn_module_args.c8["conv_blocking_and_parallelization_config_override"] = None

        conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
        update_ttnn_module_args(ttnn_module_args.c8)
        parameters["c8"], c8_parallel_config = preprocess_conv2d(
            conv8_weight, conv8_bias, ttnn_module_args.c8, return_parallel_config=True
        )

        ttnn_module_args.c9["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c9["use_shallow_conv_variant"] = (
            False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        )
        ttnn_module_args.c9["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c9["deallocate_activation"] = False
        ttnn_module_args.c9["conv_blocking_and_parallelization_config_override"] = None

        conv9_weight, conv9_bias = fold_batch_norm2d_into_conv2d(model.c9, model.b9)
        update_ttnn_module_args(ttnn_module_args.c9)
        ttnn_module_args.c9["use_1d_systolic_array"] = False
        parameters["c9"], c9_parallel_config = preprocess_conv2d(
            conv9_weight, conv9_bias, ttnn_module_args.c9, return_parallel_config=True
        )

        conv10_weight = model.c10.weight
        conv10_bias = model.c10.bias
        # conv10_bias = None
        parameters["c10"] = {}
        parameters["c10"]["weight"] = conv10_weight
        parameters["c10"]["bias"] = conv10_bias

        ttnn_module_args.c11["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c11["use_shallow_conv_variant"] = (
            False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        )
        ttnn_module_args.c11["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c11["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c11["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c11["deallocate_activation"] = False
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
        ttnn_module_args.c12["activation"] = None  # Fuse relu with conv1
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
        ttnn_module_args.c13["activation"] = None  # Fuse relu with conv1
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
        ttnn_module_args.c14["activation"] = None  # Fuse relu with conv1
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
        ttnn_module_args.c15["activation"] = None  # Fuse relu with conv1
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
        ttnn_module_args.c16["activation"] = None  # Fuse relu with conv1
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

        conv18_weight = model.c18.weight
        conv18_bias = model.c18.bias
        parameters["c18"] = {}
        parameters["c18"]["weight"] = conv18_weight
        parameters["c18"]["bias"] = conv18_bias

    return parameters
