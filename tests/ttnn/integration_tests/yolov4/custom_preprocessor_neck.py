# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.experimental.functional_yolov4.reference.neck import Neck
from models.experimental.functional_yolov4.tt.ttnn_neck import TtNeck

import time
import tt_lib as ttl
import tt_lib.profiler as profiler

import ttnn
import tt_lib
from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d
import ttnn


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi


def update_ttnn_module_argsc3(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = True
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi


def custom_preprocessor(device, model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, Neck):
        ttnn_module_args.c1["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c1["use_shallow_conv_variant"] = False  # (
        ttnn_module_args.c1["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c1["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c1["deallocate_activation"] = True
        ttnn_module_args.c1["conv_blocking_and_parallelization_config_override"] = None
        # ttnn_module_args.c1["use_1d_systolic_array"] = True
        conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.c1, model.b1)
        update_ttnn_module_args(ttnn_module_args.c1)
        parameters["c1"], c1_parallel_config = preprocess_conv2d(
            conv1_weight, conv1_bias, ttnn_module_args.c1, return_parallel_config=True
        )

        ttnn_module_args.c2["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c2["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c2["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c2["activation"] = None  # Fuse relu with conv2
        ttnn_module_args.c2["deallocate_activation"] = True
        ttnn_module_args.c2["conv_blocking_and_parallelization_config_override"] = None

        conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.c2, model.b2)
        update_ttnn_module_args(ttnn_module_args.c2)
        parameters["c2"], c2_parallel_config = preprocess_conv2d(
            conv2_weight, conv2_bias, ttnn_module_args.c2, return_parallel_config=True
        )

        ttnn_module_args.c3["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c3["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c3["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c3["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c3["deallocate_activation"] = True
        ttnn_module_args.c3["conv_blocking_and_parallelization_config_override"] = None
        update_ttnn_module_argsc3(ttnn_module_args.c3)
        conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.c3, model.b3)
        update_ttnn_module_args(ttnn_module_args.c3)
        parameters["c3"], c3_parallel_config = preprocess_conv2d(
            conv3_weight, conv3_bias, ttnn_module_args.c3, return_parallel_config=True
        )

        ttnn_module_args.c4["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c4["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
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
        ttnn_module_args.c5["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
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
        ttnn_module_args.c6["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c6["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c6["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c6["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c6["deallocate_activation"] = False
        ttnn_module_args.c6["conv_blocking_and_parallelization_config_override"] = None

        conv6_weight, conv6_bias = fold_batch_norm2d_into_conv2d(model.c6, model.b6)
        update_ttnn_module_args(ttnn_module_args.c6)
        parameters["c6"], c6_parallel_config = preprocess_conv2d(
            conv6_weight, conv6_bias, ttnn_module_args.c6, return_parallel_config=True
        )

        ttnn_module_args.c7["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c7["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c7["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c7["deallocate_activation"] = False
        ttnn_module_args.c7["conv_blocking_and_parallelization_config_override"] = None

        conv7_weight, conv7_bias = fold_batch_norm2d_into_conv2d(model.c7, model.b7)
        update_ttnn_module_args(ttnn_module_args.c7)
        parameters["c7"], c7_parallel_config = preprocess_conv2d(
            conv7_weight, conv7_bias, ttnn_module_args.c7, return_parallel_config=True
        )

        ttnn_module_args.c7_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c7_2["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c7_2["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_2["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c7_2["deallocate_activation"] = True
        ttnn_module_args.c7_2["conv_blocking_and_parallelization_config_override"] = None

        conv7_2_weight, conv7_2_bias = fold_batch_norm2d_into_conv2d(model.c7_2, model.b7_2)
        update_ttnn_module_args(ttnn_module_args.c7_2)
        parameters["c7_2"], c7_2_parallel_config = preprocess_conv2d(
            conv7_2_weight, conv7_2_bias, ttnn_module_args.c7_2, return_parallel_config=True
        )

        ttnn_module_args.c7_3["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c7_3["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c7_3["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_3["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c7_3["deallocate_activation"] = True
        ttnn_module_args.c7_3["conv_blocking_and_parallelization_config_override"] = None

        conv7_3_weight, conv7_3_bias = fold_batch_norm2d_into_conv2d(model.c7_3, model.b7_3)
        update_ttnn_module_args(ttnn_module_args.c7_3)
        parameters["c7_3"], c7_3_parallel_config = preprocess_conv2d(
            conv7_3_weight, conv7_3_bias, ttnn_module_args.c7_3, return_parallel_config=True
        )

        ttnn_module_args.c7_4["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c7_4["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c7_4["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_4["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_4["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c7_4["deallocate_activation"] = True
        ttnn_module_args.c7_4["conv_blocking_and_parallelization_config_override"] = None

        conv7_4_weight, conv7_4_bias = fold_batch_norm2d_into_conv2d(model.c7_4, model.b7_4)
        update_ttnn_module_args(ttnn_module_args.c7_4)
        parameters["c7_4"], c7_4_parallel_config = preprocess_conv2d(
            conv7_4_weight, conv7_4_bias, ttnn_module_args.c7_4, return_parallel_config=True
        )

        ttnn_module_args.c7_5["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c7_5["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c7_5["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_5["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c7_5["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c7_5["deallocate_activation"] = False
        ttnn_module_args.c7_5["conv_blocking_and_parallelization_config_override"] = None

        conv7_5_weight, conv7_5_bias = fold_batch_norm2d_into_conv2d(model.c7_5, model.b7_5)
        update_ttnn_module_args(ttnn_module_args.c7_5)
        parameters["c7_5"], c7_5_parallel_config = preprocess_conv2d(
            conv7_5_weight, conv7_5_bias, ttnn_module_args.c7_5, return_parallel_config=True
        )

        ttnn_module_args.c8["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c8["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c8["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c8["deallocate_activation"] = True
        ttnn_module_args.c8["conv_blocking_and_parallelization_config_override"] = None

        conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
        update_ttnn_module_args(ttnn_module_args.c8)
        parameters["c8"], c8_parallel_config = preprocess_conv2d(
            conv8_weight, conv8_bias, ttnn_module_args.c8, return_parallel_config=True
        )

        ttnn_module_args.c8_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c8_2["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c8_2["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8_2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8_2["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c8_2["deallocate_activation"] = True
        ttnn_module_args.c8_2["conv_blocking_and_parallelization_config_override"] = None

        conv8_2_weight, conv8_2_bias = fold_batch_norm2d_into_conv2d(model.c8_2, model.b8_2)
        update_ttnn_module_args(ttnn_module_args.c8_2)
        parameters["c8_2"], c8_2_parallel_config = preprocess_conv2d(
            conv8_2_weight, conv8_2_bias, ttnn_module_args.c8_2, return_parallel_config=True
        )

        ttnn_module_args.c9["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c9["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c9["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c9["deallocate_activation"] = True
        ttnn_module_args.c9["conv_blocking_and_parallelization_config_override"] = None

        conv9_weight, conv9_bias = fold_batch_norm2d_into_conv2d(model.c9, model.b9)
        update_ttnn_module_argsc3(ttnn_module_args.c9)
        parameters["c9"], c9_parallel_config = preprocess_conv2d(
            conv9_weight, conv9_bias, ttnn_module_args.c9, return_parallel_config=True
        )

        ttnn_module_args.c9_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c9_2["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c9_2["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9_2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9_2["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c9_2["deallocate_activation"] = True
        ttnn_module_args.c9_2["conv_blocking_and_parallelization_config_override"] = None

        conv9_2_weight, conv9_2_bias = fold_batch_norm2d_into_conv2d(model.c9_2, model.b9_2)
        update_ttnn_module_args(ttnn_module_args.c9_2)
        parameters["c9_2"], c9_2_parallel_config = preprocess_conv2d(
            conv9_2_weight, conv9_2_bias, ttnn_module_args.c9_2, return_parallel_config=True
        )

        ttnn_module_args.c9_3["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c9_3["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c9_3["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9_3["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9_3["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c9_3["deallocate_activation"] = True
        ttnn_module_args.c9_3["conv_blocking_and_parallelization_config_override"] = None

        conv9_3_weight, conv9_3_bias = fold_batch_norm2d_into_conv2d(model.c9_3, model.b9_3)
        update_ttnn_module_args(ttnn_module_args.c9_3)
        parameters["c9_3"], c9_3_parallel_config = preprocess_conv2d(
            conv9_3_weight, conv9_3_bias, ttnn_module_args.c9_3, return_parallel_config=True
        )

        ttnn_module_args.c9_4["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c9_4["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c9_4["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9_4["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9_4["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c9_4["deallocate_activation"] = True
        ttnn_module_args.c9_4["conv_blocking_and_parallelization_config_override"] = None

        conv9_4_weight, conv9_4_bias = fold_batch_norm2d_into_conv2d(model.c9_4, model.b9_4)
        update_ttnn_module_args(ttnn_module_args.c9_4)
        parameters["c9_4"], c9_4_parallel_config = preprocess_conv2d(
            conv9_4_weight, conv9_4_bias, ttnn_module_args.c9_4, return_parallel_config=True
        )

        ttnn_module_args.c9_5["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c9_5["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c9_5["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9_5["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c9_5["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c9_5["deallocate_activation"] = True
        ttnn_module_args.c9_5["conv_blocking_and_parallelization_config_override"] = None

        conv9_5_weight, conv9_5_bias = fold_batch_norm2d_into_conv2d(model.c9_5, model.b9_5)
        update_ttnn_module_args(ttnn_module_args.c9_5)
        parameters["c9_5"], c9_5_parallel_config = preprocess_conv2d(
            conv9_5_weight, conv9_5_bias, ttnn_module_args.c9_5, return_parallel_config=True
        )

        ttnn_module_args.c10["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c10["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c10["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c10["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c10["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c10["deallocate_activation"] = False
        ttnn_module_args.c10["conv_blocking_and_parallelization_config_override"] = None

        conv10_weight, conv10_bias = fold_batch_norm2d_into_conv2d(model.c10, model.b10)
        update_ttnn_module_args(ttnn_module_args.c10)
        parameters["c10"], c10_parallel_config = preprocess_conv2d(
            conv10_weight, conv10_bias, ttnn_module_args.c10, return_parallel_config=True
        )

        ttnn_module_args.c10_2["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c10_2["use_shallow_conv_variant"] = False  # (
        #                False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        #            )
        ttnn_module_args.c10_2["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c10_2["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c10_2["activation"] = None  # Fuse relu with conv1
        ttnn_module_args.c10_2["deallocate_activation"] = True
        ttnn_module_args.c10_2["conv_blocking_and_parallelization_config_override"] = None

        conv10_weight, conv10_bias = fold_batch_norm2d_into_conv2d(model.c10_2, model.b10_2)
        update_ttnn_module_args(ttnn_module_args.c10_2)
        parameters["c10_2"], c10_2_parallel_config = preprocess_conv2d(
            conv10_weight, conv10_bias, ttnn_module_args.c10_2, return_parallel_config=True
        )

    return parameters
