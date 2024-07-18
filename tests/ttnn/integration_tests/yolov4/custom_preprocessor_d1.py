# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from ttnn.model_preprocessing import preprocess_conv2d, fold_batch_norm2d_into_conv2d

from models.experimental.functional_yolov4.reference.downsample1 import DownSample1
from models.experimental.functional_yolov4.tt.ttnn_downsample1 import TtDownSample1

import ttnn
import tt_lib


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256


def custom_preprocessor(device, model, name, ttnn_module_args):
    parameters = {}
    if isinstance(model, DownSample1):
        ttnn_module_args.c1["math_fidelity"] = ttnn.MathFidelity.LoFi
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

        ttnn_module_args.c8["math_fidelity"] = ttnn.MathFidelity.LoFi
        ttnn_module_args.c8["use_shallow_conv_variant"] = (
            False if device.arch() == tt_lib.device.Arch.WORMHOLE_B0 else True
        )
        ttnn_module_args.c8["dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8["weights_dtype"] = ttnn.bfloat8_b
        ttnn_module_args.c8["activation"] = "relu"  # Fuse relu with conv1
        ttnn_module_args.c8["deallocate_activation"] = True
        ttnn_module_args.c8["conv_blocking_and_parallelization_config_override"] = None

        conv8_weight, conv8_bias = fold_batch_norm2d_into_conv2d(model.c8, model.b8)
        update_ttnn_module_args(ttnn_module_args.c8)
        parameters["c8"], c8_parallel_config = preprocess_conv2d(
            conv8_weight, conv8_bias, ttnn_module_args.c8, return_parallel_config=True
        )

    return parameters
