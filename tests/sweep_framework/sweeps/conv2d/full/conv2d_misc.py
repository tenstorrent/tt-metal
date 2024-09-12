# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import os
import itertools
import random
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.conv2d_common import run_full, get_input_specs, mesh_device_fixture

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30


# compute_kernel_config_suite: Create test vectors for following 5 Params,
# activations_dtype
# weights_dtype
# math_fidelity
# fp32_accum
# packer_l1_acc

# Keeping rest of the params constant
# batches
# activations
# kernels
# strides
# padding
# dilation
# input Channels
# output Channels
# transpose_mcast
# output_layout
# has_bias
# enable_act_double_buffer
# enable_split_reader
# enable_subblock_padding
# groups
# override_sharding_config
# core_grid
# use_shallow_conv_variant
# deallocate_activation
# enable_auto_formatting
# padded_input_channels


# Total test cases = 2 * 2 * 4 * 2 * 2 * 4(kernels) * 3 (input_channels) * 3 (output_channels) = 2304
# Note that some test cases might be invalid.

# conv_config_suite: Create test vectors for following 5 Params,
# groups
# override_sharding_config
# core_grid
# use_shallow_conv_variant
# deallocate_activation

# Keeping rest of the params constant
# batches
# activations
# kernels
# strides
# padding
# dilation
# input Channels
# output Channels
# transpose_mcast
# output_layout
# has_bias
# enable_act_double_buffer
# enable_split_reader
# enable_subblock_padding
# enable_auto_formatting
# activations_dtype
# weights_dtype
# math_fidelity
# fp32_accum
# packer_l1_acc
# padded_input_channels


# Total test cases = 2 * 3 * 7 * 8 * 3 * 9 * 1 * 1 = 9072
# Note that some test cases might be invalid.

# large_sizes_suite: Checks for large sizes for convolutions
# Total test cases = 2 * 2 * 2 * 4 * 3 * 3 * 2 * 2 * 2 * 2 * 2 = 9216
# Note that some test cases might be invalid.

parameters = {
    "compute_kernel_config_suite": {
        "input_specs": list(
            get_input_specs(
                [2],  # batch_sizes
                [32],  # activation size
                [2, 5, 7, 9],  # kernel sizes
                [1],  # stride sizes
                [1],  # padding sizes
                [1],  # dilations sizes
            )
        ),
        "input_channels": [3, 1024, 2560],
        "output_channels": [3, 1024, 2560],
        "transpose_mcast": [True],
        "output_layout": [ttnn.TILE_LAYOUT],
        "has_bias": [True],
        "enable_act_double_buffer": [False],
        "enable_split_reader": [False],
        "enable_subblock_padding": [False],
        # Parameters-to-check starts
        "activations_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "weights_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "math_fidelity": [
            ttnn.MathFidelity.LoFi,
            ttnn.MathFidelity.HiFi2,
            ttnn.MathFidelity.HiFi3,
            ttnn.MathFidelity.HiFi4,
        ],
        "fp32_accum": [True, False],
        "packer_l1_acc": [True, False],
        # Parameters-to-check ends
        "groups": [1],
        "override_sharding_config": [False],
        "core_grid": [None],
        "use_shallow_conv_variant": [False],
        "deallocate_activation": [False],
        "enable_auto_formatting": [False],
        "padded_input_channels": [None],
    },
    "conv_config_suite": {
        "input_specs": list(
            get_input_specs(
                [2, 32],  # batch_sizes
                [32],  # activation size
                [2, 3, 5],  # kernel sizes
                [1],  # stride sizes
                [1],  # padding sizes
                [1],  # dilations sizes
            )
        ),
        "input_channels": [3, 4, 12, 96, 128, 512, 1024],
        "output_channels": [3, 4, 12, 42, 96, 128, 256, 1024],
        "transpose_mcast": [True],
        "output_layout": [ttnn.TILE_LAYOUT],
        "has_bias": [True],
        "enable_act_double_buffer": [False],
        "enable_split_reader": [False],
        "enable_subblock_padding": [False],
        "activations_dtype": [ttnn.bfloat16],
        "weights_dtype": [ttnn.bfloat16],
        "math_fidelity": [ttnn.MathFidelity.HiFi4],
        "fp32_accum": [False],
        "packer_l1_acc": [False],
        # Parameters-to-check starts
        "groups": [2, 3, 8],
        "override_sharding_config": [True],
        "core_grid": [
            ((5, 5), (6, 6)),  # square 4 cores
            ((0, 0), (6, 6)),  # suqare 49 cores
            ((0, 0), (0, 6)),  # rectangle 7 cores
            ((0, 0), (6, 0)),  # rectangle 7 cores
            ((1, 1), (5, 5)),  # rectangle 25 cores
            ((4, 4), (6, 6), (0, 3), (1, 4)),  # Uneven shape, 13 cores
            ((1, 1), (4, 4), (0, 0), (0, 3)),  # Uneven shape, 20 cores
            ((0, 0), (4, 4), (0, 6), (1, 6)),  # Uneven shape, 27 cores
            ((0, 0), (5, 5), (0, 6), (2, 6)),  # Uneven shape, 39 cores
        ],
        "use_shallow_conv_variant": [True],
        "deallocate_activation": [True],
        # Parameters-to-check ends
        "enable_auto_formatting": [False],
        "padded_input_channels": [None],
    },
    "large_sizes_suite": {
        "input_specs": list(
            get_input_specs(
                [32],  # batch_sizes
                [528, 1056],  # activation size
                [6, 7],  # kernel sizes
                [1],  # stride sizes
                [1],  # padding sizes
                [1, 2],  # dilations sizes
            )
        ),
        "input_channels": [1280, 1920, 2560, 3200],
        "output_channels": [1280, 1920, 2560],
        "transpose_mcast": [True, False],
        "output_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "has_bias": [True],
        "enable_act_double_buffer": [True, False],
        "enable_split_reader": [True, False],
        "enable_subblock_padding": [True, False],
        "activations_dtype": [ttnn.bfloat16],
        "weights_dtype": [ttnn.bfloat16],
        "math_fidelity": [ttnn.MathFidelity.HiFi4],
        "fp32_accum": [False],
        "packer_l1_acc": [False],
        "groups": [1],
        "override_sharding_config": [False],
        "core_grid": [None],  # ignored
        "use_shallow_conv_variant": [False],
        "deallocate_activation": [False],
        "enable_auto_formatting": [False],
        "padded_input_channels": [None],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if (test_vector["input_channels"] % test_vector["groups"] != 0) or (
        test_vector["output_channels"] % test_vector["groups"] != 0
    ):
        return True, "input_channels and/or output_channels are not divisible by groups"
    return False, None


def run(
    input_specs,
    input_channels,
    output_channels,
    transpose_mcast,
    output_layout,
    has_bias,
    enable_act_double_buffer,
    enable_split_reader,
    enable_subblock_padding,
    activations_dtype,
    weights_dtype,
    math_fidelity,
    fp32_accum,
    packer_l1_acc,
    groups,
    override_sharding_config,
    core_grid,
    use_shallow_conv_variant,
    deallocate_activation,
    enable_auto_formatting,
    padded_input_channels=None,
    *,
    device,
) -> list:
    return run_full(
        input_specs,
        input_channels,
        output_channels,
        transpose_mcast,
        output_layout,
        has_bias,
        enable_act_double_buffer,
        enable_split_reader,
        enable_subblock_padding,
        activations_dtype,
        weights_dtype,
        math_fidelity,
        fp32_accum,
        packer_l1_acc,
        groups,
        override_sharding_config,
        core_grid,
        use_shallow_conv_variant,
        deallocate_activation,
        enable_auto_formatting,
        device,
        padded_input_channels,
    )
