# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import itertools
import random
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.conv2d_common import run_full, get_input_specs, mesh_device_fixture

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30


# Create test vectors for following 6 Params
# Batches [1, 2, 3, 4, 5, 7, 8, 16, 20, 23, 25, 27, 29, 32]
# Activations [4, 7, 16, 23, 32, 56, 60, 66, 115, 1056]
# Kernels [1, 2, 3, 4, 5, 6, 7]
# Strides [1, 2, 3, 4, 5]
# Padding [0, 1, 2, 3, 4]
# Dilation [1, 2]

# Keeping rest of the params constant as they do not affect sliding window calculations.
# Input Channels [32]
# Output Channels [32]
# Activation Datatype [ttnn.bfloat16]
# Weight Datatypes [ttnn.bfloat16]
# Groups [1]
# Transpose_mcast[True]
# use_1d_systolic_array[True]
# output_layout[ttnn.TILE_LAYOUT]

# Total test cases = 13 * 10 * 7 * 5 * 5 * 2 * 1 * 1 * 1 * 1 * 1 * 1 = 45500
# Total 7(7000*6 + 3500*1) suites are created since currently, only 10K test vectors can be generated per suite.
# Note that some test cases might be invalid.

Batches = [[1, 2], [3, 4], [5, 7], [8, 16], [20, 23], [25, 27], [32]]

parameters = {
    f"sliding_window_suite_{idx}": {
        # Parameters-to-check starts
        "input_specs": list(
            get_input_specs(
                batches,  # batch_sizes
                [4, 7, 11, 16, 23, 56, 60, 66, 115, 1056],  # activation size
                [x for x in range(1, 8, 1)],  # kernel sizes
                [x for x in range(1, 6, 1)],  # stride sizes
                [x for x in range(0, 5, 1)],  # padding sizes
                [1, 2],  # dilations sizes
            )
        ),
        # Parameters-to-check ends
        "input_channels": [32],
        "output_channels": [32],
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
        "groups": [1],
        "override_sharding_config": [False],
        "core_grid": [None],
        "use_shallow_conv_variant": [False],
        "deallocate_activation": [False],
        "enable_auto_formatting": [False],
        "padded_input_channels": [None],
    }
    for idx, batches in enumerate(Batches)
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
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
