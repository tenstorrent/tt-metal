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
from tests.sweep_framework.sweeps.conv2d.conv2d_common import run, get_input_specs, mesh_device_fixture

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30


# Create test vectors for following 9 Params
# Kernels [2, 3, 5, 7]
# Input Channels [[3, 4, 16], [17, 25, 33], [40, 48, 64], [80, 96, 100], [160, 512, 640]]
# Output Channels [3, 4, 16, 40, 64, 96, 100, 150, 512, 1024]
# Transpose_mcast
# output_layout
# has_bias
# enable_act_double_buffer
# enable_split_reader
# enable_subblock_padding

# Keeping rest of the params constant as they do not affect sharding
# Batches
# Activations
# Dilation
# Activation Datatype
# Weight Datatypes
# Groups

# Total test cases = 4 * 15 * 10 * 2 * 2 * 2 * 2 * 2 * 2 = 38400
# Total 5(7680 each) suites are created since currently, only 10K test vectors can be generated per suite.
# Note that some test cases might be invalid.

input_channels = [[3, 4, 16], [17, 25, 33], [40, 48, 64], [80, 96, 100], [160, 512, 640]]

parameters = {
    f"block_sharded_suite_{idx}": {
        "input_specs": list(
            get_input_specs(
                [2],  # batch_sizes
                [32],  # activation size
                [2, 3, 5, 7],  # kernel sizes
                [1],  # stride sizes
                [1],  # padding sizes
                [1],  # dilations sizes
            )
        ),
        # Parameters-to-check starts
        "input_channels": channels,
        "output_channels": [3, 4, 16, 40, 64, 96, 100, 150, 512, 1024],
        "sharding_scheme": [ttnn.TensorMemoryLayout.BLOCK_SHARDED],
        "transpose_mcast": [True, False],
        "output_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "has_bias": [True, False],
        "enable_act_double_buffer": [True, False],
        "enable_split_reader": [True, False],
        "enable_subblock_padding": [True, False],
        # Parameters-to-check starts
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
    for idx, channels in enumerate(input_channels)
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None
