# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.reduction_common import run_sum

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 32, 32], [6, 12, 256, 256], [1, 1, 32, 32], 16)
        + gen_shapes([1, 32, 32], [12, 256, 256], [1, 32, 32], 2)
        + gen_shapes([32, 32], [256, 256], [32, 32], 2)
        + gen_shapes([1, 1, 1, 1], [6, 12, 256, 256], [1, 1, 1, 1], 2)
        + gen_shapes([1, 1, 1], [12, 256, 256], [1, 1, 1], 2)
        + gen_shapes([1, 1], [256, 256], [1, 1], 2)
        + gen_shapes([1, 1, 1, 1], [6, 12, 100, 100], [1, 1, 1, 1], 2)
        + gen_shapes([1, 1, 1], [12, 128, 128], [1, 1, 1], 7)
        + gen_shapes([1, 1], [11, 11], [1, 1], 2)
        + gen_shapes([1], [256], [16], 2),
        "dim": [
            0,
            1,
            2,
            3,
            None,
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
        ],
        "keepdim": [True, False],
        "input_a_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT and not (
        test_vector["input_a_dtype"] == ttnn.float32 or test_vector["input_a_dtype"] == ttnn.bfloat16
    ):
        return True, "Row major is only supported for fp32 & fp16"
    if not test_vector["keepdim"]:
        return True, "keepdim = false is not supported"

    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    dim,
    keepdim,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    return run_sum(
        input_shape, dim, keepdim, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config, device
    )


import pytest


@pytest.mark.parametrize(
    "input_shape, dim, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config",
    [
        ([7, 32, 4, 96], 3, True, ttnn.float32, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_reduction_sum_localrun_fail_only(
    device, input_shape, dim, keepdim, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config
):
    run_sum(
        input_shape, dim, keepdim, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config, device
    )
