# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt, gen_constant

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import (
    MasterConfigLoader,
    unpack_traced_config,
    unpack_binary_traced_config,
)


# Ref: https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/docs/operations/aten.mul.Tensor.md

# Test to check for cases with inputs that give inf in pytorch and not in TTNN because 'inf' threshold is different for Torch and TTNN. Hence, this is tested separately.
# pytorch gives 'inf' for values beyond ±3.4e38 but in TTNN, we get inf when the value exceeds ±3.41e38

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("multiply")

parameters = {
    "check_inf_cases": {
        "input_shape": [
            {"self": [1, 1, 1, 10], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 12], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 14], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 15], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 17], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 1], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 201], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 2048], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 256], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 25], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 2], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 5], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 6], "other": -3.4028234663852886e38},
            {"self": [1, 1, 1, 7], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 8], "other": -3.3895313892515355e38},
            {"self": [1, 1, 1, 9], "other": -3.4028234663852886e38},
        ],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "model_traced": model_traced_params,
}


def run(
    input_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    traced_config_name=None,
    *,
    device,
)
) -> list:
    # if isinstance(input_shape["other"], list):
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )
    # else:
    #     input_tensor_b = input_shape["other"]

    start_time = start_measuring_time()
    result = ttnn.mul(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, pcc=0.99), e2e_perf]
