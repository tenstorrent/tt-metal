# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import (
    MasterConfigLoader,
    unpack_traced_config,
    unpack_binary_traced_config,
)


# Ref: https://github.com/tenstorrent/pytorch2.0_ttnn/blob/main/docs/operations/aten.mul.Scalar.md


loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("multiply")

parameters = {
    "nightly": {
        "input_specs": [
            {"shape": [1, 1, 64, 7], "other": 0.3535533905932738},
            {"shape": [1, 71, 7, 649], "other": 0.3535533905932738},
        ],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "model_traced": model_traced_params,
}


def run(
    input_specs=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    output_memory_config=None,
    traced_config_name=None,
    *,
    device,
)
) -> list:
    start_time = start_measuring_time()
    output_tensor = ttnn.multiply(input_tensor_a, input_specs["other"], memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
