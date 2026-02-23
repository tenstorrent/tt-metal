# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 60

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("div", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    scalar=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, dict):
        shape = (
            tuple(input_shape["self"])
            if isinstance(input_shape.get("self"), (list, tuple))
            else input_shape.get("self", (1, 1, 32, 32))
        )
        shape_b = input_shape.get("other")
        if shape_b is not None:
            shape_b = tuple(shape_b) if isinstance(shape_b, (list, tuple)) else shape_b
    elif isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
        shape_b = None
    else:
        shape = (1, 1, 32, 32)
        shape_b = None

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )

    if scalar is not None:
        divisor = scalar
    elif shape_b is not None:
        divisor = gen_func_with_cast_tt(
            partial(torch_random, low=1, high=100, dtype=torch.float32), input_b_dtype or input_a_dtype
        )(shape_b)
    else:
        divisor = 2.0

    torch_output = ttnn.get_golden_function(ttnn.div)(torch_input, divisor)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=input_a_dtype, layout=input_a_layout, device=device, memory_config=input_a_memory_config
    )

    if isinstance(divisor, torch.Tensor):
        divisor_tensor = ttnn.from_torch(
            divisor,
            dtype=input_b_dtype or input_a_dtype,
            layout=input_b_layout or input_a_layout,
            device=device,
            memory_config=input_b_memory_config or input_a_memory_config,
        )
    else:
        divisor_tensor = divisor

    start_time = start_measuring_time()
    output_tensor = ttnn.div(input_tensor, divisor_tensor, memory_config=output_memory_config or input_a_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
