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
from models.common.utility_functions import torch_random

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader, unpack_traced_config

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

# Load traced configurations from real model tests
# Simply initialize the loader and get parameters for your operation
loader = MasterConfigLoader()
# Default: Run exact 30 traced configs from real models
model_traced_params = loader.get_suite_parameters("sigmoid_accurate")
# To run all combinations (30 shapes × dtypes × layouts × memory_configs), use:
# model_traced_params = loader.get_suite_parameters("sigmoid_accurate", all_cases=True)

parameters = {
    "nightly": {
        "input_shape": gen_shapes([1, 1, 32, 32], [6, 12, 256, 256], [1, 1, 32, 32], 16)
        + gen_shapes([1, 32, 32], [12, 256, 256], [1, 32, 32], 16)
        + gen_shapes([32, 32], [256, 256], [32, 32], 32),
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    # Traced configurations from real model tests (e.g., EfficientNet)
    # Automatically loaded - just add the suite!
    "model_traced": model_traced_params,
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape=[1, 1, 32, 32],
    input_a_dtype=ttnn.bfloat16,
    input_a_layout=ttnn.TILE_LAYOUT,
    input_a_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    traced_config_name=None,
    *,
    device,
) -> list:
    # Handle traced_config_name parameter (for model_traced suite)
    # Use the helper to unpack all config values in one line
    if traced_config_name is not None:
        input_shape, input_a_dtype, input_a_layout, input_a_memory_config, output_memory_config = unpack_traced_config(
            traced_config_name
        )

    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)
    torch_output_tensor = torch.nn.functional.sigmoid(torch_input_tensor_a)
    print(f"input_shape: {input_shape}")
    print(f"input_a_memory_config: {input_a_memory_config}")
    print(f"output_memory_config: {output_memory_config}")
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.sigmoid_accurate(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
