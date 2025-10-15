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

# Import master config utilities for real-world configurations
from tests.sweep_framework.sweep_config_utils import load_unary_op_configs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

# Load master configurations for real-world test cases
# This returns 30 exact (shape, memory_config) pairs from traced models
master_traced_config = load_unary_op_configs("sigmoid_accurate")

# Create a lookup dictionary for config specs (following matmul sharded pattern)
# This avoids serialization issues when passing complex objects through sweep framework
_CONFIG_LOOKUP = {}
_config_spec_names = []
if "traced_config" in master_traced_config:
    for idx, (shape, mem_config) in enumerate(master_traced_config["traced_config"]):
        config_name = f"traced_{idx}"

        # Use traced configurations as-is with SHARDED memory
        # All configs use TILE layout and bfloat8_b dtype
        layout = ttnn.TILE_LAYOUT
        dtype = ttnn.bfloat8_b

        # Use SHARDED memory from traced config for both input and output
        # Unary operations require input and output memory layouts to match
        output_mem_config = mem_config

        _CONFIG_LOOKUP[config_name] = {
            "shape": shape,
            "memory_config": mem_config,
            "output_memory_config": output_mem_config,
            "layout": layout,
            "dtype": dtype,
        }
        _config_spec_names.append(config_name)


def get_traced_config(config_name):
    """Helper function to retrieve config from lookup by name"""
    if config_name not in _CONFIG_LOOKUP:
        # Fallback default
        return {
            "shape": [1, 32, 32],
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "layout": ttnn.TILE_LAYOUT,
            "dtype": ttnn.bfloat16,
        }
    return _CONFIG_LOOKUP[config_name]


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
    # New suite using real-world configurations from traced models
    # Uses 30 exact (shape, memory_config) pairs - no Cartesian product!
    # Pass config names (strings) instead of tuples to avoid serialization issues
    # Each config includes: shape, input/output memory_config, layout, dtype
    "model_traced": {
        "traced_config_name": _config_spec_names if _config_spec_names else ["default"],
    },
}


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    output_memory_config=None,
    traced_config_name=None,  # New parameter for model_traced suite: config spec name
    *,
    device,
) -> list:
    # Handle traced_config_name from model_traced suite
    # Following matmul sharded pattern: use config name to lookup actual config
    if traced_config_name is not None:
        config = get_traced_config(traced_config_name)
        input_shape = config["shape"]
        input_a_memory_config = config["memory_config"]
        output_memory_config = config.get("output_memory_config", ttnn.DRAM_MEMORY_CONFIG)
        # Use layout and dtype from config (auto-selected for compatibility)
        input_a_layout = config.get("layout", ttnn.TILE_LAYOUT)
        input_a_dtype = config.get("dtype", ttnn.bfloat16)

    # Validate required parameters
    if input_shape is None:
        raise ValueError("input_shape must be provided either directly or via traced_config_name")
    if input_a_dtype is None:
        input_a_dtype = ttnn.bfloat16
    if input_a_layout is None:
        input_a_layout = ttnn.TILE_LAYOUT
    if input_a_memory_config is None:
        input_a_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if output_memory_config is None:
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG

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
