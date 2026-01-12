# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("scatter", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
        "dim": [3],  # Default dimension for sample
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
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Extract dim from kwargs (from traced config) or use default
    dim = kwargs.get("dim", 0)

    # Handle tuple input_shape for sample suite or dict for model_traced
    if isinstance(input_shape, dict):
        shape = input_shape.get("self", (1, 1, 32, 32))
        index_shape = input_shape.get("index", shape)
        src_shape = input_shape.get("src", shape)
    elif isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
        index_shape = shape
        src_shape = shape
    else:
        shape = input_shape
        index_shape = shape
        src_shape = shape

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Generate index tensor with values in valid range
    torch_index_tensor = torch.randint(0, shape[dim], index_shape, dtype=torch.int64)

    torch_src_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(src_shape)

    torch_output_tensor = torch.scatter(torch_input_tensor, dim, torch_index_tensor, torch_src_tensor)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor = ttnn.from_torch(torch_input_tensor, **from_torch_kwargs)

    # Index tensor must use integer dtype (uint16 or int32)
    index_from_torch_kwargs = from_torch_kwargs.copy()
    index_from_torch_kwargs["dtype"] = ttnn.int32
    index_tensor = ttnn.from_torch(torch_index_tensor, **index_from_torch_kwargs)

    src_tensor = ttnn.from_torch(torch_src_tensor, **from_torch_kwargs)

    start_time = start_measuring_time()
    output_tensor = ttnn.scatter(input_tensor, dim, index_tensor, src_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
