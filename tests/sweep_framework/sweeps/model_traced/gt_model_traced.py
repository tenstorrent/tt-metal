# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
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
model_traced_params = loader.get_suite_parameters("gt", all_cases=False)

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
        "scalar": [0.0],  # Default scalar value for tensor-scalar comparison
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
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    scalar=None,  # Scalar value from traced configs
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Determine if this is tensor-tensor or tensor-scalar comparison
    # If input_b_dtype is provided, it's tensor-tensor comparison
    is_binary = input_b_dtype is not None

    # Handle input shapes
    # For unary operations (tensor-scalar from traced configs), shape is a list/tuple
    # For binary operations (tensor-tensor), shape might be a dict with "self" and "other"
    if isinstance(input_shape, dict) and "self" in input_shape:
        # Binary operation with dict format
        shape_a = input_shape["self"]
        shape_b = input_shape.get("other", shape_a) if is_binary else None
    elif isinstance(input_shape, (tuple, list)):
        # Unary operation or sample suite - tuple/list format
        shape_a = tuple(input_shape) if isinstance(input_shape, list) else input_shape
        shape_b = tuple(input_shape) if is_binary else None
    else:
        shape_a = input_shape
        shape_b = input_shape if is_binary else None

    # Generate input tensor A
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments for tensor A
    from_torch_kwargs_a = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }
    if not is_host:
        from_torch_kwargs_a["device"] = device
        from_torch_kwargs_a["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs_a)

    if is_binary:
        # Tensor-tensor comparison
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)

        # PyTorch reference using golden function
        golden_function = ttnn.get_golden_function(ttnn.gt)
        torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

        # Build from_torch arguments for tensor B
        from_torch_kwargs_b = {
            "dtype": input_b_dtype,
            "layout": input_b_layout,
        }
        if not is_host:
            from_torch_kwargs_b["device"] = device
            from_torch_kwargs_b["memory_config"] = input_b_memory_config

        input_tensor_b = ttnn.from_torch(torch_input_tensor_b, **from_torch_kwargs_b)

        start_time = start_measuring_time()
        output_tensor = ttnn.gt(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)
    else:
        # Tensor-scalar comparison - use scalar from traced config or default to 0
        scalar_value = scalar if scalar is not None else 0

        # PyTorch reference using golden function
        golden_function = ttnn.get_golden_function(ttnn.gt)
        torch_output_tensor = golden_function(torch_input_tensor_a, scalar_value)

        start_time = start_measuring_time()
        output_tensor = ttnn.gt(input_tensor_a, scalar_value, memory_config=output_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)

    # For comparison operations, convert to float for PCC check
    pcc = check_with_pcc(torch_output_tensor.float(), output_tensor.float(), 0.999)

    return [pcc, e2e_perf]
