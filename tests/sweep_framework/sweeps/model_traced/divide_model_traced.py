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
model_traced_params = loader.get_suite_parameters("divide", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def _ensure_tuple(shape):
    """Convert shape to tuple, handling various input formats"""
    if shape is None:
        return None
    if isinstance(shape, tuple):
        return shape
    if isinstance(shape, list):
        return tuple(shape)
    if isinstance(shape, str):
        # Handle string representations like "(1, 1, 8, 8)"
        import ast

        try:
            parsed = ast.literal_eval(shape)
            return tuple(parsed) if isinstance(parsed, (list, tuple)) else parsed
        except (ValueError, SyntaxError):
            # If parsing fails, return as-is (might be a non-shape string)
            return shape
    if isinstance(shape, dict):
        # If it's still a dict at this point, something is wrong
        raise ValueError(f"Shape should not be a dict at this point: {shape}")
    # Handle any other iterable
    if hasattr(shape, "__iter__"):
        return tuple(shape)
    # If it's a single value or something unexpected, return as-is
    return shape


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle both sample suite (tuple) and model_traced suite (dict)
    if isinstance(input_shape, dict) and "self" in input_shape:
        # This is model_traced suite - dict with 'self' and 'other' keys
        shape_a = _ensure_tuple(input_shape["self"])
        shape_b_raw = input_shape.get("other")
        shape_b = _ensure_tuple(shape_b_raw)

        # Validate that 'other' exists for divide operations (always needs two tensors)
        if shape_b is None:
            raise ValueError("Divide operation requires two tensors - 'other' key missing from input_shape")
    else:
        # This is sample suite - use same shape for both inputs
        shape_a = _ensure_tuple(input_shape)
        shape_b = _ensure_tuple(input_shape)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Avoid division by zero - generate denominator with values away from zero
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=1, high=100, dtype=torch.float32), input_b_dtype or input_a_dtype
    )(shape_b)

    torch_output_tensor = torch.divide(torch_input_tensor_a, torch_input_tensor_b)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs_a = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }
    from_torch_kwargs_b = {
        "dtype": input_b_dtype or input_a_dtype,
        "layout": input_b_layout or input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs_a["device"] = device
        from_torch_kwargs_a["memory_config"] = input_a_memory_config
        from_torch_kwargs_b["device"] = device
        from_torch_kwargs_b["memory_config"] = input_b_memory_config or input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, **from_torch_kwargs_b)

    start_time = start_measuring_time()
    output_tensor = ttnn.divide(
        input_tensor_a, input_tensor_b, memory_config=output_memory_config or ttnn.DRAM_MEMORY_CONFIG
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
