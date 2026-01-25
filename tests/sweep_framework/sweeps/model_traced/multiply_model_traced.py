# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from typing import Optional, Tuple

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("multiply", all_cases=False)

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


def mesh_device_fixture():
    """
    Override default device fixture for multiply operation.
    Using explicit DispatchCoreConfig to handle sharded memory configs.
    """
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_device(device)
    del device


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Invalidate test vectors with incompatible configurations.
    Mixed integer and float dtypes cause numerical errors in multiply operation.
    """
    input_a_dtype = test_vector.get("input_a_dtype")
    input_b_dtype = test_vector.get("input_b_dtype")

    # Define integer and float dtypes
    integer_dtypes = [ttnn.int32, ttnn.uint32, ttnn.uint16]
    float_dtypes = [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32]

    # Check if one is integer and other is float
    a_is_int = input_a_dtype in integer_dtypes
    b_is_int = input_b_dtype in integer_dtypes
    a_is_float = input_a_dtype in float_dtypes
    b_is_float = input_b_dtype in float_dtypes

    # Invalidate if mixed integer and float dtypes
    if (a_is_int and b_is_float) or (a_is_float and b_is_int):
        return True, "Mixed integer and float dtypes are not supported for multiply operation"

    return False, None


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
    scalar=None,  # For tensor-scalar operations
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Handle both sample suite (tuple) and model_traced suite (dict)
    if isinstance(input_shape, dict) and "self" in input_shape:
        # This is model_traced suite - dict with 'self' and 'other' keys
        shape_a = _ensure_tuple(input_shape["self"])
        shape_b = _ensure_tuple(input_shape.get("other"))  # May be None for scalar operations
    else:
        # This is sample suite - use same shape for both inputs
        shape_a = _ensure_tuple(input_shape)
        shape_b = _ensure_tuple(input_shape)

    # Validate that either shape_b or scalar is provided (not both None)
    if shape_b is None and scalar is None:
        raise ValueError("Either shape_b or scalar must be provided for multiply operation")

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Handle tensor-scalar operations
    if shape_b is None and scalar is not None:
        # Tensor-scalar operation: use scalar value directly
        torch_output_tensor = torch.mul(torch_input_tensor_a, scalar)
    else:
        # Tensor-tensor operation: create second tensor
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype or input_a_dtype
        )(shape_b)
        torch_output_tensor = torch.mul(torch_input_tensor_a, torch_input_tensor_b)

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

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    # Handle tensor-scalar vs tensor-tensor operations
    if shape_b is None and scalar is not None:
        # Tensor-scalar operation: pass scalar directly to ttnn.multiply
        start_time = start_measuring_time()
        output_tensor = ttnn.multiply(
            input_tensor_a, scalar, memory_config=output_memory_config or ttnn.DRAM_MEMORY_CONFIG
        )
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)
    else:
        # Tensor-tensor operation: create second tensor
        # Reuse is_host variable already computed above (no need to recompute)

        # Build from_torch arguments based on storage_type
        from_torch_kwargs = {
            "dtype": input_b_dtype or input_a_dtype,
            "layout": input_b_layout or input_a_layout,
        }

        # Only add device and memory_config if not HOST storage
        if not is_host:
            from_torch_kwargs["device"] = device
            from_torch_kwargs["memory_config"] = input_b_memory_config or input_a_memory_config

        input_tensor_b = ttnn.from_torch(torch_input_tensor_b, **from_torch_kwargs)

        start_time = start_measuring_time()
        output_tensor = ttnn.multiply(
            input_tensor_a, input_tensor_b, memory_config=output_memory_config or ttnn.DRAM_MEMORY_CONFIG
        )
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
