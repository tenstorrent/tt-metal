# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep test for ttnn.experimental.paged_fused_update_cache operation.

This operation updates the KV cache with paged memory support and fused operations
for efficient transformer attention in decode mode.
"""

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
model_traced_params = loader.get_suite_parameters("experimental::paged_fused_update_cache", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_d_dtype": [ttnn.bfloat16],
        "input_d_layout": [ttnn.TILE_LAYOUT],
        "input_d_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Validate that batch dimensions are compatible.
    The operation requires: input_tensor.padded_shape()[1] == cache_tensor.padded_shape()[0]
    """
    input_shape = test_vector.get("input_shape")

    if isinstance(input_shape, dict):
        shape_a = input_shape.get("input_a", input_shape.get("self"))
        shape_b = input_shape.get("input_b", input_shape.get("cache"))

        # Validate batch dimension compatibility
        # input_tensor shape: [1, batch, seq_len, head_dim]
        # cache_tensor shape: [batch, cache_len, n_heads, head_dim]
        if shape_a and shape_b and isinstance(shape_a, (list, tuple)) and isinstance(shape_b, (list, tuple)):
            if len(shape_a) >= 2 and len(shape_b) >= 1:
                input_batch = shape_a[1]  # Second dimension of input
                cache_batch = shape_b[0]  # First dimension of cache

                if input_batch != cache_batch:
                    return True, f"Batch mismatch: input[1]={input_batch} != cache[0]={cache_batch}"

    return False, None


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    input_d_dtype=None,
    input_d_layout=None,
    input_d_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations (multi-input)
    if isinstance(input_shape, dict):
        shape_a = input_shape.get("input_a", input_shape.get("self"))
        shape_b = input_shape.get("input_b", input_shape.get("cache"))
        shape_c = input_shape.get("input_c", input_shape.get("update_idxs"))
        shape_d = input_shape.get("input_d", input_shape.get("page_table"))
    else:
        # Fallback for sample configurations
        if isinstance(input_shape, (tuple, list)):
            shape = tuple(input_shape)
        else:
            shape = input_shape
        shape_a = shape  # New values to cache
        shape_b = (1, 32, shape[2], shape[3])  # Cache tensor
        shape_c = (1, shape[1])  # Update indices
        shape_d = (1, shape[1])  # Page table

    # Check which inputs are provided
    has_input_b = input_b_dtype is not None
    has_input_c = input_c_dtype is not None
    has_input_d = input_d_dtype is not None

    # Generate input tensors
    torch_input_a = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(
        shape_a
    )

    if has_input_b:
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype
        )(shape_b)
    else:
        torch_input_b = None

    if has_input_c:
        torch_input_c = gen_func_with_cast_tt(
            partial(torch_random, low=0, high=32, dtype=torch.float32), input_c_dtype
        )(shape_c)
    else:
        torch_input_c = None

    if has_input_d:
        torch_input_d = gen_func_with_cast_tt(
            partial(torch_random, low=0, high=32, dtype=torch.float32), input_d_dtype
        )(shape_d)
    else:
        torch_input_d = None

    # Simplified torch reference (actual paged update is complex)
    torch_output = torch_input_a  # Simplified reference

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Convert to ttnn tensors
    from_torch_kwargs_a = {"dtype": input_a_dtype, "layout": input_a_layout}
    if not is_host:
        from_torch_kwargs_a["device"] = device
        from_torch_kwargs_a["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_a, **from_torch_kwargs_a)

    input_tensors = [input_tensor_a]

    if has_input_b and torch_input_b is not None:
        from_torch_kwargs_b = {"dtype": input_b_dtype, "layout": input_b_layout}
        if not is_host:
            from_torch_kwargs_b["device"] = device
            from_torch_kwargs_b["memory_config"] = input_b_memory_config
        input_tensor_b = ttnn.from_torch(torch_input_b, **from_torch_kwargs_b)
        input_tensors.append(input_tensor_b)

    if has_input_c and torch_input_c is not None:
        from_torch_kwargs_c = {"dtype": input_c_dtype, "layout": input_c_layout}
        if not is_host:
            from_torch_kwargs_c["device"] = device
            from_torch_kwargs_c["memory_config"] = input_c_memory_config
        input_tensor_c = ttnn.from_torch(torch_input_c, **from_torch_kwargs_c)
        input_tensors.append(input_tensor_c)

    if has_input_d and torch_input_d is not None:
        from_torch_kwargs_d = {"dtype": input_d_dtype, "layout": input_d_layout}
        if not is_host:
            from_torch_kwargs_d["device"] = device
            from_torch_kwargs_d["memory_config"] = input_d_memory_config
        input_tensor_d = ttnn.from_torch(torch_input_d, **from_torch_kwargs_d)
        input_tensors.append(input_tensor_d)

    start_time = start_measuring_time()

    try:
        # paged_fused_update_cache doesn't accept memory_config parameter
        # It only accepts specific keyword arguments like update_idxs, page_table, etc.
        result = ttnn.experimental.paged_fused_update_cache(*input_tensors)
        # Handle both single tensor and tuple returns
        if isinstance(result, (list, tuple)):
            output_tensor = ttnn.to_torch(result[0]) if result else None
        else:
            output_tensor = ttnn.to_torch(result)

        e2e_perf = stop_measuring_time(start_time)

        # check_with_pcc returns (bool, message) tuple
        if output_tensor is not None:
            pcc = check_with_pcc(torch_output, output_tensor, 0.999)
        else:
            pcc = (False, "Output tensor is None")
    except Exception as e:
        # Operation may not be fully implemented yet
        print(f"Operation failed: {e}")
        e2e_perf = stop_measuring_time(start_time)
        pcc = (False, f"Operation failed: {str(e)}")

    return [pcc, e2e_perf]
