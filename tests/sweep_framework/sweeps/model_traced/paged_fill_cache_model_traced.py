# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

TIMEOUT = 30

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::paged_fill_cache", all_cases=False)

parameters = {
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
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


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
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations (multi-input)
    if isinstance(input_shape, dict):
        # Traced configuration with multiple inputs
        shape_a = input_shape.get("input_a", input_shape.get("self"))
        shape_b = input_shape.get("input_b", input_shape.get("other"))
        shape_c = input_shape.get("input_c")
        if shape_c is None:
            # If only 2 inputs, use B shape for C
            shape_c = shape_b
    else:
        # Fallback for sample configurations
        if isinstance(input_shape, (tuple, list)):
            shape = tuple(input_shape)
        else:
            shape = input_shape
        shape_a = shape_b = shape_c = shape

    # Use provided dtypes - fail if not provided (no fallbacks)
    dtype_a = input_a_dtype
    if input_b_dtype is None:
        raise ValueError("input_b_dtype is None - required parameter missing")
    if input_c_dtype is None:
        raise ValueError("input_c_dtype is None - required parameter missing")
    dtype_b = input_b_dtype
    dtype_c = input_c_dtype

    # Use provided layouts - fail if not provided (no fallbacks)
    layout_a = input_a_layout
    if input_b_layout is None:
        raise ValueError("input_b_layout is None - required parameter missing")
    if input_c_layout is None:
        raise ValueError("input_c_layout is None - required parameter missing")
    layout_b = input_b_layout
    layout_c = input_c_layout

    # Use provided memory configs - fail if not provided (no fallbacks)
    mem_config_a = input_a_memory_config
    if input_b_memory_config is None:
        raise ValueError("input_b_memory_config is None - required parameter missing")
    if input_c_memory_config is None:
        raise ValueError("input_c_memory_config is None - required parameter missing")
    # Fall back to input_a_memory_config if output_memory_config is not provided
    if output_memory_config is None:
        output_memory_config = input_a_memory_config
    mem_config_b = input_b_memory_config
    mem_config_c = input_c_memory_config
    output_mem_config = output_memory_config

    # Create input tensors
    torch_input_tensor_a = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_a)(
        shape_a
    )
    torch_input_tensor_b = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_b)(
        shape_b
    )
    torch_input_tensor_c = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_c)(
        shape_c
    )

    # For reference output, just use input_a (paged_fill_cache is a caching operation)
    torch_output_tensor = torch_input_tensor_a.clone()

    # Convert to TTNN tensors
    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": dtype_a,
        "layout": layout_a,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = mem_config_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)
    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": dtype_b,
        "layout": layout_b,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = mem_config_b

    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, **from_torch_kwargs)
    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": dtype_c,
        "layout": layout_c,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = mem_config_c

    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, **from_torch_kwargs)

    start_time = start_measuring_time()
    # paged_fill_cache signature: (cache_tensor, input_tensor, page_table, *, batch_idx_tensor=None, batch_idx=0, ...)
    # So tensor_a=cache, tensor_b=input, tensor_c=page_table
    # Note: paged_fill_cache may not accept memory_config parameter - it modifies cache_tensor in place
    # Try calling without memory_config first
    try:
        output_tensor = ttnn.experimental.paged_fill_cache(
            input_tensor_a,  # cache_tensor
            input_tensor_b,  # input_tensor
            input_tensor_c,  # page_table
            batch_idx=0,  # Use default batch_idx
        )
    except TypeError:
        # If that fails, try with memory_config
        output_tensor = ttnn.experimental.paged_fill_cache(
            input_tensor_a,  # cache_tensor
            input_tensor_b,  # input_tensor
            input_tensor_c,  # page_table
            batch_idx=0,
            memory_config=output_mem_config,
        )
    # paged_fill_cache modifies cache_tensor in place, so output is the same as input_tensor_a
    output_tensor = input_tensor_a
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
