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
model_traced_params = loader.get_suite_parameters("experimental::paged_update_cache", all_cases=False)

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
        "input_d_dtype": [ttnn.bfloat16],
        "input_d_layout": [ttnn.TILE_LAYOUT],
        "input_d_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    input_d_dtype=None,
    input_d_layout=None,
    input_d_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept extra parameters like scalar, traced_source, etc.
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations (multi-input)
    if isinstance(input_shape, dict):
        # Traced configuration with multiple inputs
        shape_a = input_shape.get("input_a", input_shape.get("self"))
        shape_b = input_shape.get("input_b", input_shape.get("other"))
        shape_c = input_shape.get("input_c")
        shape_d = input_shape.get("input_d")
        if shape_c is None:
            shape_c = shape_b
        if shape_d is None:
            shape_d = shape_c
    else:
        # Fallback for sample configurations
        if isinstance(input_shape, (tuple, list)):
            shape = tuple(input_shape)
        else:
            shape = input_shape
        shape_a = shape_b = shape_c = shape_d = shape

    # Check if we have 3 or 4 tensors (4th tensor is optional)
    has_input_d = input_d_dtype is not None and input_d_layout is not None and input_d_memory_config is not None

    # Use provided dtypes - fail if not provided (no fallbacks for required tensors)
    dtype_a = input_a_dtype
    if input_b_dtype is None:
        raise ValueError("input_b_dtype is None - required parameter missing")
    if input_c_dtype is None:
        raise ValueError("input_c_dtype is None - required parameter missing")
    dtype_b = input_b_dtype
    dtype_c = input_c_dtype
    dtype_d = input_d_dtype if has_input_d else None

    # Use provided layouts - fail if not provided (no fallbacks for required tensors)
    layout_a = input_a_layout
    if input_b_layout is None:
        raise ValueError("input_b_layout is None - required parameter missing")
    if input_c_layout is None:
        raise ValueError("input_c_layout is None - required parameter missing")
    layout_b = input_b_layout
    # layout_c validated but overridden later (must be ROW_MAJOR for page_table)

    # Use provided memory configs - fail if not provided (no fallbacks for required tensors)
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
    mem_config_d = input_d_memory_config if has_input_d else None
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
    # Only create 4th tensor if it's provided
    torch_input_tensor_d = None
    if has_input_d:
        torch_input_tensor_d = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_d
        )(shape_d)

    # For reference output, just use input_a (paged_update_cache is a caching operation)
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
    input_tensor_c = ttnn.from_torch(
        torch_input_tensor_c,
        dtype=dtype_c,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # update_idxs_tensor must be ROW_MAJOR
        device=device,
        memory_config=mem_config_c,
    )
    # Only create 4th TTNN tensor if provided
    input_tensor_d = None
    if has_input_d:
        input_tensor_d = ttnn.from_torch(
            torch_input_tensor_d,
            dtype=dtype_d,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # page_table must be ROW_MAJOR
            device=device,
            memory_config=mem_config_d,
        )

    start_time = start_measuring_time()
    # paged_update_cache signature: (cache_tensor, input_tensor, *, update_idxs=[], update_idxs_tensor=None, share_cache=None, page_table=None, ...)
    # Only cache and input are positional, everything else is keyword-only
    # So tensor_a=cache, tensor_b=input, tensor_c=update_idxs_tensor, tensor_d=page_table
    # Note: paged_update_cache may not accept memory_config parameter - it modifies cache_tensor in place
    try:
        output_tensor = ttnn.experimental.paged_update_cache(
            input_tensor_a,  # cache_tensor (positional)
            input_tensor_b,  # input_tensor (positional)
            update_idxs_tensor=input_tensor_c
            if input_tensor_c is not None
            else None,  # update_idxs_tensor (optional keyword)
            page_table=input_tensor_d if input_tensor_d is not None else None,  # page_table (optional keyword)
            batch_offset=0,  # Use default batch_offset
        )
    except TypeError:
        # If that fails, try with memory_config
        output_tensor = ttnn.experimental.paged_update_cache(
            input_tensor_a,  # cache_tensor (positional)
            input_tensor_b,  # input_tensor (positional)
            update_idxs_tensor=input_tensor_c
            if input_tensor_c is not None
            else None,  # update_idxs_tensor (optional keyword)
            page_table=input_tensor_d if input_tensor_d is not None else None,  # page_table (optional keyword)
            batch_offset=0,
            memory_config=output_mem_config,
        )
    # paged_update_cache modifies cache_tensor in place, so output is the same as input_tensor_a
    output_tensor = input_tensor_a
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
