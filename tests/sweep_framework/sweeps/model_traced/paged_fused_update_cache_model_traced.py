# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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


# Removed invalidate_vector - all configs in master JSON are valid
# Debugging why only 4/20 configs run


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
    update_idxs=[],
    update_idxs_tensor=None,
    page_table=None,
    share_cache=None,
    batch_offset=0,
    storage_type="StorageType::DEVICE",
    traced_source=None,
    traced_machine_info=None,
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

    torch_output = torch_input_a

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

    # Ensure we have exactly 4 tensors for the positional arguments
    if len(input_tensors) != 4:
        raise ValueError(f"paged_fused_update_cache requires exactly 4 tensor inputs, got {len(input_tensors)}")

    # Handle additional tensor parameters: update_idxs_tensor and page_table
    update_idxs_tensor_ttnn = None
    if update_idxs_tensor is not None and isinstance(update_idxs_tensor, dict):
        # update_idxs_tensor is a dict with shape, dtype, layout, memory_config
        shape_e = update_idxs_tensor.get("shape")
        dtype_e = update_idxs_tensor.get("dtype")
        layout_e = update_idxs_tensor.get("layout")
        memory_config_e = update_idxs_tensor.get("memory_config")

        if shape_e:
            torch_input_e = gen_func_with_cast_tt(partial(torch_random, low=0, high=32, dtype=torch.float32), dtype_e)(
                shape_e
            )
            from_torch_kwargs_e = {"dtype": dtype_e, "layout": layout_e}
            if not is_host:
                from_torch_kwargs_e["device"] = device
                from_torch_kwargs_e["memory_config"] = memory_config_e
            update_idxs_tensor_ttnn = ttnn.from_torch(torch_input_e, **from_torch_kwargs_e)

    page_table_ttnn = None
    if page_table is not None and isinstance(page_table, dict):
        # page_table is a dict with shape, dtype, layout, memory_config
        shape_f = page_table.get("shape")
        dtype_f = page_table.get("dtype")
        layout_f = page_table.get("layout")
        memory_config_f = page_table.get("memory_config")

        if shape_f:
            torch_input_f = gen_func_with_cast_tt(
                partial(torch_random, low=0, high=1024, dtype=torch.float32), dtype_f
            )(shape_f)
            from_torch_kwargs_f = {"dtype": dtype_f, "layout": layout_f}
            if not is_host:
                from_torch_kwargs_f["device"] = device
                from_torch_kwargs_f["memory_config"] = memory_config_f
            page_table_ttnn = ttnn.from_torch(torch_input_f, **from_torch_kwargs_f)

    start_time = start_measuring_time()

    # Build kwargs for paged_fused_update_cache
    op_kwargs = {}

    # update_idxs: vector<uint32_t>
    if update_idxs is not None and isinstance(update_idxs, list) and len(update_idxs) > 0:
        op_kwargs["update_idxs"] = update_idxs
    else:
        op_kwargs["update_idxs"] = []  # Empty vector

    # update_idxs_tensor: optional Tensor
    if update_idxs_tensor_ttnn is not None:
        op_kwargs["update_idxs_tensor"] = update_idxs_tensor_ttnn

    # share_cache: optional<bool>
    if share_cache is not None:
        op_kwargs["share_cache"] = share_cache

    # page_table: optional Tensor
    if page_table_ttnn is not None:
        op_kwargs["page_table"] = page_table_ttnn

    # batch_offset: uint32_t
    if batch_offset is not None:
        op_kwargs["batch_offset"] = int(batch_offset)

    # Call the operation with all parameters
    result = ttnn.experimental.paged_fused_update_cache(*input_tensors, **op_kwargs)
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

    return [pcc, e2e_perf]
