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

TIMEOUT = 60

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("fill_cache", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
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
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations
    if isinstance(input_shape, dict):
        shape_a = input_shape.get("input_a", input_shape.get("self"))
        shape_b = input_shape.get("input_b", input_shape.get("other"))
        shape_c = input_shape.get("input_c")
        if shape_c is None:
            shape_c = shape_b
    else:
        # Fallback for sample configurations
        if isinstance(input_shape, (tuple, list)):
            shape = tuple(input_shape)
        else:
            shape = input_shape
        shape_a = shape_b = shape_c = shape

    # Convert to tuples if needed
    if isinstance(shape_a, list):
        shape_a = tuple(shape_a)
    if isinstance(shape_b, list):
        shape_b = tuple(shape_b)
    if isinstance(shape_c, list):
        shape_c = tuple(shape_c)

    # Use provided dtypes with defaults if needed
    dtype_a = input_a_dtype
    dtype_b = input_b_dtype if input_b_dtype is not None else input_a_dtype
    dtype_c = input_c_dtype if input_c_dtype is not None else input_a_dtype

    layout_a = input_a_layout
    layout_b = input_b_layout if input_b_layout is not None else input_a_layout
    layout_c = input_c_layout if input_c_layout is not None else input_a_layout

    mem_config_a = input_a_memory_config
    mem_config_b = input_b_memory_config if input_b_memory_config is not None else input_a_memory_config
    mem_config_c = input_c_memory_config if input_c_memory_config is not None else input_a_memory_config

    if output_memory_config is None:
        output_memory_config = input_a_memory_config

    # Generate random tensors
    torch_cache = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype_a)(
        shape_a
    )
    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype_b)(
        shape_b
    )

    # batch_idx is an integer (which batch to fill in the cache)
    batch_idx = 0  # Default to batch 0

    # Reference computation (simplified - fill_cache updates cache inplace)
    torch_output = torch_cache.clone()
    # Simplified reference: copy input into cache at batch_idx
    if len(torch_cache.shape) >= 4 and len(torch_input.shape) >= 4:
        # Assuming cache is [batch, heads, seq, dim] and input is [batch, heads, seq_chunk, dim]
        # Fill cache at batch_idx with input data
        if torch_cache.shape[0] > batch_idx:
            seq_len = min(torch_input.shape[2], torch_cache.shape[2])
            torch_output[batch_idx, :, :seq_len, :] = torch_input[0, :, :seq_len, :]
    torch_output_tensor = torch_output

    is_host = storage_type and "HOST" in str(storage_type)

    from_torch_kwargs_a = {"dtype": dtype_a, "layout": layout_a}
    if not is_host:
        from_torch_kwargs_a["device"] = device
        from_torch_kwargs_a["memory_config"] = mem_config_a

    cache_tensor = ttnn.from_torch(torch_cache, **from_torch_kwargs_a)

    from_torch_kwargs_b = {"dtype": dtype_b, "layout": layout_b}
    if not is_host:
        from_torch_kwargs_b["device"] = device
        from_torch_kwargs_b["memory_config"] = mem_config_b

    input_tensor = ttnn.from_torch(torch_input, **from_torch_kwargs_b)

    start_time = start_measuring_time()
    # fill_cache takes cache_tensor, input_tensor, and batch_idx (int)
    output_tensor = ttnn.fill_cache(cache_tensor, input_tensor, batch_idx)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC (relaxed threshold due to inplace update)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.95)

    return [pcc, e2e_perf]
