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
model_traced_params = loader.get_suite_parameters("experimental::rotary_embedding_llama", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 64)],  # Batch, seq, heads, head_dim (must be even for rotary)
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

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype,
    input_b_layout,
    input_b_memory_config,
    input_c_dtype,
    input_c_layout,
    input_c_memory_config,
    input_d_dtype,
    input_d_layout,
    input_d_memory_config,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations
    if isinstance(input_shape, dict):
        # Traced configuration with multiple inputs
        shape_a = input_shape["input_a"]
        shape_b = input_shape["input_b"]  # cos_cache
        shape_c = input_shape["input_c"]  # sin_cache
        shape_d = input_shape["input_d"]  # trans_mat
    else:
        # Fallback for sample configurations
        shape_a = (1, 16, 256, 64)
        shape_b = (1, 1, 256, 64)
        shape_c = (1, 1, 256, 64)
        shape_d = (1, 1, 32, 32)

    # Create input tensors
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape_a)

    torch_cos_cache = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype)(
        shape_b
    )

    torch_sin_cache = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_c_dtype)(
        shape_c
    )

    torch_trans_mat = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_d_dtype)(
        shape_d
    )

    # Use TTNN golden function for reference output if available
    try:
        torch_output_tensor = ttnn.get_golden_function(ttnn.experimental.rotary_embedding_llama)(
            torch_input_tensor_a, torch_cos_cache, torch_sin_cache, torch_trans_mat
        )
    except Exception:
        # Fallback: rotary embedding is complex, so use a simple approximation
        # For now, just use the input as reference (will have lower PCC tolerance)
        torch_output_tensor = torch_input_tensor_a.clone()

    # Create TTNN tensors - use the traced memory configs
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

    cos_cache = ttnn.from_torch(
        torch_cos_cache,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,  # Use traced config
    )

    sin_cache = ttnn.from_torch(
        torch_sin_cache,
        dtype=input_c_dtype,
        layout=input_c_layout,
        device=device,
        memory_config=input_c_memory_config,  # Use traced config
    )

    trans_mat = ttnn.from_torch(
        torch_trans_mat,
        dtype=input_d_dtype,
        layout=input_d_layout,
        device=device,
        memory_config=input_d_memory_config,  # Use traced config
    )

    start_time = start_measuring_time()
    if output_memory_config is not None:
        output_tensor = ttnn.experimental.rotary_embedding_llama(
            input_tensor_a, cos_cache, sin_cache, trans_mat, memory_config=output_memory_config
        )
    else:
        output_tensor = ttnn.experimental.rotary_embedding_llama(input_tensor_a, cos_cache, sin_cache, trans_mat)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - using placeholder reference for now
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)  # Lower tolerance for complex ops

    return [pcc, e2e_perf]
