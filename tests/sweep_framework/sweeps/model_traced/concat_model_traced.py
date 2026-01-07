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
model_traced_params = loader.get_suite_parameters("concat", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [{"input_a": (1, 1, 32, 16), "input_b": (1, 1, 32, 8)}],  # Two tensors to concatenate
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    dim,
    output_memory_config,
    input_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle default values if not provided (for model_traced suite)
    if input_shape is None:
        raise ValueError("input_shape is None - required parameter missing")
    if input_a_dtype is None:
        raise ValueError("input_a_dtype is None - required parameter missing")
    if input_a_layout is None:
        raise ValueError("input_a_layout is None - required parameter missing")
    if input_a_memory_config is None:
        raise ValueError("input_a_memory_config is None - required parameter missing")
    if input_b_dtype is None:
        raise ValueError("input_b_dtype is None - required parameter missing")
    if input_b_layout is None:
        raise ValueError("input_b_layout is None - required parameter missing")
    if input_b_memory_config is None:
        raise ValueError("input_b_memory_config is None - required parameter missing")

    # Handle input_shape - can be dict (from model_traced) or tuple/list (from sample)
    if isinstance(input_shape, dict):
        # Extract shapes from dict (input_a, input_b, etc.)
        shape_a = tuple(input_shape.get("input_a", input_shape.get("input_0", [])))
        shape_b = tuple(input_shape.get("input_b", input_shape.get("input_1", [])))
    elif isinstance(input_shape, (tuple, list)):
        # Legacy format: single shape, create second tensor with modified dim
        shape_a = tuple(input_shape)
        shape_b = list(shape_a)
        # Normalize dim to positive index
        dim_idx = dim if dim >= 0 else len(shape_a) + dim
        shape_b[dim_idx] = shape_a[dim_idx] // 2  # Second tensor has half the size along concat dim
        shape_b = tuple(shape_b)
    else:
        raise ValueError(f"input_shape must be dict or tuple/list, got {type(input_shape)}")

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(shape_b)

    # Concatenate along specified dimension
    torch_output_tensor = torch.cat([torch_input_tensor_a, torch_input_tensor_b], dim=dim)

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

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_b_dtype,
        "layout": input_b_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = input_b_memory_config

    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, **from_torch_kwargs)

    start_time = start_measuring_time()
    output_tensor = ttnn.concat([input_tensor_a, input_tensor_b], dim=dim, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
