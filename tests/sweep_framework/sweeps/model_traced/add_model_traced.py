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

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("add", all_cases=False)

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


def run(
    input_shape,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Extract scalar if present (for tensor-scalar add operations)
    scalar = kwargs.get("scalar", None)

    # Handle both sample suite (tuple) and model_traced suite (dict)
    if isinstance(input_shape, dict) and "self" in input_shape and "other" in input_shape:
        # This is model_traced suite - dict with 'self' and 'other' keys
        shape_a = input_shape["self"]
        shape_b = input_shape["other"]
    else:
        # This is sample suite - use same shape for both inputs
        if isinstance(input_shape, (tuple, list)):
            shape_a = tuple(input_shape)
            shape_b = tuple(input_shape)
        else:
            shape_a = input_shape
            shape_b = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Check if this is a scalar add operation (shape_b is None and scalar is provided)
    if shape_b is None and scalar is not None:
        # Tensor-scalar add: use the scalar value directly
        torch_output_tensor = torch.add(torch_input_tensor_a, scalar)
        is_scalar_add = True
    else:
        # Tensor-tensor add: generate second tensor
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)
        is_scalar_add = False

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

    start_time = start_measuring_time()

    if is_scalar_add:
        # Tensor-scalar add: pass scalar directly
        output_tensor = ttnn.add(input_tensor_a, scalar, memory_config=output_memory_config)
    else:
        # Tensor-tensor add: convert second tensor and add
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
        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=output_memory_config)

    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
