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
model_traced_params = loader.get_suite_parameters("typecast", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_dtype": [ttnn.float32],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
# Note: output_dtype is now included in the tuple format, so no need to add defaults
if model_traced_params and any(len(v) > 0 for v in model_traced_params.values() if isinstance(v, list)):
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_dtype,
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # Handle UINT16 and UINT32 specially - PyTorch doesn't have native unsigned types
    if input_a_dtype == ttnn.uint16:
        # For uint16, create values in valid range [0, 65535]
        torch_input_tensor_a = torch.randint(0, 65536, shape, dtype=torch.int32)
        # Convert to uint16 representation (but keep as int32 for PyTorch)
        torch_input_tensor_a = torch_input_tensor_a.clamp(0, 65535)
    elif input_a_dtype == ttnn.uint32:
        # For uint32, create values in valid range [0, 2^32-1]
        # Use int64 to avoid overflow
        torch_input_tensor_a = torch.randint(0, 2**32, shape, dtype=torch.int64)
    else:
        torch_input_tensor_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(shape)

    # Create PyTorch reference output based on output_dtype
    # Handle each dtype conversion explicitly for correct reference
    if output_dtype == ttnn.float32:
        # Convert to float32
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)
    elif output_dtype == ttnn.bfloat16:
        # Convert to bfloat16 then back to float32 for comparison
        torch_output_tensor = torch_input_tensor_a.to(torch.bfloat16).to(torch.float32)
    elif output_dtype == ttnn.bfloat8_b:
        # bfloat8_b doesn't have PyTorch equivalent, use float32
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)
    elif output_dtype == ttnn.uint16:
        # PyTorch doesn't have uint16, keep as int32
        torch_output_tensor = torch_input_tensor_a.clamp(0, 65535).to(torch.int32)
    elif output_dtype == ttnn.uint32:
        # For uint32 output, clamp to uint32 range and keep as int64 to avoid overflow
        if input_a_dtype == ttnn.uint32:
            # Input is already uint32 (int64), just ensure it's in range
            torch_output_tensor = torch_input_tensor_a.clamp(0, 2**32 - 1)
        else:
            # Converting from float/int to uint32
            torch_output_tensor = torch_input_tensor_a.clamp(0, 2**32 - 1).to(torch.int64)
    elif output_dtype == ttnn.int32:
        torch_output_tensor = torch_input_tensor_a.to(torch.int32)
    else:
        # Default to float32
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)

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
    output_tensor = ttnn.typecast(input_tensor_a, output_dtype, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Convert both to float32 for comparison to avoid dtype mismatch in PCC
    # Handle uint32 specially to avoid overflow in conversion
    if output_dtype == ttnn.uint32 or input_a_dtype == ttnn.uint32:
        # For uint32, convert to int64 first to avoid overflow, then to float
        # Ensure both tensors are int64 before converting to float32
        if torch_output_tensor.dtype != torch.int64:
            torch_output_tensor_f32 = torch_output_tensor.to(torch.int64).to(torch.float32)
        else:
            torch_output_tensor_f32 = torch_output_tensor.to(torch.float32)

        if output_tensor.dtype != torch.int64:
            output_tensor_f32 = output_tensor.to(torch.int64).to(torch.float32)
        else:
            output_tensor_f32 = output_tensor.to(torch.float32)
    else:
        torch_output_tensor_f32 = torch_output_tensor.to(torch.float32)
        output_tensor_f32 = output_tensor.to(torch.float32)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor_f32, output_tensor_f32, 0.999)

    return [pcc, e2e_perf]
