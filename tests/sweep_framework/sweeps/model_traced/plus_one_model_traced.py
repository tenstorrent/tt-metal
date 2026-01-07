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
model_traced_params = loader.get_suite_parameters("plus_one", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.int32],  # plus_one requires INT32 or UINT32
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],  # plus_one requires ROW_MAJOR
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    # Override layout to ROW_MAJOR as required by plus_one operation
    # Keep the original dtypes from traced configs (they should be INT32/UINT32)
    if "input_a_layout" in model_traced_params:
        model_traced_params["input_a_layout"] = [ttnn.ROW_MAJOR_LAYOUT] * len(model_traced_params["input_a_layout"])
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
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

    # Generate tensor with correct dtype for plus_one (INT32/UINT32 required)
    # Check if dtype is int32 or uint32
    dtype_str = str(input_a_dtype).lower()
    if "int32" in dtype_str:
        torch_input_tensor_a = torch.randint(-100, 100, shape, dtype=torch.int32)
    elif "uint32" in dtype_str:
        torch_input_tensor_a = torch.randint(0, 200, shape, dtype=torch.int32)  # Will convert to uint32 in ttnn
    else:
        # Fallback for other dtypes (shouldn't happen with traced configs)
        torch_input_tensor_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(shape)

    # Plus one operation: x + 1
    torch_output_tensor = torch_input_tensor_a + 1

    # Force ROW_MAJOR layout as required by plus_one operation
    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": ttnn.ROW_MAJOR_LAYOUT,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    start_time = start_measuring_time()
    output_tensor = ttnn.plus_one(input_tensor_a)  # plus_one doesn't support memory_config
    # If needed, move to desired memory config
    if output_memory_config != input_a_memory_config:
        output_tensor = ttnn.to_memory_config(output_tensor, output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
