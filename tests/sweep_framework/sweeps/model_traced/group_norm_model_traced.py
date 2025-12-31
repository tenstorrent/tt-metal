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
model_traced_params = loader.get_suite_parameters("group_norm", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 1024, 32)],  # Shape: [N, 1, H*W, C] as per ttnn.group_norm docs
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "num_groups": [8],
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
    output_memory_config,
    num_groups,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # ttnn group_norm expects shape [N, 1, H*W, C] where C is divisible by num_groups
    # torch group_norm expects shape (N, C, *)
    # For torch, we need to reshape: [1, 1, 1024, 32] -> [1, 32, 32, 32] for group_norm
    # Reshape for torch compatibility
    if len(shape) == 4 and shape[1] == 1:
        # ttnn format: [N, 1, H*W, C] -> torch format: [N, C, H, W]
        N, _, HW, C = shape
        H = W = int(HW**0.5)
        torch_input_reshaped = torch_input_tensor_a.reshape(N, H, W, C).permute(0, 3, 1, 2)
    else:
        torch_input_reshaped = torch_input_tensor_a

    torch_output_tensor = torch.nn.functional.group_norm(torch_input_reshaped, num_groups)
    # Reshape back to ttnn format
    if len(shape) == 4 and shape[1] == 1:
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).reshape(shape)

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
    # group_norm requires inplace=False and core_grid for our test case
    num_channels = shape[1] if len(shape) > 1 else 1
    weight = None
    bias = None
    # Use a simple core grid for testing
    core_grid = ttnn.CoreGrid(y=1, x=1)
    output_tensor = ttnn.group_norm(
        input_tensor_a,
        num_groups=num_groups,
        weight=weight,
        bias=bias,
        inplace=False,
        core_grid=core_grid,
        memory_config=output_memory_config,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
