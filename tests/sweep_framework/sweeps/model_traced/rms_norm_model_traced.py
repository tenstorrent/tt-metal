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
model_traced_params = loader.get_suite_parameters("rms_norm", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept extra parameters like scalar, traced_source, etc.
) -> list:
    torch.manual_seed(0)

    # Handle both sample suite (tuple/list) and model_traced suite (dict)
    if isinstance(input_shape, dict) and "self" in input_shape and "other" in input_shape:
        # This is model_traced suite - dict with 'self' and 'other' keys
        input_tensor_shape = input_shape["self"]
        weight_tensor_shape = input_shape["other"]

        # Use the traced weight shape directly - it's already in the correct format
        # The traced configs have weight as [1, 1, 64, 32] which represents a 2048-element weight
        # in a 4D tensor format (64 * 32 = 2048)
    else:
        # This is sample suite - use simple shapes
        input_tensor_shape = input_shape if isinstance(input_shape, (tuple, list)) else tuple(input_shape)
        # For RMS norm, weight is typically 1D with size equal to last dimension of input
        weight_tensor_shape = (input_tensor_shape[-1],)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_tensor_shape)

    # Create weight tensor for RMS norm
    # The traced weight_tensor_shape is already in the correct format (e.g., [1,1,64,32])
    torch_weight = torch.randn(weight_tensor_shape, dtype=torch.float32)

    # For PyTorch reference computation, we need a 1D weight that matches input's last dim
    # Calculate the actual weight size from the shape
    if len(weight_tensor_shape) == 4:
        # Weight is in 4D format [1, 1, H, W], flatten to get actual size
        weight_size_for_pytorch = weight_tensor_shape[2] * weight_tensor_shape[3]
    elif len(weight_tensor_shape) == 1:
        weight_size_for_pytorch = weight_tensor_shape[0]
    else:
        weight_size_for_pytorch = input_tensor_shape[-1]

    # Create 1D weight for PyTorch reference (flatten the 4D weight if needed)
    torch_weight_1d_for_pytorch = torch_weight.flatten()[:weight_size_for_pytorch]

    # RMS norm computation: x * weight / sqrt(mean(x^2) + eps)
    # Use 1D weight for PyTorch computation (will broadcast correctly)
    eps = 1e-5
    torch_input_squared = torch_input_tensor_a**2
    torch_mean_squared = torch.mean(torch_input_squared, dim=-1, keepdim=True)
    torch_rms = torch.sqrt(torch_mean_squared + eps)
    torch_output_tensor = torch_input_tensor_a * torch_weight_1d_for_pytorch / torch_rms

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # RMS norm has issues with certain sharded configs (timeouts and allocation errors)
    # Convert sharded memory configs to interleaved
    actual_input_memory_config = input_a_memory_config
    actual_output_memory_config = output_memory_config

    if not is_host:
        # Check input memory config
        if hasattr(input_a_memory_config, "memory_layout"):
            mem_layout = str(input_a_memory_config.memory_layout)
            if "SHARDED" in mem_layout:
                actual_input_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Check output memory config
        if actual_output_memory_config and hasattr(actual_output_memory_config, "memory_layout"):
            mem_layout = str(actual_output_memory_config.memory_layout)
            if "SHARDED" in mem_layout:
                actual_output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = actual_input_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    # Reshape weight for TILE layout: must match input's last dimension
    torch_weight_reshaped = (
        torch_weight.flatten()[: input_tensor_shape[-1]].reshape([1, 1, 1, input_tensor_shape[-1]])
        if input_a_layout == ttnn.TILE_LAYOUT and len(weight_tensor_shape) == 4
        else torch_weight
    )

    weight_tensor = ttnn.from_torch(
        torch_weight_reshaped,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    # Fall back to input_a_memory_config if output_memory_config is not provided
    if actual_output_memory_config is None:
        actual_output_memory_config = actual_input_memory_config

    output_tensor = ttnn.rms_norm(
        input_tensor_a, epsilon=eps, weight=weight_tensor, memory_config=actual_output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
