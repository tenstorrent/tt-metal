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
model_traced_params = loader.get_suite_parameters("rms_norm_pre_all_gather", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
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
    storage_type="StorageType::DEVICE",
    input_b_dtype=None,  # Weight dtype (optional, defaults to input_a_dtype)
    input_b_layout=None,  # Weight layout (optional, defaults to input_a_layout)
    input_b_memory_config=None,  # Weight memory config (optional, defaults to input_a_memory_config)
    output_memory_config=None,  # Optional, defaults to input_a_memory_config
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Use defaults for weight parameters if not provided
    if input_b_dtype is None:
        input_b_dtype = input_a_dtype
    if input_b_layout is None:
        input_b_layout = input_a_layout
    if input_b_memory_config is None:
        input_b_memory_config = input_a_memory_config
    if output_memory_config is None:
        output_memory_config = input_a_memory_config

    # Handle both sample suite (tuple/list) and model_traced suite (dict)
    if isinstance(input_shape, dict):
        # Model traced suite - can have 'self'/'other' or 'input_a' keys
        if "input_a" in input_shape:
            # Multi-input format: use input_a as the input tensor
            input_tensor_shape = input_shape["input_a"]
            # Check if weight shape is provided in input_b
            weight_tensor_shape = input_shape.get("input_b")
            if weight_tensor_shape is None:
                # Fallback: estimate weight shape from input
                if isinstance(input_tensor_shape, list):
                    input_tensor_shape = tuple(int(x) for x in input_tensor_shape)
                weight_tensor_shape = (int(input_tensor_shape[-1]),)
            else:
                if isinstance(weight_tensor_shape, list):
                    weight_tensor_shape = tuple(int(x) for x in weight_tensor_shape)
            if isinstance(input_tensor_shape, list):
                input_tensor_shape = tuple(int(x) for x in input_tensor_shape)
        elif "self" in input_shape and "other" in input_shape:
            # Binary format: self, other
            input_tensor_shape = input_shape["self"]
            weight_tensor_shape = input_shape["other"]
            if isinstance(input_tensor_shape, list):
                input_tensor_shape = tuple(int(x) for x in input_tensor_shape)
            if isinstance(weight_tensor_shape, list):
                weight_tensor_shape = tuple(int(x) for x in weight_tensor_shape)
        elif "self" in input_shape:
            # Unary format: only self
            input_tensor_shape = input_shape["self"]
            if isinstance(input_tensor_shape, list):
                input_tensor_shape = tuple(int(x) for x in input_tensor_shape)
            weight_tensor_shape = (int(input_tensor_shape[-1]),)
        else:
            # Unknown dict format, use first value
            first_key = list(input_shape.keys())[0]
            input_tensor_shape = input_shape[first_key]
            if isinstance(input_tensor_shape, list):
                input_tensor_shape = tuple(int(x) for x in input_tensor_shape)
            weight_tensor_shape = (int(input_tensor_shape[-1]),)
    else:
        # This is sample suite - use simple shapes
        input_tensor_shape = input_shape if isinstance(input_shape, (tuple, list)) else tuple(input_shape)
        if isinstance(input_tensor_shape, list):
            input_tensor_shape = tuple(int(x) for x in input_tensor_shape)
        # For RMS norm, weight is typically 1D with size equal to last dimension of input
        weight_tensor_shape = (int(input_tensor_shape[-1]),)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_tensor_shape)

    # Create weight tensor for RMS norm
    torch_weight = torch.randn(weight_tensor_shape, dtype=torch.float32)

    # For PyTorch reference computation, we need a 1D weight that matches input's last dim
    # The weight should match the input tensor's last dimension
    input_last_dim = int(input_tensor_shape[-1])

    # Flatten weight and take only the elements we need
    torch_weight_flat = torch_weight.flatten()
    weight_size_for_pytorch = min(len(torch_weight_flat), input_last_dim)
    torch_weight_1d_for_pytorch = torch_weight_flat[:weight_size_for_pytorch]

    # If weight is shorter than input's last dim, pad with ones (or repeat)
    if weight_size_for_pytorch < input_last_dim:
        # Repeat the weight to match input size
        repeat_factor = (input_last_dim + weight_size_for_pytorch - 1) // weight_size_for_pytorch
        torch_weight_1d_for_pytorch = torch_weight_1d_for_pytorch.repeat(repeat_factor)[:input_last_dim]

    # RMS norm computation: x * weight / sqrt(mean(x^2) + eps)
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

    # Create weight tensor - must be ROW_MAJOR layout for RMS norm operations
    # Weight needs to be reshaped to [1, 1, num_elements/TILE_WIDTH, TILE_WIDTH] format
    # where TILE_WIDTH = 32
    TILE_WIDTH = 32
    weight_size = torch_weight.numel()
    if weight_size % TILE_WIDTH != 0:
        # Pad to nearest multiple of TILE_WIDTH
        pad_size = TILE_WIDTH - (weight_size % TILE_WIDTH)
        torch_weight = torch.cat([torch_weight.flatten(), torch.zeros(pad_size, dtype=torch_weight.dtype)])
        weight_size = torch_weight.numel()
    # Reshape to [1, 1, H, 32] where H = weight_size / TILE_WIDTH
    weight_h = weight_size // TILE_WIDTH
    torch_weight_reshaped = torch_weight.flatten()[:weight_size].reshape(1, 1, weight_h, TILE_WIDTH)

    weight_tensor = ttnn.from_torch(
        torch_weight_reshaped,
        dtype=input_b_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # RMS norm requires ROW_MAJOR layout for weight
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    # Fall back to input_a_memory_config if output_memory_config is not provided
    if actual_output_memory_config is None:
        actual_output_memory_config = actual_input_memory_config

    # Use rms_norm_pre_all_gather operation
    # This operation computes stats (sum(x^2)) that would be gathered across devices
    # For single-device test, we use it directly and then apply post_all_gather logic
    stats_tensor = ttnn.rms_norm_pre_all_gather(input_tensor_a, memory_config=actual_output_memory_config)
    # For single device, stats are already "gathered", so we can use post_all_gather directly
    output_tensor = ttnn.rms_norm_post_all_gather(
        input_tensor_a,
        stats_tensor,
        epsilon=eps,
        weight=weight_tensor,
        memory_config=actual_output_memory_config,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
