# Model traced sweep for rms_norm
# Generated automatically - DO NOT EDIT MANUALLY

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
model_traced_params = loader.get_suite_parameters("rms_norm", all_cases=False)

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
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle both sample suite (tuple/list) and model_traced suite (dict)
    if isinstance(input_shape, dict) and "self" in input_shape and "other" in input_shape:
        # This is model_traced suite - dict with 'self' and 'other' keys
        input_tensor_shape = input_shape["self"]
        # For RMS norm, weight should be 1D with size equal to last dimension of input
        # Ignore the traced weight shape as it may not be correct
        weight_tensor_shape = (input_tensor_shape[-1],)
    else:
        # This is sample suite - use simple shapes
        input_tensor_shape = input_shape if isinstance(input_shape, (tuple, list)) else tuple(input_shape)
        # For RMS norm, weight is typically 1D with size equal to last dimension of input
        weight_tensor_shape = (input_tensor_shape[-1],)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_tensor_shape)

    # Create weight tensor for RMS norm
    torch_weight = torch.randn(weight_tensor_shape, dtype=torch.float32)

    # RMS norm computation: x * weight / sqrt(mean(x^2) + eps)
    eps = 1e-5
    torch_input_squared = torch_input_tensor_a**2
    torch_mean_squared = torch.mean(torch_input_squared, dim=-1, keepdim=True)
    torch_rms = torch.sqrt(torch_mean_squared + eps)
    torch_output_tensor = torch_input_tensor_a * torch_weight / torch_rms

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    weight_tensor = ttnn.from_torch(
        torch_weight,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.rms_norm(input_tensor_a, epsilon=eps, weight=weight_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
