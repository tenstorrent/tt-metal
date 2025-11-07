# Model traced sweep for linear
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
model_traced_params = loader.get_suite_parameters("linear", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(32, 32)],  # Input shape (m, k)
        "weight_shape": [(32, 32)],  # Weight shape (k, n) - will be transposed internally
        "bias_shape": [(32,)],  # Bias shape (n,)
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "transpose_a": [False],
        "transpose_b": [False],
        "has_bias": [True],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params and any(len(v) > 0 for v in model_traced_params.values() if isinstance(v, list)):
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    weight_shape,
    bias_shape,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    transpose_a,
    transpose_b,
    has_bias,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle tuple shapes for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape_a = tuple(input_shape)
    else:
        shape_a = input_shape

    if isinstance(weight_shape, (tuple, list)):
        shape_b = tuple(weight_shape)
    else:
        shape_b = weight_shape

    if isinstance(bias_shape, (tuple, list)):
        shape_bias = tuple(bias_shape)
    else:
        shape_bias = bias_shape

    # Create random tensors
    torch_a = torch.randn(*shape_a, dtype=torch.float32)
    torch_b = torch.randn(*shape_b, dtype=torch.float32)

    # For torch.linear, weight matrix should be (out_features, in_features)
    torch_weight = torch_b
    if len(shape_b) >= 2:
        torch_weight = torch.transpose(torch_weight, -1, -2)

    # Create bias tensor if needed
    torch_bias = None
    ttnn_bias = None
    if has_bias and shape_bias is not None:
        torch_bias = torch.randn(*shape_bias, dtype=torch.float32) if shape_bias != tuple() else torch.randn(())
        ttnn_bias = ttnn.from_torch(torch_bias, layout=input_a_layout, device=device)

    # Golden output using PyTorch
    torch_output_tensor = torch.nn.functional.linear(torch_a, torch_weight, torch_bias)

    # Create TTNN tensors
    ttnn_a = ttnn.from_torch(
        torch_a, dtype=input_a_dtype, layout=input_a_layout, device=device, memory_config=input_a_memory_config
    )
    ttnn_b = ttnn.from_torch(
        torch_b, dtype=input_b_dtype, layout=input_b_layout, device=device, memory_config=input_b_memory_config
    )

    # Run TTNN linear
    start_time = start_measuring_time()
    output_tensor = ttnn.linear(ttnn_a, ttnn_b, bias=ttnn_bias, transpose_a=transpose_a, transpose_b=transpose_b)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
