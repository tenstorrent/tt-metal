# Model traced sweep for nlp_create_qkv_heads
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
model_traced_params = loader.get_suite_parameters("experimental::nlp_create_qkv_heads", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 768)],  # Batch, seq, 1, hidden_dim (3 * num_heads * head_dim)
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "num_heads": [12],
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
    num_q_heads,
    num_kv_heads,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # Input shape is [B, 1, S, hidden_dim] where S is the sequence length
    # For shape [1, 1, 256, 1536]: B=1, S=256, hidden_dim=1536
    batch_size = shape[0]
    seq_len = shape[2]  # Third dimension is sequence length
    hidden_dim = shape[3]
    # For GQA: hidden_dim = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
    # So head_dim = hidden_dim / (num_q_heads + 2 * num_kv_heads)
    head_dim = hidden_dim // (num_q_heads + 2 * num_kv_heads)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # nlp_create_qkv_heads reshapes input [B, 1, S, hidden_dim] to Q heads [B, num_q_heads, S, head_dim]
    # Input shape: [1, 1, 256, 1536] -> Output Q: [1, 16, 256, 64]
    # So seq_len is the third dimension (256), not the second
    expected_output_shape = (batch_size, num_q_heads, seq_len, head_dim)
    torch_output_tensor = torch.zeros(expected_output_shape, dtype=torch_input_tensor_a.dtype)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    # nlp_create_qkv_heads signature: (input, input_kv=None, *, num_heads, num_kv_heads=None, ...)
    # Note: The function uses num_heads (not num_q_heads), and num_kv_heads is optional
    # Returns a tuple of tensors (q_heads, k_heads, v_heads)
    output_result = ttnn.experimental.nlp_create_qkv_heads(
        input_tensor_a, num_heads=num_q_heads, num_kv_heads=num_kv_heads, memory_config=output_memory_config
    )
    # Handle tuple return - convert to torch
    if isinstance(output_result, tuple):
        # Take the first tensor (q_heads) for comparison, or concatenate all
        output_tensor = ttnn.to_torch(output_result[0])
    else:
        output_tensor = ttnn.to_torch(output_result)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - using lower tolerance for complex operations
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
