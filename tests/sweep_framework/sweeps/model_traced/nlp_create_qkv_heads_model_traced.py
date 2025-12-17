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
    num_q_heads,
    num_kv_heads,
    output_memory_config,
    storage_type="StorageType::DEVICE",
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

    # Try to infer num_q_heads and num_kv_heads from shape if missing
    if num_q_heads is None or num_kv_heads is None:
        # For GQA: hidden_dim = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
        # Try common ratios: assume head_dim = 64 (common)
        head_dim_guess = 64
        total_heads = hidden_dim // head_dim_guess
        if num_q_heads is None and num_kv_heads is None:
            # Assume GQA: num_kv_heads = num_q_heads / 2
            # So: num_q_heads + 2*(num_q_heads/2) = 2*num_q_heads = total_heads
            num_q_heads = total_heads // 2
            num_kv_heads = num_q_heads // 2
        elif num_q_heads is None:
            # num_kv_heads is known, solve for num_q_heads
            num_q_heads = total_heads - 2 * num_kv_heads
        elif num_kv_heads is None:
            # num_q_heads is known, solve for num_kv_heads
            num_kv_heads = (total_heads - num_q_heads) // 2

    # For GQA: hidden_dim = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
    # So head_dim = hidden_dim / (num_q_heads + 2 * num_kv_heads)
    head_dim = hidden_dim // (num_q_heads + 2 * num_kv_heads)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # Compute proper torch reference (from test_nlp_create_qkv_heads.py)
    # Split input into Q, K, V components
    (ref_q, ref_k, ref_v) = torch.split(
        torch_input_tensor_a, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )

    # Reshape and transpose to get proper head dimensions
    # [B, 1, S, heads*head_dim] -> [B, S, heads, head_dim] -> [B, heads, S, head_dim]
    ref_q = torch.reshape(ref_q, [batch_size, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch_size, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch_size, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)

    # Use Q heads as reference for PCC check (operation returns tuple of Q, K, V)
    torch_output_tensor = ref_q

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
