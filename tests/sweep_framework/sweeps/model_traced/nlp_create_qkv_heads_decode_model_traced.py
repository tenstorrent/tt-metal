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

TIMEOUT = 30

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_create_qkv_heads_decode", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 1536)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "num_heads": [16],
        "num_kv_heads": [4],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    num_heads,
    num_kv_heads,
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # Try to infer num_heads and num_kv_heads from shape if missing
    if num_heads is None or num_kv_heads is None:
        if len(shape) == 4:
            # Input shape: [1, 1, 1, hidden_dim] where hidden_dim = (num_heads + 2*num_kv_heads) * head_dim
            hidden_dim = shape[3]
            # Try common ratios: if num_kv_heads = num_heads, then hidden_dim = 3 * num_heads * head_dim
            # If num_kv_heads = num_heads / 2 (GQA), then hidden_dim = 2 * num_heads * head_dim
            # Try to infer: assume head_dim = 64 (common), then num_heads + 2*num_kv_heads = hidden_dim / 64
            head_dim_guess = 64
            total_heads = hidden_dim // head_dim_guess
            if num_heads is None and num_kv_heads is None:
                # Assume GQA: num_kv_heads = num_heads / 2
                # So: num_heads + 2*(num_heads/2) = 2*num_heads = total_heads
                num_heads = total_heads // 2
                num_kv_heads = num_heads // 2
            elif num_heads is None:
                # num_kv_heads is known, solve for num_heads
                num_heads = total_heads - 2 * num_kv_heads
            elif num_kv_heads is None:
                # num_heads is known, solve for num_kv_heads
                num_kv_heads = (total_heads - num_heads) // 2
        else:
            # Default fallbacks
            if num_heads is None:
                num_heads = 16
            if num_kv_heads is None:
                num_kv_heads = num_heads // 2

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # nlp_create_qkv_heads_decode returns Q, K, V heads with shapes:
    # Input shape: [1, seq_len, batch, hidden_dim] where hidden_dim = (num_heads + 2*num_kv_heads) * head_dim
    # Outputs: Q [seq_len, batch, num_heads, head_dim], K [seq_len, batch, num_kv_heads, head_dim], V [seq_len, batch, num_kv_heads, head_dim]
    # Reference implementation from test_nlp_create_qkv_heads_decode.py (lines 51-57)
    if len(shape) == 4:
        seq_len = shape[1]
        batch = shape[2]
        hidden_dim = shape[3]
        # Calculate head_dim from hidden_dim: hidden_dim = (num_heads + 2*num_kv_heads) * head_dim
        head_dim = hidden_dim // (num_heads + 2 * num_kv_heads)

        # Torch reference: split the input along the hidden dimension, then reshape and view
        # Q heads: first num_heads * head_dim elements
        q_heads_torch = torch_input_tensor_a[:, :, :batch, : head_dim * num_heads].view(
            seq_len, batch, num_heads, head_dim
        )
        # K and V heads: computed for completeness but only Q is used for PCC
        # (operation returns tuple of (Q, K, V) but we only validate Q)
        _ = torch_input_tensor_a[  # k_heads
            :, :, :batch, head_dim * num_heads : head_dim * (num_heads + num_kv_heads)
        ].view(seq_len, batch, num_kv_heads, head_dim)
        _ = torch_input_tensor_a[:, :, :batch, head_dim * (num_heads + num_kv_heads) :].view(  # v_heads
            seq_len, batch, num_kv_heads, head_dim
        )

        # For comparison, use Q heads (the first output from operation)
        torch_output_tensor = q_heads_torch
    else:
        torch_output_tensor = torch_input_tensor_a.clone()

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
    output_result = ttnn.experimental.nlp_create_qkv_heads_decode(
        input_tensor_a, num_heads=num_heads, num_kv_heads=num_kv_heads, memory_config=output_memory_config
    )
    # nlp_create_qkv_heads_decode returns a tuple of tensors (q_heads, k_heads, v_heads)
    # Convert to torch - handle tuple return
    if isinstance(output_result, tuple):
        # Take the first tensor (q_heads) for comparison, or concatenate all
        output_tensor = ttnn.to_torch(output_result[0])
    else:
        output_tensor = ttnn.to_torch(output_result)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - using proper torch reference from unit test
    # Compare Q heads (first output) with computed golden
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    return [pcc, e2e_perf]
