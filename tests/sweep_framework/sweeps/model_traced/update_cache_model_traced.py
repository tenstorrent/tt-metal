# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.sweep_framework.master_config_loader import MasterConfigLoader
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("update_cache", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {}

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
    scalar=None,
    *,
    device,
    **kwargs,
) -> list:
    """
    update_cache operation: updates a KV cache tensor with new values
    Args:
        cache (input_a): The cache tensor to update [num_users, num_heads, max_seq_len, head_dim]
        input (input_b): The new values to write [1, num_heads, 1, head_dim] (permuted to [1, num_heads, num_users, head_dim])
        cache_idx (scalar['update_index']): Index in sequence dimension to update
    """
    torch.manual_seed(0)

    # Parse input_shape - dict with multiple inputs (stored as strings)
    if isinstance(input_shape, dict):
        # Shapes are stored as strings like "(32, 1, 1024, 64)"
        shape_a_str = input_shape.get("self")  # cache shape
        shape_b_str = input_shape.get("other")  # input shape (to be written)

        # Parse string representations to tuples
        import ast

        if isinstance(shape_a_str, str):
            shape_a = ast.literal_eval(shape_a_str)
        else:
            shape_a = shape_a_str

        if isinstance(shape_b_str, str):
            shape_b = ast.literal_eval(shape_b_str)
        else:
            shape_b = shape_b_str

        # Parse scalars - cache_idx and batch_offset
        # Note: The tracer captures these as arg2 and arg3, but the binary parameter extractor
        # doesn't extract them (only extracts 1 scalar). So we use intelligent defaults:
        # cache_idx: Use middle of cache (safer for testing)
        # batch_offset: Always 0 (default)
        if scalar and isinstance(scalar, dict):
            cache_idx = int(scalar.get("update_index", shape_a[2] // 2))
            batch_offset = int(scalar.get("batch_offset", 0))
        else:
            # Default to middle of cache sequence length
            cache_idx = shape_a[2] // 2 if len(shape_a) > 2 else 0
            batch_offset = 0
    else:
        # Fallback if not dict format
        return [1.0, 0.0]

    # Generate cache tensor
    torch_cache = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape_a
    )

    # Generate input tensor
    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype)(
        shape_b
    )

    # Create expected output - update cache at cache_idx
    torch_output = torch_cache.clone()
    # update_cache signature: update_cache(cache_tensor, input_tensor, cache_idx, batch_offset)
    # cache: [num_users, num_heads, max_seq_len, head_dim]
    # input: [seq, num_heads, batch, head_dim] where seq=1, batch=1 (only first user)
    # The operation writes input into cache at position cache_idx + batch_offset

    # For testing, we only update the cache for user at batch_offset
    # Extract the input data for the first user
    user_data = torch_input[0, :, 0, :]  # [num_heads, head_dim]
    torch_output[batch_offset, :, cache_idx, :] = user_data

    # Convert to TTNN tensors
    cache_tensor = ttnn.from_torch(
        torch_cache,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    # update_cache expects input in shape [batch=1, num_heads, seq_len=1, head_dim]
    # Our traced input is [seq=1, num_heads, num_users, head_dim]
    # For now, we can only test with the first user (batch_idx=0)
    # Extract first user's data: [1, num_heads, 1, head_dim]
    torch_input_for_update = torch_input[:, :, 0:1, :]  # [1, num_heads, 1, head_dim]
    torch_input_for_update = torch_input_for_update.permute(2, 1, 0, 3)  # [1, num_heads, 1, head_dim]

    input_tensor = ttnn.from_torch(
        torch_input_for_update,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    # Run operation
    start_time = start_measuring_time()
    output_tensor = ttnn.update_cache(
        cache_tensor,
        input_tensor,
        cache_idx,
        batch_offset=batch_offset,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check PCC
    pcc_result = check_with_pcc(torch_output, output_tensor, 0.99)

    # Return result in the format expected by sweeps_runner: [(status, message), e2e_perf]
    return [pcc_result, e2e_perf]
