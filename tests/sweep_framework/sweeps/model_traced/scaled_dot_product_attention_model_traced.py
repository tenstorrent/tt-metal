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
TIMEOUT = 120

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("transformer::scaled_dot_product_attention", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 8, 32, 64)],  # Batch, heads, seq_len, head_dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Filter out configs that are known to cause timeouts or resource issues.
    """
    input_shape = test_vector.get("input_shape")
    input_a_memory_config = test_vector.get("input_a_memory_config")

    # If we have shape info, check for very large configs
    if isinstance(input_shape, dict):
        shape_q = input_shape.get("input_a") or input_shape.get("self")
        shape_k = input_shape.get("input_b") or input_shape.get("other")
        shape_v = input_shape.get("input_c")

        # Calculate approximate memory requirement
        if shape_q and isinstance(shape_q, (list, tuple)) and len(shape_q) >= 4:
            batch, num_heads, seq_len, head_dim = shape_q[0], shape_q[1], shape_q[2], shape_q[3]

            # Filter very large attention computations that cause timeouts
            # Attention complexity is O(batch * heads * seq_len^2 * head_dim)
            # Very large seq_len (> 4096) or very large num_heads (> 64) can cause timeouts
            if seq_len > 4096:
                return True, f"Sequence length {seq_len} too large (timeout risk)"

            if num_heads > 64:
                return True, f"Number of heads {num_heads} too large (timeout risk)"

            # Also filter configs with very large batch * heads * seq_len
            total_elements = batch * num_heads * seq_len * seq_len * head_dim
            if total_elements > 1024 * 1024 * 1024:  # 1B elements
                return (
                    True,
                    f"Attention computation too large: {total_elements / (1024**3):.2f}B elements (timeout risk)",
                )

    return False, None


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations (multi-input)
    if isinstance(input_shape, dict):
        # Traced configuration with multiple inputs (Q, K, V)
        shape_q = input_shape.get("input_a", input_shape.get("self"))
        shape_k = input_shape.get("input_b", input_shape.get("other"))
        shape_v = input_shape.get("input_c")
        if shape_v is None:
            # If only 2 inputs, use K shape for V
            shape_v = shape_k
    else:
        # Fallback for sample configurations
        if isinstance(input_shape, (tuple, list)):
            shape = tuple(input_shape)
        else:
            shape = input_shape
        shape_q = shape_k = shape_v = shape

    # Validate shapes - Q, K, V must have compatible shapes for attention
    # Q: [B, H_q, S_q, D], K: [B, H_k, S_k, D], V: [B, H_v, S_v, D]
    # For GQA (Grouped Query Attention), H_k and H_v can be less than H_q
    # But S_q, S_k, S_v must match, and D must match
    if isinstance(shape_q, (list, tuple)) and isinstance(shape_k, (list, tuple)) and isinstance(shape_v, (list, tuple)):
        if len(shape_q) == 4 and len(shape_k) == 4 and len(shape_v) == 4:
            # Check sequence length and head dimension match
            if shape_q[2] != shape_k[2] or shape_q[3] != shape_k[3]:
                # Adjust K/V to match Q if needed
                shape_k = (shape_k[0], shape_k[1], shape_q[2], shape_q[3])
            if shape_q[2] != shape_v[2] or shape_q[3] != shape_v[3]:
                shape_v = (shape_v[0], shape_v[1], shape_q[2], shape_q[3])
            # For GQA, we need to handle different num_heads
            # PyTorch SDPA requires Q and K/V to have same num_heads, so we'll replicate K/V heads if needed
            if shape_q[1] != shape_k[1]:
                # GQA case: replicate K heads to match Q
                # This is a simplification - real GQA would use grouped attention
                shape_k = (shape_k[0], shape_q[1], shape_k[2], shape_k[3])
            if shape_q[1] != shape_v[1]:
                shape_v = (shape_v[0], shape_q[1], shape_v[2], shape_v[3])

    # Use provided dtypes - fail if not provided (no fallbacks)
    dtype_q = input_a_dtype
    if input_b_dtype is None:
        raise ValueError("input_b_dtype is None - required parameter missing")
    if input_c_dtype is None:
        raise ValueError("input_c_dtype is None - required parameter missing")
    dtype_k = input_b_dtype
    dtype_v = input_c_dtype

    # Use provided layouts - fail if not provided (no fallbacks)
    layout_q = input_a_layout
    if input_b_layout is None:
        raise ValueError("input_b_layout is None - required parameter missing")
    if input_c_layout is None:
        raise ValueError("input_c_layout is None - required parameter missing")
    layout_k = input_b_layout
    layout_v = input_c_layout

    # Use provided memory configs - fail if not provided (no fallbacks)
    mem_config_q = input_a_memory_config
    if input_b_memory_config is None:
        raise ValueError("input_b_memory_config is None - required parameter missing")
    if input_c_memory_config is None:
        raise ValueError("input_c_memory_config is None - required parameter missing")
    # Fall back to input_a_memory_config if output_memory_config is not provided
    if output_memory_config is None:
        output_memory_config = input_a_memory_config
    mem_config_k = input_b_memory_config
    mem_config_v = input_c_memory_config
    output_mem_config = output_memory_config

    batch_size, num_heads_q, seq_len, head_dim = shape_q
    _, num_heads_k, _, _ = shape_k
    _, num_heads_v, _, _ = shape_v

    # Create Q, K, V tensors
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_q)(shape_q)
    torch_k = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_k)(shape_k)
    torch_v = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_v)(shape_v)

    # Handle GQA (Grouped Query Attention) - if K/V have fewer heads, replicate them
    if num_heads_k < num_heads_q:
        repeat_factor = num_heads_q // num_heads_k
        torch_k = torch_k.repeat(1, repeat_factor, 1, 1)
        if num_heads_q % num_heads_k != 0:
            remaining = num_heads_q - (repeat_factor * num_heads_k)
            torch_k = torch.cat([torch_k, torch_k[:, -num_heads_k : -num_heads_k + remaining, :, :]], dim=1)

    if num_heads_v < num_heads_q:
        repeat_factor = num_heads_q // num_heads_v
        torch_v = torch_v.repeat(1, repeat_factor, 1, 1)
        if num_heads_q % num_heads_v != 0:
            remaining = num_heads_q - (repeat_factor * num_heads_v)
            torch_v = torch.cat([torch_v, torch_v[:, -num_heads_v : -num_heads_v + remaining, :, :]], dim=1)

    # Quantize inputs to target dtype - both PyTorch and TTNN use same quantized inputs
    torch_q = ttnn.to_torch(
        ttnn.from_torch(torch_q, dtype=dtype_q, layout=layout_q, device=device, memory_config=mem_config_q)
    )
    torch_k = ttnn.to_torch(
        ttnn.from_torch(torch_k, dtype=dtype_k, layout=layout_k, device=device, memory_config=mem_config_k)
    )
    torch_v = ttnn.to_torch(
        ttnn.from_torch(torch_v, dtype=dtype_v, layout=layout_v, device=device, memory_config=mem_config_v)
    )

    # PyTorch reference
    torch_output_golden = torch.nn.functional.scaled_dot_product_attention(
        torch_q, torch_k, torch_v, attn_mask=None, dropout_p=0.0, is_causal=False
    )

    # TTNN execution
    q_tensor = ttnn.from_torch(torch_q, dtype=dtype_q, layout=layout_q, device=device, memory_config=mem_config_q)
    k_tensor = ttnn.from_torch(torch_k, dtype=dtype_k, layout=layout_k, device=device, memory_config=mem_config_k)
    v_tensor = ttnn.from_torch(torch_v, dtype=dtype_v, layout=layout_v, device=device, memory_config=mem_config_v)

    start_time = start_measuring_time()
    output_tensor = ttnn.transformer.scaled_dot_product_attention(
        q_tensor, k_tensor, v_tensor, is_causal=False, memory_config=output_mem_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Quantize PyTorch output to match TTNN output dtype for fair comparison
    torch_output_tensor = ttnn.to_torch(
        ttnn.from_torch(
            torch_output_golden, dtype=dtype_q, layout=layout_q, device=device, memory_config=output_mem_config
        )
    )

    # Check with PCC (threshold 0.99 for low-precision dtypes)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
