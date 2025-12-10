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
model_traced_params = loader.get_suite_parameters("transformer::chunked_scaled_dot_product_attention", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 8, 32, 64)],  # Batch, heads, seq_len, head_dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    input_d_dtype=None,  # Ignored (loader generates this but operation doesn't use it)
    input_d_layout=None,  # Ignored
    input_d_memory_config=None,  # Ignored
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations (multi-input)
    if isinstance(input_shape, dict):
        # Traced configuration with multiple inputs (Q, K, V, page_table)
        shape_q = input_shape.get("input_a", input_shape.get("self"))
        shape_k = input_shape.get("input_b", input_shape.get("other"))
        shape_v = input_shape.get("input_c")
        shape_page_table = input_shape.get("input_d")
        if shape_v is None:
            # If only 2 inputs, use K shape for V
            shape_v = shape_k
        if shape_page_table is None:
            # Default page table shape: [batch_size, num_pages]
            # Estimate num_pages from sequence length (assuming 64 tokens per page)
            if isinstance(shape_q, (list, tuple)) and len(shape_q) >= 3:
                batch_size = int(shape_q[0])
                seq_len = int(shape_q[2])
                num_pages = max(1, (seq_len + 63) // 64)  # Round up to nearest page
                shape_page_table = (batch_size, num_pages)
            else:
                shape_page_table = (1, 1)
        else:
            # Ensure shape_page_table is a tuple of integers
            if isinstance(shape_page_table, list):
                shape_page_table = tuple(int(x) for x in shape_page_table)
    else:
        # Fallback for sample configurations
        if isinstance(input_shape, (tuple, list)):
            shape = tuple(input_shape)
        else:
            shape = input_shape
        shape_q = shape_k = shape_v = shape
        # Default page table shape
        if len(shape) >= 3:
            batch_size = int(shape[0])
            seq_len = int(shape[2])
            num_pages = max(1, (seq_len + 63) // 64)
            shape_page_table = (batch_size, num_pages)
        else:
            shape_page_table = (1, 1)

    # Ensure shapes are tuples of integers
    if isinstance(shape_q, list):
        shape_q = tuple(int(x) for x in shape_q)
    if isinstance(shape_k, list):
        shape_k = tuple(int(x) for x in shape_k)
    if isinstance(shape_v, list):
        shape_v = tuple(int(x) for x in shape_v)

    # Validate shapes - Q, K, V must have compatible shapes for attention
    # Note: For chunked attention, K and V can have different batch/head dimensions
    # The operation handles this internally, so we don't need to adjust shapes
    # Just ensure they're valid tuples
    if not (isinstance(shape_q, tuple) and isinstance(shape_k, tuple) and isinstance(shape_v, tuple)):
        raise ValueError(f"Invalid shape types: Q={type(shape_q)}, K={type(shape_k)}, V={type(shape_v)}")

    if len(shape_q) != 4 or len(shape_k) != 4 or len(shape_v) != 4:
        raise ValueError(f"Shapes must be 4D: Q={shape_q}, K={shape_k}, V={shape_v}")

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

    batch_size_q, num_heads_q, seq_len_q, head_dim = shape_q
    batch_size_k, num_heads_k, seq_len_k, _ = shape_k
    batch_size_v, num_heads_v, seq_len_v, _ = shape_v

    # Create Q, K, V tensors
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_q)(shape_q)
    torch_k = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_k)(shape_k)
    torch_v = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_v)(shape_v)

    # Create page table tensor - maps page indices to KV cache indices
    # For testing, create a simple sequential mapping
    if isinstance(shape_page_table, (list, tuple)) and len(shape_page_table) == 2:
        page_table_batch = int(shape_page_table[0])
        num_pages = int(shape_page_table[1])
        # Create page table: each page maps to sequential indices starting from 0
        # Page table shape: [batch_size, num_pages] with int32 values
        torch_page_table = torch.zeros(page_table_batch, num_pages, dtype=torch.int32)
        for i in range(num_pages):
            torch_page_table[:, i] = i * 64  # Each page is 64 tokens
    else:
        # Fallback: create minimal page table
        torch_page_table = torch.zeros(1, 1, dtype=torch.int32)

    chunk_start_idx = 0  # Default: start from beginning of sequence

    # For chunked attention, K and V can have different batch sizes (they represent KV cache)
    # The output batch size matches Q's batch size
    # Handle batch size mismatch: if K/V have larger batch, take first batch_size_q elements
    if batch_size_k > batch_size_q:
        torch_k = torch_k[:batch_size_q]
    if batch_size_v > batch_size_q:
        torch_v = torch_v[:batch_size_q]

    # Handle sequence length mismatch: K/V might have different seq_len (cache length)
    # Use Q's sequence length for the output
    if seq_len_k != seq_len_q:
        if seq_len_k < seq_len_q:
            # Pad K if needed
            pad_size = seq_len_q - seq_len_k
            torch_k = torch.nn.functional.pad(torch_k, (0, 0, 0, pad_size, 0, 0, 0, 0))
        else:
            # Take first seq_len_q elements
            torch_k = torch_k[:, :, :seq_len_q, :]
    if seq_len_v != seq_len_q:
        if seq_len_v < seq_len_q:
            pad_size = seq_len_q - seq_len_v
            torch_v = torch.nn.functional.pad(torch_v, (0, 0, 0, pad_size, 0, 0, 0, 0))
        else:
            torch_v = torch_v[:, :, :seq_len_q, :]

    # Handle GQA (Grouped Query Attention) - if K/V have fewer heads, replicate them
    if num_heads_k < num_heads_q:
        # Replicate K heads to match Q
        repeat_factor = num_heads_q // num_heads_k
        torch_k = torch_k.repeat(1, repeat_factor, 1, 1)
        if num_heads_q % num_heads_k != 0:
            # If not divisible, pad with last head
            remaining = num_heads_q - (repeat_factor * num_heads_k)
            torch_k = torch.cat([torch_k, torch_k[:, -num_heads_k : -num_heads_k + remaining, :, :]], dim=1)

    if num_heads_v < num_heads_q:
        # Replicate V heads to match Q
        repeat_factor = num_heads_q // num_heads_v
        torch_v = torch_v.repeat(1, repeat_factor, 1, 1)
        if num_heads_q % num_heads_v != 0:
            remaining = num_heads_q - (repeat_factor * num_heads_v)
            torch_v = torch.cat([torch_v, torch_v[:, -num_heads_v : -num_heads_v + remaining, :, :]], dim=1)

    # PyTorch scaled dot product attention as golden reference
    # Note: For chunked attention with paged cache, this may not be perfectly accurate
    # but it provides a reasonable reference for validation
    torch_output_tensor = torch.nn.functional.scaled_dot_product_attention(
        torch_q, torch_k, torch_v, attn_mask=None, dropout_p=0.0, is_causal=False
    )

    # Convert to TTNN tensors
    q_tensor = ttnn.from_torch(
        torch_q,
        dtype=dtype_q,
        layout=layout_q,
        device=device,
        memory_config=mem_config_q,
    )

    k_tensor = ttnn.from_torch(
        torch_k,
        dtype=dtype_k,
        layout=layout_k,
        device=device,
        memory_config=mem_config_k,
    )

    v_tensor = ttnn.from_torch(
        torch_v,
        dtype=dtype_v,
        layout=layout_v,
        device=device,
        memory_config=mem_config_v,
    )

    # Create page table tensor (int32)
    page_table_tensor = ttnn.from_torch(
        torch_page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_config_q,  # Use Q's memory config for page table
    )

    start_time = start_measuring_time()
    # Chunked attention with page table and chunk start index
    output_tensor = ttnn.transformer.chunked_scaled_dot_product_attention(
        q_tensor, k_tensor, v_tensor, page_table_tensor, chunk_start_idx, memory_config=output_mem_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - chunked attention with paged cache may have lower PCC
    # due to paging complexity, but we use standard attention as reference
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
