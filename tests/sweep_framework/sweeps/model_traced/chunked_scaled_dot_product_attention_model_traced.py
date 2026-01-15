# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader import MasterConfigLoader

TIMEOUT = 30


def page_cache(cache, page_block_size, permutation):
    """Convert K or V cache to paged format.

    Args:
        cache: [b, nkv, s, d] tensor
        page_block_size: size of each page block
        permutation: permutation for shuffling pages

    Returns:
        paged_cache: [max_num_blocks, nkv, page_block_size, d]
    """
    b, nkv, s, d = cache.shape
    max_num_blocks_per_seq = s // page_block_size
    max_num_blocks = b * max_num_blocks_per_seq

    paged_cache = (
        cache.reshape(b, nkv, max_num_blocks_per_seq, page_block_size, d)
        .transpose(1, 2)
        .reshape(max_num_blocks, nkv, page_block_size, d)
    )

    shuffled_page_cache = paged_cache[permutation]
    return shuffled_page_cache


loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::chunked_scaled_dot_product_attention", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 8, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_d_dtype": [ttnn.bfloat16],
        "input_d_layout": [ttnn.TILE_LAYOUT],
        "input_d_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


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
    *,
    device,
    **kwargs,  # Accept any extra parameters (like input_d_*)
) -> list:
    torch.manual_seed(0)

    # Extract shapes for Q and K from input_shape dict or use defaults
    if isinstance(input_shape, dict):
        shape_q = input_shape.get("input_a", (1, 8, 32, 64))
        shape_k_paged = input_shape.get("input_b", (64, 1, 64, 64))  # [num_pages, nkv, page_size, d]
    else:
        shape_q = input_shape if isinstance(input_shape, (tuple, list)) else (1, 8, 32, 64)
        # Default paged format
        b, nh, sq, d = shape_q
        page_block_size = 64
        max_num_blocks_per_seq = max(1, (sq + page_block_size - 1) // page_block_size)
        nkv = 1
        shape_k_paged = (b * max_num_blocks_per_seq, nkv, page_block_size, d)

    # Extract dimensions from Q
    b, nh, sq, d = shape_q

    # For paged K, V: [num_pages, nkv, page_block_size, head_dim]
    num_pages, nkv, page_block_size, _ = shape_k_paged

    # Calculate unpaged sequence length from paged dimensions
    # The paged format has num_pages blocks, each of size page_block_size
    # For b batches: num_pages = b * max_num_blocks_per_seq
    max_num_blocks_per_seq = num_pages // b
    s = max_num_blocks_per_seq * page_block_size

    # Create unpaged K and V for torch reference
    # These represent the "logical" unpaged KV cache
    # Shape: [b, nkv, s, d]
    torch_k_unpaged = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype or input_a_dtype
    )((b, nkv, s, d))
    torch_v_unpaged = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_c_dtype or input_a_dtype
    )((b, nkv, s, d))

    # Create Q - note Q sequence length (sq) may be different from KV sequence length (s)
    # Q might be a chunk of the full sequence
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(shape_q)

    # Create page table and permutation
    # The page table maps logical page indices to physical page indices
    max_num_blocks = b * max_num_blocks_per_seq
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    torch_page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq)

    # Convert K, V to paged format for TTNN operation
    # This simulates what the actual KV cache would look like in paged format
    torch_k_paged = page_cache(torch_k_unpaged, page_block_size, permutation)
    torch_v_paged = page_cache(torch_v_unpaged, page_block_size, permutation)

    # Torch reference: Use unpaged K, V with standard SDPA
    # Repeat K, V for GQA if needed
    if nkv < nh:
        K_repeated = torch.cat(
            [torch_k_unpaged[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
        )
        V_repeated = torch.cat(
            [torch_v_unpaged[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
        )
    else:
        K_repeated = torch_k_unpaged
        V_repeated = torch_v_unpaged

    torch_output = torch.nn.functional.scaled_dot_product_attention(
        torch_q, K_repeated, V_repeated, attn_mask=None, dropout_p=0.0, is_causal=True
    )

    # Create TTNN tensors with paged K, V
    q_tensor = ttnn.from_torch(
        torch_q, dtype=input_a_dtype, layout=input_a_layout, device=device, memory_config=input_a_memory_config
    )
    k_tensor = ttnn.from_torch(
        torch_k_paged,
        dtype=input_b_dtype or input_a_dtype,
        layout=input_b_layout or ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_b_memory_config or input_a_memory_config,
    )
    v_tensor = ttnn.from_torch(
        torch_v_paged,
        dtype=input_c_dtype or input_a_dtype,
        layout=input_c_layout or ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_c_memory_config or input_a_memory_config,
    )
    page_table_tensor = ttnn.from_torch(
        torch_page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_a_memory_config,
    )

    # Op call - chunk_start_idx is 0 for full sequence
    start_time = start_measuring_time()
    output_tensor = ttnn.transformer.chunked_scaled_dot_product_attention(
        q_tensor, k_tensor, v_tensor, page_table_tensor, 0, memory_config=output_memory_config or input_a_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison - use 0.998 PCC threshold as in unit test
    return [check_with_pcc(torch_output, output_tensor, 0.998), e2e_perf]
