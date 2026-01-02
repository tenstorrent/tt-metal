# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
import math
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

TIMEOUT = 30

# NOTE:
# -----
# For most ops, the model_traced suite uses real traced configurations from
# production models plus a PyTorch/TTNN golden.  For paged SDPA decode, we use
# the golden reference from the unit test (torch.nn.functional.scaled_dot_product_attention).
loader = MasterConfigLoader()

# Load traced configurations for paged_scaled_dot_product_attention_decode
_model_traced_params = loader.get_suite_parameters(
    "paged_scaled_dot_product_attention_decode", suite_name="model_traced", all_cases=False
)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 8, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Attach traced configurations
if _model_traced_params:
    parameters["model_traced"] = _model_traced_params


def nearest_n(x, n):
    """Round x up to nearest multiple of n"""
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    """Round x up to nearest power of 2"""
    if x < 1:
        raise ValueError("x must be >= 1")
    power = math.ceil(math.log2(x))
    return 1 << power


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
    input_d_dtype=None,
    input_d_layout=None,
    input_d_memory_config=None,
    input_e_dtype=None,
    input_e_layout=None,
    input_e_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, config_id, etc.
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations (multi-input)
    if isinstance(input_shape, dict):
        # Traced configuration with multiple inputs
        # Q, K, V, page_table, cur_pos
        shape_q = input_shape.get("input_a", input_shape.get("self"))  # Q
        shape_k_paged = input_shape.get("input_b")  # K (paged)
        shape_v_paged = input_shape.get("input_c")  # V (paged)
        shape_page_table = input_shape.get("input_d")  # page_table
        shape_cur_pos = input_shape.get("input_e")  # cur_pos
    else:
        # Fallback for sample configurations - use defaults
        shape_q = input_shape if isinstance(input_shape, (list, tuple)) else (1, 8, 32, 64)
        # For sample, create simple paged cache structure
        b, nh, d = 8, 32, 64
        block_size = 32
        max_num_blocks_per_seq = 16
        nkv = 1
        num_blocks = b * max_num_blocks_per_seq
        shape_k_paged = (num_blocks, nkv, block_size, d)
        shape_v_paged = (num_blocks, nkv, block_size, d)
        shape_page_table = (b, max_num_blocks_per_seq)
        shape_cur_pos = (b,)

    # Extract dimensions from shapes
    # Q: [1, b, nh, d]
    # K_paged: [num_blocks, nkv, block_size, d]
    # V_paged: [num_blocks, nkv, block_size, d]
    # page_table: [b, max_num_blocks_per_seq]
    # cur_pos: [b]

    _, b, nh, d = shape_q
    num_blocks, nkv, block_size, _ = shape_k_paged
    _, max_num_blocks_per_seq = shape_page_table

    # Calculate sequence length
    s = max_num_blocks_per_seq * block_size

    # Generate contiguous K and V caches first
    torch.manual_seed(0)
    K_contiguous = torch.randn(b, nkv, s, d, dtype=torch.float32)
    V_contiguous = torch.randn(b, nkv, s, d, dtype=torch.float32)

    # Generate cur_pos (decode positions for each batch)
    # Use random positions between 0 and s-1
    cur_pos = torch.randint(0, s, (b,), dtype=torch.int32)

    # Page the K and V caches
    def to_paged_cache(cache, batch, num_kv, max_num_blocks_per_seq, block_size, head_dim, max_seq_len):
        return (
            cache.reshape(batch, num_kv, max_num_blocks_per_seq, block_size, head_dim)
            .transpose(1, 2)
            .reshape(batch * max_num_blocks_per_seq, num_kv, block_size, head_dim)
        )

    K_paged = to_paged_cache(K_contiguous, b, nkv, max_num_blocks_per_seq, block_size, d, s)
    V_paged = to_paged_cache(V_contiguous, b, nkv, max_num_blocks_per_seq, block_size, d, s)

    # Create page table (identity mapping for simplicity - no shuffling)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(b, max_num_blocks_per_seq)

    # Generate Q
    Q = torch.randn(1, b, nh, d, dtype=torch.float32)

    # Compute PyTorch golden reference
    scale = d**-0.5
    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))

    # Compute expected output using unpaged K/V
    # For each batch, mask out tokens after cur_pos[batch]
    max_cur_pos = cur_pos.max().item()
    padded_layer_len = nearest_n(max_cur_pos + 1, n=32)  # Round up to tile size

    # Create causal mask
    attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
    for i in range(b):
        pos = cur_pos[i].item()
        attn_mask[i, :, :, pos + 1 :] = torch.finfo(torch.float32).min

    # Prepare inputs for PyTorch attention
    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K_contiguous[:, :, :padded_layer_len, :]  # b, nkv, S, d
    # Expand KV heads to match Q heads (GQA)
    K_slice = torch.cat(
        [K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, S, d
    V_slice = V_contiguous[:, :, :padded_layer_len, :]  # b, nkv, S, d
    V_slice = torch.cat(
        [V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1
    )  # b, nh, S, d
    attn_mask_slice = attn_mask[:, :nh, :, :]  # b, nh, 1, S

    torch_output_tensor = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    torch_output_tensor = torch_output_tensor.squeeze(2).unsqueeze(0)  # 1, b, nh, d

    # Convert to TTNN tensors
    tt_Q = ttnn.from_torch(
        Q,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    tt_K = ttnn.from_torch(
        K_paged,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    tt_V = ttnn.from_torch(
        V_paged,
        dtype=input_c_dtype,
        layout=input_c_layout,
        device=device,
        memory_config=input_c_memory_config,
    )

    tt_page_table = ttnn.Tensor(page_table, ttnn.int32).to(device)

    tt_cur_pos = ttnn.Tensor(cur_pos, ttnn.int32).to(device)

    start_time = start_measuring_time()

    # Run TTNN paged attention
    output_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        tt_page_table,
        is_causal=True,
        cur_pos_tensor=tt_cur_pos,
        scale=scale,
        memory_config=output_memory_config if output_memory_config else ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with 0.99 PCC (proper golden reference)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
