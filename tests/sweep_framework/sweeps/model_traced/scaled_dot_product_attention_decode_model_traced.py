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

TIMEOUT = 60

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::scaled_dot_product_attention_decode", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 8, 1, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    input_d_dtype=None,  # cur_pos tensor dtype
    input_d_layout=None,  # cur_pos tensor layout
    input_d_memory_config=None,  # cur_pos tensor memory config
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    scale=None,  # Extracted from arg8
    k_chunk_size=None,  # Extracted from arg9
    is_causal=None,  # Extracted from arg3
    *,
    device,
    **kwargs,  # Accept any extra parameters
) -> list:
    torch.manual_seed(0)

    # Handle both sample suite (tuple/list) and model_traced suite (dict with keys for multi-input ops)
    if isinstance(input_shape, dict):
        # Multi-input operation - extract individual shapes for Q, K, V
        shape_q = tuple(input_shape.get("input_a", (1, 8, 1, 64)))
        shape_k = tuple(input_shape.get("input_b", (1, 8, 2048, 64)))
        shape_v = tuple(input_shape.get("input_c", shape_k))
    else:
        # Convert list to tuple if needed
        shape_q = tuple(input_shape) if isinstance(input_shape, list) else input_shape
        # Default K, V shapes for sample - use larger sequence length for KV cache
        b, nh, sq, d = shape_q
        # For decode, K and V have accumulated cache, use 2048 as default cache size
        shape_k = (b, nh, 2048, d)
        shape_v = shape_k

    # Extract dimensions
    b, nh_q, sq, d = shape_q
    _, nh_kv, s_kv, _ = shape_k

    # Tensor creation with correct shapes
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(shape_q)
    torch_k = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype or input_a_dtype
    )(shape_k)
    torch_v = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_c_dtype or input_a_dtype
    )(shape_v)

    # For torch reference, handle GQA (Grouped Query Attention) if num_kv_heads < num_q_heads
    if nh_kv < nh_q:
        # Repeat K, V to match number of Q heads
        K_repeated = torch.cat(
            [torch_k[:, i : i + 1, :, :].repeat(1, nh_q // nh_kv, 1, 1) for i in range(nh_kv)], dim=1
        )
        V_repeated = torch.cat(
            [torch_v[:, i : i + 1, :, :].repeat(1, nh_q // nh_kv, 1, 1) for i in range(nh_kv)], dim=1
        )
    else:
        K_repeated = torch_k
        V_repeated = torch_v

    torch_output = torch.nn.functional.scaled_dot_product_attention(
        torch_q.to(torch.bfloat16),
        K_repeated.to(torch.bfloat16),
        V_repeated.to(torch.bfloat16),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,  # Decode is causal
    )

    # Create TTNN tensors
    q_tensor = ttnn.from_torch(
        torch_q, dtype=input_a_dtype, layout=input_a_layout, device=device, memory_config=input_a_memory_config
    )
    k_tensor = ttnn.from_torch(
        torch_k,
        dtype=input_b_dtype or input_a_dtype,
        layout=input_b_layout or ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_b_memory_config or input_a_memory_config,
    )
    v_tensor = ttnn.from_torch(
        torch_v,
        dtype=input_c_dtype or input_a_dtype,
        layout=input_c_layout or ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_c_memory_config or input_a_memory_config,
    )

    # Op call - uses extracted parameters from traced configs
    # Key parameters:
    # - scale: attention scale factor (extracted from arg8)
    # - k_chunk_size: chunk size for K processing (extracted from arg9, bypasses auto-calculation)
    # - is_causal: whether to use causal masking (extracted from arg3, default True for decode)
    start_time = start_measuring_time()
    try:
        # Parse is_causal parameter (default to True for decode)
        is_causal_flag = True
        if is_causal is not None:
            if isinstance(is_causal, str):
                is_causal_flag = is_causal.lower() in ["true", "1", "yes"]
            else:
                is_causal_flag = bool(is_causal)

        # Force output to be DRAM_INTERLEAVED as operation doesn't support sharded output
        output_mem_cfg = ttnn.DRAM_MEMORY_CONFIG

        # Build operation arguments
        op_kwargs = {"is_causal": is_causal_flag, "memory_config": output_mem_cfg}

        # Add optional parameters if extracted from traced config
        if scale is not None:
            op_kwargs["scale"] = float(scale)

        if k_chunk_size is not None and k_chunk_size != "nullopt":
            # This is critical! Using extracted k_chunk_size bypasses automatic calculation
            # that was causing the k_chunk_size < 32 constraint violation
            try:
                op_kwargs["k_chunk_size"] = int(k_chunk_size)
            except (ValueError, TypeError):
                pass  # Skip if conversion fails

        output_tensor = ttnn.transformer.scaled_dot_product_attention_decode(q_tensor, k_tensor, v_tensor, **op_kwargs)
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)

        # Comparison - decode operation typically has slightly lower accuracy
        return [check_with_pcc(torch_output, output_tensor, 0.98), e2e_perf]
    except Exception as e:
        e2e_perf = stop_measuring_time(start_time)
        # If operation fails, return failure with error message
        # With extracted k_chunk_size, constraint violations should be avoided
        return [(False, f"Operation failed: {str(e)}"), e2e_perf]
