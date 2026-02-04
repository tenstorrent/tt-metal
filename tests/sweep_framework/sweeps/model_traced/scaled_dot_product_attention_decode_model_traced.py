# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Import helper functions from unit test
from tests.tt_eager.python_api_testing.unit_testing.misc.test_scaled_dot_product_attention_decode import (
    nearest_n,
    nearest_pow_2,
    fa_rand,
    get_chunk_size,
)

TIMEOUT = 60


loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::scaled_dot_product_attention_decode", all_cases=False)


def mesh_device_fixture():
    """
    Device fixture with DispatchCoreConfig to maximize available compute cores.
    Required for operations needing large compute grids (e.g., 8x8 = 64 cores).
    """
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())
    yield (device, "Wormhole with DispatchCoreConfig")
    ttnn.close_device(device)


parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 8, 1, 64)],
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
        "input_e_dtype": [ttnn.bfloat16],
        "input_e_layout": [ttnn.TILE_LAYOUT],
        "input_e_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_b_dtype=None,
    input_c_dtype=None,  # cur_pos tensor dtype
    # curtensor layout
    # curtensor memory config
    scale=None,  # Extracted from arg8
    sliding_window_size=None,  # Extracted from arg9
    is_causal=None,  # Extracted from arg3
    program_config_compute_grid=None,  # Extracted from arg11
    program_config_q_chunk_size=None,  # Extracted from arg11
    program_config_k_chunk_size=None,  # Extracted from arg11
    compute_kernel_config_math_fidelity=None,  # Extracted from arg12
    compute_kernel_config_math_approx_mode=None,  # Extracted from arg12
    compute_kernel_config_fp32_dest_acc_en=None,  # Extracted from arg12
    compute_kernel_config_packer_l1_acc=None,  # Extracted from arg12
    *,
    device,
    **kwargs,  # Accept any extra parameters
) -> list:
    torch.manual_seed(1234)  # Match unit test seed

    # Handle both sample suite (tuple/list) and model_traced suite (dict with keys for multi-input ops)
    if isinstance(input_shape, dict):
        # Multi-input operation - extract individual shapes for Q and K
        shape_q = tuple(input_shape.get("input_a", (1, 8, 1, 64)))
        shape_k = tuple(input_shape.get("input_b", (1, 8, 2048, 64)))
    else:
        # Convert list to tuple if needed
        shape_q = tuple(input_shape) if isinstance(input_shape, list) else input_shape
        # Default K shape for sample - use larger sequence length for KV cache
        b, nh, sq, d = shape_q
        # For decode, K and V have accumulated cache, use 2048 as default cache size
        shape_k = (b, nh, 2048, d)

    # Extract dimensions following unit test pattern
    # Q shape: [1, b, nh_q, d]
    # K/V shape: [b, nh_kv, s_kv, d]
    _, b, nh_q, d = shape_q
    b_k, nh_kv, s_kv, _ = shape_k

    # Use unit test's fa_rand() for realistic attention patterns
    # This creates Gaussian + sparse outliers instead of uniform random
    Q = fa_rand(1, b, nh_q, d)
    K = fa_rand(b, nh_kv, s_kv, d)
    V = fa_rand(b, nh_kv, s_kv, d)

    # Current position (decode uses last position in cache)
    cur_pos = s_kv - 1
    start_indices = [cur_pos] * b

    # Get k_chunk_size from config or calculate it using unit test's helper
    if program_config_k_chunk_size is not None:
        k_chunk_size = int(program_config_k_chunk_size)
    else:
        # Use unit test's get_chunk_size function
        k_chunk_size = get_chunk_size(cur_pos + 1, s_kv)

    # Calculate padded_layer_len (unit test uses this for K/V slicing)
    padded_layer_len = nearest_n(cur_pos + 1, k_chunk_size)

    # Calculate padded_num_heads for attention mask
    padded_num_heads = nearest_pow_2(nearest_n(nh_q, n=32))

    # PyTorch reference - EXACTLY following unit test pattern
    # Create explicit attention mask (unit test approach)
    attn_mask = torch.zeros((b, padded_num_heads, 1, padded_layer_len))
    for i in range(b):
        start_idx = start_indices[i]
        attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min

    # Prepare Q, K, V for PyTorch SDPA
    Q_slice = Q[:, :, :nh_q, :].permute(1, 2, 0, 3)  # [1, b, nh_q, d] -> [b, nh_q, 1, d]

    # Slice K, V to padded_layer_len BEFORE GQA expansion (unit test approach)
    K_slice = K[:, :, :padded_layer_len, :]  # [b, nh_kv, padded_layer_len, d]
    V_slice = V[:, :, :padded_layer_len, :]  # [b, nh_kv, padded_layer_len, d]

    # GQA: Expand K, V heads to match Q heads if needed
    if nh_kv < nh_q and nh_q % nh_kv == 0:
        K_slice = torch.cat([K_slice[:, i : i + 1, :, :].repeat(1, nh_q // nh_kv, 1, 1) for i in range(nh_kv)], dim=1)
        V_slice = torch.cat([V_slice[:, i : i + 1, :, :].repeat(1, nh_q // nh_kv, 1, 1) for i in range(nh_kv)], dim=1)

    attn_mask_slice = attn_mask[:, :nh_q, :, :]  # [b, nh_q, 1, padded_layer_len]

    # Compute scale
    compute_scale = d**-0.5 if scale is None else float(scale)

    # PyTorch SDPA with explicit mask (unit test uses is_causal=False with explicit mask)
    torch_output_ref = torch.nn.functional.scaled_dot_product_attention(
        Q_slice,
        K_slice,
        V_slice,
        attn_mask=attn_mask_slice,
        scale=compute_scale,
        is_causal=False,  # Use explicit mask instead of implicit causal
    )  # [b, nh_q, 1, d]

    # Reshape to match TTNN output format: [b, nh_q, 1, d] -> [1, b, nh_q, d]
    torch_output = torch_output_ref.squeeze(2).unsqueeze(0)

    # Create TTNN tensors using unit test approach
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    # Q tensor: slice to actual heads (unit test does this)
    tt_Q = ttnn.as_tensor(
        Q[:, :, :nh_q],
        device=device,
        dtype=input_a_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
    )

    # K, V tensors
    tt_K = ttnn.as_tensor(
        K,
        device=device,
        dtype=input_b_dtype or input_a_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
    )
    tt_V = ttnn.as_tensor(
        V,
        device=device,
        dtype=input_c_dtype or input_a_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
    )

    # Create cur_pos tensor
    torch_cur_pos = torch.tensor(start_indices, dtype=torch.int32)
    cur_pos_tensor = ttnn.Tensor(torch_cur_pos, ttnn.int32).to(device)

    # Op call - uses extracted parameters from traced configs
    start_time = start_measuring_time()

    # Parse is_causal parameter (default to True for decode)
    is_causal_flag = True
    if is_causal is not None:
        if isinstance(is_causal, str):
            is_causal_flag = is_causal.lower() in ["true", "1", "yes"]
        else:
            is_causal_flag = bool(is_causal)

    # Force output to be DRAM_INTERLEAVED as operation doesn't support sharded output
    output_mem_cfg = ttnn.DRAM_MEMORY_CONFIG

    # Build program_config if parameters are provided (from arg11)
    program_config = None
    if all([program_config_compute_grid, program_config_q_chunk_size, program_config_k_chunk_size]):
        # Parse compute_grid if it's a list/tuple
        if isinstance(program_config_compute_grid, (list, tuple)) and len(program_config_compute_grid) == 2:
            grid = tuple(program_config_compute_grid)
        else:
            grid = (8, 8)  # Default fallback

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=int(program_config_q_chunk_size),
            k_chunk_size=int(program_config_k_chunk_size),
            exp_approx_mode=False,  # Default
        )

    # Build compute_kernel_config if parameters are provided (from arg12)
    compute_kernel_config = None
    if compute_kernel_config_math_fidelity is not None:
        # Map string fidelity to enum
        math_fidelity_map = {
            "HiFi4": ttnn.MathFidelity.HiFi4,
            "HiFi3": ttnn.MathFidelity.HiFi3,
            "HiFi2": ttnn.MathFidelity.HiFi2,
            "LoFi": ttnn.MathFidelity.LoFi,
        }
        fidelity = math_fidelity_map.get(compute_kernel_config_math_fidelity, ttnn.MathFidelity.HiFi2)

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=bool(compute_kernel_config_math_approx_mode)
            if compute_kernel_config_math_approx_mode is not None
            else False,
            fp32_dest_acc_en=bool(compute_kernel_config_fp32_dest_acc_en)
            if compute_kernel_config_fp32_dest_acc_en is not None
            else False,
            packer_l1_acc=bool(compute_kernel_config_packer_l1_acc)
            if compute_kernel_config_packer_l1_acc is not None
            else False,
        )

    # Build operation arguments
    op_kwargs = {
        "is_causal": is_causal_flag,
        "memory_config": output_mem_cfg,
        "cur_pos_tensor": cur_pos_tensor,
    }

    # Add optional parameters if extracted from traced config
    op_kwargs["scale"] = compute_scale

    if sliding_window_size is not None:
        op_kwargs["sliding_window_size"] = int(sliding_window_size)

    if program_config is not None:
        op_kwargs["program_config"] = program_config

    if compute_kernel_config is not None:
        op_kwargs["compute_kernel_config"] = compute_kernel_config

    # Run TTNN operation
    output_tensor = ttnn.transformer.scaled_dot_product_attention_decode(tt_Q, tt_K, tt_V, **op_kwargs)
    output_tensor = ttnn.to_torch(output_tensor)

    # Slice output to match Q heads (following unit test pattern)
    # Output shape: [1, b, ?, d] - slice to [1, b, nh_q, d]
    output_tensor = output_tensor[:, :, :nh_q, :]

    e2e_perf = stop_measuring_time(start_time)

    # Comparison - decode operation with unit test approach achieves high accuracy
    return [check_with_pcc(torch_output, output_tensor, 0.99), e2e_perf]
