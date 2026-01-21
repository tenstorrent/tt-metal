# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep test for ttnn.experimental.rotary_embedding_llama_fused_qk operation.

This operation fuses rotary position embedding with Q/K preparation in one kernel,
optimizing memory bandwidth and reducing overhead in transformer attention layers.

The operation returns two tensors: rotated Q and rotated K.
"""

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Import helper functions for proper cos/sin generation and transformation matrix
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.rope import compute_gather_cos_sin

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("experimental::rotary_embedding_llama_fused_qk", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 8, 128, 64)],  # batch, n_heads, seq_len, head_dim
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

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def apply_rotary_emb_golden(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
    """
    Golden function for rotary embedding that handles TTNN format cos/sin.

    The TTNN op expects cos/sin in "doubled" format where each frequency
    value is duplicated: [c0, c0, c1, c1, c2, c2, ...] with shape [..., head_dim].
    """
    seq_len = x.shape[2]  # For prefill mode

    # Slice cos/sin to match seq_len (cache may be larger)
    cos = cos_cache[..., :seq_len, :]
    sin = sin_cache[..., :seq_len, :]

    # cos/sin are in TTNN "doubled" format: [c0, c0, c1, c1, ...]
    # Extract the "un-doubled" version: [c0, c1, c2, ...]
    freqs_cos = cos[..., 0::2]  # [..., seq_len, head_dim//2]
    freqs_sin = sin[..., 0::2]

    # Split input into even/odd (real/imaginary parts of complex rotation)
    x_even = x[..., 0::2]  # [batch, n_heads, seq_len, head_dim//2]
    x_odd = x[..., 1::2]

    # 2D rotation: [cos -sin; sin cos] @ [even; odd]
    cos_part = x_even * freqs_cos - x_odd * freqs_sin
    sin_part = x_even * freqs_sin + x_odd * freqs_cos

    # Interleave back to original format
    out = torch.stack([cos_part, sin_part], dim=-1).flatten(-2)
    return out


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
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations (multi-input)
    if isinstance(input_shape, dict):
        shape_a = input_shape.get("input_a", input_shape.get("q_input_tensor"))  # Q tensor
        shape_b = input_shape.get("input_b", input_shape.get("k_input_tensor"))  # K tensor
        shape_c = input_shape.get("input_c", input_shape.get("cos_cache"))  # cos cache
        shape_d = input_shape.get("input_d", input_shape.get("sin_cache"))  # sin cache
        # Need to get the 5th input - trans_mat
        shape_e = input_shape.get("input_e", input_shape.get("trans_mat"))
    else:
        # Fallback for sample configurations
        if isinstance(input_shape, (tuple, list)):
            shape = tuple(input_shape)
        else:
            shape = input_shape
        # For sample, assume standard shapes
        batch, n_heads, seq_len, head_dim = shape
        shape_a = shape  # Q input
        shape_b = shape  # K input (may have different n_heads for GQA)
        shape_c = (1, 1, seq_len, head_dim)  # cos cache
        shape_d = (1, 1, seq_len, head_dim)  # sin cache
        shape_e = (1, 1, head_dim, head_dim)  # transformation matrix

    # Check which inputs are provided
    has_input_e = kwargs.get("input_e_dtype") is not None or (
        isinstance(input_shape, dict) and "input_e" in input_shape
    )
    input_e_dtype = kwargs.get("input_e_dtype")
    input_e_layout = kwargs.get("input_e_layout")
    input_e_memory_config = kwargs.get("input_e_memory_config")

    # Extract dimensions
    batch, n_heads_q, seq_len, head_dim = shape_a
    _, n_heads_k, _, _ = shape_b

    # Generate random input tensors for Q and K
    torch_input_a = (torch.rand(shape_a) * 2 - 1).to(torch.bfloat16)
    torch_input_b = (torch.rand(shape_b) * 2 - 1).to(torch.bfloat16)

    # Generate proper cos/sin using the same code path as production
    cos_matrix, sin_matrix = compute_gather_cos_sin(
        dhead=head_dim,
        end=max(seq_len * 2, 2048),  # Compute extra positions for safety
        theta=500000.0,  # LLaMA 3 default
        rope_scaling=None,
    )

    # Match the traced cache shape
    # For fused_qk, cos/sin dim[1] must equal q_dim[1] + k_dim[1]
    cos_dim1 = shape_c[1]  # Should be n_heads_q + n_heads_k
    cache_size = shape_c[2]

    # Repeat cos/sin along dimension 1 to match the required size
    torch_input_c = cos_matrix[:, :, :cache_size, :].repeat(1, cos_dim1, 1, 1).to(torch.bfloat16)
    torch_input_d = sin_matrix[:, :, :cache_size, :].repeat(1, cos_dim1, 1, 1).to(torch.bfloat16)

    # Generate transformation matrix using production code
    torch_input_e = (
        get_rot_transformation_mat(dhead=head_dim).to(torch.bfloat16) if has_input_e or shape_e is not None else None
    )

    # Compute golden reference outputs for both Q and K
    # For fused_qk, the cos/sin have dim[1] = n_heads_q + n_heads_k
    # We need to split them for Q and K separately
    cos_q = torch_input_c[:, :n_heads_q, :, :]
    sin_q = torch_input_d[:, :n_heads_q, :, :]
    cos_k = torch_input_c[:, n_heads_q:, :, :]
    sin_k = torch_input_d[:, n_heads_q:, :, :]

    torch_output_q = apply_rotary_emb_golden(
        torch_input_a.float(),
        cos_q.float(),
        sin_q.float(),
    ).to(torch.bfloat16)

    torch_output_k = apply_rotary_emb_golden(
        torch_input_b.float(),
        cos_k.float(),
        sin_k.float(),
    ).to(torch.bfloat16)

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Use defaults for non-traced parameters
    if input_b_dtype is None:
        input_b_dtype = input_a_dtype
    if input_b_layout is None:
        input_b_layout = input_a_layout
    if input_b_memory_config is None:
        input_b_memory_config = input_a_memory_config
    if input_c_dtype is None:
        input_c_dtype = ttnn.bfloat16
    if input_c_layout is None:
        input_c_layout = ttnn.TILE_LAYOUT
    if input_c_memory_config is None:
        input_c_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if input_d_dtype is None:
        input_d_dtype = ttnn.bfloat16
    if input_d_layout is None:
        input_d_layout = ttnn.TILE_LAYOUT
    if input_d_memory_config is None:
        input_d_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Convert to ttnn tensors
    input_tensor_a = ttnn.from_torch(
        torch_input_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    input_tensor_c = ttnn.from_torch(
        torch_input_c,
        dtype=input_c_dtype,
        layout=input_c_layout,
        device=device,
        memory_config=input_c_memory_config,
    )

    input_tensor_d = ttnn.from_torch(
        torch_input_d,
        dtype=input_d_dtype,
        layout=input_d_layout,
        device=device,
        memory_config=input_d_memory_config,
    )

    # Convert transformation matrix
    if torch_input_e is not None:
        if input_e_dtype is None:
            input_e_dtype = ttnn.bfloat16
        if input_e_layout is None:
            input_e_layout = ttnn.TILE_LAYOUT
        if input_e_memory_config is None:
            input_e_memory_config = ttnn.DRAM_MEMORY_CONFIG

        input_tensor_e = ttnn.from_torch(
            torch_input_e,
            dtype=input_e_dtype,
            layout=input_e_layout,
            device=device,
            memory_config=input_e_memory_config,
        )
    else:
        # If no trans_mat, create a default one
        torch_input_e = get_rot_transformation_mat(dhead=head_dim).to(torch.bfloat16)
        input_tensor_e = ttnn.from_torch(
            torch_input_e,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    start_time = start_measuring_time()

    # rotary_embedding_llama_fused_qk returns a tuple of (Q_rotated, K_rotated)
    # API signature: (q_input_tensor, k_input_tensor, cos_cache, sin_cache, trans_mat)
    result = ttnn.experimental.rotary_embedding_llama_fused_qk(
        input_tensor_a,  # q_input_tensor
        input_tensor_b,  # k_input_tensor
        input_tensor_c,  # cos_cache
        input_tensor_d,  # sin_cache
        input_tensor_e,  # trans_mat
    )

    # The operation returns a tuple of (Q_rotated, K_rotated)
    if isinstance(result, (list, tuple)) and len(result) == 2:
        output_tensor_q = ttnn.to_torch(result[0])
        output_tensor_k = ttnn.to_torch(result[1])
    else:
        e2e_perf = stop_measuring_time(start_time)
        return [(False, f"Expected tuple of 2 tensors, got {type(result)}"), e2e_perf]

    e2e_perf = stop_measuring_time(start_time)

    # Check PCC for both Q and K outputs
    pcc_q = check_with_pcc(torch_output_q, output_tensor_q, 0.999)
    pcc_k = check_with_pcc(torch_output_k, output_tensor_k, 0.999)

    # Both must pass for the test to pass
    if pcc_q[0] and pcc_k[0]:
        pcc = (True, f"Q PCC: {pcc_q[1]}, K PCC: {pcc_k[1]}")
    else:
        pcc = (False, f"Q PCC: {pcc_q[1]}, K PCC: {pcc_k[1]}")

    return [pcc, e2e_perf]
