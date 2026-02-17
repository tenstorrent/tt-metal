# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep test for ttnn.experimental.rotary_embedding operation.

This test validates the rotary positional embedding operation used
in Falcon and other transformer models.

Mathematical basis:
- Rotary embedding treats pairs of dimensions as 2D rotation
- For each pair (x_even, x_odd), applies: [cos -sin; sin cos] @ [x_even; x_odd]

Operation signature:
    ttnn.experimental.rotary_embedding(input, cos_cache, sin_cache, token_idx=None, memory_config=None)

Inputs:
- input: [W, Z, Y, X] - typically [batch, n_heads, seq_len, head_dim]
- cos_cache: [1, 1, cache_size, X] - cosine cache in TILE layout
- sin_cache: [1, 1, cache_size, X] - sine cache in TILE layout
- token_idx: Optional token index for decode mode (default None for prefill)
"""

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("experimental::rotary_embedding", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 71, 32, 64)],  # Falcon-style: [batch=1, n_heads=71, seq_len=32, head_dim=64]
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
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def rotate_half(x):
    """Helper function for golden reference computation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx=None):
    """Golden function for rotary embedding (from unit test)."""
    seq_len = x.shape[-2]
    if token_idx is None:
        cos = cos_cached[:, :, :seq_len, ...]
        sin = sin_cached[:, :, :seq_len, ...]
    else:
        cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
        sin = sin_cached[:, :, token_idx : token_idx + 1, ...]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


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
    **kwargs,
) -> list:
    """
    Run the rotary_embedding sweep test.

    This function handles both:
    1. Traced configurations (input_shape is a dict with all tensor shapes)
    2. Sample configurations (input_shape is a simple tuple)
    """
    torch.manual_seed(0)

    # Determine if this is a traced config (dict) or sample config (tuple)
    is_traced_config = isinstance(input_shape, dict)

    if is_traced_config:
        # Traced configuration with explicit shapes for all inputs
        shape_a = input_shape["input_a"]  # Main input: [W, Z, Y, X]
        shape_b = input_shape["input_b"]  # cos_cache: [1, 1, cache_size, X]
        shape_c = input_shape["input_c"]  # sin_cache: [1, 1, cache_size, X]
    else:
        # Sample configuration - derive shapes from input_shape
        # input_shape format: [W, Z, Y, X] e.g., [batch, n_heads, seq_len, head_dim]
        shape_a = list(input_shape)
        W, Z, Y, X = shape_a
        # Generate cos/sin cache shapes (cache should be large enough)
        cache_size = max(Y, 1024)  # At least as large as seq_len
        shape_b = [1, 1, cache_size, X]  # cos cache
        shape_c = [1, 1, cache_size, X]  # sin cache

    # Extract dimensions from shape_a
    W, Z, Y, X = shape_a

    # --- Generate Random Input Tensor ---
    torch_input_tensor = (torch.rand(shape_a) * 2 - 1).to(torch.bfloat16)

    # --- Generate cos/sin cache (random for now - real implementation would compute properly) ---
    torch_cos_cache = (torch.rand(shape_b) * 2 - 1).to(torch.bfloat16)
    torch_sin_cache = (torch.rand(shape_c) * 2 - 1).to(torch.bfloat16)

    # --- Compute Golden Reference Output ---
    torch_output_tensor = apply_rotary_pos_emb(
        torch_input_tensor.float(),
        torch_cos_cache.float(),
        torch_sin_cache.float(),
        token_idx=None,  # Prefill mode (None) for traced configs
    ).to(torch.bfloat16)

    # --- Create TTNN Tensors ---
    # Use defaults for non-traced parameters
    if input_b_dtype is None:
        input_b_dtype = ttnn.bfloat16
    if input_b_layout is None:
        input_b_layout = ttnn.TILE_LAYOUT
    if input_b_memory_config is None:
        input_b_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if input_c_dtype is None:
        input_c_dtype = ttnn.bfloat16
    if input_c_layout is None:
        input_c_layout = ttnn.TILE_LAYOUT
    if input_c_memory_config is None:
        input_c_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Convert input tensor to TTNN
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    # Convert cos cache to TTNN
    cos_cache_tt = ttnn.from_torch(
        torch_cos_cache,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    # Convert sin cache to TTNN
    sin_cache_tt = ttnn.from_torch(
        torch_sin_cache,
        dtype=input_c_dtype,
        layout=input_c_layout,
        device=device,
        memory_config=input_c_memory_config,
    )

    # --- Execute TTNN Operation ---
    start_time = start_measuring_time()

    if output_memory_config is not None:
        output_tensor = ttnn.experimental.rotary_embedding(
            input_tensor_a,
            cos_cache_tt,
            sin_cache_tt,
            memory_config=output_memory_config,
        )
    else:
        output_tensor = ttnn.experimental.rotary_embedding(
            input_tensor_a,
            cos_cache_tt,
            sin_cache_tt,
        )

    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # --- Check Results ---
    # Use standard PCC threshold (0.999)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
