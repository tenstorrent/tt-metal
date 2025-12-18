# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep test for ttnn.experimental.rotary_embedding_llama operation.

This test validates the Llama-style rotary positional embedding operation used
in transformer attention layers. The operation applies position-dependent
rotation to query/key vectors.

Mathematical basis:
- Rotary embedding treats pairs of dimensions as 2D rotation
- For each pair (x_even, x_odd), applies: [cos -sin; sin cos] @ [x_even; x_odd]
- This is equivalent to complex multiplication in the frequency domain

Tensor formats:
- Input: [batch, n_heads, seq_len, head_dim] for prefill mode
- cos/sin: [1, n_heads_or_1, seq_len, head_dim] in TTNN "doubled" format
- trans_mat: [1, 1, 32, 32] fixed transformation matrix
"""

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import helper functions for proper cos/sin generation and transformation matrix
from models.tt_transformers.tt.common import (
    precompute_freqs,
    gather_cos_sin,
    get_rot_transformation_mat,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("experimental::rotary_embedding_llama", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    # Shape format for prefill: [batch, n_heads, seq_len, head_dim]
    "model_traced_sample": {
        "input_shape": [(1, 8, 128, 64)],  # batch=1, n_heads=8, seq_len=128, head_dim=64
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Invalidate test vectors that are not supported by this sweep test.

    Currently, decode mode (HEIGHT_SHARDED memory) is not supported because it requires:
    - Complex sharding setup for input/cos/sin/trans_mat
    - RotarySetup class for on-device cos/sin generation
    - Special tensor layouts

    Returns:
        Tuple of (is_invalid: bool, reason: str or None)
    """
    # Check memory config for HEIGHT_SHARDED (indicates decode mode)
    mem_config = test_vector.get("input_a_memory_config")

    # Handle ttnn.MemoryConfig object (during generation)
    if hasattr(mem_config, "memory_layout"):
        mem_layout_str = str(mem_config.memory_layout)
        if "HEIGHT_SHARDED" in mem_layout_str:
            return True, "Decode mode (HEIGHT_SHARDED) not supported - requires complex sharding setup"
    # Handle serialized dict (after JSON load)
    elif isinstance(mem_config, dict):
        data = mem_config.get("data", {})
        if data.get("memory_layout") == "HEIGHT_SHARDED":
            return True, "Decode mode (HEIGHT_SHARDED) not supported - requires complex sharding setup"

    return False, None


def apply_rotary_emb_golden(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
    """
    Golden function for rotary embedding that handles TTNN format cos/sin.

    The TTNN op expects cos/sin in "doubled" format where each frequency
    value is duplicated: [c0, c0, c1, c1, c2, c2, ...] with shape [..., head_dim].

    This golden function "un-doubles" the cos/sin to compute the correct rotation,
    then interleaves the result back.

    Args:
        x: Input tensor [batch, n_heads, seq_len, head_dim] for prefill
        cos_cache: Cos cache [..., cache_size, head_dim] (TTNN doubled format)
        sin_cache: Sin cache [..., cache_size, head_dim] (TTNN doubled format)

    Returns:
        Output tensor with same shape as x
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


def generate_cos_sin_for_prefill(seq_len: int, head_dim: int, theta: float = 10000.0) -> tuple:
    """
    Generate properly formatted cos/sin tensors for sweep test (prefill mode).

    Args:
        seq_len: Sequence length
        head_dim: Head dimension (must be even)
        theta: RoPE theta parameter (default 10000.0)

    Returns:
        Tuple of (cos, sin) tensors in TTNN format [1, 1, seq_len, head_dim]
    """
    # Compute raw frequencies using precompute_freqs
    # This returns cos/sin with shape [end, head_dim//2]
    cos_raw, sin_raw = precompute_freqs(
        head_dim,
        seq_len * 2,  # Compute extra positions for safety
        theta=theta,
        scale_factor=None,
        orig_context_len=131072,
    )

    # Gather and format for TTNN (this doubles the dimension)
    # Output shape: [1, 1, seq_len, head_dim]
    cos_gathered, sin_gathered = gather_cos_sin(torch.arange(seq_len), cos_raw, sin_raw)

    return cos_gathered, sin_gathered


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
    storage_type=None,
    *,
    device,
) -> list:
    """
    Run the rotary_embedding_llama sweep test.

    This function handles both:
    1. Traced configurations (input_shape is a dict with all tensor shapes)
    2. Sample configurations (input_shape is a simple tuple)

    Note: Decode mode (HEIGHT_SHARDED) requires regenerating vectors with invalidate_vector.
    """
    torch.manual_seed(0)

    # Determine if this is a traced config (dict) or sample config (tuple)
    is_traced_config = isinstance(input_shape, dict)

    if is_traced_config:
        # Traced configuration with explicit shapes for all inputs
        shape_a = input_shape["input_a"]  # Main input: [batch, n_heads, seq_len, head_dim]
        shape_b = input_shape["input_b"]  # cos_cache: [1, n_heads_or_1, cache_size, head_dim]
        shape_c = input_shape["input_c"]  # sin_cache: [1, n_heads_or_1, cache_size, head_dim]
        shape_d = input_shape["input_d"]  # trans_mat: [1, 1, 32, 32]
    else:
        # Sample configuration - derive shapes from input_shape
        # input_shape format: [batch, n_heads, seq_len, head_dim]
        shape_a = list(input_shape)
        batch, n_heads, seq_len, head_dim = shape_a
        # Generate cos/sin cache shapes (cache can be larger than seq_len)
        cache_size = max(seq_len, 1024)  # Use at least 1024 for cache
        shape_b = [1, 1, cache_size, head_dim]  # cos cache
        shape_c = [1, 1, cache_size, head_dim]  # sin cache
        shape_d = [1, 1, 32, 32]  # trans_mat (fixed size)

    # Extract dimensions (for prefill mode: [batch, n_heads, seq_len, head_dim])
    batch, n_heads, seq_len, head_dim = shape_a

    # For prefill mode, is_decode_mode is always False
    # Decode mode configs should be filtered out by invalidate_vector
    is_decode_mode = False

    # --- Generate Input Tensor (random) ---
    torch_input_tensor = (torch.rand(shape_a) * 2 - 1).to(torch.bfloat16)

    # --- Generate cos/sin (properly computed, not random!) ---
    if is_traced_config:
        # For traced configs, generate cos/sin that match the traced shapes
        cache_size = shape_b[2]
        cos_cache, sin_cache = generate_cos_sin_for_prefill(cache_size, head_dim)
        # Ensure shapes match traced config (handle n_heads dimension)
        if shape_b[1] != 1:
            # Broadcast cos/sin to match n_heads if needed
            cos_cache = cos_cache.expand(-1, shape_b[1], -1, -1)
            sin_cache = sin_cache.expand(-1, shape_c[1], -1, -1)
    else:
        # For sample configs, generate based on cache_size
        cache_size = shape_b[2]
        cos_cache, sin_cache = generate_cos_sin_for_prefill(cache_size, head_dim)

    # Convert to bfloat16 for consistency
    torch_cos_cache = cos_cache.to(torch.bfloat16)
    torch_sin_cache = sin_cache.to(torch.bfloat16)

    # --- Generate Transformation Matrix (exact structure, not random!) ---
    torch_trans_mat = get_rot_transformation_mat(head_dim).to(torch.bfloat16)

    # --- Compute Golden Reference Output ---
    torch_output_tensor = apply_rotary_emb_golden(
        torch_input_tensor.float(),  # Use float for golden computation
        torch_cos_cache.float(),
        torch_sin_cache.float(),
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
    if input_d_dtype is None:
        input_d_dtype = ttnn.bfloat16
    if input_d_layout is None:
        input_d_layout = ttnn.TILE_LAYOUT
    if input_d_memory_config is None:
        input_d_memory_config = ttnn.DRAM_MEMORY_CONFIG

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

    # Convert transformation matrix to TTNN
    trans_mat_tt = ttnn.from_torch(
        torch_trans_mat,
        dtype=input_d_dtype,
        layout=input_d_layout,
        device=device,
        memory_config=input_d_memory_config,
    )

    # --- Execute TTNN Operation ---
    start_time = start_measuring_time()

    if output_memory_config is not None:
        output_tensor = ttnn.experimental.rotary_embedding_llama(
            input_tensor_a,
            cos_cache_tt,
            sin_cache_tt,
            trans_mat_tt,
            is_decode_mode=is_decode_mode,
            memory_config=output_memory_config,
        )
    else:
        output_tensor = ttnn.experimental.rotary_embedding_llama(
            input_tensor_a,
            cos_cache_tt,
            sin_cache_tt,
            trans_mat_tt,
            is_decode_mode=is_decode_mode,
        )

    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # --- Check Results ---
    # Use high PCC threshold (0.999) since we have a proper golden function
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
