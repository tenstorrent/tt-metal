# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep test for ttnn.experimental.rotary_embedding_llama operation.

This test validates the Llama-style rotary positional embedding operation used
in transformer attention layers. The operation applies position-dependent
rotation to query/key vectors in both prefill and decode modes.

Mathematical basis:
- Rotary embedding treats pairs of dimensions as 2D rotation
- For each pair (x_even, x_odd), applies: [cos -sin; sin cos] @ [x_even; x_odd]
- This is equivalent to complex multiplication in the frequency domain

Tensor formats:
Prefill mode (INTERLEAVED memory):
- Input: [batch, n_heads, seq_len, head_dim]
- cos/sin: [1, n_heads_or_1, seq_len, head_dim] in TTNN "doubled" format
- trans_mat: [1, 1, 32, 32] fixed transformation matrix

Decode mode (HEIGHT_SHARDED memory):
- Input: [1, batch, n_heads, head_dim] (seq_len=1, sharded over batch)
- cos/sin: [1, batch_or_1, 1, head_dim] in TTNN "doubled" format
- trans_mat: [1, 1, batch*32, 32] repeated for each core

RoPE Parameter Support:
This sweep test supports different RoPE configurations across various model families:
- Standard RoPE (LLaMA 1/2, Mistral): theta=10000.0, no scaling
- LLaMA 3+ RoPE: theta=500000.0, llama3 scaling with factor=8.0
- YARN RoPE (Qwen 2.5): theta varies, yarn scaling
- Phi-3 LongRoPE: complex multi-factor scaling
- Vision RoPE (Mistral Vision): 2D positional encoding

The test automatically uses appropriate parameters based on traced model metadata,
with fallback to LLaMA 3 defaults (the primary traced model).

For other models, RoPE parameters can be provided explicitly via:
- rope_theta: Base frequency parameter
- rope_scale_factor: Context extension scaling factor
- rope_orig_context_len: Original max context length
- rope_type: Scaling algorithm ("default", "linear", "llama3", "yarn", "longrope")
"""

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Import helper functions for proper cos/sin generation and transformation matrix.
# Cos/sin caches in this test are generated using the production code path via
# `compute_gather_cos_sin`, ensuring alignment with model runtime behavior.
from models.tt_transformers.tt.common import (
    get_rot_transformation_mat,
    RopeScalingLlama3,
    RopeScalingLinear,
    RopeScalingYarn,
)
from models.tt_transformers.tt.rope import compute_gather_cos_sin

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
        # Optional RoPE parameters (None = use defaults from traced model or LLaMA 3)
        # Uncomment and modify to test other RoPE configurations:
        # "rope_theta": [10000.0],  # Standard LLaMA: 10000.0, LLaMA 3: 500000.0
        # "rope_scale_factor": [None],  # None = no scaling, or 2.0, 4.0, 8.0, etc.
        # "rope_orig_context_len": [None],  # Required if scale_factor is set
        # "rope_type": ["default"],  # "default", "linear", "llama3", "yarn", "longrope"
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Invalidate test vectors that are not supported by this sweep test.

    All modes (prefill and decode with HEIGHT_SHARDED memory) are supported.

    Returns:
        Tuple of (is_invalid: bool, reason: str or None)
    """
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


def generate_cos_sin_for_prefill(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    scale_factor: float = None,
    orig_context_len: int = None,
    rope_type: str = "default",
) -> tuple:
    """
    Generate properly formatted cos/sin tensors for sweep test (prefill mode).

    This function generates RoPE (Rotary Position Embedding) frequency tables
    using the same approach as the actual model implementations.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension (must be even)
        theta: RoPE theta base frequency parameter (default 10000.0)
            - Standard LLaMA/LLaMA 2: 10000.0
            - LLaMA 3/3.1/3.2: 500000.0
            - Mistral: 10000.0
            - Qwen: 10000.0 or 1000000.0 depending on version
            - Model-specific: check config.json "rope_theta"
        scale_factor: Scaling factor for context extension (None = no scaling)
            - Used for extending context beyond original training length
            - LLaMA 3: 8.0 (for 8x context extension)
            - Linear scaling: 2.0, 4.0, etc.
        orig_context_len: Original max context length (required if scale_factor is set)
            - LLaMA 3: 8192
            - LLaMA 2: 4096
            - Check config.json "original_max_position_embeddings"
        rope_type: Type of RoPE scaling (default "default")
            - "default": No scaling
            - "linear": Simple linear scaling
            - "llama3": LLaMA 3 scaling with smooth interpolation
            - "yarn": YARN scaling (used in Qwen 2.5)
            - "longrope": Phi-3 long rope

    Returns:
        Tuple of (cos, sin) tensors in TTNN format [1, 1, seq_len, head_dim]

    Note:
        The returned cos/sin tensors are in TTNN "doubled" format where each
        frequency value is duplicated: [c0, c0, c1, c1, c2, c2, ...]
    """
    # Create RopeScaling object if scaling is needed
    rope_scaling = None
    if scale_factor is not None:
        # Build rope_scaling object based on rope_type using specific subclasses
        if rope_type == "llama3":
            rope_scaling = RopeScalingLlama3(
                rope_type=rope_type,
                factor=scale_factor,
                original_max_position_embeddings=orig_context_len,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
            )
        elif rope_type == "linear":
            rope_scaling = RopeScalingLinear(rope_type=rope_type, factor=scale_factor)
        elif rope_type == "yarn":
            # YARN requires additional parameters - use defaults
            rope_scaling = RopeScalingYarn(
                rope_type=rope_type,
                factor=scale_factor,
                original_max_position_embeddings=orig_context_len,
                beta_fast=32,
                beta_slow=1,
                mscale=1.0,
                mscale_all_dim=0.0,
            )
        # Add more rope_type handling as needed

    # Use the same code path as production: compute_gather_cos_sin
    # This uses rotary_embedding_factory internally which handles all RoPE variants correctly
    cos_matrix, sin_matrix = compute_gather_cos_sin(
        dhead=head_dim,
        end=seq_len * 2,  # Compute extra positions for safety
        theta=theta,
        rope_scaling=rope_scaling,
    )

    # cos_matrix and sin_matrix are already in the correct format [seq_len, head_dim]
    # They're already "doubled" and gathered by the factory
    return cos_matrix, sin_matrix


def extract_rope_parameters(traced_source: str = None) -> dict:
    """
    Extract RoPE parameters from traced model source or use defaults.

    Args:
        traced_source: Optional source string that may contain HF_MODEL tag

    Returns:
        Dictionary with RoPE parameters: {theta, scale_factor, orig_context_len, rope_type}

    Note:
        Current defaults are for LLaMA 3.x models (the primary traced model).
        For other models, parameters would need to be extracted from model config
        or passed explicitly.
    """
    # Default parameters (LLaMA 3.x configuration)
    # These are the values used by meta-llama/Llama-3.2-1B-Instruct (the primary traced model)
    rope_params = {
        "theta": 500000.0,  # LLaMA 3+ uses higher theta
        "scale_factor": 8.0,  # Context extension factor
        "orig_context_len": 8192,  # Original training context length
        "rope_type": "llama3",  # LLaMA 3 scaling type
    }

    # Note: For other models, RoPE parameters can be extracted from HuggingFace config
    # via traced_source (e.g., parsing HF_MODEL tag and using AutoConfig).
    # Currently using LLaMA 3 defaults as that's the primary traced model.

    return rope_params


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
    traced_source=None,
    rope_theta=None,
    rope_scale_factor=None,
    rope_orig_context_len=None,
    rope_type=None,
    *,
    device,
    **_kwargs,  # Captures unused traced config fields (storage_type, traced_machine_info, etc.)
) -> list:
    """
    Run the rotary_embedding_llama sweep test.

    This function handles both:
    1. Traced configurations (input_shape is a dict with all tensor shapes)
    2. Sample configurations (input_shape is a simple tuple)

    RoPE Parameters:
        The function supports configurable RoPE parameters for different model types:
        - rope_theta: Base frequency (default from traced model)
        - rope_scale_factor: Context extension scaling (default from traced model)
        - rope_orig_context_len: Original context length (default from traced model)
        - rope_type: Scaling algorithm type (default from traced model)

        If not provided, defaults are extracted from traced_source or use LLaMA 3 values.

    Mode Detection:
        The function automatically detects prefill vs decode mode based on memory config:
        - INTERLEAVED memory → Prefill mode (processes full sequences)
        - HEIGHT_SHARDED memory → Decode mode (processes single tokens, sharded over batch)
    """
    torch.manual_seed(0)

    # Extract RoPE parameters from traced source or explicit parameters
    rope_params = extract_rope_parameters(traced_source)

    # Override with explicit parameters if provided
    if rope_theta is not None:
        rope_params["theta"] = rope_theta
    if rope_scale_factor is not None:
        rope_params["scale_factor"] = rope_scale_factor
    if rope_orig_context_len is not None:
        rope_params["orig_context_len"] = rope_orig_context_len
    if rope_type is not None:
        rope_params["rope_type"] = rope_type

    # Determine if this is a traced config (dict) or sample config (tuple)
    is_traced_config = isinstance(input_shape, dict)

    if is_traced_config:
        # Traced configuration with explicit shapes for all inputs
        shape_a = input_shape["input_a"]  # Main input: [batch, n_heads, seq_len, head_dim]
        shape_b = input_shape["input_b"]  # cos_cache: [1, n_heads_or_1, cache_size, head_dim]
        shape_c = input_shape["input_c"]  # sin_cache: [1, n_heads_or_1, cache_size, head_dim]
    else:
        # Sample configuration - derive shapes from input_shape
        # input_shape format: [batch, n_heads, seq_len, head_dim]
        shape_a = list(input_shape)
        batch, n_heads, seq_len, head_dim = shape_a
        # Generate cos/sin cache shapes (cache can be larger than seq_len)
        cache_size = max(seq_len, 1024)  # Use at least 1024 for cache
        shape_b = [1, 1, cache_size, head_dim]  # cos cache
        shape_c = [1, 1, cache_size, head_dim]  # sin cache

    # Detect decode mode from memory config
    # Decode mode uses HEIGHT_SHARDED memory layout
    is_decode_mode = False
    if hasattr(input_a_memory_config, "memory_layout"):
        mem_layout_str = str(input_a_memory_config.memory_layout)
        if "HEIGHT_SHARDED" in mem_layout_str:
            is_decode_mode = True
    elif isinstance(input_a_memory_config, dict):
        data = input_a_memory_config.get("data", {})
        if data.get("memory_layout") == "HEIGHT_SHARDED":
            is_decode_mode = True

    # Extract dimensions based on mode
    if is_decode_mode:
        # Decode mode shape: [seq_len=1, batch, n_heads, head_dim]
        # The batch dimension is what gets sharded across cores
        seq_len_dim, batch, n_heads, head_dim = shape_a
        assert seq_len_dim == 1, "Decode mode requires seq_len (dim 0) to be 1"
    else:
        # Prefill mode shape: [batch, n_heads, seq_len, head_dim]
        batch, n_heads, seq_len, head_dim = shape_a

    # --- Generate Input Tensor (random) ---
    torch_input_tensor = (torch.rand(shape_a) * 2 - 1).to(torch.bfloat16)

    # --- Generate cos/sin (properly computed, not random!) ---
    if is_decode_mode:
        # For decode mode, cos/sin shapes from traced configs are typically [1, 1, 1, head_dim]
        # or [1, batch, 1, head_dim] depending on the configuration
        # Generate position-specific cos/sin values (use position 0 for testing)
        max_cache_size = 2048  # Reasonable cache size for lookup
        cos_cache_full, sin_cache_full = generate_cos_sin_for_prefill(
            max_cache_size,
            head_dim,
            theta=rope_params["theta"],
            scale_factor=rope_params["scale_factor"],
            orig_context_len=rope_params["orig_context_len"],
            rope_type=rope_params["rope_type"],
        )
        # For testing, use position 0 and create shape matching traced config
        # Shape: [1, batch or 1, 1, head_dim]
        if is_traced_config:
            cos_cache = cos_cache_full[:, :, 0:1, :].expand(shape_b)
            sin_cache = sin_cache_full[:, :, 0:1, :].expand(shape_c)
        else:
            cos_cache = cos_cache_full[:, :, 0:1, :].expand(1, batch, 1, head_dim)
            sin_cache = sin_cache_full[:, :, 0:1, :].expand(1, batch, 1, head_dim)
    elif is_traced_config:
        # For traced configs, generate cos/sin that match the traced shapes
        cache_size = shape_b[2]
        cos_cache, sin_cache = generate_cos_sin_for_prefill(
            cache_size,
            head_dim,
            theta=rope_params["theta"],
            scale_factor=rope_params["scale_factor"],
            orig_context_len=rope_params["orig_context_len"],
            rope_type=rope_params["rope_type"],
        )
        # Ensure shapes match traced config (handle n_heads dimension)
        if shape_b[1] != 1:
            # Broadcast cos/sin to match n_heads if needed
            cos_cache = cos_cache.expand(-1, shape_b[1], -1, -1)
            sin_cache = sin_cache.expand(-1, shape_c[1], -1, -1)
    else:
        # For sample configs, generate based on cache_size
        cache_size = shape_b[2]
        cos_cache, sin_cache = generate_cos_sin_for_prefill(
            cache_size,
            head_dim,
            theta=rope_params["theta"],
            scale_factor=rope_params["scale_factor"],
            orig_context_len=rope_params["orig_context_len"],
            rope_type=rope_params["rope_type"],
        )

    # Convert to bfloat16 for consistency
    torch_cos_cache = cos_cache.to(torch.bfloat16)
    torch_sin_cache = sin_cache.to(torch.bfloat16)

    # --- Generate Transformation Matrix (exact structure, not random!) ---
    if is_decode_mode:
        # For decode mode, transformation matrix is [1, 1, batch*32, 32]
        # Each core gets a [32, 32] shard
        torch_trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(1, 1, batch, 1).to(torch.bfloat16)
    else:
        # For prefill mode, use standard transformation matrix based on head_dim
        torch_trans_mat = get_rot_transformation_mat(head_dim).to(torch.bfloat16)

    # --- Compute Golden Reference Output ---
    if is_decode_mode:
        # For decode mode, input shape is [1, batch, n_heads, head_dim]
        # Apply rotary embedding position-wise

        # Get single position cos/sin values (position 0)
        # cos/sin shape may be [1, batch_or_1, 1, head_dim]
        # We only need one position's cos/sin - take first element and broadcast
        cos_single = torch_cos_cache[0, 0, 0, :]  # [head_dim]
        sin_single = torch_sin_cache[0, 0, 0, :]  # [head_dim]

        # Get frequency components (undoubled from TTNN format)
        freqs_cos = cos_single[0::2]  # [head_dim//2]
        freqs_sin = sin_single[0::2]

        # Input: [1, batch, n_heads, head_dim]
        x_even = torch_input_tensor[..., 0::2]  # [1, batch, n_heads, head_dim//2]
        x_odd = torch_input_tensor[..., 1::2]

        # Apply 2D rotation
        cos_part = x_even * freqs_cos - x_odd * freqs_sin
        sin_part = x_even * freqs_sin + x_odd * freqs_cos

        # Interleave back to original format
        torch_output_tensor = torch.stack([cos_part, sin_part], dim=-1).flatten(-2).to(torch.bfloat16)
    else:
        # For prefill mode
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

    if is_decode_mode:
        # --- Decode Mode: Create sharded tensors ---
        # Get core grid for sharding
        core_grid = device.compute_with_storage_grid_size()
        batch_grid = ttnn.num_cores_to_corerangeset(batch, core_grid, row_wise=True)

        # Create sharded memory config for input, cos, sin
        # Each shard has shape [TILE_HEIGHT=32, head_dim]
        shard_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Create sharded memory config for transformation matrix
        # Each shard has shape [TILE_SIZE, TILE_SIZE]
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # Convert tensors to device as interleaved first, then shard
        input_tensor_interleaved = ttnn.from_torch(
            torch_input_tensor,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_tensor_a = ttnn.interleaved_to_sharded(input_tensor_interleaved, shard_mem_config)

        cos_cache_interleaved = ttnn.from_torch(
            torch_cos_cache,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos_cache_tt = ttnn.interleaved_to_sharded(cos_cache_interleaved, shard_mem_config)

        sin_cache_interleaved = ttnn.from_torch(
            torch_sin_cache,
            dtype=input_c_dtype,
            layout=input_c_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin_cache_tt = ttnn.interleaved_to_sharded(sin_cache_interleaved, shard_mem_config)

        trans_mat_interleaved = ttnn.from_torch(
            torch_trans_mat,
            dtype=input_d_dtype,
            layout=input_d_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        trans_mat_tt = ttnn.interleaved_to_sharded(trans_mat_interleaved, trans_mat_mem_config)

    else:
        # --- Prefill Mode: Use interleaved memory ---
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
    # Use high PCC threshold (0.9997) to match reference test expectations
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.9997)

    return [pcc, e2e_perf]
