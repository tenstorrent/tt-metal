# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Attention1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. Attention1D class matches HuggingFace/Meta reference model
3. Attention1D correctly rejects TG/Galaxy devices
4. Sliding window attention works correctly (seq_len > window_size)

Test coverage notes:
- Paged attention: Tested via (paged_attention, chunked_prefill) parameter combinations.
- Chunked prefill: Tested via paged-chunked variant. Requires paged=True and mode="prefill".
- Variants: non-paged, paged, paged-chunked (3 combinations per test case).
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1DConfig
from models.common.utility_functions import comp_allclose, comp_pcc


def _reverse_permute(tensor, n_heads, dim1, dim2):
    """Convert HuggingFace Q/K weights to Meta format for RoPE compatibility.

    HuggingFace stores Q/K weights in a format optimized for their attention implementation,
    while Meta format is required for TTNN's RoPE implementation.
    """
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _reverse_permute_1d(tensor):
    """Convert the last dim from separate real/imaginary (r1,r2,i1,i2,...) to interleaved (r1,i1,r2,i2,...)"""
    shape = tensor.shape
    dim = shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    interleaved = torch.stack((reals, imags), dim=-1).flatten(start_dim=len(shape) - 1)
    return interleaved


def get_attention_weights_from_ref_model(
    reference_attn, num_devices: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Extract attention weights from a reference attention module in TTNN layout.

    Applies reverse_permute to Q and K weights to convert from HuggingFace format
    to Meta format, which is required for TTNN's RoPE implementation.

    Returns:
        (wqkv, wo, q_norm, k_norm, wqkv_bias) tensors in TTNN layout
    """
    # Get raw weights from HF module
    wq_raw = reference_attn.q_proj.weight  # (n_heads * head_dim, dim)
    wk_raw = reference_attn.k_proj.weight  # (n_kv_heads * head_dim, dim)
    wv_raw = reference_attn.v_proj.weight  # (n_kv_heads * head_dim, dim)
    wo_raw = reference_attn.o_proj.weight  # (dim, n_heads * head_dim)

    # Compute head_dim from weight shapes
    dim = wq_raw.shape[1]
    n_heads_times_head_dim = wq_raw.shape[0]
    n_kv_heads_times_head_dim = wk_raw.shape[0]

    # For head_dim calculation, we need n_heads. Use the ratio of Q/K sizes.
    # Q: (n_heads * head_dim, dim), K: (n_kv_heads * head_dim, dim)
    # If n_heads == n_kv_heads (no GQA), just use q shape
    # Otherwise, we need to infer from config or assume head_dim from common values
    if hasattr(reference_attn, "head_dim"):
        head_dim = reference_attn.head_dim
    elif hasattr(reference_attn, "config") and hasattr(reference_attn.config, "head_dim"):
        head_dim = reference_attn.config.head_dim
    else:
        # Common head_dim values for LLaMA models
        head_dim = 128 if n_heads_times_head_dim >= 4096 else 64

    n_heads = n_heads_times_head_dim // head_dim
    n_kv_heads = n_kv_heads_times_head_dim // head_dim

    # Apply reverse_permute to convert HF format to Meta format for RoPE compatibility
    # This transformation is critical for Q and K weights
    wq_meta = _reverse_permute(wq_raw, n_heads, n_heads_times_head_dim, dim)
    wk_meta = _reverse_permute(wk_raw, n_kv_heads, n_kv_heads_times_head_dim, dim)
    # V and O don't need permutation
    wv_meta = wv_raw
    wo_meta = wo_raw

    # Transpose to TTNN layout: (dim, out_features)
    wq = wq_meta.T  # (dim, n_heads * head_dim)
    wk = wk_meta.T  # (dim, n_kv_heads * head_dim)
    wv = wv_meta.T  # (dim, n_kv_heads * head_dim)
    wo = wo_meta.T  # (n_heads * head_dim, dim)

    # Build combined QKV weight
    # Shape: (1, 1, dim, qkv_size_per_device * num_devices)
    qkv_list = []
    for i in range(num_devices):
        wq_chunk = torch.chunk(wq, num_devices, dim=1)[i]
        wk_chunk = torch.chunk(wk, num_devices, dim=1)[i]
        wv_chunk = torch.chunk(wv, num_devices, dim=1)[i]
        qkv = torch.cat([wq_chunk, wk_chunk, wv_chunk], dim=-1)
        qkv_list.append(qkv)

    wqkv = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

    # WO weight: (1, 1, n_heads * head_dim, dim)
    wo = wo.unsqueeze(0).unsqueeze(0)

    # Q/K norm weights (optional, e.g., for Qwen models)
    # These also need reverse_permute_1d transformation
    q_norm = None
    k_norm = None
    if hasattr(reference_attn, "q_norm") and reference_attn.q_norm is not None:
        q_norm = _reverse_permute_1d(reference_attn.q_norm.weight)
    if hasattr(reference_attn, "k_norm") and reference_attn.k_norm is not None:
        k_norm = _reverse_permute_1d(reference_attn.k_norm.weight)

    # QKV bias (optional, e.g., for Qwen2/Qwen2.5 models)
    # Bias also needs the same chunking/concat pattern as weights
    wqkv_bias = None
    if hasattr(reference_attn.q_proj, "bias") and reference_attn.q_proj.bias is not None:
        bq_raw = reference_attn.q_proj.bias  # (n_heads * head_dim,)
        bk_raw = reference_attn.k_proj.bias  # (n_kv_heads * head_dim,)
        bv_raw = reference_attn.v_proj.bias  # (n_kv_heads * head_dim,)

        # Apply reverse_permute to Q and K biases (same as weights)
        bq_meta = _reverse_permute_1d(bq_raw.view(n_heads, head_dim)).view(-1)
        bk_meta = _reverse_permute_1d(bk_raw.view(n_kv_heads, head_dim)).view(-1)
        bv_meta = bv_raw  # V doesn't need permutation

        # Build combined QKV bias with chunking for multi-device
        qkv_bias_list = []
        for i in range(num_devices):
            bq_chunk = torch.chunk(bq_meta, num_devices, dim=0)[i]
            bk_chunk = torch.chunk(bk_meta, num_devices, dim=0)[i]
            bv_chunk = torch.chunk(bv_meta, num_devices, dim=0)[i]
            qkv_bias = torch.cat([bq_chunk, bk_chunk, bv_chunk], dim=-1)
            qkv_bias_list.append(qkv_bias)

        wqkv_bias = torch.cat(qkv_bias_list, dim=-1)

    return wqkv, wo, q_norm, k_norm, wqkv_bias


def get_rotary_embedding_from_ref_model(
    reference_model, seq_len: int, device: torch.device = None, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract rotary embedding cos/sin from reference model.

    Args:
        reference_model: HuggingFace model with rotary embedding
        seq_len: Sequence length for position embeddings
        device: Device to create tensors on
        dtype: Data type for cos/sin tensors (should match hidden_states dtype)

    Returns:
        (cos, sin) tensors for rotary embedding
    """
    if device is None:
        device = torch.device("cpu")

    # Get rotary embedding from the model
    if hasattr(reference_model.model, "rotary_emb"):
        rotary_emb = reference_model.model.rotary_emb
    elif hasattr(reference_model.model.layers[0].self_attn, "rotary_emb"):
        rotary_emb = reference_model.model.layers[0].self_attn.rotary_emb
    else:
        raise AttributeError("Could not find rotary embedding in reference model")

    # Generate position ids
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Get cos and sin - use matching dtype for hidden_states compatibility
    # The rotary embedding forward expects (x, position_ids) where x is used for dtype/device
    cos, sin = rotary_emb(torch.zeros(1, seq_len, dtype=dtype, device=device), position_ids)

    return cos, sin


def run_reference_attention(
    reference_attn,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run HuggingFace reference attention module with appropriate API.

    Different HF model architectures have different attention APIs. This helper
    abstracts the differences for common models.

    Args:
        reference_attn: HuggingFace attention module (e.g., LlamaAttention, Qwen2Attention)
        hidden_states: Input tensor (batch, seq_len, dim)
        position_ids: Position indices (batch, seq_len)
        cos: Cosine rotation tensor (optional, for models with external RoPE)
        sin: Sine rotation tensor (optional, for models with external RoPE)

    Returns:
        Output tensor (batch, seq_len, dim)
    """
    # Get the class name to determine the API
    class_name = reference_attn.__class__.__name__

    # Most modern HF models (Llama, Qwen2, Mistral) use this signature:
    # forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
    #         output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None)

    def _extract_output(result):
        """Extract attention output from various HF return formats."""
        if isinstance(result, tuple):
            return result[0]
        return result

    # Some models (like newer Llama) pass position_embeddings (cos, sin) directly
    if cos is not None and sin is not None:
        # Try newer API with position_embeddings
        try:
            result = reference_attn(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=(cos, sin),
            )
            return _extract_output(result)
        except TypeError:
            pass  # Fall back to older API

    # Standard API - let model compute RoPE internally
    try:
        result = reference_attn(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        return _extract_output(result)
    except TypeError:
        # Some models have simpler API
        result = reference_attn(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
        )
        return _extract_output(result)


# ============================================================================
# Weight Caching - Avoid expensive torch.randn_like() per test
# ============================================================================

_CACHED_ATTN_WEIGHTS: dict[str, dict[str, torch.Tensor]] = {}


def _init_weight_scaled_normal(param: torch.Tensor, name: str) -> torch.Tensor:
    """Initialize a weight tensor using scaled normal distribution.

    Uses a scaling factor based on typical pretrained LLM weight statistics
    (std ~0.02) rather than pure randn (std=1.0) which can cause
    extreme activations and numerical issues in softmax.

    For 2D weights (linear layers): Uses normal with std = 0.02
    For 1D weights (biases, norms): Uses appropriate initialization
    """
    if param.dim() >= 2:
        # Linear layer weights: scaled normal initialization
        # std=0.02 matches typical pretrained transformer weights
        return torch.randn_like(param) * 0.02
    elif "norm" in name.lower() or "weight" in name.lower():
        # Norm weights (e.g., q_norm.weight, k_norm.weight): use ones
        return torch.ones_like(param)
    else:
        # Biases: use small random values
        return torch.randn_like(param) * 0.01


def _get_or_init_attn_weights(model_name: str, reference_attn) -> None:
    """Initialize attention weights once per model, cache and reuse across tests.

    Uses scaled normal initialization (std=0.02) for better numerical conditioning
    compared to pure random noise (std=1.0). This helps maintain reasonable
    activation magnitudes through the attention computation.
    """
    if model_name not in _CACHED_ATTN_WEIGHTS:
        logger.info(f"\033[33m[cache miss]\033[0m Initializing weights for {model_name}")
        _CACHED_ATTN_WEIGHTS[model_name] = {}
        with torch.no_grad():
            for name, param in reference_attn.named_parameters():
                _CACHED_ATTN_WEIGHTS[model_name][name] = _init_weight_scaled_normal(param, name)
    else:
        logger.info(f"\033[32m[cache hit]\033[0m Reusing cached weights for {model_name}")

    # Load cached weights into model
    with torch.no_grad():
        for name, param in reference_attn.named_parameters():
            param.copy_(_CACHED_ATTN_WEIGHTS[model_name][name])


# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_attention_1d_config_creation():
    """Test that Attention1DConfig dataclass can be created with explicit values."""
    from unittest.mock import MagicMock

    from models.common.modules.attention.attention_1d import Attention1DConfig

    mock_mesh_device = MagicMock()
    mock_tt_ccl = MagicMock()
    mock_wqkv = MagicMock()
    mock_wo = MagicMock()

    # Create config with explicit values
    config = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=mock_wo,
        mesh_device=mock_mesh_device,
        tt_ccl=mock_tt_ccl,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=64,
        topology=ttnn.Topology.Ring,
    )

    # Verify explicit values are preserved
    assert config.wqkv is mock_wqkv
    assert config.wo is mock_wo
    assert config.mesh_device is mock_mesh_device
    assert config.tt_ccl is mock_tt_ccl
    assert config.dim == 4096
    assert config.n_heads == 32
    assert config.n_kv_heads == 8
    assert config.head_dim == 128
    assert config.max_batch_size == 64
    assert config.topology == ttnn.Topology.Ring


def test_attention_1d_config_defaults():
    """Test that Attention1DConfig has sensible defaults."""
    from unittest.mock import MagicMock

    from models.common.modules.attention.attention_1d import Attention1DConfig

    # Minimal creation - only required fields
    config = Attention1DConfig(wqkv=MagicMock(), wo=MagicMock())

    # Check defaults
    assert config.max_batch_size == 32
    assert config.max_seq_len == 128 * 1024
    assert config.use_vllm_paged_kv_cache is False
    assert config.kv_cache_dtype == ttnn.bfloat8_b
    assert config.use_qk_fused is False
    assert config.num_reduce_scatter_links == 1
    assert config.num_all_gather_links == 2

    # Optional fields default to None
    assert config.mesh_device is None
    assert config.tt_ccl is None
    assert config.dim is None
    assert config.n_heads is None
    assert config.n_kv_heads is None
    assert config.q_norm_config is None
    assert config.k_norm_config is None


def test_attention_1d_config_power_user_overrides():
    """Test that Attention1DConfig accepts power-user overrides for program configs."""
    from unittest.mock import MagicMock

    from models.common.modules.attention.attention_1d import Attention1DConfig

    mock_prg_config = MagicMock()
    mock_mem_config = MagicMock()
    mock_compute_kernel = MagicMock()

    config = Attention1DConfig(
        wqkv=MagicMock(),
        wo=MagicMock(),
        decode_xqkv_prg_config=mock_prg_config,
        decode_sdpa_prg_config=mock_prg_config,
        decode_attn_output_prg_config=mock_prg_config,
        decode_residual_memcfg=mock_mem_config,
        li_qkv_decode_compute_kernel_cfg=mock_compute_kernel,
        activation_dtype=ttnn.bfloat16,
        scale=0.08838834764831843,  # 1/sqrt(128)
    )

    # User-provided overrides should be preserved
    assert config.decode_xqkv_prg_config is mock_prg_config
    assert config.decode_sdpa_prg_config is mock_prg_config
    assert config.decode_attn_output_prg_config is mock_prg_config
    assert config.decode_residual_memcfg is mock_mem_config
    assert config.li_qkv_decode_compute_kernel_cfg is mock_compute_kernel
    assert config.activation_dtype == ttnn.bfloat16
    assert config.scale == pytest.approx(0.08838834764831843)


# ============================================================================
# Model name constants
# ============================================================================

LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_90B = "meta-llama/Llama-3.2-90B-Vision-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-v0.1"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN25_CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEEPSEEK_R1_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
QWEN3_32B = "Qwen/Qwen3-32B"

_slow = pytest.mark.slow


# ============================================================================
# Test cases from attn_1d_performance.csv - hardcoded as pytest parameters
# ============================================================================


def _list_test_cases() -> list[pytest.param]:
    """
    Hardcoded test cases from attn_1d_performance.csv.

    Parameters: mesh_shape, seq_len, mode, x_dtype, wqkv_dtype, hf_model_name, pcc
    """
    # fmt: off
    return [
        # === Fast tests (minimal coverage set) ===
        # Single device (1x1)
        pytest.param((1, 1), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-128-1B"),
        pytest.param((1, 1), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-decode-32-1B"),
        # Dual device (1x2)
        pytest.param((1, 2), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-128-8B"),
        pytest.param((1, 2), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-decode-32-8B"),
        # Multi-device (1x8)
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-128-8B"),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-decode-32-8B"),

        # === Slow tests (full coverage from models sweep) ===
        # --- Llama-3.2-1B on N150 (1x1) ---
        pytest.param((1, 1), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-1024-1B", marks=_slow),
        pytest.param((1, 1), 2048, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-2048-1B", marks=_slow),
        pytest.param((1, 1), 4096, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-4096-1B", marks=_slow),
        pytest.param((1, 1), 8192, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-8192-1B", marks=_slow),
        pytest.param((1, 1), 16384, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-16384-1B", marks=_slow),
        pytest.param((1, 1), 32768, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-32768-1B", marks=_slow),

        # --- Llama-3.2-3B on N150 (1x1) ---
        pytest.param((1, 1), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-128-3B", marks=_slow),
        pytest.param((1, 1), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-1024-3B", marks=_slow),
        pytest.param((1, 1), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-decode-32-3B", marks=_slow),

        # --- Llama-3.1-8B on N150 (1x1) ---
        pytest.param((1, 1), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-128-8B", marks=_slow),
        pytest.param((1, 1), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-1024-8B", marks=_slow),
        pytest.param((1, 1), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-decode-32-8B", marks=_slow),

        # --- Mistral-7B on N150 (1x1) ---
        pytest.param((1, 1), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-128-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-decode-32-Mistral-7B", marks=_slow),

        # --- Llama-3.2-1B on N300 (1x2) ---
        pytest.param((1, 2), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-128-1B", marks=_slow),
        pytest.param((1, 2), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-1024-1B", marks=_slow),
        pytest.param((1, 2), 2048, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-2048-1B", marks=_slow),
        pytest.param((1, 2), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-decode-32-1B", marks=_slow),

        # --- Llama-3.2-3B on N300 (1x2) ---
        pytest.param((1, 2), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-128-3B", marks=_slow),
        pytest.param((1, 2), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-1024-3B", marks=_slow),
        pytest.param((1, 2), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-decode-32-3B", marks=_slow),

        # --- Llama-3.1-8B on N300 (1x2) ---
        pytest.param((1, 2), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-1024-8B", marks=_slow),
        pytest.param((1, 2), 2048, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-2048-8B", marks=_slow),

        # --- Llama-3.2-11B on N300 (1x2) ---
        pytest.param((1, 2), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-prefill-128-11B", marks=_slow),
        # NOTE: 11B 1024 prefill has lower PCC (0.9845) due to vision model complexity
        pytest.param((1, 2), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x2-prefill-1024-11B", marks=_slow),
        pytest.param((1, 2), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-decode-32-11B", marks=_slow),

        # --- Mistral-7B on N300 (1x2) ---
        pytest.param((1, 2), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-128-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-decode-32-Mistral-7B", marks=_slow),

        # --- Qwen2-7B on N300 (1x2) ---
        # NOTE: Qwen2-7B has Q/K biases causing numerical precision issues
        pytest.param((1, 2), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN2_7B, 0.98, id="1x2-prefill-128-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN2_7B, 0.97, id="1x2-prefill-1024-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, QWEN2_7B, 0.99, id="1x2-decode-32-Qwen2-7B", marks=_slow),

        # --- Qwen2.5-7B on N300 (1x2) ---
        # NOTE: Qwen2.5-7B has large Q/K biases causing numerical precision issues
        pytest.param((1, 2), 128, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.98, id="1x2-prefill-128-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.97, id="1x2-prefill-1024-Qwen2.5-7B", marks=_slow),
        # NOTE: Qwen2.5-7B has lower PCC for prefill+decode due to Q/K biases + RoPE interaction.
        # TTTv1's test_attention.py also shows ~0.984 min PCC. With 128-token prefill, accumulated
        # numerical error in SDPA over the larger KV cache causes further degradation.
        # See models/common/tests/modules/attention/low_pcc_notes.md for detailed analysis
        pytest.param((1, 2), 32, "decode", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.95, id="1x2-decode-32-Qwen2.5-7B", marks=_slow),

        # --- DeepSeek-R1-14B on N300 (1x2) ---
        pytest.param((1, 2), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-prefill-128-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-prefill-1024-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-decode-32-DeepSeek-R1-14B", marks=_slow),

        # --- Llama-3.2-1B on T3K (1x8) ---
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-128-1B", marks=_slow),
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-1024-1B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-decode-32-1B", marks=_slow),

        # --- Llama-3.2-3B on T3K (1x8) ---
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-128-3B", marks=_slow),
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-1024-3B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-decode-32-3B", marks=_slow),

        # --- Llama-3.1-8B on T3K (1x8) ---
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-1024-8B", marks=_slow),
        pytest.param((1, 8), 2048, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-2048-8B", marks=_slow),

        # --- Llama-3.2-11B on T3K (1x8) ---
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-prefill-128-11B", marks=_slow),
        # NOTE: 11B 1024 prefill has lower PCC (0.9844) due to vision model complexity
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x8-prefill-1024-11B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-decode-32-11B", marks=_slow),

        # --- Llama-3.3-70B on T3K (1x8) ---
        # NOTE: 70B has slightly lower PCC (0.997) due to model size and multi-device communication
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_70B, 0.99, id="1x8-prefill-128-70B", marks=_slow),
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_70B, 0.99, id="1x8-prefill-1024-70B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_70B, 0.97, id="1x8-decode-32-70B", marks=_slow),

        # --- Llama-3.2-90B on T3K (1x8) ---
        # NOTE: 90B has slightly lower PCC (0.995-0.996) due to model size
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_90B, 0.99, id="1x8-prefill-128-90B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_90B, 0.99, id="1x8-decode-32-90B", marks=_slow),

        # --- Mistral-7B on T3K (1x8) ---
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-128-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-decode-32-Mistral-7B", marks=_slow),

        # --- Mixtral-8x7B on T3K (1x8) ---
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MIXTRAL_8X7B, 0.99, id="1x8-prefill-128-Mixtral-8x7B", marks=_slow),
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MIXTRAL_8X7B, 0.99, id="1x8-prefill-1024-Mixtral-8x7B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, MIXTRAL_8X7B, 0.99, id="1x8-decode-32-Mixtral-8x7B", marks=_slow),

        # --- Qwen2.5-72B on T3K (1x8) ---
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-prefill-128-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-prefill-1024-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-decode-32-Qwen2.5-72B", marks=_slow),

        # --- Qwen2.5-Coder-32B on T3K (1x8) ---
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-128-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-1024-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-decode-32-Qwen2.5-Coder-32B", marks=_slow),
        # BF16 weights
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_CODER_32B, 0.99, id="1x8-prefill-128-Qwen2.5-Coder-32B-bf16", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat16, QWEN25_CODER_32B, 0.99, id="1x8-decode-32-Qwen2.5-Coder-32B-bf16", marks=_slow),

        # --- Qwen3-32B on T3K (1x8) ---
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-128-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-1024-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-decode-32-Qwen3-32B", marks=_slow),
        # BF16 weights
        pytest.param((1, 8), 128, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN3_32B, 0.99, id="1x8-prefill-128-Qwen3-32B-bf16", marks=_slow),
        pytest.param((1, 8), 32, "decode", ttnn.bfloat16, ttnn.bfloat16, QWEN3_32B, 0.99, id="1x8-decode-32-Qwen3-32B-bf16", marks=_slow),
    ]
    # fmt: on


# ============================================================================
# Integration Tests - Require device
# ============================================================================


def _run_prefill_test(
    tt_model,
    reference_wrapper,
    ttnn_mesh_device,
    mesh_shape,
    batch_size,
    seq_len,
    dim,
    n_heads,
    n_kv_heads,
    head_dim,
    act_dtype,
    rope_theta,
    rope_scaling,
    transformation_mats,
    freqs_cis,
    paged_attention,
    chunked_prefill,
    chunk_size,
    paged_attention_config,
    page_table,
    page_table_tt,
    reverse_permutation,
    pcc,
):
    """Run prefill test and compare against HuggingFace reference."""
    from models.tt_transformers.tt.rope import get_rot_mats

    pt_attention_input = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    # Prepare TT input for prefill
    tt_input = ttnn.from_torch(
        pt_attention_input.unsqueeze(0),  # [1, batch, seq_len, dim]
        device=ttnn_mesh_device,
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # Get rot_mats for prefill (cos/sin matrices)
    if chunked_prefill:
        # Chunked prefill: process sequence in chunks
        # Compute full rotation matrices once (covering all positions 0 to seq_len)
        from models.tt_transformers.tt.rope import compute_gather_cos_sin

        full_cos, full_sin = compute_gather_cos_sin(
            dhead=head_dim,
            end=2 * seq_len,
            theta=rope_theta,
            rope_scaling=rope_scaling,
        )

        num_chunks = seq_len // chunk_size
        tt_outputs_chunked = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size

            # Extract chunk input
            pt_chunk = pt_attention_input[:, chunk_start:chunk_end, :]

            # Prepare TT input for this chunk
            tt_chunk = ttnn.from_torch(
                pt_chunk.unsqueeze(0),
                device=ttnn_mesh_device,
                dtype=act_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
            )

            # Chunk page table: subset of pages for this chunk
            block_size = paged_attention_config.block_size
            chunk_page_table_pt = page_table[:, chunk_start // block_size : chunk_end // block_size]
            chunk_page_table_tt = ttnn.from_torch(
                chunk_page_table_pt,
                device=ttnn_mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
            )

            # Slice rotation matrices for this chunk's positions
            chunk_cos = full_cos[:, :, chunk_start:chunk_end, :]
            chunk_sin = full_sin[:, :, chunk_start:chunk_end, :]

            chunk_cos_tt = ttnn.from_torch(
                chunk_cos,
                device=ttnn_mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=act_dtype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
            )
            chunk_sin_tt = ttnn.from_torch(
                chunk_sin,
                device=ttnn_mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=act_dtype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
            )
            chunk_rot_mats = [chunk_cos_tt, chunk_sin_tt]

            # Run chunked prefill
            tt_chunk_out = tt_model.forward(
                tt_chunk,
                None,
                chunk_rot_mats,
                mode="prefill",
                page_table=page_table_tt,
                chunk_page_table=chunk_page_table_tt,
                chunk_start_idx=chunk_start,
            )

            # Collect output
            tt_chunk_out_torch = to_torch_auto_compose(tt_chunk_out)
            tt_chunk_output = tt_chunk_out_torch[:, 0:1, :chunk_size, :dim].view(batch_size, chunk_size, dim)
            tt_outputs_chunked.append(tt_chunk_output)

        # Concatenate all chunk outputs
        tt_output_torch = torch.cat(tt_outputs_chunked, dim=1)
    else:
        # Standard prefill: single forward pass
        rot_mats = get_rot_mats(
            head_dim=head_dim,
            device=ttnn_mesh_device,
            seq_len=seq_len,
            theta=rope_theta,
            rope_scaling=rope_scaling,
        )

        # Run TT model - verify forward pass executes without error
        tt_out = tt_model.forward(
            tt_input,
            None,  # current_pos not used in prefill
            rot_mats,
            mode="prefill",
            page_table=page_table_tt,
        )

        # Convert output to torch and verify shape
        tt_out = to_torch_auto_compose(tt_out)
        tt_output_torch = tt_out[:, 0:1, :seq_len, :dim].view(batch_size, seq_len, dim)
    ttnn.SetDefaultDevice(None)

    # Verify output shape and content
    assert tt_output_torch.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Expected shape {(batch_size, seq_len, dim)}, got {tt_output_torch.shape}"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN values"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf values"

    # Run reference HuggingFace attention using HfAttentionWrapper
    freqs_cis_slice = freqs_cis[:seq_len]
    with torch.no_grad():
        reference_output = reference_wrapper(pt_attention_input, start_pos=0, freqs_cis_i=freqs_cis_slice, mask=None)

    # Compare TT output with reference using PCC
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch.to(reference_output.dtype), pcc)
    logger.info(f"  PCC comparison: {pcc_message}")
    logger.info(comp_allclose(reference_output, tt_output_torch.to(reference_output.dtype)))
    assert passing, f"Prefill PCC failed: {pcc_message} (expected >= {pcc})"

    # Verify KV cache was correctly populated
    # This ensures prefill correctly writes K/V values to cache
    if not chunked_prefill:
        # Get reference KV cache from HfAttentionWrapper
        # cache_k/cache_v are in shape [batch_size, seq_len, n_kv_heads, head_dim]
        ref_k_cache = reference_wrapper.cache_k[:, :seq_len, :, :]  # [batch, seq, n_kv_heads, head_dim]
        ref_v_cache = reference_wrapper.cache_v[:, :seq_len, :, :]

        # Get TT KV cache - extraction differs for paged vs non-paged
        if paged_attention:
            # Paged attention: cache is stored in blocks that need to be reordered
            # TT paged cache shape: [num_blocks, n_kv_heads, block_size, head_dim]
            block_size = paged_attention_config.block_size
            max_num_blocks = paged_attention_config.max_num_blocks

            tt_k_cache_raw = ttnn.to_torch(
                tt_model.kv_cache[0],
                mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, dims=(0, 1), mesh_shape=mesh_shape),
            )
            tt_v_cache_raw = ttnn.to_torch(
                tt_model.kv_cache[1],
                mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, dims=(0, 1), mesh_shape=mesh_shape),
            )

            # Reorder blocks using reverse_permutation to get virtual order
            # Then reshape to [batch, n_kv_heads, seq, head_dim]
            def extract_paged_cache(cache_raw):
                # cache_raw shape: [num_blocks, n_kv_heads, block_size, head_dim]
                cache_reordered = cache_raw[reverse_permutation][:, :n_kv_heads, :, :head_dim]
                # Reshape: [num_blocks] -> [batch, blocks_per_batch]
                blocks_per_batch = max_num_blocks // batch_size
                cache_reshaped = cache_reordered.reshape(batch_size, blocks_per_batch, n_kv_heads, block_size, head_dim)
                # Transpose to [batch, n_kv_heads, blocks_per_batch, block_size, head_dim]
                cache_transposed = cache_reshaped.transpose(1, 2)
                # Flatten blocks: [batch, n_kv_heads, total_seq, head_dim]
                cache_flat = cache_transposed.reshape(batch_size, n_kv_heads, -1, head_dim)
                # Slice to actual seq_len
                return cache_flat[:, :, :seq_len, :]

            tt_k_cache = extract_paged_cache(tt_k_cache_raw)
            tt_v_cache = extract_paged_cache(tt_v_cache_raw)
        else:
            # Non-paged attention: cache is stored contiguously
            # TT cache shape: [batch_size, n_kv_heads, max_seq_len, head_dim]
            tt_k_cache = ttnn.to_torch(
                tt_model.kv_cache[0],
                mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, dims=(0, 1), mesh_shape=mesh_shape),
            )[:batch_size, :n_kv_heads, :seq_len, :head_dim]

            tt_v_cache = ttnn.to_torch(
                tt_model.kv_cache[1],
                mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, dims=(0, 1), mesh_shape=mesh_shape),
            )[:batch_size, :n_kv_heads, :seq_len, :head_dim]

        # Permute reference to match TT layout: [batch, n_kv_heads, seq, head_dim]
        ref_k_cache = ref_k_cache.permute(0, 2, 1, 3)
        ref_v_cache = ref_v_cache.permute(0, 2, 1, 3)

        # Compare K cache
        passing_k, pcc_k = comp_pcc(ref_k_cache, tt_k_cache.to(ref_k_cache.dtype), pcc)
        logger.info(f"  K cache PCC: {pcc_k}")
        assert passing_k, f"K cache PCC failed: {pcc_k} (expected >= {pcc})"

        # Compare V cache
        passing_v, pcc_v = comp_pcc(ref_v_cache, tt_v_cache.to(ref_v_cache.dtype), pcc)
        logger.info(f"  V cache PCC: {pcc_v}")
        assert passing_v, f"V cache PCC failed: {pcc_v} (expected >= {pcc})"

        logger.info("  KV cache validation: PASSED")

    paged_str = "paged" if paged_attention else "non-paged"
    chunked_str = "chunked" if chunked_prefill else "non-chunked"
    logger.info(
        f"test_attention_1d_vs_reference (from_config): PASSED for mode=prefill, seq_len={seq_len}, {paged_str}, {chunked_str}"
    )
    logger.info(f"  Config: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    logger.info(f"  Output shape: {tt_output_torch.shape}, dtype: {tt_output_torch.dtype}")


def _run_decode_test(
    tt_model,
    reference_wrapper,
    ttnn_mesh_device,
    mesh_shape,
    dim,
    n_heads,
    n_kv_heads,
    head_dim,
    max_seq_len,
    act_dtype,
    rope_theta,
    rope_scaling,
    transformation_mats,
    freqs_cis,
    paged_attention,
    page_table_tt,
    pcc,
):
    """Run decode test (prefill + decode) and compare against HuggingFace reference."""
    from models.tt_transformers.tt.rope import RotarySetup, get_rot_mats

    # Decode mode - first prefill to populate KV cache, then decode
    # Use batch_size=1 for simpler PCC comparison with HuggingFace reference
    decode_batch_size = 1

    # Step 1: Prefill pass to populate KV cache (use 128 tokens as initial context)
    prefill_seq_len = 128
    pt_prefill_input = torch.randn(1, prefill_seq_len, dim, dtype=torch.bfloat16)

    tt_prefill_input = ttnn.from_torch(
        pt_prefill_input.unsqueeze(0),  # [1, 1, prefill_seq_len, dim]
        device=ttnn_mesh_device,
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    prefill_rot_mats = get_rot_mats(
        head_dim=head_dim,
        device=ttnn_mesh_device,
        seq_len=prefill_seq_len,
        theta=rope_theta,
        rope_scaling=rope_scaling,
    )

    # Run TT prefill to populate KV cache
    _ = tt_model.forward(
        tt_prefill_input,
        None,
        prefill_rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
    )

    # Run HuggingFace prefill to populate its KV cache (using wrapper)
    freqs_cis_prefill = freqs_cis[:prefill_seq_len]
    with torch.no_grad():
        _ = reference_wrapper(pt_prefill_input, start_pos=0, freqs_cis_i=freqs_cis_prefill, mask=None)

    # Step 2: Decode pass - single token
    pt_decode_input = torch.randn(decode_batch_size, 1, dim, dtype=torch.bfloat16)

    tt_decode_input = ttnn.from_torch(
        pt_decode_input.unsqueeze(0),  # [1, 1, 1, dim]
        device=ttnn_mesh_device,
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # Convert to decode input memory config
    tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)

    # Position for decode: right after prefill
    position_idxs = torch.tensor([prefill_seq_len], dtype=torch.long)

    # Create decode-specific RotarySetup
    decode_rope_setup = RotarySetup(
        ttnn_mesh_device,
        decode_batch_size,
        head_dim,
        max_seq_len,
        rope_theta,
        rope_scaling,
        use_qk_fused=False,
    )
    decode_transformation_mats = decode_rope_setup.get_both_trans_mats()

    decode_rot_mats = decode_rope_setup.get_rot_mats(position_idxs)

    current_pos = ttnn.from_torch(
        position_idxs,
        device=ttnn_mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(ttnn_mesh_device),
    )

    # Run TT decode
    tt_out = tt_model.forward(
        tt_decode_input,
        current_pos,
        decode_rot_mats,
        mode="decode",
        page_table=page_table_tt,
    )

    # Convert output to torch
    tt_out = to_torch_auto_compose(tt_out)
    ttnn.SetDefaultDevice(None)

    # Extract TT output
    tt_output_torch = tt_out[:, 0:1, :decode_batch_size, :dim].view(decode_batch_size, 1, dim)

    # Run HuggingFace decode using wrapper
    freqs_cis_decode = freqs_cis[prefill_seq_len, :].unsqueeze(0)
    with torch.no_grad():
        reference_output = reference_wrapper(
            pt_decode_input, start_pos=prefill_seq_len, freqs_cis_i=freqs_cis_decode, mask=None
        )

    # Verify output content
    assert tt_output_torch.numel() > 0, "Output is empty"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN values"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf values"

    # Compare TT output with reference using PCC
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch.to(reference_output.dtype), pcc)
    logger.info(f"  PCC comparison: {pcc_message}")
    logger.info(comp_allclose(reference_output, tt_output_torch.to(reference_output.dtype)))
    assert passing, f"Decode PCC failed: {pcc_message} (expected >= {pcc})"

    paged_str = "paged" if paged_attention else "non-paged"
    logger.info(
        f"test_attention_1d_vs_reference (from_config): PASSED for mode=decode, decode_batch_size={decode_batch_size}, {paged_str}"
    )
    logger.info(f"  Config: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    logger.info(f"  Output shape: {tt_output_torch.shape}, dtype: {tt_output_torch.dtype}")


@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape,seq_len,mode,act_dtype,wqkv_dtype,hf_model_name,pcc",
    _list_test_cases(),
)
@pytest.mark.parametrize(
    "paged_attention,chunked_prefill",
    [
        (False, False),  # non-paged, non-chunked
        (True, False),  # paged, non-chunked
        (True, True),  # paged + chunked (chunked requires paged)
    ],
    ids=["non-paged", "paged", "paged-chunked"],
)
def test_attention_1d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mesh_shape,
    seq_len,
    mode,
    act_dtype,
    wqkv_dtype,
    hf_model_name,
    pcc,
    paged_attention,
    chunked_prefill,
):
    """
    Test Attention1D constructed via from_config (direct API) executes successfully.

    This test uses the TTTv2 pattern:
    1. Load HF config directly via AutoConfig.from_pretrained
    2. Extract weights from HF reference model
    3. Create LazyWeight objects
    4. Build Attention1DConfig with explicit parameters
    5. Create Attention1D via from_config (NOT from_model_args)
    6. Run forward pass and verify execution

    Note: Full numerical comparison against HF reference is done in
    test_attention_1d_vs_reference_from_model_args which uses existing infrastructure
    that handles HF API differences.
    """
    from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
    from models.common.modules.tt_ccl import TT_CCL
    from models.tt_transformers.tt.rope import RotarySetup

    # Skip if mesh_shape doesn't match device
    device_shape = (ttnn_mesh_device.shape[0], ttnn_mesh_device.shape[1])
    if device_shape != mesh_shape:
        pytest.skip(f"Test requires {mesh_shape} mesh, got {device_shape}")

    # Chunked prefill only applies to prefill mode
    if chunked_prefill and mode != "prefill":
        pytest.skip("Chunked prefill only applies to prefill mode")

    # Chunked prefill requires seq_len > chunk_size (128 minimum)
    chunk_size = 128  # Fixed chunk size for testing
    if chunked_prefill and seq_len <= chunk_size:
        pytest.skip(f"Chunked prefill requires seq_len > chunk_size ({chunk_size})")

    # For decode mode, use batch_size=1 for simpler HF reference comparison
    # For prefill mode, batch_size=1 and seq_len is the sequence length
    # Note: seq_len parameter for decode tests is ignored (kept for parameterization compatibility)
    batch_size = 1
    max_seq_len = max(2048, seq_len * 2) if mode == "prefill" else 2048
    num_devices = ttnn_mesh_device.get_num_devices()

    # Seed for reproducibility
    seed = 1234
    torch.manual_seed(seed)

    # Load HF config directly (no ModelArgs)
    hf_config = AutoConfig.from_pretrained(hf_model_name)

    # Handle multimodal models (Mllama) which have nested text_config
    is_multimodal = hasattr(hf_config, "text_config") and hf_config.text_config is not None
    # todo)) for multimodal models, hf_config = text_config
    if is_multimodal:
        # For Mllama and similar multimodal models, use text_config for dimensions
        text_config = hf_config.text_config
        text_config.num_hidden_layers = 1  # Only need 1 layer for testing
        dim = text_config.hidden_size
        n_heads = text_config.num_attention_heads
        n_kv_heads = getattr(text_config, "num_key_value_heads", n_heads)
        # Use explicit head_dim if available (e.g., Qwen3 models), else calculate from dim/n_heads
        # Note: some configs have head_dim=None explicitly, so we use `or` to fallback
        head_dim = getattr(text_config, "head_dim", None) or (dim // n_heads)
        rope_theta = getattr(text_config, "rope_theta", 10000.0)
        sliding_window = getattr(text_config, "sliding_window", None)
    else:
        hf_config.num_hidden_layers = 1  # Only need 1 layer for testing
        dim = hf_config.hidden_size
        n_heads = hf_config.num_attention_heads
        n_kv_heads = getattr(hf_config, "num_key_value_heads", n_heads)
        # Use explicit head_dim if available (e.g., Qwen3 models), else calculate from dim/n_heads
        # Note: some configs have head_dim=None explicitly, so we use `or` to fallback
        head_dim = getattr(hf_config, "head_dim", None) or (dim // n_heads)
        rope_theta = getattr(hf_config, "rope_theta", 10000.0)
        sliding_window = getattr(hf_config, "sliding_window", None)

    # Load HF model structure without weights, then initialize with random weights
    # This avoids slow network downloads and ensures reproducible deterministic testing
    if is_multimodal:
        # For multimodal models, import and use the specific model class
        from transformers import MllamaForConditionalGeneration

        with no_init_weights():
            # MllamaForConditionalGeneration uses _from_config (internal method) instead of from_config
            hf_model = MllamaForConditionalGeneration._from_config(hf_config, torch_dtype=torch.bfloat16)
        # Mllama has layers directly at language_model.layers (not language_model.model.layers)
        first_layer = hf_model.language_model.layers[0]
        rotary_emb = getattr(hf_model.language_model, "rotary_emb", None)
    else:
        with no_init_weights():
            hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
        first_layer = hf_model.model.layers[0]
        rotary_emb = getattr(hf_model.model, "rotary_emb", None)

    # Get reference attention from first layer
    reference_attn = first_layer.self_attn

    # Initialize attention weights deterministically (cached for speed across test cases)
    _get_or_init_attn_weights(hf_model_name, reference_attn)

    # Wrap in HfAttentionWrapper for consistent KV cache and RoPE handling
    from models.tt_transformers.tt.model_config import HfAttentionWrapper

    reference_wrapper = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)

    # Extract attention weights in TTNN layout
    wqkv_torch, wo_torch, q_norm_torch, k_norm_torch, wqkv_bias_torch = get_attention_weights_from_ref_model(
        reference_attn, num_devices
    )

    # Create LazyWeights with caching enabled
    # Cache keys include model name to ensure uniqueness across different models
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/attention_1d"))
    model_cache_prefix = hf_model_name.replace("/", "_").replace("-", "_")

    # QKV weight: shard on dim=-1
    lazy_wqkv = LazyWeight(
        source=wqkv_torch,
        dtype=wqkv_dtype,
        cache_dir_weight_name=(cache_dir, f"{model_cache_prefix}_layer0_wqkv"),
    )

    # WO weight: shard on dim=-2
    lazy_wo = LazyWeight(
        source=wo_torch,
        dtype=wqkv_dtype,
        cache_dir_weight_name=(cache_dir, f"{model_cache_prefix}_layer0_wo"),
    )

    # Q/K norm configs (optional) - using RMSNorm1DConfig composition pattern
    q_norm_config = None
    k_norm_config = None
    if q_norm_torch is not None:
        lazy_q_norm = LazyWeight(
            source=q_norm_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            cache_dir_weight_name=(cache_dir, f"{model_cache_prefix}_layer0_q_norm"),
        )
        q_norm_config = RMSNorm1DConfig(
            weight=lazy_q_norm,
            mesh_device=ttnn_mesh_device,
            eps=1e-5,
            decode_in_sharded=False,  # Q/K heads are interleaved
            decode_out_sharded=False,
            prefill_distributed=False,
        )
    if k_norm_torch is not None:
        lazy_k_norm = LazyWeight(
            source=k_norm_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            cache_dir_weight_name=(cache_dir, f"{model_cache_prefix}_layer0_k_norm"),
        )
        k_norm_config = RMSNorm1DConfig(
            weight=lazy_k_norm,
            mesh_device=ttnn_mesh_device,
            eps=1e-5,
            decode_in_sharded=False,  # Q/K heads are interleaved
            decode_out_sharded=False,
            prefill_distributed=False,
        )

    # Create TT_CCL for multi-device
    tt_ccl = TT_CCL(ttnn_mesh_device) if num_devices > 1 else None

    # Determine topology
    if num_devices == 1:
        topology = None
    elif num_devices == 2:
        topology = ttnn.Topology.Linear
    else:
        topology = ttnn.Topology.Ring

    # Handle rope_scaling using proper Pydantic models
    rope_scaling = None
    if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
        from models.tt_transformers.tt.common import rope_scaling_model_factory

        rope_scaling_dict = dict(hf_config.rope_scaling)
        rope_scaling = rope_scaling_model_factory(rope_scaling_dict, hf_config.max_position_embeddings)

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        ttnn_mesh_device,
        batch_size,
        head_dim,
        max_seq_len,
        rope_theta,
        rope_scaling,
        use_qk_fused=False,  # Use non-fused for testing
    )

    # Precompute freqs_cis for HfAttentionWrapper reference
    from models.tt_transformers.tt.common import precompute_freqs

    cos, sin = precompute_freqs(
        head_dim,
        max_seq_len * 2,
        rope_theta,
        rope_scaling.factor if rope_scaling else None,
        rope_scaling.original_max_position_embeddings if rope_scaling else None,
        rope_scaling.rope_type.value if rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)
    transformation_mats = rope_setup.get_both_trans_mats()

    # Setup paged attention config and page table if enabled
    paged_attention_config = None
    page_table_tt = None
    page_table = None
    reverse_permutation = None

    if paged_attention:
        from models.tt_transformers.tt.common import PagedAttentionConfig

        # Paged attention parameters
        block_size = 64
        max_num_blocks = max(128, (max_seq_len // block_size) * batch_size * 2)  # Ensure enough blocks

        paged_attention_config = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

        # Create page table: random permutation simulates block allocation
        permutation = torch.randperm(max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(batch_size, max_num_blocks // batch_size)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

    # Build Attention1DConfig with explicit parameters
    # Note: kv_cache is auto-created by config resolution if not using paged attention
    config = Attention1DConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        q_norm_config=q_norm_config,
        k_norm_config=k_norm_config,
        wqkv_bias=LazyWeight(source=wqkv_bias_torch) if wqkv_bias_torch is not None else None,
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        scale=head_dim**-0.5,
        sliding_window=sliding_window,
        use_qk_fused=False,
        # Note: use_vllm_paged_kv_cache=False means kv_cache is managed internally (auto-created).
        # paged_attention_config controls whether the cache uses paged layout.
        use_vllm_paged_kv_cache=False,
        paged_attention_config=paged_attention_config,
        wqkv_dtype=wqkv_dtype,
        wo_dtype=wqkv_dtype,
        activation_dtype=act_dtype,
    )

    # Create Attention1D via from_config (TTTv2 pattern)
    tt_model = Attention1D.from_config(config)

    # Verify config is properly resolved
    assert tt_model.config.is_resolved(), "Config should be resolved after from_config"
    assert tt_model.config.dim == dim
    assert tt_model.config.n_heads == n_heads
    assert tt_model.config.n_kv_heads == n_kv_heads
    assert tt_model.config.head_dim == head_dim

    if mode == "prefill":
        _run_prefill_test(
            tt_model=tt_model,
            reference_wrapper=reference_wrapper,
            ttnn_mesh_device=ttnn_mesh_device,
            mesh_shape=mesh_shape,
            batch_size=batch_size,
            seq_len=seq_len,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            act_dtype=act_dtype,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            transformation_mats=transformation_mats,
            freqs_cis=freqs_cis,
            paged_attention=paged_attention,
            chunked_prefill=chunked_prefill,
            chunk_size=chunk_size,
            paged_attention_config=paged_attention_config,
            page_table=page_table,
            page_table_tt=page_table_tt,
            reverse_permutation=reverse_permutation,
            pcc=pcc,
        )
    else:
        _run_decode_test(
            tt_model=tt_model,
            reference_wrapper=reference_wrapper,
            ttnn_mesh_device=ttnn_mesh_device,
            mesh_shape=mesh_shape,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            act_dtype=act_dtype,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            transformation_mats=transformation_mats,
            freqs_cis=freqs_cis,
            paged_attention=paged_attention,
            page_table_tt=page_table_tt,
            pcc=pcc,
        )


@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),  # single device
        (1, 2),  # 1D mesh, 2 devices
        (1, 8),  # 1D mesh, 8 devices
    ],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_attention_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that Attention1D class created via from_model_args matches reference model.

    This test validates backward compatibility with the ModelArgs factory method
    and performs numerical PCC comparison against the reference attention.
    """
    from models.common.modules.attention.attention_1d import Attention1D
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.common import precompute_freqs
    from models.tt_transformers.tt.model_config import ModelArgs
    from models.tt_transformers.tt.rope import RotarySetup, get_rot_mats

    # Use HF_MODEL env var if set, otherwise use appropriate default based on device count
    # Multi-device requires larger models (dim >= 4096) for proper sharding
    num_devices = ttnn_mesh_device.get_num_devices()
    env_model = os.environ.get("HF_MODEL")

    if not env_model:
        # Set default model based on device configuration
        if num_devices == 1:
            # Small model works for single device
            default_model = "meta-llama/Llama-3.2-1B"
        else:
            # Multi-device needs larger model with dim >= 4096
            default_model = "meta-llama/Llama-3.1-8B-Instruct"
        os.environ["HF_MODEL"] = default_model
        logger.info(f"HF_MODEL not set, using default: {default_model}")

    dtype = ttnn.bfloat8_b
    pcc = 0.98
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=batch_size, max_seq_len=2048, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("Attention1D test only runs on non-TG devices")

    # Verify model dimensions are sufficient for multi-device
    if num_devices > 1 and model_args.dim < 4096:
        pytest.skip(f"Model dim={model_args.dim} too small for {num_devices} devices - use 8B+ models")

    state_dict = model_args.load_state_dict()

    # Load reference attention model
    first_layer_prefix = model_args.get_state_dict_prefix("Attention", 0) + "."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }
    reference_model = model_args.reference_attention()
    reference_model.load_state_dict(partial_state_dict)

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        ttnn_mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.use_qk_fused,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Precompute freqs_cis for reference model
    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
        model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
        model_args.rope_scaling.rope_type.value if model_args.rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)

    # Cache path
    def topology_aware_cache_path(dtype):
        if model_args.instruct:
            return model_args.model_cache_path / {
                ttnn.bfloat16: f"tensor_cache_instruct_bf16_{ttnn_mesh_device.shape}",
                ttnn.bfloat8_b: f"tensor_cache_instruct_bfp8_{ttnn_mesh_device.shape}",
            }.get(dtype, f"tensor_cache_instruct_{ttnn_mesh_device.shape}")
        return model_args.model_cache_path / {
            ttnn.bfloat16: f"tensor_cache_bf16_{ttnn_mesh_device.shape}",
            ttnn.bfloat8_b: f"tensor_cache_bfp8_{ttnn_mesh_device.shape}",
        }.get(dtype, f"tensor_cache_{ttnn_mesh_device.shape}")

    weight_cache_path = topology_aware_cache_path(dtype)

    # Create TT_CCL for multi-device
    tt_ccl = TT_CCL(ttnn_mesh_device) if num_devices > 1 else None

    # Create Attention1D via from_model_args
    tt_model = Attention1D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        layer_num=0,
        transformation_mats=transformation_mats,
        use_paged_kv_cache=False,
    )

    # Verify the model was created successfully
    assert tt_model is not None
    assert tt_model.config.is_resolved()
    assert tt_model.config.dim == model_args.dim
    assert tt_model.config.n_heads == model_args.n_heads
    assert tt_model.config.n_kv_heads == model_args.n_kv_heads

    logger.info(f"test_attention_1d_vs_reference_from_model_args: Testing mode={mode}, seq_len={seq_len}")

    if mode == "prefill":
        # Prefill mode test
        pt_attention_input = torch.randn(
            batch_size, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
        )

        # Prepare TT input
        tt_input = ttnn.from_torch(
            pt_attention_input.unsqueeze(0),  # [1, batch, seq_len, dim]
            device=ttnn_mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                ttnn_mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape
            ),
        )

        # Get rot_mats for prefill
        rot_mats = get_rot_mats(
            head_dim=model_args.head_dim,
            device=ttnn_mesh_device,
            seq_len=seq_len,
            theta=model_args.rope_theta,
            rope_scaling=model_args.rope_scaling,
        )

        # Run TT model
        tt_out = tt_model.forward(
            tt_input,
            None,  # current_pos not used in prefill
            rot_mats,
            mode="prefill",
        )

        # Convert TT output to torch
        tt_out = to_torch_auto_compose(tt_out)
        tt_output_torch = tt_out[:, 0:1, :seq_len, : model_args.dim].view(batch_size, seq_len, model_args.dim)

        # Run reference model
        freqs_cis_slice = freqs_cis[:seq_len]
        reference_output = reference_model(pt_attention_input, start_pos=0, freqs_cis_i=freqs_cis_slice, mask=None)

        # Compare with PCC
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch.to(reference_output.dtype), pcc)
        logger.info(f"  PCC comparison: {pcc_message}")
        logger.info(comp_allclose(reference_output, tt_output_torch.to(reference_output.dtype)))
        assert passing, f"Prefill PCC failed: {pcc_message} (expected >= {pcc})"

        logger.info(f"test_attention_1d_vs_reference_from_model_args: PASSED for mode={mode}, seq_len={seq_len}")

    else:
        # Decode mode test - requires prefill first to populate KV cache
        prefill_seq_len = 128
        pt_prefill_input = torch.randn(
            batch_size,
            prefill_seq_len,
            model_args.dim,
            dtype=get_ref_model_dype(reference_model, model_args.model_name),
        )

        # Prefill TT model
        tt_prefill_input = ttnn.from_torch(
            pt_prefill_input.unsqueeze(0),
            device=ttnn_mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                ttnn_mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape
            ),
        )

        prefill_rot_mats = get_rot_mats(
            head_dim=model_args.head_dim,
            device=ttnn_mesh_device,
            seq_len=prefill_seq_len,
            theta=model_args.rope_theta,
            rope_scaling=model_args.rope_scaling,
        )

        _ = tt_model.forward(
            tt_prefill_input,
            None,
            prefill_rot_mats,
            user_id=0,
            mode="prefill",
        )

        # Prefill reference model (to populate its KV cache state)
        freqs_cis_prefill = freqs_cis[:prefill_seq_len]
        _ = reference_model(pt_prefill_input, start_pos=0, freqs_cis_i=freqs_cis_prefill, mask=None)

        # Decode pass
        current_pos = torch.tensor([prefill_seq_len])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
        )

        pt_decode_input = torch.randn(
            batch_size, 1, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
        )

        # Prepare decode input for TT
        attention_input = model_args.prepare_residual_tensor_decode(
            pt_decode_input.clone(),
            model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            force_replicated=True,
        )

        decode_rot_mats = rope_setup.get_rot_mats(current_pos)

        # Run TT decode
        tt_out = tt_model.forward(
            attention_input,
            current_pos_tensor,
            decode_rot_mats,
            mode="decode",
        )

        # Convert TT output
        tt_out = to_torch_auto_compose(tt_out)
        tt_output_torch = tt_out[:, 0:1, :batch_size, : model_args.dim].view(batch_size, 1, model_args.dim)

        # Run reference decode
        freqs_cis_i = freqs_cis[prefill_seq_len, :].unsqueeze(0)
        reference_output = reference_model(
            pt_decode_input, start_pos=prefill_seq_len, freqs_cis_i=freqs_cis_i, mask=None
        )

        # Compare with PCC
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch.to(reference_output.dtype), pcc)
        logger.info(f"  PCC comparison: {pcc_message}")
        logger.info(comp_allclose(reference_output, tt_output_torch.to(reference_output.dtype)))
        assert passing, f"Decode PCC failed: {pcc_message} (expected >= {pcc})"

        logger.info(f"test_attention_1d_vs_reference_from_model_args: PASSED for mode={mode}, seq_len={seq_len}")


def test_attention_1d_rejects_galaxy():
    """Test that Attention1D.from_model_args rejects Galaxy/TG devices."""
    from unittest.mock import MagicMock

    from models.common.modules.attention.attention_1d import Attention1D

    # Mock args with is_galaxy = True
    mock_args = MagicMock()
    mock_args.is_galaxy = True

    with pytest.raises(ValueError, match="cannot be used for Galaxy"):
        Attention1D.from_model_args(
            mesh_device=MagicMock(),
            tt_ccl=MagicMock(),
            args=mock_args,
            state_dict={},
            weight_cache_path=None,
            layer_num=0,
            transformation_mats={},
        )


@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=True,
)
@pytest.mark.parametrize(
    "sliding_window,seq_len,pcc",
    [
        pytest.param(64, 128, 0.99, id="sw64-seq128"),
        pytest.param(64, 256, 0.99, id="sw64-seq256"),
        pytest.param(128, 256, 0.99, id="sw128-seq256"),
    ],
)
def test_attention_1d_sliding_window(
    ttnn_mesh_device: ttnn.MeshDevice,
    sliding_window: int,
    seq_len: int,
    pcc: float,
):
    """
    Test Attention1D with sliding window attention.

    This test explicitly verifies that sliding window masking works correctly by:
    1. Using seq_len > sliding_window to ensure the window is actually applied
    2. Creating a reference implementation with sliding window causal mask
    3. Comparing TT output against the masked reference

    The sliding window limits attention to the last `sliding_window` tokens,
    which affects which KV entries are attended to during SDPA.
    """
    from models.common.auto_compose import to_torch_auto_compose
    from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
    from models.common.modules.lazy_weight import LazyWeight
    from models.tt_transformers.tt.common import precompute_freqs
    from models.tt_transformers.tt.rope import RotarySetup

    # Use Llama-3.2-1B as base model (fast to load, no sliding window by default)
    hf_model_name = LLAMA_1B

    mesh_shape = tuple(ttnn_mesh_device.shape)
    num_devices = ttnn_mesh_device.get_num_devices()
    batch_size = 1
    max_seq_len = max(512, seq_len * 2)

    torch.manual_seed(42)

    # Load HuggingFace model
    hf_config = AutoConfig.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch.bfloat16)
    reference_attn = hf_model.model.layers[0].self_attn
    rotary_emb = getattr(hf_model.model, "rotary_emb", None)

    dim = hf_config.hidden_size
    n_heads = hf_config.num_attention_heads
    n_kv_heads = hf_config.num_key_value_heads
    head_dim = dim // n_heads
    rope_theta = getattr(hf_config, "rope_theta", 10000.0)

    # Get weights from reference model
    wqkv_torch, wo_torch, _, _, _ = get_attention_weights_from_ref_model(reference_attn, num_devices)

    # Create TT model with sliding window
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lazy_wqkv = LazyWeight(source=wqkv_torch, dtype=ttnn.bfloat8_b, cache_dir_weight_name=None)
    lazy_wo = LazyWeight(source=wo_torch, dtype=ttnn.bfloat8_b, cache_dir_weight_name=None)

    # Note: kv_cache is auto-created by config resolution
    config = Attention1DConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        mesh_device=ttnn_mesh_device,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        scale=head_dim**-0.5,
        sliding_window=sliding_window,  # Enable sliding window
        use_vllm_paged_kv_cache=False,
        activation_dtype=ttnn.bfloat16,
    )

    tt_model = Attention1D.from_config(config)

    # Setup RoPE
    rope_setup = RotarySetup(ttnn_mesh_device, batch_size, head_dim, max_seq_len, rope_theta, None, use_qk_fused=False)
    from models.tt_transformers.tt.rope import get_rot_mats

    rot_mats = get_rot_mats(
        head_dim=head_dim,
        device=ttnn_mesh_device,
        seq_len=seq_len,
        theta=rope_theta,
        rope_scaling=None,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Prepare input
    pt_input = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        pt_input.unsqueeze(0),  # [1, batch, seq_len, dim]
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # Run TT model with sliding window
    tt_out = tt_model.forward(
        tt_input,
        None,  # current_pos not used in prefill
        rot_mats,
        mode="prefill",
    )

    tt_out_torch = to_torch_auto_compose(tt_out)
    tt_output = tt_out_torch[:, 0:1, :seq_len, :dim].view(batch_size, seq_len, dim)

    # Create reference with sliding window causal mask
    # Sliding window mask: position i can only attend to positions max(0, i - sliding_window + 1) to i
    def create_sliding_window_mask(seq_len: int, sliding_window: int) -> torch.Tensor:
        """Create a sliding window causal attention mask for HuggingFace attention."""
        # HF expects mask shape: (batch, 1, seq_len, seq_len) where 0 = attend, -inf = mask
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bfloat16)

        for i in range(seq_len):
            # Position i can attend to [max(0, i - sliding_window + 1), i]
            for j in range(seq_len):
                if j > i:
                    # Causal: can't attend to future
                    mask[i, j] = torch.finfo(torch.bfloat16).min
                elif j < i - sliding_window + 1:
                    # Sliding window: can't attend beyond window
                    mask[i, j] = torch.finfo(torch.bfloat16).min

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    sliding_mask = create_sliding_window_mask(seq_len, sliding_window)

    # Setup reference attention
    cos, sin = precompute_freqs(head_dim, max_seq_len * 2, rope_theta, None, None, "llama3")
    freqs_cis = torch.complex(cos, sin)[:seq_len]

    from models.tt_transformers.tt.model_config import HfAttentionWrapper

    reference_wrapper = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)

    # Run reference WITH sliding window mask
    ref_output_sw = reference_wrapper(pt_input, start_pos=0, freqs_cis_i=freqs_cis, mask=sliding_mask)

    # Also run reference WITHOUT sliding window (full attention) for comparison
    # Need a fresh wrapper since it caches KV
    reference_wrapper_full = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)
    ref_output_full = reference_wrapper_full(pt_input, start_pos=0, freqs_cis_i=freqs_cis, mask=None)

    ttnn.SetDefaultDevice(None)

    # Basic sanity checks
    assert not torch.isnan(tt_output).any(), "TT output contains NaN"
    assert not torch.isinf(tt_output).any(), "TT output contains Inf"
    assert tt_output.shape == (batch_size, seq_len, dim), f"Shape mismatch: {tt_output.shape}"

    logger.info(f"Sliding window test: sw={sliding_window}, seq_len={seq_len}")

    # Compare TT output (with sliding window) against HF reference (with sliding window mask)
    passing_sw, pcc_sw = comp_pcc(ref_output_sw, tt_output.to(ref_output_sw.dtype), pcc)
    logger.info(f"  PCC TT vs HF (both with sliding window): {pcc_sw}")
    assert passing_sw, f"Sliding window PCC failed: {pcc_sw} (expected >= {pcc})"

    # Verify sliding window actually has an effect by comparing to full attention
    if seq_len > sliding_window:
        # The sliding window reference should differ from full attention
        ref_diff = (ref_output_sw - ref_output_full).abs().mean()
        logger.info(f"  HF sliding window vs full attention diff: {ref_diff:.6f}")
        assert ref_diff > 1e-4, "HF sliding window mask had no effect on reference"

        # TT output should also differ from full attention
        tt_diff = (tt_output - ref_output_full).abs().mean()
        logger.info(f"  TT sliding window vs HF full attention diff: {tt_diff:.6f}")
        assert tt_diff > 1e-4, "TT sliding window had no effect"

        # The TT-vs-reference-sliding-window PCC should be higher than TT-vs-full-attention
        _, pcc_full = comp_pcc(ref_output_full, tt_output.to(ref_output_full.dtype), 0.0)
        logger.info(f"  PCC TT (sliding) vs HF (full): {pcc_full}")

        # Compare PCC values
        pcc_sw_val = float(pcc_sw)
        pcc_full_val = float(pcc_full)
        assert pcc_sw_val > pcc_full_val, (
            f"TT sliding window should match HF sliding window better than HF full attention. "
            f"PCC(TT,HF_sw)={pcc_sw_val:.4f} should be > PCC(TT,HF_full)={pcc_full_val:.4f}"
        )

    logger.info(f"test_attention_1d_sliding_window: PASSED (sw={sliding_window}, seq_len={seq_len})")


# ============================================================================
# Benchmark Tests - Fused vs Non-Fused QK Operations
# ============================================================================


def _create_attention_model_for_benchmark(
    ttnn_mesh_device: ttnn.MeshDevice,
    hf_model_name: str,
    use_qk_fused: bool,
    paged_attention: bool = True,
    max_seq_len: int = 2048,
) -> tuple:
    """
    Create an Attention1D model with specified use_qk_fused setting for benchmarking.

    Returns:
        (tt_model, reference_wrapper, config_params) tuple for running benchmarks
    """

    from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from models.tt_transformers.tt.rope import RotarySetup

    batch_size = 1
    mesh_shape = ttnn_mesh_device.shape

    # Load HF config
    hf_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)

    # Extract dimensions
    text_config = getattr(hf_config, "text_config", hf_config)
    dim = text_config.hidden_size
    n_heads = text_config.num_attention_heads
    n_kv_heads = getattr(text_config, "num_key_value_heads", n_heads)
    # Use explicit head_dim if available (e.g., Qwen3 models), else calculate from dim/n_heads
    head_dim = getattr(text_config, "head_dim", None) or (dim // n_heads)

    # Calculate num_devices for topology
    num_devices = mesh_shape[0] * mesh_shape[1]

    # RoPE parameters
    rope_theta = getattr(text_config, "rope_theta", 10000.0)
    rope_scaling = None
    if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
        from models.tt_transformers.tt.common import rope_scaling_model_factory

        rope_scaling_dict = dict(hf_config.rope_scaling)
        rope_scaling = rope_scaling_model_factory(rope_scaling_dict, hf_config.max_position_embeddings)

    # Topology
    topology = ttnn.Topology.Ring if num_devices > 1 else None
    tt_ccl = TT_CCL(ttnn_mesh_device) if num_devices > 1 else None

    # Create reference model with random weights
    is_multimodal = "vision_config" in hf_config.__dict__ or "Vision" in hf_model_name

    if is_multimodal:
        from transformers import MllamaForConditionalGeneration

        with no_init_weights():
            hf_model = MllamaForConditionalGeneration._from_config(hf_config, torch_dtype=torch.bfloat16)
        # Mllama has layers directly at language_model.layers (not language_model.model.layers)
        first_layer = hf_model.language_model.layers[0]
        rotary_emb = getattr(hf_model.language_model, "rotary_emb", None)
    else:
        with no_init_weights():
            hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
        first_layer = hf_model.model.layers[0]
        rotary_emb = getattr(hf_model.model, "rotary_emb", None)

    # Get reference attention from first layer
    reference_attn = first_layer.self_attn

    # Initialize weights with scaled normal
    _get_or_init_attn_weights(hf_model_name, reference_attn)

    # Create HF wrapper
    from models.tt_transformers.tt.model_config import HfAttentionWrapper

    reference_wrapper = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)

    # Extract and prepare weights
    wqkv_torch, wo_torch, q_norm_torch, k_norm_torch, wqkv_bias_torch = get_attention_weights_from_ref_model(
        reference_attn, num_devices
    )

    # Create LazyWeight objects WITHOUT caching
    # Note: Disable caching to ensure weights match the random HF reference weights
    # (cached weights from previous runs would mismatch current random weights)
    wqkv_dtype = ttnn.bfloat8_b
    lazy_wqkv = LazyWeight(
        source=wqkv_torch,
        dtype=wqkv_dtype,
        cache_dir_weight_name=None,  # Disable caching for benchmark
    )
    lazy_wo = LazyWeight(
        source=wo_torch,
        dtype=wqkv_dtype,
        cache_dir_weight_name=None,  # Disable caching for benchmark
    )

    # Q/K norm configs (if present)
    q_norm_config = None
    k_norm_config = None
    if q_norm_torch is not None:
        # Add 3 dimensions to match expected shape (1, 1, 1, head_dim)
        q_norm_4d = q_norm_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        q_norm_config = RMSNorm1DConfig(
            weight=LazyWeight(source=q_norm_4d, dtype=ttnn.bfloat16),
            mesh_device=ttnn_mesh_device,
            decode_in_sharded=False,  # Q/K heads are interleaved after create_qkv_heads
            decode_out_sharded=False,
            prefill_distributed=False,  # Q/K norm doesn't need distributed prefill
        )
    if k_norm_torch is not None:
        # Add 3 dimensions to match expected shape (1, 1, 1, head_dim)
        k_norm_4d = k_norm_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        k_norm_config = RMSNorm1DConfig(
            weight=LazyWeight(source=k_norm_4d, dtype=ttnn.bfloat16),
            mesh_device=ttnn_mesh_device,
            decode_in_sharded=False,  # Q/K heads are interleaved after create_qkv_heads
            decode_out_sharded=False,
            prefill_distributed=False,  # Q/K norm doesn't need distributed prefill
        )

    # Paged attention config
    paged_attention_config = None
    if paged_attention:
        paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=2048)

    # RotarySetup for transformation mats
    rope_setup = RotarySetup(
        ttnn_mesh_device,
        batch_size,
        head_dim,
        max_seq_len,
        rope_theta,
        rope_scaling,
        use_qk_fused=use_qk_fused,  # Use the parameterized value
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Precompute freqs_cis
    from models.tt_transformers.tt.common import precompute_freqs

    cos, sin = precompute_freqs(
        head_dim,
        max_seq_len * 2,
        rope_theta,
        rope_scaling.factor if rope_scaling else None,
        rope_scaling.original_max_position_embeddings if rope_scaling else None,
        rope_scaling.rope_type.value if rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)

    # Build Attention1DConfig with specified use_qk_fused
    # Note: kv_cache is auto-created by config resolution if not using paged attention
    config = Attention1DConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        q_norm_config=q_norm_config,
        k_norm_config=k_norm_config,
        wqkv_bias=LazyWeight(source=wqkv_bias_torch) if wqkv_bias_torch is not None else None,
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        scale=head_dim**-0.5,
        use_qk_fused=use_qk_fused,  # KEY: this is what we're benchmarking
        use_vllm_paged_kv_cache=False,  # kv_cache managed internally
        paged_attention_config=paged_attention_config,
        wqkv_dtype=wqkv_dtype,
        wo_dtype=wqkv_dtype,
        activation_dtype=ttnn.bfloat16,
    )

    # Create Attention1D
    tt_model = Attention1D.from_config(config)

    config_params = {
        "dim": dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "batch_size": batch_size,
        "mesh_shape": mesh_shape,
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "transformation_mats": transformation_mats,
        "freqs_cis": freqs_cis,
        "rope_setup": rope_setup,
        "is_multimodal": is_multimodal,
    }

    return tt_model, reference_wrapper, config_params


@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
def test_attention_1d_fused_qk_profiling(ttnn_mesh_device: ttnn.MeshDevice):
    """
    Comprehensive profiling test comparing fused vs non-fused QK operations across multiple models.

    This test:
    1. Benchmarks multiple model architectures
    2. Compares fused vs non-fused performance and accuracy
    3. Outputs a detailed analysis table with findings
    4. Runs on 1x1, 1x2, and 1x8 mesh configurations
    """
    import time

    from models.tt_transformers.tt.rope import RotarySetup, get_rot_mats

    mesh_shape = ttnn_mesh_device.shape
    num_devices = mesh_shape[0] * mesh_shape[1]

    # ANSI color codes
    BOLD = "\033[1m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE_BG = "\033[47m"
    BLACK = "\033[30m"

    # Models to benchmark - filter based on mesh size
    # Hardware constraints:
    # - nlp_create_qkv_heads_decode supports max 32 q heads per device
    # - DeepSeek-R1-14B has 40 q heads, needs 1x2+ (40/2=20 per device)
    # - Large models (70B, 90B, 72B) need 1x8 for memory
    if num_devices == 1:
        models_to_test = [
            (LLAMA_1B, "Llama-1B"),
            (LLAMA_3B, "Llama-3B"),
            (LLAMA_8B, "Llama-8B"),
            (LLAMA_11B, "Llama-11B-Vision"),
            (MISTRAL_7B, "Mistral-7B"),
            (QWEN2_7B, "Qwen2-7B"),
            (QWEN25_7B, "Qwen2.5-7B"),
            # Note: DeepSeek-R1-14B has 40 q heads, exceeds 32 head limit on single device
        ]
    elif num_devices == 2:
        models_to_test = [
            (LLAMA_1B, "Llama-1B"),
            (LLAMA_3B, "Llama-3B"),
            (LLAMA_8B, "Llama-8B"),
            (LLAMA_11B, "Llama-11B-Vision"),
            (MISTRAL_7B, "Mistral-7B"),
            (QWEN2_7B, "Qwen2-7B"),
            (QWEN25_7B, "Qwen2.5-7B"),
            (DEEPSEEK_R1_14B, "DeepSeek-R1-14B"),  # 40 heads / 2 devices = 20 per device, OK
        ]
    else:
        # 1x8: all models including very large ones
        # Note: Qwen2-7B and Qwen2.5-7B excluded (dim=3584 not divisible by 8)
        models_to_test = [
            (LLAMA_8B, "Llama-8B"),
            (LLAMA_11B, "Llama-11B-Vision"),
            (LLAMA_70B, "Llama-70B"),
            (LLAMA_90B, "Llama-90B-Vision"),
            (MISTRAL_7B, "Mistral-7B"),
            (MIXTRAL_8X7B, "Mixtral-8x7B"),
            (QWEN25_72B, "Qwen2.5-72B"),
            (QWEN25_CODER_32B, "Qwen2.5-Coder-32B"),
            (DEEPSEEK_R1_14B, "DeepSeek-R1-14B"),
            (QWEN3_32B, "Qwen3-32B"),
        ]

    # Collect results for all models (no printing during benchmark)
    all_results = {}
    log_messages = []  # Collect log messages to print at end

    for hf_model_name, model_label in models_to_test:
        log_messages.append(f"Benchmarking {model_label} ({hf_model_name})...")

        model_results = {}

        for use_qk_fused in [False, True]:  # Test non-fused first for comparison
            fused_label = "fused" if use_qk_fused else "non-fused"

            try:
                tt_model, reference_wrapper, params = _create_attention_model_for_benchmark(
                    ttnn_mesh_device=ttnn_mesh_device,
                    hf_model_name=hf_model_name,
                    use_qk_fused=use_qk_fused,
                    paged_attention=False,
                )

                dim = params["dim"]
                head_dim = params["head_dim"]
                batch_size = params["batch_size"]
                rope_theta = params["rope_theta"]
                rope_scaling = params["rope_scaling"]
                transformation_mats = params["transformation_mats"]
                freqs_cis = params["freqs_cis"]

                # Prefill to populate KV cache
                prefill_seq_len = 128
                pt_prefill_input = torch.randn(batch_size, prefill_seq_len, dim, dtype=torch.bfloat16)

                tt_prefill_input = ttnn.from_torch(
                    pt_prefill_input.unsqueeze(0),
                    device=ttnn_mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                )

                prefill_rot_mats = get_rot_mats(
                    head_dim=head_dim,
                    device=ttnn_mesh_device,
                    seq_len=prefill_seq_len,
                    theta=rope_theta,
                    rope_scaling=rope_scaling,
                )

                _ = tt_model.forward(
                    tt_prefill_input,
                    None,
                    prefill_rot_mats,
                    mode="prefill",
                )

                # Run HF prefill for accuracy check
                freqs_cis_prefill = freqs_cis[:prefill_seq_len]
                _ = reference_wrapper(pt_prefill_input, start_pos=0, freqs_cis_i=freqs_cis_prefill, mask=None)

                # Prepare decode
                pt_decode_input = torch.randn(batch_size, 1, dim, dtype=torch.bfloat16)
                position_idxs = torch.tensor([prefill_seq_len], dtype=torch.long)

                decode_rope_setup = RotarySetup(
                    ttnn_mesh_device,
                    batch_size,
                    head_dim,
                    2048,
                    rope_theta,
                    rope_scaling,
                    use_qk_fused=use_qk_fused,
                )
                decode_transformation_mats = decode_rope_setup.get_both_trans_mats()
                decode_rot_mats = decode_rope_setup.get_rot_mats(position_idxs)

                current_pos = ttnn.from_torch(
                    position_idxs,
                    device=ttnn_mesh_device,
                    dtype=ttnn.int32,
                    mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(ttnn_mesh_device),
                )

                # First decode for accuracy check
                tt_decode_input = ttnn.from_torch(
                    pt_decode_input.unsqueeze(0),
                    device=ttnn_mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                )
                tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)

                tt_out = tt_model.forward(
                    tt_decode_input,
                    current_pos,
                    decode_rot_mats,
                    mode="decode",
                )
                ttnn.synchronize_device(ttnn_mesh_device)

                # Check PCC
                tt_out_torch = to_torch_auto_compose(tt_out)
                tt_output = tt_out_torch[:, 0:1, :batch_size, :dim].view(batch_size, 1, dim)

                freqs_cis_decode = freqs_cis[prefill_seq_len, :].unsqueeze(0)
                reference_output = reference_wrapper(
                    pt_decode_input, start_pos=prefill_seq_len, freqs_cis_i=freqs_cis_decode, mask=None
                )

                _, pcc_value = comp_pcc(reference_output, tt_output.to(reference_output.dtype), 0.9)
                pcc = float(pcc_value)

                # Warmup
                for _ in range(5):
                    tt_decode_input = ttnn.from_torch(
                        pt_decode_input.unsqueeze(0),
                        device=ttnn_mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                    )
                    tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)
                    _ = tt_model.forward(
                        tt_decode_input,
                        current_pos,
                        decode_rot_mats,
                        mode="decode",
                    )
                    ttnn.synchronize_device(ttnn_mesh_device)

                # Timed runs
                num_runs = 100
                timings = []
                for _ in range(num_runs):
                    tt_decode_input = ttnn.from_torch(
                        pt_decode_input.unsqueeze(0),
                        device=ttnn_mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                    )
                    tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)

                    start = time.perf_counter()
                    _ = tt_model.forward(
                        tt_decode_input,
                        current_pos,
                        decode_rot_mats,
                        mode="decode",
                    )
                    ttnn.synchronize_device(ttnn_mesh_device)
                    end = time.perf_counter()
                    timings.append((end - start) * 1e6)

                avg_time = sum(timings) / len(timings)
                min_time = min(timings)
                std_time = (sum((t - avg_time) ** 2 for t in timings) / len(timings)) ** 0.5

                model_results[fused_label] = {
                    "mean": avg_time,
                    "min": min_time,
                    "std": std_time,
                    "pcc": pcc,
                }

                log_messages.append(
                    f"  {fused_label:10s}: mean={avg_time:6.1f} μs, std={std_time:5.1f} μs, PCC={pcc:.4f}"
                )

            except Exception as e:
                # Extract short error message (first line or up to 80 chars)
                error_msg = str(e).split("\n")[0][:80]
                log_messages.append(f"  [SKIP] {fused_label}: {error_msg}")
                continue

        all_results[model_label] = model_results

    # =========================================================================
    # ALL OUTPUT AT THE END - AFTER BENCHMARKING COMPLETES
    # =========================================================================

    print()
    print()
    print(f"{BOLD}{WHITE_BG}{BLACK}{'=' * 80}{RESET}")
    print(
        f"{BOLD}{WHITE_BG}{BLACK}  MESH SHAPE: {mesh_shape[0]}x{mesh_shape[1]}  ({num_devices} device{'s' if num_devices > 1 else ''})  {RESET}"
    )
    print(f"{BOLD}{WHITE_BG}{BLACK}{'=' * 80}{RESET}")
    print()

    print(f"{BOLD}{CYAN}FUSED vs NON-FUSED QK COMPREHENSIVE BENCHMARK{RESET}")
    print(f"{CYAN}{'─' * 80}{RESET}")

    # Print collected log messages
    for msg in log_messages:
        if "[SKIP]" in msg:
            print(f"{RED}{msg}{RESET}")
        elif "Benchmarking" in msg:
            print(f"{YELLOW}{msg}{RESET}")
        else:
            print(msg)

    # =========================================================================
    # COMPREHENSIVE ANALYSIS OUTPUT
    # =========================================================================
    print()
    print(f"{BOLD}{MAGENTA}{'=' * 80}{RESET}")
    print(f"{BOLD}{MAGENTA}BENCHMARK ANALYSIS: FUSED vs NON-FUSED QK OPERATIONS{RESET}")
    print(f"{BOLD}{MAGENTA}  Mesh Shape: {mesh_shape[0]}x{mesh_shape[1]} ({num_devices} devices){RESET}")
    print(f"{BOLD}{MAGENTA}{'=' * 80}{RESET}")

    # Performance Comparison Table
    print()
    print(f"{BOLD}{GREEN}PERFORMANCE COMPARISON (Decode Latency){RESET}")
    print(f"{GREEN}{'─' * 80}{RESET}")
    print(f"{BOLD}{'Model':<14} │ {'Non-fused (μs)':<16} │ {'Fused (μs)':<14} │ {'Diff (μs)':<12} │ {'%':<10}{RESET}")
    print(f"{GREEN}{'─' * 80}{RESET}")

    total_diff = 0
    total_pct = 0
    count = 0
    fused_faster_count = 0
    fused_slower_count = 0

    for model_label, results in all_results.items():
        if "non-fused" in results and "fused" in results:
            non_fused = results["non-fused"]["mean"]
            fused = results["fused"]["mean"]
            diff = fused - non_fused
            pct = (diff / non_fused) * 100

            total_diff += diff
            total_pct += pct
            count += 1

            if diff < 0:
                fused_faster_count += 1
            else:
                fused_slower_count += 1

            diff_color = GREEN if diff < 0 else RED if diff > 0 else RESET
            print(
                f"{model_label:<14} │ {non_fused:>14.1f}   │ {fused:>12.1f}   │ {diff_color}{diff:>+10.1f}{RESET}   │ {diff_color}{pct:>+8.1f}%{RESET}"
            )

    print(f"{GREEN}{'─' * 80}{RESET}")
    if count > 0:
        avg_diff = total_diff / count
        avg_pct = total_pct / count
        avg_color = GREEN if avg_diff < 0 else RED if avg_diff > 0 else RESET
        print(
            f"{BOLD}{'AVERAGE':<14}{RESET} │ {'':<16} │ {'':<14} │ {avg_color}{BOLD}{avg_diff:>+10.1f}{RESET}   │ {avg_color}{BOLD}{avg_pct:>+8.1f}%{RESET}"
        )

    # Accuracy Comparison Table
    print()
    print(f"{BOLD}{BLUE}ACCURACY COMPARISON (PCC vs HuggingFace Reference){RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    print(f"{BOLD}{'Model':<14} │ {'Non-fused PCC':<16} │ {'Fused PCC':<14}{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")

    avg_pcc_nf = 0
    avg_pcc_f = 0
    pcc_count = 0

    for model_label, results in all_results.items():
        non_fused_pcc = results.get("non-fused", {}).get("pcc", "N/A")
        fused_pcc = results.get("fused", {}).get("pcc", "N/A")

        non_fused_str = f"{non_fused_pcc:.4f}" if isinstance(non_fused_pcc, float) else non_fused_pcc
        fused_str = f"{fused_pcc:.4f}" if isinstance(fused_pcc, float) else fused_pcc

        if isinstance(non_fused_pcc, float) and isinstance(fused_pcc, float):
            avg_pcc_nf += non_fused_pcc
            avg_pcc_f += fused_pcc
            pcc_count += 1

        nf_color = (
            GREEN
            if isinstance(non_fused_pcc, float) and non_fused_pcc >= 0.99
            else (
                YELLOW
                if isinstance(non_fused_pcc, float) and non_fused_pcc >= 0.95
                else RED
                if isinstance(non_fused_pcc, float)
                else RESET
            )
        )
        f_color = (
            GREEN
            if isinstance(fused_pcc, float) and fused_pcc >= 0.99
            else (
                YELLOW
                if isinstance(fused_pcc, float) and fused_pcc >= 0.95
                else RED
                if isinstance(fused_pcc, float)
                else RESET
            )
        )

        print(f"{model_label:<14} │ {nf_color}{non_fused_str:>14}{RESET}   │ {f_color}{fused_str:>12}{RESET}")

    print(f"{BLUE}{'─' * 60}{RESET}")

    if pcc_count > 0:
        avg_pcc_nf /= pcc_count
        avg_pcc_f /= pcc_count

    # Dynamic Root Cause Analysis based on results
    print()
    print(f"{BOLD}{YELLOW}ROOT CAUSE ANALYSIS{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    print()
    print(f"  {BOLD}{RED}FUSED PATH{RESET} (in decode_forward):")
    print(f"    1. {CYAN}_to_qk_fused_memory_config(){RESET} - 2x ttnn.to_memory_config calls")
    print(f"       └─ Converts Q/K to HEIGHT_SHARDED layout for fused kernel")
    print(f"    2. {CYAN}rotary_embedding_llama_fused_qk(){RESET} - 1 fused kernel")
    print(f"    3. {CYAN}paged_fused_update_cache(){RESET} - 1 fused kernel")
    print()
    print(f"  {BOLD}{GREEN}NON-FUSED PATH{RESET} (in decode_forward):")
    print(f"    1. {CYAN}rotary_embedding_llama(q){RESET} - Q rotary embedding")
    print(f"    2. {CYAN}rotary_embedding_llama(k){RESET} - K rotary embedding")
    print(f"    3. {CYAN}paged_update_cache(keys, k){RESET} - K cache update")
    print(f"    4. {CYAN}paged_update_cache(values, v){RESET} - V cache update")
    print()

    # Dynamic Conclusion based on actual results
    print(f"{BOLD}{MAGENTA}CONCLUSION (Mesh {mesh_shape[0]}x{mesh_shape[1]}){RESET}")
    print(f"{MAGENTA}{'─' * 80}{RESET}")
    print()

    if count > 0:
        # Determine overall winner
        if avg_pct < -2:
            perf_summary = f"{GREEN}Fused path is {abs(avg_pct):.1f}% FASTER{RESET} on average"
            recommendation = f"{GREEN}Prefer fused path{RESET} for better performance"
        elif avg_pct > 2:
            perf_summary = f"{RED}Fused path is {avg_pct:.1f}% SLOWER{RESET} on average"
            recommendation = f"{GREEN}Prefer non-fused path{RESET} for better performance"
        else:
            perf_summary = f"{YELLOW}Performance is similar{RESET} (within 2%)"
            recommendation = f"{YELLOW}Either path is acceptable{RESET}"

        print(f"  1. {BOLD}PERFORMANCE:{RESET} {perf_summary}")
        print(f"     - Fused faster: {GREEN}{fused_faster_count}{RESET} models")
        print(f"     - Fused slower: {RED}{fused_slower_count}{RESET} models")

        if pcc_count > 0:
            pcc_color = GREEN if avg_pcc_nf >= 0.99 and avg_pcc_f >= 0.99 else YELLOW
            print(
                f"  2. {BOLD}ACCURACY:{RESET} Both paths produce similar results ({pcc_color}PCC ~{avg_pcc_f:.3f}{RESET})"
            )

        print(f"  3. {BOLD}RECOMMENDATION:{RESET} {recommendation}")
    else:
        print(f"  {RED}No valid benchmark results collected.{RESET}")

    print()
    print(f"{BOLD}{MAGENTA}{'=' * 80}{RESET}")
