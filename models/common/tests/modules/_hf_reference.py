# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace reference helpers shared by the 1D and 2D module tests.

The 1D and 2D flavours of a module implement the same math on different mesh
topologies, so they must be qualified against the same reference: the
HuggingFace module itself, never a hand-written torch re-implementation of it.
This module owns that reference plumbing (wrappers, weight extraction, and the
HF <-> Meta weight layout conversions) so both test suites share one definition.
"""

from __future__ import annotations

import inspect

import torch

from models.common.utility_functions import hf_cache_layer_kv, hf_cache_num_layers, hf_dynamic_cache_from_legacy

# =============================================================================
# Rotary embeddings
# =============================================================================


class IdentityRotaryEmbedding:
    """Rotary embedding that leaves Q/K untouched (cos=1, sin=0).

    Tests whose module under test is wired with an identity rotation - the 2D
    modules delegate RoPE to a separately qualified RotarySetup - drive the HF
    reference with this so the two sides apply the same (no-op) rotation while
    every other stage stays the real HuggingFace implementation.
    """

    def __init__(self, head_dim: int):
        self.head_dim = head_dim

    def __call__(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*position_ids.shape, self.head_dim)
        return torch.ones(shape, dtype=x.dtype), torch.zeros(shape, dtype=x.dtype)


# =============================================================================
# HfAttentionWrapper
# =============================================================================


class HfAttentionWrapper:
    """
    Wrapper for HuggingFace attention modules with KV cache support.
    Provides a consistent interface for running HF attention as reference.
    """

    def __init__(self, attention, head_dim: int, rotary_emb):
        from transformers import DynamicCache

        self.attention = attention
        self.past_key_value = DynamicCache()
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self._uses_past_key_values = "past_key_values" in inspect.signature(attention.forward).parameters

    def forward(self, x: torch.Tensor, start_pos: int, mask=None):
        """Run attention forward pass using rotary_emb directly."""
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])

        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)

        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(x, position_ids)
            cache_kwargs = (
                {"past_key_values": self.past_key_value}
                if self._uses_past_key_values
                else {"past_key_value": self.past_key_value, "use_cache": True}
            )
            output, *_ = self.attention(x, position_embeddings=position_embeddings, attention_mask=mask, **cache_kwargs)
        else:
            cache_kwargs = (
                {"past_key_values": self.past_key_value}
                if self._uses_past_key_values
                else {"past_key_value": self.past_key_value, "use_cache": True}
            )
            outputs = self.attention(x, position_ids=position_ids, attention_mask=mask, **cache_kwargs)
            output = outputs[0]
            if not self._uses_past_key_values and len(outputs) > 2:
                self.past_key_value = outputs[2]
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset_cache(self):
        """Reset KV cache for new sequence."""
        from transformers import DynamicCache

        self.past_key_value = DynamicCache()

    def reset_cache_to_zeros(self, batch_size: int, n_kv_heads: int, length: int, dtype=torch.float32):
        """Reset the KV cache to `length` zero-filled positions.

        Device KV caches are allocated zero-filled, so a decode step at position
        p attends over p zero entries plus the token it just wrote. Seeding the
        reference cache the same way keeps both sides attending over identical
        history when a test decodes without prefilling first.
        """
        if length == 0:
            self.reset_cache()
            return
        zeros = torch.zeros(batch_size, n_kv_heads, length, self.head_dim, dtype=dtype)
        self.past_key_value = hf_dynamic_cache_from_legacy([(zeros.clone(), zeros.clone())])

    @property
    def cache_k(self) -> torch.Tensor:
        """Get key cache in shape [batch, seq_len, n_kv_heads, head_dim]."""
        if hf_cache_num_layers(self.past_key_value) == 0:
            return torch.zeros(0)
        # DynamicCache stores as [batch, n_heads, seq_len, head_dim]
        # Transpose to [batch, seq_len, n_heads, head_dim]
        return hf_cache_layer_kv(self.past_key_value, 0)[0].transpose(1, 2)

    @property
    def cache_v(self) -> torch.Tensor:
        """Get value cache in shape [batch, seq_len, n_kv_heads, head_dim]."""
        if hf_cache_num_layers(self.past_key_value) == 0:
            return torch.zeros(0)
        # DynamicCache stores as [batch, n_heads, seq_len, head_dim]
        # Transpose to [batch, seq_len, n_heads, head_dim]
        return hf_cache_layer_kv(self.past_key_value, 0)[1].transpose(1, 2)


# =============================================================================
# Weight extraction helpers
# =============================================================================


def reverse_permute(tensor, n_heads, dim1, dim2):
    """Convert HuggingFace Q/K weights to Meta format for RoPE compatibility.

    HuggingFace stores Q/K weights in a format optimized for their attention implementation,
    while Meta format is required for TTNN's RoPE implementation.
    """
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def reverse_permute_1d(tensor):
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
    # Phi-3 / Phi-4 ship a FUSED qkv_proj (single Linear) instead of separate q/k/v projections.
    # Split it into Q/K/V rows here so the rest of the pipeline is architecture-agnostic.
    fused_qkv = hasattr(reference_attn, "qkv_proj") and not hasattr(reference_attn, "q_proj")
    if fused_qkv:
        cfg = reference_attn.config
        _n_heads = cfg.num_attention_heads
        _n_kv = getattr(cfg, "num_key_value_heads", _n_heads)
        _hd = (
            getattr(reference_attn, "head_dim", None) or getattr(cfg, "head_dim", None) or (cfg.hidden_size // _n_heads)
        )
        _q = _n_heads * _hd
        _kv = _n_kv * _hd
        qkv_w = reference_attn.qkv_proj.weight  # (n_heads*hd + 2*n_kv*hd, dim), order Q|K|V
        wq_raw = qkv_w[:_q]  # (n_heads * head_dim, dim)
        wk_raw = qkv_w[_q : _q + _kv]  # (n_kv_heads * head_dim, dim)
        wv_raw = qkv_w[_q + _kv : _q + 2 * _kv]  # (n_kv_heads * head_dim, dim)
        wo_raw = reference_attn.o_proj.weight  # (dim, n_heads * head_dim)
    else:
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
    wq_meta = reverse_permute(wq_raw, n_heads, n_heads_times_head_dim, dim)
    wk_meta = reverse_permute(wk_raw, n_kv_heads, n_kv_heads_times_head_dim, dim)
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
        q_norm = reverse_permute_1d(reference_attn.q_norm.weight)
    if hasattr(reference_attn, "k_norm") and reference_attn.k_norm is not None:
        k_norm = reverse_permute_1d(reference_attn.k_norm.weight)

    # QKV bias (optional, e.g., for Qwen2/Qwen2.5 models)
    # Bias also needs the same chunking/concat pattern as weights
    wqkv_bias = None
    if not fused_qkv and hasattr(reference_attn.q_proj, "bias") and reference_attn.q_proj.bias is not None:
        bq_raw = reference_attn.q_proj.bias  # (n_heads * head_dim,)
        bk_raw = reference_attn.k_proj.bias  # (n_kv_heads * head_dim,)
        bv_raw = reference_attn.v_proj.bias  # (n_kv_heads * head_dim,)

        # Apply reverse_permute to Q and K biases (same as weights)
        bq_meta = reverse_permute_1d(bq_raw.view(n_heads, head_dim)).view(-1)
        bk_meta = reverse_permute_1d(bk_raw.view(n_kv_heads, head_dim)).view(-1)
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


def get_mlp_weights_from_ref_model(reference_mlp) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract w1, w2, w3 weights from a reference MLP module in TTNN layout (transposed).

    Handles both standard LLaMA-style MLPs (gate_proj, up_proj, down_proj) and
    fused gate_up_proj models (Phi-3/Phi-4).

    Returns:
        (w1, w2, w3) tensors in TTNN layout: (in_features, out_features)
    """
    if hasattr(reference_mlp, "gate_proj"):
        w1_torch = reference_mlp.gate_proj.weight.T  # (dim, hidden_dim)
        w3_torch = reference_mlp.up_proj.weight.T  # (dim, hidden_dim)
    elif hasattr(reference_mlp, "gate_up_proj"):
        # Handle models like Phi-3/Phi-4 that use fused gate_up_proj
        gate_up_weight = reference_mlp.gate_up_proj.weight
        hidden_dim = gate_up_weight.shape[0] // 2
        w1_torch = gate_up_weight[:hidden_dim, :].T  # (dim, hidden_dim)
        w3_torch = gate_up_weight[hidden_dim:, :].T  # (dim, hidden_dim)
    else:
        raise AttributeError(f"Reference MLP {type(reference_mlp)} has no gate_proj or gate_up_proj")

    w2_torch = reference_mlp.down_proj.weight.T  # (hidden_dim, dim)
    return w1_torch, w2_torch, w3_torch


# =============================================================================
# Sampling reference
# =============================================================================


def hf_valid_token_set(logits_row: torch.Tensor, k: int, p: float, temp: float) -> set:
    """Compute the set of tokens eligible under top-k / top-p / temperature filtering.

    Mirrors the pipeline inside ttnn.sampling:
      1. Temperature: divide logits by temp  (skipped if temp == 1.0)
      2. Top-k:       zero out all but top-k tokens
      3. Top-p:       zero out tokens outside the cumulative-probability nucleus

    Uses HuggingFace's LogitsWarper classes so this reference is auditable against
    the transformers library rather than a hand-rolled implementation.

    Returns the set of token ids that have finite logit after filtering - any
    sampled token MUST come from this set.
    """
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    # Warpers expect input_ids (unused here, pass None) and a [1, V] float32 scores tensor.
    scores = logits_row.float().unsqueeze(0)  # [1, V]
    if temp != 1.0:
        scores = TemperatureLogitsWarper(temperature=temp)(None, scores)
    if k > 0:
        scores = TopKLogitsWarper(top_k=k)(None, scores)
    if 0.0 < p < 1.0:
        scores = TopPLogitsWarper(top_p=p)(None, scores)
    # Tokens with -inf logit are filtered out; all others are valid candidates.
    return set(scores[0].isfinite().nonzero(as_tuple=False).squeeze(-1).tolist())
