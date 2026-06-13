# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Pure-PyTorch reference functions for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

No TTNN imports.  Each function mirrors the HuggingFace NemotronH module that
carries the same name in the component inventory so that golden tensors can be
generated and PCC-checked against the upstream model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
# The HuggingFace NemotronHModel uses a plain nn.Embedding:
#   self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
# and in forward:
#   inputs_embeds = self.embeddings(input_ids)   # -> [B, S, hidden_size]
#
# Weight stored in checkpoint as: backbone.embeddings.weight
# Shape: [vocab_size=131072, hidden_size=2688], dtype bfloat16
# ---------------------------------------------------------------------------


def embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Token-embedding lookup for Nemotron-3 Nano.

    Replicates ``NemotronHModel.embeddings(input_ids)``.

    Args:
        input_ids: Long tensor of shape [B, S] — token ids in [0, vocab_size).
        weight:    Float/BF16 tensor of shape [vocab_size, hidden_size] —
                   loaded from ``backbone.embeddings.weight`` in the checkpoint.

    Returns:
        Tensor of shape [B, S, hidden_size] in the same dtype as ``weight``.
    """
    return F.embedding(input_ids, weight)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------
# Despite the component being named "LayerNorm" in the architecture inventory,
# the HuggingFace implementation (NemotronHRMSNorm) is an RMSNorm variant —
# identical to T5LayerNorm / LlamaRMSNorm.  It has a single learnable scale
# vector (weight) and no bias.
#
# Forward path from modeling_nemotron_h.py lines 733-739:
#   input_dtype = hidden_states.dtype
#   hidden_states = hidden_states.to(torch.float32)
#   variance = hidden_states.pow(2).mean(-1, keepdim=True)
#   hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#   return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)
# ---------------------------------------------------------------------------


def layer_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """RMSNorm used as the pre-block LayerNorm in Nemotron-3-Nano-30B.

    Equivalent to NemotronHRMSNorm.forward().

    Args:
        hidden_states: Input tensor of arbitrary leading dims followed by
                       hidden_size (e.g. [batch, seq, hidden_size]).
                       Supports any dtype; computation is done in float32.
        weight:        Scale vector of shape [hidden_size], float32 or bfloat16.
        eps:           Variance epsilon (default matches model config: 1e-5).

    Returns:
        Tensor with the same shape and dtype as ``hidden_states``.
    """
    input_dtype = hidden_states.dtype
    hidden_states_f32 = hidden_states.to(torch.float32)
    variance = hidden_states_f32.pow(2).mean(-1, keepdim=True)
    hidden_states_normed = hidden_states_f32 * torch.rsqrt(variance + eps)
    out = weight.to(torch.float32) * hidden_states_normed
    return out.to(input_dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
# NemotronH uses standard HF-style Rotary Position Embedding (RoPE) in its
# attention layers.  Config parameters from config.json:
#   rope_theta          = 10000
#   partial_rotary_factor = 1.0   (all head_dim dimensions are rotated)
#   head_dim            = 128
#
# The embedding follows NemotronRotaryEmbedding (copied into NemotronH):
#   inv_freq = 1 / (rope_theta ^ (2i / dim))    for i in [0, dim//2)
#   freqs    = outer(positions, inv_freq)
#   emb      = cat(freqs, freqs, dim=-1)         shape [seq_len, dim]
#   cos      = emb.cos() * attention_scaling
#   sin      = emb.sin() * attention_scaling
#
# apply_rotary_pos_emb then performs:
#   q_embed = (q * cos) + (rotate_half(q) * sin)
#   k_embed = (k * cos) + (rotate_half(k) * sin)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the second half of the last dimension into the first, negated."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_compute_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float = 10000.0,
    partial_rotary_factor: float = 1.0,
    attention_scaling: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple:
    """Compute cosine/sine position embeddings for NemotronH RoPE.

    Replicates NemotronRotaryEmbedding.forward() from modeling_nemotron.py
    (which NemotronH inherits).

    Args:
        position_ids:          Long tensor of shape [batch, seq_len] — token positions.
        head_dim:              Head dimension (128 for NemotronH-30B).
        rope_theta:            RoPE base frequency (10000 for NemotronH-30B).
        partial_rotary_factor: Fraction of head_dim to rotate (1.0 = all dims).
        attention_scaling:     Scale applied to cos/sin (1.0 for default rope_type).
        dtype:                 Output dtype for cos/sin tensors.

    Returns:
        Tuple (cos, sin) each of shape [batch, seq_len, rot_dim] in ``dtype``.
    """
    rot_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))  # [rot_dim // 2]

    # position_ids: [batch, seq_len] -> expand inv_freq to [batch, rot_dim//2, 1]
    batch, seq_len = position_ids.shape
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch, -1, 1)  # [B, rot_dim//2, 1]
    position_ids_expanded = position_ids[:, None, :].float()  # [B, 1, S]

    with torch.autocast(device_type="cpu", enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # freqs: [B, S, rot_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, rot_dim]
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int = 128,
    rope_theta: float = 10000.0,
    partial_rotary_factor: float = 1.0,
    attention_scaling: float = 1.0,
) -> tuple:
    """Apply Rotary Position Embedding to query and key tensors.

    Replicates the combined NemotronRotaryEmbedding.forward() +
    apply_rotary_pos_emb() call from NemotronH attention layers.

    Args:
        query:                 Float tensor [batch, num_heads, seq_len, head_dim].
        key:                   Float tensor [batch, num_kv_heads, seq_len, head_dim].
        position_ids:          Long tensor [batch, seq_len] — token positions.
        head_dim:              Per-head feature dimension (128 for NemotronH-30B).
        rope_theta:            RoPE base (10000 for NemotronH-30B).
        partial_rotary_factor: Fraction of head_dim rotated (1.0 for NemotronH-30B).
        attention_scaling:     Multiplicative scale for cos/sin (1.0 for default RoPE).

    Returns:
        Tuple (query_rotated, key_rotated) with the same shapes and dtype as
        the inputs.
    """
    cos, sin = rope_compute_cos_sin(
        position_ids=position_ids,
        head_dim=head_dim,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        attention_scaling=attention_scaling,
        dtype=query.dtype,
    )
    # cos/sin: [B, S, rot_dim] -> unsqueeze head dim -> [B, 1, S, rot_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    rot_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rot_dim], query[..., rot_dim:]
    k_rot, k_pass = key[..., :rot_dim], key[..., rot_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q_out = torch.cat((q_embed, q_pass), dim=-1)
    k_out = torch.cat((k_embed, k_pass), dim=-1)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Convenience nn.Module wrapper (not used by TTNN, but useful for golden gen)
# ---------------------------------------------------------------------------


class NemotronHRMSNorm(nn.Module):
    """Drop-in replacement for HuggingFace NemotronHRMSNorm for golden tensor
    generation.  Weights can be loaded directly from a HuggingFace checkpoint.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return layer_norm(hidden_states, self.weight, self.variance_epsilon)
