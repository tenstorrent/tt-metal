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
# MoEAttention
# ---------------------------------------------------------------------------
# In NemotronH the MoE-layer attention block (``NemotronHAttention``) is
# architecturally identical to the DenseAttention block.  Both use standard
# GQA (Group Query Attention) with KV-cache, RoPE, and a causal attention mask.
#
# Config for NemotronH-30B:
#   num_attention_heads   = 32  (Q heads)
#   num_key_value_heads   = 2   (KV heads)
#   head_dim              = 128
#   hidden_size           = 2688
#   rope_theta            = 10000
#   partial_rotary_factor = 1.0
#
# Weight layout (model.layers.{i}.self_attn.*):
#   q_proj.weight : [num_heads * head_dim, hidden_size]     = [4096, 2688]
#   k_proj.weight : [num_kv_heads * head_dim, hidden_size]  = [256,  2688]
#   v_proj.weight : [num_kv_heads * head_dim, hidden_size]  = [256,  2688]
#   o_proj.weight : [hidden_size, num_heads * head_dim]     = [2688, 4096]
# All biases are absent in this model.
#
# Forward (prefill, no KV cache, single sequence):
#   1. q = x @ q_proj.T                  [B, S, num_heads  * head_dim]
#   2. k = x @ k_proj.T                  [B, S, num_kv_heads * head_dim]
#   3. v = x @ v_proj.T                  [B, S, num_kv_heads * head_dim]
#   4. Reshape to multi-head layout       [B, num_heads, S, head_dim]
#   5. Apply RoPE to q, k
#   6. Expand k, v for GQA               [B, num_heads, S, head_dim]
#   7. Scaled dot-product attention       [B, num_heads, S, head_dim]
#   8. Reshape back to [B, S, hidden_size]
#   9. out = attn @ o_proj.T             [B, S, hidden_size]
# ---------------------------------------------------------------------------


def moe_attention(
    hidden_states: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int = 32,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    rope_theta: float = 10000.0,
    partial_rotary_factor: float = 1.0,
    attention_scaling: float = 1.0,
) -> torch.Tensor:
    """GQA attention block used in MoE transformer layers of NemotronH-30B.

    This function is architecturally identical to the DenseAttention reference;
    the distinction is purely in which layer index it is called on.

    Replicates ``NemotronHAttention.forward()`` (prefill / full-sequence mode,
    no KV-cache state passed in).

    Args:
        hidden_states:  [batch, seq_len, hidden_size] in bfloat16.
        wq:             Q projection weight [num_heads * head_dim, hidden_size].
        wk:             K projection weight [num_kv_heads * head_dim, hidden_size].
        wv:             V projection weight [num_kv_heads * head_dim, hidden_size].
        wo:             Output projection weight [hidden_size, num_heads * head_dim].
        position_ids:   Long tensor [batch, seq_len] — token positions.
        num_heads:      Number of query heads (32 for NemotronH-30B).
        num_kv_heads:   Number of KV heads (2 for NemotronH-30B).
        head_dim:       Head dimension (128 for NemotronH-30B).
        rope_theta:     RoPE base frequency (10000 for NemotronH-30B).
        partial_rotary_factor: Fraction of head_dim to rotate (1.0).
        attention_scaling:    Multiplicative scale for cos/sin (1.0).

    Returns:
        Tensor [batch, seq_len, hidden_size] in the same dtype as
        ``hidden_states``.
    """
    bsz, seq_len, hidden_size = hidden_states.shape
    num_groups = num_heads // num_kv_heads  # GQA expansion factor

    # --- QKV projections ---
    q = F.linear(hidden_states, wq)  # [B, S, num_heads * head_dim]
    k = F.linear(hidden_states, wk)  # [B, S, num_kv_heads * head_dim]
    v = F.linear(hidden_states, wv)  # [B, S, num_kv_heads * head_dim]

    # Reshape to multi-head layout
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, nH,  S, D]
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # [B, nKV, S, D]
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # [B, nKV, S, D]

    # --- Rotary Position Embedding ---
    cos, sin = rope_compute_cos_sin(
        position_ids=position_ids,
        head_dim=head_dim,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        attention_scaling=attention_scaling,
        dtype=hidden_states.dtype,
    )
    # cos/sin: [B, S, rot_dim] -> [B, 1, S, rot_dim] for broadcasting over heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat((q_embed, q_pass), dim=-1)  # [B, nH,  S, D]
    k = torch.cat((k_embed, k_pass), dim=-1)  # [B, nKV, S, D]

    # --- GQA: expand K and V to full num_heads ---
    # Each KV head serves num_groups Q heads
    k = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(bsz, num_heads, seq_len, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(bsz, num_heads, seq_len, head_dim)

    # --- Scaled dot-product attention (causal) ---
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        scale=head_dim**-0.5,
    )  # [B, nH, S, D]

    # --- Reshape and output projection ---
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)
    output = F.linear(attn_output, wo)  # [B, S, hidden_size]
    return output


# ---------------------------------------------------------------------------
# DenseAttention
# ---------------------------------------------------------------------------
# NemotronH "dense" attention layers are the 6 blocks where the
# hybrid_override_pattern character is '*'.  Layer indices: 5, 12, 19, 26, 33,
# 42 (0-based).
#
# Each block wraps:
#   1. Pre-block RMSNorm  (backbone.layers.{i}.norm.weight)
#   2. NemotronHAttention (GQA, 32 Q heads / 2 KV heads, head_dim=128)
#      Weights: backbone.layers.{i}.mixer.{q,k,v,o}_proj.weight
#      NOTE: The HuggingFace implementation has RoPE commented out with a TODO,
#            so *no* positional encoding is applied in the reference.
#   3. Residual addition
#
# NemotronHBlock.forward():
#   residual      = hidden_states
#   hidden_states = norm(hidden_states)
#   hidden_states = mixer(hidden_states, cache_position=...)[0]
#   hidden_states = residual + hidden_states
#
# NemotronHAttention.forward() (no RoPE — TODO comment in source):
#   Q = q_proj(x)    [B, S, n_heads*head_dim]
#   K = k_proj(x)    [B, S, n_kv*head_dim]
#   V = v_proj(x)    [B, S, n_kv*head_dim]
#   reshape/transpose to [B, H, S, D]
#   repeat_kv(K, groups), repeat_kv(V, groups)
#   SDPA (causal when seqlen > 1)
#   transpose/reshape -> [B, S, n_heads*head_dim]
#   output = o_proj(attn_output)
# ---------------------------------------------------------------------------


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q head count (GQA -> MHA expansion)."""
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


def dense_attention(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    num_heads: int = 32,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Pure-PyTorch DenseAttention block for Nemotron-3 Nano 30B.

    Replicates the NemotronHBlock (block_type='attention') forward pass with
    the NemotronHAttention eager implementation.

    NOTE: RoPE is intentionally absent — the HuggingFace implementation has
    positional embeddings commented out with a TODO for this model variant.

    Args:
        hidden_states:  Input tensor of shape [batch, seq_len, hidden_size].
        norm_weight:    RMSNorm scale vector of shape [hidden_size].
        wq:             Q projection weight  [num_heads*head_dim, hidden_size].
        wk:             K projection weight  [num_kv_heads*head_dim, hidden_size].
        wv:             V projection weight  [num_kv_heads*head_dim, hidden_size].
        wo:             O projection weight  [hidden_size, num_heads*head_dim].
        num_heads:      Number of Q attention heads (32 for NemotronH-30B).
        num_kv_heads:   Number of KV heads (2 for NemotronH-30B GQA).
        head_dim:       Per-head feature dimension (128 for NemotronH-30B).
        norm_eps:       RMSNorm epsilon (1e-5 for NemotronH-30B).

    Returns:
        Tensor of shape [batch, seq_len, hidden_size] — post-residual output.
    """
    residual = hidden_states

    # 1. Pre-norm (RMSNorm)
    normed = layer_norm(hidden_states, norm_weight, eps=norm_eps)

    # 2. Attention (no RoPE — HF source has it TODO-commented)
    bsz, q_len, _ = normed.shape
    num_kv_groups = num_heads // num_kv_heads

    q = F.linear(normed, wq)  # [B, S, num_heads*head_dim]
    k = F.linear(normed, wk)  # [B, S, num_kv_heads*head_dim]
    v = F.linear(normed, wv)  # [B, S, num_kv_heads*head_dim]

    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)  # [B, nH,  S, D]
    k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # [B, nKV, S, D]
    v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # [B, nKV, S, D]

    # GQA: expand KV heads to match Q heads
    k = _repeat_kv(k, num_kv_groups)  # [B, nH, S, D]
    v = _repeat_kv(v, num_kv_groups)  # [B, nH, S, D]

    # Scaled dot-product attention (causal for seqlen > 1)
    is_causal = q_len > 1
    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
    )  # [B, nH, S, D]

    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
    attn_out = F.linear(attn_out, wo)  # [B, S, hidden_size]

    # 3. Residual
    return residual + attn_out


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
