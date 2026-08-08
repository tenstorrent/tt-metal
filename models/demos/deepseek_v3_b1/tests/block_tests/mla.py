# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MLA (Multi-Head Latent Attention) — reference implementation."""

import torch
import torch.nn.functional as F

from helpers import repeat_kv, rms_norm, apply_rotary_pos_emb_interleaved


# ---------------------------------------------------------------------------
# MLA — reference implementation
# ---------------------------------------------------------------------------

def mla_attention_torch(
    hidden_states,
    *,
    wq, w_kv_a, w_kv_b, wo,
    num_q_heads,
    num_kv_heads,
    kv_latent_dim,
    kv_a_layernorm_weight,
    qk_rope_head_dim,
    position_embeddings,
    # Q LoRA (optional): q_a_proj → q_a_layernorm → q_b_proj
    w_q_a=None,
    q_a_layernorm_weight=None,
    q_a_layernorm_eps=1e-6,
    # KV layernorm
    kv_a_layernorm_eps=1e-6,
    # Scaling
    mscale=1.0,
    # Attention mode
    # Mask / cache
    attention_mask=None,
    past_key_value=None,
    use_cache=False,
):
    """
    MLA with latent KV path, Q LoRA, KV layernorm, and RoPE.
    Matches the official DeepSeek V3 MLA implementation.

    Q path (two modes):
      - Direct: wq projects hidden → Q  (when w_q_a is None)
      - LoRA:   w_q_a → q_a_layernorm → wq (q_b_proj)  (when w_q_a is provided)

    KV path:
      - w_kv_a → kv_a_layernorm → w_kv_b → [k_nope, v] split

    hidden_states: [b, seq, hidden]
    wq: [num_q_heads * qk_head_dim, hidden_or_q_lora_rank]
    w_q_a: [q_lora_rank, hidden]  (optional Q LoRA down-projection)
    q_a_layernorm_weight: [q_lora_rank]  (required when w_q_a is provided)
    w_kv_a: [kv_latent_dim + rope_dim, hidden]
    kv_a_layernorm_weight: [kv_latent_dim]
    w_kv_b: [num_kv_heads * (nope_dim + v_head_dim), kv_latent_dim]
    wo: [hidden, num_q_heads * v_head_dim]
    qk_rope_head_dim: number of RoPE dims per head
    position_embeddings: (cos, sin) each [b, seq, rope_dim]
    mscale: attention scale multiplier for extended context (YaRN)
    """
    b, q_seq, h = hidden_states.shape

    # Derive dimensions from weight shapes
    qk_head_dim = wq.shape[0] // num_q_heads
    nope_dim = qk_head_dim - qk_rope_head_dim
    kv_b_per_head = w_kv_b.shape[0] // num_kv_heads
    v_head_dim = kv_b_per_head - nope_dim

    # Softmax scale with mscale (YaRN extended context)
    scale = (qk_head_dim ** -0.5) * mscale * mscale

    # --- Q path ---
    if w_q_a is not None:
        q_latent = F.linear(hidden_states, w_q_a)
        q_latent = rms_norm(q_latent, q_a_layernorm_weight, q_a_layernorm_eps)
        q = F.linear(q_latent, wq)
    else:
        q = F.linear(hidden_states, wq)
    q = q.view(b, q_seq, num_q_heads, qk_head_dim).transpose(1, 2)

    # --- KV compressed path ---
    compressed_kv = F.linear(hidden_states, w_kv_a)
    kv_latent, k_pe = compressed_kv.split([kv_latent_dim, qk_rope_head_dim], dim=-1)

    # KV layernorm (applied to latent part only)
    kv_latent = rms_norm(kv_latent, kv_a_layernorm_weight, kv_a_layernorm_eps)

    # --- RoPE on rope parts ---
    q_nope, q_pe = q.split([nope_dim, qk_rope_head_dim], dim=-1)
    k_pe_4d = k_pe.unsqueeze(1)  # [b, 1, seq, rope_dim]
    cos, sin = position_embeddings
    q_pe, k_pe_4d = apply_rotary_pos_emb_interleaved(q_pe, k_pe_4d, cos, sin)
    k_pe = k_pe_4d.squeeze(1)  # [b, seq, rope_dim]

    # --- Expand KV, standard attention ---
    kv = F.linear(kv_latent, w_kv_b)
    kv = kv.view(b, -1, num_kv_heads, kv_b_per_head).transpose(1, 2)
    k_nope, v = kv.split([nope_dim, v_head_dim], dim=-1)

    k_pe_expanded = k_pe.unsqueeze(1).expand(-1, num_kv_heads, -1, -1)
    k = torch.cat([k_nope, k_pe_expanded], dim=-1)
    q = torch.cat([q_nope, q_pe], dim=-1)

    if past_key_value is not None:
        past_k, past_v = past_key_value
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    present_kv = (k, v) if use_cache else None

    n_rep = num_q_heads // num_kv_heads
    k_exp = repeat_kv(k, n_rep)
    v_exp = repeat_kv(v, n_rep)

    scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scale

    if attention_mask is not None:
        scores = scores + attention_mask

    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(hidden_states)
    attn_out = torch.matmul(attn_weights, v_exp)

    attn_out = attn_out.transpose(1, 2).contiguous().view(b, q_seq, num_q_heads * v_head_dim)
    attn_out = F.linear(attn_out, wo)
    return attn_out, present_kv


mla_attention_tt = mla_attention_torch


# ---------------------------------------------------------------------------
# Absorbed attention (optimization example)
# ---------------------------------------------------------------------------
# Instead of expanding KV via w_kv_b, absorbed attention works in latent space:
# - Absorb w_kv_b[:k_nope] into Q:  q_absorbed = q_nope @ w_k_nope
# - Cache (kv_latent, k_pe) instead of (K, V) — smaller KV cache
# - Compute scores:  q_absorbed @ kv_latent^T + q_pe @ k_pe^T
# - After softmax, apply w_kv_b[v:] to recover output in head space
#
# This produces identical results to the naive path above.
#
#     n_rep = num_q_heads // num_kv_heads
#     w_kv_b_3d = w_kv_b.view(num_kv_heads, kv_b_per_head, kv_latent_dim)
#     w_kv_b_3d = w_kv_b_3d.repeat_interleave(n_rep, dim=0)
#
#     # Absorb w_k into Q: q_nope @ w_k → q in latent space
#     q_absorbed = torch.einsum("bhsd,hdc->bhsc", q_nope, w_kv_b_3d[:, :nope_dim, :])
#
#     # Cache compressed latent + PE
#     if past_key_value is not None:
#         past_kv, past_pe = past_key_value
#         kv_latent = torch.cat([past_kv, kv_latent], dim=1)
#         k_pe = torch.cat([past_pe, k_pe], dim=1)
#
#     present_kv = (kv_latent, k_pe) if use_cache else None
#
#     # Scores in latent space
#     scores = (torch.einsum("bhsc,btc->bhst", q_absorbed, kv_latent)
#               + torch.einsum("bhsr,btr->bhst", q_pe, k_pe)) * scale
#
#     if attention_mask is not None:
#         scores = scores + attention_mask
#
#     attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(hidden_states)
#
#     # Output in latent space, then apply w_v
#     attn_out = torch.einsum("bhst,btc->bhsc", attn_weights, kv_latent)
#     attn_out = torch.einsum("bhsc,hdc->bhsd", attn_out, w_kv_b_3d[:, -v_head_dim:, :])
