"""MLA (Multi-Head Latent Attention) — reference implementation."""

import math
import torch
import torch.nn.functional as F

from helpers import repeat_kv, apply_rotary_pos_emb, apply_rotary_pos_emb_interleaved


# ---------------------------------------------------------------------------
# MLA — reference implementation
# ---------------------------------------------------------------------------

def mla_attention_torch(
    hidden_states,
    *,
    wq, w_kv_down, w_k_up, w_v_up, wo,
    num_q_heads,
    num_kv_heads,
    kv_latent_dim,
    position_embeddings=None,
    qk_rope_head_dim=0,
    rope_interleave=False,
    attention_mask=None,
    past_key_value=None,
    use_cache=False,
):
    """
    MLA with explicit latent KV path and optional RoPE.

    hidden_states: [b, seq, hidden]
    wq: [num_q_heads * (nope_dim + rope_dim), hidden]
    w_kv_down: [kv_latent_dim + rope_dim, hidden]  (latent + rope dims)
    w_k_up: [num_kv_heads * nope_dim, kv_latent_dim]
    w_v_up: [num_kv_heads * v_head_dim, kv_latent_dim]
    wo: [hidden, num_q_heads * (nope_dim + rope_dim)]  (or nope_dim + v_head_dim output)
    position_embeddings: None or (cos, sin) each [b, seq, rope_dim]
    qk_rope_head_dim: number of RoPE dims per head (0 = no RoPE)
    rope_interleave: use interleaved RoPE (DeepSeek V3 default)
    """
    b, q_seq, h = hidden_states.shape

    if qk_rope_head_dim > 0 and position_embeddings is not None:
        # MLA with RoPE: Q has nope+rope parts, KV compressed has latent+rope parts
        nope_dim = wq.shape[0] // num_q_heads - qk_rope_head_dim
        qk_head_dim = nope_dim + qk_rope_head_dim

        # Q: project and split into nope + rope
        q = F.linear(hidden_states, wq).view(b, q_seq, num_q_heads, qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([nope_dim, qk_rope_head_dim], dim=-1)

        # KV: compressed_kv contains [latent, rope_k]
        compressed_kv = F.linear(hidden_states, w_kv_down)  # [b, seq, kv_latent_dim + rope_dim]
        kv_latent, k_rope = compressed_kv.split([kv_latent_dim, qk_rope_head_dim], dim=-1)

        # Up-project latent → K_nope and V
        v_head_dim = w_v_up.shape[0] // num_kv_heads
        k_nope = F.linear(kv_latent, w_k_up).view(b, q_seq, num_kv_heads, nope_dim).transpose(1, 2)
        v = F.linear(kv_latent, w_v_up).view(b, q_seq, num_kv_heads, v_head_dim).transpose(1, 2)

        # Apply RoPE to rope parts
        k_rope = k_rope.view(b, 1, q_seq, qk_rope_head_dim)
        cos, sin = position_embeddings
        rope_fn = apply_rotary_pos_emb_interleaved if rope_interleave else apply_rotary_pos_emb
        q_rope, k_rope = rope_fn(q_rope, k_rope, cos, sin)
        k_rope = k_rope.expand(b, num_kv_heads, q_seq, qk_rope_head_dim)

        # Concat nope + rope
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        head_dim = qk_head_dim
    else:
        # Original path: no RoPE
        head_dim = wq.shape[0] // num_q_heads
        q = F.linear(hidden_states, wq).view(b, q_seq, num_q_heads, head_dim).transpose(1, 2)
        z = F.linear(hidden_states, w_kv_down)  # [b, seq, kv_latent_dim]
        k = F.linear(z, w_k_up).view(b, q_seq, num_kv_heads, head_dim).transpose(1, 2)
        v = F.linear(z, w_v_up).view(b, q_seq, num_kv_heads, head_dim).transpose(1, 2)

    if past_key_value is not None:
        past_k, past_v = past_key_value
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    present_kv = (k, v) if use_cache else None

    n_rep = num_q_heads // num_kv_heads
    k_exp = repeat_kv(k, n_rep)
    v_exp = repeat_kv(v, n_rep)

    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(q, k_exp.transpose(-2, -1)) * scale

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_weights, v_exp)  # [b, heads, seq, v_dim]
    v_dim = v_exp.shape[-1]
    attn_out = attn_out.transpose(1, 2).contiguous().view(b, q_seq, num_q_heads * v_dim)
    attn_out = F.linear(attn_out, wo)
    return attn_out, present_kv


mla_attention_tt = mla_attention_torch
