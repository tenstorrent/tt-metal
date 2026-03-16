# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""GQA (Grouped Query Attention) — reference implementation."""

import math
import torch
import torch.nn.functional as F

from helpers import repeat_kv, apply_rotary_pos_emb, apply_rotary_pos_emb_glm4


# ---------------------------------------------------------------------------
# GQA — reference implementation
# ---------------------------------------------------------------------------

def gqa_attention_torch(
    hidden_states,
    *,
    wq, wk, wv, wo,
    num_q_heads,
    num_kv_heads,
    position_embeddings=None,
    rope_variant="standard",
    attention_mask=None,
    past_key_value=None,
    use_cache=False,
):
    """
    Grouped Query Attention.

    hidden_states: [b, seq, hidden]
    wq: [num_q_heads * head_dim, hidden]
    wk: [num_kv_heads * head_dim, hidden]
    wv: [num_kv_heads * head_dim, hidden]
    wo: [hidden, num_q_heads * head_dim]
    position_embeddings: None or (cos, sin) each [b, seq, rope_dim]
    rope_variant: "standard" (Llama/Qwen) or "glm4" (interleaved, partial)
    attention_mask: None, bool [b,1,q,kv] (True=keep), or additive [b,1,q,kv]
    past_key_value: None or (past_k, past_v) each [b, kv_heads, past_seq, hd]

    Returns (attn_out [b, seq, hidden], present_kv or None)
    """
    b, q_seq, h = hidden_states.shape
    head_dim = wq.shape[0] // num_q_heads

    q = F.linear(hidden_states, wq).view(b, q_seq, num_q_heads, head_dim).transpose(1, 2)
    k = F.linear(hidden_states, wk).view(b, q_seq, num_kv_heads, head_dim).transpose(1, 2)
    v = F.linear(hidden_states, wv).view(b, q_seq, num_kv_heads, head_dim).transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        if rope_variant == "glm4":
            q, k = apply_rotary_pos_emb_glm4(q, k, cos, sin)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

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
    attn_out = torch.matmul(attn_weights, v_exp)
    attn_out = attn_out.transpose(1, 2).contiguous().view(b, q_seq, num_q_heads * head_dim)
    attn_out = F.linear(attn_out, wo)
    return attn_out, present_kv


gqa_attention_tt = gqa_attention_torch
