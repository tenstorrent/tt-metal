"""MLA (Multi-Head Latent Attention) — reference implementation."""

import math
import torch
import torch.nn.functional as F

from helpers import repeat_kv


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
    attention_mask=None,
    past_key_value=None,
    use_cache=False,
):
    """
    MLA with explicit latent KV path.

    hidden_states: [b, seq, hidden]
    wq: [num_q_heads * head_dim, hidden]
    w_kv_down: [kv_latent_dim, hidden]
    w_k_up: [num_kv_heads * head_dim, kv_latent_dim]
    w_v_up: [num_kv_heads * head_dim, kv_latent_dim]
    wo: [hidden, num_q_heads * head_dim]
    """
    b, q_seq, h = hidden_states.shape
    head_dim = wq.shape[0] // num_q_heads

    # Q projection
    q = F.linear(hidden_states, wq).view(b, q_seq, num_q_heads, head_dim).transpose(1, 2)

    # Latent KV path
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
    attn_out = torch.matmul(attn_weights, v_exp)
    attn_out = attn_out.transpose(1, 2).contiguous().view(b, q_seq, num_q_heads * head_dim)
    attn_out = F.linear(attn_out, wo)
    return attn_out, present_kv


mla_attention_tt = mla_attention_torch
