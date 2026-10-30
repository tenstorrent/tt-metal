"""Bounded-memory HF reference for selected output rows at the real context size.

Projects every prefix token with actual HF modules. Only attention/output rows
are selected; no prefix KV is skipped. Short tests compare against HF forward.
"""

import torch
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb


@torch.no_grad()
def selected_reference(layer, x, cos, sin, indices):
    attn = layer.self_attn
    b, length, _ = x.shape
    dim = attn.head_dim
    nq = layer.config.num_attention_heads
    nk = nq // attn.num_key_value_groups
    # Sliding attention cannot use keys earlier than the first selected query's window.
    start = max(0, min(indices) - attn.sliding_window + 1) if attn.sliding_window else 0
    keys, values = [], []
    for offset in range(start, length, 2048):
        end = min(length, offset + 2048)
        z = layer.input_layernorm(x[:, offset:end])
        kr = attn.k_proj(z).view(b, -1, nk, dim)
        vr = attn.v_proj(z).view(b, -1, nk, dim) if attn.v_proj is not None else kr
        keys.append(
            apply_rotary_pos_emb(attn.k_norm(kr), cos[:, offset:end], sin[:, offset:end], unsqueeze_dim=2).transpose(
                1, 2
            )
        )
        values.append(attn.v_norm(vr).transpose(1, 2))
    k, v = torch.cat(keys, dim=2), torch.cat(values, dim=2)
    residual = x[:, indices]
    q = attn.q_norm(attn.q_proj(layer.input_layernorm(residual)).view(b, -1, nq, dim))
    q = apply_rotary_pos_emb(q, cos[:, indices], sin[:, indices], unsqueeze_dim=2).transpose(1, 2)
    query_pos = torch.tensor(indices)[:, None]
    key_pos = torch.arange(start, length)[None, :]
    allowed = key_pos <= query_pos
    if attn.sliding_window:
        allowed &= key_pos > query_pos - attn.sliding_window
    mask = torch.where(allowed, 0.0, float("-inf")).to(x.dtype)[None, None]
    outputs = []
    group = attn.num_key_value_groups
    for i in range(nk):
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                q[:, i * group : (i + 1) * group],
                k[:, i : i + 1].expand(-1, group, -1, -1),
                v[:, i : i + 1].expand(-1, group, -1, -1),
                attn_mask=mask,
                scale=1.0,
            )
        )
    a = torch.cat(outputs, dim=1).transpose(1, 2).reshape(b, len(indices), -1)
    y = residual + layer.post_attention_layernorm(attn.o_proj(a))
    return (y + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(y)))) * layer.layer_scalar
