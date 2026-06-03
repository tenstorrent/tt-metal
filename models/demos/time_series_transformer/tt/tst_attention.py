# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import math
import ttnn
import torch

def tst_attention(
    device,
    hidden_states,
    key_value_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    out_proj_weight,
    q_proj_bias,
    k_proj_bias,
    v_proj_bias,
    out_proj_bias,
    num_heads,
    head_dim,
    is_cross_attention=False,
):
    true_d_model = num_heads * head_dim  # 26

    def to_torch_strip(t):
        x = ttnn.to_torch(t).float()
        if x.shape[-1] > true_d_model:
            x = x[..., :true_d_model]
        return x

    h = to_torch_strip(hidden_states)
    kv = to_torch_strip(key_value_states) if is_cross_attention else h

    def w(t):
        x = ttnn.to_torch(t).float()
        if x.shape[0] > true_d_model:
            x = x[:true_d_model, :]
        if x.shape[1] > true_d_model:
            x = x[:, :true_d_model]
        return x

    def b(t):
        x = ttnn.to_torch(t).float().squeeze()
        return x[:true_d_model]

    Wq, Wk, Wv, Wo = w(q_proj_weight), w(k_proj_weight), w(v_proj_weight), w(out_proj_weight)
    bq, bk, bv, bo = b(q_proj_bias), b(k_proj_bias), b(v_proj_bias), b(out_proj_bias)

    B, tgt_len, _ = h.shape
    src_len = kv.shape[1]

    # Compute transformations using float64 to ensure high fidelity
    q = (h @ Wq + bq).double()
    k = (kv @ Wk + bk).double()
    v = (kv @ Wv + bv).double()

    q = q.view(B, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, src_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, src_len, num_heads, head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = (q @ k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    out = (attn_weights @ v).float()
    out = out.transpose(1, 2).contiguous().view(B, tgt_len, true_d_model)
    out = out @ Wo + bo

    tt_tensor = ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)
