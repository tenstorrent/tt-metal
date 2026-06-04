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

    def get_w(t):
        x = ttnn.to_torch(t).float()
        return x[:true_d_model, :true_d_model]

    def get_b(t):
        x = ttnn.to_torch(t).float().squeeze()
        return x[:true_d_model]

    Wq = get_w(q_proj_weight)
    Wk = get_w(k_proj_weight)
    Wv = get_w(v_proj_weight)
    Wo = get_w(out_proj_weight)
    bq = get_b(q_proj_bias)
    bk = get_b(k_proj_bias)
    bv = get_b(v_proj_bias)
    bo = get_b(out_proj_bias)

    B, tgt_len, _ = h.shape
    src_len = kv.shape[1]

    q = h @ Wq + bq
    k = kv @ Wk + bk
    v = kv @ Wv + bv

    q = q.view(B, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, src_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, src_len, num_heads, head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    out = attn @ v

    out = out.transpose(1, 2).contiguous().view(B, tgt_len, true_d_model)
    out = out @ Wo + bo

    pad = 32 - (true_d_model % 32)
    if pad < 32:
        out = torch.nn.functional.pad(out, (0, pad))

    return ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
