# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `q_k_v_attention_legacy` of coqui/XTTS-v2.

Reference submodule: `gpt.conditioning_encoder.attn.0.attention`, a
`TTS.tts.layers.tortoise.arch_utils.QKVAttentionLegacy` (n_heads=8):

    bs, width, length = qkv.shape                       # [1, 3072, 259]
    ch = width // (3*n_heads)                           # 128
    q, k, v = qkv.reshape(bs*n_heads, ch*3, length).split(ch, dim=1)  # head-major
    scale = 1/sqrt(sqrt(ch))
    weight = einsum("bct,bcs->bts", q*scale, k*scale)   # [b, t, s]
    weight = softmax(weight, dim=-1)
    a = einsum("bts,bcs->bct", weight, v)               # [b, ch, t]
    return a.reshape(bs, n_heads*ch, length)            # [1, 1024, 259]

No learned parameters (mask / rel_pos are None in this call). The fused qkv is
head-major (`[head0 q|k|v, head1 q|k|v, ...]`), so we reshape `[1, H*3C, T]` ->
`[H, 3C, T]` and slice each head's q/k/v. Attention is:

    weight = (qᵀ @ k) * scale²   (scale² = 1/sqrt(ch))
    a      = v @ weightᵀ

Everything runs natively in ttnn (float32 + HiFi4 batched matmuls).
"""

from __future__ import annotations

import math

import ttnn


def build(device, torch_module):
    """Bind n_heads and return a native ttnn QKV-attention forward closure."""
    n_heads = int(torch_module.n_heads)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def forward(qkv, *args, **kwargs):
        if qkv.get_dtype() != ttnn.float32:
            qkv = ttnn.typecast(qkv, ttnn.float32)
        bs, width, length = int(qkv.shape[0]), int(qkv.shape[1]), int(qkv.shape[2])
        ch = width // (3 * n_heads)
        scale2 = 1.0 / math.sqrt(ch)  # (1/sqrt(sqrt(ch)))^2, folded onto the qk product

        # [bs, H*3C, T] -> [bs*H, 3C, T] (head-major), then split q|k|v.
        x = ttnn.reshape(ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT), (bs * n_heads, 3 * ch, length))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        q = ttnn.slice(x, [0, 0, 0], [bs * n_heads, ch, length])            # [B, ch, T]
        k = ttnn.slice(x, [0, ch, 0], [bs * n_heads, 2 * ch, length])
        v = ttnn.slice(x, [0, 2 * ch, 0], [bs * n_heads, 3 * ch, length])

        # weight[b,t,s] = sum_c q[b,c,t]*k[b,c,s] = (qᵀ @ k)
        qt = ttnn.transpose(q, -2, -1)                                       # [B, T, ch]
        weight = ttnn.matmul(qt, k, compute_kernel_config=compute_config)    # [B, T, T]
        weight = ttnn.multiply(weight, scale2)
        weight = ttnn.softmax(weight, dim=-1)

        # a[b,c,t] = sum_s weight[b,t,s]*v[b,c,s] = v @ weightᵀ
        wt = ttnn.transpose(weight, -2, -1)                                  # [B, s, t]
        a = ttnn.matmul(v, wt, compute_kernel_config=compute_config)         # [B, ch, T]

        # [bs*H, ch, T] -> [bs, H*ch, T]
        a = ttnn.reshape(ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT), (bs, n_heads * ch, length))
        return ttnn.to_layout(a, ttnn.TILE_LAYOUT)

    return forward


def q_k_v_attention_legacy(*args, **kwargs):
    raise RuntimeError(
        "q_k_v_attention_legacy requires build(device, torch_module) to bind n_heads; "
        "use build(device, torch_module)."
    )
