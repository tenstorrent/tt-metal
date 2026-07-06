# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `attention_block` of coqui/XTTS-v2.

Reference submodule: `gpt.conditioning_encoder.attn.0`, an instance of
`TTS.tts.layers.tortoise.arch_utils.AttentionBlock` (tortoise-style spatial
self-attention over a channels-first `(B, C, T)` activation).

Instance config (probed from the trained model):
    channels=1024, num_heads=16 (head_ch=64), tortoise_norm=False,
    relative_pos_embeddings=None, GroupNorm32(groups=32).

Reference forward (tortoise_norm=False):
    x_norm = GroupNorm32(x)                 # 32 groups, float32 stats
    qkv    = Conv1d(1024, 3072, 1)(x_norm)  # 1x1 conv == per-position linear
    h      = QKVAttentionLegacy(16)(qkv)    # multi-head attn over T
    h      = Conv1d(1024, 1024, 1)(h)       # proj_out
    out    = x_norm + h

QKVAttentionLegacy detail (num_heads=16, ch=64):
    reshape qkv `(1, 3072, T)` -> `(16, 192, T)` and split -> q,k,v `(16, 64, T)`.
    So the 3072 channel axis is ordered head-major then q|k|v then head-channel:
    channel = head*192 + {q:0,k:1,v:2}*64 + c.
    scale = ch**-0.25 applied to BOTH q and k (== 1/sqrt(ch) combined).
    weight = softmax(qᵀk * scale²) over keys; out = weight·v; reshape -> (1,1024,T).

Captured shapes: in/out `[1, 1024, 259]`.

Harness note: the PCC harness converts the single primary arg `x` to a ttnn
tensor; there are no extra tensor kwargs. All math below is native ttnn.
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.group_norm32 import build as _build_gn
from models.demos.xtts_v2._stubs.q_k_v_attention_legacy import build as _build_qkv

_CHANNELS = 1024


def _linear_1x1(x_tl, weight_t, bias):
    """1x1 Conv1d as a per-token linear on tokens-last `[1, T, C_in]`.

    weight_t is the transposed conv weight `[C_in, C_out]`; bias is `[1,1,C_out]`.
    """
    y = ttnn.matmul(x_tl, weight_t)
    y = ttnn.add(y, bias)
    return y


def build(device, torch_module):
    """Precompute ttnn weights from the trained torch AttentionBlock."""
    import torch

    m = torch_module.float()

    def _w(t):
        return ttnn.as_tensor(
            t.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # GroupNorm32 handled by the graduated leaf stub (reads groups/eps/affine
    # from m.norm); takes/returns channels-first [1, C, T].
    gn_fwd = _build_gn(device, m.norm)

    # Multi-head QKV attention handled by the graduated leaf stub (reads
    # n_heads from m.attention); takes head-major qkv [1, 3C, T] -> [1, C, T].
    qkv_fwd = _build_qkv(device, m.attention)

    # 1x1 conv weights -> [C_in, C_out] (transpose of conv's [C_out, C_in, 1]).
    qkv_w = _w(m.qkv.weight.detach().squeeze(-1).t())        # [1024, 3072]
    qkv_b = _w(m.qkv.bias.detach().reshape(1, 1, 3 * _CHANNELS))
    proj_w = _w(m.proj_out.weight.detach().squeeze(-1).t())  # [1024, 1024]
    proj_b = _w(m.proj_out.bias.detach().reshape(1, 1, _CHANNELS))

    def forward(x, mask=None):
        x_norm = gn_fwd(x)                                   # [1, C, T]

        # qkv = 1x1 conv on the normalized activation, produced head-major.
        # tokens-last for the linear, then back to channels-first for the leaf.
        xn_tl = ttnn.transpose(x_norm, -2, -1)               # [1, T, C]
        qkv = _linear_1x1(xn_tl, qkv_w, qkv_b)               # [1, T, 3C]
        qkv_cf = ttnn.transpose(qkv, -2, -1)                 # [1, 3C, T] head-major

        # multi-head QKVAttentionLegacy via the graduated leaf stub.
        a_cf = qkv_fwd(qkv_cf)                               # [1, C, T]

        a_tl = ttnn.transpose(a_cf, -2, -1)                  # [1, T, C]
        h = _linear_1x1(a_tl, proj_w, proj_b)                # [1, T, C]
        h_cf = ttnn.transpose(h, -2, -1)                     # [1, C, T]

        return ttnn.add(x_norm, h_cf)                        # tortoise_norm=False

    return forward


def attention_block(x, mask=None, **kwargs):
    raise RuntimeError(
        "attention_block requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
