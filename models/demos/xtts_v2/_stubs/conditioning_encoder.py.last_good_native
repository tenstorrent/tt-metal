# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `conditioning_encoder` of coqui/XTTS-v2.

Reference submodule: `gpt.conditioning_encoder`, an instance of
`TTS.tts.layers.tortoise.autoregressive.ConditioningEncoder`:

    h = init(x)     # nn.Conv1d(spec_dim=80, embedding_dim=1024, kernel_size=1)
    h = attn(h)     # nn.Sequential of 6 x AttentionBlock(1024, num_heads=16)
    return h

Input `x` is a channels-first `(1, 80, T)` mel-like activation; output is
`(1, 1024, T)`. Captured shapes: in `[1, 80, 259]`, out `[1, 1024, 259]`.

The 1x1 init conv is a per-position linear over channels. Each of the 6
attention blocks is structurally identical to the `attention_block` component,
so we reuse its native ttnn builder (`_stubs.attention_block.build`) per block
rather than duplicating the group-norm/qkv/attention math.
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.attention_block import build as _build_attention_block


def build(device, torch_module):
    """Precompute the init-conv ttnn weights and per-block attention forwards."""
    import torch

    m = torch_module.float()

    # 1x1 init conv weight -> [C_in=80, C_out=1024] (transpose of [C_out,C_in,1]).
    init_w = ttnn.from_torch(
        m.init.weight.detach().squeeze(-1).t().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )
    c_out = m.init.weight.shape[0]
    init_b = ttnn.from_torch(
        m.init.bias.detach().reshape(1, 1, c_out).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )

    # Reuse the native AttentionBlock port for each block in the Sequential.
    block_forwards = [_build_attention_block(device, blk) for blk in m.attn]

    def forward(x, *args, **kwargs):
        # init conv: channels-first [1,C_in,T] -> tokens-last linear -> [1,C_out,T]
        x_tl = ttnn.transpose(x, -2, -1)            # [1, T, C_in]
        h = ttnn.matmul(x_tl, init_w)               # [1, T, C_out]
        h = ttnn.add(h, init_b)
        h = ttnn.transpose(h, -2, -1)               # [1, C_out, T] channels-first

        for blk in block_forwards:
            h = blk(h)
        return h

    return forward


def conditioning_encoder(x, *args, **kwargs):
    raise RuntimeError(
        "conditioning_encoder requires build(device, torch_module) to bind "
        "trained weights; the bare callable has no parameters."
    )
