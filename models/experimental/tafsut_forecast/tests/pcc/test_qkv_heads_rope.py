# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-1: the ported ``ops.qkv_heads_rope`` against torch on the same bf16 inputs, at every s_pad it runs at (128 for
contexts up to 2048, 256 for 3072 / 4096: long_horizon's forward 2). Exact head split; RoPE error at bf16 rounding."""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.tafsut_forecast import ops
from models.experimental.tafsut_forecast.reference import dataflow as df

D, NH, DH = df.D, df.NH, df.DH


def rotate_half(x):
    return torch.cat([-x[..., DH // 2 :], x[..., : DH // 2]], -1)


@pytest.mark.parametrize("fp32_dest", [True, False])
@pytest.mark.parametrize("ctx", [2048, 3072])
def test_qkv_heads_rope(device, ctx, fp32_dest):
    P = ctx // df.PATCH
    s_pad = df.PAD_TABLE[ctx][0]
    B = 3
    x = torch.randn(B, s_pad, 3 * D, generator=torch.Generator().manual_seed(ctx)).bfloat16()
    cos, sin = (t.bfloat16() for t in df.rope_tables(df.layout(P, s_pad)))
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    q, k, v = ops.qkv_heads_rope(up(x), up(cos[None, None]), up(sin[None, None]), num_heads=NH,
                                 fp32_dest_acc_en=fp32_dest)
    q, k, v = (ttnn.to_torch(t).float() for t in (q, k, v))
    xs = x.float().view(B, s_pad, 3, NH, DH).permute(2, 0, 3, 1, 4)  # [3, B, NH, S, DH]
    c, s = cos.float(), sin.float()
    ref = [xs[i] * c + rotate_half(xs[i]) * s for i in (0, 1)]
    assert torch.equal(v, xs[2]), "V must be the exact head split"
    for name, got, r in (("q", q, ref[0]), ("k", k, ref[1])):
        err = (got - r).abs()
        rel = err / r.abs().clamp_min(1e-3)
        # per-position worst error: a wrong cos/sin row would show up as one sequence tile far off
        per_tile = err.amax((0, 1, 3)).view(-1, 32).amax(1)
        print(f"\n[rope] ctx {ctx} s_pad {s_pad} fp32_dest {fp32_dest} {name}: max abs {float(err.max()):.3e}, "
              f"p99.9 rel {float(torch.quantile(rel.flatten()[::7], 0.999)):.3e}, per seq tile {per_tile.tolist()}")
        # fp32 dest: one bf16 rounding of the output; bf16 dest (not used) rounds the products too
        assert float(err.max()) <= (2**-7 if fp32_dest else 2**-6) * float(r.abs().max()), name
