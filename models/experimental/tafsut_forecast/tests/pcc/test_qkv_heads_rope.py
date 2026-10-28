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


def run(device, x, cos, sin, fp32_dest):
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ops.qkv_heads_rope(up(x), up(cos[None, None]), up(sin[None, None]), num_heads=NH, fp32_dest_acc_en=fp32_dest)
    return [ttnn.to_torch(t).float() for t in out]


def bad_tiles(got, ref, fp32_dest, label=""):
    """Sequence tiles whose worst error exceeds one output rounding of that tile's own largest value (v1.24: the
    bound is per tile, so one misread cos/sin row cannot hide under the global maximum)."""
    err = (got - ref).abs().amax((0, 1, 3)).view(-1, 32).amax(1)
    mag = ref.abs().amax((0, 1, 3)).view(-1, 32).amax(1)
    ratio = err / mag
    print(f"\n[rope] {label}: max abs {float(err.max()):.3e}, per seq tile err/max {[round(float(r), 5) for r in ratio]}")
    # fp32 dest: one bf16 rounding of the output; bf16 dest (not used) rounds the products too
    return (ratio > (2**-7 if fp32_dest else 2**-6)).nonzero().flatten().tolist()


def reference(x, cos, sin):
    B, s_pad = x.shape[:2]
    xs = x.float().view(B, s_pad, 3, NH, DH).permute(2, 0, 3, 1, 4)  # [3, B, NH, S, DH]
    c, s = cos.float(), sin.float()
    return [xs[i] * c + rotate_half(xs[i]) * s for i in (0, 1)] + [xs[2]]


# B = 3 puts 1-2 units on a core; B = 1024 (the batch the bench issues at s_pad 128) many units per core
@pytest.mark.parametrize("fp32_dest", [True, False])
@pytest.mark.parametrize("ctx, B", [(2048, 3), (2048, 1024), (3072, 3)])
def test_qkv_heads_rope(device, ctx, B, fp32_dest):
    P = ctx // df.PATCH
    s_pad = df.PAD_TABLE[ctx][0]
    x = torch.randn(B, s_pad, 3 * D, generator=torch.Generator().manual_seed(ctx + B)).bfloat16()
    cos, sin = (t.bfloat16() for t in df.rope_tables(df.layout(P, s_pad)))
    q, k, v = run(device, x, cos, sin, fp32_dest)
    rq, rk, rv = reference(x, cos, sin)
    assert torch.equal(v, rv), "V must be the exact head split"
    for name, got, r in (("q", q, rq), ("k", k, rk)):
        bad = bad_tiles(got, r, fp32_dest, f"ctx {ctx} s_pad {s_pad} B {B} fp32_dest {fp32_dest} {name}")
        assert not bad, f"{name}: sequence tiles {bad} over one output rounding"


def test_qkv_heads_rope_control(device):
    """Positive control (v1.13): a cos/sin table with one sequence tile misread (tile 1 holds tile 0's rows) must fail
    the per-tile check on that tile and only there."""
    ctx, B = 2048, 3
    P = ctx // df.PATCH
    s_pad = df.PAD_TABLE[ctx][0]
    x = torch.randn(B, s_pad, 3 * D, generator=torch.Generator().manual_seed(7)).bfloat16()
    cos, sin = (t.bfloat16() for t in df.rope_tables(df.layout(P, s_pad)))
    bad_cos, bad_sin = cos.clone(), sin.clone()
    bad_cos[32:64], bad_sin[32:64] = cos[0:32], sin[0:32]
    q, _, _ = run(device, x, bad_cos, bad_sin, True)
    rq = reference(x, cos, sin)[0]
    assert bad_tiles(q, rq, True, "control: tile 1 misread") == [1]
