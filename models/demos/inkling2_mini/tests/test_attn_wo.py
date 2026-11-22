# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Attention o-projection with the post-attention conv in the same program (tt/fused/attn_wo.py) must be bit-equal
to attention + FusedShortConv.conv_add as model._app runs them: conv output (dense twin) and ring slot, eager over
several steps (all 4 phases, with / without the residual) and traced (replays at one phase)."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tests import test_attention as ta
from models.demos.inkling2_mini.tests.test_attention_fast import Runner
from models.demos.inkling2_mini.tt import residual as R
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv

C = 2304
PARAMS = [{"trace_region_size": 64 * 1024 * 1024}]


def dense(t):
    return ttnn.to_torch(t).float()


@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.parametrize("kind", ["local", "global"])
def test_attn_wo_conv(device, kind):
    is_global = kind == "global"
    g = torch.Generator().manual_seed(5)
    sd = ta.random_state_dict(is_global, seed=11 if is_global else 7)
    rA = Runner(device, is_global, sd, 2048, ttnn.bfloat8_b)
    stB = rA.attn.new_state(2048, ttnn.bfloat8_b)
    rows = R.RowOps(device)
    w = {
        "post_attn_conv": torch.randn(1, 1, 4, C, generator=g) * 0.5,
        "post_mlp_conv": torch.randn(1, 1, 4, C, generator=g),
    }
    conv, cm0 = FusedShortConv(rows, w, "post_attn_conv"), FusedShortConv(rows, w, "post_mlp_conv")
    ringA, ringB = conv.alloc_state(0), conv.alloc_state(1)

    def row(t, dtype):
        return ttnn.from_torch(t.reshape(1, 1, -1, C), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # h as in the model: a bf16 conv output row with a single-vector dense twin
    h = cm0.conv_add(row(torch.randn(1, C, generator=g), ttnn.bfloat16), cm0.alloc_state(0), 0,
                     row(torch.randn(1, C, generator=g), ttnn.float32), ttnn.bfloat16)  # fmt: skip
    assert F.twin(h) is not None and rA.attn.FUSED and rA.attn.wo is not None
    xs = ta.make_inputs(10, is_global, seed=3)
    xs = [xs[n] for n in ("xq", "xk", "xv")][: 3 if is_global else 1]

    def step_a(resid, phase):
        rA.positions.prepare()
        x = rA.x
        a = rA.attn.decode(x[0], rA.state, rA.positions, *x[1:])
        y = conv.conv_add(a, ringA, phase, resid, ttnn.float32, row_out=False)
        return F.twin(y)[0]

    def step_b(resid, phase):
        rA.positions.prepare()
        x = rA.x
        y = rA.attn.decode(x[0], stB, rA.positions, *x[1:], post=(conv, ringB, phase, resid))
        return F.twin(y)[0]

    bad = []
    for t in range(10):
        resid = None if t % 3 == 2 else h  # None: the block-boundary form (conv only, fp32 out)
        rA.set_inputs(t, [x[t] for x in xs])
        ya = dense(step_a(resid, t % 4))
        yb = dense(step_b(resid, t % 4))
        assert ya.abs().max() > 0
        same_ring = all(torch.equal(dense(p), dense(q)) for p, q in zip(ringA, ringB))
        if not (torch.equal(ya, yb) and same_ring):
            bad.append((t, (ya - yb).abs().max().item(), same_ring))
    assert not bad, bad

    # traced: capture the fused step at phase 2, replay at positions 10, 14, 18; eager unfused alongside
    def reset():
        for t in rA.state.tensors() + stB.tensors() + ringA + ringB:
            ttnn.copy(ttnn.zeros_like(t), t)

    rA.set_inputs(10, [x[0] for x in xs])
    tr = ttnn.begin_trace_capture(device, cq_id=0)
    out = step_b(h, 2)
    ttnn.end_trace_capture(device, tr, cq_id=0)
    reset()
    try:
        for i, pos in enumerate((10, 14, 18)):
            rA.set_inputs(pos, [x[i] for x in xs])
            ya = dense(step_a(h, 2))
            rA.set_inputs(pos, [x[i] for x in xs])
            ttnn.execute_trace(device, tr, cq_id=0, blocking=True)
            yb = dense(out)
            assert torch.equal(ya, yb), (pos, (ya - yb).abs().max().item())
    finally:
        ttnn.release_trace(device, tr)
