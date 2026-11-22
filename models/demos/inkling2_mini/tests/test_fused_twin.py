# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""E10c dense twins: one layer application of the fused residual path (model._app dataflow, attention / MoE
replaced by fixed rows) with twins on must be bit-equal to twins off (every input gathered from its row), traced
twice bit-equal, and is timed both ways. Device ops before/after are the same programs, so op count is unchanged."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt import residual as R
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.attnres import FusedAttnRes, FusedRowOps
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv
from models.demos.inkling2_mini.tt.fused.gate import FusedMessageGate, FusedQKVGate
from models.demos.inkling2_mini.tt.fused.norm import FusedRowNorm

C = 2304
TRACE_PARAMS = [{"trace_region_size": 32 * 1024 * 1024}]
RESULTS = os.environ.get("INKLING_E10_RESULTS")


def weights(g, is_global):
    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=g) * scale

    nq = 3 if is_global else 1
    w = {
        "attn_res_q": torch.nn.functional.pad(rnd(nq, C, scale=2), (0, 0, 0, 32 - nq))[None, None],
        "mlp_res_q": torch.nn.functional.pad(rnd(1, C, scale=2), (0, 0, 0, 31))[None, None],
        "post_attn_conv": rnd(1, 1, 4, C, scale=0.5),
        "post_mlp_conv": rnd(1, 1, 4, C, scale=0.5),
    }
    for gg in ("q", "k", "v", "mlp"):
        w[f"msg_gate.{gg}.down"] = rnd(1, 1, 2 * C, 64, scale=0.05).bfloat16()
        w[f"msg_gate.{gg}.forget"] = rnd(1, 1, 64, C, scale=0.2)
        w[f"msg_gate.{gg}.gain"] = rnd(1, 1, 64, C, scale=0.2)
    norms = [(1 + rnd(C, scale=0.1)).bfloat16().float() for _ in range(3 if is_global else 1)]
    return w, norms


def host(t):
    return ttnn.to_torch(t).float().reshape(-1, t.shape[-1])


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
@pytest.mark.parametrize("kind", ["local", "local_msg", "global", "global_msg"])
def test_twin_app(device, kind):
    is_global, with_msg = kind.startswith("global"), kind.endswith("msg")
    g = torch.Generator().manual_seed(21)
    w, norms = weights(g, is_global)
    rows, frows = R.RowOps(device), FusedRowOps(device)
    nq = 3 if is_global else 1
    attn, mlp = FusedAttnRes(frows, w, "attn_res_q", nq), FusedAttnRes(frows, w, "mlp_res_q", 1)
    ca, cm = FusedShortConv(rows, w, "post_attn_conv"), FusedShortConv(rows, w, "post_mlp_conv")
    qkv, mg = FusedQKVGate(rows, w), FusedMessageGate(rows, w, "mlp")
    norm = FusedRowNorm(device, norms)
    ring_a, ring_m = ca.alloc_state(0), cm.alloc_state(0)

    def row(t, dtype):
        return ttnn.from_torch(t.reshape(1, 1, -1, C), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    A, Y = row(torch.randn(1, C, generator=g), ttnn.bfloat16), row(torch.randn(1, C, generator=g), ttnn.bfloat16)
    msgs = (row(torch.randn(3, C, generator=g), ttnn.float32), row(torch.randn(1, C, generator=g), ttnn.float32))
    for k in range(2):
        frows.seal(None, row(torch.randn(1, C, generator=g) * (k + 1), ttnn.bfloat16), k)
    # the app input h: a fused output (has a twin), as in the model
    cm0 = FusedShortConv(rows, w, "post_mlp_conv")  # the previous layer's conv: its own output buffer
    h0 = cm0.conv_add(Y, cm0.alloc_state(0), 0, row(torch.randn(1, C, generator=g), ttnn.float32), ttnn.bfloat16)

    seen = {}

    def app(h, skip_rows=False):
        rep = 3 if (with_msg and not is_global) else None
        # x feeds only the fused norm without messages (the gate's combine reads its rows); resid feeds only fused ops
        x = attn(None, h, 3, split=False, repeat_rows=rep, row_out=not (skip_rows and not with_msg))
        if with_msg:
            x = qkv(x, msgs[0])
        qkv_rows = norm(x)
        resid = ca.conv_add(A, ring_a, 1, h, ttnn.float32, row_out=not skip_rows)
        m = mlp(None, resid, 3)[0]
        if with_msg:
            m = mg(m, msgs[1])
        out = cm.conv_add(Y, ring_m, 1, resid, ttnn.bfloat16)
        seen.update(h=h, x=x, resid=resid, m=m, out=out)
        return [out, m] + qkv_rows

    def run(twins, skip_rows=False):
        saved = F.twin
        if not twins:
            F.twin = lambda t: None  # every input gathered from its row
        try:
            app(h0, skip_rows)
            first = [host(t) for t in app(h0, skip_rows)]
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            outs = app(h0, skip_rows)
            ttnn.end_trace_capture(device, tid, cq_id=0)
        finally:
            F.twin = saved
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        r1 = [host(t) for t in outs]
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        r2 = [host(t) for t in outs]
        t0 = time.perf_counter()
        for _ in range(200):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        us = (time.perf_counter() - t0) / 200 * 1e6
        ttnn.release_trace(device, tid)
        assert all(torch.equal(a, b) for a, b in zip(r1, r2)), "trace replays differ"
        assert all(torch.equal(a, b) for a, b in zip(first, r1)), "trace differs from eager"
        return r1, us

    off, us_off = run(False)
    on, us_on = run(True)
    missing = [k for k, t in seen.items() if F.twin(t) is None]
    assert not missing, f"fused outputs without a dense twin: {missing}"
    skip, us_skip = run(True, skip_rows=True)
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(
                json.dumps(dict(name=f"twin_app_{kind}", us_gather=us_off, us_twin=us_on, us_twin_norows=us_skip))
                + "\n"
            )
    assert all(torch.equal(a, b) for a, b in zip(on, off)), "twins change the result"
    assert all(torch.equal(a, b) for a, b in zip(skip, off)), "skipping rows changes the result"
