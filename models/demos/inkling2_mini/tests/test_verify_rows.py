# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""tt/verify_rows.py RowsStep on the real model (weights from INKLING_CACHE): one R-row step vs R 1-row model.step
calls at the same positions, with attention and the routed experts replaced by fixed rows in both (the R-row parts
of those are not this module's). Every row's logits, token, router outputs and every app's output bit-equal; then
commit(n) leaves the conv rings as n 1-row steps do. Also times both (eager, so host-bound: a smoke number)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt import model as MD
from models.demos.inkling2_mini.tt.attention_fast import Inkling2AttentionFast
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.verify_rows import RowsStep

C = MD.HIDDEN


def _host(t):
    tw = F.twin(t)
    if tw is not None:
        return F.from_dense_matrix(ttnn.to_torch(tw[0]).float(), C, tw[1]).clone()
    return ttnn.to_torch(t).float().reshape(-1, t.shape[-1]).clone()


@pytest.mark.skipif(not os.environ.get("INKLING_CACHE"), reason="needs the converted weights (INKLING_CACHE)")
@pytest.mark.parametrize("n_rows", [3])
def test_rows_step(device, n_rows):
    m = MD.Inkling2Model(device, root=os.environ["INKLING_CACHE"], max_ctx=4096, log=lambda s: None)
    g = torch.Generator().manual_seed(4)
    warm, pos0 = 3, 3
    n_pos = warm + n_rows
    A = {(L, p): torch.randn(n_pos, C, generator=g) for L, p in MD.APPS}  # attention output per app and position
    Y = {(L, p): torch.randn(n_pos, C, generator=g) * 0.3 for L, p in MD.APPS}  # expert output
    toks = [int(t) for t in torch.randint(0, 200000, (n_pos,), generator=g)]

    def dram(v):
        return ttnn.from_torch(v.reshape(1, 1, -1, C), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def with_twin(v):
        v = v.reshape(-1, C)
        # DRAM, not L1 as the real expert output: conv reads the twin either way, and ~300 fake rows stay alive
        t = ttnn.from_torch(v.reshape(1, 1, -1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        F.set_twin(t, F.dense_tensor(device, v, ttnn.bfloat16), v.shape[0])
        return t

    keep = []  # device tensors the fakes hand out stay alive
    state = {"pos": 0, "app": 0, "route": []}
    apps_of = {}
    for i, (L, p) in enumerate(MD.APPS):
        apps_of.setdefault(L, []).append(p)

    # 1-row fakes, patched into every layer: attention -> post conv of A[app][pos]; MoE -> route, then Y[app][pos]
    orig_dec, orig_moe = {}, {}
    for L, lay in enumerate(m.layers):

        def dec(xq, st, positions, xk=None, xv=None, post=None, L=L):
            p = MD.APPS[state["app"]][1]
            a = dram(A[(L, p)][state["pos"]])
            keep.append(a)
            return Inkling2AttentionFast._post(a, post)

        def moe(x, normed=False, L=L, lay=lay):
            p = MD.APPS[state["app"]][1]
            w, ids = lay.moe.route(x)
            state["route"].append(
                (_host(w)[0, :8].clone(), ttnn.to_torch(ids).reshape(-1, ids.shape[-1])[0, :8].clone())
            )
            y = with_twin(Y[(L, p)][state["pos"]])
            keep.append(y)
            state["app"] += 1
            return y, ids

        orig_dec[L], orig_moe[L] = lay.attn.decode, lay.moe.__call__
        lay.attn.decode = dec
        lay.moe.__class__ = type(
            "FakeMoE", (lay.moe.__class__,), {"__call__": lambda self, x, normed=False, f=moe: f(x, normed)}
        )

    def one_step(pos):
        state["pos"], state["app"] = pos, 0
        m.set_inputs(toks[pos], pos)
        out = m.step()
        return _host(out["logits"]).reshape(1, -1)[:, : MD.VOCAB_REAL], int(ttnn.to_torch(out["token"]).reshape(-1)[0])

    m.reset()
    for pos in range(warm):
        one_step(pos)
    state["route"] = []
    t0 = time.perf_counter()
    ref = [one_step(pos) for pos in range(pos0, pos0 + n_rows)]
    t_one = time.perf_counter() - t0
    ref_route = state["route"]
    rings_ref = {c: [_host(s) for s in st[1] + st[2]] for c, st in m.states.items()}

    # R-row step from the same state
    m.reset()
    for pos in range(warm):
        one_step(pos)
    rs = RowsStep(m, n_rows)
    rs.set_inputs(toks[pos0 : pos0 + n_rows], pos0)
    route_rows, cnt = [], {"app": 0}

    def attention(L, p, xq, xk, xv):
        a = dram(A[(L, p)][pos0 : pos0 + n_rows])
        keep.append(a)
        return a

    def experts(L, x, w, ids):
        p = MD.APPS[cnt["app"]][1]
        cnt["app"] += 1
        route_rows.append((_host(w)[:, :8].clone(), ttnn.to_torch(ids).reshape(-1, ids.shape[-1])[:, :8].clone()))
        y = with_twin(Y[(L, p)][pos0 : pos0 + n_rows])
        keep.append(y)
        return y

    t0 = time.perf_counter()
    got = rs.step(attention, experts)
    t_rows = time.perf_counter() - t0
    print(f"eager: {n_rows} x 1-row {t_one * 1e3:.0f} ms, R-row {t_rows * 1e3:.0f} ms")
    for r in range(n_rows):
        lg = _host(got[r][0]).reshape(1, -1)[:, : MD.VOCAB_REAL]
        assert torch.equal(lg, ref[r][0]), (r, float((lg - ref[r][0]).abs().max()))
        assert int(ttnn.to_torch(got[r][1]).reshape(-1)[0]) == ref[r][1]
    n_apps = len(MD.APPS)
    for i in range(n_apps):
        for r in range(n_rows):
            w1, i1 = ref_route[r * n_apps + i]
            assert torch.equal(route_rows[i][0][r], w1) and torch.equal(route_rows[i][1][r], i1), (i, r)
    # rings: untouched by the step; commit(n) = n 1-row steps
    rs.commit(n_rows)
    for c, st in m.states.items():
        for k, s in enumerate(st[1] + st[2]):
            assert torch.equal(_host(s), rings_ref[c][k]), ("ring", c, k)
    for L, lay in enumerate(m.layers):
        lay.attn.decode = orig_dec[L]
