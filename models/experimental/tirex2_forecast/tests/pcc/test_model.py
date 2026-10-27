# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC of TTNN TiRex-2 on real weights against the upstream fp32 CPU goldens.

Metrics (CONTRACT section 3):
    pcc_normalized  head output (scaled space) at the read-out tokens vs the upstream ``output_patch_embedding``
    pcc_final       single-pass ``[targets, 9, H]`` forecast vs ``q_one``; every median within 1e-2 x its scale

Path coverage: univariate (no variate mixer), multivariate groups (variate attention on 32-row blocks),
groups that do not divide 32 rows (row packing with dummy rows, singleton pass-through mixed with groups),
H = 928 (29 read-out tokens), sign flip batched as 2B rows, and trace replay against the eager forward.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.golden.goldens import golden, have_goldens
from models.experimental.tirex2_forecast.tests.pcc.helpers import host_weights

pytestmark = pytest.mark.skipif(not have_goldens(), reason="goldens not generated")

# Floors: doc/optimize/gates.json (D3 accuracy, D4 performance), read through check_gates; the suite has no
# constants of its own (D4 condition a). Single-series cases keep pcc_normalized >= 0.9975 and pcc_final >= 0.98
# (D3).
SINGLE_SERIES = {"pcc_normalized": 0.9975, "pcc_final": 0.98}
# D5: population band metrics need enough active rows. p50 applies at n_active >= 8, p01 / p99 at >= 100; below
# that the floor-only min / max and the PCC metrics carry the case (suite only, the bench gates everything).
MIN_ACTIVE = {"spread_ratio_p50": 8, "spread_ratio_p01": 100, "spread_ratio_p99": 100}


def applicable(f: dict, n_active: int, name: str = "") -> dict:
    out = {}
    for k, v in f.items():
        if n_active < MIN_ACTIVE.get(k, 0):
            print(f"[PCC] {name}: {k} not applicable (n_active={n_active} < {MIN_ACTIVE[k]})")
        else:
            out[k] = v
    return out


# D6: the suite cases that take a bench shape's per-shape floors (inactive_max_err)
SUITE_SHAPE = {"paper64": "paper", "long_context": "long_context", "multivar4": "multivariate"}


def floors(precision: str, single: bool = False, name: str = "") -> dict:
    import json

    from models.experimental.tirex2_forecast.tests.perf.check_gates import DEFAULT_GATES, _floors

    f = _floors(json.loads(DEFAULT_GATES.read_text()), SUITE_SHAPE.get(name.split("[")[0], "suite"), precision)
    if single:
        f.update(SINGLE_SERIES)
    return f


def model(device, precision="accuracy", variate=True, pad_skip=True):
    from models.experimental.tirex2_forecast.tt.common import Precision
    from models.experimental.tirex2_forecast.tt.model import TtTiRex2

    return TtTiRex2(device, host_weights(), Precision.by_name(precision), variate=variate, pad_skip=pad_skip)


LAST_CUT = []  # the pad cut of every ``run`` (asserted per test: which path ran)


def run(tt, context, horizon, group_ids=None, target_mask=None):
    """-> (head output at the read-out tokens [rows, n, 288] in source order, q_one [targets, 9, H])."""
    import ttnn

    from models.experimental.tirex2_forecast.tt.model import WINDOW

    prep = tt.prepare_inputs(context, horizon, group_ids, target_mask)
    LAST_CUT.append(prep.cut)
    inputs = tt.upload_inputs(prep)
    out = tt.forward_device(inputs)
    tt.deallocate_inputs(inputs)
    win = ttnn.to_torch(out).float().reshape(prep.device_rows, WINDOW[1] - WINDOW[0], -1)
    q = tt.postprocess(out, prep)
    ttnn.deallocate(out)
    readout = [t - WINDOW[0] for t in df.readout_tokens(horizon)]
    head = torch.empty(prep.rows, len(readout), win.shape[-1])
    valid = prep.order >= 0
    head[prep.order[valid]] = win[valid][:, readout]
    return head, q


def check_forecast(
    name, head, q, g_head, g_q, scale, target_mask, precision="accuracy", loc=None, horizon=64, single=False
):
    from models.experimental.tirex2_forecast.tests.pcc import metrics as M

    from models.experimental.tirex2_forecast.tests.perf.check_gates import passes

    m = M.gate_metrics(head, g_head, q, g_q, scale[target_mask])
    f = applicable(floors(precision, single, name), m["active_rows"], name)
    print(f"[PCC] {name}: {M.fmt(m)}")
    print(f"[PCC] {name}: floors (gates.json {precision}{', single series' if single else ''}): {f}")
    if loc is not None:
        c = M.collapsed_metrics(g_head, g_q, loc, scale, horizon, target_mask)
        print(f"[PCC] {name}.collapsed: {M.fmt(c)}")
        assert [k for k, v in f.items() if not passes(k, c[k], v)], "the collapsed output passes every floor"
    failed = {k: (m[k], v) for k, v in f.items() if not passes(k, m[k], v)}
    assert not failed, f"{name}: floors failed {failed}"
    return m


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
def test_forward_16(device, precision):
    """16 univariate series x 2048 -> 64, single pass."""
    g = golden("model16")
    tt = model(device, precision, variate=False)
    head, q = run(tt, g["context"], g["horizon"])
    assert LAST_CUT[-1] == 192  # 2048 steps: 192 pad tokens skipped
    tok = df.readout_tokens(g["horizon"])
    check_forecast(
        f"model16[{precision}]",
        head,
        q,
        g["head_out"][:, tok],
        g["q_one"],
        g["scale"],
        torch.ones(16, dtype=torch.bool),
        precision,
    )


def test_multivariate_4x8(device):
    """4 groups x (1 target + 7 past covariates) x 2048 -> 64: all 12 variate mixers run."""
    g = golden("multivar4")
    groups, per, T = g["context"].shape
    gid = torch.arange(groups).repeat_interleave(per)
    tm = (torch.arange(groups * per) % per) == 0
    tt = model(device)
    head, q = run(tt, g["context"].reshape(-1, T), g["horizon"], gid, tm)
    assert LAST_CUT[-1] == 0  # grouped rows: full path
    tok = df.readout_tokens(g["horizon"])
    check_forecast("multivar4", head, q, g["head_out"][:, tok], g["q_one"], g["scale"], tm, loc=g["loc"])


def test_long_context_h928(device):
    """32 series x 8192 (no left pad) -> 928: 29 read-out tokens."""
    g = golden("bench_long_context")
    tt = model(device, variate=False)
    head, q = run(tt, g["context"], g["horizon"])
    assert LAST_CUT[-1] == 0  # no left pad: full path
    check_forecast(
        "long_context",
        head,
        q,
        g["head_out"],
        g["q_one"],
        g["scale"],
        torch.ones(32, dtype=torch.bool),
        loc=g["loc"],
        horizon=g["horizon"],
    )


def test_single_series(device):
    """One univariate series x 2048 -> 64 (one real row among 31 dummy rows)."""
    g = golden("single")
    tt = model(device, variate=False)
    head, q = run(tt, g["context"], g["horizon"])
    tok = df.readout_tokens(g["horizon"])
    check_forecast(
        "single", head, q, g["head_out"][:, tok], g["q_one"], g["scale"], torch.ones(1, dtype=torch.bool), single=True
    )


@pytest.mark.parametrize("row", [0, 7])
def test_single_long_context(device, row):
    """One series of the long-context bench alone (8192 -> 928): rows are independent, so its golden is a row."""
    g = golden("bench_long_context")
    tt = model(device, variate=False)
    head, q = run(tt, g["context"][row : row + 1], g["horizon"])
    check_forecast(
        f"long_context[row {row}]",
        head,
        q,
        g["head_out"][row : row + 1],
        g["q_one"][row : row + 1],
        g["scale"][row : row + 1],
        torch.ones(1, dtype=torch.bool),
        single=True,
    )


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
def test_paper_rows(device, precision):
    """64 rows of the paper bench (realistic input mix: walks, intermittent, level shifts, trends)."""
    g = golden("bench_paper")
    rows = slice(0, 64)
    tt = model(device, precision, variate=False)
    head, q = run(tt, g["context"][rows], g["horizon"])
    assert LAST_CUT[-1] == 192
    check_forecast(
        f"paper64[{precision}]",
        head,
        q,
        g["head_out"][rows],
        g["q_one"][rows],
        g["scale"][rows],
        torch.ones(64, dtype=torch.bool),
        precision,
        loc=g["loc"][rows],
        horizon=g["horizon"],
    )


def sign_flip_case():
    """model16 as the sign-flip batch ``[x; -x]`` (32 rows, two groups per series pair) and its references: the
    head output of both halves from the CPU dataflow (upstream to 1e-7; the ``x`` half is the golden), the upstream
    two-pass ``q_full``, and ``forecast(head)`` = the sign-flip average of the two single-pass halves (D7 a)."""
    g = golden("model16")
    both = torch.cat([g["context"], -g["context"]])
    n = g["context"].shape[0]
    prep = df.prepare(both, g["horizon"], torch.arange(2 * n), torch.ones(2 * n, dtype=torch.bool))
    ref_head = df.forward(host_weights(), prep)[:, df.readout_tokens(g["horizon"])]

    def forecast(head):
        from models.experimental.tirex2_forecast.tests.pcc import metrics as M

        q = M.forecast_from_head(head, prep.loc, prep.scale, g["horizon"], prep.target_mask)
        return df.sign_flip_average(q[:n], q[n:])

    return g, both, prep, ref_head, forecast


@pytest.mark.parametrize("precision", ["accuracy", "performance"])
def test_sign_flip_16(device, precision):
    """Sign-flip TTA: x and -x as one batch of 32 rows, against the upstream two-pass ``q_full``, gated on every
    head-space floor of model16 (D7 a); the collapsed output must fail them."""
    from models.experimental.tirex2_forecast.tests.pcc import metrics as M

    from models.experimental.tirex2_forecast.tests.perf.check_gates import passes

    g, both, prep, ref_head, forecast = sign_flip_case()
    n = g["context"].shape[0]
    tt = model(device, precision, variate=False)
    head, q = run(tt, both, g["horizon"])
    assert LAST_CUT[-1] == 192
    q = df.sign_flip_average(q[:n], q[n:])
    assert torch.allclose(q, tt.forecast(g["context"], g["horizon"], sign_flip=True), rtol=1e-3, atol=1e-3)
    name = f"sign_flip16[{precision}]"
    m = M.gate_metrics(head, ref_head, q, g["q_full"], g["scale"])
    f = applicable(floors(precision, name=name), m["active_rows"], name)
    print(f"[PCC] {name}: {M.fmt(m)}")
    print(f"[PCC] {name}: floors (gates.json {precision}): {f}")
    c = M.collapsed(ref_head)
    cm = M.gate_metrics(c, ref_head, forecast(c), g["q_full"], g["scale"])
    print(f"[PCC] {name}.collapsed: {M.fmt(cm)}")
    assert [k for k, v in f.items() if not passes(k, cm[k], v)], "the collapsed output passes every floor"
    failed = {k: (m[k], v) for k, v in f.items() if not passes(k, m[k], v)}
    assert not failed, f"{name}: floors failed {failed}"


def test_ragged_groups(device):
    """Groups of 5 and 1 rows (they do not divide 32): dummy-row packing and singleton pass-through.

    No upstream golden has this layout, so the oracle is the CPU dataflow (checked against upstream to 1e-7
    in ``tests/golden/test_dataflow.py``) on the same inputs.
    """
    from models.experimental.tirex2_forecast.tt.model import pack_rows

    g = golden("multivar4")
    ctx = g["context"].reshape(-1, g["context"].shape[-1])
    src = torch.cat([ctx, -ctx[:16]])  # 48 rows of realistic series
    sizes = [5] * 8 + [1] * 8  # the 7th 5-row group would straddle the first 32-row block
    gid = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
    tm = torch.cat([torch.arange(s) == 0 for s in sizes])
    order = pack_rows(gid)
    assert order.numel() == 64 and int((order < 0).sum()) == 16, "packing must insert dummy rows"
    assert int(order[30]) == -1 and int(order[32]) == 30, "the 7th group must start the second block"
    tt = model(device)
    head, q = run(tt, src, 64, gid, tm)
    prep = df.prepare(src, 64, gid, tm)
    ref_head = df.forward(host_weights(), prep)[:, df.readout_tokens(64)]
    ref_q = df.postprocess(ref_head, prep)
    check_forecast("ragged", head, q, ref_head, ref_q, prep.scale, tm, loc=prep.loc)


def test_pad_skip_vs_full(device):
    """model16 through both paths (pad skip at cut 192, and the full 320-token path): each meets the floors
    against the goldens, and they agree with each other."""
    from models.experimental.tirex2_forecast.tests.pcc import metrics as M

    g = golden("model16")
    tok = df.readout_tokens(g["horizon"])
    outs = {}
    for skip in (True, False):
        tt = model(device, variate=False, pad_skip=skip)
        outs[skip] = run(tt, g["context"], g["horizon"])
        assert LAST_CUT[-1] == (192 if skip else 0)
        tt.release()
        check_forecast(
            f"model16[pad_skip={skip}]",
            *outs[skip],
            g["head_out"][:, tok],
            g["q_one"],
            g["scale"],
            torch.ones(16, dtype=torch.bool),
        )
    agree = M.pearson(outs[False][0], outs[True][0])
    print(f"[PCC] model16 pad skip vs full: head pcc {agree:.6f}")
    assert agree >= 0.9995
