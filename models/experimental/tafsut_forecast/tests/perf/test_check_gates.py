# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""check_gates.py unit tests (TiRex-2 651e8c4af93 pattern, on synthetic records; no device)."""

import copy
from datetime import datetime, timezone

from models.experimental.tafsut_forecast.tests.perf import check_gates as CG
from models.experimental.tafsut_forecast.tests.perf.check_gates import LOWER_IS_BETTER, check_coverage

COMMITTED = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def check(cand, base, gates, **kw):
    """check_gates.check with a fake commit time for the synthetic head "x" (no git lookup)."""
    return CG.check(cand, base, gates, commit_time=lambda h: COMMITTED if h == "x" else None, **kw)


GATES = {
    "floors": {"*/accuracy": {"pcc_normalized": 0.999, "abs_err_max": 0.5, "spread_ratio_p50": [0.98, 1.02]}},
    "drift": {"pcc_spread": {"abs": 0.001}, "spread_ratio_p50": {"two_sided": True, "abs": 0.01}},
}
REC = {
    "name": "paper",
    "precision": "accuracy",
    "git_head": "x",
    "git_dirty": "",
    "started_at": "2026-10-28T12:05:00+00:00",
    "pcc_normalized": 0.9999,
    "abs_err_max": 0.1,
    "spread_ratio_p50": 1.0,
    "pcc_spread": 0.9999,
}


def test_lower_is_better_set():
    assert {
        "inactive_max_err",
        "abs_err_max",
        "flat_max_err",
        "median_err",
        "spread_ratio_max",
        "bias_row_p50",
        "bias_row_max",
        "bias_row_p95",
    } <= LOWER_IS_BETTER


def test_passes():
    assert not check([REC], [REC], GATES)


def test_floor_and_drift_fail():
    assert check([dict(REC, pcc_normalized=0.99)], [REC], GATES)
    assert check([dict(REC, abs_err_max=0.9)], [REC], GATES)
    assert check([dict(REC, spread_ratio_p50=1.015)], [REC], GATES)  # in range, outside two-sided drift
    assert check([dict(REC, pcc_spread=0.99)], [REC], GATES)


def test_not_applicable_skips_floor_only():
    assert not check(
        [dict(REC, spread_ratio_p50=None)], [dict(REC, spread_ratio_p50=None)], {"floors": GATES["floors"], "drift": {}}
    )


def test_missing_drift_metric_fails():
    r = copy.deepcopy(REC)
    del r["pcc_spread"]
    assert any("pcc_spread" in f for f in check([r], [REC], GATES))


def test_dirty_or_unstamped_fails():
    assert any("dirty" in f for f in check([dict(REC, git_dirty=" M tt/model.py")], [REC], GATES))
    r = {k: v for k, v in REC.items() if k not in ("git_head", "git_dirty")}
    assert any("stamp" in f for f in check([r], [REC], GATES))
    r = {k: v for k, v in REC.items() if k != "started_at"}
    assert any("stamp" in f for f in check([r], [REC], GATES))


def test_started_before_commit_fails():
    """v1.23: a record whose job started before its commit is not evidence; an unknown commit fails too."""
    assert any(
        "before its commit" in f for f in check([dict(REC, started_at="2026-10-28T11:59:59+00:00")], [REC], GATES)
    )
    assert not check([dict(REC, started_at="2026-10-28T12:00:00+00:00")], [REC], GATES)
    assert any("unknown" in f for f in check([dict(REC, git_head="nosuch")], [REC], GATES))


F2 = {"f2_pcc_normalized": 0.9999, "f2_bias_row_max": 0.01}
GATES_F2 = {
    "floors": {"*/accuracy": dict(GATES["floors"]["*/accuracy"], f2_pcc_normalized=0.999, f2_bias_row_max=0.03)},
    "drift": dict(GATES["drift"], f2_bias_row_max={"abs": 0.01}),
}


def test_f2_family_gated():
    """v1.24 / D3 item 1: forward-2 keys are gated with their base metric's direction; a family without floors or
    drift fails."""
    r = dict(REC, **F2)
    assert not check([r], [r], GATES_F2)
    assert check([dict(r, f2_pcc_normalized=0.99)], [r], GATES_F2)
    assert check([dict(r, f2_bias_row_max=0.05)], [r], GATES_F2)  # lower is better, floor and drift
    assert not check([dict(r, f2_bias_row_max=0.0)], [r], GATES_F2)
    assert not check([REC], [REC], GATES_F2)  # a single-forward case skips the f2_* floors
    assert any("f2_pcc_normalized" in f for f in check([REC], [r], GATES_F2))  # dropped forward 2 fails
    assert any("no floors" in f for f in check([r], [r], GATES))
    assert any("no drift" in f for f in check([r], [r], dict(GATES_F2, drift=GATES["drift"])))


def test_bias_row_max_gated():
    """D3 item 3: a shift on a minority of rows moves bias_row_max past its floor and drift."""
    g = {"floors": {"*/accuracy": {"bias_row_max": 0.03}}, "drift": {"bias_row_max": {"abs": 0.01}}}
    r = dict(REC, bias_row_max=0.01)
    assert not check([r], [r], g)
    assert len(check([dict(r, bias_row_max=0.06)], [r], g)) == 2
    assert len(check([dict(r, bias_row_max=0.025)], [r], g)) == 1  # under the floor, over the drift


ROWS = {"row_active_idx": [0, 3, 7], "bias_row_vec": [0.0, -0.056, 0.01], "spread_ratio_row_vec": [1.0, 1.1, 0.9]}
GATES_ROW = {"floors": {"*/accuracy": {"bias_row_drift_max": 0.03, "spread_ratio_row_drift_max": 0.15}}, "drift": {}}


def test_row_drift_gated():
    """D4 items 1-2: a +0.05 shift on one row, or a 1.3x band on one row, fails the row-indexed drift, also on a
    row whose own bias is already large (no relative term); missing vectors or a different active set fail."""
    r = dict(REC, **ROWS)
    assert not check([r], [r], GATES_ROW)
    assert CG.row_drift(r, r) == {
        "bias_row_drift_max": 0.0,
        "bias_row_drift_max_row": 0,
        "spread_ratio_row_drift_max": 0.0,
        "spread_ratio_row_drift_max_row": 0,
    }
    shifted = dict(r, bias_row_vec=[0.0, -0.006, 0.01])  # +0.05 on row 3 (e0 -0.056)
    assert any("bias_row_drift_max" in f for f in check([shifted], [r], GATES_ROW))
    assert CG.row_drift(shifted, r)["bias_row_drift_max_row"] == 3
    assert not check([dict(r, bias_row_vec=[0.02, -0.07, 0.0])], [r], GATES_ROW)  # under 0.03 on every row
    band = dict(r, spread_ratio_row_vec=[1.0, 1.43, 0.9])  # 1.3x on row 3
    assert any("spread_ratio_row_drift_max" in f for f in check([band], [r], GATES_ROW))
    assert check([REC], [r], GATES_ROW)  # no vectors: missing
    assert check([dict(r, row_active_idx=[0, 3, 8])], [r], GATES_ROW)  # another active set: missing
    f2 = {"f2_" + k: v for k, v in ROWS.items()}
    g2 = {"floors": {"*/accuracy": {"f2_bias_row_drift_max": 0.03}}, "drift": {"f2_bias_row_drift_max": None}}
    r2 = dict(REC, **f2)
    g2["drift"] = {"f2_pcc_normalized": {"abs": 1}}
    assert any("f2_bias_row_drift_max" in f for f in check([dict(r2, f2_bias_row_vec=[0.05, -0.056, 0.01])], [r2], g2))


def test_spread_ratio_p50_toward_one():
    """D4 item 5: |p50 - 1| <= |p50_0 - 1| + tol: a fixed record off 1 passes itself and moves toward 1 freely."""
    g = {"floors": {}, "drift": {"spread_ratio_p50": {"abs": 0.01, "toward_one": True}}}
    r = dict(REC, spread_ratio_p50=1.0322)
    assert not check([r], [r], g)
    assert not check([dict(r, spread_ratio_p50=1.0)], [r], g)
    assert not check([dict(r, spread_ratio_p50=0.96)], [r], g)  # 0.04 from 1, within 0.0322 + 0.01
    assert check([dict(r, spread_ratio_p50=1.045)], [r], g)
    assert check([dict(r, spread_ratio_p50=0.955)], [r], g)


def test_stamp_covers_gate_files():
    """D4 item 4: the porcelain stamp includes the floor files."""
    assert "doc/optimize/gates.json" in CG.GATE_FILES


def test_shift_rows_sizes():
    from models.experimental.tafsut_forecast.tests.perf import score_broken as SB

    assert [SB.n_shift_rows(b) for b in (4, 16, 32, 64, 512, 1024)] == [1, 1, 2, 4, 6, 11]


def test_coverage():
    """v1.24: the candidate set must hold every case and precision of the fixed record."""
    perf = dict(REC, precision="performance")
    assert not check_coverage([REC, perf], [REC, perf])
    assert check_coverage([REC], [REC, perf]) == ["paper/performance: missing from the candidate set"]


def _g1():
    import json
    from pathlib import Path

    from models.experimental.tafsut_forecast.tests.perf.check_gates import DEFAULT_BASELINES, load_records

    doc = Path(__file__).resolve().parents[2] / "doc" / "optimize"
    return json.loads((doc / "gates.json").read_text()), load_records(DEFAULT_BASELINES)


def test_g1_proposal_passes_round0():
    """v1.17: the fixed round-0 bench and suite records pass every proposed floor and their own drift."""
    gates, r0 = _g1()
    assert not CG.check(r0, r0, gates)
    assert not check_coverage(r0, r0)


def test_g1_drift_covers_every_case():
    """v1.22: drift budgets and the abs_err_max ceiling exist, and every bench and suite case has a round-0 record."""
    from models.experimental.tafsut_forecast.tests.perf import score_broken as SB

    gates, r0 = _g1()
    assert {"abs_err_max", "spread_ratio_p50", "bias_row_p50", "bias_row_max", "pcc_step"} <= set(gates["drift"])
    assert {"f2_" + m for m in gates["drift"] if not m.startswith(("pcc_final", "f2_"))} <= set(gates["drift"])
    assert gates["drift"]["abs_err_max"] == {"rel": 0.5}
    assert gates["drift"]["spread_ratio_p50"]["toward_one"]
    for prec in ("accuracy", "performance"):
        f = gates["floors"][f"*/{prec}"]
        assert {p + m for p in ("", "f2_") for m in CG.ROW_DRIFT} <= set(f)
        assert f["bias_row_drift_max"] == f["f2_bias_row_drift_max"] == 0.03  # D4 item 1: absolute, both presets
        assert f["spread_ratio_row_drift_max"] <= 0.15
    have = {(r["name"], r["precision"]) for r in r0}
    assert {(c, p) for c in SB.BENCH + SB.SUITE for p in ("accuracy", "performance")} <= have


def test_level_shift_and_lag1_shapes():
    import torch

    from models.experimental.tafsut_forecast.tests.pcc import metrics as M
    from models.experimental.tafsut_forecast.tests.perf import score_broken as SB

    h = torch.randn(3, 2, M.Q * M.P)
    out = SB.broken(h)
    assert torch.allclose(out["level_shift"][0] - h, torch.full_like(h, SB.LEVEL_SHIFT))
    lag = out["lag1"][0].reshape(3, 2, M.Q, M.P)
    hh = h.reshape(3, 2, M.Q, M.P)
    assert torch.equal(lag[..., 1:], hh[..., :-1]) and torch.equal(lag[..., 0], hh[..., 0])


PENDING_RULING = set()  # D4 ruled shift10pct is not a documented limit: the row-indexed drift catches it


def test_g1_proposal_fails_broken():
    """v1.9 / v1.17 / v1.22: every broken output (level_shift and lag1 included) fails at least one proposed floor
    or drift check, on every case, in both precisions."""
    from models.experimental.tafsut_forecast.tests.perf import score_broken as SB

    gates, r0 = _g1()
    results = SB.score(gates, r0)  # skips when the goldens are missing
    assert {r["broken"] for r in results} >= {"level_shift", "lag1", "sign_flip", "collapsed", "shift10pct"}
    assert {(r["shape"], r["forward"]) for r in results} >= {("long_horizon", 2), ("horizon4", 2)}
    passed = [f"{r['shape']}/f{r['forward']}/{r['precision']}/{r['broken']}" for r in results if not r["fails"]]
    assert set(passed) <= PENDING_RULING, passed
