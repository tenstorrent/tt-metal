# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit check of check_gates.py (PLAN section 9 v1.17, v1.20; D2 fix 4) on synthetic records."""

import copy

from pathlib import Path

from models.experimental.kairos_forecast.tests.perf.check_gates import FIXED_RECORDS, LOWER_IS_BETTER, default_baseline
from models.experimental.kairos_forecast.tests.perf.check_gates import check as _check

FLOORS = {
    "pcc_normalized": 0.999,
    "pcc_final": 0.99,
    "route_mismatch_rows": 0,
    "bias_max_abs_diff": 1e-6,
    "bias_row_p50": 0.01,
    "bias_row_max": 0.03,
    "pcc_step": 0.999,
    "bias_row_p50_final": 0.01,
    "bias_row_max_final": 0.03,
    "pcc_step_final": 0.999,
    "inactive_max_err": 0.05,
    "spread_ratio_p50": [0.97, 1.03],
    "spread_ratio_p01": 0.9,
}
GATES = {
    "floors": {"*/accuracy": dict(FLOORS, abs_err_max=0.05), "*/performance": dict(FLOORS, abs_err_max=0.1)},
    "drift": {"pcc_normalized": {"abs": 1e-3}, "route_mismatch_rows": {"abs": 0}, "abs_err_max": {"rel": 0.1}},
}
GOOD = {
    "pcc_normalized": 0.9995,
    "pcc_final": 0.995,
    "route_mismatch_rows": 0,
    "bias_max_abs_diff": 2e-7,
    "min_top3_margin": 1e-3,
    "abs_err_max": 0.02,
    "bias_row_p50": 0.001,
    "bias_row_max": 0.005,
    "pcc_step": 0.9999,
    "bias_row_p50_final": 0.001,
    "bias_row_max_final": 0.004,
    "pcc_step_final": 0.9999,
    "inactive_max_err": 0.01,
    "spread_ratio_p50": 1.0,
    "spread_ratio_p01": 0.98,
    "active_rows": 400,
    "excluded_rows": 50,
    "git_head": "x",
    "git_dirty": "",
    "started_at": 2000.0,
}
TIMES = {"x": 1000.0}


def _rec(name="paper", precision="accuracy", **kw):
    return dict(GOOD, name=name, precision=precision, **kw)


def check(cand, base, gates, **kw):
    return _check(cand, base, gates, commit_times=TIMES, **kw)


def _fails(r, metric):
    return [f for f in check([r], [_rec(r["name"], r["precision"])], GATES) if metric in f]


def test_lower_is_better_set():
    assert {
        "route_mismatch_rows",
        "bias_max_abs_diff",
        "abs_err_max",
        "bias_row_p50",
        "bias_row_max",
        "bias_row_p50_final",
        "bias_row_max_final",
        "inactive_max_err",
    } <= LOWER_IS_BETTER


def test_bias_row_fails():  # v1.24, lower is better
    for k in ("bias_row_p50", "bias_row_max", "bias_row_p50_final"):
        assert _fails(_rec(**{k: 0.05}), k), k


def test_inactive_max_err_fails():
    assert _fails(_rec(inactive_max_err=0.2), "inactive_max_err")
    # not applicable without inactive rows: prints, does not fail
    assert not _fails(_rec(inactive_max_err=0.2, excluded_rows=0), "inactive_max_err")


def test_pcc_step_fails():
    assert _fails(_rec(pcc_step=0.99), "pcc_step")
    assert _fails(_rec(pcc_step_final=0.99), "pcc_step_final")


def test_not_applicable_below_active_rows(capfd):  # capfd: tests/conftest.py holds it
    r = _rec(active_rows=9, spread_ratio_p01=0.1)  # p01 needs >= 100 active rows
    assert not _fails(r, "spread_ratio_p01")
    assert "not applicable (n_active=9 < 100)" in capfd.readouterr().out
    assert _fails(_rec(active_rows=9, spread_ratio_p50=1.2), "spread_ratio_p50")  # p50 applies at >= 8
    assert not _fails(_rec(active_rows=2, spread_ratio_p50=1.2), "spread_ratio_p50")


def test_started_before_commit_fails():  # v1.22 / v1.23
    assert any("started before" in f for f in check([_rec(started_at=999.0)], [_rec()], GATES))
    r = {k: v for k, v in _rec().items() if k != "started_at"}
    assert any("stamp" in f for f in check([r], [_rec()], GATES))


def test_missing_case_or_precision_fails():  # v1.24
    base = [_rec(), _rec(precision="performance"), _rec("ragged")]
    fails = check([_rec(), _rec("ragged")], base, GATES)
    assert any("missing from the candidate set" in f for f in fails)
    assert not check(base, base, GATES)


def test_good_passes():
    for p in ("accuracy", "performance"):
        assert not check([_rec(precision=p)], [_rec(precision=p)], GATES)


def test_route_mismatch_fails():
    for p in ("accuracy", "performance"):
        assert _fails(_rec(precision=p, route_mismatch_rows=1), "route_mismatch_rows")


def test_bias_diff_fails():
    assert _fails(_rec(bias_max_abs_diff=2e-6), "bias_max_abs_diff")


def test_abs_err_max_fails():
    assert _fails(_rec(abs_err_max=0.06), "abs_err_max")


def test_report_only_routing_cannot_fail():
    for r in (_rec("long_horizon_free_s2", free_step=2), _rec("e2e_stream")):
        bad = dict(r, route_mismatch_rows=7, bias_max_abs_diff=1e-3)
        assert not check([bad], [r], GATES), r["name"]
    # step 1 of a free-running rollout is still gated
    assert _fails(_rec("long_horizon_free_s1", free_step=1, route_mismatch_rows=1), "route_mismatch_rows")


def test_report_only_does_not_cover_pcc():
    assert _fails(_rec("e2e_stream", pcc_normalized=0.9), "pcc_normalized")


def test_missing_absolute_floor_fails():
    gates = copy.deepcopy(GATES)
    del gates["floors"]["*/performance"]["pcc_final"]
    r = _rec(precision="performance")
    assert any("no floor" in f for f in check([r], [r], gates))


def test_missing_drift_metric_fails():
    r = _rec()
    del r["abs_err_max"]
    assert any("abs_err_max" in f for f in check([r], [_rec()], GATES))


def test_dirty_or_unstamped_fails():
    assert any("dirty" in f for f in check([_rec(git_dirty=" M tt/model.py")], [_rec()], GATES))
    r = {k: v for k, v in _rec().items() if k not in ("git_head", "git_dirty")}
    assert any("stamp" in f for f in check([r], [_rec()], GATES))


ROW_GATES = dict(GATES, drift={"bias_row_drift_max": {"abs": 0.025}, "bias_row_drift_max_final": {"abs": 0.025}})


def test_row_drift():  # D4 item 1: one row moved by the v1.25 probe fails, in head and final space
    rows = [0.01 * i - 0.08 for i in range(16)]
    base = _rec(bias_rows=rows, bias_rows_final=rows)
    assert not check([copy.deepcopy(base)], [base], ROW_GATES)
    for vec, metric in (("bias_rows", "bias_row_drift_max"), ("bias_rows_final", "bias_row_drift_max_final")):
        for row in (0, 7, 15):
            moved = list(rows)
            moved[row] += 0.05
            fails = check([_rec(**{**base, vec: moved})], [base], ROW_GATES)
            assert any(metric in f and f"row {row}" in f for f in fails), (metric, row)
        small = [x + 0.02 for x in rows]  # every row inside the bound
        assert not check([_rec(**{**base, vec: small})], [base], ROW_GATES)
    # inactive rows (None) are skipped; a vector of another length or a missing one fails
    holes = [None if i % 3 == 0 else x for i, x in enumerate(rows)]
    assert not check(
        [_rec(bias_rows=holes, bias_rows_final=rows)], [_rec(bias_rows=holes, bias_rows_final=rows)], ROW_GATES
    )
    assert check([_rec(bias_rows=rows[:-1], bias_rows_final=rows)], [base], ROW_GATES)
    assert check([_rec(bias_rows_final=rows)], [base], ROW_GATES)


def test_row_drift_against_row_base():  # broken_outputs: the move is taken against the golden itself
    gold = _rec(bias_rows=[0.0] * 8, bias_rows_final=[0.0] * 8)
    dev = _rec(bias_rows=[0.3] + [0.0] * 7, bias_rows_final=[0.0] * 8)  # a fixed record with an outlier row
    shifted = _rec(bias_rows=[0.0, 0.05] + [0.0] * 6, bias_rows_final=[0.0] * 8)
    assert check([shifted], [dev], ROW_GATES, row_base=[gold])
    assert not check([gold], [dev], ROW_GATES, row_base=[gold])


def test_spread_ratio_p50_toward_one():  # D4 item 3
    g = dict(GATES, drift={"spread_ratio_p50": {"toward": 1.0, "abs": 0.01}})
    base = _rec(precision="performance", spread_ratio_p50=1.012)
    assert not check([_rec(precision="performance", spread_ratio_p50=1.0)], [base], g)  # the golden passes
    assert not check([_rec(precision="performance", spread_ratio_p50=0.99)], [base], g)
    assert check([_rec(precision="performance", spread_ratio_p50=1.042)], [base], g)  # 0.03 away from 1
    assert check([_rec(precision="performance", spread_ratio_p50=0.97)], [base], g)


def test_bias_row_max_final_floor_required():  # D4 item 1
    g = copy.deepcopy(GATES)
    del g["floors"]["*/accuracy"]["bias_row_max_final"]
    assert any("bias_row_max_final" in f and "no floor" in f for f in check([_rec()], [_rec()], g))
    assert _fails(_rec(bias_row_max_final=0.05), "bias_row_max_final")


def test_default_baseline_follows_kind():  # D4 item 4
    assert default_baseline(Path("x/suite.jsonl")) == FIXED_RECORDS["suite"]
    assert default_baseline(Path("x/bench.json")) == FIXED_RECORDS["bench"]
