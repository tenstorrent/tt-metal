# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit check of check_gates.py (PLAN section 9 v1.17, v1.20; D2 fix 4) on synthetic records."""

import copy

from models.experimental.kairos_forecast.tests.perf.check_gates import LOWER_IS_BETTER
from models.experimental.kairos_forecast.tests.perf.check_gates import check as _check

FLOORS = {"pcc_normalized": 0.999, "pcc_final": 0.99, "route_mismatch_rows": 0, "bias_max_abs_diff": 1e-6,
          "bias_row_p50": 0.01, "bias_row_max": 0.03, "pcc_step": 0.999, "bias_row_p50_final": 0.01,
          "pcc_step_final": 0.999, "inactive_max_err": 0.05, "spread_ratio_p50": [0.97, 1.03],
          "spread_ratio_p01": 0.9}
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
    assert {"route_mismatch_rows", "bias_max_abs_diff", "abs_err_max", "bias_row_p50", "bias_row_max",
            "bias_row_p50_final", "bias_row_max_final", "inactive_max_err"} <= LOWER_IS_BETTER


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


def test_not_applicable_below_active_rows(capsys):
    r = _rec(active_rows=9, spread_ratio_p01=0.1)  # p01 needs >= 100 active rows
    assert not _fails(r, "spread_ratio_p01")
    assert "not applicable (n_active=9 < 100)" in capsys.readouterr().out
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
