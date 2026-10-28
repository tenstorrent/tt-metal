# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit check of check_gates.py (PLAN section 9 v1.17, v1.20; D2 fix 4) on synthetic records."""

import copy

from models.experimental.kairos_forecast.tests.perf.check_gates import LOWER_IS_BETTER, check

FLOORS = {"pcc_normalized": 0.999, "pcc_final": 0.99, "route_mismatch_rows": 0, "bias_max_abs_diff": 1e-6}
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
    "git_head": "x",
    "git_dirty": "",
}


def _rec(name="paper", precision="accuracy", **kw):
    return dict(GOOD, name=name, precision=precision, **kw)


def _fails(r, metric):
    return [f for f in check([r], [_rec(r["name"], r["precision"])], GATES) if metric in f]


def test_lower_is_better_set():
    assert {"route_mismatch_rows", "bias_max_abs_diff", "abs_err_max"} <= LOWER_IS_BETTER


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
