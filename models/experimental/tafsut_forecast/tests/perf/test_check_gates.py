# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""check_gates.py unit tests (TiRex-2 651e8c4af93 pattern, on synthetic records; no device)."""

import copy

from models.experimental.tafsut_forecast.tests.perf.check_gates import LOWER_IS_BETTER, check

GATES = {
    "floors": {"*/accuracy": {"pcc_normalized": 0.999, "abs_err_max": 0.5, "spread_ratio_p50": [0.98, 1.02]}},
    "drift": {"pcc_spread": {"abs": 0.001}, "spread_ratio_p50": {"two_sided": True, "abs": 0.01}},
}
REC = {"name": "paper", "precision": "accuracy", "git_head": "x", "git_dirty": "", "pcc_normalized": 0.9999,
       "abs_err_max": 0.1, "spread_ratio_p50": 1.0, "pcc_spread": 0.9999}


def test_lower_is_better_set():
    assert {"inactive_max_err", "abs_err_max", "flat_max_err", "median_err", "spread_ratio_max"} <= LOWER_IS_BETTER


def test_passes():
    assert not check([REC], [REC], GATES)


def test_floor_and_drift_fail():
    assert check([dict(REC, pcc_normalized=0.99)], [REC], GATES)
    assert check([dict(REC, abs_err_max=0.9)], [REC], GATES)
    assert check([dict(REC, spread_ratio_p50=1.015)], [REC], GATES)  # in range, outside two-sided drift
    assert check([dict(REC, pcc_spread=0.99)], [REC], GATES)


def test_not_applicable_skips_floor_only():
    assert not check([dict(REC, spread_ratio_p50=None)], [dict(REC, spread_ratio_p50=None)], {"floors": GATES["floors"], "drift": {}})


def test_missing_drift_metric_fails():
    r = copy.deepcopy(REC)
    del r["pcc_spread"]
    assert any("pcc_spread" in f for f in check([r], [REC], GATES))


def test_dirty_or_unstamped_fails():
    assert any("dirty" in f for f in check([dict(REC, git_dirty=" M tt/model.py")], [REC], GATES))
    r = {k: v for k, v in REC.items() if k not in ("git_head", "git_dirty")}
    assert any("stamp" in f for f in check([r], [REC], GATES))
