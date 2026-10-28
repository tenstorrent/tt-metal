# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit check of check_gates.py (PLAN §9 v1.17; copied from TiRex-2 651e8c4af93, adapted to FlowState's
round-0 records): the measured record passes, broken ones fail, missing or unstamped ones fail."""

import copy
import json

from models.experimental.flowstate_forecast.tests.perf.check_gates import (
    DEFAULT_BASELINE,
    DEFAULT_GATES,
    LOWER_IS_BETTER,
    check,
)


def _load():
    gates = json.loads(DEFAULT_GATES.read_text())
    base = json.loads(DEFAULT_BASELINE.read_text())
    return gates, copy.deepcopy(base), base


def test_lower_is_better_set():
    assert {"inactive_max_err", "abs_err_max", "spread_ratio_p99", "spread_ratio_max"} <= LOWER_IS_BETTER


def test_measured_passes():
    gates, cand, base = _load()
    assert not check(cand, base, gates)


def test_inactive_collapsed_fails():
    """An inactive-row error at the collapsed level (0.5 normalized units) fails every record."""
    gates, cand, base = _load()
    for r in cand:
        broken = dict(r, inactive_max_err=0.5)
        assert any("inactive_max_err" in f for f in check([broken], base, gates)), (r["name"], r["precision"])


def test_gross_element_fails():
    gates, cand, base = _load()
    r = dict(cand[0], abs_err_max=cand[0]["abs_err_max"] * 1.6)
    assert any("abs_err_max" in f for f in check([r], base, gates))


def test_missing_drift_metric_fails():
    gates, cand, base = _load()
    r = copy.deepcopy(cand[0])
    del r["pcc_spread"]
    assert any("pcc_spread" in f for f in check([r], base, gates))


def test_dirty_or_unstamped_fails():
    gates, cand, base = _load()
    assert any("dirty" in f for f in check([dict(cand[0], git_dirty=" M tt/model.py")], base, gates))
    r = {k: v for k, v in cand[0].items() if k not in ("git_head", "git_dirty")}
    assert any("stamp" in f for f in check([r], base, gates))
