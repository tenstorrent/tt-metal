# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit check of check_gates.py (PLAN section 9 v1.17, D6 a): the measured record passes, broken ones fail."""

import copy
import json

from models.experimental.flowstate_forecast.tests.perf.check_gates import DEFAULT_GATES, LOWER_IS_BETTER, check

OPT = DEFAULT_GATES.parent
MEASURED = OPT / "iter_01" / "r1_2f" / "variance" / "base.json"  # pre-R1-2f, carries inactive_max_err
BASE = OPT / "iter_01" / "r1_1" / "b2b_base.json"


def _load():
    gates = json.loads(DEFAULT_GATES.read_text())
    gates["drift_by"] = {k: {m: t for m, t in v.items() if m != "abs_err_max"} for k, v in gates["drift_by"].items()}
    gates["drift"].pop("abs_err_max", None)  # its fixed record postdates these records
    cand = [dict(r, git_head="x", git_dirty="") for r in json.loads(MEASURED.read_text())]
    return gates, cand, json.loads(BASE.read_text())


def test_lower_is_better_set():
    assert {"inactive_max_err", "abs_err_max", "flat_max_err", "median_err"} <= LOWER_IS_BETTER


def test_measured_passes_floors():
    gates, cand, base = _load()
    fails = [f for f in check(cand, base, gates) if "inactive_max_err" in f]
    assert not fails, fails


def test_inactive_collapsed_fails():
    """An inactive row off by the collapsed-output error (D6: 0.469 / 0.1116 / 0.349) fails every shape."""
    gates, cand, base = _load()
    for r, err in zip(cand, [0.469, 0.469, 0.1116, 0.1116, 0.349, 0.349]):
        broken = copy.deepcopy(r)
        broken["inactive_max_err"] = err
        assert any("inactive_max_err" in f for f in check([broken], base, gates)), (r["name"], r["precision"])


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
