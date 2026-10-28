# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit check of check_gates.py (PLAN §9 v1.17, v1.22-v1.26, D3 items 2, 3, 5).

The records are made here, from two test goldens scored against themselves plus small noise (both precisions), so
the checks do not depend on a fixed record on disk: the measured-like record passes, and each broken variant fails
its named check."""

import copy
import json

import pytest
import torch

from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS
from models.experimental.flowstate_forecast.tests.pcc import metrics as M
from models.experimental.flowstate_forecast.tests.perf.check_gates import (
    DEFAULT_GATES,
    LOWER_IS_BETTER,
    REQUIRED_FLOORS,
    check,
)

HEAD, T0 = "f" * 40, 1_000_000.0
CASES = ("model16", "nan_mix")


def _record(case, precision, noise):
    g = torch.load(GOLDENS / f"{case}.pt", weights_only=True)
    gen = torch.Generator().manual_seed(3)
    gh, gq = M.head(g["combined_normalized"]), M.head(g["quantile_outputs"])
    h = gh + noise * torch.randn(gh.shape, generator=gen)
    m = M.gate_metrics(h, gh, gq, gq, g["prediction_outputs"], g["prediction_outputs"])
    return {"name": f"suite:{case}", "precision": precision, **m, "git_head": HEAD, "git_dirty": "", "started_at": T0 + 5}


@pytest.fixture(scope="module")
def recs():
    gates = json.loads(DEFAULT_GATES.read_text())
    base = [_record(c, p, 1e-4) for c in CASES for p in ("accuracy", "performance")]
    return gates, base


def _check(cand, base, gates, **kw):
    return check(cand, base, gates, commit_times={HEAD: T0}, **kw)


def test_lower_is_better_set():
    assert {"inactive_max_err", "abs_err_max", "bias_row_p50", "bias_row_max", "bias_row_drift_max"} <= LOWER_IS_BETTER


def test_measured_passes(recs):
    gates, base = recs
    assert not _check(copy.deepcopy(base), base, gates)


def test_missing_case_or_precision_fails(recs):
    gates, base = recs
    assert any("missing from the candidate set" in f for f in _check(copy.deepcopy(base[1:]), base, gates))


def test_missing_floor_family_fails(recs):
    gates, base = recs
    for metric in sorted(REQUIRED_FLOORS):
        g = copy.deepcopy(gates)
        for v in g["floors"].values():
            v.pop(metric, None)
        assert any(f"{metric}: no floor" in f for f in _check(copy.deepcopy(base), base, g)), metric
    g = copy.deepcopy(gates)
    del g["drift"]["bias_row_drift_max"]
    assert any("no drift" in f for f in _check(copy.deepcopy(base), base, g))


def test_one_row_shift_fails_row_drift(recs):
    """+0.05 on one active row: bias_row_drift_max (0.03) fails."""
    gates, base = recs
    for r in base:
        c = copy.deepcopy(r)
        i = next(i for i, v in enumerate(c["bias_rows"]) if v is not None)
        c["bias_rows"][i] += 0.05
        assert any("bias_row_drift_max" in f for f in _check([c], [r], gates, require_coverage=False))


def test_one_row_band_fails_row_drift(recs):
    gates, base = recs
    for r in base:
        c = copy.deepcopy(r)
        i = next(i for i, v in enumerate(c["spread_ratio_rows"]) if v is not None)
        c["spread_ratio_rows"][i] *= 1.3
        assert any("spread_ratio_row_drift_max" in f for f in _check([c], [r], gates, require_coverage=False))


def test_p50_drift_centred_on_one(recs):
    """A candidate at the reference (p50 = 1) passes even when the baseline sat at 0.975; moving away fails."""
    gates, base = recs
    r = next(x for x in base if x["active_rows"] >= 8)
    b = dict(r, spread_ratio_p50=0.975)
    assert not [f for f in _check([dict(r, spread_ratio_p50=1.0)], [b], gates, require_coverage=False) if "p50" in f]
    assert any("p50" in f for f in _check([dict(r, spread_ratio_p50=0.96)], [b], gates, require_coverage=False))


def test_inactive_collapsed_fails(recs):
    gates, base = recs
    for r in base:
        if r["excluded_rows"]:
            fails = _check([dict(r, inactive_max_err=0.5)], [r], gates, require_coverage=False)
            assert any("inactive_max_err" in f for f in fails)


def test_gross_element_fails(recs):
    gates, base = recs
    r = base[0]
    assert any("abs_err_max" in f for f in _check([dict(r, abs_err_max=r["abs_err_max"] * 1.6)], [r], gates, require_coverage=False))


def test_missing_drift_metric_fails(recs):
    gates, base = recs
    r = {k: v for k, v in base[0].items() if k != "pcc_normalized"}
    assert any("pcc_normalized" in f for f in _check([r], [base[0]], gates, require_coverage=False))


def test_stamp_failures(recs):
    gates, base = recs
    r = base[0]
    one = lambda c: _check([c], [r], gates, require_coverage=False)  # noqa: E731
    assert any("dirty" in f for f in one(dict(r, git_dirty=" M tt/model.py")))
    assert any("dirty" in f for f in one(dict(r, git_dirty="?? tt/new.py")))
    assert any("started before its commit" in f for f in one(dict(r, started_at=T0 - 1)))
    assert any("session stamp" in f for f in one({k: v for k, v in r.items() if k != "started_at"}))
