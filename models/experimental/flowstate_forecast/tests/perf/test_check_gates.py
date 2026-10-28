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


def _first_active(r, vec):
    return next(i for i, v in enumerate(r[vec]) if v is not None)


def test_row_drift_toward_reference_passes(recs):
    """v1.27: a row whose baseline bias is 0.1 may move to 0 (toward the reference), not to 0.14 or -0.04."""
    gates, base = recs
    r = base[0]
    i = _first_active(r, "bias_rows")
    b = copy.deepcopy(r)
    b["bias_rows"][i] = 0.1
    one = lambda v: [f for f in _check([dict(r, bias_rows=r["bias_rows"][:i] + [v] + r["bias_rows"][i + 1 :])], [b], gates, require_coverage=False) if "bias_row_drift_max" in f]  # noqa: E731
    assert not one(0.0) and not one(0.12)
    assert one(0.14) and one(-0.04)
    assert any(f"row {i}" in f for f in one(0.14))


def test_spread_row_drift_toward_one_passes(recs):
    gates, base = recs
    r = base[0]
    i = _first_active(r, "spread_ratio_rows")
    b = copy.deepcopy(r)
    b["spread_ratio_rows"][i] = 0.8
    one = lambda v: [f for f in _check([dict(r, spread_ratio_rows=r["spread_ratio_rows"][:i] + [v] + r["spread_ratio_rows"][i + 1 :])], [b], gates, require_coverage=False) if "spread_ratio_row_drift_max" in f]  # noqa: E731
    assert not one(1.0) and not one(1.14)
    assert one(1.16) and one(0.64)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_row_vector_fails_closed(recs, bad):
    """v1.27: NaN / inf / None in a later row against a baseline value fails."""
    gates, base = recs
    r = base[0]
    for vec, metric in (("bias_rows", "bias_row_drift_max"), ("spread_ratio_rows", "spread_ratio_row_drift_max")):
        c = copy.deepcopy(r)
        i = max(i for i, v in enumerate(c[vec]) if v is not None)
        c[vec][i] = bad
        assert any(metric in f and f"row {i}" in f for f in _check([c], [r], gates, require_coverage=False))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_scalar_fails_closed(recs, bad):
    gates, base = recs
    r = base[0]
    for metric in ("pcc_normalized", "abs_err_max", "spread_ratio_p50", "flat_max_err", "pcc_step"):
        assert any(metric in f for f in _check([dict(r, **{metric: bad})], [r], gates, require_coverage=False)), metric


def test_nan_output_is_nan_not_zero():
    """pearson / row_pccs of a NaN output are NaN (fail closed), not 0."""
    import math

    a = torch.randn(2, 9, 8)
    b = a.clone()
    b[1, 0, 0] = float("nan")
    assert math.isnan(M.pearson(a, b)) and math.isnan(float(M.row_pccs(a, b)[1]))


def test_ungated_key_fails(recs):
    """R1-0b: a gated key the record carries with neither floor nor drift fails."""
    gates, base = recs
    g = copy.deepcopy(gates)
    for v in g["floors"].values():
        v.pop("pcc_row_min_var", None)
    assert any("pcc_row_min_var: ungated" in f for f in _check(copy.deepcopy(base), base, g))


def test_pcc_step_drift(recs):
    """R1-0c: pcc_step 0.006 below the baseline fails in both presets, though above the floor."""
    gates, base = recs
    for r in base:
        c = dict(r, pcc_step=r["pcc_step"] - 0.006)
        assert any("pcc_step drift" in f for f in _check([c], [r], gates, require_coverage=False)), r["precision"]
