# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden files (made by ``tests/golden/make_goldens.py`` in the reference venv) and PCC helpers."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import torch

GOLDEN_DIR = Path(os.environ.get("TIMESFM3_GOLDENS", "/home/andy/bringup/goldens/timesfm3"))
MANIFEST = Path(__file__).resolve().parent / "golden" / "MANIFEST.json"


@functools.lru_cache(maxsize=None)
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@functools.lru_cache(maxsize=4)
def load_golden(name: str) -> dict:
    """A golden by name, after checking its sha256 against ``MANIFEST.json``."""
    path = GOLDEN_DIR / f"{name}.pt"
    expected = manifest()["files"][name]["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise RuntimeError(f"{path}: sha256 {h.hexdigest()} != MANIFEST {expected}; regenerate the goldens")
    return torch.load(path, weights_only=True)


def pcc(expected: torch.Tensor, got: torch.Tensor) -> float:
    from tests.ttnn.utils_for_testing import comp_pcc

    return float(comp_pcc(expected.double(), got.double(), 0.0)[1])


def check_pcc(name: str, expected: torch.Tensor, got: torch.Tensor, floor: float, where=None) -> float:
    """Shape, finiteness (everywhere) and PCC (over ``where`` if given) against ``floor``."""
    expected, got = torch.as_tensor(expected), torch.as_tensor(got)
    assert tuple(got.shape) == tuple(expected.shape), f"{name}: shape {tuple(got.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(got).all(), f"{name}: non-finite output"
    if where is not None:
        expected, got = expected[where], got[where]
    value = pcc(expected, got)
    print(f"[PCC] {name}: {value:.6f} (floor {floor})", flush=True)
    assert value >= floor, f"{name}: PCC {value:.6f} < {floor}"
    return value


# ------------------------------------------------------------------------------------------ gate metrics
# PLAN section 9 (v1.1): ``pcc_normalized`` alone is dominated by per-row offsets, so a forecast whose
# quantiles all collapse to their mean can still score ~0.9997. The head output is (rows, F, 64 * 9) in
# normalized space; these metrics look at the quantile spread and at the worst rows.
NUM_QUANTILES = 9
# Rows whose reference head varies by less than this fraction of the whole head's std (e.g. an intermittent
# series forecast flat at zero, std ~0.002 vs 1.19) have an ill-conditioned per-row PCC: their absolute error is
# smaller than a typical row's, yet it is comparable to their own variation. ``*_active`` skips those rows.
ACTIVE_ROW_STD = 0.05
# The gate table: metric -> (">=" or "<=", bound) per precision, used by the PCC suite and the bench records
# (``floor_ok``) and by ``tests/perf/check_gates.py``. Set by the orchestrator in D3 (G1); they change only through an
# orchestrator decision (PLAN section 9).
GATES = {
    "accuracy": {
        "pcc_normalized": (">=", 0.999),
        "pcc_spread": (">=", 0.999),
        "pcc_row_p01": (">=", 0.7),
        "pcc_row_min": (">=", 0.5),
        "pcc_row_p01_active": (">=", 0.99),
        "pcc_row_min_active": (">=", 0.95),
        "spread_ratio_p01": (">=", 0.9),
        "spread_ratio_p50": ("in", (0.97, 1.03)),
        "spread_ratio_p99": ("<=", 1.1),
        "spread_ratio_min": (">=", 0.6),
        "spread_ratio_max": ("<=", 1.25),
        "pcc_final": (">=", 0.99),
        "pcc_final_spread": (">=", 0.999),
    },
    "performance": {
        "pcc_normalized": (">=", 0.995),
        "pcc_spread": (">=", 0.998),
        "pcc_row_p01": (">=", 0.5),
        "pcc_row_min": (">=", 0.3),
        "pcc_row_p01_active": (">=", 0.97),
        "pcc_row_min_active": (">=", 0.85),
        "spread_ratio_p01": (">=", 0.9),
        "spread_ratio_p50": ("in", (0.97, 1.03)),
        "spread_ratio_p99": ("<=", 1.1),
        "spread_ratio_min": (">=", 0.5),
        "spread_ratio_max": ("<=", 1.25),
        "pcc_final": (">=", 0.98),
        "pcc_final_spread": (">=", 0.998),
    },
}
# Allowed drift from round 0 for an accepted candidate (``check_gates.py``), per D3 and D4: higher-is-better metrics
# may fall at most this far, ``spread_ratio_p99`` may rise at most this far, ``spread_ratio_p50`` may move this far
# either way. Min/max metrics and the all-row ``pcc_row_p01`` (D4) are floor-only (no drift).
DRIFT = {
    "pcc_normalized": 0.0005,
    "pcc_spread": 0.0005,
    "pcc_final_spread": 0.0005,
    "pcc_final": 0.01,
    "pcc_row_p01_active": 0.01,
    "spread_ratio_p01": 0.01,
    "spread_ratio_p50": 0.005,
    "spread_ratio_p99": 0.01,
}
# Per-case drift overrides, (case name, metric) -> band (D4: the paper golden's p01 moves ~0.04 under order-only changes).
DRIFT_CASE = {("paper", "spread_ratio_p01"): 0.03}


# D5: ``inactive_err_max_rel`` floor (<=) per case, both precisions; a case is matched by name ("bench_paper[:64]"
# is paper). Not applicable when the golden has no inactive rows (``rows_inactive`` 0): printed as N/A.
INACTIVE_ERR_MAX_REL = {"paper": 0.15, "long_context": 0.06, "multivariate": 0.06}


def inactive_floor(case: str | None) -> float | None:
    """The ``inactive_err_max_rel`` bound of the case named ``case``, or None if it has none."""
    hits = [b for k, b in INACTIVE_ERR_MAX_REL.items() if case and k in case]
    return hits[0] if len(hits) == 1 else None


def gate_failures(metrics: dict, precision: str, case: str | None = None) -> dict:
    """The metrics outside their gate: {metric: (value, op, bound)}. With ``case``, also ``inactive_err_max_rel``
    against the case's floor when the metrics have inactive rows; a case with inactive rows and no floor fails."""
    out = {}
    if case is not None:
        if "rows_inactive" not in metrics:
            out["rows_inactive"] = (None, ">=", 0)
        elif metrics["rows_inactive"] > 0:
            bound, v = inactive_floor(case), metrics.get("inactive_err_max_rel")
            if bound is None or v is None or not v <= bound:
                out["inactive_err_max_rel"] = (v, "<=", bound)
    for k, (op, bound) in GATES[precision].items():
        if k not in metrics:
            out[k] = (None, op, bound)
            continue
        v = metrics[k]
        if (
            (op == ">=" and not v >= bound)
            or (op == "<=" and not v <= bound)
            or (op == "in" and not bound[0] <= v <= bound[1])
        ):
            out[k] = (v, op, bound)
    return out


def _pcc_rows(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Row-wise PCC of two (rows, n) float64 tensors; a constant row scores 1 if both are equal, else 0."""
    e = e - e.mean(-1, keepdim=True)
    g = g - g.mean(-1, keepdim=True)
    den = e.norm(dim=-1) * g.norm(dim=-1)
    num = (e * g).sum(-1)
    out = torch.where(den > 0, num / den.clamp_min(1e-300), torch.zeros_like(num))
    same = (e.abs().amax(-1) == 0) & (g.abs().amax(-1) == 0)
    return torch.where(same, torch.ones_like(out), out)


def active_rows(golden_head: torch.Tensor) -> torch.Tensor:
    """(rows,) bool: the golden head's rows whose std is >= ACTIVE_ROW_STD of the whole golden head's std. Taken
    from the full golden only, so the set (and the excluded count) is fixed per golden; slice it for subsets."""
    e = torch.as_tensor(golden_head).double().reshape(golden_head.shape[0], -1)
    return e.std(-1) >= ACTIVE_ROW_STD * e.std()


def head_metrics(expected: torch.Tensor, got: torch.Tensor, active: torch.Tensor | None = None) -> dict:
    """``pcc_normalized`` (all values), ``pcc_spread`` (the head minus its mean over the 9 quantiles),
    ``pcc_row_p01`` / ``pcc_row_min`` (per-row PCC, 1st percentile and minimum) of (rows, F, 576) heads."""
    e = torch.as_tensor(expected).double().reshape(expected.shape[0], -1, NUM_QUANTILES)
    g = torch.as_tensor(got).double().reshape(got.shape[0], -1, NUM_QUANTILES)
    flat = lambda t: t.reshape(1, -1)  # noqa: E731
    spread = lambda t: t - t.mean(-1, keepdim=True)  # noqa: E731
    er, gr = e.reshape(e.shape[0], -1), g.reshape(g.shape[0], -1)
    rows = _pcc_rows(er, gr)
    active = active_rows(expected) if active is None else torch.as_tensor(active)
    act = rows[active] if active.any() else rows
    # Per-row forecast band amplitude: |h - mean_q h| of the device over the reference's (active rows).
    es, gs = spread(e).reshape(e.shape[0], -1), spread(g).reshape(g.shape[0], -1)
    ratio = gs.norm(dim=-1) / es.norm(dim=-1).clamp_min(1e-30)
    ratio = ratio[active] if active.any() else ratio
    return {
        "spread_ratio_p01": float(torch.quantile(ratio, 0.01)),
        "spread_ratio_p50": float(torch.quantile(ratio, 0.5)),
        "spread_ratio_p99": float(torch.quantile(ratio, 0.99)),
        "spread_ratio_min": float(ratio.min()),
        "spread_ratio_max": float(ratio.max()),
        "pcc_normalized": float(_pcc_rows(flat(e), flat(g))[0]),
        "pcc_spread": float(_pcc_rows(flat(spread(e)), flat(spread(g)))[0]),
        "pcc_row_p01": float(torch.quantile(rows, 0.01)),
        "pcc_row_min": float(rows.min()),
        "pcc_row_p01_active": float(torch.quantile(act, 0.01)),
        "pcc_row_min_active": float(act.min()),
        "rows_inactive": int((~active).sum()),
        # PLAN v1.16: a gross single-element check (not a band check), drift-gated at 1.5x the fixed record.
        "abs_err_max": float((g - e).abs().max()),
        # PLAN v1.11: the inactive rows' own metric, max |error| over the golden-defined inactive set relative to
        # the whole golden head's std; not applicable (absent) when the golden has no inactive rows.
        **(
            {"inactive_err_max_rel": float((gr[~active] - er[~active]).abs().max() / er.std())}
            if (~active).any()
            else {}
        ),
    }


def inflate_quantiles(head: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    """The complementary broken output: every quantile's distance from the quantile mean scaled by ``factor``."""
    h = torch.as_tensor(head).reshape(head.shape[0], -1, NUM_QUANTILES)
    m = h.mean(-1, keepdim=True)
    return (m + factor * (h - m)).reshape(head.shape).contiguous()


def scale_bands(head: torch.Tensor, factor, rows=None) -> torch.Tensor:
    """PLAN v1.9 broken outputs: the band of ``rows`` (all rows if None) scaled by ``factor`` around the quantile mean."""
    h = torch.as_tensor(head).reshape(head.shape[0], -1, NUM_QUANTILES).clone()
    sel = slice(None) if rows is None else torch.as_tensor(rows)
    m = h[sel].mean(-1, keepdim=True)
    h[sel] = m + factor * (h[sel] - m)
    return h.reshape(head.shape).contiguous()


def collapse_quantiles(head: torch.Tensor) -> torch.Tensor:
    """The deliberately broken output: every quantile replaced by the mean over the 9 quantiles."""
    h = torch.as_tensor(head).reshape(head.shape[0], -1, NUM_QUANTILES)
    return h.mean(-1, keepdim=True).expand_as(h).reshape(head.shape).contiguous()


def check_gates(
    name: str, expected_head, got_head, expected_q=None, got_q=None, precision: str = "accuracy", active=None
) -> dict:
    """Every gate metric of a head output (and of the final quantiles when given) asserted against ``GATES``."""
    got_head = torch.as_tensor(got_head)
    assert torch.isfinite(got_head).all(), f"{name}: non-finite head output"
    m = head_metrics(expected_head, got_head, active)
    if expected_q is not None:
        m.update(final_metrics(expected_q, got_q))
    gates = GATES[precision]
    print(
        f"[METRICS] {name} ({precision}): "
        + ", ".join(f"{k} {v:.6f}" + (f" ({gates[k][0]} {gates[k][1]})" if k in gates else "") for k, v in m.items()),
        flush=True,
    )
    failures = gate_failures(m, precision, name)
    if m["rows_inactive"]:
        print(f"[METRICS] {name}: inactive_err_max_rel <= {inactive_floor(name)}", flush=True)
    else:
        print(f"[METRICS] {name}: inactive_err_max_rel N/A (rows_inactive 0)", flush=True)
    if expected_q is None:  # head-only check: the data-space metrics are not computed here
        failures = {k: v for k, v in failures.items() if not k.startswith("pcc_final")}
    assert not failures, f"{name}: outside the gates: {failures}"
    return m


def final_metrics(expected, got) -> dict:
    """``pcc_final`` (data-space quantiles) and ``pcc_final_spread`` (minus their mean over the quantile axis).
    Data-space levels dominate ``pcc_final``: a collapsed forecast still scores ~0.9996 on it."""
    e, g = torch.as_tensor(expected).double(), torch.as_tensor(got).double()
    spread = lambda t: (t - t.mean(-1, keepdim=True)).reshape(1, -1)  # noqa: E731
    return {
        "pcc_final": float(_pcc_rows(e.reshape(1, -1), g.reshape(1, -1))[0]),
        "pcc_final_spread": float(_pcc_rows(spread(e), spread(g))[0]),
    }
