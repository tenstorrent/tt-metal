# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden files written by ``make_goldens.py`` (reference venv), read here from python_env (no reference import)."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch

GOLDEN_DIR = Path(os.environ.get("TOTO2_GOLDEN_DIR", "/home/andy/bringup/goldens/toto2"))
MANIFEST = Path(__file__).resolve().parent / "MANIFEST.json"


def golden_path(ckpt: str, case: str) -> Path:
    return GOLDEN_DIR / ckpt / f"{case}.pt"


def have_golden(ckpt: str, case: str) -> bool:
    return golden_path(ckpt, case).is_file()


@lru_cache(maxsize=4)
def load_golden(ckpt: str, case: str, verify: bool = False) -> dict:
    """The golden dict (``inputs``, ``captures``, ``cpu_s``, ...); ``verify`` checks its sha256 against MANIFEST.json."""
    path = golden_path(ckpt, case)
    if verify:
        expected = json.loads(MANIFEST.read_text())["goldens"][f"{ckpt}/{case}.pt"]["sha256"]
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h == expected, f"{path}: sha256 {h} != MANIFEST {expected}"
    return torch.load(path, weights_only=False)


def rel_err(got: torch.Tensor, expected: torch.Tensor) -> float:
    """``||got - expected|| / ||expected||`` in float64."""
    got, expected = got.double(), expected.double()
    return ((got - expected).norm() / expected.norm()).item()


def pcc(got: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = got.double().flatten(), expected.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def _pcc_or_zero(got: torch.Tensor, expected: torch.Tensor) -> float:
    """PCC, 0.0 when either side has no variance (a collapsed output)."""
    a, b = got.double().flatten(), expected.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return 0.0 if denom == 0 else (a @ b / denom).item()


# Row sets (fixed by the golden alone):
# - active (PLAN section 9 v1.2): golden head std >= 5 % of the golden's global head std. Per-row PCC is
#   ill-conditioned on flat rows (intermittent series forecast flat), where bf16 noise is comparable to the row's
#   own variation; the per-row metrics with suffix _active and spread_ratio use the active rows.
# - degenerate: near-constant reference series, within-series std below 1 % of its mean magnitude (floor 0.1).
#   A case made only of them (a single flat series is "active" relative to itself) is judged by abs_err_max alone:
#   its PCCs measure bf16 noise (2.5B-FT single_sparse_1x2048: head std 6e-4 around -0.22, device PCC 0.98 at max
#   abs error 1.5e-3).
ACTIVE_REL = 0.05
DEGENERATE_REL = 0.01


def _rows(head: torch.Tensor) -> torch.Tensor:
    """(9, B, V, h, 32) -> (B * V, 9 * h * 32)."""
    return head.double().permute(1, 2, 0, 3, 4).reshape(head.shape[1] * head.shape[2], -1)


def active_rows(golden_head: torch.Tensor) -> torch.Tensor:
    return _rows(golden_head).std(-1) >= ACTIVE_REL * golden_head.double().std()


def degenerate_rows(golden_head: torch.Tensor) -> torch.Tensor:
    rows = _rows(golden_head)
    return rows.std(-1) < DEGENERATE_REL * rows.abs().mean(-1).clamp_min(0.1)


def _quantiles(values: list[float], *qs: float) -> list[float]:
    t = torch.tensor(values if values else [float("nan")], dtype=torch.float64)
    return [torch.quantile(t, q).item() for q in qs]


def gate_metrics(head: torch.Tensor, q: torch.Tensor, golden_head: torch.Tensor, golden_q: torch.Tensor) -> dict:
    """Gate metrics of a forecast against the reference (PLAN section 9, v1.1-v1.3).

    ``head`` (9, B, V, h, 32): head output at the read-out tokens, asinh-normalized; ``q`` (9, B, V, H) final.
        pcc_normalized             PCC of the head output
        pcc_spread                 PCC of the head output minus its mean over the 9 quantiles
        pcc_row_p01 / _min         per-series PCC of the head output over every series: 1st percentile, minimum
        pcc_row_p01_active / _min_active   the same over the active series (``active_rows``)
        spread_ratio_p50 / _min / _max   the median (catches a uniform band scale), extremes (floor-only, v1.9)
        spread_ratio_inactive_p01 / _p50 / _p99   the same ratio over the inactive series (1.0 when there are none)
        spread_ratio_p01 / _p99    per active series, |h - mean_q h| / |h_ref - mean_q h_ref| (norms over the
                                   series' 9 x h x 32 values): 1st and 99th percentiles; a collapse scores 0
        pcc_final                  PCC of the final quantiles
        pcc_final_spread           PCC of the final quantiles minus their mean over the 9 quantiles
        abs_err_max                max |head - reference| in normalized space
        rows, rows_active, rows_degenerate
    """
    got, exp = _rows(head), _rows(golden_head)
    active, degenerate = active_rows(golden_head), degenerate_rows(golden_head)
    per_row = [_pcc_or_zero(a, b) for a, b in zip(got, exp)]
    per_active = [p for p, a in zip(per_row, active) if a]
    dev_spread = _rows(head - head.mean(0)).norm(dim=-1)
    ref_spread = _rows(golden_head - golden_head.mean(0)).norm(dim=-1)
    ratio = [(d / r).item() for d, r, a in zip(dev_spread, ref_spread, active) if a and r > 0]
    # Inactive rows (flat forecasts) still carry a quantile band; a collapse of theirs must fail a floor (v1.9).
    ratio_in = [(d / r).item() for d, r, a in zip(dev_spread, ref_spread, active) if not a and r > 0]
    row_p01, row_min = _quantiles(per_row, 0.01, 0.0)
    act_p01, act_min = _quantiles(per_active, 0.01, 0.0)
    ratio_p01, ratio_p50, ratio_p99 = _quantiles(ratio, 0.01, 0.5, 0.99)
    ratio_min, ratio_max = _quantiles(ratio, 0.0, 1.0)
    in_p01, in_p50, in_p99 = _quantiles(ratio_in, 0.01, 0.5, 0.99) if ratio_in else (1.0, 1.0, 1.0)
    return {
        "pcc_normalized": _pcc_or_zero(head, golden_head),
        "pcc_spread": _pcc_or_zero(head - head.mean(0), golden_head - golden_head.mean(0)),
        "pcc_row_p01": row_p01,
        "pcc_row_min": row_min,
        "pcc_row_p01_active": act_p01,
        "pcc_row_min_active": act_min,
        "spread_ratio_p01": ratio_p01,
        "spread_ratio_p99": ratio_p99,
        "spread_ratio_p50": ratio_p50,
        "spread_ratio_min": ratio_min,
        "spread_ratio_max": ratio_max,
        "spread_ratio_inactive_p01": in_p01,
        "spread_ratio_inactive_p50": in_p50,
        "spread_ratio_inactive_p99": in_p99,
        "pcc_final": _pcc_or_zero(q, golden_q),
        "pcc_final_spread": _pcc_or_zero(q - q.mean(0), golden_q - golden_q.mean(0)),
        "abs_err_max": (head.double() - golden_head.double()).abs().max().item(),
        "rows": int(active.numel()),
        "rows_active": int(active.sum()),
        "rows_degenerate": int(degenerate.sum()),
    }


# Metrics with an upper floor (a ceiling) and those recorded but not gated.
CEILINGS = {"spread_ratio_p99", "abs_err_max"}
RECORDED = {"pcc_row_p01", "pcc_row_min", "rows", "rows_active", "rows_degenerate"}


# Gated on an all-degenerate case (its PCCs measure bf16 noise; its band ratio and abs error do not).
DEGENERATE_GATED = {"abs_err_max", "spread_ratio_p50"}


def _misses(key: str, value: float, floor) -> bool:
    """``floor``: a ``(low, high)`` range, a ceiling (``CEILINGS``) or a floor."""
    if isinstance(floor, tuple):
        return not floor[0] <= value <= floor[1]
    return not value <= floor if key in CEILINGS else not value >= floor


def below_floors(metrics: dict, floors: dict) -> dict:
    """The gated metrics that miss their floor (``CEILINGS``: exceed it; tuples: leave the range); a gated metric
    missing from ``metrics`` is a miss. ``pcc_row_min_active`` takes ``pcc_row_min_active_many`` past ``MANY_ROWS``
    series. A case whose series are all degenerate is judged by ``DEGENERATE_GATED`` alone; ``abs_err_max`` gates
    nothing else."""
    degenerate = all_degenerate(metrics)
    out = {}
    for key, floor in floors.items():
        if key == "pcc_row_min_active_many":
            continue
        if key == "spread_ratio_p50_degenerate":
            if not degenerate:
                continue
            key = "spread_ratio_p50"  # D2: the degenerate range; gated even where spread_ratio_p50 is INFO (D3)
        elif key == "spread_ratio_p50":
            if degenerate:
                continue
        elif degenerate != (key in DEGENERATE_GATED):
            continue
        if key == "pcc_row_min_active" and metrics["rows"] > MANY_ROWS:
            floor = floors["pcc_row_min_active_many"]
        if key not in metrics:
            out[key] = (None, floor)
        elif _misses(key, metrics[key], floor):
            out[key] = (round(metrics[key], 6), floor)
    return out


def collapsed(head: torch.Tensor) -> torch.Tensor:
    """Every quantile replaced by the mean over the quantile axis: the broken output the gate metrics must reject."""
    return head.mean(0, keepdim=True).expand_as(head).clone()


# Gate floors (asserted by tests/pcc/test_model.py; the bench records them). pcc_normalized / pcc_final are the
# CONTRACT section 3 floors. The others are this worker's proposals for G1, chosen AFTER measuring round 0 (as PLAN
# section 9 asks; history in GATE.md): from iter_00/pcc_metrics.json and bench.json with margin, each scored below
# by a quantile-collapsed output (tests/golden/test_gate_metrics.py -> doc/golden/collapsed_metrics.json). They change
# only by decision.
# - pcc_row_min_active over more than MANY_ROWS series is set by ill-conditioned rows of the reference itself (313m
#   paper_mix row 756: rounding the patch tokens to bf16 moves the fp32 reference head by up to 0.33).
# - The unit-test checkpoints 4m and 22m have no contract floor; theirs sit lower (bf16 stream error of small,
#   high-tau models at context 8192: 4m 0.9989 / 22m 0.9996 normalized).
MANY_ROWS = 64
GATE_FLOORS = {
    "accuracy": {
        "pcc_normalized": 0.999,
        "pcc_spread": 0.998,
        "pcc_row_p01_active": 0.99,
        "pcc_row_min_active": 0.98,
        "pcc_row_min_active_many": 0.9,
        "spread_ratio_p01": 0.9,
        "spread_ratio_p99": 1.25,
        "pcc_final": 0.99,
        "pcc_final_spread": 0.99,
        "abs_err_max": 0.01,
    },
    "performance": {
        "pcc_normalized": 0.995,
        "pcc_spread": 0.99,
        "pcc_row_p01_active": 0.97,
        "pcc_row_min_active": 0.95,
        "pcc_row_min_active_many": 0.9,
        "spread_ratio_p01": 0.85,
        "spread_ratio_p99": 1.25,
        "pcc_final": 0.98,
        "pcc_final_spread": 0.93,
        "abs_err_max": 0.02,
    },
}
SMALL_CHECKPOINT_FLOORS = {
    "accuracy": {
        "pcc_normalized": 0.995,
        "pcc_spread": 0.995,
        "pcc_row_p01_active": 0.99,
        "pcc_row_min_active": 0.99,
        "pcc_row_min_active_many": 0.9,
        "spread_ratio_p01": 0.8,
        "spread_ratio_p99": 1.2,
        "pcc_final": 0.99,
        "pcc_final_spread": 0.99,
        "abs_err_max": 0.01,
    },
    "performance": {
        "pcc_normalized": 0.99,
        "pcc_spread": 0.98,
        "pcc_row_p01_active": 0.9,
        "pcc_row_min_active": 0.9,
        "pcc_row_min_active_many": 0.8,
        "spread_ratio_p01": 0.85,
        "spread_ratio_p99": 1.7,
        "pcc_final": 0.98,
        "pcc_final_spread": 0.93,
        "abs_err_max": 0.02,
    },
}


# Band floors set by D2 (G1 approval, 2026-10-26 13:56). spread_ratio_p50 is two-sided; an all-degenerate case
# (rows_degenerate == rows) is judged by abs_err_max plus spread_ratio_p50_degenerate. min / max / inactive are
# floor-only (no drift).
_BAND_COMMON = dict(
    spread_ratio_p50_degenerate=(0.5, 2.0),
    spread_ratio_min=0.6,
    spread_ratio_inactive_p01=0.4,
    spread_ratio_inactive_p50=2.0,
)
TARGET_BAND = {
    "accuracy": dict(_BAND_COMMON, spread_ratio_p50=(0.97, 1.03), spread_ratio_max=2.0),
    "performance": dict(_BAND_COMMON, spread_ratio_p50=(0.95, 1.05), spread_ratio_max=2.0),
}
SMALL_BAND = dict(_BAND_COMMON, spread_ratio_p50=(0.95, 1.05), spread_ratio_max=2.5)
for _prec in ("accuracy", "performance"):
    GATE_FLOORS[_prec].update(TARGET_BAND[_prec])
    SMALL_CHECKPOINT_FLOORS[_prec].update(SMALL_BAND)
# D3 ruling 1: 4m / 22m spread_ratio_p50 is 0.9-1.1 in both precisions (2 to 8 active rows; broken-output detection
# is required on 2.5B-FT and 313m only). D3 ruling 3: small-checkpoint p99 is the tighter of the pre-D2 and D2 values.
for _prec, _p99 in (("accuracy", 1.2), ("performance", 1.5)):
    SMALL_CHECKPOINT_FLOORS[_prec].update(spread_ratio_p50=(0.9, 1.1), spread_ratio_p99=_p99)
SMALL_313M_P99 = {"accuracy": 1.25, "performance": 1.5}

# D3 ruling 2 put four 313m performance band metrics in an INFO set (recorded, not gated); D4 ruling 2 (performance
# = P6) empties it: every D2 floor is gated on 313m performance.
INFO: dict = {}


def _all_floors(ckpt: str, prec: str) -> dict:
    if ckpt in ("4m", "22m"):
        return SMALL_CHECKPOINT_FLOORS[prec]
    if ckpt == "2.5B-FT":
        return GATE_FLOORS[prec]
    return {**GATE_FLOORS[prec], **SMALL_BAND, "spread_ratio_p99": SMALL_313M_P99[prec]}


def info_keys(ckpt: str, precision: str) -> set:
    return INFO.get((ckpt, precision.split("_")[0]), set())


def floors_for(ckpt: str, precision: str) -> dict:
    """The gated floors (D3: the INFO metrics are left out)."""
    info = info_keys(ckpt, precision)
    return {k: v for k, v in _all_floors(ckpt, precision.split("_")[0]).items() if k not in info}


def info_misses(metrics: dict, ckpt: str, precision: str) -> dict:
    """The INFO metrics (D3) that would miss their D2 floor: printed and recorded, never gated."""
    floors = _all_floors(ckpt, precision.split("_")[0])
    return below_floors(metrics, {k: floors[k] for k in info_keys(ckpt, precision)})


# Drift from round 0 of the same precision (tests/perf/check_gates.py): a candidate may lose at most this much
# (CEILINGS: gain); twice that off the target. Proposed at G1 (D1 item 4); min / max / range metrics are floor-only
# (PLAN section 9 v1.9).
DRIFT = {
    "pcc_normalized": 2e-4,
    "pcc_spread": 5e-4,
    "pcc_row_p01_active": 5e-3,
    "spread_ratio_p01": 0.02,
    "spread_ratio_p99": 0.05,
    "pcc_final": 2e-4,
    "pcc_final_spread": 2e-3,
    "spread_ratio_p50": 0.01,
}
TWO_SIDED = {"spread_ratio_p50"}
# D3 ruling 2's extra 313m performance drift (inactive_p50 +0.15) went away with the INFO set (D4 ruling 2).
DRIFT_EXTRA: dict = {}
# D4 ruling 1: an all-degenerate case gets no drift on PCC, pcc_row_* or spread_ratio (floors only: abs_err_max and
# the 0.5-2.0 p50 range); abs_err_max may grow at most this fraction of the fixed record. A gross-error check, not
# a band check: a collapse (0.00214) and a 0.3x band (0.00150) on 2.5B-FT single_sparse both pass it.
DEGENERATE_ABS_ERR_GROWTH = 0.5


def all_degenerate(metrics: dict) -> bool:
    """D4 ruling 1: every series of the case is degenerate."""
    return metrics["rows_degenerate"] == metrics["rows"]


CEILINGS.update({"spread_ratio_max", "spread_ratio_inactive_p50"})
# The G1 proposals are in force (D2); score_broken.py --proposed now scores the same floors.
PROPOSED_FLOORS = GATE_FLOORS
