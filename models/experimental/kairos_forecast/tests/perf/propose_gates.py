# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Writes the G1 floor and drift proposal (doc/optimize/gates.json) from one round-0 suite record, by fixed rules.

    python -m models.experimental.kairos_forecast.tests.perf.propose_gates <suite records .jsonl>

Floors change only through a decision; this script records how the proposal was derived (PLAN v1.9, v1.11, v1.14,
v1.16, v1.20, v1.22, v1.24; D3 items 1 and 5), so the gate can cite the rule instead of hand-fitted numbers.

Absolute floors on every case (`*/<precision>`): the contract and D0 ones (pcc_normalized, pcc_normalized_full,
pcc_final, routing), the band shape (pcc_spread, pcc_final_spread, pcc_row_min), and the band ratios: p01 / p99 at
>= 100 active rows, p50 at >= 8 (check_gates applies them), and loose min / max on every case.
Per case, from the measured value m of that case and precision:
    bias_row_p50, bias_row_p50_final   <= max(2 m, 0.005)
    bias_row_max                       <= max(2 m, 0.01)
    inactive_max_err                   <= max(2 m, 0.01)
    pcc_step, pcc_step_final           >= m - max(2 (1 - m), 1e-4)
    spread_ratio_min / max on cases with < 100 active rows (v1.14): accuracy [0.95, 1.05] (long_horizon [0.95, 1.10]),
                                       performance [0.92, 1.08] (long_horizon [0.90, 1.20])
Drift against the fixed record (iter_00), both precisions: the PCCs by an absolute margin (accuracy / performance),
pcc_row_p01_active and pcc_row_min_active (not the all-row pcc_row_min, which is floor-only, v1.11),
spread_ratio_p50 two-sided +-0.01 (v1.11, D3), and the lower-is-better errors at 1.5x (abs_err_max is the gross
single-element ceiling, v1.16). pcc_step drift per case: max(1 - m, 1e-4).
"""
from __future__ import annotations

import json
import sys

from models.experimental.kairos_forecast.tests.perf.check_gates import DEFAULT_GATES, load_records

PRECS = ("accuracy", "performance")
GLOBAL = {
    "accuracy": {
        "pcc_normalized": 0.999, "pcc_normalized_full": 0.999, "pcc_final": 0.99, "pcc_final_spread": 0.999,
        "pcc_spread": 0.9995, "pcc_row_min": 0.99, "route_mismatch_rows": 0, "bias_max_abs_diff": 1e-06,
        "spread_ratio_p01": 0.97, "spread_ratio_p99": [0.0, 1.1], "spread_ratio_p50": [0.97, 1.03],
        "spread_ratio_min": 0.8, "spread_ratio_max": 1.3,
    },
    "performance": {
        "pcc_normalized": 0.995, "pcc_normalized_full": 0.995, "pcc_final": 0.98, "pcc_final_spread": 0.995,
        "pcc_spread": 0.998, "pcc_row_min": 0.95, "route_mismatch_rows": 0, "bias_max_abs_diff": 1e-06,
        "spread_ratio_p01": 0.9, "spread_ratio_p99": [0.0, 1.3], "spread_ratio_p50": [0.93, 1.07],
        "spread_ratio_min": 0.7, "spread_ratio_max": 2.0,
    },
}
SMALL_BAND = {"accuracy": (0.95, 1.05), "performance": (0.92, 1.08)}
LONG_BAND = {"accuracy": (0.95, 1.10), "performance": (0.90, 1.20)}
DRIFT = {
    "accuracy": {"pcc_normalized": 2e-4, "pcc_normalized_full": 2e-4, "pcc_final": 5e-4, "pcc_spread": 2e-4,
                 "pcc_final_spread": 2e-4, "pcc_row_p01_active": 2e-3, "pcc_row_min_active": 5e-3},
    "performance": {"pcc_normalized": 1e-3, "pcc_normalized_full": 1e-3, "pcc_final": 1e-3, "pcc_spread": 1e-3,
                    "pcc_final_spread": 1e-3, "pcc_row_p01_active": 1e-2, "pcc_row_min_active": 2e-2},
}
REL = ("abs_err_max", "inactive_max_err", "bias_row_p50", "bias_row_max", "bias_row_p50_final", "bias_row_max_final")


def _up(x, digits=2):
    """Round a ceiling up / a floor down to `digits` significant figures (never tighter than the rule)."""
    import math

    if x == 0:
        return 0.0
    e = math.floor(math.log10(abs(x))) - digits + 1
    return math.ceil(x / 10**e) * 10**e


def _down(x, digits=4):
    import math

    return math.floor(x * 10**digits) / 10**digits


def propose(records):
    floors = {f"*/{p}": dict(GLOBAL[p]) for p in PRECS}
    drift = {k: {"abs": v} for k, v in DRIFT["accuracy"].items()}
    drift["spread_ratio_p50"] = {"two_sided": True, "abs": 0.01}
    drift.update({k: {"rel": 0.5} for k in REL})
    drift_by = {"*/performance": {k: {"abs": v} for k, v in DRIFT["performance"].items()}}
    for r in records:
        key = f"{r['name']}/{r['precision']}"
        p = r["precision"]
        f = {
            "bias_row_p50": _up(max(2 * r["bias_row_p50"], 0.005)),
            "bias_row_max": _up(max(2 * r["bias_row_max"], 0.01)),
            "bias_row_p50_final": _up(max(2 * r["bias_row_p50_final"], 0.005)),
            "inactive_max_err": _up(max(2 * r["inactive_max_err"], 0.01)),
        }
        db = {}
        for k in ("pcc_step", "pcc_step_final"):
            f[k] = _down(r[k] - max(2 * (1 - r[k]), 1e-4))
            db[k] = {"abs": _up(max(1 - r[k], 1e-4))}
        if r["active_rows"] < 100:
            lo, hi = (LONG_BAND if r["name"] == "long_horizon" else SMALL_BAND)[p]
            f["spread_ratio_min"], f["spread_ratio_max"] = lo, hi
        floors[key] = f
        drift_by[key] = db
    return {
        "_doc": "G1 re-raise proposal, written by tests/perf/propose_gates.py (its docstring has the rules) from the "
                "round-0 suite record of pcc job 1047. Absolute floors on every case and precision (PLAN v1.20); p01 / "
                "p99 at >= 100 active rows and p50 at >= 8 (check_gates prints `not applicable` otherwise). Drift is "
                "against the fixed records iter_00/bench.json (bench) and iter_00/suite.jsonl (suite).",
        "floors": floors, "drift": drift, "drift_by": drift_by,
    }


if __name__ == "__main__":
    recs = load_records(sys.argv[1])
    DEFAULT_GATES.write_text(json.dumps(propose(recs), indent=1) + "\n")
    print("wrote", DEFAULT_GATES, len(recs), "records")
