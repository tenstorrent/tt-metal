# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Check bench records against the gate floors and the per-metric drift tolerances.

    python models/experimental/tirex2_forecast/tests/perf/check_gates.py <candidate bench.json>
        [--baseline <baseline bench.json>] [--gates doc/optimize/gates.json]

``gates.json`` holds, per ``<shape>/<precision>`` (or ``*/<precision>``), the floor of every gate metric, and one
drift tolerance per metric. Floors and tolerances change only through a decision (PLAN section 9): this script
only reads them. Exit status 1 when any floor or drift check fails; every check is printed.

Direction of each metric: higher is better except ``median_err``, ``flat_max_err`` (upper bounds) and
``spread_ratio_p99`` / ``spread_ratio_max`` (upper bounds near 1). A floor given as ``[lo, hi]`` is a range. Drift is ``candidate >= baseline - tol`` for higher-is-better
metrics; ``candidate <= baseline + tol`` (absolute) or ``<= baseline x (1 + rel)`` for the others.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOWER_IS_BETTER = {"median_err", "flat_max_err", "spread_ratio_p99", "spread_ratio_max"}
DEFAULT_GATES = Path(__file__).resolve().parents[2] / "doc" / "optimize" / "gates.json"


def _floors(gates: dict, name: str, precision: str) -> dict:
    out = dict(gates["floors"].get(f"*/{precision}", {}))
    out.update(gates["floors"].get(f"{name}/{precision}", {}))
    return out


def passes(metric: str, v: float, floor) -> bool:
    """``floor`` is a bound in the metric's direction, or a ``[lo, hi]`` range."""
    if isinstance(floor, list):
        return floor[0] <= v <= floor[1]
    return v <= floor if metric in LOWER_IS_BETTER else v >= floor


def check(cand: list[dict], base: list[dict] | None, gates: dict) -> list[str]:
    failures = []
    base_by = {(r["name"], r["precision"]): r for r in (base or [])}
    for r in cand:
        key = (r["name"], r["precision"])
        for metric, floor in _floors(gates, *key).items():
            v = r.get(metric)
            if v is None:
                failures.append(f"{key} {metric}: missing")
                continue
            ok = passes(metric, v, floor)
            rel = "in" if isinstance(floor, list) else "<=" if metric in LOWER_IS_BETTER else ">="
            print(f"{'ok  ' if ok else 'FAIL'} floor {key[0]}/{key[1]} {metric} {v:.6g} {rel} {floor}")
            if not ok:
                failures.append(f"{key} {metric} {v} vs floor {floor}")
        b = base_by.get(key)
        if b is None:
            continue
        for metric, tol in gates["drift"].items():
            if metric not in r or metric not in b:
                continue
            v, bv = r[metric], b[metric]
            if metric in LOWER_IS_BETTER:
                limit = bv * (1 + tol["rel"]) if "rel" in tol else bv + tol["abs"]
                ok = v <= limit
                rel = "<="
            else:
                limit = bv - tol["abs"]
                ok = v >= limit
                rel = ">="
            print(
                f"{'ok  ' if ok else 'FAIL'} drift {key[0]}/{key[1]} {metric} {v:.6g} {rel} {limit:.6g} (baseline {bv:.6g})"
            )
            if not ok:
                failures.append(f"{key} {metric} drift {v} vs {limit}")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    args = ap.parse_args()
    gates = json.loads(args.gates.read_text())
    cand = json.loads(args.candidate.read_text())
    base = json.loads(args.baseline.read_text()) if args.baseline else None
    failures = check(cand, base, gates)
    print(f"{len(failures)} failure(s)" + (": " + "; ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
