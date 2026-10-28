# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Copied from TiRex-2 651e8c4af93 (CONTRACT §3, PLAN v1.20). Check bench records against the gate floors and the per-metric drift tolerances.

    python models/experimental/tafsut_forecast/tests/perf/check_gates.py <candidate bench.json>
        [--baseline <record.json> ...] [--gates doc/optimize/gates.json]

``gates.json`` holds, per ``<shape>/<precision>`` (or ``*/<precision>``), the floor of every gate metric, and one
drift tolerance per metric. Floors and tolerances change only through a decision (PLAN section 9): this script
only reads them. Exit status 1 when any floor or drift check fails; every check is printed.

Direction of each metric: higher is better except ``median_err``, ``flat_max_err``, ``inactive_max_err``,
``abs_err_max``, ``bias_row_p50`` (upper bounds) and
``spread_ratio_p99`` / ``spread_ratio_max`` (upper bounds near 1). A floor given as ``[lo, hi]`` is a range; a drift with
``two_sided`` is ``|candidate - baseline| <= abs``. Drift is ``candidate >= baseline - tol`` for higher-is-better
metrics; ``candidate <= baseline + tol`` (absolute) or ``<= baseline x (1 + rel)`` for the others.

``drift_by`` overrides ``drift`` per ``<shape>/<precision>`` (or ``*/<precision>``); ``null`` removes a metric. A
tolerance with ``ref`` (a path relative to gates.json) checks that metric against that record instead of the
baseline (D6). A drift metric missing from either record is a FAIL (v1.17). A candidate record must carry
``git_head`` and an empty ``git_dirty`` (v1.16); ``--allow-unstamped`` admits records written before that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOWER_IS_BETTER = {
    "median_err",
    "flat_max_err",
    "inactive_max_err",
    "abs_err_max",
    "bias_row_p50",
    "spread_ratio_p99",
    "spread_ratio_max",
}
DEFAULT_GATES = Path(__file__).resolve().parents[2] / "doc" / "optimize" / "gates.json"
# drift is always against the fixed round-0 records (both precisions, every bench and suite case; v1.14, v1.17)
DEFAULT_BASELINES = (DEFAULT_GATES.parent / "iter_00" / "bench.json", DEFAULT_GATES.parent / "iter_00" / "suite.json")


def load_records(paths) -> list[dict]:
    return [r for p in paths for r in json.loads(Path(p).read_text())]


def git_stamp() -> dict:
    """v1.16 / v1.20: HEAD and the porcelain status of the model's tt/, tests/ and ops/, untracked files included."""
    import subprocess

    model = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", *a], cwd=model, capture_output=True, text=True, check=True).stdout.strip()
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain", "--", "tt", "tests", "ops", "reference"),
    }


def _floors(gates: dict, name: str, precision: str) -> dict:
    out = dict(gates["floors"].get(f"*/{precision}", {}))
    out.update(gates["floors"].get(f"{name}/{precision}", {}))
    return out


def _drift(gates: dict, name: str, precision: str) -> dict:
    out = dict(gates["drift"])
    for key in (f"*/{precision}", f"{name}/{precision}"):
        out.update(gates.get("drift_by", {}).get(key, {}))
    return {k: v for k, v in out.items() if v is not None}


def _by_key(records: list[dict]) -> dict:
    return {(r["name"], r["precision"]): r for r in records}


def passes(metric: str, v: float, floor) -> bool:
    """``floor`` is a bound in the metric's direction, or a ``[lo, hi]`` range."""
    if isinstance(floor, list):
        return floor[0] <= v <= floor[1]
    return v <= floor if metric in LOWER_IS_BETTER else v >= floor


def check(
    cand: list[dict], base: list[dict] | None, gates: dict, gates_dir: Path | None = None, allow_unstamped=False
) -> list[str]:
    failures = []
    base_by = _by_key(base or [])
    refs = {}
    for r in cand:
        key = (r["name"], r["precision"])
        if "git_head" not in r and not allow_unstamped:
            failures.append(f"{key}: no git_head / git_dirty stamp")
        if r.get("git_dirty"):
            print(f"FAIL tree {key[0]}/{key[1]} dirty at {r.get('git_head')}: {r['git_dirty']!r}")
            failures.append(f"{key}: dirty tree")
        for metric, floor in _floors(gates, *key).items():
            if metric in r and r[metric] is None:  # v1.14: skipped for too few active rows
                print(f"n/a  floor {key[0]}/{key[1]} {metric}: not applicable (n_active={r.get('active_rows')})")
                continue
            v = r.get(metric)
            if v is None:
                failures.append(f"{key} {metric}: missing")
                continue
            ok = passes(metric, v, floor)
            rel = "in" if isinstance(floor, list) else "<=" if metric in LOWER_IS_BETTER else ">="
            print(f"{'ok  ' if ok else 'FAIL'} floor {key[0]}/{key[1]} {metric} {v:.6g} {rel} {floor}")
            if not ok:
                failures.append(f"{key} {metric} {v} vs floor {floor}")
        for metric, tol in _drift(gates, *key).items():
            if "ref" in tol:
                path = (gates_dir or DEFAULT_GATES.parent) / tol["ref"]
                if path not in refs:
                    refs[path] = _by_key(json.loads(path.read_text()))
                b = refs[path].get(key)
            else:
                b = base_by.get(key)
            if b is None or metric not in r or metric not in b:
                where = (
                    "no baseline record"
                    if b is None
                    else "missing from " + ("candidate" if metric not in r else "baseline")
                )
                print(f"FAIL drift {key[0]}/{key[1]} {metric}: {where}")
                failures.append(f"{key} {metric} drift: {where}")
                continue
            v, bv = r[metric], b[metric]
            if v is None and bv is None:  # v1.14: too few active rows in both records
                print(f"n/a  drift {key[0]}/{key[1]} {metric}: not applicable (n_active={r.get('active_rows')})")
                continue
            if v is None or bv is None:
                print(f"FAIL drift {key[0]}/{key[1]} {metric}: n/a in one record only")
                failures.append(f"{key} {metric} drift: n/a in one record only")
                continue
            if tol.get("two_sided"):
                ok = abs(v - bv) <= tol["abs"]
                print(
                    f"{'ok  ' if ok else 'FAIL'} drift {key[0]}/{key[1]} {metric} {v:.6g} within {tol['abs']} of {bv:.6g}"
                )
                if not ok:
                    failures.append(f"{key} {metric} drift {v} vs {bv} +- {tol['abs']}")
                continue
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
    ap.add_argument(
        "--baseline", nargs="*", type=Path, default=list(DEFAULT_BASELINES), help="none given = floors only"
    )
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--allow-unstamped", action="store_true", help="admit records without git_head (pre-v1.16)")
    args = ap.parse_args()
    gates = json.loads(args.gates.read_text())
    cand = json.loads(args.candidate.read_text())
    base = load_records(args.baseline) if args.baseline else None
    failures = check(cand, base, gates, args.gates.parent, args.allow_unstamped)
    print(f"{len(failures)} failure(s)" + (": " + "; ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
