# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Check bench records against the gate floors and the per-metric drift tolerances.

    python models/experimental/kairos_forecast/tests/perf/check_gates.py <candidate bench.json>
        [--baseline <baseline bench.json>] [--gates doc/optimize/gates.json]

``gates.json`` holds, per ``<shape>/<precision>`` (or ``*/<precision>``), the floor of every gate metric, and one
drift tolerance per metric. Floors and tolerances change only through a decision (PLAN section 9): this script
only reads them. Exit status 1 when any floor or drift check fails; every check is printed.

Direction of each metric: higher is better except ``median_err``, ``flat_max_err``, ``inactive_max_err``,
``abs_err_max`` (upper bounds) and
``spread_ratio_p99`` / ``spread_ratio_max`` (upper bounds near 1). A floor given as ``[lo, hi]`` is a range; a drift with
``two_sided`` is ``|candidate - baseline| <= abs``. Drift is ``candidate >= baseline - tol`` for higher-is-better
metrics; ``candidate <= baseline + tol`` (absolute) or ``<= baseline x (1 + rel)`` for the others.

``drift_by`` overrides ``drift`` per ``<shape>/<precision>`` (or ``*/<precision>``); ``null`` removes a metric. A
tolerance with ``ref`` (a path relative to gates.json) checks that metric against that record instead of the
baseline (D6). A drift metric missing from either record is a FAIL (v1.17), and so is a candidate set that lacks a
case or precision of the baseline. A candidate record must carry the session stamp (v1.16, v1.23): ``git_head``, an
empty ``git_dirty`` and a ``started_at`` no earlier than ``git_head``'s commit time, all taken once when the job's
session started (``session_stamp()``); ``--allow-unstamped`` admits records written before that.

Applicability (v1.12, v1.14): band percentiles p01 / p99 apply at >= 100 active rows, p50 at >= 8, and
``inactive_max_err`` only when the case has inactive rows; every skipped check prints ``not applicable (...)``.
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
    "spread_ratio_p99",
    "spread_ratio_max",
    "route_mismatch_rows",  # D0 amendment 1: floor == 0
    "bias_max_abs_diff",  # D0 amendment 2: <= 1e-6
    "bias_row_p50",  # v1.24: row |mean error|, head space
    "bias_row_max",
    "bias_row_p50_final",  # the same in data space (over the target's scale)
    "bias_row_max_final",
}
# D0 amendment 1 / D2 fix 4: gated on the target and every suite / bench case in both precisions, report-only for
# free-running rollout steps >= 2 and e2e_stream.
ROUTING_KEYS = {"route_mismatch_rows", "bias_max_abs_diff", "min_top3_margin"}
# PLAN v1.20: absolute floors on every case, both precisions; a gates.json without them is a FAIL.
# D3 / v1.24: the level-shift, lag and inactive-row metrics are gated on every case too.
REQUIRED_FLOORS = {"pcc_normalized", "pcc_final", "route_mismatch_rows", "bias_max_abs_diff", "bias_row_p50",
                   "bias_row_max", "pcc_step", "bias_row_p50_final", "pcc_step_final", "inactive_max_err"}
# v1.14: (metric, minimum active rows). Population percentiles over fewer rows are really the min / max.
MIN_ACTIVE = {"spread_ratio_p01": 100, "spread_ratio_p99": 100, "pcc_row_p01_active": 100, "pcc_row_p01": 100,
              "spread_ratio_p50": 8}


def not_applicable(record: dict, metric: str) -> str | None:
    """Why `metric` is not checked on `record` (None: it is checked)."""
    need = MIN_ACTIVE.get(metric)
    n = record.get("active_rows")
    if need is not None and n is not None and n < need:
        return f"n_active={n} < {need}"
    if metric == "inactive_max_err" and record.get("excluded_rows") == 0:
        return "no inactive rows"
    return None


def report_only(record: dict, metric: str) -> bool:
    if metric not in ROUTING_KEYS:
        return False
    return record["name"].startswith("e2e_stream") or record.get("free_step", 0) >= 2
DEFAULT_GATES = Path(__file__).resolve().parents[2] / "doc" / "optimize" / "gates.json"
# D5: drift is always against this fixed round-0 record (both precisions, all shapes); a b2b base is timing only
DEFAULT_BASELINE = DEFAULT_GATES.parent / "iter_00" / "bench.json"


_SESSION = {}


def git_stamp() -> dict:
    """v1.16 / v1.20: HEAD and the porcelain status of the model's tt/, tests/ and ops/, untracked files included."""
    import subprocess

    model = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", *a], cwd=model, capture_output=True, text=True, check=True).stdout.strip()
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain", "--", "tt", "tests", "ops"),
    }


def session_stamp() -> dict:
    """v1.23: the stamp taken once, at the first call in this process (tests call it at import, i.e. at session
    start), with ``started_at`` (unix time); every record of the session carries this same stamp."""
    import time

    if not _SESSION:
        _SESSION.update(started_at=time.time(), **git_stamp())
    return dict(_SESSION)


def commit_time(head: str) -> float | None:
    import subprocess

    r = subprocess.run(["git", "show", "-s", "--format=%ct", head], cwd=Path(__file__).resolve().parent,
                       capture_output=True, text=True)
    return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None


def _floors(gates: dict, name: str, precision: str) -> dict:
    out = dict(gates["floors"].get(f"*/{precision}", {}))
    out.update(gates["floors"].get(f"{name}/{precision}", {}))
    return out


def _drift(gates: dict, name: str, precision: str) -> dict:
    out = dict(gates["drift"])
    for key in (f"*/{precision}", f"{name}/{precision}"):
        out.update(gates.get("drift_by", {}).get(key, {}))
    return {k: v for k, v in out.items() if v is not None}


def load_records(path: Path) -> list[dict]:
    """bench.json (a list) or the suite's JSON lines (suite.jsonl)."""
    text = Path(path).read_text()
    return [json.loads(x) for x in text.splitlines() if x.strip()] if str(path).endswith(".jsonl") else json.loads(text)


def _by_key(records: list[dict]) -> dict:
    return {(r["name"], r["precision"]): r for r in records}


def passes(metric: str, v: float, floor) -> bool:
    """``floor`` is a bound in the metric's direction, or a ``[lo, hi]`` range."""
    if isinstance(floor, list):
        return floor[0] <= v <= floor[1]
    return v <= floor if metric in LOWER_IS_BETTER else v >= floor


def check(
    cand: list[dict], base: list[dict] | None, gates: dict, gates_dir: Path | None = None, allow_unstamped=False,
    require_coverage=True, commit_times: dict | None = None,
) -> list[str]:
    """commit_times: {git_head: unix time} overrides the git lookup (unit tests)."""
    failures = []
    base_by = _by_key(base or [])
    refs = {}
    if require_coverage and base:
        for key in sorted(base_by.keys() - _by_key(cand).keys()):
            print(f"FAIL coverage {key[0]}/{key[1]}: in the baseline, missing from the candidate set (v1.24)")
            failures.append(f"{key}: missing from the candidate set")
    for r in cand:
        key = (r["name"], r["precision"])
        if ("git_head" not in r or "started_at" not in r) and not allow_unstamped:
            print(f"FAIL stamp {key[0]}/{key[1]}: no session stamp (git_head, started_at)")
            failures.append(f"{key}: no git_head / started_at session stamp")
        if r.get("git_dirty"):
            print(f"FAIL tree {key[0]}/{key[1]} dirty at {r.get('git_head')}: {r['git_dirty']!r}")
            failures.append(f"{key}: dirty tree")
        if "started_at" in r and "git_head" in r:
            ct = commit_times.get(r["git_head"]) if commit_times else commit_time(r["git_head"])
            if ct is None or r["started_at"] < ct:
                print(f"FAIL stamp {key[0]}/{key[1]}: started_at {r['started_at']} before commit {r['git_head']} "
                      f"({ct}) (v1.22)")
                failures.append(f"{key}: started before its commit")
        floors = _floors(gates, *key)
        for metric in sorted(REQUIRED_FLOORS - floors.keys()):
            print(f"FAIL floor {key[0]}/{key[1]} {metric}: no absolute floor (v1.20)")
            failures.append(f"{key} {metric}: no floor")
        for metric, floor in floors.items():
            why = not_applicable(r, metric)
            if why:
                print(f"n/a  floor {key[0]}/{key[1]} {metric}: not applicable ({why})")
                continue
            v = r.get(metric)
            if v is None:
                failures.append(f"{key} {metric}: missing")
                continue
            ok = passes(metric, v, floor)
            rel = "in" if isinstance(floor, list) else "<=" if metric in LOWER_IS_BETTER else ">="
            tag = "info" if report_only(r, metric) else "ok  " if ok else "FAIL"
            print(f"{tag} floor {key[0]}/{key[1]} {metric} {v:.6g} {rel} {floor}")
            if not ok and not report_only(r, metric):
                failures.append(f"{key} {metric} {v} vs floor {floor}")
        for metric, tol in _drift(gates, *key).items():
            if report_only(r, metric):
                continue
            why = not_applicable(r, metric)
            if why:
                print(f"n/a  drift {key[0]}/{key[1]} {metric}: not applicable ({why})")
                continue
            if "ref" in tol:
                path = (gates_dir or DEFAULT_GATES.parent) / tol["ref"]
                if path not in refs:
                    refs[path] = _by_key(load_records(path))
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
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--allow-unstamped", action="store_true", help="admit records without git_head (pre-v1.16)")
    args = ap.parse_args()
    gates = json.loads(args.gates.read_text())
    cand = load_records(args.candidate)
    base = load_records(args.baseline) if args.baseline else None
    failures = check(cand, base, gates, args.gates.parent, args.allow_unstamped)
    print(f"{len(failures)} failure(s)" + (": " + "; ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
