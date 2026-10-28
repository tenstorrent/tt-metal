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
``git_head`` and an empty ``git_dirty`` (v1.16); ``--allow-unstamped`` admits records written before that. The stamp
is taken once per session (``SESSION_STAMP``) with ``started_at``; a record started before its commit fails (v1.23).

Key families (v1.24, D3 item 1): a metric ``f2_<m>`` is forward 2 (teacher-forced) of ``<m>``, same direction. A
record that carries a family (any ``f2_*`` key) with no floor or no drift of that family for its case fails. The
candidate set must hold every case and precision of the baseline records (``--partial`` for screens only).
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
    "bias_row_max",
    "bias_row_p95",
    "spread_ratio_p99",
    "spread_ratio_max",
}
DEFAULT_GATES = Path(__file__).resolve().parents[2] / "doc" / "optimize" / "gates.json"
# drift is always against the fixed round-0 records (both precisions, every bench and suite case; v1.14, v1.17)
DEFAULT_BASELINES = (DEFAULT_GATES.parent / "iter_00" / "bench.json", DEFAULT_GATES.parent / "iter_00" / "suite.json")


def load_records(paths) -> list[dict]:
    return [r for p in paths for r in json.loads(Path(p).read_text())]


FAMILIES = ("f2_",)  # forward 2, teacher-forced; the unprefixed keys are forward 1
MODEL_DIR = Path(__file__).resolve().parents[2]


def _git(*a) -> str:
    import subprocess

    return subprocess.run(["git", *a], cwd=MODEL_DIR, capture_output=True, text=True, check=True).stdout.strip()


def git_stamp() -> dict:
    """v1.16 / v1.20 / v1.23: HEAD, the porcelain status of the model's tt/, tests/, ops/ and reference/ (untracked
    files included) and the wall-clock start, taken together."""
    from datetime import datetime, timezone

    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain", "--", "tt", "tests", "ops", "reference"),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


_SESSION = None


def session_stamp() -> dict:
    """The stamp of this process, taken once at its first call (session start: test modules call it at import)."""
    global _SESSION
    if _SESSION is None:
        _SESSION = git_stamp()
    return dict(_SESSION)


def commit_time(head: str):
    """Committer time of ``head`` (aware datetime), or None when git does not know it."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(_git("show", "-s", "--format=%cI", head))
    except Exception:
        return None


def _family(metric: str) -> str:
    return next((f for f in FAMILIES if metric.startswith(f)), "")


def _base(metric: str) -> str:
    return metric[len(_family(metric)) :]


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
    return v <= floor if _base(metric) in LOWER_IS_BETTER else v >= floor


def check_stamp(r: dict, commit_time=commit_time) -> list[str]:
    """v1.16 / v1.23: a stamped record from a clean tree that started at or after its commit."""
    from datetime import datetime

    key = f"{r['name']}/{r['precision']}"
    if "git_head" not in r or "started_at" not in r:
        print(f"FAIL stamp {key}: no git_head / git_dirty / started_at")
        return [f"{key}: no git_head / git_dirty / started_at stamp"]
    out = []
    if r.get("git_dirty"):
        out.append(f"{key}: dirty tree at {r['git_head']}: {r['git_dirty']!r}")
    ct = commit_time(r["git_head"])
    if ct is None:
        out.append(f"{key}: commit {r['git_head']} unknown")
    elif datetime.fromisoformat(r["started_at"]) < ct:
        out.append(f"{key}: started {r['started_at']} before its commit {r['git_head'][:11]} ({ct.isoformat()})")
    for f in out:
        print(f"FAIL stamp {f}")
    return out


def check_coverage(cand: list[dict], base: list[dict] | None) -> list[str]:
    """v1.24: every case and precision of the fixed record is in the candidate set."""
    missing = sorted(set(_by_key(base or [])) - set(_by_key(cand)))
    for name, prec in missing:
        print(f"FAIL coverage {name}/{prec}: in the fixed record, not in the candidate set")
    return [f"{n}/{p}: missing from the candidate set" for n, p in missing]


def check(
    cand: list[dict],
    base: list[dict] | None,
    gates: dict,
    gates_dir: Path | None = None,
    allow_unstamped=False,
    commit_time=commit_time,
) -> list[str]:
    failures = []
    base_by = _by_key(base or [])
    refs = {}
    for r in cand:
        key = (r["name"], r["precision"])
        if not allow_unstamped:
            failures += check_stamp(r, commit_time)
        floors, drift = _floors(gates, *key), _drift(gates, *key)
        for fam in {_family(k) for k in r} - {""}:  # v1.24: a gated forward with no floors or no drift fails
            for what, table in (("floors", floors), ("drift", drift)):
                if not any(_family(m) == fam for m in table):
                    print(f"FAIL family {key[0]}/{key[1]} {fam}*: no {what}")
                    failures.append(f"{key} {fam}* family: no {what}")
        for metric, floor in floors.items():
            if metric in r and r[metric] is None:  # v1.14: skipped for too few active rows
                print(f"n/a  floor {key[0]}/{key[1]} {metric}: not applicable (n_active={r.get('active_rows')})")
                continue
            v = r.get(metric)
            if v is None:
                failures.append(f"{key} {metric}: missing")
                continue
            ok = passes(metric, v, floor)
            rel = "in" if isinstance(floor, list) else "<=" if _base(metric) in LOWER_IS_BETTER else ">="
            print(f"{'ok  ' if ok else 'FAIL'} floor {key[0]}/{key[1]} {metric} {v:.6g} {rel} {floor}")
            if not ok:
                failures.append(f"{key} {metric} {v} vs floor {floor}")
        for metric, tol in drift.items():
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
            if _base(metric) in LOWER_IS_BETTER:
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
    ap.add_argument("candidate", type=Path, nargs="+", help="bench and suite records together")
    ap.add_argument(
        "--baseline", nargs="*", type=Path, default=list(DEFAULT_BASELINES), help="none given = floors only"
    )
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--allow-unstamped", action="store_true", help="admit records without git_head (pre-v1.16)")
    ap.add_argument(
        "--partial", action="store_true", help="screens only: the candidate set may miss fixed-record cases"
    )
    args = ap.parse_args()
    st = session_stamp()
    print(f"check_gates at {st['git_head']} dirty={st['git_dirty']!r} started_at {st['started_at']}")
    gates = json.loads(args.gates.read_text())
    cand = load_records(args.candidate)
    base = load_records(args.baseline) if args.baseline else None
    failures = check(cand, base, gates, args.gates.parent, args.allow_unstamped)
    if not args.partial:
        failures += check_coverage(cand, base)
    print(f"{len(failures)} failure(s)" + (": " + "; ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
