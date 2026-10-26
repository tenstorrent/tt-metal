# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Check a candidate bench JSON against the gates and the fixed drift records (CPU only, no device):

    python models/experimental/timesfm3_forecast/tests/perf/check_gates.py <candidate.json>

For every record (``test``, ``name``, ``precision``) of the candidate:
- every metric of ``golden_io.GATES[precision]`` must hold, and ``inactive_err_max_rel`` its per-case floor
  (``golden_io.INACTIVE_ERR_MAX_REL``; N/A when the record has 0 inactive rows);
- against the fixed record of the metric (``REFERENCES``, D5), a higher-is-better metric may fall by at most
  ``golden_io.DRIFT``, a ``LOWER_IS_BETTER`` metric may rise by at most it, a ``TWO_SIDED`` metric may move either
  way by at most it (``golden_io.DRIFT_CASE`` overrides a band per case); metrics without a ``DRIFT`` entry are
  floor-only;
- ``abs_err_max`` may grow to at most 1.5x its fixed record's (PLAN v1.16);
- a drift metric missing from the candidate or from its fixed record fails (PLAN v1.17), as does a missing gated
  metric or a missing ``git_dirty``; a non-empty ``git_dirty`` fails (PLAN v1.16);
- every record of each fixed record file must be present in the candidate.
Device time is judged separately. Exits 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.experimental.timesfm3_forecast.tests.golden_io import (  # noqa: E402
    DRIFT,
    DRIFT_CASE,
    GATES,
    gate_failures,
    inactive_floor,
)

LOWER_IS_BETTER = {"spread_ratio_p99", "abs_err_max", "inactive_err_max_rel"}
TWO_SIDED = {"spread_ratio_p50"}
ABS_ERR_GROWTH = 1.5

_OPT = Path(__file__).resolve().parents[2] / "doc" / "optimize"
# D5: the fixed record per metric (and per case where a case has no round-0 record).
DEFAULT_REFERENCE = _OPT / "iter_00" / "r1_1" / "bench.json"  # D4
METRIC_REFERENCES = {
    "spread_ratio_p50": _OPT / "iter_00" / "r1_4c" / "bench.json",  # first record with p50 (52594075d0c)
    "abs_err_max": _OPT / "final" / "r0_abs_err" / "bench.json",  # r1_1's commit, HEAD's head_metrics (job 810)
}
CASE_REFERENCES = {"gift_e2e": _OPT / "final" / "gift_e2e_798.json"}  # no round-0 record: job 798 is fixed


def _key(r: dict) -> tuple:
    return r.get("test"), r["name"], r["precision"]


def _load(path: Path) -> dict:
    return {_key(r): r for r in json.loads(Path(path).read_text())}


def reference(metric: str, key: tuple, refs: dict) -> tuple[float | None, str]:
    """(fixed value or None, file name) of ``metric`` for the record ``key``."""
    path = CASE_REFERENCES.get(key[1]) or METRIC_REFERENCES.get(metric, DEFAULT_REFERENCE)
    return refs[path].get(key, {}).get(metric), Path(path).parent.name


def drift_ok(metric: str, case: str, v: float, ref: float) -> bool:
    if metric == "abs_err_max":
        return v <= ABS_ERR_GROWTH * ref
    band = DRIFT_CASE.get((case, metric), DRIFT[metric])
    if metric in TWO_SIDED:
        return abs(v - ref) <= band
    return v <= ref + band if metric in LOWER_IS_BETTER else v >= ref - band


def check(cand: dict, refs: dict) -> int:
    bad = 0
    for path, base in refs.items():
        if path in CASE_REFERENCES.values():
            continue
        for key in base.keys() - cand.keys():
            print(f"== {'/'.join(str(k) for k in key)}: fixed record ({Path(path).parent.name}) missing from the candidate")
            bad += 1
    for key, r in cand.items():
        print(f"== {'/'.join(str(k) for k in key)}")
        if "git_dirty" not in r:
            print("   git_dirty missing  FAIL")
            bad += 1
        elif r["git_dirty"]:
            print(f"   dirty tree at {r.get('git_head', '?')[:11]}: {r['git_dirty']!r}  FAIL")
            bad += 1
        gated = dict(GATES[r["precision"]])
        n_inactive = r.get("rows_inactive")
        if n_inactive is None:
            print("   rows_inactive missing  FAIL")
            bad += 1
        elif n_inactive == 0:
            print("   inactive_err_max_rel   N/A (rows_inactive 0)")
        else:
            gated["inactive_err_max_rel"] = ("<=", inactive_floor(r["name"]))
        for metric in [*gated, *(m for m in ("abs_err_max",) if m not in gated)]:
            op, bound = gated.get(metric, ("", ""))
            if metric not in r:
                print(f"   {metric:22s} missing  FAIL")
                bad += 1
                continue
            v = r[metric]
            if metric == "inactive_err_max_rel":
                gate_ok = metric not in gate_failures(r, r["precision"], r["name"])
            else:
                gate_ok = not bound or metric not in gate_failures({metric: v}, r["precision"])
            ok, ref_s = gate_ok, ""
            if metric in DRIFT or metric == "abs_err_max":
                ref, src = reference(metric, key, refs)
                if ref is None:
                    ref_s, ok = f"  fixed record ({src}) lacks it", False
                else:
                    d = drift_ok(metric, r["name"], v, ref)
                    ok = ok and d
                    ref_s = f"  fixed {ref:.6f} ({src}){'' if d else ' DRIFT'}"
            status = "ok" if ok else ("GATE" if not gate_ok else "FAIL")
            bad += not ok
            print(f"   {metric:22s} {v:.6f}  {op} {bound}{ref_s}  {status}")
    print("all gates hold" if not bad else f"{bad} check(s) failed")
    return 1 if bad else 0


def main(candidate: str) -> int:
    paths = {DEFAULT_REFERENCE, *METRIC_REFERENCES.values(), *CASE_REFERENCES.values()}
    return check(_load(Path(candidate)), {p: _load(p) for p in paths})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
