# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Check a candidate bench JSON against the gates and the round-0 drift band (CPU only, no device):

    python models/experimental/timesfm3_forecast/tests/perf/check_gates.py <candidate.json> <round0.json>

For every record (``test``, ``name``, ``precision``) of the candidate:
- every metric of ``golden_io.GATES[precision]`` must hold;
- against the matching round-0 record, a higher-is-better metric may fall by at most ``golden_io.DRIFT`` and
  ``spread_ratio_p99`` may rise by at most its drift, ``spread_ratio_p50`` may move either way by at most its drift
  (``golden_io.DRIFT_CASE`` overrides a band per case); metrics without a ``golden_io.DRIFT`` entry (the min/max
  metrics) are floor-only;
- every round-0 record must be present in the candidate.
Device time is judged separately by ``compare_bench.py``. Exits 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.experimental.timesfm3_forecast.tests.golden_io import DRIFT, DRIFT_CASE, GATES, gate_failures  # noqa: E402

LOWER_IS_BETTER = {"spread_ratio_p99"}
TWO_SIDED = {"spread_ratio_p50"}


def _key(r: dict) -> tuple:
    return r.get("test"), r["name"], r["precision"]


def main(candidate: str, round0: str) -> int:
    cand = {_key(r): r for r in json.loads(Path(candidate).read_text())}
    base = {_key(r): r for r in json.loads(Path(round0).read_text())}
    bad = 0
    for key in base.keys() - cand.keys():
        print(f"== {'/'.join(str(k) for k in key)}: round-0 record missing from the candidate")
        bad += 1
    for key, r in cand.items():
        print(f"== {'/'.join(str(k) for k in key)}")
        for metric, (op, bound) in GATES[r["precision"]].items():
            if metric not in r:
                print(f"   {metric:22s} missing")
                bad += 1
                continue
            v = r[metric]
            gate_ok = metric not in gate_failures({metric: v}, r["precision"])
            drift_ok, ref = True, None
            if key in base and metric in base[key] and metric in DRIFT:
                ref = base[key][metric]
                band = DRIFT_CASE.get((r["name"], metric), DRIFT[metric])
                if metric in TWO_SIDED:
                    drift_ok = abs(v - ref) <= band
                else:
                    drift_ok = v <= ref + band if metric in LOWER_IS_BETTER else v >= ref - band
            status = "ok" if gate_ok and drift_ok else ("GATE" if not gate_ok else "DRIFT")
            bad += status != "ok"
            ref_s = "" if ref is None else f"  round 0 {ref:.6f}"
            print(f"   {metric:22s} {v:.6f}  {op} {bound}{ref_s}  {status}")
    print("all gates hold" if not bad else f"{bad} check(s) failed")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
