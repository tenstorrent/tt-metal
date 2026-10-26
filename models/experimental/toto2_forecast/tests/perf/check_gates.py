# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy gate for a candidate bench run: the gate floors plus drift from round 0 of the same precision.

    python models/experimental/toto2_forecast/tests/perf/check_gates.py <candidate bench.json> <round-0 bench.json>

Each full-depth record must meet ``floors_for(checkpoint, precision)`` and stay within ``DRIFT`` of the round-0
record with the same (name, precision); ``DRIFT`` applies on the target and twice it elsewhere. A missing metric,
a missing round-0 record and a round-0 record the candidate lacks are failures (D2). ``BENCH_LAYERS`` records are skipped (their metrics mean nothing).
Exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from models.experimental.toto2_forecast.tests.golden.goldens import CEILINGS, DRIFT, TWO_SIDED, below_floors, floors_for

TARGET = ("paper", "accuracy")


def check(rec: dict, ref: dict | None) -> list[str]:
    if ref is None:
        return ["no round-0 record"]
    notes = [f"{k} {v[0]} misses floor {v[1]}" for k, v in below_floors(rec, floors_for(rec["checkpoint"], rec["precision"])).items()]
    scale = 1.0 if (rec["name"], rec["precision"]) == TARGET else 2.0
    for metric, margin in DRIFT.items():
        if metric not in rec or metric not in ref:
            notes.append(f"{metric} missing")
            continue
        if metric in TWO_SIDED:
            bad = abs(rec[metric] - ref[metric]) > scale * margin
        elif metric in CEILINGS:
            bad = rec[metric] > ref[metric] + scale * margin
        else:
            bad = rec[metric] < ref[metric] - scale * margin
        if bad:
            notes.append(f"{metric} {rec[metric]:.6f} drifts from round 0 {ref[metric]:.6f} by over {scale * margin:g}")
    return notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("baseline", type=Path)
    args = ap.parse_args(argv)
    base = {(r["name"], r["precision"]): r for r in json.loads(args.baseline.read_text()) if not r.get("bench_layers")}
    failed = False
    for rec in json.loads(args.candidate.read_text()):
        if rec.get("bench_layers"):
            continue
        key = (rec["name"], rec["precision"])
        notes = check(rec, base.get(key))
        failed |= bool(notes)
        print(("FAIL " if notes else "PASS ") + "-".join(key) + (": " + "; ".join(notes) if notes else ""))
    seen = {(r["name"], r["precision"]) for r in json.loads(args.candidate.read_text()) if not r.get("bench_layers")}
    for key in sorted(set(base) - seen):  # D2 item 4: a round-0 record the candidate run lacks is a failure
        failed = True
        print("FAIL " + "-".join(key) + ": missing from the candidate run")
    print("gates:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
