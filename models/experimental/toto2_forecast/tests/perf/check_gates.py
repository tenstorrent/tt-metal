# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy gate for a candidate bench run: the gate floors plus drift from round 0 of the same precision.

    python models/experimental/toto2_forecast/tests/perf/check_gates.py <candidate bench.json> <round-0 bench.json>

Each full-depth record must meet ``floors_for(checkpoint, precision)`` and stay within ``DRIFT`` of the round-0
record with the same (name, precision); ``DRIFT`` applies on the target and twice it elsewhere. With ``--suite`` both files are
suite records (``pcc_metrics.json``); only 313m and 2.5B-FT are checked there (D4 condition 1; 4m / 22m stay
floor-only, D3 ruling 1). An all-degenerate case gets no drift except abs_err_max (D4 ruling 1). A missing metric,
a missing round-0 record and a round-0 record the candidate lacks are failures (D2). ``BENCH_LAYERS`` records are skipped (their metrics mean nothing).
Exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from models.experimental.toto2_forecast.tests.golden.goldens import (
    CEILINGS,
    DRIFT,
    DEGENERATE_ABS_ERR_GROWTH,
    DRIFT_EXTRA,
    TWO_SIDED,
    all_degenerate,
    below_floors,
    floors_for,
    info_misses,
)

TARGET = ("paper", "accuracy")


def check(rec: dict, ref: dict | None) -> list[str]:
    if ref is None:
        return ["no round-0 record"]
    notes = [
        f"{k} {v[0]} misses floor {v[1]}"
        for k, v in below_floors(rec, floors_for(rec["checkpoint"], rec["precision"])).items()
    ]
    if all_degenerate(rec):  # D4 ruling 1: floors only, plus a gross-error ceiling on abs_err_max
        ceiling = ref["abs_err_max"] * (1 + DEGENERATE_ABS_ERR_GROWTH)
        if rec["abs_err_max"] > ceiling:
            notes.append(
                f"abs_err_max {rec['abs_err_max']:.6f} over {ceiling:.6f} (round 0 +{DEGENERATE_ABS_ERR_GROWTH:.0%})"
            )
        return notes
    scale = 1.0 if (rec["name"], rec["precision"]) == TARGET else 2.0
    extra = DRIFT_EXTRA.get((rec["checkpoint"], rec["precision"].split("_")[0]), {})  # D3 ruling 2, scale 1
    for metric, margin in [*DRIFT.items(), *((m, x / scale) for m, x in extra.items())]:
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
    ap.add_argument("--suite", action="store_true", help="suite records (pcc_metrics.json), 313m and 2.5B-FT only")
    ap.add_argument("--precision", help="check only records of this precision")
    args = ap.parse_args(argv)

    def records(path):
        out = []
        for r in json.loads(path.read_text()):
            if r.get("bench_layers") or (args.precision and r["precision"] != args.precision):
                continue
            if args.suite:
                if r["ckpt"] not in ("313m", "2.5B-FT"):
                    continue
                r = {**r, "name": f"{r['ckpt']}/{r['case']}", "checkpoint": r["ckpt"]}
            out.append(r)
        return out

    base = {(r["name"], r["precision"]): r for r in records(args.baseline)}
    failed = False
    cand = records(args.candidate)
    for rec in cand:
        key = (rec["name"], rec["precision"])
        notes = check(rec, base.get(key))
        failed |= bool(notes)
        info = info_misses(rec, rec["checkpoint"], rec["precision"])  # D3: printed, not gated
        print(
            ("FAIL " if notes else "PASS ")
            + "-".join(key)
            + (": " + "; ".join(notes) if notes else "")
            + ("  [info: " + "; ".join(f"{k} {v[0]} vs {v[1]}" for k, v in info.items()) + "]" if info else "")
        )
    seen = {(r["name"], r["precision"]) for r in cand}
    for key in sorted(set(base) - seen):  # D2 item 4: a round-0 record the candidate run lacks is a failure
        failed = True
        print("FAIL " + "-".join(key) + ": missing from the candidate run")
    print("gates:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
