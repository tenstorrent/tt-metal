# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy gate for a candidate bench run: the D3 / D4 floors and the drift margins.

    python models/experimental/t0_forecast/tests/perf/check_gates.py <candidate bench.json> <round-0 bench.json> \
        [--band-baseline <round-2 bench.json>]

Every candidate record must meet its precision class's floors (``tests/helpers.FLOORS``) and must not drift more than
the margin from the matching baseline record (on the target, ``paper-accuracy``; twice that on every other record):
``tests/helpers.DRIFT`` below round 0, and ``DRIFT_BAND`` against the round-2 bench (job 614; D4), where
``spread_ratio_p99`` drifts upwards. Metrics missing from either record are skipped and listed. ``compare_bench.py`` judges
device time; this judges accuracy. Exit status 1 when any record fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TARGET = ("paper", "accuracy")


def main(argv=None) -> int:
    from models.experimental.t0_forecast.tests.helpers import DRIFT, DRIFT_BAND, FLOORS, below

    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("baseline", type=Path, help="the round-0 bench.json")
    ap.add_argument(
        "--band-baseline",
        type=Path,
        default=Path(__file__).parents[2] / "doc/optimize/iter_02/candidates/round2_final.json",
        help="the round-2 bench.json (job 614) for DRIFT_BAND",
    )
    args = ap.parse_args(argv)
    load = lambda p: {(r["name"], r["precision"]): r for r in json.loads(p.read_text())}
    base, band = load(args.baseline), load(args.band_baseline)
    failed = False
    for rec in json.loads(args.candidate.read_text()):
        if rec.get("test", "forward") != "forward":  # timing-only records (e2e_gift)
            print(f"INFO {rec['name']}-{rec['precision']}: timing only, steady {rec['steady_s'] * 1e3:.1f} ms")
            continue
        key = (rec["name"], rec["precision"])
        ref = base.get(key)
        scale = 1.0 if key == TARGET else 2.0
        notes, skipped = [], []
        for metric, floor in FLOORS[rec["precision"]].items():
            if metric not in rec:
                skipped.append(metric)
                continue
            value = rec[metric]
            if below(value, floor):
                notes.append(f"{metric} {value:.6f} outside floor {floor}")
            if metric in DRIFT and ref is not None and metric in ref:
                limit = ref[metric] - scale * DRIFT[metric]
                if value < limit:
                    notes.append(
                        f"{metric} {value:.6f} drifted below round 0 {ref[metric]:.6f} - {scale * DRIFT[metric]:g}"
                    )
            r2 = band.get(key)
            if metric in DRIFT_BAND and r2 is not None and metric in r2:
                margin = scale * DRIFT_BAND[metric]
                up = metric == "spread_ratio_p99"
                if (value > r2[metric] + margin) if up else (value < r2[metric] - margin):
                    notes.append(f"{metric} {value:.6f} drifted from round 2 {r2[metric]:.6f} by more than {margin:g}")
        verdict = "FAIL" if notes else "PASS"
        failed |= bool(notes)
        extra = f"; skipped {', '.join(skipped)}" if skipped else ""
        missing = "; no round-0 record" if ref is None else ""
        print(f"{verdict} {key[0]}-{key[1]}{': ' + '; '.join(notes) if notes else ''}{extra}{missing}")
    print("gates:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
