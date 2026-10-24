# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy gate for a candidate bench run: the D3 floors and the drift margins against round 0.

    python models/experimental/t0_forecast/tests/perf/check_gates.py <candidate bench.json> <round-0 bench.json>

Every candidate record must meet its precision class's floors (``tests/helpers.FLOORS``) and must not fall more than
the drift margin below the matching round-0 record (``tests/helpers.DRIFT`` on the target, ``paper-accuracy``; twice
that on every other record). Metrics missing from either record are skipped and listed. ``compare_bench.py`` judges
device time; this judges accuracy. Exit status 1 when any record fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TARGET = ("paper", "accuracy")


def main(argv=None) -> int:
    from models.experimental.t0_forecast.tests.helpers import DRIFT, FLOORS

    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path)
    ap.add_argument("baseline", type=Path, help="the round-0 bench.json")
    args = ap.parse_args(argv)
    base = {(r["name"], r["precision"]): r for r in json.loads(args.baseline.read_text())}
    failed = False
    for rec in json.loads(args.candidate.read_text()):
        key = (rec["name"], rec["precision"])
        ref = base.get(key)
        scale = 1.0 if key == TARGET else 2.0
        notes, skipped = [], []
        for metric, floor in FLOORS[rec["precision"]].items():
            if metric not in rec:
                skipped.append(metric)
                continue
            value = rec[metric]
            if value < floor:
                notes.append(f"{metric} {value:.6f} < floor {floor}")
            if metric in DRIFT and ref is not None and metric in ref:
                limit = ref[metric] - scale * DRIFT[metric]
                if value < limit:
                    notes.append(
                        f"{metric} {value:.6f} drifted below round 0 {ref[metric]:.6f} - {scale * DRIFT[metric]:g}"
                    )
        verdict = "FAIL" if notes else "PASS"
        failed |= bool(notes)
        extra = f"; skipped {', '.join(skipped)}" if skipped else ""
        missing = "; no round-0 record" if ref is None else ""
        print(f"{verdict} {key[0]}-{key[1]}{': ' + '; '.join(notes) if notes else ''}{extra}{missing}")
    print("gates:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
