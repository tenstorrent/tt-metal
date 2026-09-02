# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Apply the strict release gate to batch-1 latency sweep JSON files.

An initially slower point requires two same-shape retries. With all three
samples present, their median is compared with the committed cardrun2 baseline
at the precision shown in the model card. Any confirmed slowdown fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = {
    "ttft_ms": 1,
    "tpot_ms": 2,
    "e2el_ms": 1,
}


def _displayed(value: float, places: int) -> float:
    return float(f"{value:.{places}f}")


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data.get("rows"), list):
        raise ValueError(f"{path}: missing rows list")
    return data


def evaluate(baseline: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in candidates:
        for key in ("batch_size", "osl", "hf_advertised_context"):
            if candidate.get(key) != baseline.get(key):
                raise ValueError(
                    f"candidate {key}={candidate.get(key)!r} does not match " f"baseline {baseline.get(key)!r}"
                )

    samples: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        for row in candidate["rows"]:
            samples.setdefault(int(row["isl"]), []).append(row)

    report_rows = []
    for baseline_row in baseline["rows"]:
        isl = int(baseline_row["isl"])
        shape_samples = samples.get(isl, [])
        if not shape_samples:
            report_rows.append({"isl": isl, "status": "missing", "samples": 0})
            continue

        metrics = {}
        regressed = False
        for metric, places in METRICS.items():
            median = statistics.median(float(row[metric]) for row in shape_samples)
            baseline_display = _displayed(float(baseline_row[metric]), places)
            candidate_display = _displayed(median, places)
            slower = candidate_display > baseline_display
            regressed |= slower
            metrics[metric] = {
                "baseline": baseline_display,
                "candidate_median": candidate_display,
                "regressed": slower,
            }

        if regressed and len(shape_samples) < 3:
            status = "retry_required"
        elif regressed:
            status = "fail"
        else:
            status = "pass"
        report_rows.append(
            {
                "isl": isl,
                "status": status,
                "samples": len(shape_samples),
                "metrics": metrics,
            }
        )

    statuses = {row["status"] for row in report_rows}
    if "fail" in statuses:
        outcome = "fail"
    elif statuses & {"missing", "retry_required"}:
        outcome = "incomplete"
    else:
        outcome = "pass"
    return {
        "gate": "strict_no_displayed_precision_regression",
        "outcome": outcome,
        "rows": report_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = evaluate(_load(args.baseline), [_load(path) for path in args.candidate])
    for row in report["rows"]:
        details = " ".join(
            f"{name}={values['candidate_median']}/{values['baseline']}"
            for name, values in row.get("metrics", {}).items()
        )
        print(f"ISL {row['isl']:>6}: {row['status']:<14} " f"samples={row['samples']} {details}".rstrip())
    print(f"OUTCOME: {report['outcome']}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if report["outcome"] == "incomplete":
        raise SystemExit(2)
    if report["outcome"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
