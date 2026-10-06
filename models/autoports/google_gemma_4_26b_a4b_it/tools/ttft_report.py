# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare matched native benchmark artifacts without discarding requests."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-c1", type=Path, required=True)
    parser.add_argument("--baseline-occupancy", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--warmup-note",
        default="Consult native benchmark command logs for each cohort's warmup count; do not infer identical warmup.",
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(args.candidate.glob("s*-o*-c*.json")):
        if "request-map" in path.name:
            continue
        candidate = json.loads(path.read_text())
        if "median_ttft_ms" not in candidate:
            continue
        baseline_root = args.baseline_c1 if path.stem.endswith("-c1") else args.baseline_occupancy
        baseline_path = baseline_root / path.name
        baseline = json.loads(baseline_path.read_text())
        row = dict(
            shape=path.stem,
            baseline_path=str(baseline_path),
            candidate_path=str(path),
            baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            completed=[baseline["completed"], candidate["completed"]],
            input_lengths_equal=baseline["input_lens"] == candidate["input_lens"],
            output_lengths_equal=baseline["output_lens"] == candidate["output_lens"],
            generated_texts_equal=baseline["generated_texts"] == candidate["generated_texts"],
            differing_output_rows=[
                i
                for i, (old, new) in enumerate(zip(baseline["generated_texts"], candidate["generated_texts"]))
                if old != new
            ],
        )
        for metric in (
            "median_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_itl_ms",
            "mean_e2el_ms",
            "output_throughput",
        ):
            row[metric] = dict(baseline=baseline[metric], candidate=candidate[metric])
        row["ttft_reduction_percent"] = 100 * (1 - candidate["median_ttft_ms"] / baseline["median_ttft_ms"])
        rows.append(row)
    if not rows:
        raise RuntimeError("No benchmark rows found")
    report = dict(
        scope="Warmed end-to-end native vLLM serving; no device-profiler timings",
        occupancy_warmup_note=args.warmup_note,
        device_prefill_ms=None,
        device_decode_ms=None,
        rows=rows,
        all_outputs_equal=all(row["generated_texts_equal"] for row in rows),
        all_lengths_equal=all(row["input_lengths_equal"] and row["output_lengths_equal"] for row in rows),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_outputs_equal"] or not report["all_lengths_equal"]:
        raise AssertionError("Benchmark outputs or lengths differ; inspect saved comparison")


if __name__ == "__main__":
    main()
