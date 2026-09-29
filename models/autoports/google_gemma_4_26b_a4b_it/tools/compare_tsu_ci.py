# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare exact-shape raw vLLM CI cohorts without confusing TSU and throughput."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

SHAPE = re.compile(r"_isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)\.json$")


def load_cohorts(root):
    cohorts = {}
    for path in sorted(root.rglob("benchmark_*.json")):
        match = SHAPE.search(path.name)
        if match is None:
            continue
        shape = tuple(map(int, match.groups()))
        assert shape not in cohorts, f"Duplicate shape: {shape}: {path}"
        cohorts[shape] = (path, json.loads(path.read_text()))
    assert cohorts, f"No raw benchmark cohorts found under {root}"
    return cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument(
        "--concurrency", type=int, action="append", help="Compare only explicitly selected concurrency rows"
    )
    args = parser.parse_args()
    control, candidate = load_cohorts(args.control), load_cohorts(args.candidate)
    excluded = {"control": [], "candidate": []}
    if args.concurrency:
        for label, cohorts in (("control", control), ("candidate", candidate)):
            for shape in list(cohorts):
                if shape[2] not in args.concurrency:
                    excluded[label].append(shape)
                    del cohorts[shape]
        assert control and candidate, "No cohorts match the requested concurrency filter"
    assert set(candidate) <= set(control), "Candidate contains unmatched shapes"
    if not args.allow_subset:
        assert set(control) == set(candidate), "Incomplete matrix"
    report = {
        "scope": "Exact-shape raw CI cohorts; source revisions/configuration require separate provenance review",
        "tsu_definition": "1000 / mean_tpot_ms, not aggregate output throughput",
        "control": str(args.control),
        "candidate": str(args.candidate),
        "allow_subset": args.allow_subset,
        "concurrency_filter": args.concurrency,
        "excluded_shapes": excluded,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
        "passed": False,
    }
    for shape in sorted(candidate):
        row = {"shape": shape, "checks": {}}
        baseline, selected = control[shape][1], candidate[shape][1]
        for label, (path, data) in (("control", control[shape]), ("candidate", candidate[shape])):
            assert math.isfinite(data["mean_tpot_ms"]) and data["mean_tpot_ms"] > 0, path
            row[label] = {
                "file": str(path),
                "canonical_json_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                **{key: data[key] for key in ("mean_tpot_ms", "mean_ttft_ms", "mean_e2el_ms", "median_itl_ms")},
                "tsu": 1000 / data["mean_tpot_ms"],
                "completed": data["completed"],
                "failed": data["failed"],
            }
            row["checks"][label + "_complete"] = data["completed"] == shape[3] and data["failed"] == 0
            row["checks"][label + "_text_count"] = len(data["generated_texts"]) == shape[3]
            row["checks"][label + "_exact_lengths"] = (
                data["input_lens"] == [shape[0]] * shape[3] and data["output_lens"] == [shape[1]] * shape[3]
            )
        for key in ("model_id", "tokenizer_id", "input_lens", "output_lens", "generated_texts"):
            row["checks"][key + "_equal"] = baseline[key] == selected[key]
        row["text_mismatch_indices"] = [
            index
            for index, (old, new) in enumerate(zip(baseline["generated_texts"], selected["generated_texts"]))
            if old != new
        ]
        row["tsu_gain_percent"] = 100 * (row["candidate"]["tsu"] / row["control"]["tsu"] - 1)
        report["rows"].append(row)
    report["passed"] = all(all(row["checks"].values()) for row in report["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"], "Matrix correctness/shape mismatch: inspect saved comparison"


if __name__ == "__main__":
    main()
