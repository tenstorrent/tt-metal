# SPDX-License-Identifier: Apache-2.0
"""Validate Stage5 artifacts without importing TTNN or opening devices."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/optimized_multichip_decoder"


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-review", action="store_true")
    args = parser.parse_args()
    digest = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    checks = []

    def check(label, condition):
        assert condition, label
        checks.append(label)

    for layer in (0, 4):
        for tokens in (128, 8193):
            item = read(f"final_{layer}_{tokens}.json")
            check(
                f"default {layer}/{tokens}", item["model_class"] == "MultichipDecoder" and item["mesh_shape"] == [1, 4]
            )
            check(f"current source {layer}/{tokens}", item["source_sha256"] == digest)
            check(
                f"PCC {layer}/{tokens}",
                all(v >= (0.99999 if k.startswith("replica") else 0.995) for k, v in item["pcc"].items()),
            )
            check(
                f"trace timing {layer}/{tokens}",
                item["decode_trace"]
                and not item["trace_blocking"]
                and item["repetitions"] >= 100
                and all(v > 0 for v in item["latency_ms"].values()),
            )
            before = read(f"before_{layer}_{tokens}.json")
            check(
                f"baseline {layer}/{tokens}",
                before["repetitions"] >= 100 and before["pcc"]["prefill"] >= 0.995 and before["pcc"]["decode"] >= 0.995,
            )
        for tag in ("coverage", "watcher", "watcher_boundaries"):
            item = read(f"{tag}_{layer}.json")
            check(f"coverage source {tag}/{layer}", item["source_sha256"] == digest)
            check(
                f"coverage contracts {tag}/{layer}",
                item["tracking"] == 1
                and item["skip_program_cache"] == 0
                and item["runtime_audit"]
                and item["trace_variants"] == [1, 3, 17, 32]
                and item["repeated_replays"] >= 96,
            )
            check(
                f"coverage PCC {tag}/{layer}",
                all(r["pcc"] >= 0.995 and min(r.get("per_request_pcc", [1])) >= 0.995 for r in item["rows"]),
            )
            if tag.startswith("watcher"):
                check(f"worker watcher {tag}/{layer}", bool(item["watcher"]))
            if tag == "watcher_boundaries":
                check(
                    f"projection and chunk boundaries {layer}",
                    {1023, 1024, 1025, 8191, 8192, 8193} <= set(item["lengths"]),
                )
                check(f"new ring {layer}", layer != 0 or item["ring_tokens"] == 8704)
        item = read(f"context_{layer}.json")
        check(
            f"context {layer}",
            item["capacity"] == 1048576
            and item["largest_physical_chunk"] == 8192
            and item["reservation_bytes"] >= 26575110144
            and item["source_sha256"] == digest
            and item["runtime_audit"]
            and item["tracking"] == 1
            and item["skip_program_cache"] == 0,
        )
        check(f"context PCC {layer}", all(r["pcc"] >= 0.995 for r in item["rows"]))
    stack = read("stack.json")
    check(
        "stack contract",
        stack["layers"] == [0, 4]
        and stack["boundary_conversions"] == 0
        and stack["shared_collective_workspace"]
        and stack["runtime_audit"]
        and stack["tracking"] == 1
        and all(r["pcc"] >= 0.995 for r in stack["rows"]),
    )
    ring = read("ring_batch.json")
    check(
        "ring owner PCC",
        ring["batch"] == 3 and ring["source_sha256"] == digest and all(r["pcc"] >= 0.995 for r in ring["rows"]),
    )
    for name in ("profile_final_0_128", "profile_final_4_128", "profile_final_0_8193"):
        for phase in ("prefill", "decode"):
            for rank in range(4):
                path = OUT / name / f"{phase}_device{rank}_report"
                rows = list(csv.DictReader(path.with_suffix(".csv").open()))
                check(
                    f"profile table {name}/{phase}/{rank}",
                    len(rows) > 10 and path.with_suffix(".txt").stat().st_size > 1000,
                )
                if phase == "decode":
                    matmuls = [r for r in rows if "Matmul" in r["OP Code"]]
                    check(
                        f"runtime precision {name}/{rank}",
                        len(matmuls) >= 7
                        and all(
                            r["Input 1 Datatype"] == "BFLOAT4_B" and r["Math Fidelity"].startswith("LoFi ")
                            for r in matmuls
                        ),
                    )
    reader = read("reader_profile/reader_summary.json")
    check(
        "reader matrix",
        len(reader["rows"]) == 12
        and {r["readers"] for r in reader["rows"]} == {1, 2, 3}
        and all(r["same_dtype_control_pcc"] >= 0.99999 and r["device_median_ns"] > 0 for r in reader["rows"]),
    )
    check("context manifest", read("memory_capacity_plan.json")["status"] == "validated")
    for name in (
        "README.md",
        "work_log.md",
        "topology_audit.md",
        "candidate_index.csv",
        "optimization_results.md",
        "decode_accounting.json",
        "profile_analysis.json",
        "AUTODEBUG.md",
        "AUTOFIX.md",
    ):
        check(f"artifact {name}", (OUT / name).stat().st_size > 100)
    if args.require_review:
        review = (OUT / "stage_review_final.md").read_text()
        check("independent review", "Verdict: clean-pass" in review)
    result = dict(source_sha256=digest, passed=True, checks=checks, independent_review_required=args.require_review)
    (OUT / "evidence_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
