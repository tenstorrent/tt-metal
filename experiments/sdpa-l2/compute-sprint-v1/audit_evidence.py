"""Audit explicitly selected result files, not arbitrary successful log lines.

Usage: python audit_evidence.py RESULT.json [...] --output AUDIT.json
This checks saved equality/hash and throughput arithmetic; it does not rerun
silicon, establish transitive compiler provenance, or qualify untested inputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def check_rate(record, flops):
    ms, rate = record.get("median_ms"), record.get("tflops_per_core")
    if ms is not None and rate is not None:
        assert ms > 0
        assert math.isclose(rate, flops / (ms * 1e9), rel_tol=1e-10)


def audit(path):
    data = json.loads(path.read_text())
    comparisons = 0
    if isinstance(data, list):
        baselines = {}
        for row in data:
            if row.get("current_candidate", row.get("candidate")) == "baseline":
                key = (row.get("variant"), row.get("distribution", row.get("current_distribution")))
                baselines[key] = row["full_output_sha256"]
        assert baselines, "Missing baseline"
        for row in data:
            key = (row.get("variant"), row.get("distribution", row.get("current_distribution")))
            assert row["baseline_bitwise_equal"]
            assert row["full_output_sha256"] == baselines[key], "Raw output hash mismatch"
            if row.get("useful_flops"):
                check_rate(row, row["useful_flops"])
            if row.get("median_ms") is not None:
                assert row["trace_equal"]
            comparisons += 1
    elif isinstance(data.get("results"), list):
        baselines = {r["distribution"]: r["output_sha256"] for r in data["results"]
                     if r["mode"] == "canonical"}
        assert baselines
        for row in data["results"]:
            assert row["bitwise_equal"] and row["unequal_elements"] == 0
            assert row["output_sha256"] == baselines[row["distribution"]]
            comparisons += 1
    elif "results" in data:
        baseline = data["results"]["canonical"]
        for row in data["results"].values():
            assert row["bitwise_equal_canonical"]
            assert row["unequal_elements"] == 0
            assert row["output_sha256"] == baseline["output_sha256"]
            check_rate(row, data["useful_flops"])
            comparisons += 1
    else:
        assert data["baseline_candidate_equal"] and data["trace_equal"]
        assert data["output_sha256"] == data["baseline_output_sha256"]
        assert data["source_stable"]
        assert not any(data["preprocessing_mismatches"])
        assert all(data[name] for name in (
            "original_host_immutable", "original_device_immutable", "prepared_device_immutable"
        ))
        check_rate(data["baseline"], data["useful_flops"])
        check_rate(data["candidate"], data["useful_flops"])
        comparisons = 1
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records_checked": comparisons, "pass": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite evidence"
    results = [audit(path) for path in args.files]
    report = {"files": results, "all_pass": True,
              "scope": "Saved output equality, preparation gates and useful-throughput arithmetic only"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_pass": True, "files": len(results),
                      "records_checked": sum(r["records_checked"] for r in results)}))
