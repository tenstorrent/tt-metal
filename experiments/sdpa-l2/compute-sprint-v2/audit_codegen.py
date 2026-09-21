"""Audit selected saved compiler-experiment results; does not rerun hardware."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(path):
    rows = json.loads(path.read_text())
    baselines = {
        (row["variant"], row["distribution"], row["seed"]): row
        for row in rows if row["codegen_level"] == "original"
    }
    assert baselines
    for row in rows:
        baseline = baselines[row["variant"], row["distribution"], row["seed"]]
        assert row["raw_bits_equal"] and row["mismatch_count"] == 0
        assert row["raw_trace_equal"] and row["trace_equal"]
        assert row["full_output_sha256"] == baseline["full_output_sha256"]
        for key in ("q_chunk", "k_chunk", "dim", "cores", "kv_slots", "cb_bytes",
                    "q_repeats", "k_chunks", "distinct_kv", "mode", "candidate"):
            assert row[key] == baseline[key], (path, key)
        assert len(row["actual_descriptors"]) == len(baseline["actual_descriptors"])
        for candidate_desc, baseline_desc in zip(row["actual_descriptors"], baseline["actual_descriptors"]):
            candidate_defines = dict(candidate_desc["defines"])
            baseline_defines = dict(baseline_desc["defines"])
            candidate_defines.pop("SDPA_CODEGEN_OPT", None)
            baseline_defines.pop("SDPA_CODEGEN_OPT", None)
            assert candidate_defines == baseline_defines
            assert candidate_desc["compile_time_args"] == baseline_desc["compile_time_args"]
        assert math.isclose(row["tflops_per_core"], row["useful_flops"] / (row["median_ms"] * 1e9), rel_tol=1e-10)
        for name, expected in row["private_source_sha256"].items():
            assert hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == expected
        provenance = json.loads((path.parent / (row["label"] + ".provenance.json")).read_text())
        repo = Path(__file__).resolve().parents[3]
        for name, expected in provenance["source_sha256"].items():
            # AppleDouble transfer metadata is not a kernel source dependency.
            if Path(name).name.startswith("._"):
                continue
            assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records_checked": len(rows), "pass": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    results = [audit(path) for path in args.files]
    report = {"all_pass": True, "files": results,
              "scope": "Saved raw-output/trace equality, private wrapper source hashes, unchanged numeric defines/geometry and useful FLOP arithmetic; not an independent device run or transitive compiler audit"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_pass": True, "files": len(results),
                      "records_checked": sum(result["records_checked"] for result in results)}))
