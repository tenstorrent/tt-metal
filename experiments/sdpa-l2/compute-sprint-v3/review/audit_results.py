"""Independent standard-library checks of B records and shared E/G valid evidence."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def gate(comparison):
    baseline, candidate = comparison["baseline_metrics"], comparison["candidate_metrics"]
    assert baseline["finite"] and candidate["finite"]
    assert baseline["reference_is_zero"] == candidate["reference_is_zero"]
    if baseline["reference_is_zero"]:
        limit = max(1e-6, 1.05 * baseline["max_abs"])
        value = candidate["max_abs"]
    else:
        limit = 1.05 * baseline["l2_pct"] + 0.0001
        value = candidate["l2_pct"]
    assert math.isfinite(value) and value <= limit
    assert comparison["acceptance"]["pass"]
    assert math.isclose(limit, comparison["acceptance"]["limit"], rel_tol=1e-12, abs_tol=1e-12)


def timing(record, flops, list_key):
    values = record[list_key]
    if not values:
        return
    median = statistics.median(values)
    assert math.isclose(median, record["median_ms"], rel_tol=1e-12)
    assert math.isclose(flops / (median*1e9), record["tflops_per_core"], rel_tol=1e-12)


def run(kind):
    files = sorted(HERE.glob("B-*.json")) if kind == "B" else sorted(
        (HERE.parent / "compensated").glob("*-valid-*.json"))
    cache, mismatches, total, detail = {}, [], 0, []
    def sources(pins, filename):
        for name, expected in pins.items():
            path = ROOT / name
            if name not in cache:
                cache[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            if cache[name] != expected:
                mismatches.append(dict(result=filename, source=name, expected=expected, actual=cache[name]))
    for path in files:
        report = json.loads(path.read_text())
        if kind == "B":
            assert report["all_numerical_gates_pass"], path.name
            sources(report["source_sha256"], path.name)
            records = report["results"]
            if isinstance(records, dict):
                for comparison in report["comparisons"].values():
                    gate(comparison)
                for record in records.values():
                    assert all(record["trace_raw_equal"])
                    timing(record, report["useful_flops"], "times_ms")
                assert report["input_immutability_pass"] and report["source_immutability_pass"]
            else:
                assert report["completed"] and report["input_immutability_pass"] and report["source_immutability_pass"]
                for record in records:
                    assert all(record["trace_raw_equal"])
                    for key in ("vs_v1", "vs_v2_early"):
                        if key in record:
                            gate(record[key])
                    if record["mode"] == "v2_early":
                        assert record["raw_equal_v1"]
        else:
            records = report if isinstance(report, list) else [report]
            for record in records:
                gate(record["numerical_comparison"])
                for key in ("source_stable", "original_host_immutable", "original_device_immutable",
                            "prepared_device_immutable", "canonical_adapter_equal", "trace_equal"):
                    assert record[key], (path.name, key)
                assert record["mandatory_trace_replays_per_kernel"] >= 2
                assert not any(record["preprocessing_mismatches"])
                base, candidate = record["baseline_metadata"], record["candidate_metadata"]
                for key in ("cb_specs", "input_slots", "fidelity", "fp32_dst"):
                    assert base[key] == candidate[key]
                strip = lambda d: {k:v for k,v in d.items() if k != "SDPA_SPRINT_CANDIDATE_HEADER"}
                assert strip(base["defines"]) == strip(candidate["defines"])
                assert "group2_valid/compute_streaming.hpp" in candidate["defines"]["SDPA_SPRINT_CANDIDATE_HEADER"]
                assert base["input_slots"] == 2 and not base["fp32_dst"]
                for label in ("baseline", "candidate"):
                    timing(record[label], record["useful_flops"], "replay_ms")
                sources(record["source_sha256"], path.name)
        total += len(records)
        detail.append(dict(file=path.name, records=len(records)))
    result = dict(kind=kind, files=len(files), records=total, checks_pass=not mismatches,
                  source_mismatches=mismatches, detail=detail)
    print(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--kind", choices=("B", "EG"), required=True)
    p.add_argument("--output")
    args=p.parse_args()
    result=run(args.kind)
    if args.output:
        (HERE / args.output).write_text(json.dumps(result,indent=2)+"\n")
    raise SystemExit(0 if result["checks_pass"] else 1)
