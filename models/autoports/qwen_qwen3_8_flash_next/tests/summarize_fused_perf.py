# SPDX-License-Identifier: Apache-2.0
"""Aggregate source-matched warmed timings and validated profiler windows."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="reviewed")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / "doc/fused_decoder"
    source_hash = hashlib.sha256((root / "tt/fused_decoder.py").read_bytes()).hexdigest()
    result = dict(runtime_sha256=source_hash, layers={})
    for layer in (0, 1, 3):
        item = {}
        for label, tag, profile in (
            ("functional", f"matched_baseline_l{layer}", f"functional_l{layer}"),
            ("fused", f"delivered_l{layer}_{args.suffix}", f"{args.suffix}_fused_l{layer}"),
        ):
            timing_path = evidence / tag / "timing.json"
            timing = json.loads(timing_path.read_text())
            assert timing["decode_replays"] == 50 and len(timing["prefill_host_ms"]) == 5
            assert not timing["profiler"] and timing["tracking"] == "0"
            if label == "fused":
                assert timing["source_sha256"] == source_hash
            profile_path = evidence / "tracy" / profile / "provenance.json"
            provenance = json.loads(profile_path.read_text())
            assert provenance["report_tool_version"] == "1.4.1"
            windows = {}
            for mode in ("prefill", "decode"):
                record = provenance["windows"][mode]
                assert record["repetitions"] == (1 if mode == "prefill" else 10)
                paths = [profile_path.parent / (mode + "_perf_report" + extension) for extension in (".csv", ".txt")]
                assert all(path.stat().st_size for path in paths)
                windows[mode] = dict(
                    per_pass_device_us=record["per_pass_device_us"],
                    operations_per_pass=record["device_operation_rows"] / record["repetitions"],
                    artifacts={
                        str(path.relative_to(evidence)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
                    },
                )
            item[label] = dict(
                timing_artifact=str(timing_path.relative_to(evidence)),
                timing_sha256=hashlib.sha256(timing_path.read_bytes()).hexdigest(),
                prefill_ms=timing["median_prefill_host_ms"],
                traced_decode_ms=timing["median_traced_decode_ms"],
                profiler_artifact=str(profile_path.relative_to(evidence)),
                profiler=windows,
            )
        item["speedup"] = {
            phase: item["functional"][phase] / item["fused"][phase] for phase in ("prefill_ms", "traced_decode_ms")
        }
        result["layers"][str(layer)] = item
    (evidence / "reviewed_performance_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
