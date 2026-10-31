"""Whole reduced-model device windows and advice-backed perf tables."""
import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument(
        "--output", type=Path, default=Path("models/autoports/google_gemma_4_31b_it/doc/optimized_full_model")
    )
    args = p.parse_args()
    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    result = dict(
        scope="reduced real layers0/5 plus complete terminal and buffered sampling; not full-model device latency",
        source_csv=str(args.csv),
        window_basis="max per-device FW end minus min FW start; includes all operations and gaps",
        windows={},
    )
    for name in ("PREFILL", "DECODE", "MODEL_ONLY", "SAMPLER_ONLY", "TERMINAL"):
        start = next(i for i, r in enumerate(rows) if r["OP CODE"] == f"PERF_{name}")
        end = next(i for i, r in enumerate(rows[start + 1 :], start + 1) if r["OP CODE"] == f"PERF_{name}_END")
        devices = defaultdict(list)
        for row in rows[start + 1 : end]:
            if row["DEVICE FW START CYCLE"] and row["DEVICE FW DURATION [ns]"]:
                devices[row["DEVICE ID"]].append(row)
        measurements = {}
        for device, ops in devices.items():
            ratios = [
                (float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"]))
                / float(r["DEVICE FW DURATION [ns]"])
                for r in ops
                if float(r["DEVICE FW DURATION [ns]"]) > 0
            ]
            ratio = statistics.median(ratios)
            us = (
                (
                    max(float(r["DEVICE FW END CYCLE"]) for r in ops)
                    - min(float(r["DEVICE FW START CYCLE"]) for r in ops)
                )
                / ratio
                / 1000
            )
            measurements[device] = dict(
                device_us=us, cycles_per_ns=ratio, ops=len(ops), trace_ids=sorted(set(r["METAL TRACE ID"] for r in ops))
            )
        assert measurements, name
        result["windows"][name] = dict(
            device_us=max(v["device_us"] for v in measurements.values()), devices=measurements
        )
        stem = args.output / f"{name.lower()}_perf_report"
        common = [
            "python_env/bin/tt-perf-report",
            str(args.csv),
            "--start-signpost",
            f"PERF_{name}",
            "--end-signpost",
            f"PERF_{name}_END",
            "--no-color",
        ]
        with stem.with_suffix(".console.log").open("w") as f:
            subprocess.run(
                common + ["--csv", str(stem.with_suffix(".csv"))], stdout=f, stderr=subprocess.STDOUT, check=True
            )
        with stem.with_suffix(".txt").open("w") as f:
            subprocess.run(common + ["--no-summary"], stdout=f, stderr=subprocess.STDOUT, check=True)
    (args.output / "terminal_attribution.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
