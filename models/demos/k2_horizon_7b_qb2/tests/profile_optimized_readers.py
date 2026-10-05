"""Opposite-order device measurements of 1/2/3 readers at each selected role geometry."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .analyze_fused_profile import window

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parents[1] / "doc/optimized_decoder/readers"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multichip", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    doc = args.output_dir or DOC
    records = []
    for direction in ["forward", "reverse"]:
        dest = doc / direction
        dest.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for key in ["TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"]:
            env.pop(key, None)
        env["TT_METAL_PROFILER_DIR"] = str(dest)
        cmd = [
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-p",
            "-v",
            "--check-exit-code",
            "--op-support-count",
            "5000",
            "--web-app-port",
            "55183",
            "-o",
            str(dest),
            "-m",
            "models.demos.k2_horizon_7b_qb2.tests.sweep_optimized_matmuls",
            "--shortlist",
            "--profile",
            "--cores",
            "64",
            "--output",
            str(dest / "results.json"),
        ]
        if args.multichip:
            cmd += ["--multichip", "--roles", "qkv", "o", "gate", "up", "down"]
        if direction == "reverse":
            cmd.append("--reverse")
        with (dest / "capture.log").open("w") as f:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        ops = max(dest.glob("reports/*/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
        with ops.open() as f:
            raw = list(csv.DictReader(f))
        with (dest / ".logs/profile_log_device.csv").open() as f:
            header = "".join(next(f) for _ in range(3))
        freq = float(re.search(r"CHIP_FREQ\[MHz\]\s*:\s*([0-9.]+)", header).group(1))
        for row in json.loads((dest / "results.json").read_text())["records"]:
            if not row.get("passed"):
                continue
            name = f"{row['role']}_c{row['cores']}_k{row['block_w']}_r{row['readers']}_f{int(row['fp32'])}"
            if args.multichip:
                from .analyze_multichip_profile import window as mesh_window

                timing = mesh_window(raw, name, name + "_END", freq)
            else:
                timing = window(raw, name, name + "_END", freq)
            physical = row["k"] * row["nphysical"] * 576 / 1024
            logical = row["k"] * row["n"] * 0.5
            records.append(
                {
                    **row,
                    "order": direction,
                    "device_us": timing["device_us"],
                    "physical_weight_bytes": physical,
                    "logical_weight_bytes": logical,
                    "physical_GB_s": physical / timing["device_us"] / 1000,
                    "logical_GB_s": logical / timing["device_us"] / 1000,
                    "dram_pct": 100 * physical / (512e9 * timing["device_us"] / 1e6),
                    "evidence": str(ops),
                }
            )
        (doc / "comparison.json").write_text(
            json.dumps(
                {
                    "records": records,
                    "basis": "maximum per-rank whole matmul firmware span; opposite reader/role order; per-rank bandwidth against 512GB/s P300c chip DRAM peak",
                    "participating_devices": 4 if args.multichip else 1,
                    "commands": "profile_optimized_readers --multichip; sweep_optimized_matmuls --multichip --shortlist --profile (forward/reverse)",
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
