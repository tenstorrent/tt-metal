# SPDX-License-Identifier: Apache-2.0
"""Render retained Tracy CSVs and derive per-pass time with explicit units."""

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/functional_decoder"
summary = {"units": "Device Time and Op-to-Op Gap are microseconds; sum / repetitions / 1000 gives ms", "rows": []}
for variant in ("before_numerical_fix/untracked_perf", ""):
    for layer in (0, 4):
        directory = ROOT / variant / f"tracy/layer_{layer}"
        source = max((directory / "raw/reports").glob("*/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
        profile = json.loads((ROOT / variant / f"profile_{layer}.json").read_text())
        assert profile["provenance"]["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "0"
        assert profile["provenance"]["environment"]["TT_METAL_WATCHER"] is None
        for mode in ("prefill", "decode"):
            target = directory / f"{mode}_ops.csv"
            shutil.copy2(source, target)
            args = [
                "tt-perf-report",
                str(target),
                "--start-signpost",
                f"PERF_{mode.upper()}",
                "--end-signpost",
                f"PERF_{mode.upper()}_END",
                "--no-advice",
            ]
            commands = [
                args + ["--csv", str(directory / f"{mode}_perf_report.csv")],
                args + ["--no-summary", "--no-color"],
            ]
            for command, filename in zip(commands, (f"{mode}_perf_report.console.log", f"{mode}_perf_report.txt")):
                with (directory / filename).open("w") as output:
                    subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
            with (directory / f"{mode}_perf_report.csv").open() as source_csv:
                rows = list(csv.DictReader(source_csv))
            assert rows and all(r["Device"] == "0" and r["Device Time"] not in (None, "") for r in rows)
            repeats = profile["repetitions"]
            kernel_ms = sum(float(r["Device Time"] or 0) for r in rows) / repeats / 1000
            gap_ms = sum(float(r["Op-to-Op Gap"] or 0) for r in rows) / repeats / 1000
            if variant:
                continue
            old_path = ROOT / f"before_numerical_fix/untracked_perf/tracy/layer_{layer}/{mode}_perf_report.csv"
            with old_path.open() as old_file:
                baseline = sum(float(r["Device Time"] or 0) for r in csv.DictReader(old_file)) / repeats / 1000
            summary["rows"].append(
                dict(
                    layer=layer,
                    allocation_tracking=False,
                    runtime_diagnostics=False,
                    baseline_csv=str(old_path.relative_to(ROOT)),
                    mode=mode,
                    batch=1,
                    prefill_tokens=128,
                    decode_position=128,
                    repetitions=repeats,
                    device_kernel_ms=kernel_ms,
                    device_gap_ms=gap_ms,
                    host_elapsed_ms=profile["host_elapsed_ms"][mode],
                    baseline_device_kernel_ms=baseline,
                    numerical_repair_cost_ms=kernel_ms - baseline,
                    numerical_repair_cost_percent=(kernel_ms / baseline - 1) * 100,
                    measured_op_rows=len(rows),
                    source_csv=str(source.relative_to(ROOT)),
                    source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    commands=commands,
                )
            )
(ROOT / "performance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
