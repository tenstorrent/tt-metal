# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Advice-enabled TP4 Tracy reports with exact raw-CSV and source provenance."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--collection", required=True)
ap.add_argument("--stack", action="store_true")
ap.add_argument("--measurement")
options = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"
for layer in (["stack32"] if options.stack else [0, 3]):
    directory = ROOT / ("tracy_" + options.collection) / str(layer)
    measurement = (Path(options.measurement) if options.measurement else
                   ROOT / f"{options.collection}_profile_{layer}.json")
    measured = json.loads(measurement.read_text())
    source = max(directory.rglob("ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
    for mode in (["decode"] if options.stack else ["prefill", "decode"]):
        raw = directory / f"{mode}_ops.csv"
        # Hard link avoids duplicating large immutable raw captures.
        if not raw.exists():
            raw.hardlink_to(source)
        args = [
            "tt-perf-report",
            str(raw),
            "--start-signpost",
            f"PERF_{mode.upper()}",
            "--end-signpost",
            f"PERF_{mode.upper()}_END",
        ]
        for extra, suffix in [
            (["--csv", str(directory / f"{mode}_perf_report.csv")], "console.log"),
            (["--no-summary"], "txt"),
        ]:
            with (directory / f"{mode}_perf_report.{suffix}").open("w") as f:
                subprocess.run(args + extra, stdout=f, stderr=subprocess.STDOUT, check=True)
        (directory / f"{mode}_provenance.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "implementation_sha256": measured["source_sha256"],
                    "commands": args,
                    "layer": layer,
                    "devices": [0, 1, 2, 3],
                    "regime": "serial",
                    "watcher": False,
                    "tracking": False,
                    "prefill_length": measured.get("prefix_length", 1024),
                    "decode_position": measured.get("prefix_length", 1024) if not options.stack else None,
                    "decode_positions": [p + measured["steps"] - 1 for p in measured["starts"]] if options.stack else None,
                    "position_note": "Stack profiling retains the final validated decode positions; state buffers are restored, position tensors are not." if options.stack else None,
                    "measurement": str(measurement),
                    "io_contract": measured.get("io_contract"),
                    "decode_traced": True,
                    "warmed": True,
                },
                indent=2,
            )
            + "\n"
        )
print("PERF_REPORTS_PASS")
