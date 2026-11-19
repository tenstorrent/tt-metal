# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Advice-enabled TP4 Tracy reports with exact raw-CSV and source provenance."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/multichip_decoder"
for layer in [0, 3]:
    directory = ROOT / "tracy" / str(layer)
    measured = json.loads((ROOT / f"profile_{layer}.json").read_text())
    source = max(directory.rglob("ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
    for mode in ["prefill", "decode"]:
        raw = directory / f"{mode}_ops.csv"
        shutil.copy2(source, raw)
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
                    "prefill_length": 1024,
                    "decode_position": 1024,
                    "decode_traced": True,
                    "warmed": True,
                },
                indent=2,
            )
            + "\n"
        )
print("PERF_REPORTS_PASS")
