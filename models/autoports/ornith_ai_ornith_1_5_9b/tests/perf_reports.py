# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Retain raw Tracy CSV and render the two warmed signposted windows."""
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/functional_decoder"
for kind in ("linear", "full"):
    directory = ROOT / "tracy" / kind
    sources = list(directory.rglob("ops_perf_results_*.csv"))
    if not sources:
        for archived in directory.rglob("ops_perf_results_*.csv.gz"):
            restored = archived.with_suffix("")
            restored.write_bytes(gzip.decompress(archived.read_bytes()))
            sources.append(restored)
    source = max(sources, key=lambda p: p.stat().st_mtime)
    for mode in ("prefill", "decode"):
        raw = directory / f"{mode}_ops.csv"
        shutil.copy2(source, raw)
        args = [
            "tt-perf-report",
            str(raw),
            "--start-signpost",
            f"PERF_{mode.upper()}",
            "--end-signpost",
            f"PERF_{mode.upper()}_END",
            "--no-advice",
        ]
        with (directory / f"{mode}_perf_report.console.log").open("w") as out:
            subprocess.run(
                args + ["--csv", str(directory / f"{mode}_perf_report.csv")],
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with (directory / f"{mode}_perf_report.txt").open("w") as out:
            subprocess.run(args + ["--no-summary"], stdout=out, stderr=subprocess.STDOUT, check=True)
        (directory / f"{mode}_provenance.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "implementation_sha256": hashlib.sha256(
                        (ROOT.parents[1] / "tt/functional_decoder.py").read_bytes()
                    ).hexdigest(),
                    "capture_runner": "tests/run_profiles.sh",
                    "capture_cwd": str(ROOT.parents[1]),
                    "kernel_overlay": "doc/functional_decoder/profiler_kernel_overlay",
                    "kernel_overlay_source_identity_only": True,
                    "commands": [args + ["--csv", str(directory / f"{mode}_perf_report.csv")], args + ["--no-summary"]],
                    "chip": 0,
                    "regime": "serial",
                    "trace_allocation_tracking": False,
                    "watcher": False,
                    "logical_prefill_length": 1024,
                    "decode_position": 1024,
                    "warmed": True,
                    "decode_traced": True,
                },
                indent=2,
            )
            + "\n"
        )
