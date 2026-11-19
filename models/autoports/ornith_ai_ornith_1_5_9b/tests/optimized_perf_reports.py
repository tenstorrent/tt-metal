# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Retain raw Tracy CSV and render the two warmed signposted windows."""
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
ROOT = MODEL_ROOT / "doc/optimized_decoder"
if len(sys.argv) > 2 and sys.argv[1] == "--root":
    ROOT = Path(sys.argv[2]).resolve()
    del sys.argv[1:3]
for kind in sys.argv[1:] or ["linear", "full"]:
    directory = ROOT / "tracy_final" / kind
    measured = json.loads((ROOT / f"{kind}_profile_final.json").read_text())
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
                    "implementation_sha256": measured["implementation_sha256"],
                    "policy": measured.get("policy", {}),
                    "experiment_sources": measured.get("experiment_sources", {}),
                    "capture_runner": "tests/run_optimized_profiles.sh",
                    "capture_cwd": str(MODEL_ROOT),
                    "kernel_overlay": "doc/optimized_decoder/profiler_kernel_overlay",
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
