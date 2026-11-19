# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render signposted Tracy evidence without opening a TT device.

Keep the original CSV, filtered CSVs, human-readable tables and exact command
provenance together. XZ copies preserve large CSV evidence within the repo's
file-size gate; the uncompressed files remain available for tt-perf-report.
"""

import argparse
import csv
import hashlib
import json
import lzma
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--program-support-count", type=int, required=True)
    parser.add_argument(
        "--streaming-merge-provenance", type=Path, help="Offline streaming join identity; gaps remain unmeasured"
    )
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    args = parser.parse_args()
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "ops.csv"
    if source.resolve() != args.source_csv.resolve():
        shutil.copyfile(args.source_csv, source)
    executable = Path(sys.executable).parent / "tt-perf-report"
    provenance = {
        "source_csv": str(args.source_csv),
        "copied_csv": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "profile_json": str(args.profile_json),
        "profile": json.loads(args.profile_json.read_text()),
        "profiler_mode": "standard per-op; decode is captured/replayed",
        "program_support_count": args.program_support_count,
        "watcher": False,
        "report_arch": "blackhole",
        "commands": [],
        "compressed_csv_artifacts": [],
    }
    if args.streaming_merge_provenance:
        merge = json.loads(args.streaming_merge_provenance.read_text())
        if not merge.get("passed") or merge["output_sha256"] != provenance["sha256"]:
            raise ValueError("Streaming merge provenance does not match source CSV")
        if merge.get("profile_sha256") != hashlib.sha256(args.profile_json.read_bytes()).hexdigest():
            raise ValueError("Streaming merge provenance does not match profile JSON")
        provenance["profiler_mode"] = "continuous streaming per-op; selected single warmed pass"
        provenance["program_support_count"] = None
        provenance["streaming_merge"] = merge
        provenance["device_gaps_measured"] = False
    with source.open() as stream:
        source_rows = list(csv.DictReader(stream))
    modes = (args.mode,)
    for mode in modes:
        for marker in (f"PERF_{mode.upper()}", f"PERF_{mode.upper()}_END"):
            if sum(row["OP CODE"] == marker and row["OP TYPE"] == "signpost" for row in source_rows) != 1:
                raise ValueError(f"Missing or duplicate measurement signpost: {marker}")
    summaries = {}
    for mode in modes:
        command = [
            str(executable),
            str(source),
            "--start-signpost",
            f"PERF_{mode.upper()}",
            "--end-signpost",
            f"PERF_{mode.upper()}_END",
            "--no-advice",
            "--no-summary",
            "--no-color",
            "--arch",
            "blackhole",
        ]
        if args.streaming_merge_provenance:
            command.append("--tracing-mode")  # Preserve the proven replay/program order, not capture host timestamps.
        filtered = directory / f"{mode}_perf_report.csv"
        for extra, suffix in ((["--csv", str(filtered)], "console.log"), ([], "txt")):
            invocation = command + extra
            provenance["commands"].append(invocation)
            with (directory / f"{mode}_perf_report.{suffix}").open("w") as stream:
                subprocess.run(invocation, stdout=stream, stderr=subprocess.STDOUT, check=True)
        with filtered.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows or "Device Time [μs]" not in rows[0]:
            raise ValueError(f"Missing signposted device measurements: {filtered}")
        if args.streaming_merge_provenance and len(rows) != merge["selected_ops"]:
            raise ValueError("tt-perf-report did not retain every selected streaming operation")
        kernel_ms = sum(float(row["Device Time [μs]"] or 0) for row in rows) / 1000
        gap_ms = sum(float(row["Op-to-Op Gap [μs]"] or 0) for row in rows) / 1000
        summaries[mode] = {
            "device_kernel_sum_ms": kernel_ms,
            "op_gap_sum_ms": None if args.streaming_merge_provenance else gap_ms,
            "device_kernel_plus_gap_ms": None if args.streaming_merge_provenance else kernel_ms + gap_ms,
            "device_gaps_measured": not bool(args.streaming_merge_provenance),
            "ops": len(rows),
            "source_column": "Device Time [μs]",
            "gap_column": "Op-to-Op Gap [μs]",
            "source_file": str(filtered),
        }
    for path in (source, *(directory / f"{mode}_perf_report.csv" for mode in modes)):
        compressed = path.with_suffix(path.suffix + ".xz")
        compressed.write_bytes(lzma.compress(path.read_bytes(), preset=9))
        provenance["compressed_csv_artifacts"].append(
            {
                "path": str(compressed),
                "sha256": hashlib.sha256(compressed.read_bytes()).hexdigest(),
                "bytes": compressed.stat().st_size,
                "uncompressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (directory / "latency_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
