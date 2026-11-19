# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused-stage official whole-trace reports, retaining strict selection and advice.

The measured row is a complete trace interval including internal idle gaps.
It cannot provide individual op bandwidth/utilization; use per-op reports for that.
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

from models.autoports.qwen_qwen3_8_flash_next.tests.report_trace_performance import (
    analyze_device,
    select_device_replay,
    select_host_replay,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-csv", type=Path, required=True)
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--host-metadata-json", type=Path)
    authority.add_argument("--device-protocol", action="store_true")
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--capture-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile_json.read_text())
    if args.device_protocol:
        proof = select_device_replay(profile, args.capture_log.read_text())
        if profile.get("native_csv", {}).get("sha256") != hashlib.sha256(args.device_csv.read_bytes()).hexdigest():
            raise ValueError("Native CSV differs from the file bound after device close")
    else:
        metadata = json.loads(args.host_metadata_json.read_text())
        proof = select_host_replay(profile, metadata, args.capture_log.read_text())
    analysis = analyze_device(args.device_csv, proof)
    mode = profile["mode"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw_device.csv"
    if args.device_csv.resolve() != raw.resolve():
        shutil.copyfile(args.device_csv, raw)
    row = {
        "OP TYPE": "tt_dnn_device",
        "OP CODE": f"WHOLE_TRACE_{mode.upper()}_AGGREGATE",
        "GLOBAL CALL COUNT": (2 << 10) | proof["device"],
        "DEVICE ID": proof["device"],
        "CORE COUNT": len(analysis["cores"]),
        "DEVICE ARCH": "blackhole",
        "AVAILABLE WORKER CORE COUNT": analysis["device_info"]["max_compute_cores"],
        "DEVICE KERNEL START CYCLE": analysis["trace_kernel"]["start_cycle"],
        "DEVICE KERNEL END CYCLE": analysis["trace_kernel"]["end_cycle"],
        "DEVICE KERNEL DURATION [ns]": analysis["trace_kernel"]["duration_ns"],
        "DEVICE FW DURATION [ns]": analysis["trace_fw"]["duration_ns"],
        "OP TO OP LATENCY [ns]": "",
        "INPUT_0_MEMORY": "",
        "ATTRIBUTES": f"Complete captured decoder pass; aggregate elapsed includes internal gaps; no per-op timing/details; marker authority={proof['metadata_authority']}",
        "HOST START TS": proof["host_replay_timestamps"][1],
        "HOST END TS": "",
        "HOST DURATION [ns]": "",
    }
    rows = [
        {
            "OP TYPE": "signpost",
            "OP CODE": f"PERF_{mode.upper()}",
            "HOST START TS": proof["markers"][f"PERF_{mode.upper()}"],
            "ATTRIBUTES": f"Actual measurement marker from {proof['metadata_authority']}; host timestamp is ordering only",
        },
        row,
        {
            "OP TYPE": "signpost",
            "OP CODE": f"PERF_{mode.upper()}_END",
            "HOST START TS": proof["markers"][f"PERF_{mode.upper()}_END"],
            "ATTRIBUTES": f"Actual measurement marker from {proof['metadata_authority']}; host timestamp is ordering only",
        },
    ]
    csv_path = output / "whole_trace_ops.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    command = [
        str(Path(sys.executable).parent / "tt-perf-report"),
        str(csv_path),
        "--start-signpost",
        f"PERF_{mode.upper()}",
        "--end-signpost",
        f"PERF_{mode.upper()}_END",
        "--tracing-mode",
        "--no-color",
        "--arch",
        "blackhole",
    ]
    filtered = output / f"{mode}_perf_report.csv"
    commands = []
    for extra, suffix in ((["--csv", str(filtered)], "console.log"), ([], "txt")):
        invocation = command + extra
        commands.append(invocation)
        with (output / f"{mode}_perf_report.{suffix}").open("w") as stream:
            subprocess.run(invocation, stdout=stream, stderr=subprocess.STDOUT, check=True)
    with filtered.open() as stream:
        rendered = list(csv.DictReader(stream))
    if len(rendered) != 1 or "WHOLE_TRACE_" not in str(rendered[0]) or not rendered[0].get("Device Time [μs]"):
        raise ValueError("tt-perf-report did not preserve exactly one measured aggregate row")
    reported_us = float(rendered[0]["Device Time [μs]"])
    if abs(reported_us - analysis["trace_kernel"]["duration_ns"] / 1000) > 1:
        raise ValueError("Rendered device time differs from official aggregate")
    summary = {
        "mode": mode,
        "measurement": "single warmed whole-trace replay",
        "metadata_authority": proof["metadata_authority"],
        "device_trace_elapsed_ms": analysis["trace_kernel"]["duration_ns"] / 1e6,
        "device_trace_firmware_elapsed_ms": analysis["trace_fw"]["duration_ns"] / 1e6,
        "device_kernel_sum_ms": None,
        "op_gap_sum_ms": None,
        "per_op_timings_available": False,
        "internal_gaps_included_in_trace_elapsed": True,
        "captured_operation_count": proof["captured_operation_count"],
        "aggregate_rows": 1,
        "report_column": "Device Time [μs] (whole-trace elapsed, not a per-op kernel sum)",
    }
    provenance = {
        "scope": __doc__,
        "advice_enabled": True,
        "validation_source_sha256": hashlib.sha256(
            Path(__import__(analyze_device.__module__, fromlist=["__file__"]).__file__).read_bytes()
        ).hexdigest(),
        "selection": proof,
        "analysis": analysis,
        "profile": profile,
        "commands": commands,
        "sources": {},
        "archives": [],
        "passed": True,
    }
    source_paths = [args.device_csv, args.profile_json, args.capture_log, csv_path]
    if args.host_metadata_json is not None:
        source_paths.append(args.host_metadata_json)
    for path in source_paths:
        provenance["sources"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (raw, csv_path, filtered):
        packed = path.with_suffix(path.suffix + ".xz")
        packed.write_bytes(lzma.compress(path.read_bytes(), preset=6, check=lzma.CHECK_CRC64))
        provenance["archives"].append({"path": str(packed), "sha256": hashlib.sha256(packed.read_bytes()).hexdigest()})
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "latency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
