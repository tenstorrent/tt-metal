"""Preserve a Tracy ops CSV and render both mandatory measured windows."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def validate_windows(source, decode_replays):
    """Reject incomplete captures before tt-perf-report's permissive filtering."""
    if decode_replays <= 0:
        raise ValueError("decode-replays must be positive")
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    windows = {}
    for mode in ("prefill", "decode"):
        markers = (f"PERF_{mode.upper()}", f"PERF_{mode.upper()}_END")
        indices = [
            [i for i, row in enumerate(rows) if row.get("OP TYPE") == "signpost" and row.get("OP CODE") == marker]
            for marker in markers
        ]
        if any(len(found) != 1 for found in indices):
            raise ValueError(f"Expected exactly one of each {markers}; found {indices}")
        start, end = indices[0][0], indices[1][0]
        if start >= end:
            raise ValueError(f"Reversed {mode} signpost window")
        device_rows = [row for row in rows[start + 1 : end] if row.get("OP TYPE") == "tt_dnn_device"]
        devices = {row.get("DEVICE ID", "").strip() for row in device_rows}
        if not device_rows or len(devices) != 1 or "" in devices:
            raise ValueError(f"Expected single-device operations inside {mode} window; got {devices}")
        evidence = {
            "start_csv_line": start + 2,
            "end_csv_line": end + 2,
            "raw_device_operation_rows": len(device_rows),
            "device_ids": sorted(devices),
        }
        if mode == "decode":
            sessions = {}
            for row in device_rows:
                trace_id = row.get("METAL TRACE ID", "").strip()
                session_id = row.get("METAL TRACE REPLAY SESSION ID", "").strip()
                if not trace_id or not session_id:
                    raise ValueError("Decode window contains device operations without trace/replay IDs")
                sessions.setdefault((trace_id, session_id), []).append(row.get("OP CODE"))
            if len(sessions) != decode_replays or len({key[0] for key in sessions}) != 1:
                raise ValueError(f"Expected {decode_replays} replays of one trace, got {list(sessions)}")
            sequences = list(sessions.values())
            if any(sequence != sequences[0] for sequence in sequences[1:]):
                raise ValueError("Decode replay operation sequences differ or are incomplete")
            evidence["trace_replay_sessions"] = [list(key) for key in sessions]
            evidence["operations_per_replay"] = len(sequences[0])
        windows[mode] = evidence
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ops_csv", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--decode-replays", type=int, default=10)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-decode-programs", type=int)
    args = parser.parse_args()
    window_evidence = validate_windows(args.ops_csv, args.decode_replays)
    if args.expected_decode_programs is not None:
        assert window_evidence["decode"]["operations_per_replay"] == args.expected_decode_programs
    root = Path(__file__).resolve().parents[1]
    dest = root / "doc/optimized_decoder/tracy" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    source = dest / "ops.csv"
    if args.ops_csv.resolve() != source.resolve():
        shutil.copyfile(args.ops_csv, source)
    executable = str(Path(sys.executable).with_name("tt-perf-report"))
    results = {}
    for mode, repetitions in (("prefill", 1), ("decode", args.decode_replays)):
        base = [
            executable,
            str(source),
            "--start-signpost",
            f"PERF_{mode.upper()}",
            "--end-signpost",
            f"PERF_{mode.upper()}_END",
            "--no-color",
        ]
        if mode == "decode":
            base.append("--tracing-mode")
        report = dest / f"{mode}_perf_report.csv"
        commands = [base + ["--csv", str(report)], base + ["--no-summary"]]
        for command, suffix in zip(commands, ("console.log", "txt")):
            with (dest / f"{mode}_perf_report.{suffix}").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        with report.open() as handle:
            rows = list(csv.DictReader(handle))
        # Version1.4.1's filtered report exposes microseconds. Do not silently
        # fall back to raw device durations with a different unit/window.
        column = "Device Time"
        if not rows or column not in rows[0]:
            raise RuntimeError(f"Inspect report time units/columns before summing: {report}")
        durations = []
        gaps = []
        for row in rows:
            value = row[column].replace(",", "").strip()
            if value and value not in ("-", "N/A"):
                durations.append(float(value))
            gap = row.get("Op-to-Op Gap", "").replace(",", "").strip()
            if gap and gap not in ("-", "N/A"):
                gaps.append(float(gap))
        if len(durations) != window_evidence[mode]["raw_device_operation_rows"]:
            raise RuntimeError(f"Missing device durations in {report}")
        if not durations or sum(durations) <= 0 or not all(math.isfinite(x) and x >= 0 for x in durations + gaps):
            raise RuntimeError(f"Missing or invalid measured device durations in {report}")
        results[mode] = {
            "commands": commands,
            "report": str(report.relative_to(root)),
            "window_validation": window_evidence[mode],
            "time_column": column,
            "time_unit": "microseconds",
            "device_operation_rows": len(durations),
            "repetitions": repetitions,
            "summed_device_us": sum(durations),
            "per_pass_device_us": sum(durations) / repetitions,
            "summed_op_gap_us": sum(gaps),
            "per_pass_device_plus_gap_us": (sum(durations) + sum(gaps)) / repetitions,
        }
    result = {
        "layer": args.layer,
        "expected_decode_programs": args.expected_decode_programs,
        "source_csv": str(args.ops_csv.resolve()),
        "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "report_tool_version": version("tt-perf-report"),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "windows": results,
    }
    (dest / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
