# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-window timing for the reduced full-path profile, including device gaps."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope",
        default="Reduced real layers0/5 plus embedding, final norm, LM head, split sampler and recorder; NOT all-layer model",
    )
    parser.add_argument(
        "--phases", nargs="+", choices=("PERF_PREFILL", "PERF_DECODE"), default=["PERF_PREFILL", "PERF_DECODE"]
    )
    args = parser.parse_args()
    windows = defaultdict(list)
    phase = None
    host_starts, host_windows = {}, {}
    with args.csv.open() as stream:
        for row in csv.DictReader(stream):
            if row["OP TYPE"] == "signpost":
                code = row["OP CODE"]
                if code in ("PERF_PREFILL", "PERF_DECODE"):
                    phase = code
                    host_starts[phase] = float(row["HOST START TS"])
                elif code.endswith("_END"):
                    if phase:
                        host_windows[phase] = (float(row["HOST START TS"]) - host_starts[phase]) / 1e6
                    phase = None
            elif phase and row.get("DEVICE FW START CYCLE"):
                windows[phase].append(row)
    report = {
        "scope": args.scope,
        "source": str(args.csv),
        "sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "basis": "Maximum per-device first firmware start to final firmware end, including all internal gaps; per-device cycles/ns inferred from firmware durations",
        "windows": {},
    }
    for phase, rows in windows.items():
        devices = defaultdict(list)
        for row in rows:
            devices[row["DEVICE ID"]].append(row)
        result = {}
        for device, ops in devices.items():
            start = lambda row: float(row["DEVICE FW START CYCLE"])
            end = lambda row: float(row["DEVICE FW END CYCLE"])
            ratios = [
                (end(row) - start(row)) / float(row["DEVICE FW DURATION [ns]"])
                for row in ops
                if float(row["DEVICE FW DURATION [ns]"]) > 0
            ]
            clock = statistics.median(ratios)
            traces = defaultdict(list)
            for row in ops:
                if row.get("METAL TRACE ID"):
                    traces[row["METAL TRACE ID"]].append(row)
            result[device] = {
                "whole_window_us": (max(map(end, ops)) - min(map(start, ops))) / clock / 1000,
                "cycles_per_ns": clock,
                "traces": {
                    key: {
                        "duration_us": (max(map(end, value)) - min(map(start, value))) / clock / 1000,
                        "first_op": value[0]["OP CODE"],
                        "last_op": value[-1]["OP CODE"],
                        "op_count": len(value),
                    }
                    for key, value in traces.items()
                },
            }
        report["windows"][phase] = {
            "devices": result,
            "max_device_window_us": max(x["whole_window_us"] for x in result.values()),
            "host_signpost_window_ms": host_windows[phase],
        }
    assert set(report["windows"]) == set(args.phases)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
