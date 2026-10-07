# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recover device-only P0 windows without importing the full host timing CSV.

Use the pinned Tracy metadata parser, preserving its cache and trace semantics.
Only its host-zone input is empty: this report never claims host time or TPOT.
Device durations come verbatim from the native compact device CSV.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.layer_profile_report import SIGNPOST, analyze, write_report


def join_windows(ops, signposts, device_rows):
    markers = []
    intervals = []
    start = None
    for post in sorted(signposts.values(), key=lambda post: int(post["tracy_time"])):
        label = post["data"].split(": ")[-1].split("\n")[0]
        match = SIGNPOST.fullmatch(label)
        if match is None:
            continue
        timestamp = int(post["tracy_time"])
        markers.append((timestamp, {"OP TYPE": "signpost", "OP CODE": label}))
        if match.group(1).endswith("_MODEL"):
            if match.group(4) == "BEGIN":
                if start is not None:
                    raise ValueError("Overlapping model windows")
                start = timestamp
            else:
                if start is None or timestamp <= start:
                    raise ValueError("Invalid model window end")
                intervals.append((start, timestamp))
                start = None
    if start is not None or not intervals:
        raise ValueError("Incomplete model windows")

    selected = {
        int(op_id): op
        for op_id, op in ops.items()
        if any(begin < int(op["tracy_time"]) < end for begin, end in intervals)
    }
    for op in selected.values():
        if op.get("metal_trace_id") is not None:
            raise ValueError("Trace operation inside eager diagnostic window")
    timings = {}
    for row in device_rows:
        op_id = int(row["GLOBAL CALL COUNT"])
        if op_id not in selected:
            continue
        device = int(row["DEVICE ID"])
        if int(selected[op_id].get("device_id", -1)) != device:
            raise ValueError("Device metadata mismatch")
        if row.get("METAL TRACE ID") not in (None, "", "-"):
            raise ValueError("Trace timing inside eager diagnostic window")
        if op_id in timings:
            raise ValueError("Duplicate compact device timing")
        timings[op_id] = dict(row)

    events = markers[:]
    for op_id, op in selected.items():
        # Preserve an absent timing as a device row: analyze() must count it as
        # missing, never silently drop the operation or substitute zero time.
        row = timings.get(op_id, {}).copy()
        row.update(
            {
                "GLOBAL CALL COUNT": str(op_id),
                "OP CODE": op["op_code"],
                "OP TYPE": op["op_type"],
                "DEVICE ID": str(op["device_id"]) if "device_id" in op else "",
            }
        )
        events.append((int(op["tracy_time"]), row))
    return [row for _, row in sorted(events, key=lambda event: event[0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logs = args.source / "tracy/.logs"
    metadata = logs / "tracy_ops_data.csv"
    compact = logs / "cpp_device_perf_report.csv"
    receipt = args.source / "profile.json"
    staging = args.output / "metadata-only"
    staging.mkdir()
    (staging / metadata.name).symlink_to(metadata.resolve())
    (staging / "tracy_ops_times.csv").write_text(
        "name,src_file,src_line,zone_name,zone_text,ns_since_start,exec_time_ns,thread,special_parent_text\n"
    )

    from tracy.process_ops_logs import import_tracy_op_logs

    ops, signposts, _ = import_tracy_op_logs(staging)
    with compact.open(newline="") as stream:
        rows = join_windows(ops, signposts, csv.DictReader(stream))
    report = analyze(rows, json.loads(receipt.read_text()))
    report["recovery_method"] = (
        "Pinned native Tracy metadata parser plus compact native device timings; "
        "only P0 signpost windows selected. Full host-zone timing CSV was not read. "
        "No host-time, dispatch-overhead or full-model TPOT conclusion is supported."
    )
    report["sources"] = {
        path.name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (metadata, compact, receipt)
    }
    write_report(report, args.output / "analysis")
    with (args.output / "diagnostic-ops.csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(measurements_complete=report["measurements_complete"], rows=len(rows))), flush=True)
    if not report["measurements_complete"]:
        raise RuntimeError("Diagnostic windows have missing device timings")


if __name__ == "__main__":
    main()
