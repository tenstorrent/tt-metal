# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inspect native whole-layer captures without opening TT devices."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
source = next(root.glob("reports/tp4/*/ops_perf_results*.csv"))
phase = None
sessions = defaultdict(lambda: defaultdict(list))
native = []
first = None
invalid = []
for row in csv.DictReader(source.open()):
    op = row["OP CODE"]
    if op in ("PERF_PREFILL", "PERF_DECODE"):
        phase = op[5:].lower()
    elif op in ("PERF_PREFILL_END", "PERF_DECODE_END"):
        phase = None
    elif phase and row["OP TYPE"] == "tt_dnn_device":
        device = row["DEVICE ID"]
        session = row["METAL TRACE REPLAY SESSION ID"] if phase == "decode" else "prefill"
        sessions[(phase, device)][session].append(op)
        start, end = float(row["DEVICE FW START CYCLE"]), float(row["DEVICE FW END CYCLE"])
        if end < start:
            invalid.append(dict(device=device, session=session, op=op))
        if phase == "decode" and device == "0":
            if first is None:
                first = session
            if session == first:
                native.append(
                    {
                        k: v
                        for k, v in row.items()
                        if k
                        in (
                            "OP CODE",
                            "GLOBAL CALL COUNT",
                            "MATH FIDELITY",
                            "CORE COUNT",
                            "ATTRIBUTES",
                            "DEVICE KERNEL DURATION [ns]",
                            "DEVICE FW DURATION [ns]",
                        )
                        or k.startswith(("INPUT_", "OUTPUT_"))
                    }
                )
summary = {}
for device in sorted({key[1] for key in sessions}):
    replays = sessions["decode", device]
    values = list(replays.values())
    summary[device] = dict(
        prefill_programs=len(sessions["prefill", device]["prefill"]),
        decode_replay_sessions=len(values),
        programs_per_decode_replay=len(values[0]),
        same_op_set_all_replays=all(value == values[0] for value in values),
    )
passed = (
    len(summary) == 4
    and not invalid
    and all(item["decode_replay_sessions"] == 128 and item["same_op_set_all_replays"] for item in summary.values())
)
(root / "capture_integrity.json").write_text(
    json.dumps(dict(source=str(source), per_device=summary, invalid_fw_spans=invalid, passed=passed), indent=2) + "\n"
)
(root / "native_decode_policy_rows.json").write_text(
    json.dumps(
        dict(
            source=str(source),
            device=0,
            scope="First traced replay; all128 replay op sequences checked on all4 devices",
            rows=native,
        ),
        indent=2,
    )
    + "\n"
)
assert passed
print("Native capture integrity passed:", summary)
