# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete profiler captures and retain dtype/fidelity evidence."""

import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("csv", type=Path)
ap.add_argument("log", type=Path)
ap.add_argument("output", type=Path)
a = ap.parse_args()
with a.csv.open() as f:
    rows = list(csv.DictReader(f))
groups = collections.defaultdict(list)
for row in rows:
    if row["DEVICE ID"] and row["CORE COUNT"]:
        # EncodePerDeviceProgramID reserves ten low bits for device ID.
        # tt_metal/impl/profiler/tt_metal_profiler.cpp: DEVICE_ID_NUM_BITS.
        encoded = int(row["GLOBAL CALL COUNT"])
        assert encoded & 1023 == int(row["DEVICE ID"]), row
        groups[(encoded >> 10, row["METAL TRACE ID"], row["METAL TRACE REPLAY SESSION ID"])].append(row)
bad = []
for key, group in groups.items():
    if (
        len(group) != 4
        or {r["DEVICE ID"] for r in group} != {"0", "1", "2", "3"}
        or len({r["CORE COUNT"] for r in group}) != 1
        or len({r["OP CODE"] for r in group}) != 1
    ):
        bad.append(key)
log = a.log.read_text()
# Commit size limits may replace a verbose log with a lossless archive pointer.
# Inspect the full original for drop markers, not the shortened excerpt.
if log.startswith("Archived full log:"):
    log = gzip.decompress(a.log.with_name(a.log.name + ".gz").read_bytes()).decode()
drops = [
    line
    for line in log.splitlines()
    if "dropped" in line.lower() and ("marker" in line.lower() or "profiler" in line.lower())
]
policy = collections.Counter(
    (r["OP CODE"], r["MATH FIDELITY"], r["INPUT_0_DATATYPE"], r["INPUT_1_DATATYPE"], r["OUTPUT_0_DATATYPE"])
    for r in rows
    if "Matmul" in r["OP CODE"]
)
result = dict(
    status="pass" if not bad and not drops else "fail",
    grouping="GLOBAL CALL COUNT >> 10, trace ID, replay session; low10 bits verified as device ID",
    raw_csv=str(a.csv),
    raw_sha256=hashlib.sha256(a.csv.read_bytes()).hexdigest(),
    rows=len(rows),
    operation_groups=len(groups),
    asymmetric_core_coverage=bad,
    dropped_marker_warnings=drops,
    matmul_policy_counts=[
        dict(op=k[0], fidelity=k[1], input0=k[2], input1=k[3], output=k[4], rows=v) for k, v in policy.items()
    ],
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
assert result["status"] == "pass", f"{len(bad)} invalid groups, {len(drops)} drop warnings; see {a.output}"
print("PASS", len(groups), "complete four-device operation groups")
