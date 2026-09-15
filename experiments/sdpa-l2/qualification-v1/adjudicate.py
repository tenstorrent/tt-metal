# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Honor the agreed undefined-PCC exception; preserve raw measurements/scores.

This does not waive defined PCC failures, change thresholds, or alter L2,
row-tail, common-mode, masking, trace, or structural checks.
"""

import argparse
import json
from pathlib import Path


def adjudicate(row):
    row = json.loads(json.dumps(row))
    row["raw_status"] = row["status"]
    if "failed_gates" not in row:
        return row
    row["raw_failed_gates"] = row["failed_gates"][:]
    row["failed_gates"] = [f for f in row["failed_gates"] if f.split(":")[-1] != "pcc_undefined_actual_constant"]
    for field in ("per_head", "common_k_centered_per_head"):
        for head in row.get(field, []):
            if head["pcc"] is None:
                head["pcc_status"] = "NOT_APPLICABLE_CONSTANT_OUTPUT_OR_REFERENCE"
            head["failed_gates"] = [f for f in head["failed_gates"] if f != "pcc_undefined_actual_constant"]
    if row["status"] in ("PASS", "FAIL"):
        row["status"] = "FAIL" if row["failed_gates"] else "PASS"
    row["scoring_revision"] = "agreed_constant_output_PCC_exception"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rows = {(r["id"], r["mode"]): r for r in map(json.loads, args.source.read_text().splitlines())}
    changed = 0
    with args.destination.open("x") as output:
        for row in rows.values():
            scored = adjudicate(row)
            changed += scored["status"] != row["status"]
            output.write(json.dumps(scored, allow_nan=False) + "\n")
    print(json.dumps(dict(rows=len(rows), changed_status=changed, output=str(args.destination))))


if __name__ == "__main__":
    main()
