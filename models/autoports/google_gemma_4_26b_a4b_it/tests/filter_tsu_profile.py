# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Preserve all raw CSV fields inside one signposted reduced decode window."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("0", "1", "2", "3"))
    args = parser.parse_args()
    assert args.source.resolve() != args.output.resolve()
    selected = []
    active = False
    starts = ends = 0
    with args.source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            if row["OP TYPE"] == "signpost" and row["OP CODE"] == "PERF_DECODE":
                starts += 1
                active = True
            if active and (args.device is None or row["OP TYPE"] == "signpost" or row.get("DEVICE ID") == args.device):
                selected.append(row)
            if row["OP TYPE"] == "signpost" and row["OP CODE"] == "PERF_DECODE_END":
                ends += 1
                active = False
    assert starts == ends == 1 and not active and len(selected) > 2
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "scope": "All original columns/field values within PERF_DECODE/PERF_DECODE_END; only out-of-window rows omitted",
        "device_filter": args.device,
        "original_source": str(args.source),
        "original_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "window_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "rows_including_signposts": len(selected),
        "columns": len(fields),
    }
    if args.device is not None:
        manifest[
            "scope"
        ] = "All original columns/field values for the selected device within PERF_DECODE/PERF_DECODE_END, including both signposts"
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
