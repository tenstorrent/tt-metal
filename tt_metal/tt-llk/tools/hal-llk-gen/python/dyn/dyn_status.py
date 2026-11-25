#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Completion summary of a dynamic analysis, without instruction counts.

Per kernel and TRISC: status records, completed variants and variants with any error, from
<dyn dir>/status_<kernel>_<core>.jsonl. With --reference <dir> (direct stepping of the original kernel,
files full_<core>.jsonl), also how many original variants agree exactly with it; only that number is printed.

Usage: dyn_status.py <dyn dir> [--kernels original hal] [--reference DIR]
"""
import argparse
import json
from pathlib import Path

ERRORS = [
    "flow_errors",
    "unknown_pc",
    "unparsed_store",
    "time_limit",
    "lost_halt",
    "client_error_type",
]
FIELDS = ["steps", "tti", "tt", "sfp", "other_store"]
CORES = ["unpack", "math", "pack"]


def records(path):
    return [json.loads(line) for line in path.open()] if path.is_file() else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dyn_dir")
    ap.add_argument("--kernels", nargs="+", default=["original", "hal"])
    ap.add_argument("--reference")
    a = ap.parse_args()
    d = Path(a.dyn_dir)
    for kernel in a.kernels:
        for core in CORES:
            status = records(d / f"status_{kernel}_{core}.jsonl")
            completed = sum(1 for r in status if r.get("completed"))
            with_errors = sum(1 for r in status if any(r.get(k) for k in ERRORS))
            line = f"{kernel} {core}: {len(status)} variants, {completed} completed, {with_errors} with errors"
            if kernel == "original" and a.reference:
                reference = {
                    r["nodeid"]: r
                    for r in records(Path(a.reference) / f"full_{core}.jsonl")
                }
                measured = records(d / f"original_{core}.jsonl")
                agree = sum(
                    1
                    for r in measured
                    if r["nodeid"] in reference
                    and all(
                        r.get(f, 0) == reference[r["nodeid"]].get(f, 0) for f in FIELDS
                    )
                )
                line += (
                    f", {agree} of {len(measured)} agree with the reference stepping"
                )
            print(line)


if __name__ == "__main__":
    main()
