#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Append one row to runs.csv from a run's logs/<id>.env (SHA, dirty files, full env).

  append_runs.py <run_id> --block P0A2 --start 2026-..Z --end 2026-..Z --status OK --cmd "..." [--notes "..."]
"""

import argparse
import csv
from pathlib import Path

RES = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--block", default="P0A2")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    kv = {}
    for line in open(RES / "logs" / f"{a.run_id}.env"):
        k, _, v = line.rstrip("\n").partition("=")
        kv.setdefault(k, v)
    sha = kv.pop("git_sha", "?")
    dirty = kv.pop("dirty", "").strip()
    for k in ("run_id", "date"):
        kv.pop(k, None)
    if dirty and not dirty.isdigit():
        files = [f for f in dirty.split() if f != "M"]
        sha += f" dirty({len(files)}: {' '.join(files)})"
    elif dirty and dirty != "0":
        sha += f" dirty({dirty})"
    env = " ".join(f"{k}={v}" for k, v in sorted(kv.items()))
    log = f"m3_budget_study/results_ops/logs/{a.run_id}.log"
    with open(RES / "runs.csv", "a", newline="") as f:
        csv.writer(f).writerow([a.run_id, a.block, a.start, a.end, sha, env, a.cmd, a.status, log, a.notes])
    print(f"appended {a.run_id}")


if __name__ == "__main__":
    main()
