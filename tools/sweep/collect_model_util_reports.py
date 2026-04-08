#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Copy each run's model_util_report.csv into <experiment_dir>/perf_csvs/<label>.csv.

Scans for model_util_report.csv under the experiment root; label is the parent
directory name (e.g. seqlen_1k_batch_1_layers_1).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect model_util_report.csv files into perf_csvs/")
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Root directory containing one subdirectory per sweep point",
    )
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    out_dir = root / "perf_csvs"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = sorted(root.glob("*/model_util_report.csv"))
    if not reports:
        print(f"No model_util_report.csv found in direct subdirectories of {root}", file=sys.stderr)
        sys.exit(1)

    written = 0
    for src in reports:
        label = src.parent.name
        dest = out_dir / f"{label}.csv"
        if dest.exists():
            print(f"  Warning: overwriting {dest}")
        shutil.copy2(src, dest)
        written += 1
        print(f"  {src} -> {dest}")

    print(f"Done. Copied {written} file(s) to {out_dir}")


if __name__ == "__main__":
    main()
