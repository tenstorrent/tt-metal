#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pick smallest CB at grid peak BW ('s shrink recipe).

From cb_grid BUFFER_TUNE log lines:
  1. Find peak dram_pipeline_gbps across all (input, output) pairs.
  2. Keep pairs within tolerance_pct of peak.
  3. Pick smallest input_cb_depth with any qualifying output.
  4. For that input, pick smallest output_cb_depth still at peak.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class GridRow:
    input_cb: int
    output_cb: int
    gbps: float


ROW_RE = re.compile(
    r"BUFFER_TUNE,phase=cb_grid,input_cb_depth=(\d+),output_cb_depth=(\d+)," r".*dram_pipeline_gbps=([0-9.]+)"
)


def parse_grid_rows(text: str) -> list[GridRow]:
    rows: list[GridRow] = []
    for line in text.splitlines():
        m = ROW_RE.search(line)
        if m:
            rows.append(GridRow(int(m.group(1)), int(m.group(2)), float(m.group(3))))
    return rows


def pick_min_cb_at_peak(rows: list[GridRow], tolerance_pct: float) -> tuple[float, int, int]:
    if not rows:
        raise ValueError("no cb_grid rows found")

    peak = max(r.gbps for r in rows)
    threshold = peak * (1.0 - tolerance_pct / 100.0)
    qual = [r for r in rows if r.gbps >= threshold]
    if not qual:
        raise ValueError("no rows within tolerance of peak")

    best_in = min(r.input_cb for r in qual)
    outs = [r for r in qual if r.input_cb == best_in]
    best_out = min(r.output_cb for r in outs)
    return peak, best_in, best_out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_file", type=argparse.FileType("r"), help="grid_tune.log from buffer-tune grid")
    p.add_argument("--tolerance-pct", type=float, default=2.0)
    p.add_argument("--min-input-cb", type=int, default=0, help="floor for mode2 reader (e.g. 2*N)")
    args = p.parse_args()

    rows = parse_grid_rows(args.log_file.read())
    peak, in_cb, out_cb = pick_min_cb_at_peak(rows, args.tolerance_pct)
    if args.min_input_cb and in_cb < args.min_input_cb:
        in_cb = args.min_input_cb

    # peak,in,out — easy for bash
    print(f"{peak:.4f} {in_cb} {out_cb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
