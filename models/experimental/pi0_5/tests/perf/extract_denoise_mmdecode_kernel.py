# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Extract per-shape matmul_decode DEVICE KERNEL DURATION (ns, col 20) from a tracy
ops_perf_results CSV produced by bench_matmul_decode_denoise.py.

Per signpost region MMD_DENOISE_<label>:
  - sum DEVICE KERNEL DURATION over MatmulDecodeDeviceOperation rows
  - count those rows
  - per-call kernel us = sum / count  (region = N_ITERS forwards x 1 call/forward;
    averaging over the actual call count is robust to a stray warm-up row)

Usage:
  python extract_denoise_mmdecode_kernel.py ops_perf_results_*.csv
"""
from __future__ import annotations

import csv
import sys

MMD = "MatmulDecodeDeviceOperation"
NATIVE = "MatmulDeviceOperation"
KERNEL_COL = "DEVICE KERNEL DURATION [NS]"


def _find_col(header, name):
    up = [h.strip().upper() for h in header]
    target = name.upper()
    for i, h in enumerate(up):
        if h == target:
            return i
    for i, h in enumerate(up):
        if target in h:
            return i
    raise SystemExit(f"column {name!r} not found in {header}")


def parse(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    ci_k = _find_col(header, KERNEL_COL)
    ci_op = _find_col(header, "OP CODE")
    # region markers
    regions = []
    for idx, r in enumerate(rows[1:], start=1):
        tok = next((c for c in r if c.strip().startswith("MMD_DENOISE_")), None)
        if tok:
            regions.append((tok.strip(), idx))
    out = {}
    for ri, (name, start) in enumerate(regions):
        end = regions[ri + 1][1] if ri + 1 < len(regions) else len(rows)
        body = name[len("MMD_DENOISE_") :]
        # NATIVE regions sum MatmulDeviceOperation; matmul_decode regions sum MMD.
        want_op = NATIVE if body.startswith("NATIVE_") else MMD
        total = 0.0
        calls = 0
        for r in rows[start:end]:
            if len(r) <= max(ci_k, ci_op):
                continue
            if r[ci_op].strip() != want_op:
                continue
            try:
                total += float(r[ci_k])
            except (ValueError, IndexError):
                continue
            calls += 1
        label = body
        per_call_us = (total / 1000.0 / calls) if calls else 0.0
        out[label] = (per_call_us, calls)
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"\n=== {path} ===")
        d = parse(path)
        print(f"  {'shape':<14} {'kernel us/call':>14} {'calls':>7}")
        for label, (us, calls) in d.items():
            print(f"  {label:<14} {us:>14.3f} {calls:>7}")
