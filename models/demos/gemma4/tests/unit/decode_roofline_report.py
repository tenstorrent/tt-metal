# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Split a Tracy ops CSV of signposted decode steps into a roofline table.

    python decode_roofline_report.py ops_perf_results_<date>.csv --steps 10 \\
        --params-b 31 --chips 4 --dram-gbps 512 [--bytes-per-param 1.0625]

Reads the rows between the ``start`` and ``stop`` signposts (device 0 only,
so one chip's view of the tensor-parallel step), sums device firmware time
per category, and reports per-step: op count, matmul time, collective time,
other-op time, op-to-op gaps, and the DRAM-bandwidth floor
(params * bytes_per_param / chips / bandwidth).
"""

import argparse
import csv
from collections import Counter, defaultdict

CCL = ("AllGather", "ReduceScatter", "AllReduce", "all_gather", "reduce_scatter", "all_reduce", "Ccl", "CCL")
MATMUL = ("Matmul", "matmul", "LinearOperation")


def category(op_code):
    if any(k in op_code for k in CCL):
        return "ccl"
    if any(k in op_code for k in MATMUL):
        return "matmul"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--params-b", type=float, required=True, help="parameters in billions")
    ap.add_argument("--chips", type=int, required=True, help="tensor-parallel chips")
    ap.add_argument("--dram-gbps", type=float, default=512.0)
    ap.add_argument("--bytes-per-param", type=float, default=1.0625)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    cols = rows[0].keys()
    code_col = next(c for c in cols if c.strip() == "OP CODE")
    dur_col = next(c for c in cols if "DEVICE FW DURATION" in c)
    gap_col = next((c for c in cols if "OP TO OP LATENCY" in c), None)
    dev_col = next((c for c in cols if c.strip() == "DEVICE ID"), None)

    # Keep rows between the signposts (the signpost rows carry the name in OP CODE).
    inside, kept = False, []
    for r in rows:
        code = r[code_col]
        if "signpost" in code.lower() and "start" in code.lower():
            inside = True
            continue
        if "signpost" in code.lower() and "stop" in code.lower():
            inside = False
            continue
        if inside and (dev_col is None or str(r[dev_col]).strip() == str(args.device)):
            kept.append(r)
    if not kept:  # no signposts: take everything on the device
        kept = [r for r in rows if dev_col is None or str(r[dev_col]).strip() == str(args.device)]

    dur = defaultdict(float)
    counts = Counter()
    gaps = 0.0
    per_op = defaultdict(float)
    for r in kept:
        d = float(r[dur_col] or 0) / 1e6  # ns -> ms
        c = category(r[code_col])
        dur[c] += d
        counts[c] += 1
        per_op[r[code_col]] += d
        if gap_col and r[gap_col]:
            gaps += max(0.0, float(r[gap_col]) - float(r[dur_col] or 0)) / 1e6

    n = args.steps
    total_ops = sum(counts.values())
    device_busy = sum(dur.values())
    floor_ms = args.params_b * 1e9 * args.bytes_per_param / args.chips / (args.dram_gbps * 1e9) * 1e3
    print(f"rows between signposts (device {args.device}): {len(kept)}  steps: {n}")
    print(f"ops/step: {total_ops / n:.0f}   matmul {counts['matmul'] / n:.0f}  ccl {counts['ccl'] / n:.0f}  other {counts['other'] / n:.0f}")
    print(f"device time/step: {device_busy / n:.2f} ms  = matmul {dur['matmul'] / n:.2f} + ccl {dur['ccl'] / n:.2f} + other {dur['other'] / n:.2f}")
    if gap_col:
        print(f"op-to-op gaps/step: {gaps / n:.2f} ms   (busy + gaps = {(device_busy + gaps) / n:.2f} ms)")
    print(f"DRAM floor/step: {floor_ms:.2f} ms  -> matmul at {floor_ms / (dur['matmul'] / n) * 100:.0f}% of floor, "
          f"whole step at {floor_ms / ((device_busy + gaps) / n) * 100:.0f}% of floor")
    print("\ntop ops by device time per step:")
    for code, d in sorted(per_op.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {d / n:7.3f} ms  {code}")


if __name__ == "__main__":
    main()
