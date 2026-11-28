"""Summarize a test_eager_calls profiler CSV: median kernel / FW ns per group of EAGER_CALLS, dropping 2 warm calls.

Usage: python perf_csv_summary.py <csv> <group label> [<group label> ...]   (labels in test execution order)
"""
import csv
import statistics
import sys

path, labels = sys.argv[1], sys.argv[2:]
with open(path) as f:
    # A device-only report (no host op columns) lists the same programs in call order.
    rows = [r for r in csv.DictReader(f) if r.get("OP CODE", "GenericOp").startswith("GenericOp")]
calls = 12
assert len(rows) == calls * len(labels), f"{len(rows)} rows for {len(labels)} groups"
for i, label in enumerate(labels):
    group = rows[i * calls + 2 : (i + 1) * calls]
    k = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in group]
    fw = [float(r["DEVICE FW DURATION [ns]"]) for r in group]
    print(
        f"{label:24s} cores={group[0].get('CORE COUNT', '?'):>4s} kernel med={statistics.median(k):7.1f} "
        f"min={min(k):6.0f} max={max(k):6.0f}  fw med={statistics.median(fw):7.1f}"
    )
