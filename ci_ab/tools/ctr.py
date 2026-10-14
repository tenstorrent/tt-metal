#!/usr/bin/env python3
"""Counter columns of the ops report for the long ops."""
import csv, glob, os, sys
out, arm = sys.argv[1:3]
fs = sorted(glob.glob(os.path.join(out, "**", "ops_perf_results*.csv"), recursive=True))
if not fs:
    print(f"NOREPORT {arm}"); sys.exit(0)
rows = list(csv.DictReader(open(fs[-1])))
keys = ("BUSY", "Busy", "busy", "Util", "util", "(%)", "Stall", "stall", "STALL", "FPU", "UNPACK", "Unpack", "PACK", "Pack", "MATH", "Math", "INSTRN", "Instrn", "THREAD", "SRC", "Src")
i = 0
for r in rows:
    try:
        t = int(float(r.get("DEVICE KERNEL DURATION [ns]", "0") or 0))
    except ValueError:
        continue
    if t < 3000:
        continue
    vals = [f"{k}={v}" for k, v in r.items() if k and v not in ("", None) and any(s in k for s in keys)]
    print(f"CTR {arm} op{i} {t} ns | " + " | ".join(vals))
    i += 1
