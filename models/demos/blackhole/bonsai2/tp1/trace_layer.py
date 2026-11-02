#!/usr/bin/env python3
"""Print the op sequence (kernel us, gap us, cores) of one traced decode step from a device-side ops CSV,
between two occurrences of an anchor op. Usage: trace_layer.py <csv> <anchor substring> [nth] [session]"""
import collections, csv, sys
path, anchor = sys.argv[1], sys.argv[2]
nth = int(sys.argv[3]) if len(sys.argv) > 3 else 1
which = int(sys.argv[4]) if len(sys.argv) > 4 else 1
f = lambda r, k: float(r[k]) if (r.get(k) or "").strip() not in ("", "-") else 0.0
rows = [r for r in csv.DictReader(open(path)) if r["METAL TRACE REPLAY SESSION ID"] == str(which)]
rows.sort(key=lambda r: f(r, "DEVICE FW START CYCLE"))
idx = [i for i, r in enumerate(rows) if anchor in r["OP NAME"]]
a, b = idx[nth], idx[nth + 1]
tot = collections.Counter()
for r in rows[a + 1 : b + 1]:
    k, g = f(r, "DEVICE KERNEL DURATION [ns]") / 1e3, f(r, "OP TO OP LATENCY [ns]") / 1e3
    print(f"{r['OP NAME'].replace('DeviceOperation', ''):<34}{k:8.2f}{g:7.2f}  cores {r['CORE COUNT']}")
    tot["k"] += k; tot["g"] += g
print(f"{b - a} ops, kernel {tot['k']:.1f} us, gap {tot['g']:.1f} us")
