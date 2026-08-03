#!/usr/bin/env python3
"""Join each completed vllm-bench sweep point with the board power drawn during it."""
import glob
import json
import os
import re
import sys
import time

POW = "/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/power_watch.log"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep_report.log"
PAT = "/tmp/v2_*.json"
TDP_MESH = 500.0


def power_between(t0, t1):
    """mean/max total board W from power_watch rows whose HH:MM:SS falls in [t0,t1]."""
    vals = []
    try:
        for ln in open(POW):
            m = re.match(r"(\d\d:\d\d:\d\d) .*?\s{2,}.*?\s+(\d+)\s+\d+%", ln)
            if m and t0 <= m.group(1) <= t1:
                vals.append(int(m.group(2)))
    except OSError:
        pass
    if not vals:
        return None, None, 0
    return sum(vals) / len(vals), max(vals), len(vals)


def hms(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts))


seen = {}
hdr = (
    f"{'point (ISL/OSL/C)':<22}{'t/s/u':>7}{'agg t/s':>9}{'TTFT s':>8}{'E2EL s':>8}"
    f"{'mesh W':>8}{'%TDP':>6}{'peak W':>8}  note"
)
with open(OUT, "a", buffering=1) as f:
    f.write("\n=== sweep_report start ===\n" + hdr + "\n")
    while True:
        for p in sorted(glob.glob(PAT), key=os.path.getmtime):
            mt = os.path.getmtime(p)
            if seen.get(p) == mt:
                continue
            seen[p] = mt
            try:
                d = json.load(open(p))
            except Exception:
                continue
            name = os.path.basename(p)[3:-5].replace("_", "/")
            if not d.get("completed"):
                f.write(
                    f"{name:<22}{'--':>7}{'--':>9}{'--':>8}{'--':>8}"
                    f"{'--':>8}{'--':>6}{'--':>8}  FAILED (completed=0)\n"
                )
                continue
            dur = d["duration"]
            mean_w, peak_w, n = power_between(hms(mt - dur), hms(mt))
            tsu = 1000 / d["mean_tpot_ms"] if d["mean_tpot_ms"] else 0
            wtxt = (
                f"{mean_w:8.0f}{100*mean_w/TDP_MESH:5.0f}%{peak_w:8.0f}"
                if mean_w
                else f"{'n/a':>8}{'n/a':>6}{'n/a':>8}"
            )
            f.write(
                f"{name:<22}{tsu:7.2f}{d['output_throughput']:9.1f}"
                f"{d['mean_ttft_ms']/1000:8.2f}{d['mean_e2el_ms']/1000:8.1f}{wtxt}  ({n} pwr samples)\n"
            )
        time.sleep(5)
