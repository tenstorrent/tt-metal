#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Joins shm_probe.py --raw output against calib_duty.py phase boundaries. Separate from the
# probe because the phase file only exists once the workload has finished, so the live
# probe cannot align to it -- it has to be done after the fact, off the recorded samples.

import json
import sys
from collections import defaultdict


def main() -> int:
    raw_path, ph_path = sys.argv[1], sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "F"
    rows = [json.loads(l) for l in open(raw_path) if l.strip()]
    ph = json.load(open(ph_path))
    per = defaultdict(list)
    for r in rows:
        per[r["chip"]].append(r)

    print(f"=== CONSISTENCY ({len(rows)} samples over {len(per)} chips) ===")
    print(f"{'chip':22s} {'n':>5s} {'max age':>9s} {'mean age':>9s} {'stalls>1s':>10s} {'aiclk=0':>8s}")
    for name in sorted(per):
        rs = per[name]
        ages = [r["age"] for r in rs]
        print(
            f"{name:22s} {len(rs):5d} {max(ages):8.2f}s {sum(ages)/len(ages):8.2f}s "
            f"{sum(1 for a in ages if a > 1.0):10d} {sum(1 for r in rs if r['aiclk']==0):8d}"
        )

    print(f"\n=== ACCURACY: monitor {metric} vs host-measured busy % (matmul {ph.get('size')}) ===")
    hdr = f"{'target':>7s} {'host':>7s} |"
    for name in sorted(per):
        hdr += f" {name.split('_')[2]:>6s}"
    print(hdr)
    fits = defaultdict(list)
    for p in ph["phases"]:
        lo = p["t_start"] + 0.25 * (p["t_end"] - p["t_start"])
        line = f"{p['duty_target']:6.0f}% {p['duty_actual_host']:6.1f}% |"
        for name in sorted(per):
            sel = [r[metric] for r in per[name] if lo <= r["t"] <= p["t_end"]]
            if sel:
                m = sum(sel) / len(sel)
                line += f" {m:6.2f}"
                fits[name].append((p["duty_actual_host"], m))
            else:
                line += "      -"
        print(line)

    print(f"\n=== LINEARITY: monitor_{metric} = slope x host_busy + intercept ===")
    print(f"{'chip':22s} {'slope':>8s} {'intercept':>10s} {'R^2':>7s}   verdict")
    for name in sorted(fits):
        pts = fits[name]
        if len(pts) < 3:
            continue
        n = len(pts)
        sx = sum(x for x, _ in pts)
        sy = sum(y for _, y in pts)
        sxx = sum(x * x for x, _ in pts)
        sxy = sum(x * y for x, y in pts)
        syy = sum(y * y for _, y in pts)
        den = n * sxx - sx * sx
        if den == 0:
            continue
        slope = (n * sxy - sx * sy) / den
        icept = (sy - slope * sx) / n
        dr = n * syy - sy * sy
        r2 = ((n * sxy - sx * sy) ** 2 / (den * dr)) if dr > 0 else 0.0
        if slope <= 0.02:
            verdict = "FLAT -- does not respond to compute at all"
        elif r2 > 0.9:
            verdict = "TRACKS duty (linear)"
        else:
            verdict = "responds but NOT linear"
        print(f"{name:22s} {slope:8.3f} {icept:10.2f} {r2:7.3f}   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
