#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Joins shm_probe.py --raw output against calib_duty.py phase boundaries. Separate from the
# probe because the phase file only exists once the workload has finished, so the live
# probe cannot align to it -- it has to be done after the fact, off the recorded samples.
#
# Usage: calib_report.py RAW.jsonl PHASES.json [metric]
#
# The metric names are the C field names from PerCoreView (common/shm_schema.hpp:72),
# not the old single letters. Raw files recorded before 2026-08-31 carry "F"/"S"/"D"
# keys whose LABELS DO NOT MATCH THEIR FIELDS (shm_probe.py used a same-size but wrong
# struct format), so this deliberately refuses to read them rather than silently
# reproducing the mislabelled slope.

import json
import sys
from collections import defaultdict

# probe key -> the PerCoreView field it actually reads
METRICS = {
    "fpu": "compute_busy_p1000 (FPU / MATH pipe)",
    "sfpu": "sfpu_busy_p1000 (vector pipe)",
    "dispatch": "dispatch_busy_p1000 (go_msg occupancy)",
}
LEGACY_KEYS = ("F", "S", "D", "Fnz", "dram_rd", "dram_wr")


def main() -> int:
    raw_path, ph_path = sys.argv[1], sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "fpu"
    if metric not in METRICS:
        print(f"unknown metric {metric!r}; choose one of {', '.join(sorted(METRICS))}", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in open(raw_path) if l.strip()]
    # An empty or blank-only raw file used to produce a structurally normal, entirely
    # data-free report and exit 0. A gate script that checks only the exit code would
    # then pass on NO DATA, which is the same class of silent-success failure that let
    # the mislabelled 5w slopes through. Refuse instead.
    if not rows:
        print(f"{raw_path} contains no samples -- nothing to report (was the probe running?)", file=sys.stderr)
        return 3
    if metric not in rows[0]:
        stale = [k for k in LEGACY_KEYS if k in rows[0]]
        if stale:
            print(
                f"{raw_path} was recorded by the pre-fix shm_probe.py (keys {stale}). Those\n"
                f"samples are mislabelled at the source -- 'F' is dispatch_busy_p1000 and 'D' is\n"
                f"a field the collector never writes. Re-record with the corrected probe.",
                file=sys.stderr,
            )
        else:
            print(f"{raw_path} has no {metric!r} key (keys: {sorted(rows[0])})", file=sys.stderr)
        return 2
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

    print(f"\n=== ACCURACY: monitor {metric} [{METRICS[metric]}] vs host-measured busy % (matmul {ph.get('size')}) ===")
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

    # Same reasoning as the empty-file case: a report whose ACCURACY table is entirely
    # "-" is not a result. If no chip yielded enough in-phase samples to fit, say so and
    # exit non-zero rather than printing an empty LINEARITY table and returning success.
    fitted = [n for n in fits if len(fits[n]) >= 3]
    if not fitted:
        print(
            f"\nNO FIT: no chip had >=3 phases with samples in range. The probe run and the\n"
            f"phase file ({ph_path}) do not overlap, or the probe saw no SHM files.",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
