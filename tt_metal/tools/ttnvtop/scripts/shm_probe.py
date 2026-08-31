#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Samples every /dev/shm/tt_device_*_util at a fixed rate and answers two questions the
# TUI can only gesture at:
#
#   1. CONSISTENCY -- per chip: worst staleness, and how many times the publish stalled
#      past a threshold. "It looks hung" becomes a number, per chip, so a chip that really
#      does stall (chip 7 was reported stalling) is separated from one that never did.
#   2. ACCURACY -- mean F/S/D/DRAM within each calibration phase, so the reading can be
#      regressed against a known duty cycle.
#
# Reads the schema defensively: the sizes are asserted, because two earlier verifier
# versions silently mis-strided PerCoreView and produced 205% and 2046% readings.

import argparse
import glob
import json
import os
import struct
import sys
import time
from collections import defaultdict

H = struct.Struct("<4sHHQIIQQIIIIIIII")
# PerCoreView, field for field against common/shm_schema.hpp.
#
# The previous format string was "<IIHHHHIIIIII". It is also 40 bytes, so
# `assert V.size == 40` PASSED while every field was shifted: what was read as
# compute_busy was dispatch_busy, what was read as sfpu_busy was compute_busy,
# and samples_seen was noc1_out_mbps. A size check cannot catch a wrong SHAPE --
# only naming the fields can, so they are named here and indexed by name below.
V_FIELDS = [
    "noc_x",
    "noc_y",
    "logical_x",
    "logical_y",
    "is_remote",
    "dispatched",  # 6 x u8
    "sfpu_busy_p1000",
    "dispatch_busy_p1000",
    "compute_busy_p1000",  # u16...
    "unpack_busy_p1000",
    "pack_busy_p1000",
    "stall_p1000",
    "noc0_in_mbps",
    "noc0_out_mbps",
    "noc1_in_mbps",
    "noc1_out_mbps",
    "samples_seen",
    "last_kernel_id",
    "reserved_1",  # 3 x u32 after 2 pad
]
V = struct.Struct("<6B10H2x3I")
assert V.size == 40, V.size
assert len(V_FIELDS) == 19, "field names must match the format"
VI = {n: i for i, n in enumerate(V_FIELDS)}

assert H.size == 72, H.size
assert V.size == 40, V.size


def sample():
    out = {}
    now = time.monotonic()
    for f in glob.glob("/dev/shm/tt_device_*_util"):
        try:
            b = open(f, "rb").read()
        except OSError:
            continue
        if len(b) < H.size:
            continue
        h = H.unpack_from(b, 0)
        if h[0] != b"TTUT":
            continue
        nc = h[8]
        if len(b) < H.size + nc * V.size:
            continue
        fsum = ssum = dsum = 0
        fnz = 0
        for i in range(nc):
            v = V.unpack_from(b, H.size + i * V.size)
            fsum += v[VI["compute_busy_p1000"]]
            ssum += v[VI["sfpu_busy_p1000"]]
            dsum += v[VI["dispatch_busy_p1000"]]
            if v[VI["compute_busy_p1000"]]:
                fnz += 1
        out[os.path.basename(f)] = {
            "t": now,
            "age": now - h[7] / 1e6,
            "aiclk": h[11],
            "F": fsum / nc / 10.0,
            "S": ssum / nc / 10.0,
            "D": dsum / nc / 10.0,
            "Fnz": fnz,
            "dram_rd": h[12],
            "dram_wr": h[13],
            "cores": nc,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--stall-ms", type=float, default=1000.0, help="age above this counts as a stall")
    ap.add_argument("--phases", type=str, default="", help="calib_duty.py JSON to align against")
    ap.add_argument("--raw", type=str, default="", help="write every sample here as JSONL")
    args = ap.parse_args()

    period = 1.0 / args.hz
    end = time.monotonic() + args.seconds
    rows = []
    raw = open(args.raw, "w") if args.raw else None
    while time.monotonic() < end:
        s = sample()
        for name, r in s.items():
            r["chip"] = name
            rows.append(r)
            if raw:
                raw.write(json.dumps(r) + "\n")
        time.sleep(period)
    if raw:
        raw.close()

    if not rows:
        print("NO SHM FILES -- is the collector running?")
        return 1

    # ---- consistency, per chip
    per = defaultdict(list)
    for r in rows:
        per[r["chip"]].append(r)
    print(f"\n=== CONSISTENCY  ({len(rows)} samples, {args.hz} Hz, stall > {args.stall_ms:.0f} ms) ===")
    print(f"{'chip':22s} {'n':>5s} {'max age':>9s} {'mean age':>9s} {'stalls':>7s} {'worst gap':>10s} {'aiclk=0':>8s}")
    for name in sorted(per):
        rs = per[name]
        ages = [r["age"] for r in rs]
        stalls = sum(1 for a in ages if a > args.stall_ms / 1000.0)
        # worst gap between successive observations of a CHANGED last_update
        gap = 0.0
        prev_t = None
        for r in rs:
            if prev_t is not None:
                gap = max(gap, r["t"] - prev_t)
            prev_t = r["t"]
        zc = sum(1 for r in rs if r["aiclk"] == 0)
        print(
            f"{name:22s} {len(rs):5d} {max(ages):8.2f}s {sum(ages)/len(ages):8.2f}s " f"{stalls:7d} {gap:9.2f}s {zc:7d}"
        )

    # ---- accuracy, aligned to the calibration phases
    if args.phases and os.path.exists(args.phases):
        ph = json.load(open(args.phases))
        print(f"\n=== ACCURACY vs known duty cycle (matmul size {ph.get('size')}) ===")
        print(f"{'target':>7s} {'host busy':>10s} | per-chip mean FPU% (monitor)")
        fits = defaultdict(list)
        for p in ph["phases"]:
            # skip the first 25% of each phase: the EWMA and the aggregator sweep both
            # need time to reach the new level, and including the ramp biases the fit.
            lo = p["t_start"] + 0.25 * (p["t_end"] - p["t_start"])
            line = f"{p['duty_target']:6.0f}% {p['duty_actual_host']:9.1f}% |"
            for name in sorted(per):
                sel = [r["F"] for r in per[name] if lo <= r["t"] <= p["t_end"]]
                if sel:
                    m = sum(sel) / len(sel)
                    line += f" {name.split('_')[2]}:{m:5.2f}"
                    fits[name].append((p["duty_actual_host"], m))
            print(line)

        print(f"\n=== LINEARITY  (monitor% = slope x host_busy% + intercept) ===")
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
            if r2 > 0.9 and slope > 0.02:
                verdict = "TRACKS duty"
            elif slope <= 0.02:
                verdict = "FLAT -- reading does not respond to compute"
            else:
                verdict = "responds but NOT linear"
            print(f"{name:22s} {slope:8.3f} {icept:10.2f} {r2:7.3f}   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
