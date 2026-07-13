#!/usr/bin/env python3
# Read-only depth sweep: separate stride vs outstanding-depth; find contiguous multi-reader ceiling.
import csv, os, subprocess
from collections import defaultdict

BIN = "build_Release/test/tt_metal/perf_microbenchmark/sp_bankread/test_bank_read"
CSV = "generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
KT, NBAND, TB = 192, 18, 2048
TOTAL = KT * NBAND * 8 * TB  # 54 MB


def brisc_cyc():
    ev = defaultdict(list)
    try:
        rows = list(csv.reader(open(CSV)))
    except:
        return None
    for r in rows[2:]:
        if len(r) >= 12 and r[3] == "BRISC" and r[10].strip() == "BRISC-KERNEL":
            ev[(r[1], r[2])].append((r[11].strip(), int(r[5])))
    pc = {}
    for kk, l in ev.items():
        ds = []
        st = None
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        pc[kk] = ds
    n = min((len(v) for v in pc.values()), default=0)
    return min(max(v[i] for v in pc.values()) for i in range(1, n)) if n >= 2 else None


def run(mode, P, depth):
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    try:
        r = subprocess.run(
            [
                "timeout",
                "120",
                BIN,
                "--kt",
                str(KT),
                "--nband",
                str(NBAND),
                "--preaders",
                str(P),
                "--mode",
                mode,
                "--depth",
                str(depth),
                "--num-tests",
                "6",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=140,
        )
    except subprocess.TimeoutExpired:
        return None
    if "PASS" not in r.stdout:
        return None
    c = brisc_cyc()
    return TOTAL / (c / FREQ) / 1e9 if c else None


print(f"{'config':>22} " + " ".join(f"d{d:<5}" for d in [2, 4, 8, 16, 32]))
for mode, P, label in [
    ("contig", 1, "P1 contig(whole bank)"),
    ("contig", 2, "P2 contig(Kslice)"),
    ("strided", 2, "P2 strided(ns=9)"),
    ("contig", 3, "P3 contig(Kslice)"),
    ("strided", 3, "P3 strided(ns=6)"),
]:
    row = f"{label:>22} "
    for d in [2, 4, 8, 16, 32]:
        bw = run(mode, P, d)
        row += f"{bw:5.0f} " if bw else "  --  "
    print(row, flush=True)
