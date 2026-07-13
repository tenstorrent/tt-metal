#!/usr/bin/env python3
# N-sub-division sweep: run M=64/128 shapes at P readers/bank, sweep kb, report composite BW-util +
# speedup vs branch + vs the P=1 result. Constant-input correctness (out==K).
import csv, os, subprocess
from collections import defaultdict

BIN = "build_Release/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
CSV = "generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
# (M,K,N,branch_us, P1_us_from_prev_sweep, [(P,kb),...])
CFG = [
    (64, 6144, 1536, 51.7, 64.2, [(2, 2), (2, 4)]),  # Nt=48 -> P2 ns=3
    (64, 6144, 4608, 156.1, 188.5, [(2, 4), (2, 8)]),  # Nt=144 -> P2 ns=9
    (64, 4608, 6144, 153.8, 190.5, [(2, 4), (2, 8)]),  # Nt=192 -> P2 ns=12
    (64, 6144, 9216, 296.1, None, [(2, 2), (2, 4)]),  # was FAIL/L1 at P1; Nt=288 P2 ns=18
    (128, 6144, 2304, 82.5, 189.4, [(3, 4), (3, 8)]),  # Nt=72 -> P3 ns=3, 24 cores
    (128, 6144, 4608, 155.4, None, [(3, 4), (6, 4)]),  # Nt=144 -> P3 ns=6 (24c) / P6 ns=3 (48c)
    (128, 2304, 6144, 84.9, None, [(3, 4), (6, 4)]),  # Nt=192 -> P3 ns=8 (24c) / P6 ns=4 (48c)
]


def cyc():
    ev = defaultdict(list)
    try:
        rows = list(csv.reader(open(CSV)))
    except:
        return None
    for r in rows[2:]:
        if len(r) >= 12 and r[10].strip().endswith("-KERNEL"):
            ev[(r[1], r[2], r[3])].append((r[11].strip(), int(r[5])))
    pc = {}
    for k, l in ev.items():
        ds = []
        st = None
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        pc[k] = ds
    n = min((len(v) for v in pc.values()), default=0)
    return min(max(v[i] for v in pc.values()) for i in range(1, n)) if n >= 2 else None


def run(M, K, N, P, kb):
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    try:
        r = subprocess.run(
            [
                "timeout",
                "150",
                BIN,
                "--m",
                str(M),
                "--k",
                str(K),
                "--n",
                str(N),
                "--gx",
                "4",
                "--gy",
                "2",
                "--kb",
                str(kb),
                "--sharded",
                "--preaders",
                str(P),
                "--num-tests",
                "5",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=170,
        )
    except subprocess.TimeoutExpired:
        return None, "hang"
    if "PASS" not in r.stdout:
        return None, "fail"
    return cyc(), "ok"


print(
    f"{'shape':>16} {'P':>2} {'cores':>5} {'kb':>3} {'us':>7} {'BWutil%':>7} {'branch':>7} {'sp_br':>6} {'P1':>7} {'sp_P1':>6}"
)
for M, K, N, bru, p1us, cfgs in CFG:
    minb = 2.0 * (M * K + K * N + M * N)
    for P, kb in cfgs:
        c, st = run(M, K, N, P, kb)
        if st == "hang":
            subprocess.run(["tt-smi", "-r"], capture_output=True)
        if c is None:
            print(f"{f'{M}x{K}x{N}':>16} {P:>2} {8*P:>5} {kb:>3} {'FAIL/'+st:>7}", flush=True)
            continue
        us = c / FREQ * 1e6
        bw = minb / (us * 1e-6) / 500e9 * 100
        spb = bru / us
        spp1 = (p1us / us) if p1us else 0
        print(
            f"{f'{M}x{K}x{N}':>16} {P:>2} {8*P:>5} {kb:>3} {us:>7.1f} {bw:>7.1f} {bru:>7.1f} {spb:>5.2f}x {str(p1us):>7} {spp1:>5.2f}x",
            flush=True,
        )
