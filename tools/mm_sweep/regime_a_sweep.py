#!/usr/bin/env python3
# Sweep the Regime-A skinny shapes through the regime_a_mm prototype (8 bank-adjacent cores),
# sweeping kb (K-block). Reports composite (all-RISC) op time, BW-util (table metric 2*(MK+KN+MN)),
# and speedup vs the existing branch. Constant-input correctness (out==K) validates the pipeline.
import csv, os, subprocess, sys

BIN = "build_Release/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
CSV = "generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9

# Regime-A shapes (N>=M) from bh_skinny_results.md, with the existing-branch best us.
SHAPES = [
    (32, 256, 6144, 11.2),
    (32, 2048, 512, 9.4),
    (32, 2048, 1536, 18.8),
    (32, 2048, 2048, 23.1),
    (32, 6144, 1536, 51.9),
    (32, 6144, 2304, 78.3),
    (32, 6144, 3072, 105.5),
    (32, 6144, 6144, 199.6),
    (32, 6144, 9216, 293.4),
    (64, 6144, 1536, 51.7),
    (64, 15360, 1536, 137.4),
    (64, 4608, 6144, 153.8),
    (64, 6144, 4608, 156.1),
    (64, 6144, 9216, 296.1),
    (128, 6144, 768, 37.1),
    (128, 15360, 768, 86.6),
    (128, 6144, 2304, 82.5),
    (128, 2304, 6144, 84.9),
    (128, 6144, 4608, 155.4),
    (512, 6144, 1536, 87.0),
]
KBS = [8, 4, 2]


def all_risc_cyc():
    from collections import defaultdict

    ev = defaultdict(list)
    try:
        rows = list(csv.reader(open(CSV)))
    except Exception:
        return None
    for row in rows[2:]:
        if len(row) < 12:
            continue
        if row[10].strip().endswith("-KERNEL"):
            ev[(row[1], row[2], row[3])].append((row[11].strip(), int(row[5])))
    runmax = None
    percore = {}
    for k, l in ev.items():
        ds = []
        st = None
        for typ, cyc in l:
            if typ == "ZONE_START":
                st = cyc
            elif typ == "ZONE_END" and st is not None:
                ds.append(cyc - st)
                st = None
        percore[k] = ds
    n = min((len(v) for v in percore.values()), default=0)
    if n < 2:
        return None
    best = min(max(v[i] for v in percore.values()) for i in range(1, n))  # skip cold run 0
    return best


def run(M, K, N, kb):
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
    ok = "PASS" in r.stdout
    if not ok:
        return None, "fail"
    return all_risc_cyc(), "ok"


print(f"{'shape':>18} {'AI':>4} {'bestkb':>6} {'us':>7} {'BWutil%':>7} {'branch_us':>9} {'speedup':>7}")
for M, K, N, bru in SHAPES:
    ai = 2.0 * M * K * N / (2.0 * (M * K + K * N + M * N))
    minb = 2.0 * (M * K + K * N + M * N)  # table BW-util traffic (bytes)
    best_us = None
    best_kb = None
    for kb in KBS:
        cyc, st = run(M, K, N, kb)
        if st == "hang":
            subprocess.run(["tt-smi", "-r"], capture_output=True)
            continue
        if cyc is None:
            continue
        us = cyc / FREQ * 1e6
        if best_us is None or us < best_us:
            best_us = us
            best_kb = kb
    if best_us is None:
        print(f"{f'{M}x{K}x{N}':>18} {ai:>4.0f} {'--':>6} {'FAIL/L1':>7}")
        continue
    bw = minb / (best_us * 1e-6) / 500e9 * 100
    print(
        f"{f'{M}x{K}x{N}':>18} {ai:>4.0f} {best_kb:>6} {best_us:>7.1f} {bw:>7.1f} {bru:>9.1f} {bru/best_us:>6.2f}x",
        flush=True,
    )
