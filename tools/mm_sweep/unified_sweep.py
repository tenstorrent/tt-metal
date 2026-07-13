#!/usr/bin/env python3
# Unified (Ns,Pk,Sm) regime-A sweep across Mt=1..16, varying Kt/Nt. Reports aggregate DRAM BW-util.
# BW-util% = 2*(MK+KN+MN)/kernel_time/512GB/s. Timing = max-core all-RISC KERNEL cycles, best of steady runs.
# Configs chosen divisible (no padding) so BW-util reflects REAL traffic. kb = Kt_local/8 (ring-max deep block).
import csv, os, subprocess, sys, json
from collections import defaultdict

ROOT = "/localdev/cglagovich/tt-metal"
BIN = f"{ROOT}/build/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
PEAK = 512e9
NRUN = 6

# (M,K,N, [ (Ns,Pk,Sm), ... ])
SHAPES = [
    # Mt=1
    (32, 6144, 4608, [(1, 12, 1), (1, 8, 1)]),
    (32, 6144, 9216, [(1, 12, 1), (2, 6, 1)]),
    (32, 2048, 2048, [(1, 8, 1), (1, 4, 1)]),
    (32, 4608, 2304, [(1, 6, 1), (1, 3, 1)]),
    # Mt=2
    (64, 6144, 4608, [(1, 12, 1), (1, 6, 2)]),
    (64, 6144, 1536, [(1, 12, 1), (1, 6, 2)]),
    (64, 2048, 6144, [(1, 8, 1), (1, 4, 2)]),
    # Mt=4
    (128, 6144, 4608, [(1, 6, 2), (1, 3, 4), (1, 6, 4)]),
    (128, 6144, 2304, [(1, 6, 2), (1, 3, 4)]),
    (128, 2304, 6144, [(1, 3, 4), (1, 3, 2), (1, 3, 1)]),
    # Mt=8
    (256, 6144, 4608, [(1, 6, 2), (1, 3, 4), (2, 3, 2)]),
    (256, 4608, 4608, [(1, 6, 2), (1, 3, 4), (2, 6, 2)]),
    # Mt=16
    (512, 6144, 4608, [(1, 6, 2), (1, 3, 4), (2, 3, 2)]),
    (512, 3072, 6144, [(1, 6, 2), (1, 3, 4), (2, 6, 2)]),
    (512, 6144, 2304, [(1, 6, 2), (1, 3, 4)]),
]


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def plan(M, K, N, Ns, Pk, Sm):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    if Mt % Sm:
        return None
    if Kt % Pk:
        return None
    Ktl = Kt // Pk
    if Ktl % 8:
        return None
    kb = Ktl // 8
    if Nt % 8:
        return None
    Nband = Nt // 8
    if Nband % Ns:
        return None
    Nown = Nband // Ns
    Mblk = Mt // Sm
    # nsb: largest divisor of Nown with Mblk*nsb<=48 (out/intermediate cb) AND kb*nsb<=48 (in1 cb, 4-deep); >=1
    nsb = 1
    for d in sorted(divisors(Nown), reverse=True):
        if Mblk * d <= 48 and kb * d <= 48:
            nsb = d
            break
    cores = 8 * Pk * Ns * Sm
    if cores > 110:
        return None
    return dict(kb=kb, nsb=nsb, cores=cores, Ktl=Ktl, Mblk=Mblk)


def all_risc_cyc():
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
    return min(max(v[i] for v in percore.values()) for i in range(1, n))  # skip cold run 0


def run(M, K, N, Ns, Pk, Sm, p):
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    args = [
        "timeout",
        "-s",
        "TERM",
        "110",
        BIN,
        "--unified",
        "--m",
        str(M),
        "--k",
        str(K),
        "--n",
        str(N),
        "--ksplit",
        str(Pk),
        "--kb",
        str(p["kb"]),
        "--nsb",
        str(p["nsb"]),
        "--num-tests",
        str(NRUN),
    ]
    if Ns > 1:
        args += ["--nslice", str(Ns)]
    if Sm > 1:
        args += ["--msplit", str(Sm)]
    try:
        r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=140)
    except subprocess.TimeoutExpired:
        return None, "hang"
    if "PASS" not in r.stdout:
        if "beyond max L1" in r.stdout:
            return None, "L1OOM"
        return None, "fail"
    return all_risc_cyc(), "ok"


results = []
outp = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/tools/mm_sweep/unified_sweep_results.json"
print(
    f"{'shape':>18} {'Mt':>3} {'AI':>4} {'best cfg (Ns,Pk,Sm,kb,nsb)':>28} {'cores':>5} {'us':>7} {'GB/s':>6} {'BW%':>5}",
    flush=True,
)
for M, K, N, cands in SHAPES:
    Mt = M // 32
    ai = 2.0 * M * K * N / (2.0 * (M * K + K * N + M * N))
    bytes_ = 2.0 * (M * K + K * N + M * N)
    best = None
    for Ns, Pk, Sm in cands:
        p = plan(M, K, N, Ns, Pk, Sm)
        if p is None:
            print(f"  skip non-divisible cfg Ns{Ns}Pk{Pk}Sm{Sm} for {M}x{K}x{N}", flush=True)
            continue
        cyc, st = run(M, K, N, Ns, Pk, Sm, p)
        if st == "hang":
            print(f"  HANG cfg Ns{Ns}Pk{Pk}Sm{Sm} {M}x{K}x{N} -> reset", flush=True)
            subprocess.run(["tt-smi", "-r"], capture_output=True)
            continue
        if st != "ok":
            print(f"  {st} cfg Ns{Ns}Pk{Pk}Sm{Sm} {M}x{K}x{N}", flush=True)
            continue
        us = cyc / FREQ * 1e6
        bw = bytes_ / (cyc / FREQ) / 1e9
        bwp = bw / PEAK * 1e9 * 100
        rec = dict(
            M=M,
            K=K,
            N=N,
            Mt=Mt,
            Ns=Ns,
            Pk=Pk,
            Sm=Sm,
            kb=p["kb"],
            nsb=p["nsb"],
            cores=p["cores"],
            us=us,
            gbs=bw,
            bwp=bwp,
            ai=ai,
        )
        results.append(rec)
        if best is None or us < best["us"]:
            best = rec
    if best:
        c = best
        cfg = f"({c['Ns']},{c['Pk']},{c['Sm']},{c['kb']},{c['nsb']})"
        print(
            f"{f'{M}x{K}x{N}':>18} {Mt:>3} {ai:>4.0f} {cfg:>28} {c['cores']:>5} {c['us']:>7.1f} {c['gbs']:>6.0f} {c['bwp']:>5.1f}",
            flush=True,
        )
    else:
        print(f"{f'{M}x{K}x{N}':>18} {Mt:>3} {ai:>4.0f} {'ALL FAILED':>28}", flush=True)
    json.dump(results, open(outp, "w"), indent=1)
print("DONE", flush=True)
