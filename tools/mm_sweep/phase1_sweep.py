#!/usr/bin/env python3
# PHASE 1 done right: sweep (Ns,Pk,Sm,kb,nsb) with in0 read+forward AND reduction ABLATED
# (--skipin0 --skipfwd --noreduce), to find the slicing that MINIMIZES in1+compute time.
# Metric = kernel time; compare to DRAM roofline (in1 read at peak).
import csv, os, subprocess, sys, json
from collections import defaultdict

ROOT = "/localdev/cglagovich/tt-metal"
BIN = f"{ROOT}/build/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
PEAK = 512e9
NRUN = 4
TB = 2048
L1BUDGET = 1440 * 1024


def rup(x, y):
    return ((x + y - 1) // y) * y


def cdiv(x, y):
    return (x + y - 1) // y


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


SHAPES = [(256, 6144, 4608), (512, 6144, 4608), (512, 6144, 2304)]
Pk_list = [1, 2, 3, 4, 6, 8, 12]
Ns_list = [1, 2, 3, 4, 6]
Sm_list = [1, 2, 4]
kb_list = [1, 2, 4]


def plan(M, K, N, Ns, Pk, Sm, kb, nsb):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cores = 8 * Pk * Ns * Sm
    if not (48 <= cores <= 104):
        return None
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    Kts = Pk * Ktl
    if Kts / Kt - 1 > 0.20:
        return None
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    if nsb > Nown:
        return None
    Nbpc = cdiv(Nown, nsb)
    cb0 = Ktl * Mblk * TB
    cb1 = 4 * kb * nsb * TB
    cb2 = 2 * Mblk * nsb * TB
    cb3 = Mblk * nsb * 4096
    cb7 = 2 * Mblk * nsb * TB
    if cb0 + cb1 + cb2 + cb3 + cb7 > L1BUDGET:
        return None
    return dict(cores=cores, Mblk=Mblk)


def gen(M, K, N):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cfgs = []
    for Pk in Pk_list:
        for Ns in Ns_list:
            for Sm in Sm_list:
                if Mt < Sm or not (6 <= Pk * Ns * Sm <= 13):
                    continue
                for kb in kb_list:
                    if kb > cdiv(Kt, Pk):
                        continue
                    Nown = cdiv(cdiv(Nt, 8), Ns)
                    divs = sorted(set(divisors(Nown)))
                    for nsb in sorted(set([divs[0], divs[len(divs) // 2], divs[-1]])):
                        p = plan(M, K, N, Ns, Pk, Sm, kb, nsb)
                        if p:
                            cfgs.append((Ns, Pk, Sm, kb, nsb, p))
    return cfgs


def cyc():
    ev = defaultdict(list)
    try:
        rows = list(csv.reader(open(CSV)))
    except:
        return None
    for row in rows[2:]:
        if len(row) < 12:
            continue
        if row[10].strip().endswith("-KERNEL"):
            ev[(row[1], row[2], row[3])].append((row[11].strip(), int(row[5])))
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


def run(M, K, N, Ns, Pk, Sm, kb, nsb):
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    a = [
        "timeout",
        "-s",
        "TERM",
        "80",
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
        str(kb),
        "--nsb",
        str(nsb),
        "--skipin0",
        "--skipfwd",
        "--noreduce",
        "--num-tests",
        str(NRUN),
    ]
    if Ns > 1:
        a += ["--nslice", str(Ns)]
    if Sm > 1:
        a += ["--msplit", str(Sm)]
    try:
        r = subprocess.run(a, env=env, capture_output=True, text=True, timeout=110)
    except subprocess.TimeoutExpired:
        return None, "hang"
    # noreduce/skipin0 -> wrong output, that's expected; we only need timing (profiler CSV written on run)
    if "Always | Profiler" not in r.stdout and "Test |" not in r.stdout:
        return None, "fail"
    c = cyc()
    return (c, "ok") if c else (None, "nodata")


allres = []
for M, K, N in SHAPES:
    bytes_ = 2 * (M * K + K * N + M * N)
    roof = bytes_ / PEAK * 1e6
    cfgs = gen(M, K, N)
    print(f"\n### {M}x{K}x{N} Mt={M//32} roofline={roof:.0f}us ({len(cfgs)} cfgs)", flush=True)
    sh = []
    for Ns, Pk, Sm, kb, nsb, p in cfgs:
        c, st = run(M, K, N, Ns, Pk, Sm, kb, nsb)
        if st == "hang":
            print("  HANG->reset", flush=True)
            subprocess.run(["tt-smi", "-r"], capture_output=True)
            continue
        if st != "ok":
            continue
        us = c / FREQ * 1e6
        rec = dict(M=M, K=K, N=N, Ns=Ns, Pk=Pk, Sm=Sm, kb=kb, nsb=nsb, cores=p["cores"], us=us, pct=roof / us * 100)
        sh.append(rec)
        allres.append(rec)
        json.dump(allres, open(f"{ROOT}/tools/mm_sweep/phase1_sweep.json", "w"))
    sh.sort(key=lambda r: r["us"])
    print(f"  TOP8 in1+compute (min time):", flush=True)
    for r in sh[:8]:
        print(
            f"    ({r['Ns']},{r['Pk']},{r['Sm']}) kb{r['kb']} nsb{r['nsb']} {r['cores']}c -> {r['us']:.0f}us ({r['pct']:.0f}% of roofline)",
            flush=True,
        )
print("\nDONE", flush=True)
