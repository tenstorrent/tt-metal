#!/usr/bin/env python3
# Validate the cost-model FALLBACK picker on a WIDE variety of unseen shapes. For each shape the picker
# enumerates feasible (Ns,Pk,Sm=1,kb,nsb), scores with the calibrated cost model, picks the best, and we
# run ONLY that config on device -> BW-util. Pass 1 (this file, mode=pick): one run/shape across the grid.
# cwd=ROOT (TT_METAL_HOME detection). SIGTERM timeout + tt-smi -r on hang. Resumable.
import csv, os, subprocess, sys, json
from collections import defaultdict

ROOT = "/localdev/cglagovich/tt-metal"
BIN = f"{ROOT}/build/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
PEAK = 512e9
NRUN = 3
TB = 2048
L1BUDGET = 1440 * 1024
RIDGE = 594.0  # flop/byte at 304 TFLOP / 512 GB/s


def rup(x, y):
    return ((x + y - 1) // y) * y


def cdiv(x, y):
    return (x + y - 1) // y


# ---- calibrated cost-model params (from picker_backtest.py: 96.2% geomean) ----
P = dict(Csat=24, kk=0.5, aa=2, acap=6, ovl=1.0, red=0.0, fwd=0.0, start=0.0)


def plan(M, K, N, Ns, Pk, Sm, kb, nsb):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cores = 8 * Pk * Ns * Sm
    if not (16 <= cores <= 104):
        return None
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    Kts = Pk * Ktl
    wasteK = Kts / Kt - 1
    if wasteK > 0.20:
        return None
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    if nsb > Nown:
        return None
    Nsub = nsb
    Nbpc = cdiv(Nown, Nsub)
    Nowns = Nbpc * Nsub
    Nbands = Ns * Nowns
    Nts = 8 * Nbands
    wasteN = Nts / Nt - 1
    if wasteN > 0.20:
        return None
    cb0 = Ktl * Mblk * TB
    cb1 = 4 * kb * Nsub * TB
    cb2 = 2 * Mblk * Nsub * TB
    cb3 = Mblk * Nsub * 4096
    cb7 = 2 * Mblk * Nsub * TB
    l1 = cb0 + cb1 + cb2 + cb3 + cb7
    if l1 > L1BUDGET:
        return None
    real = (Mt * Kt + Kt * Nt + Mt * Nt) * TB
    return dict(cores=cores, real=real, Ktl=Ktl, Mblk=Mblk, Nown=Nown, Nbpc=Nbpc, wasteK=wasteK, wasteN=wasteN)


def cost(M, K, N, c):
    Ns, Pk, Sm, kb, nsb = c
    g = plan(M, K, N, *c)
    Kt, Nt = K // 32, N // 32
    readT = Kt * Nt / min(g["cores"], P["Csat"])
    comp_pc = g["Mblk"] * g["Nown"] * g["Ktl"]
    area = min(g["Mblk"] * nsb, P["acap"])
    eff_kb = kb / (kb + P["kk"])
    eff_area = area / (area + P["aa"])
    compT = comp_pc / (eff_kb * eff_area)
    ovlT = P["ovl"] * comp_pc / g["Nbpc"]
    return (
        max(readT, compT)
        + ovlT
        + P["red"] * Pk * g["Mblk"] * g["Nown"]
        + P["fwd"] * g["Mblk"] * Pk
        + P["start"] * g["Ktl"]
    )


def pick(M, K, N):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    Nband = cdiv(Nt, 8)
    cands = []
    for Pk in range(1, 13):
        for Ns in range(1, 7):
            Sm = 1
            Nown = cdiv(cdiv(Nt, 8), Ns)
            for kb in (1, 2, 4, 8):
                for nsb in range(1, Nown + 1):
                    if plan(M, K, N, Ns, Pk, Sm, kb, nsb):
                        cands.append((Ns, Pk, Sm, kb, nsb))
    if not cands:
        return None
    return min(cands, key=lambda c: cost(M, K, N, c))


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
        for t, cc in l:
            if t == "ZONE_START":
                st = cc
            elif t == "ZONE_END" and st is not None:
                ds.append(cc - st)
                st = None
        pc[k] = ds
    n = min((len(v) for v in pc.values()), default=0)
    if n < 2:
        return None
    return min(max(v[i] for v in pc.values()) for i in range(1, n))


def run(M, K, N, c):
    Ns, Pk, Sm, kb, nsb = c
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_HOME"] = ROOT
    a = [
        "timeout",
        "-s",
        "TERM",
        "120",
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
        "--num-tests",
        str(NRUN),
    ]
    if Ns > 1:
        a += ["--nslice", str(Ns)]
    if Sm > 1:
        a += ["--msplit", str(Sm)]
    try:
        r = subprocess.run(a, env=env, capture_output=True, text=True, timeout=160, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None, "hang"
    if "PASS" not in r.stdout:
        return None, ("L1" if "beyond max L1" in r.stdout else "fail")
    return cyc(), "ok"


# ---- wide shape grid: per Mt, main K x N grid + edge cases (shallow/deep K, tiny/huge N, non-divisible) ----
def shapes_for(M):
    Mt = M // 32
    Kmain = [2048, 4096, 6144, 12288]
    Nmain = [1536, 3072, 6144, 9216]
    sh = set()
    for K in Kmain:
        for N in Nmain:
            if N > M:
                sh.add((M, K, N))
    # edges
    edge = [
        (256, 6144),
        (512, 4608),
        (15360, 1536),
        (20480, 1536),  # shallow K, deep K
        (2560, 4608),
        (5120, 6144),
        (6144, 2560),
        (6144, 5120),  # non-divisible K and N
        (6144, max(512, M * 2)),
        (6144, 12288),
        (4096, max(512, M * 2)),
    ]  # tiny N, huge N
    for K, N in edge:
        if N > M:
            sh.add((M, K, N))
    return sorted(sh)


GRID = []
for M in [32, 64, 128, 256, 512]:
    GRID += [(M, K, N) for (M, K, N) in shapes_for(M)]

outp = f"{ROOT}/tools/mm_sweep/picker_validate.json"
res = []
done = set()
if os.path.exists(outp):
    try:
        res = json.load(open(outp))
        for r in res:
            done.add((r["M"], r["K"], r["N"]))
        print(f"RESUME: {len(res)} shapes done", flush=True)
    except:
        res = []
        done = set()

print(f"grid = {len(GRID)} shapes", flush=True)
for M, K, N in GRID:
    if (M, K, N) in done:
        continue
    Mt = M // 32
    ai = (M * K * N) / (M * K + K * N + M * N)
    c = pick(M, K, N)
    if c is None:
        print(f"  {M}x{K}x{N}: NO FEASIBLE CONFIG", flush=True)
        continue
    cyc_, st = run(M, K, N, c)
    if st == "hang":
        print(f"  {M}x{K}x{N} {c}: HANG -> reset", flush=True)
        subprocess.run(["tt-smi", "-r"], capture_output=True)
        continue
    if st != "ok":
        print(f"  {M}x{K}x{N} {c}: {st}", flush=True)
        continue
    g = plan(M, K, N, *c)
    bwp = g["real"] / (cyc_ / FREQ) / PEAK * 100
    ceil = min(100.0, RIDGE / ai * 100)  # roofline BW-util ceiling (memory-bound=100)
    rec = dict(
        M=M,
        K=K,
        N=N,
        Mt=Mt,
        ai=round(ai, 1),
        ceil=round(ceil, 0),
        Ns=c[0],
        Pk=c[1],
        Sm=c[2],
        kb=c[3],
        nsb=c[4],
        cores=g["cores"],
        bwp=round(bwp, 1),
        us=round(cyc_ / FREQ * 1e6, 1),
        wasteK=round(g["wasteK"], 2),
        wasteN=round(g["wasteN"], 2),
    )
    res.append(rec)
    done.add((M, K, N))
    json.dump(res, open(outp, "w"))
    print(
        f"  {M}x{K}x{N:>6} Mt{Mt} ai{ai:>5.0f} pick({c[0]},{c[1]},{c[2]})kb{c[3]}nsb{c[4]} {g['cores']:>3}c -> {bwp:>4.0f}%  (ceil {ceil:.0f}, {cyc_/FREQ*1e6:.0f}us)",
        flush=True,
    )
print("\nDONE", flush=True)
