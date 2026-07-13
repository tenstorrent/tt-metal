#!/usr/bin/env python3
# Pass 2: classify picker outliers. For each suspect shape, run the picker's pick PLUS a curated set of
# alternatives (cost-model top-K + lowest-waste high-core configs). Report best-found vs picker -> tells us
# heuristic-miss (better config existed) vs shape-floor (nothing better).
import csv, os, subprocess, json
from collections import defaultdict

ROOT = "/localdev/cglagovich/tt-metal"
BIN = f"{ROOT}/build/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
PEAK = 512e9
NRUN = 3
TB = 2048
L1BUDGET = 1440 * 1024


def rup(x, y):
    return ((x + y - 1) // y) * y


def cdiv(x, y):
    return (x + y - 1) // y


P = dict(Csat=24, kk=0.5, aa=2, acap=6, ovl=1.0, red=0.0, fwd=0.0, start=0.0)


def plan(M, K, N, Ns, Pk, Sm, kb, nsb):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cores = 8 * Pk * Ns * Sm
    if not (16 <= cores <= 104):
        return None
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    wasteK = Pk * Ktl / Kt - 1
    if wasteK > 0.20:
        return None
    Mblk = cdiv(Mt, Sm)
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    if nsb > Nown:
        return None
    Nbpc = cdiv(Nown, nsb)
    Nts = 8 * Ns * Nbpc * nsb
    wasteN = Nts / Nt - 1
    if wasteN > 0.20:
        return None
    cb0 = Ktl * Mblk * TB
    cb1 = 4 * kb * nsb * TB
    cb2 = 2 * Mblk * nsb * TB
    cb3 = Mblk * nsb * 4096
    cb7 = 2 * Mblk * nsb * TB
    if cb0 + cb1 + cb2 + cb3 + cb7 > L1BUDGET:
        return None
    return dict(
        cores=cores,
        real=(Mt * Kt + Kt * Nt + Mt * Nt) * TB,
        Ktl=Ktl,
        Mblk=Mblk,
        Nown=Nown,
        Nbpc=Nbpc,
        wasteK=wasteK,
        wasteN=wasteN,
    )


def cost(M, K, N, c):
    Ns, Pk, Sm, kb, nsb = c
    g = plan(M, K, N, *c)
    Kt, Nt = K // 32, N // 32
    readT = Kt * Nt / min(g["cores"], P["Csat"])
    comp_pc = g["Mblk"] * g["Nown"] * g["Ktl"]
    area = min(g["Mblk"] * nsb, P["acap"])
    compT = comp_pc / ((kb / (kb + P["kk"])) * (area / (area + P["aa"])))
    return max(readT, compT) + P["ovl"] * comp_pc / g["Nbpc"]


def feasible(M, K, N):
    Nt = N // 32
    out = []
    for Pk in range(1, 13):
        for Ns in range(1, 7):
            Nown = cdiv(cdiv(Nt, 8), Ns)
            for kb in (1, 2, 4, 8):
                for nsb in range(1, Nown + 1):
                    if plan(M, K, N, Ns, Pk, 1, kb, nsb):
                        out.append((Ns, Pk, 1, kb, nsb))
    return out


def candidates(M, K, N):
    fs = feasible(M, K, N)
    pick = min(fs, key=lambda c: cost(M, K, N, c))
    topc = sorted(fs, key=lambda c: cost(M, K, N, c))[:10]  # cost-model's own top-10
    zw = [
        c
        for c in fs
        if plan(M, K, N, *c)["wasteK"] == 0 and plan(M, K, N, *c)["wasteN"] == 0 and plan(M, K, N, *c)["cores"] >= 64
    ]
    zw = sorted(zw, key=lambda c: cost(M, K, N, c))[:10]  # lowest-cost zero-waste high-core
    seen = {}
    for c in [pick] + topc + zw:
        seen[c] = True
    return pick, list(seen)


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
    try:
        r = subprocess.run(a, env=env, capture_output=True, text=True, timeout=160, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None, "hang"
    if "PASS" not in r.stdout:
        return None, ("L1" if "beyond max L1" in r.stdout else "fail")
    return cyc(), "ok"


OUTLIERS = [
    (128, 6144, 2560),
    (128, 6144, 12288),
    (128, 12288, 6144),
    (128, 4096, 6144),
    (256, 6144, 6144),
    (256, 5120, 6144),
    (256, 4096, 9216),
    (256, 6144, 9216),
    (512, 6144, 6144),
    (512, 6144, 5120),
]
for M, K, N in OUTLIERS:
    pick, cands = candidates(M, K, N)
    results = []
    for c in cands:
        cyc_, st = run(M, K, N, c)
        if st == "hang":
            subprocess.run(["tt-smi", "-r"], capture_output=True)
            continue
        if st != "ok":
            continue
        g = plan(M, K, N, *c)
        bw = g["real"] / (cyc_ / FREQ) / PEAK * 100
        results.append((c, bw, g["wasteK"], g["wasteN"], g["cores"]))
    results.sort(key=lambda x: -x[1])
    pbw = next((bw for c, bw, _, _, _ in results if c == pick), 0)
    best = results[0]
    gain = best[1] / pbw if pbw else 0
    verdict = "HEURISTIC MISS" if gain > 1.05 else "shape floor (picker~best)"
    print(
        f"\n{M}x{K}x{N}: picker {pick} -> {pbw:.0f}%  |  BEST {best[0]} -> {best[1]:.0f}% (wK{best[2]:.0%} wN{best[3]:.0%} {best[4]}c)  = {verdict} ({gain:.2f}x)",
        flush=True,
    )
    for c, bw, wk, wn, cc in results[:5]:
        mark = " <-pick" if c == pick else ""
        print(f"    {c} {cc}c wK{wk:.0%} wN{wn:.0%} -> {bw:.0f}%{mark}", flush=True)
print("\nDONE", flush=True)
