#!/usr/bin/env python3
# GENUINE (Ns,Pk,Sm,kb,nsb) sweep per shape. Finds the true achievable top per shape (no hard-coded kb,
# no auto-nsb). Effective BW-util on REAL bytes (padding-heavy configs naturally rank lower). L1-pruned
# (predict cb budget) and padding-pruned (skip >20% K-waste). SIGTERM timeout + reset on hang, incremental.
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


def rup(x, y):
    return ((x + y - 1) // y) * y


def cdiv(x, y):
    return (x + y - 1) // y


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


SHAPES = [
    (32, 6144, 4608),
    (64, 6144, 4608),
    (128, 6144, 4608),
    (256, 6144, 4608),
    (512, 6144, 4608),
    (512, 6144, 2304),
]
Pk_list = [1, 2, 3, 4, 6, 8, 12]
Ns_list = [1, 2, 3, 4, 6]
Sm_list = [1, 2, 4]
kb_list = [2, 4, 8]  # kb1 dropped (proven compute-poor)


def plan(M, K, N, Ns, Pk, Sm, kb, nsb):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cores = 8 * Pk * Ns * Sm
    if not (24 <= cores <= 104):
        return None
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    Kts = Pk * Ktl
    wasteK = Kts / Kt - 1
    if wasteK > 0.20:
        return None  # padding-heavy configs can't win on effective BW
    Mblk = cdiv(Mt, Sm)
    Mts = Sm * Mblk
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
    Keff = Ktl // kb
    cb0 = Ktl * Mblk * TB
    cb1 = 4 * kb * Nsub * TB
    cb2 = 2 * Mblk * Nsub * TB
    cb3 = Mblk * Nsub * 4096
    cb7 = 2 * Mblk * Nsub * TB
    l1 = cb0 + cb1 + cb2 + cb3 + cb7
    if l1 > L1BUDGET:
        return None
    real = (Mt * Kt + Kt * Nt + Mt * Nt) * TB
    return dict(cores=cores, real=real, wasteK=wasteK, l1=l1)


def gen(M, K, N):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cfgs = []
    for Pk in Pk_list:
        for Ns in Ns_list:
            for Sm in Sm_list:
                if not (Mt >= Sm):
                    continue
                prod = Pk * Ns * Sm
                lo = 3 if Mt == 1 else 6  # Mt=1 can win on fewer cores; others need >=48
                if prod > 13 or prod < lo:
                    continue
                for kb in kb_list:
                    if kb > cdiv(Kt, Pk):
                        continue
                    Nown = cdiv(cdiv(Nt, 8), Ns)
                    divs = sorted(set(divisors(Nown)))
                    # nsb candidates: smallest, ~middle, largest (cap 3) to sample the non-monotonic nsb axis
                    cand = sorted(set([divs[0], divs[len(divs) // 2], divs[-1]]))
                    for nsb in cand:
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
    if n < 2:
        return None
    return min(max(v[i] for v in pc.values()) for i in range(1, n))


def run(M, K, N, Ns, Pk, Sm, kb, nsb):
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    a = [
        "timeout",
        "-s",
        "TERM",
        "90",
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
        r = subprocess.run(a, env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "hang"
    if "PASS" not in r.stdout:
        return None, ("L1" if "beyond max L1" in r.stdout else "fail")
    return cyc(), "ok"


allres = []
outp = f"{ROOT}/tools/mm_sweep/comprehensive_sweep.json"
for M, K, N in SHAPES:
    cfgs = gen(M, K, N)
    bytes_ = cfgs[0][5]["real"] if cfgs else 0
    print(f"\n### {M}x{K}x{N}  Mt={M//32}  ({len(cfgs)} feasible configs)", flush=True)
    shres = []
    for Ns, Pk, Sm, kb, nsb, p in cfgs:
        c, st = run(M, K, N, Ns, Pk, Sm, kb, nsb)
        if st == "hang":
            print(f"  HANG ({Ns},{Pk},{Sm},kb{kb},nsb{nsb}) -> reset", flush=True)
            subprocess.run(["tt-smi", "-r"], capture_output=True)
            continue
        if st != "ok":
            continue
        bwp = p["real"] / (c / FREQ) / PEAK * 100
        rec = dict(
            M=M,
            K=K,
            N=N,
            Mt=M // 32,
            Ns=Ns,
            Pk=Pk,
            Sm=Sm,
            kb=kb,
            nsb=nsb,
            cores=p["cores"],
            bwp=bwp,
            us=c / FREQ * 1e6,
            wasteK=p["wasteK"],
        )
        shres.append(rec)
        allres.append(rec)
        json.dump(allres, open(outp, "w"))
    shres.sort(key=lambda r: -r["bwp"])
    print(f"  TOP5 for {M}x{K}x{N}:", flush=True)
    for r in shres[:5]:
        print(
            f"    ({r['Ns']},{r['Pk']},{r['Sm']}) kb{r['kb']} nsb{r['nsb']} {r['cores']}c -> {r['bwp']:.0f}%  ({r['us']:.0f}us)",
            flush=True,
        )
print("\nDONE", flush=True)
