#!/usr/bin/env python3
# Comprehensive (Ns,Pk,Sm,kb,nsb) sweep over the FLUX/LTX Regime-A shapes (M<N), ring all-gather in0
# (--unified). Purpose: build the heuristic picker (M,K,N)->(Ns,Pk,Sm,kb,nsb). Effective BW-util on REAL
# bytes (padding-heavy configs rank lower). L1-pruned + padding-pruned. SIGTERM timeout + tt-smi -r on hang.
# RESUMABLE: reloads the JSON, skips configs already measured. Run in background; monitor the .log.
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


# FLUX/LTX Regime-A shapes (M<N: in1[K,N] is the big shardable operand). Mt=1..16.
SHAPES = [
    (32, 2048, 512),
    (32, 2048, 1536),
    (32, 6144, 1536),
    (32, 2048, 2048),
    (32, 6144, 2304),
    (32, 6144, 3072),
    (32, 256, 6144),
    (32, 6144, 6144),
    (32, 6144, 9216),
    (64, 6144, 1536),
    (64, 15360, 1536),
    (64, 6144, 4608),
    (64, 4608, 6144),
    (64, 6144, 9216),
    (128, 6144, 768),
    (128, 15360, 768),
    (128, 6144, 2304),
    (128, 6144, 4608),
    (128, 2304, 6144),
    (512, 6144, 1536),
]
Pk_list = [1, 2, 3, 4, 6, 8, 12]
Ns_list = [1, 2, 3, 4, 6]
Sm_list = [1, 2, 4]
kb_list = [1, 2, 4, 8]  # kb1 RESTORED (wins Mt=8)


def plan(M, K, N, Ns, Pk, Sm, kb, nsb):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cores = 8 * Pk * Ns * Sm
    if not (16 <= cores <= 104):
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
    cb0 = Ktl * Mblk * TB
    cb1 = 4 * kb * Nsub * TB
    cb2 = 2 * Mblk * Nsub * TB
    cb3 = Mblk * Nsub * 4096
    cb7 = 2 * Mblk * Nsub * TB
    l1 = cb0 + cb1 + cb2 + cb3 + cb7
    if l1 > L1BUDGET:
        return None
    real = (Mt * Kt + Kt * Nt + Mt * Nt) * TB
    return dict(cores=cores, real=real, wasteK=wasteK, wasteN=wasteN, l1=l1)


def gen(M, K, N):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cfgs = []
    for Pk in Pk_list:
        for Ns in Ns_list:
            for Sm in Sm_list:
                if not (Mt >= Sm):
                    continue
                prod = Pk * Ns * Sm
                lo = 2 if Mt == 1 else 6  # Mt=1 can win on fewer cores; others need >=48
                if prod > 13 or prod < lo:
                    continue
                for kb in kb_list:
                    if kb > cdiv(Kt, Pk):
                        continue
                    Nown = cdiv(cdiv(Nt, 8), Ns)
                    for nsb in divisors(Nown):  # sweep ALL divisors (nsb is a 28-pt non-monotonic lever)
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
    env["TT_METAL_HOME"] = ROOT
    try:
        r = subprocess.run(a, env=env, capture_output=True, text=True, timeout=160, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None, "hang"
    if "PASS" not in r.stdout:
        return None, ("L1" if "beyond max L1" in r.stdout else "fail")
    return cyc(), "ok"


outp = f"{ROOT}/tools/mm_sweep/fluxltx_regimeA_sweep.json"
allres = []
done = set()
if os.path.exists(outp):
    try:
        allres = json.load(open(outp))
        for r in allres:
            done.add((r["M"], r["K"], r["N"], r["Ns"], r["Pk"], r["Sm"], r["kb"], r["nsb"]))
        print(f"RESUME: {len(allres)} configs already measured", flush=True)
    except:
        allres = []
        done = set()

for M, K, N in SHAPES:
    Mt = M // 32
    ridge_ai = (M * K * N * 2) / ((M * K + K * N + M * N) * 2)  # flop/byte (bf16)
    cfgs = gen(M, K, N)
    todo = [c for c in cfgs if (M, K, N, c[0], c[1], c[2], c[3], c[4]) not in done]
    print(f"\n### {M}x{K}x{N}  Mt={Mt}  ai={ridge_ai:.0f}  ({len(cfgs)} feasible, {len(todo)} to run)", flush=True)
    for Ns, Pk, Sm, kb, nsb, p in todo:
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
            Mt=Mt,
            ai=round(ridge_ai, 1),
            Ns=Ns,
            Pk=Pk,
            Sm=Sm,
            kb=kb,
            nsb=nsb,
            cores=p["cores"],
            bwp=round(bwp, 1),
            us=round(c / FREQ * 1e6, 1),
            wasteK=round(p["wasteK"], 3),
            wasteN=round(p["wasteN"], 3),
        )
        allres.append(rec)
        done.add((M, K, N, Ns, Pk, Sm, kb, nsb))
        json.dump(allres, open(outp, "w"))
    sh = [r for r in allres if (r["M"], r["K"], r["N"]) == (M, K, N)]
    sh.sort(key=lambda r: -r["bwp"])
    print(f"  TOP5:", flush=True)
    for r in sh[:5]:
        print(
            f"    ({r['Ns']},{r['Pk']},{r['Sm']}) kb{r['kb']} nsb{r['nsb']} {r['cores']}c -> {r['bwp']:.0f}%  ({r['us']:.0f}us)",
            flush=True,
        )
print("\nDONE", flush=True)
