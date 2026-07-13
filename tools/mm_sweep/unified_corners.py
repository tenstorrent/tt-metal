#!/usr/bin/env python3
# NON-DIVISIBLE corner sweep: configs impossible without first-class padding (odd Sm, non-divisor Pk,
# grid-fill). Reports EFFECTIVE BW-util on REAL bytes (honest: padding waste lowers it), the DELIVERED
# BW on padded bytes (engine quality), and padding waste%. Compares to the divisible best per shape.
import csv, os, subprocess, sys, json
from collections import defaultdict

ROOT = "/localdev/cglagovich/tt-metal"
BIN = f"{ROOT}/build/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
PEAK = 512e9
NRUN = 6
TB = 2048


def rup(x, y):
    return ((x + y - 1) // y) * y


def cdiv(x, y):
    return (x + y - 1) // y


def pad(M, K, N, Ns, Pk, Sm, kb, nsb):
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    Kts = Pk * Ktl
    Mblk = cdiv(Mt, Sm)
    Mts = Sm * Mblk
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    Nsub = nsb if nsb else Nown
    Nbpc = cdiv(Nown, Nsub)
    Nowns = Nbpc * Nsub
    Nbands = Ns * Nowns
    Nts = 8 * Nbands
    cores = 8 * Pk * Ns * Sm
    # L1 cb budget (bytes)
    Keff = Ktl // kb
    cb0 = Ktl * Mblk * TB
    cb1 = 4 * kb * Nsub * TB
    cb2 = 2 * Mblk * Nsub * TB
    cb3 = Mblk * Nsub * 4096
    cb7 = 2 * Mblk * Nsub * TB
    l1 = cb0 + cb1 + cb2 + cb3 + cb7
    real = (Mt * Kt + Kt * Nt + Mt * Nt) * TB
    padb = (Mts * Kts + Kts * Nts + Mts * Nts) * TB
    return dict(
        Mts=Mts,
        Kts=Kts,
        Nts=Nts,
        cores=cores,
        l1=l1,
        real=real,
        padb=padb,
        waste=100 * (padb - real) / real,
        Keff=Keff,
        Mblk=Mblk,
        ok=cores <= 110 and l1 <= 1450 * 1024,
    )


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
        r = subprocess.run(a, env=env, capture_output=True, text=True, timeout=140)
    except subprocess.TimeoutExpired:
        return None, "hang"
    if "PASS" not in r.stdout:
        if "beyond max L1" in r.stdout:
            return None, "L1OOM"
        return None, "fail(" + (r.stdout.strip().splitlines()[-1][:40] if r.stdout.strip() else "?") + ")"
    return cyc(), "ok"


# (label, M,K,N, Ns,Pk,Sm,kb,nsb, divisible-best-BW% for reference)
CORNERS = [
    # odd M-split Sm=3 (impossible divisibly for Mt=16/8): deeper M-parallel + shallower reduction
    ("Sm3 Pk4", 512, 6144, 4608, 1, 4, 3, 6, 3, 52),
    ("Sm3 Pk4", 512, 6144, 2304, 1, 4, 3, 6, 3, 44),
    ("Sm3 Pk3", 512, 3072, 6144, 1, 3, 3, 4, 6, 52),
    ("Sm3 Pk4", 256, 6144, 4608, 1, 4, 3, 6, 6, 61),
    ("Sm3 Pk3", 256, 6144, 4608, 1, 3, 3, 8, 6, 61),
    # non-divisor Pk (reduction-depth tuning), Sm2
    ("Pk5 Sm2", 512, 6144, 4608, 1, 5, 2, 1, 6, 52),
    ("Pk5 Sm2", 512, 6144, 2304, 1, 5, 2, 1, 3, 44),
    ("Pk7 Sm1", 512, 6144, 4608, 1, 7, 1, 1, 3, 52),  # 56c, Sm1 M_block16 (may OOM)
    # grid-fill Mt=1: use >96 cores via non-divisor Pk
    ("Pk13", 32, 6144, 4608, 1, 13, 1, 1, 9, 82),
    ("Pk11", 32, 6144, 4608, 1, 11, 1, 1, 9, 82),
    ("Pk10", 32, 6144, 4608, 1, 10, 1, 2, 9, 82),
    # odd Sm for Mt=4
    ("Sm3 Pk4", 128, 6144, 4608, 1, 4, 3, 6, 6, 64),
]
res = []
print(
    f"{'label':>9} {'shape':>16} {'cfg(Ns,Pk,Sm,kb,nsb)':>20} {'c':>4} {'wst%':>4} {'us':>6} {'eff%':>5} {'dlv%':>5} {'divbest%':>7} {'v.div':>6}"
)
for lab, M, K, N, Ns, Pk, Sm, kb, nsb, divb in CORNERS:
    p = pad(M, K, N, Ns, Pk, Sm, kb, nsb)
    if not p["ok"]:
        why = "cores>110" if p["cores"] > 110 else f"L1={p['l1']//1024}KB"
        print(f"{lab:>9} {f'{M}x{K}x{N}':>16} {f'({Ns},{Pk},{Sm},{kb},{nsb})':>20} {p['cores']:>4} skip:{why}")
        continue
    c, st = run(M, K, N, Ns, Pk, Sm, kb, nsb)
    if st == "hang":
        print(f"{lab:>9} {f'{M}x{K}x{N}':>16} HANG -> reset")
        subprocess.run(["tt-smi", "-r"], capture_output=True)
        continue
    if st != "ok":
        print(f"{lab:>9} {f'{M}x{K}x{N}':>16} {f'({Ns},{Pk},{Sm},{kb},{nsb})':>20} {p['cores']:>4} {st}")
        continue
    t = c / FREQ
    eff = p["real"] / t / PEAK * 100
    dlv = p["padb"] / t / PEAK * 100
    res.append(
        dict(
            lab=lab,
            M=M,
            K=K,
            N=N,
            Ns=Ns,
            Pk=Pk,
            Sm=Sm,
            kb=kb,
            nsb=nsb,
            cores=p["cores"],
            waste=p["waste"],
            us=t * 1e6,
            eff=eff,
            dlv=dlv,
            divb=divb,
        )
    )
    print(
        f"{lab:>9} {f'{M}x{K}x{N}':>16} {f'({Ns},{Pk},{Sm},{kb},{nsb})':>20} {p['cores']:>4} {p['waste']:>4.0f} {t*1e6:>6.1f} {eff:>5.1f} {dlv:>5.1f} {divb:>7} {eff-divb:>+5.1f}",
        flush=True,
    )
    json.dump(res, open(f"{ROOT}/tools/mm_sweep/unified_corners.json", "w"), indent=1)
print("DONE", flush=True)
