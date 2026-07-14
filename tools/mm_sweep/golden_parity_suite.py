#!/usr/bin/env python3
# Step 1 of the regime_a_matmul productization: FREEZE a compact golden parity suite from the C++
# prototype (test_regime_a_mm --unified). This becomes the ORACLE the ttnn op must reproduce within 5%.
#
# For each parity shape we record: exact (Ns,Pk,Sm,kb,nsb), kernel cycles + kernel time, effective BW
# (logical bytes) and delivered BW (padded physical bytes) separately, per-RISC BRISC/NCRISC/TRISC
# max-core cycles (BRISC=in1 reader, NCRISC=in0 ring/reduce/writer, TRISC=compute), core count, L1, and
# the physical in1 shard-shape assumption ([Kt_pad, Nt_pad/8] tiles across 8 DRAM banks).
#
# Table shapes use the oracle-best config from the 3262-config sweep (picker_table.py). Off-table shapes
# measure a small candidate set (top-N by v2 cost) and freeze the BEST-MEASURED as golden.
#
# Safe: one config at a time, SIGTERM timeout, tt-smi -r on hang. Run from repo root cwd is handled here.
import csv, os, subprocess, json, importlib.util, io, contextlib
from collections import defaultdict

ROOT = "/localdev/cglagovich/tt-metal"
BIN = f"{ROOT}/build/test/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm"
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
PEAK = 512e9
TB = 2048
L1BUDGET = 1440 * 1024
NRUN = 6  # best-of over runs 1.. (run 0 is warmup)


def cdiv(a, b):
    return (a + b - 1) // b


def rup(x, y):
    return ((x + y - 1) // y) * y


# ---- load picker table + v2 fallback ----
def _load(name):
    spec = importlib.util.spec_from_file_location(name, f"{ROOT}/tools/mm_sweep/{name}.py")
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        cwd = os.getcwd()
        os.chdir(f"{ROOT}/tools/mm_sweep")
        try:
            spec.loader.exec_module(m)
        finally:
            os.chdir(cwd)
    return m


pt = _load("picker_table")
pv = _load("picker_v2")


def geo_plan(M, K, N, Ns, Pk, Sm, kb, nsb):
    """Padded tile geometry + byte accounting (mirrors sweep plan())."""
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    cores = 8 * Pk * Ns * Sm
    Ktl = rup(cdiv(Kt, Pk), kb * 8)
    Kts = Pk * Ktl
    Mblk = cdiv(Mt, Sm)
    Mts = Sm * Mblk
    Nband = cdiv(Nt, 8)
    Nown = cdiv(Nband, Ns)
    Nsub = nsb
    Nbpc = cdiv(Nown, Nsub)
    Nowns = Nbpc * Nsub
    Nbands = Ns * Nowns
    Nts = 8 * Nbands
    cb0 = Ktl * Mblk * TB
    cb1 = 4 * kb * Nsub * TB
    cb2 = 2 * Mblk * Nsub * TB
    cb3 = Mblk * Nsub * 4096
    cb7 = 2 * Mblk * Nsub * TB
    l1 = cb0 + cb1 + cb2 + cb3 + cb7
    real = (Mt * Kt + Kt * Nt + Mt * Nt) * TB
    phys = (Mts * Kts + Kts * Nts + Mts * Nts) * TB
    return dict(
        cores=cores,
        real=real,
        phys=phys,
        l1=l1,
        Kt=Kt,
        Nt=Nt,
        Mt=Mt,
        Kts=Kts,
        Nts=Nts,
        Mts=Mts,
        wasteK=Kts / Kt - 1,
        wasteN=Nts / Nt - 1,
        in1_shard_tiles=(Kts, Nts // 8),  # per-bank in1 shard [Kt_pad, Nt_pad/8] tiles
    )


def parse_profiler():
    """Return (total_cyc, {risc: cyc}) using max-across-cores per run, min over runs 1.."""
    try:
        rows = list(csv.reader(open(CSV)))
    except Exception:
        return None, None
    # per (core, risc) list of (marker, cyc)
    ev = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12:
            continue
        zname = row[10].strip()
        if not zname.endswith("-KERNEL"):
            continue
        risc = zname.replace("-KERNEL", "")  # BRISC / NCRISC / TRISC / TRISC0..
        core = (row[1], row[2], row[3])
        ev[(core, risc)].append((row[11].strip(), int(row[5])))
    # durations per (core,risc)
    dur = {}
    for k, l in ev.items():
        ds, st = [], None
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                ds.append(c - st)
                st = None
        dur[k] = ds
    if not dur:
        return None, None
    nruns = min((len(v) for v in dur.values()), default=0)
    if nruns < 2:
        return None, None
    # total kernel = max across ALL (core,risc) per run, min over runs 1..
    total = min(max(v[i] for v in dur.values()) for i in range(1, nruns))
    # per-risc = max across cores of that risc, min over runs 1..
    per = {}
    riscs = set(r for (_, r) in dur.keys())
    for r in riscs:
        vs = [v for (c, rr), v in dur.items() if rr == r]
        per[r] = min(max(v[i] for v in vs) for i in range(1, nruns))
    return total, per


def run(M, K, N, Ns, Pk, Sm, kb, nsb):
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_HOME"] = ROOT
    env["ARCH_NAME"] = "blackhole"
    env["PYTHONPATH"] = ROOT
    env.setdefault("TT_METAL_CACHE", "/localdev/cglagovich/tt_metal_cache")
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
    total, per = parse_profiler()
    if total is None:
        return None, "noprof"
    return (total, per), "ok"


def candidates(M, K, N, topn=4):
    """Off-table: top-N feasible configs by v2 cost."""
    fs = pv.feasible(M, K, N)
    fs = sorted(fs, key=lambda c: pv.cost(M, K, N, c, pv.bestP))
    return fs[:topn]


# ---- parity suite ----
PARITY = [
    ("Mt1  base", 32, 6144, 4608),
    ("Mt2  base", 64, 6144, 4608),
    ("Mt4  base", 128, 6144, 4608),
    ("Mt8  base", 256, 6144, 4608),
    ("Mt4  small-N", 128, 6144, 768),
    ("Mt1  non-divis", 32, 6080, 4640),
]

results = []
for label, M, K, N in PARITY:
    key = (M, K, N)
    if key in pt.PICKER_TABLE:
        cfgs = [pt.PICKER_TABLE[key]]
        src = "table-oracle"
    else:
        cfgs = candidates(M, K, N)
        src = "v2-candidates"
    print(f"\n### {label}  {M}x{K}x{N}  ({src}, {len(cfgs)} cfg)", flush=True)
    shape_runs = []
    for Ns, Pk, Sm, kb, nsb in cfgs:
        g = geo_plan(M, K, N, Ns, Pk, Sm, kb, nsb)
        res, st = run(M, K, N, Ns, Pk, Sm, kb, nsb)
        if st == "hang":
            print(f"  ({Ns},{Pk},{Sm},{kb},{nsb}) HANG -> reset", flush=True)
            subprocess.run(["tt-smi", "-r"], capture_output=True)
            continue
        if st != "ok":
            print(f"  ({Ns},{Pk},{Sm},{kb},{nsb}) {st}", flush=True)
            continue
        total, per = res
        us = total / FREQ * 1e6
        eff = g["real"] / (total / FREQ)
        deliv = g["phys"] / (total / FREQ)
        rec = dict(
            label=label,
            M=M,
            K=K,
            N=N,
            Mt=M // 32,
            Ns=Ns,
            Pk=Pk,
            Sm=Sm,
            kb=kb,
            nsb=nsb,
            cores=g["cores"],
            cyc=total,
            us=round(us, 2),
            eff_bw=round(eff / 1e9, 1),
            eff_pct=round(eff / PEAK * 100, 1),
            deliv_bw=round(deliv / 1e9, 1),
            deliv_pct=round(deliv / PEAK * 100, 1),
            per_risc={k: v for k, v in sorted(per.items())},
            l1_kb=round(g["l1"] / 1024, 1),
            wasteK_pct=round(g["wasteK"] * 100, 1),
            wasteN_pct=round(g["wasteN"] * 100, 1),
            in1_shard_tiles=list(g["in1_shard_tiles"]),
        )
        shape_runs.append(rec)
        pr = " ".join(f"{k}={v}" for k, v in rec["per_risc"].items())
        print(
            f"  ({Ns},{Pk},{Sm},{kb},{nsb}) {g['cores']}c  {us:.1f}us  eff={rec['eff_pct']}%  "
            f"deliv={rec['deliv_pct']}%  [{pr}]",
            flush=True,
        )
    if shape_runs:
        best = max(shape_runs, key=lambda r: r["eff_bw"])
        best["GOLDEN"] = True
        results.append(dict(label=label, shape=[M, K, N], golden=best, all_candidates=shape_runs))
        print(
            f"  => GOLDEN ({best['Ns']},{best['Pk']},{best['Sm']},{best['kb']},{best['nsb']})  "
            f"{best['us']}us  eff={best['eff_pct']}%  deliv={best['deliv_pct']}%",
            flush=True,
        )

out = f"{ROOT}/tools/mm_sweep/golden_parity_suite.json"
json.dump(results, open(out, "w"), indent=2)
print(f"\nWROTE {out}")
