#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run -> per launch (node, repetition k, op) the op-level columns of the ops report
(duration, per-thread kernel durations, compute CB wait front / reserve back, NoC and DRAM utilisation, compute kernel source)
and the Tensix counters of the raw device log (timer 9090), as fractions of the counter window per core, median over cores:
U unpacker 0 busy, F FPU, P packer busy, S0/S1/S2 semaphore waits of thread 0/1/2 (nonzero + nonfull), T0/T1/T2 thread stalls.
Printed as "PRO <arm> <run> <case> k<k> <op> <src> cores=.. dur=.. ..."; the first repetition (k0) is the warm-up."""
import csv, glob, os, re, statistics as st, sys
from collections import defaultdict

out, arm, run = sys.argv[1:4]
fs = sorted(glob.glob(os.path.join(out, "reports", "*", "ops_perf_results_*.csv")))
if not fs:
    print(f"PRO NOREPORT {arm} {run}")
    sys.exit(0)
rows = list(csv.DictReader(open(fs[-1])))
PAT = re.compile(r'"counter type":"([^"]+)".*?"ref cnt":(\d+).*?"value":(\d+)')
cnt = defaultdict(lambda: defaultdict(dict))  # call -> core -> name -> (ref, value)
raw = os.path.join(out, ".logs", "profile_log_device.csv")
if os.path.exists(raw):
    with open(raw, errors="replace") as f:
        f.readline()
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        ix, iy, itid, irun, imeta = (hdr.index(c) for c in ("core_x", "core_y", "timer_id", "run host ID", "meta data"))
        for row in r:
            if len(row) <= imeta or row[itid].strip() != "9090":
                continue
            m = PAT.search(",".join(row[imeta:]))
            if m:
                cnt[int(row[irun])][(row[ix], row[iy])][m.group(1)] = (int(m.group(2)), int(m.group(3)))


def med(call, names):
    per = []
    for d in cnt.get(call, {}).values():
        ref = max((v[0] for v in d.values()), default=0)
        if ref and all(n in d for n in names):
            per.append(100.0 * sum(d[n][1] for n in names) / ref)
    return f"{st.median(per):.1f}" if per else "na"


def col(r, name):
    try:
        return f"{float(r[name]):.0f}"
    except (ValueError, KeyError):
        return "na"


cur = None
for r in rows:
    if r["OP TYPE"] == "signpost":
        node, _, k = r["OP CODE"].rpartition("#")
        cur = (node.split("::")[-1].replace(" ", ""), int(k)) if k.isdigit() else None
        continue
    if cur is None:
        continue
    call = int(float(r["GLOBAL CALL COUNT"])) if r.get("GLOBAL CALL COUNT") else -1
    src = ",".join(sorted(set(re.findall(r"([\w.]+\.(?:cpp|hpp))", r.get("COMPUTE KERNEL SOURCE") or "")))) or "-"
    f = [("dur", "DEVICE KERNEL DURATION [ns]"), ("t0", "DEVICE TRISC0 KERNEL DURATION [ns]"),
         ("t1", "DEVICE TRISC1 KERNEL DURATION [ns]"), ("t2", "DEVICE TRISC2 KERNEL DURATION [ns]"),
         ("br", "DEVICE BRISC KERNEL DURATION [ns]"), ("nc", "DEVICE NCRISC KERNEL DURATION [ns]"),
         ("wf", "DEVICE COMPUTE CB WAIT FRONT [ns]"), ("rb", "DEVICE COMPUTE CB RESERVE BACK [ns]"),
         ("noc", "NOC UTIL (%)"), ("dram", "DRAM BW UTIL (%)"), ("cores", "CORE COUNT")]
    s = " ".join(f"{a}={col(r, b)}" for a, b in f)
    c = (f"U={med(call, ['UNPACK0_BUSY_THREAD0'])} F={med(call, ['FPU_COUNTER'])} P={med(call, ['PACKER_BUSY'])} "
         + " ".join(f"S{i}={med(call, [f'WAITING_FOR_NONZERO_SEM_{i}', f'WAITING_FOR_NONFULL_SEM_{i}'])}" for i in range(3)) + " "
         + " ".join(f"T{i}={med(call, [f'THREAD_STALLS_{i}'])}" for i in range(3)))
    print(f"PRO {arm} {run} {cur[0]} k{cur[1]} {r['OP CODE']} {src} {s} {c}")
