#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize a run_bench.sh output dir into a human-readable report + machine CSV.

Joins three sources:
  phase_results.csv  per-phase wall/user/sys/%CPU/peak-RSS/rc + parsed JIT telemetry
  marks.csv          phase START/END epochs (the integration windows for the sampler)
  sampler.csv        per-tick CPU-seconds split into tree/compiler/python over time

For each phase it integrates the sampler over the phase window to get mean/peak cores,
the compiler-vs-host CPU split, and core utilization (of nproc). WARMUP is further split
into its collect vs compile sub-windows using the plugin's reported compile_wall.

Then it builds the north-star comparison: COLD total wall vs PRECOMPILE total wall
(probe_real + probe_mock + warmup + warm), per ccache condition, with speedups; aggregated
across repeats by median.

Usage: summarize.py <bench_out_dir> [nproc]
"""
import csv
import os
import statistics
import sys

OUT = sys.argv[1]
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 8)
# the container is cgroup-limited; nproc (8) is the real parallelism ceiling, not os.cpu_count (32)
try:
    NPROC = int(open(os.path.join(OUT, "nproc.txt")).read().strip())
except Exception:
    pass


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_extra(s):
    d = {}
    if not s or s == "none":
        return d
    for kv in s.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
    return d


phases = load_csv(os.path.join(OUT, "phase_results.csv"))
marks = load_csv(os.path.join(OUT, "marks.csv"))
sampler = []
try:
    sampler = load_csv(os.path.join(OUT, "sampler.csv"))
    for r in sampler:
        for k in r:
            r[k] = float(r[k])
except FileNotFoundError:
    pass

# windows[(label,phase)] = (start_epoch, end_epoch)
win = {}
starts = {}
for m in marks:
    key = (m["label"], m["phase"])
    if m["event"] == "START":
        starts[key] = float(m["epoch"])
    elif m["event"] == "END" and key in starts:
        win[key] = (starts[key], float(m["epoch"]))


def integrate(t0, t1):
    """Integrate sampler over [t0,t1] -> dict of cpu-seconds + peak cores."""
    tree = comp = py = oth = 0.0
    peak = peak_comp = 0.0
    rows = [r for r in sampler if t0 <= r["epoch"] <= t1]
    for r in rows:
        tree += r["tree_cpu_s"]
        comp += r["compiler_cpu_s"]
        py += r["python_cpu_s"]
        oth += r["other_cpu_s"]
        dt = r["dt"] or 1e9
        peak = max(peak, r["tree_cpu_s"] / dt)
        peak_comp = max(peak_comp, r["compiler_cpu_s"] / dt)
    dur = max(t1 - t0, 1e-9)
    return {
        "cpu_s": tree,
        "comp_s": comp,
        "py_s": py,
        "oth_s": oth,
        "mean_cores": tree / dur,
        "peak_cores": peak,
        "mean_comp_cores": comp / dur,
        "peak_comp_cores": peak_comp,
        "comp_share": (comp / tree * 100.0) if tree > 0 else 0.0,
        "n": len(rows),
    }


# attach sampler-integrated metrics to each phase row
for p in phases:
    p["wall_s"] = float(p["wall_s"])
    p["user_s"] = float(p["user_s"])
    p["sys_s"] = float(p["sys_s"])
    p["cpu_pct"] = float(p["cpu_pct"])
    p["maxrss_mb"] = float(p["maxrss_mb"])
    p["ex"] = parse_extra(p["extra"])
    key = (p["label"], p["phase"])
    if key in win:
        p["w0"], p["w1"] = win[key]
        p["s"] = integrate(p["w0"], p["w1"])
    else:
        p["w0"] = p["w1"] = None
        p["s"] = None


def fmt_s(x):
    return f"{x:.1f}s" if x < 60 else f"{int(x // 60)}m{int(x % 60):02d}s"


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0


# group phases by config (method,ccache,ccstate,phase) across repeats
from collections import defaultdict

grp = defaultdict(list)
for p in phases:
    grp[(p["method"], p["ccache"], p["ccstate"], p["phase"])].append(p)

REP = []  # report lines


def w(s=""):
    REP.append(s)


CONDS = [
    ("off", "na", "ccache OFF (default/CI)", "off"),
    ("on", "deleted", "ccache ON, deleted (cold compiler cache)", "on/del"),
    ("on", "warm", "ccache ON, warm (reused compiler cache)", "on/warm"),
]

w("=" * 100)
w(f"PRECOMPILE BENCHMARK SUMMARY   out={OUT}   nproc(parallelism ceiling)={NPROC}")
try:
    w("  " + open(os.path.join(OUT, "run.log")).readline().strip())
except Exception:
    pass
w("=" * 100)

# ---------------- NORTH STAR ----------------
w("")
w("################  NORTH STAR: total end-to-end wall  (COLD inline  vs  PRECOMPILE warmup+warm)  ################")
w("")
hdr = f"{'ccache condition':<42} {'COLD':>9} {'PRECOMPILE':>26} {'speedup':>9}"
w(hdr)
w("-" * len(hdr))
northstar_rows = []
for cc, ccst, label, short in CONDS:
    cold = grp.get(("cold", cc, ccst, "cold"), [])
    cold_wall = med([x["wall_s"] for x in cold])
    pc_total_per_rep = defaultdict(float)
    pc_phase_meds = {}
    for ph in ("probe_real", "probe_mock", "warmup", "warm"):
        rows = grp.get(("precompile", cc, ccst, ph), [])
        pc_phase_meds[ph] = med([x["wall_s"] for x in rows])
        for x in rows:
            pc_total_per_rep[x["repeat"]] += x["wall_s"]
    pc_total = med(list(pc_total_per_rep.values())) if pc_total_per_rep else 0.0
    sp = (cold_wall / pc_total) if pc_total > 0 else 0.0
    breakdown = (
        f"{fmt_s(pc_total)} (pr{pc_phase_meds['probe_real']:.0f}+mk{pc_phase_meds['probe_mock']:.0f}"
        f"+wu{pc_phase_meds['warmup']:.0f}+wm{pc_phase_meds['warm']:.0f})"
    )
    w(f"{label:<42} {fmt_s(cold_wall):>9} {breakdown:>26} {sp:>8.2f}x")
    northstar_rows.append((label, cold_wall, pc_total, pc_phase_meds, sp))
w("")
w("  (pr=probe_real  mk=probe_mock  wu=warmup[collect+compile]  wm=warm-run.  speedup = COLD/PRECOMPILE;")
w("   <1.00x means precompile is SLOWER for this suite size — warmup overhead not yet amortized.)")


# ---------------- PER-PHASE DETAIL ----------------
def phase_line(method, cc, ccst, ph):
    rows = grp.get((method, cc, ccst, ph), [])
    if not rows:
        return None
    wall = med([x["wall_s"] for x in rows])
    cpu = med([x["cpu_pct"] for x in rows])
    rss = med([x["maxrss_mb"] for x in rows])
    user = med([x["user_s"] for x in rows])
    sysc = med([x["sys_s"] for x in rows])
    ss = [x["s"] for x in rows if x["s"]]
    mc = med([s["mean_cores"] for s in ss]) if ss else 0.0
    pk = med([s["peak_cores"] for s in ss]) if ss else 0.0
    cs = med([s["comp_share"] for s in ss]) if ss else 0.0
    comp_s = med([s["comp_s"] for s in ss]) if ss else 0.0
    # AUTHORITATIVE parallelism/utilization from getrusage (exact, counts every reaped child):
    ru_cores = cpu / 100.0  # %CPU/100 == mean cores busy over the phase
    ru_util = cpu / NPROC  # % of the nproc ceiling
    cpu_s = user + sysc  # exact total CPU-seconds for the phase tree
    util = mc / NPROC * 100.0  # sampler-based (lower bound; short procs slip 0.25s ticks)
    ex = rows[-1]["ex"]
    return dict(
        wall=wall,
        cpu=cpu,
        rss=rss,
        mc=mc,
        pk=pk,
        cs=cs,
        comp_s=comp_s,
        util=util,
        ru_cores=ru_cores,
        ru_util=ru_util,
        cpu_s=cpu_s,
        ex=ex,
    )


w("")
w("################  PER-PHASE DETAIL (median across repeats)  ################")
w("  cores/util/CPU-s = AUTHORITATIVE (getrusage, counts every reaped compiler child).")
w("  peak/comp% = sampler (0.25s /proc): peak cores is real; comp% is a LOWER BOUND on the")
w("  compiler's CPU share (short gcc procs slip between ticks), i.e. true share is >= shown.")
colhdr = (
    f"{'phase':<12} {'wall':>8} {'%CPU':>6} {'cores':>6} {'util':>6} {'CPU-s':>7} "
    f"{'pkCor':>6} {'comp%':>6} {'pkRSS':>7}  notes"
)


def emit_phase(method, cc, ccst, ph, disp):
    pl = phase_line(method, cc, ccst, ph)
    if not pl:
        return
    note = ""
    if ph == "cold":
        note = f"jit:{pl['ex'].get('jit_jitted','?')} jitted/{pl['ex'].get('jit_total','?')} (hit {pl['ex'].get('jit_hitpct','?')}%)"
    elif ph == "warmup":
        note = (
            f"collect+compile  unique={pl['ex'].get('unique','?')} compiled={pl['ex'].get('compiled','?')} "
            f"compile_wall={pl['ex'].get('compile_wall','?')}s"
        )
    elif ph == "warm":
        note = (
            f"jit:{pl['ex'].get('jit_hits','?')}/{pl['ex'].get('jit_total','?')} hits "
            f"({pl['ex'].get('jit_hitpct','?')}%) jitted={pl['ex'].get('jit_jitted','?')}"
        )
    w(
        f"           {disp:<12} {fmt_s(pl['wall']):>8} {pl['cpu']:>5.0f}% {pl['ru_cores']:>6.1f} "
        f"{pl['ru_util']:>5.0f}% {pl['cpu_s']:>6.0f}s {pl['pk']:>6.1f} {pl['cs']:>5.0f}% {pl['rss']:>6.0f}MB  {note}"
    )


for cc, ccst, label, short in CONDS:
    w("")
    w(f"=== {label} ===")
    w(f"  [COLD]   {colhdr}")
    emit_phase("cold", cc, ccst, "cold", "cold")
    w(f"  [PRECOMPILE]")
    for ph in ("probe_real", "probe_mock", "warmup", "warm"):
        emit_phase("precompile", cc, ccst, ph, ph)

# ---------------- WARMUP collect vs compile sub-split ----------------
w("")
w("################  WARMUP sub-split: collect vs compile (the two phase-1 subphases)  ################")
w(f"{'condition':<42} {'collect':>20} {'compile':>34}")
w(f"{'':42} {'wall  cores util':>20} {'wall  cores util  comp%':>34}")
for cc, ccst, label, short in CONDS:
    rows = grp.get(("precompile", cc, ccst, "warmup"), [])
    rows = [r for r in rows if r["s"] and r["ex"].get("compile_wall", "?") != "?"]
    if not rows:
        continue
    col_w = comp_w = []
    cstats = []
    mstats = []
    for r in rows:
        cw = float(r["ex"]["compile_wall"])
        t0, t1 = r["w0"], r["w1"]
        comp0 = max(t1 - cw, t0)
        cint = integrate(t0, comp0)  # collect window
        mint = integrate(comp0, t1)  # compile window
        cstats.append((t1 - cw - t0, cint["mean_cores"]))
        mstats.append((cw, mint["mean_cores"], mint["comp_share"]))
    coll_wall = med([a for a, _ in cstats])
    coll_cores = med([b for _, b in cstats])
    comp_wall = med([a for a, _, _ in mstats])
    comp_cores = med([b for _, b, _ in mstats])
    comp_share = med([c for _, _, c in mstats])
    w(
        f"{label:<42} "
        f"{fmt_s(coll_wall):>6} {coll_cores:>4.1f} {coll_cores/NPROC*100:>3.0f}%   "
        f"{fmt_s(comp_wall):>6} {comp_cores:>4.1f} {comp_cores/NPROC*100:>3.0f}% {comp_share:>4.0f}%"
    )
w("  (compile = last <compile_wall>s of the warmup window; collect = the rest — both walls authoritative.")
w("   cores/util here are sampler-based LOWER BOUNDS; the whole-warmup getrusage cores in the per-phase")
w("   table above (~6-7 cores @ ccache off/del) is exact and is dominated by this compile burst.")
w("   collect is single-process meta shape-prop -> low cores; compile is the nproc-parallel JIT burst.)")

# ---------------- JIT TELEMETRY ----------------
w("")
w("################  JIT TELEMETRY (kernels jitted vs served from cache)  ################")
w(f"{'run':<46} {'total':>7} {'jitted':>7} {'cached/hits':>12} {'hit%':>6}")


def tel(method, cc, ccst, ph, label):
    rows = grp.get((method, cc, ccst, ph), [])
    rows = [r for r in rows if r["ex"].get("jit_total")]
    if not rows:
        return
    r = rows[-1]["ex"]
    w(
        f"{label:<46} {r.get('jit_total','?'):>7} {r.get('jit_jitted','?'):>7} {r.get('jit_hits','?'):>12} {r.get('jit_hitpct','?'):>6}"
    )


for cc, ccst, label, short in CONDS:
    tel("cold", cc, ccst, "cold", f"COLD  [{short}]")
    tel("precompile", cc, ccst, "warm", f"PRECOMPILE warm  [{short}]")

# ---------------- WHERE TIME GOES IN THE COLD PASS ----------------
w("")
w("################  WHERE THE TIME GOES IN THE COLD PASS (inline compile attribution)  ################")
w("  Method: same 75 tests run cold (compile inline) vs warm (cache hit, ~0 compile). The delta is the")
w("  inline-compile cost smeared across the cold run; warm wall ~= pure framework+execution floor.")
w("  Δ are exact (getrusage). NB warm still cold-compiles ~93 kernels (81% hit), so Δ is a LOWER bound")
w("  on the full inline-compile cost (the warmup's compile phase below shows the full parallel-compile CPU).")
w(f"{'ccache condition':<42} {'cold wall':>10} {'warm wall':>10} {'Δwall(compile)':>15} {'ΔCPU-s(compile)':>16}")
for cc, ccst, label, short in CONDS:
    cold = phase_line("cold", cc, ccst, "cold")
    warm = phase_line("precompile", cc, ccst, "warm")
    if not cold or not warm:
        continue
    dwall = cold["wall"] - warm["wall"]
    dcpu = cold["cpu_s"] - warm["cpu_s"]
    w(f"{label:<42} {fmt_s(cold['wall']):>10} {fmt_s(warm['wall']):>10} {fmt_s(dwall):>15} {dcpu:>14.0f}s")
w("")
w("  Full inline-compile CPU (cold pass, getrusage total CPU-s) and how the warmup re-spends it in parallel:")
w(f"{'ccache condition':<42} {'cold CPU-s':>11} {'warmup CPU-s':>13} {'cold cores':>11} {'warmup cores':>13}")
for cc, ccst, label, short in CONDS:
    cold = phase_line("cold", cc, ccst, "cold")
    wu = phase_line("precompile", cc, ccst, "warmup")
    if not cold or not wu:
        continue
    w(f"{label:<42} {cold['cpu_s']:>10.0f}s {wu['cpu_s']:>12.0f}s {cold['ru_cores']:>10.1f} {wu['ru_cores']:>12.1f}")

report = "\n".join(REP)
print(report)
with open(os.path.join(OUT, "SUMMARY.txt"), "w") as f:
    f.write(report + "\n")
print(f"\n[written: {os.path.join(OUT, 'SUMMARY.txt')}]")
