#!/usr/bin/env python3
"""Analyze conv_bench_data_unify.csv: per-conv main/sbm/trm + deltas + aggregates + v2 compare."""
import csv, sys, statistics

UNIFY = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/localdev/wransom/tt-metal/.claude/worktrees/agent-a48fa14207415d0cb/conv_bench_data_unify.csv"
)
V2 = "/tmp/v2_bh.csv"


def load(path):
    by = {}  # label -> {mode: row}
    order = []
    with open(path) as f:
        for r in csv.DictReader(f):
            lbl = r["label"]
            if lbl not in by:
                by[lbl] = {}
                order.append(lbl)
            by[lbl][r["mode"]] = r  # last wins (latest run)
    return by, order


def med(r):
    try:
        return float(r["median_ns"])
    except (ValueError, TypeError, KeyError):
        return None


def pct(new, base):
    if new is None or base is None or base == 0:
        return None
    return 100.0 * (new - base) / base


uni, order = load(UNIFY)
try:
    v2, _ = load(V2)
except FileNotFoundError:
    v2 = {}

print(
    f"{'label':32s} | {'main_ns':>9s} | {'sbm_ns':>9s} | {'trm_ns':>9s} | {'sbm%':>7s} | {'trm%':>7s} | TRM? | status"
)
print("-" * 110)
sbm_deltas, trm_deltas, trm_engaged = [], [], []
didnotfit = []
for lbl in order:
    m = uni[lbl]
    rmain, rsbm, rtrm = m.get("main"), m.get("helper_sbm"), m.get("helper_trm")
    # did-not-fit if any mode non-ok
    statuses = {k: (m[k]["status"] if k in m else "MISSING") for k in ("main", "helper_sbm", "helper_trm")}
    bad = [f"{k}:{v}" for k, v in statuses.items() if v not in ("ok",)]
    mm, ms, mt = med(rmain) if rmain else None, med(rsbm) if rsbm else None, med(rtrm) if rtrm else None
    trm_flag = rtrm and rtrm.get("trm", "").strip() == "trm=true"
    sd = pct(ms, mm)
    td = pct(mt, mm)
    if not bad and sd is not None:
        sbm_deltas.append(sd)
    if not bad and td is not None:
        trm_deltas.append(td)
        if trm_flag:
            trm_engaged.append((lbl, td, pct(mt, ms)))
    if bad:
        didnotfit.append((lbl, ";".join(bad)))
    print(
        f"{lbl:32s} | {('%.0f'%mm) if mm else '-':>9s} | {('%.0f'%ms) if ms else '-':>9s} | "
        f"{('%.0f'%mt) if mt else '-':>9s} | {('%+.1f'%sd) if sd is not None else '-':>7s} | "
        f"{('%+.1f'%td) if td is not None else '-':>7s} | {'YES' if trm_flag else ' no':>4s} | "
        f"{','.join(bad) if bad else 'ok'}"
    )

print("\n=== AGGREGATES (fitting convs only) ===")
if sbm_deltas:
    print(
        f"SBM vs main: mean {statistics.mean(sbm_deltas):+.2f}%  range [{min(sbm_deltas):+.1f}, {max(sbm_deltas):+.1f}]  n={len(sbm_deltas)}"
    )
if trm_deltas:
    print(
        f"TRM vs main: mean {statistics.mean(trm_deltas):+.2f}%  range [{min(trm_deltas):+.1f}, {max(trm_deltas):+.1f}]  n={len(trm_deltas)}"
    )
print(f"\nTRM ENGAGED on {len(trm_engaged)} convs:")
for lbl, td, tvss in trm_engaged:
    print(f"   {lbl:32s}  trm%vs main {td:+.1f}  trm%vs sbm {('%+.1f'%tvss) if tvss is not None else '-'}")
if didnotfit:
    print(f"\n=== DID-NOT-FIT / FAIL ({len(didnotfit)}) ===")
    for lbl, why in didnotfit:
        print(f"   {lbl:32s}  {why}")

# v2 comparison
if v2:
    print("\n=== v2 (pin-era BH) comparison ===")
    v2_sbm, v2_trm = [], []
    for lbl, modes in v2.items():
        rm, rs, rt = modes.get("main"), modes.get("helper_sbm"), modes.get("helper_trm")
        if not (rm and rs and rt):
            continue
        if any(modes[k]["status"] != "ok" for k in ("main", "helper_sbm", "helper_trm")):
            continue
        sd, td = pct(med(rs), med(rm)), pct(med(rt), med(rm))
        if sd is not None:
            v2_sbm.append(sd)
        if td is not None:
            v2_trm.append(td)
    if v2_sbm:
        print(
            f"v2 SBM vs main: mean {statistics.mean(v2_sbm):+.2f}%  range [{min(v2_sbm):+.1f}, {max(v2_sbm):+.1f}]  n={len(v2_sbm)}"
        )
    if v2_trm:
        print(
            f"v2 TRM vs main: mean {statistics.mean(v2_trm):+.2f}%  range [{min(v2_trm):+.1f}, {max(v2_trm):+.1f}]  n={len(v2_trm)}"
        )
