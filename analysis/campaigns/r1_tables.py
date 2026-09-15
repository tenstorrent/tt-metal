#!/usr/bin/env python3
"""R1 campaign tables: per run, zones-off device wall (cycles, us), wall-setting core, iteration list; for zones-on twins the
UNPACK K/V wait share and the dominant thread. Also decode ops-report durations. Writes data/bh_zones/r1_walls.csv and
r1_tables.md; prints the markdown.

Dominant thread rule (MEASURED shares on the zones-on wall core, mean of invocations 1 and 2): 'reader/DRAM' when UNPACK
K_WAIT + V_WAIT >= 25 percent of the zones-on wall; otherwise 'compute (PACK)' when the PACK thread's non-wait zone sum
exceeds the MATH thread's, else 'compute (MATH)'.
"""
import glob
import os
import re
from pathlib import Path

import pandas as pd

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, DD that directory;
# or copied into $TTM/analysis/campaigns of a checkout, where DD falls back to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = os.environ.get("DD") or str(Path.cwd() if _REPO else _SD)
ACC_IN = {"TRISC_0": 0.0, "TRISC_1": 2.0, "TRISC_2": 2.0, "BRISC": 2.0, "NCRISC": 2.0}
WORK = [
    "QK_MM",
    "PV_MM",
    "SUBEXP",
    "REDUCE",
    "SALAD_EXP",
    "SALAD_CORR",
    "NORM",
    "RECONFIG",
    "MASK",
    "MASK_DIAG",
    "EXP_INIT",
    "PUSHES",
    "PUSH_HOLD",
    "POPS",
]


def sel_sdpa(runs):
    """Keep SDPA invocations only (mask harness also runs a mask-gen op; sparse/joint may run extra ops)."""
    if len(runs) <= 3:
        return runs
    keep = runs[runs.wall_dev_cycles > 0.2 * runs.wall_dev_cycles.max()].copy()
    return keep


def zon_view(tag):
    """K/V wait share and dominant thread on the wall core of the zones-on twin."""
    zr = f"{DD}/{tag}_zon_runs.csv"
    zc = f"{DD}/{tag}_zon_cores.csv"
    if not (os.path.exists(zr) and os.path.exists(zc)):
        return None
    runs = pd.read_csv(zr)
    cores = pd.read_csv(zc)
    runs = sel_sdpa(runs)
    its = list(runs.run_idx)[1:] if len(runs) > 1 else list(runs.run_idx)
    kv, dom, wz = [], [], []
    for it in its:
        wc = cores[(cores.run_idx == it) & (cores.is_wall_core == 1)]
        wall = runs[runs.run_idx == it].wall_dev_cycles.iloc[0]
        wz.append(wall)

        def v(risc, p):
            r = wc[wc.risc == risc]
            if not len(r) or p not in r or pd.isna(r[p].iloc[0]):
                return 0.0
            n = r[p + "_N"].iloc[0] if p + "_N" in r else 0
            return float(r[p].iloc[0] - (0 if pd.isna(n) else n) * ACC_IN[risc])

        kvw = v("TRISC_0", "K_WAIT") + v("TRISC_0", "V_WAIT")
        kv.append(100 * kvw / wall)
        t1 = sum(v("TRISC_1", p) for p in WORK)
        t2 = sum(v("TRISC_2", p) for p in WORK)
        rb = v("NCRISC", "R_BARRIER")
        dom.append("reader/DRAM" if kv[-1] >= 25 else ("compute (PACK)" if t2 > t1 else "compute (MATH)"))
    return dict(
        wall_zon=sum(wz) / len(wz),
        kv_wait_pct=sum(kv) / len(kv),
        dominant=max(sorted(set(dom)), key=dom.count),
        n_zon=len(its),
    )


def main():
    rows = []
    tags = sorted(
        {re.sub(r"_(zoff|zon)(_mp)?(_runs\.csv)$", "", os.path.basename(f)) for f in glob.glob(f"{DD}/r1*_runs.csv")}
    )
    for t in tags:
        zoff = (
            f"{DD}/{t}_zoff_runs.csv"
            if os.path.exists(f"{DD}/{t}_zoff_runs.csv")
            else (f"{DD}/{t}_runs.csv" if os.path.exists(f"{DD}/{t}_runs.csv") else None)
        )
        if zoff is None:
            continue
        r = sel_sdpa(pd.read_csv(zoff))
        prov = open(zoff.replace("_runs.csv", ".csv")).readline().strip()
        fw = re.search(r"fw_bundle=(\S+);", prov)
        sha = re.search(r"tt-metal(?:-fresh)? (\w+)", prov)
        w = r.wall_dev_cycles.iloc[1:] if len(r) > 1 else r.wall_dev_cycles
        block = t.split("_")[0]
        row = dict(
            block=block,
            tag=t,
            n_invocations=len(r),
            walls_cycles=" / ".join(f"{int(x):,}" for x in r.wall_dev_cycles),
            wall_mean_cycles=int(round(w.mean())),
            wall_mean_us=round(w.mean() / 1350, 1),
            spread_pct=round(100 * (w.max() - w.min()) / w.mean(), 2) if len(w) > 1 else 0.0,
            wall_core=" / ".join(f"({x},{y})" for x, y in zip(r.wall_core_x, r.wall_core_y)),
            cores=int(r.n_cores.iloc[0]),
            fw=fw.group(1) if fw else "",
            sha=sha.group(1)[:11] if sha else "",
            fresh="fresh" in prov,
        )
        z = zon_view(t)
        if z:
            row.update(
                wall_zon_mean=int(round(z["wall_zon"])),
                zone_tax_pct=round(100 * (z["wall_zon"] / w.mean() - 1), 1),
                kv_wait_pct_T0=round(z["kv_wait_pct"], 1),
                dominant_thread=z["dominant"],
            )
        # decode / regime ops report durations when present
        for suf in ("_zoff", ""):
            f = f"{DD}/{t}{suf}_ops_perf_results.csv"
            if os.path.exists(f):
                o = pd.read_csv(f, low_memory=False)
                dur = [c for c in o.columns if c.startswith("DEVICE KERNEL DURATION")]
                if dur:
                    main_ops = o[o["OP TYPE"].astype(str) != "signpost"]
                    top = main_ops.groupby("OP CODE")[dur[0]].sum().sort_values(ascending=False)
                    row["ops_report_top_op"] = top.index[0] if len(top) else ""
                    row["ops_report_durations_ns"] = (
                        ";".join(
                            f"{x:.0f}" for x in main_ops[main_ops["OP CODE"] == top.index[0]][dur[0]].astype(float)
                        )
                        if len(top)
                        else ""
                    )
                break
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{DD}/r1_walls.csv", index=False)
    cols = [
        "tag",
        "n_invocations",
        "walls_cycles",
        "wall_mean_cycles",
        "wall_mean_us",
        "spread_pct",
        "wall_core",
        "cores",
        "wall_zon_mean",
        "zone_tax_pct",
        "kv_wait_pct_T0",
        "dominant_thread",
    ]
    out = []
    for block, g in df.groupby("block"):
        out.append(f"\n### {block}\n")
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "---|" * len(cols))
        for _, r in g.iterrows():
            out.append(
                "| "
                + " | ".join(
                    "" if (c not in r or pd.isna(r[c])) else (f"{r[c]:,}" if isinstance(r[c], (int,)) else str(r[c]))
                    for c in cols
                )
                + " |"
            )
    text = "\n".join(out)
    open(f"{DD}/r1_tables.md", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
