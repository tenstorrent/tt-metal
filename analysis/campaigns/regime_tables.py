#!/usr/bin/env python3
"""T2.3 regime tables: walls (zones off/on), wall core, steps, and per-thread parts on the wall core for each regime run."""
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
PARTS = [
    "K_WAIT",
    "V_WAIT",
    "Q_WAIT",
    "QKTIM_WAIT",
    "RECONFIG",
    "MASK",
    "MASK_DIAG",
    "EXP_INIT",
    "PUSHES",
    "PUSH_HOLD",
    "POPS",
    "QK_MM",
    "PV_MM",
    "SUBEXP",
    "EXP",
    "REDUCE",
    "SALAD_EXP",
    "SALAD_CORR",
    "NORM",
    "STEP",
]
RPARTS = ["R_K_READ", "R_V_READ", "R_Q_READ", "R_RESERVE", "R_BARRIER", "R_KCHUNK"]


def main():
    tags = sorted({os.path.basename(f)[: -len("_zoff_runs.csv")] for f in glob.glob(f"{DD}/t23r_*_zoff_runs.csv")})
    lines = [
        "| regime run | zoff wall iter1 / iter2 (cycles) | mean | us | zon wall mean | gross tax pct | wall core (zoff) | steps / q chunks on zon wall core | invocations in CSV |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    parts_out = []
    for t in tags:
        zo = pd.read_csv(f"{DD}/{t}_zoff_runs.csv")
        # the mask harness also runs a mask-generation op per iteration: keep only SDPA invocations (wall > 20 percent of max)
        zo = zo[zo.wall_dev_cycles > 0.2 * zo.wall_dev_cycles.max()].reset_index(drop=True)
        zo["run_idx"] = range(len(zo))
        w = zo[zo.run_idx > 0].wall_dev_cycles if len(zo) > 1 else zo.wall_dev_cycles
        zon_path = f"{DD}/{t}_zon_runs.csv"
        zon = pd.read_csv(zon_path) if os.path.exists(zon_path) else None
        steps = ""
        if zon is not None:
            keep = zon.wall_dev_cycles > 0.2 * zon.wall_dev_cycles.max()
            zon = zon[keep].reset_index(drop=True)
            kept_idx = list(pd.read_csv(zon_path)[keep].run_idx)
            zon["run_idx"] = range(len(zon))
            cores_all = pd.read_csv(f"{DD}/{t}_zon_cores.csv")
            cores_all = cores_all[cores_all.run_idx.isin(kept_idx)].copy()
            cores_all["run_idx"] = cores_all.run_idx.map({o: n for n, o in enumerate(kept_idx)})
            wz = zon[zon.run_idx > 0].wall_dev_cycles if len(zon) > 1 else zon.wall_dev_cycles
            cores = cores_all
            for it in sorted(cores.run_idx.unique())[1:]:
                wc = cores[(cores.run_idx == it) & (cores.is_wall_core == 1)]
                t1 = wc[wc.risc == "TRISC_1"]
                if len(t1) and "STEP_N" in t1:
                    steps = f"{int(t1.STEP_N.iloc[0])} / " + (
                        str(int(cores[(cores.run_idx == it)].groupby(["core_x", "core_y"]).size().max()))
                        if False
                        else ""
                    )
                wall = zon[zon.run_idx == it].wall_dev_cycles.iloc[0]
                row = {"run": t, "iter": it, "wall_zon": wall}
                for _, cr in wc.iterrows():
                    for p in PARTS + RPARTS + ["W_WAIT", "W_DRAIN"]:
                        if p in cr and not pd.isna(cr[p]):
                            n = cr.get(p + "_N", 0)
                            row[f"{cr.risc}:{p}"] = cr[p] - (n if not pd.isna(n) else 0) * ACC_IN[cr.risc]
                            row[f"{cr.risc}:{p}_N"] = n
                parts_out.append(row)
            qn = ""
            raw = pd.read_csv(f"{DD}/{t}_zon_raw.csv")
            it = sorted(zon.run_idx.unique())[1]
            rid = zon[zon.run_idx == it].run_id.iloc[0]
            cx, cy = zon[zon.run_idx == it].wall_core_x.iloc[0], zon[zon.run_idx == it].wall_core_y.iloc[0]
            q = raw[
                (raw.run_id == rid)
                & (raw.core_x == cx)
                & (raw.core_y == cy)
                & (raw.risc == "TRISC_1")
                & (raw.zone == "QCHUNK")
            ]
            wc1 = cores[(cores.run_idx == it) & (cores.is_wall_core == 1) & (cores.risc == "TRISC_1")]
            steps = f"{int(wc1.STEP_N.iloc[0]) if len(wc1) and 'STEP_N' in wc1 and not pd.isna(wc1.STEP_N.iloc[0]) else 'n/a'} / {len(q)}"
            tax = 100 * (wz.mean() / w.mean() - 1)
            zon_s = f"{wz.mean():,.0f}"
            tax_s = f"{tax:.1f}"
        else:
            zon_s = "n/a"
            tax_s = "n/a"
        core = f"({zo.wall_core_x.iloc[-1]},{zo.wall_core_y.iloc[-1]})"
        lines.append(
            f"| {t} | {' / '.join(f'{int(x):,}' for x in w)} | {w.mean():,.0f} | {w.mean()/1350:,.1f} | {zon_s} | {tax_s} | {core} | {steps} | {len(zo)} |"
        )
    pd.DataFrame(parts_out).to_csv(f"{DD}/regime_parts.csv", index=False)
    open(f"{DD}/regime_tables.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    # per-thread parts, percent of the zones-on wall, per regime (mean over iterations 1, 2)
    P = pd.DataFrame(parts_out)
    for t in tags:
        sub = P[P.run == t]
        if sub.empty:
            continue
        wall = sub.wall_zon.mean()
        print(f"\n### {t} parts on wall core, percent of zones-on wall {wall:,.0f}")
        for risc in ["TRISC_0", "TRISC_1", "TRISC_2", "NCRISC", "BRISC"]:
            items = [
                (c.split(":")[1], sub[c].mean())
                for c in sub.columns
                if c.startswith(risc + ":") and not c.endswith("_N") and sub[c].mean() > 0.005 * wall
            ]
            items.sort(key=lambda x: -x[1])
            print(risc, ", ".join(f"{k} {v/wall*100:.1f}" for k, v in items))


if __name__ == "__main__":
    main()
