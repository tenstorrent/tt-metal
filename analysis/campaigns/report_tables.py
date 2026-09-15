#!/usr/bin/env python3
"""Build the report tables for bh/zone_decomposition.md from decomp_<tag>.csv and decomp_summary.csv.

Tables (markdown printed to stdout, also written to report_tables.md):
 T1 walls per config: zones off (iter1, iter2, mean), zones on, gross tax, model floor, residuals, wall core, steps
 T2 per-thread decomposition at a config: parts in cycles (mean of iter 1,2 on each iteration's wall core) and % wall
 T3 acceptance: TRISC1 tiling and residual split
 T4 per-k-tile flatness (q128: k128/256/512) and q prefactor (k128: q64/128/256), both bases
 T5 causal vs non-causal per component at (128,128)
 T6 ablation deltas (t22 tags vs t21 base)
 T7 counters cross-check
"""
import os
import sys
from pathlib import Path

import pandas as pd

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, DD that directory;
# or copied into $TTM/analysis/campaigns of a checkout, where DD falls back to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = Path(os.environ.get("DD") or (Path.cwd() if _REPO else _SD))
INIT_MODEL, INIT_FIT = 30000, 102075
ORDER_T0 = [
    "K_WAIT",
    "V_WAIT",
    "Q_WAIT",
    "QKTIM_WAIT",
    "RESERVE_QKT",
    "OUT_RESERVE",
    "PACK_DONE",
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
    "REDUCE",
    "SALAD_EXP",
    "SALAD_CORR",
    "NORM",
    "UNZONED_IN_STEP",
    "OUTSIDE_STEP_IN_QCHUNK",
    "INIT_PRE_FIRST_QCHUNK",
    "TAIL_AFTER_LAST_QCHUNK",
    "ZONE_TAX_IN_STEP",
]
WAITS = ["K_WAIT", "V_WAIT", "Q_WAIT", "QKTIM_WAIT", "RESERVE_QKT", "OUT_RESERVE", "PACK_DONE"]
FE = ["RECONFIG", "MASK", "MASK_DIAG", "EXP_INIT", "PUSHES", "PUSH_HOLD", "POPS"]
MATH = ["QK_MM", "PV_MM", "SUBEXP", "REDUCE", "SALAD_EXP", "SALAD_CORR", "NORM"]
CTRL = ["UNZONED_IN_STEP", "OUTSIDE_STEP_IN_QCHUNK", "INIT_PRE_FIRST_QCHUNK", "TAIL_AFTER_LAST_QCHUNK"]


def load(tag):
    d = pd.read_csv(DD / f"decomp_{tag}.csv")
    return d[d["iter"] > 0]


def part_mean(d, risc, part):
    x = d[(d.risc == risc) & (d.part == part)].cycles_corr
    return float(x.mean()) if len(x) else float("nan")


def thread_table(tag, wall):
    d = load(tag)
    rows = []
    for p in ORDER_T0:
        r = {"part": p}
        for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
            v = part_mean(d, risc, p)
            r[risc] = v
        rows.append(r)
    t = pd.DataFrame(rows).set_index("part")
    pct = t / wall * 100
    return t, pct


def reader_writer_table(tag, wall):
    d = load(tag)
    rd = d[d.risc == "NCRISC"]
    wr = d[d.risc == "BRISC"]
    out = {}
    for p in [
        "R_K_READ",
        "R_V_READ",
        "R_Q_READ",
        "R_RESERVE",
        "R_BARRIER",
        "R_ISSUE_AND_PUSH",
        "R_CONTROL_IN_KCHUNK",
        "R_KCHUNK",
        "INIT_PRE_FIRST_QCHUNK",
        "TAIL_AFTER_LAST_QCHUNK",
        "OUTSIDE_STEP_IN_QCHUNK",
    ]:
        out[("reader", p)] = part_mean(d, "NCRISC", p)
    for p in ["W_WAIT", "W_WRITE", "W_DRAIN", "INIT_PRE_FIRST_QCHUNK", "TAIL_AFTER_LAST_QCHUNK"]:
        out[("writer", p)] = part_mean(d, "BRISC", p)
    return out


def md(df, fmt="{:,.0f}"):
    cols = list(df.columns)
    s = "| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |\n"
    s += "|" + "---|" * (len(cols) + 1) + "\n"
    for i, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(
                fmt.format(v) if isinstance(v, (int, float)) and not pd.isna(v) else ("" if pd.isna(v) else str(v))
            )
        s += "| " + " | ".join([str(i)] + vals) + " |\n"
    return s


def main():
    S = pd.read_csv(DD / "decomp_summary.csv").set_index("tag")
    out = []
    # T1
    t1 = S[
        [
            "causal",
            "q_chunk",
            "k_chunk",
            "wall_zoff_iter1",
            "wall_zoff_iter2",
            "wall_zoff_mean",
            "wall_zon_mean",
            "gross_zone_tax_pct",
            "floor_model",
            "inner_iters_model",
            "it1_steps_wc",
            "it1_qchunks_wc",
            "init_measured_pre_first_qchunk_T1",
            "resid_dev_init_model",
            "resid_dev_init_fit",
            "resid_wc_init_meas",
        ]
    ].copy()
    out.append(
        "### T1 walls, floors and residuals per config (cycles; zones-off walls, iterations 1 and 2)\n\n"
        + md(t1, "{:,.1f}")
        + "\n"
    )
    # T2 per config
    for tag in S.index:
        wall = S.loc[tag, "wall_zoff_mean"]
        t, pct = thread_table(tag, wall)
        both = t.copy()
        for c in ["TRISC_0", "TRISC_1", "TRISC_2"]:
            both[c + " %"] = pct[c]
        out.append(
            f"### T2 {tag}: compute-thread parts on the wall-setting core (mean of iterations 1 and 2; cycles and percent of the zones-off wall {wall:,.0f})\n\n"
            + md(both, "{:,.1f}")
            + "\n"
        )
        rw = reader_writer_table(tag, wall)
        rwdf = pd.DataFrame(
            [{"thread": k[0], "part": k[1], "cycles": v, "pct_wall": 100 * v / wall} for k, v in rw.items()]
        ).set_index("part")
        out.append(f"### T2b {tag}: reader (NCRISC) and writer (BRISC) parts\n\n" + md(rwdf, "{:,.1f}") + "\n")
        # T3 acceptance on TRISC_1
        d = load(tag)
        t1v = {p: part_mean(d, "TRISC_1", p) for p in ORDER_T0}
        floor_wc = S.loc[tag, ["it1_floor_wc", "it2_floor_wc"]].mean()
        init_meas = t1v["INIT_PRE_FIRST_QCHUNK"]
        kd = d[(d.risc == "TRISC_1")].kernel_dur.mean()
        wait = sum(t1v[p] for p in WAITS)
        fe = sum(t1v[p] for p in FE)
        math = sum(t1v[p] for p in MATH)
        ctrl = t1v["UNZONED_IN_STEP"] + t1v["OUTSIDE_STEP_IN_QCHUNK"] + t1v["TAIL_AFTER_LAST_QCHUNK"]
        tax = t1v["ZONE_TAX_IN_STEP"]
        resid = wall - floor_wc - init_meas
        parts_sum = wait + fe + (math - floor_wc) + ctrl
        wall_zon = S.loc[tag, "wall_zon_mean"]
        displaced = tax - (wall_zon - wall)
        acc = pd.DataFrame(
            [
                dict(quantity="zones-off wall (dev, = wall-core span)", cycles=wall, pct_resid=float("nan")),
                dict(quantity="TRISC1 KERNEL span on wall core (zones on)", cycles=kd, pct_resid=float("nan")),
                dict(
                    quantity="model floor on wall core (per-step floor x actual steps)",
                    cycles=floor_wc,
                    pct_resid=float("nan"),
                ),
                dict(
                    quantity="init measured (KERNEL start to first QCHUNK, TRISC1)",
                    cycles=init_meas,
                    pct_resid=float("nan"),
                ),
                dict(quantity="RESIDUAL = wall - floor_wc - init", cycles=resid, pct_resid=100),
                dict(
                    quantity="  waits on TRISC1 (K/V/Q/qkt_im/reserve/pack_done)",
                    cycles=wait,
                    pct_resid=100 * wait / resid,
                ),
                dict(
                    quantity="  front end on TRISC1 (reconfig, mask, exp_init, pushes, pops)",
                    cycles=fe,
                    pct_resid=100 * fe / resid,
                ),
                dict(
                    quantity="  in-phase excess = math zones on TRISC1 - floor_wc",
                    cycles=math - floor_wc,
                    pct_resid=100 * (math - floor_wc) / resid,
                ),
                dict(
                    quantity="  un-zoned control flow (in-step + between steps + tail)",
                    cycles=ctrl,
                    pct_resid=100 * ctrl / resid,
                ),
                dict(
                    quantity="SUM of named parts (zones-on measurement)",
                    cycles=parts_sum,
                    pct_resid=100 * parts_sum / resid,
                ),
                dict(
                    quantity="  slack displaced by the zone tax = tax_T1 - (wall_zon - wall_zoff) (INFERRED: waiting that the instrumentation overhead replaced)",
                    cycles=displaced,
                    pct_resid=100 * displaced / resid,
                ),
                dict(
                    quantity="SUM including displaced slack",
                    cycles=parts_sum + displaced,
                    pct_resid=100 * (parts_sum + displaced) / resid,
                ),
                dict(
                    quantity="zone tax inside STEP on TRISC1 (removed, not a part)", cycles=tax, pct_resid=float("nan")
                ),
                dict(
                    quantity="zones-on wall minus zones-off wall (gross tax on the wall)",
                    cycles=wall_zon - wall,
                    pct_resid=float("nan"),
                ),
                dict(
                    quantity="check: floor_wc + init + parts + tax vs zones-on TRISC1 span (should be ~0)",
                    cycles=floor_wc + init_meas + parts_sum + tax - kd,
                    pct_resid=float("nan"),
                ),
            ]
        ).set_index("quantity")
        # per-thread views of the same split
        tv = []
        for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
            v = {p: part_mean(d, risc, p) for p in ORDER_T0}
            w_ = sum(v[p] for p in WAITS)
            f_ = sum(v[p] for p in FE)
            m_ = sum(v[p] for p in MATH)
            c_ = (
                v["UNZONED_IN_STEP"]
                + v["OUTSIDE_STEP_IN_QCHUNK"]
                + v["TAIL_AFTER_LAST_QCHUNK"]
                + v["INIT_PRE_FIRST_QCHUNK"]
            )
            tv.append(
                dict(
                    thread=risc,
                    waits=w_,
                    front_end=f_,
                    math_zones=m_,
                    math_minus_floor=m_ - floor_wc,
                    control=c_,
                    tax=v["ZONE_TAX_IN_STEP"],
                    sum_all=w_ + f_ + m_ + c_ + v["ZONE_TAX_IN_STEP"],
                    kernel_span=d[d.risc == risc].kernel_dur.mean(),
                    waits_pct_wall=100 * w_ / wall,
                    fe_pct_wall=100 * f_ / wall,
                    math_excess_pct_wall=100 * (m_ - floor_wc) / wall,
                    control_pct_wall=100 * c_ / wall,
                )
            )
        tvd = pd.DataFrame(tv).set_index("thread")
        out.append(
            f"### T3b {tag}: the same split seen from each compute thread (cycles; percent of the zones-off wall)\n\n"
            + md(tvd, "{:,.1f}")
            + "\n"
        )
        out.append(f"### T3 {tag}: acceptance on the wall-setting core, TRISC1 view\n\n" + md(acc, "{:,.1f}") + "\n")

    # T4 flatness and prefactor
    def sel(c, q, k):
        m = S[(S.causal == c) & (S.q_chunk == q) & (S.k_chunk == k)]
        return m.iloc[0] if len(m) else None

    rows = []
    for c, q, k in [
        (1, 128, 128),
        (1, 128, 256),
        (1, 128, 512),
        (1, 64, 128),
        (1, 256, 128),
        (1, 512, 128),
        (1, 512, 512),
        (0, 128, 128),
        (0, 128, 256),
        (0, 128, 512),
        (0, 64, 128),
        (0, 256, 128),
        (0, 512, 128),
        (0, 512, 512),
    ]:
        r = sel(c, q, k)
        if r is None:
            continue
        rows.append(
            dict(
                config=f"{'causal' if c else 'noncausal'} q{q} k{k}",
                wall=r.wall_zoff_mean,
                floor=r.floor_model,
                per_ktile_dev_init_fit=r.resid_per_ktile_dev_init_fit,
                per_ktile_dev_init_model=r.resid_per_ktile_dev_init_model,
                per_ktile_wc_init_meas=r.resid_per_ktile_wc_init_meas,
                steps_wc=r.it1_steps_wc,
                T0_KV_wait_per_ktile=(r.it1_T0_K_WAIT + r.it1_T0_V_WAIT + r.it2_T0_K_WAIT + r.it2_T0_V_WAIT)
                / 2
                / (r.it1_steps_wc * r.kct),
                T2_EXP_INIT_per_step=(r.get("it1_T2_EXP_INIT", float("nan")) + r.get("it2_T2_EXP_INIT", float("nan")))
                / 2
                / r.it1_steps_wc,
                T2_MASK_per_step=(r.it1_T2_MASK + r.it2_T2_MASK) / 2 / r.it1_steps_wc,
                R_BARRIER_per_ktile=(r.it1_R_BARRIER + r.it2_R_BARRIER) / 2 / (r.it1_steps_wc * r.kct),
            )
        )
    t4 = pd.DataFrame(rows).set_index("config")
    out.append(
        "### T4 residual per k-tile (flatness across k_chunk at q128, prefactor across q_chunk at k128), both bases\n\n"
        + md(t4, "{:,.1f}")
        + "\n"
    )
    # T5 causal vs non-causal at (128,128): per component per k-chunk step on the wall core, by thread
    if "t21_causal_q128k128" in S.index and "t21_noncausal_q128k128" in S.index:
        rows = []
        for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
            for p in ORDER_T0[:-1]:
                r = {"thread": risc, "part": p}
                for tag, lab in [("t21_causal_q128k128", "causal"), ("t21_noncausal_q128k128", "noncausal")]:
                    d = load(tag)
                    steps = S.loc[tag, ["it1_steps_wc", "it2_steps_wc"]].mean()
                    v = part_mean(d, risc, p)
                    r[lab + "_total"] = v
                    r[lab + "_per_step"] = v / steps
                r["delta_per_step"] = r["causal_per_step"] - r["noncausal_per_step"]
                rows.append(r)
        t5 = pd.DataFrame(rows).set_index(["thread", "part"])
        out.append(
            "### T5 causal vs non-causal at q128 k128: wall-core parts, total cycles and per k-chunk step (causal 165 steps, non-causal 320 steps on their wall cores)\n\n"
            + md(t5.reset_index().set_index("part"), "{:,.1f}")
            + "\n"
        )
        rows = []
        for p in [
            "R_K_READ",
            "R_V_READ",
            "R_Q_READ",
            "R_RESERVE",
            "R_BARRIER",
            "R_ISSUE_AND_PUSH",
            "R_CONTROL_IN_KCHUNK",
            "R_KCHUNK",
        ]:
            r = {"part": p}
            for tag, lab in [("t21_causal_q128k128", "causal"), ("t21_noncausal_q128k128", "noncausal")]:
                d = load(tag)
                steps = S.loc[tag, ["it1_steps_wc", "it2_steps_wc"]].mean()
                v = part_mean(d, "NCRISC", p)
                r[lab + "_total"] = v
                r[lab + "_per_step"] = v / steps
            rows.append(r)
        out.append(
            "### T5b reader (NCRISC) causal vs non-causal\n\n"
            + md(pd.DataFrame(rows).set_index("part"), "{:,.1f}")
            + "\n"
        )
    # T6 ablations: t22_abl_<name>_causal_q<q>k<k> vs t21_causal_q<q>k<k>
    abl_tags = [t for t in S.index if t.startswith("t22_abl_")]
    if abl_tags:
        rows = []
        for at in abl_tags:
            name = at.split("_")[2]
            base = "t21_" + at.split("_causal_")[0].split("_")[-1] if False else "t21_causal_" + at.split("_causal_")[1]
            if base not in S.index:
                continue
            wb, wa = S.loc[base, "wall_zoff_mean"], S.loc[at, "wall_zoff_mean"]
            r = dict(
                ablation=name,
                config=base.replace("t21_", ""),
                wall_base=wb,
                wall_abl=wa,
                delta_wall=wa - wb,
                delta_pct=100 * (wa - wb) / wb,
            )
            db, da = load(base), load(at)
            for risc, parts in [
                (
                    "TRISC_0",
                    [
                        "K_WAIT",
                        "V_WAIT",
                        "Q_WAIT",
                        "RECONFIG",
                        "MASK",
                        "QK_MM",
                        "PV_MM",
                        "REDUCE",
                        "SALAD_CORR",
                        "UNZONED_IN_STEP",
                    ],
                ),
                ("TRISC_1", ["QK_MM", "PV_MM", "SUBEXP", "REDUCE", "SALAD_CORR", "RECONFIG", "MASK"]),
                (
                    "TRISC_2",
                    [
                        "MASK",
                        "MASK_DIAG",
                        "EXP_INIT",
                        "EXP",
                        "SUBEXP",
                        "SALAD_EXP",
                        "REDUCE",
                        "RECONFIG",
                        "PUSHES",
                        "UNZONED_IN_STEP",
                    ],
                ),
                ("NCRISC", ["R_K_READ", "R_V_READ", "R_BARRIER", "R_RESERVE", "R_ISSUE_AND_PUSH"]),
            ]:
                for p in parts:
                    r[f"d_{risc[-1] if risc.startswith('T') else 'R'}_{p}"] = part_mean(da, risc, p) - part_mean(
                        db, risc, p
                    )
            rows.append(r)
        t6 = pd.DataFrame(rows).set_index(["ablation", "config"])
        out.append(
            "### T6 ablations: wall delta (zones-off) and per-part deltas on the wall core (zones-on, ablated minus base; d_0 UNPACK, d_1 MATH, d_2 PACK, d_R reader)\n\n"
            + md(t6.reset_index().set_index("ablation"), "{:,.0f}")
            + "\n"
        )
    # T7 counters
    ctr_cols = [c for c in S.columns if c.startswith("ctr_") and c.endswith("_mean")]
    if ctr_cols:
        t7 = S[["wall_zoff_mean", "wall_mp_mean"] + ctr_cols].copy()
        t7.columns = [c.replace("ctr_", "").replace("_mean", "") for c in t7.columns]
        out.append(
            "### T7 perf counters (zones-on multipass runs, mean over the 110 cores, cycles; iterations 1 and 2)\n\n"
            + md(t7.T, "{:,.0f}")
            + "\n"
        )
    text = "\n".join(out)
    (DD / "report_tables.md").write_text(text)
    print(text[:20000])


if __name__ == "__main__":
    main()
