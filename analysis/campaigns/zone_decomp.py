#!/usr/bin/env python3
"""Decompose the SDPA device wall on the wall-setting core from the zone campaign outputs.

Inputs (per config tag, e.g. t21_causal_q128k128): <tag>_zoff_{runs,cores}.csv, <tag>_zon_{runs,cores,raw}.csv,
optional <tag>_zon_mp_{runs,counters}.csv. Iteration 0 of every run is discarded.

Zone tax (T0.2, data/bh_zones/t02_zone_tax_summary.csv): inside-window inflation per accumulate occurrence
ACC_IN = {TRISC_0: 0, others: 2} cycles; whole-thread inflation ACC_OUT = 26 cycles per occurrence.

Outputs: decomp_<tag>.csv (long table: iteration, risc, part, cycles_raw, count, cycles_corrected) and a
summary row appended to decomp_summary.csv. Model floor from tt-metal analysis/roofline.py predict()
(read-only; identical to polaris roofline_sdpa.py per FACTS s3).
"""
import argparse
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/data/bh_zones, DD that directory;
# or copied into $TTM/analysis/campaigns of a checkout, where DD falls back to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
DD = Path(os.environ.get("DD") or (Path.cwd() if _REPO else _SD))
WORK = Path(os.environ.get("SDPA_WORK", _REPO.parent if _REPO else _SD.parents[3]))
TTM = Path(os.environ.get("TTM", _REPO or WORK / "tt-metal"))
sys.path.insert(0, str(TTM / "analysis"))
from roofline import SdpaConfig, predict  # noqa: E402

ACC_IN = {"TRISC_0": 0.0, "TRISC_1": 2.0, "TRISC_2": 2.0, "BRISC": 2.0, "NCRISC": 2.0}
ACC_OUT = 26.0
INIT_MODEL = 30000
INIT_FIT = 102075

COMPUTE_PARTS = [
    "K_WAIT",
    "Q_WAIT",
    "RESERVE_QKT",
    "RECONFIG",
    "SUBEXP",
    "QK_MM",
    "MASK",
    "REDUCE",
    "OUT_RESERVE",
    "QKTIM_WAIT",
    "V_WAIT",
    "PV_MM",
    "PACK_DONE",
    "SALAD_EXP",
    "SALAD_CORR",
    "NORM",
    "PUSH_HOLD",
    "POPS",
    "MASK_DIAG",
    "EXP_INIT",
    "PUSHES",
]  # EXP is nested inside SUBEXP and STEP is the envelope
MATH_PARTS = ["SUBEXP", "QK_MM", "REDUCE", "PV_MM", "SALAD_EXP", "SALAD_CORR", "NORM"]
WAIT_PARTS = ["K_WAIT", "Q_WAIT", "V_WAIT", "QKTIM_WAIT", "RESERVE_QKT", "OUT_RESERVE", "PACK_DONE"]
FE_PARTS = ["RECONFIG", "MASK", "MASK_DIAG", "PUSH_HOLD", "POPS", "EXP_INIT", "PUSHES"]
READER_PARTS = ["R_K_READ", "R_V_READ", "R_Q_READ"]


def parse_tag(tag):
    m = re.search(r"(causal|noncausal)_(?:S\d+_)?q(\d+)k(\d+)", tag)
    return m.group(1) == "causal", int(m.group(2)), int(m.group(3))


def model_floor(causal, qc, kc, S=4096, nh=32, nkv=8, d=128):
    r = predict(
        SdpaConfig(
            S=S, head_dim=d, q_chunk=qc, k_chunk=kc, num_heads=nh, num_kv_heads=nkv, num_cores=110, is_causal=causal
        )
    )
    return r


def corr(sum_cyc, n, risc):
    return sum_cyc - n * ACC_IN[risc]


def decompose(tag, mp_tag=None):
    causal, qc, kc = parse_tag(tag)
    kct = kc // 32
    # config from the PROVENANCE line of the zones-on raw CSV (S, grid, fidelity, exp approx, head count)
    prov = open(DD / f"{tag}_zon.csv").readline()

    def env(key, default):
        m = re.search(r"\b" + key + r"=(\S+)", prov)
        return m.group(1) if m else default

    S = int(env("S", 4096))
    nh = int(env("nh", 32))
    nkv = int(env("nkv", 8))
    d = int(env("d", 128))
    fid = env("fid", "HiFi2")
    exp_approx = env("exp_approx", "1") == "1"
    grid = env("grid", "full")
    cores = 110 if grid in ("full", "11x10") else int(grid.split("x")[0]) * int(grid.split("x")[1])
    zoff_runs = pd.read_csv(DD / f"{tag}_zoff_runs.csv") if (DD / f"{tag}_zoff_runs.csv").exists() else None
    zon_runs = pd.read_csv(DD / f"{tag}_zon_runs.csv")
    zon_cores = pd.read_csv(DD / f"{tag}_zon_cores.csv")
    zon_raw = pd.read_csv(DD / f"{tag}_zon_raw.csv")
    r = predict(
        SdpaConfig(
            S=S,
            head_dim=d,
            q_chunk=qc,
            k_chunk=kc,
            num_heads=nh,
            num_kv_heads=nkv,
            num_cores=cores,
            is_causal=causal,
            fidelity=fid,
            exp_approx_mode=exp_approx,
        )
    )
    floor = r.math_active_cycles
    per_iter_floor = floor / r.inner_iters  # cycles per k-chunk step (model, equal chunks)
    rows = []
    summ = dict(
        tag=tag,
        causal=int(causal),
        q_chunk=qc,
        k_chunk=kc,
        kct=kct,
        S=S,
        cores=cores,
        fidelity=fid,
        exp_approx=int(exp_approx),
        floor_model=floor,
        inner_iters_model=r.inner_iters,
        fpu_model=r.fpu_cycles,
        sfpu_model=r.sfpu_cycles,
    )
    if zoff_runs is not None:
        w = zoff_runs[zoff_runs.run_idx > 0].wall_dev_cycles
        summ.update(
            wall_zoff_mean=w.mean(),
            wall_zoff_iter1=w.iloc[0],
            wall_zoff_iter2=w.iloc[1],
            wall_zoff_spread_pct=100 * (w.max() - w.min()) / w.mean(),
        )
    wz = zon_runs[zon_runs.run_idx > 0].wall_dev_cycles
    summ.update(wall_zon_mean=wz.mean(), wall_zon_iter1=wz.iloc[0], wall_zon_iter2=wz.iloc[1])
    if zoff_runs is not None:
        summ["gross_zone_tax_pct"] = 100 * (summ["wall_zon_mean"] / summ["wall_zoff_mean"] - 1)
    # per iteration, wall core
    for _, run in zon_runs[zon_runs.run_idx > 0].iterrows():
        it = int(run.run_idx)
        wc = zon_cores[(zon_cores.run_idx == it) & (zon_cores.is_wall_core == 1)]
        cx, cy = int(run.wall_core_x), int(run.wall_core_y)
        raw = zon_raw[(zon_raw.run_id == run.run_id) & (zon_raw.core_x == cx) & (zon_raw.core_y == cy)]
        wall = int(run.wall_dev_cycles)
        for _, cr in wc.iterrows():
            risc = cr.risc
            kd = cr.kernel_dur
            zones = {c[:-2]: cr[c] for c in wc.columns if c.endswith("_N") and not pd.isna(cr[c])}
            base = dict(tag=tag, iter=it, core=f"({cx},{cy})", risc=risc, wall=wall, kernel_dur=kd)
            # raw q-chunk zones
            rq = raw[raw.risc == risc]
            rz = rq[rq.zone.isin(["QCHUNK", "R_QCHUNK", "W_QCHUNK"])].sort_values("start")
            k0 = cr.kernel_start + zon_runs.loc[zon_runs.run_idx == it].index  # unused placeholder
            if len(rz):
                # kernel_start/end in _cores are relative to run t_min; raw start/end are absolute; recover t_min
                # from the KERNEL raw pair of this risc
                kp = rq[rq.zone.str.contains("KERNEL")]
                kstart, kend = int(kp.start.iloc[0]), int(kp.end.iloc[0])
                pre = int(rz.start.iloc[0] - kstart)
                tail = int(kend - rz.end.iloc[-1])
                qsum = int(rz.dur.sum())
                gaps = int(qsum - 0) if False else int((rz.end.iloc[-1] - rz.start.iloc[0]) - qsum)
                rows.append(dict(base, part="INIT_PRE_FIRST_QCHUNK", cycles_raw=pre, count=1, cycles_corr=pre))
                rows.append(dict(base, part="TAIL_AFTER_LAST_QCHUNK", cycles_raw=tail, count=1, cycles_corr=tail))
                rows.append(
                    dict(base, part="QCHUNK_SUM", cycles_raw=qsum, count=len(rz), cycles_corr=qsum - len(rz) * 51.5)
                )
                rows.append(dict(base, part="BETWEEN_QCHUNKS", cycles_raw=gaps, count=len(rz) - 1, cycles_corr=gaps))
                nq = len(rz)
            else:
                nq = 0
            nacc = 0
            for z, n in zones.items():
                s = cr[z]
                rows.append(dict(base, part=z, cycles_raw=s, count=int(n), cycles_corr=corr(s, n, risc)))
                nacc += n
            rows.append(
                dict(
                    base,
                    part="ZONE_OCCURRENCES_ACC",
                    cycles_raw=nacc * ACC_OUT,
                    count=int(nacc),
                    cycles_corr=nacc * ACC_OUT,
                )
            )
            # derived tiling
            if "STEP" in zones:
                parts_sum = sum(corr(cr[p], zones[p], risc) for p in COMPUTE_PARTS if p in zones)
                step = corr(cr["STEP"], zones["STEP"], risc)
                # inner occurrences (all accumulate zones except STEP itself) each leave ACC_OUT - ACC_IN cycles of
                # zone tax inside STEP but outside their own window; EXP is nested in SUBEXP so it is inner too
                n_inner = sum(zones[p] for p in zones if p != "STEP")
                tax_in_step = n_inner * (ACC_OUT - ACC_IN[risc])
                unz = step - parts_sum - tax_in_step
                rows.append(
                    dict(
                        base,
                        part="UNZONED_IN_STEP",
                        cycles_raw=step - parts_sum,
                        count=int(zones["STEP"]),
                        cycles_corr=unz,
                    )
                )
                rows.append(
                    dict(
                        base,
                        part="ZONE_TAX_IN_STEP",
                        cycles_raw=tax_in_step,
                        count=int(n_inner),
                        cycles_corr=tax_in_step,
                    )
                )
                if nq:
                    rows.append(
                        dict(
                            base,
                            part="OUTSIDE_STEP_IN_QCHUNK",
                            cycles_raw=qsum - cr["STEP"],
                            count=nq,
                            cycles_corr=(qsum - nq * 51.5) - step,
                        )
                    )
                math_sum = sum(corr(cr[p], zones[p], risc) for p in MATH_PARTS if p in zones)
                wait_sum = sum(corr(cr[p], zones[p], risc) for p in WAIT_PARTS if p in zones)
                fe_sum = sum(corr(cr[p], zones[p], risc) for p in FE_PARTS if p in zones)
                # wall-core floor: model per-step floor times the core's actual k-chunk steps
                floor_wc = per_iter_floor * zones["STEP"]
                rows.append(dict(base, part="MATH_ZONES_SUM", cycles_raw=math_sum, count=0, cycles_corr=math_sum))
                rows.append(
                    dict(
                        base,
                        part="FLOOR_MODEL_WALLCORE",
                        cycles_raw=floor_wc,
                        count=int(zones["STEP"]),
                        cycles_corr=floor_wc,
                    )
                )
                rows.append(
                    dict(
                        base,
                        part="MATH_ZONES_MINUS_FLOOR",
                        cycles_raw=math_sum - floor_wc,
                        count=0,
                        cycles_corr=math_sum - floor_wc,
                    )
                )
                rows.append(dict(base, part="WAIT_ZONES_SUM", cycles_raw=wait_sum, count=0, cycles_corr=wait_sum))
                rows.append(dict(base, part="FRONTEND_ZONES_SUM", cycles_raw=fe_sum, count=0, cycles_corr=fe_sum))
                if risc == "TRISC_1":
                    summ[f"it{it}_steps_wc"] = int(zones["STEP"])
                    summ[f"it{it}_qchunks_wc"] = nq
                    summ[f"it{it}_floor_wc"] = floor_wc
                    summ[f"it{it}_T1_math_zones"] = math_sum
                    summ[f"it{it}_T1_wait"] = wait_sum
                    summ[f"it{it}_T1_fe"] = fe_sum
                    summ[f"it{it}_T1_unzoned_in_step"] = unz
                    summ[f"it{it}_T1_pre"] = pre if nq else float("nan")
                    summ[f"it{it}_T1_tail"] = tail if nq else float("nan")
                if risc == "TRISC_0":
                    for p in ("K_WAIT", "V_WAIT", "Q_WAIT", "QKTIM_WAIT"):
                        summ[f"it{it}_T0_{p}"] = corr(cr[p], zones[p], risc) if p in zones else 0.0
                    summ[f"it{it}_T0_RECONFIG"] = corr(cr["RECONFIG"], zones["RECONFIG"], risc)
                if risc == "TRISC_2":
                    for p in ("MASK", "MASK_DIAG", "EXP", "SUBEXP", "SALAD_EXP", "RECONFIG", "EXP_INIT", "PUSHES"):
                        if p in zones:
                            summ[f"it{it}_T2_{p}"] = corr(cr[p], zones[p], risc)
                    summ[f"it{it}_T2_unzoned_in_step"] = unz
                if risc == "TRISC_0":
                    summ[f"it{it}_T0_unzoned_in_step"] = unz
            if "R_KCHUNK" in zones:
                rd_sum = sum(corr(cr[p], zones[p], risc) for p in READER_PARTS if p in zones)
                kch = corr(cr["R_KCHUNK"], zones["R_KCHUNK"], risc)
                rows.append(
                    dict(
                        base,
                        part="R_CONTROL_IN_KCHUNK",
                        cycles_raw=kch - rd_sum,
                        count=int(zones["R_KCHUNK"]),
                        cycles_corr=kch - rd_sum,
                    )
                )
                issue = (
                    rd_sum
                    - corr(cr["R_RESERVE"], zones["R_RESERVE"], risc)
                    - corr(cr["R_BARRIER"], zones["R_BARRIER"], risc)
                )
                rows.append(dict(base, part="R_ISSUE_AND_PUSH", cycles_raw=issue, count=0, cycles_corr=issue))
                summ[f"it{it}_R_BARRIER"] = corr(cr["R_BARRIER"], zones["R_BARRIER"], risc)
                summ[f"it{it}_R_RESERVE"] = corr(cr["R_RESERVE"], zones["R_RESERVE"], risc)
                summ[f"it{it}_R_ISSUE"] = issue
                summ[f"it{it}_R_KCHUNK"] = kch
                summ[f"it{it}_reader_risc"] = risc
            if "W_DRAIN" in zones:
                ww = corr(cr["W_WAIT"], zones["W_WAIT"], risc)
                wd = corr(cr["W_DRAIN"], zones["W_DRAIN"], risc)
                rows.append(
                    dict(base, part="W_WRITE", cycles_raw=wd - ww, count=int(zones["W_DRAIN"]), cycles_corr=wd - ww)
                )
                summ[f"it{it}_W_WAIT"] = ww
                summ[f"it{it}_W_WRITE"] = wd - ww
    # residuals (device basis, zones-off wall) and per-k-tile
    if zoff_runs is not None:
        wall = summ["wall_zoff_mean"]
        summ["resid_dev_init_model"] = wall - floor - INIT_MODEL
        summ["resid_dev_init_fit"] = wall - floor - INIT_FIT
        summ["resid_per_ktile_dev_init_model"] = summ["resid_dev_init_model"] / (r.inner_iters * kct)
        summ["resid_per_ktile_dev_init_fit"] = summ["resid_dev_init_fit"] / (r.inner_iters * kct)
        steps = [summ[k] for k in summ if k.endswith("_steps_wc")]
        if steps:
            st = sum(steps) / len(steps)
            fwc = per_iter_floor * st
            pre = [
                summ[k]
                for k in summ
                if k.endswith("_T1_pre") and not (isinstance(summ[k], float) and math.isnan(summ[k]))
            ]
            init_meas = sum(pre) / len(pre) if pre else float("nan")
            summ["init_measured_pre_first_qchunk_T1"] = init_meas
            summ["resid_wc_init_meas"] = wall - fwc - init_meas
            summ["resid_per_ktile_wc_init_meas"] = summ["resid_wc_init_meas"] / (st * kct)
            summ["resid_per_ktile_wc_init_model"] = (wall - fwc - INIT_MODEL) / (st * kct)
    # counters
    if mp_tag and (DD / f"{mp_tag}_counters.csv").exists():
        c = pd.read_csv(DD / f"{mp_tag}_counters.csv")
        mr = pd.read_csv(DD / f"{mp_tag}_runs.csv")
        rids = sorted(c.run_id.unique())[1:]
        cc = c[c.run_id.isin(rids)]
        summ["wall_mp_mean"] = mr[mr.run_idx > 0].wall_dev_cycles.mean()
        for name in [
            "FPU_COUNTER",
            "SFPU_COUNTER",
            "MATH_COUNTER",
            "WAITING_FOR_NONZERO_SEM_0",
            "WAITING_FOR_NONZERO_SEM_1",
            "WAITING_FOR_NONZERO_SEM_2",
            "WAITING_FOR_SRCA_VALID",
            "WAITING_FOR_SRCB_VALID",
            "THREAD_INSTRUCTIONS_0",
            "THREAD_INSTRUCTIONS_1",
            "THREAD_INSTRUCTIONS_2",
            "MATH_INSTRN_AVAILABLE",
            "UNPACK_INSTRN_AVAILABLE_0",
            "PACKER_BUSY",
            "UNPACK0_BUSY_THREAD0",
            "L1_0_UNPACKER_0",
            "L1_0_NOC_RING0_INCOMING_0",
        ]:
            sub = cc[(cc.counter == name) & (cc.value >= 0)]
            act = sub[sub.core_x.isin(cc[(cc.counter == "FPU_COUNTER") & (cc.value > 0)].core_x)]
            if len(sub):
                summ[f"ctr_{name}_mean"] = sub.value.mean()
                summ[f"ctr_{name}_max"] = sub.value.max()
        summ["ctr_ref_cnt_mean"] = cc.ref_cnt.mean()
    df = pd.DataFrame(rows)
    df.to_csv(DD / f"decomp_{tag}.csv", index=False)
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--mp-suffix", default="_zon_mp")
    ap.add_argument(
        "--config", default="", help="config label column (e.g. 'R1g hold-out noncausal_S2048_q128k128_hd64')"
    )
    a = ap.parse_args()
    summs = []
    for tag in a.tags:
        mp = tag + a.mp_suffix
        df, s = decompose(tag, mp)
        s["config"] = a.config or f"{tag.split('_')[0]} {tag}"
        summs.append(s)
        print(
            f"== {tag}: wall_zoff={s.get('wall_zoff_mean')} wall_zon={s.get('wall_zon_mean')} floor={s['floor_model']} "
            f"resid_dev_init_model={s.get('resid_dev_init_model')} per_ktile_dev_fit={s.get('resid_per_ktile_dev_init_fit')} "
            f"per_ktile_wc={s.get('resid_per_ktile_wc_init_meas')}"
        )
    out = pd.DataFrame(summs)
    # merge into the shared summary: replace rows with the same tag, keep every other row (never overwrite the union)
    p = DD / "decomp_summary.csv"
    if p.exists():
        old = pd.read_csv(p)
        old = old[~old.tag.isin(out.tag)]
        out = pd.concat([old, out], ignore_index=True, sort=False)
    out.to_csv(p, index=False)
    print(f"wrote decomp_summary.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
