#!/usr/bin/env python
"""Floor-constant verification and refit on this campaign's perf-counter captures (model/floor_verification.md).

Read-only on data/, bh/ and the polaris tree (the model module is imported from the polaris working tree at
HEAD f1adecb and monkeypatched IN MEMORY only for the "refit adopted" columns; no file under polaris is written).
Writes: data/floor_verification_configs.csv, data/floor_verification_fits.json and prints the markdown tables the
report quotes. The figure is drawn by figs/m_floor_refit.py from the CSV and JSON.

Run with the polaris venv python:
    $POLARIS/.venv/bin/python model/floor_verification.py
"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

# Paths per PORTABLE_CONTRACT.md: ROOT is handoff/revamp (the directory above this file's), POLARIS the
# polaris checkout beside it in the workspace; both are environment overrides.
ROOT = Path(os.environ.get("HANDOFF", Path(__file__).resolve().parents[1]))
DD = ROOT / "data" / "bh_zones"
POLARIS = os.environ.get("POLARIS", str(ROOT.parents[1] / "polaris"))
sys.path.insert(0, POLARIS)
import ttsim.perf.roofline_sdpa as M  # noqa: E402  (HEAD f1adecb, read-only import)

C3 = ["FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"]
ITERS = (1, 2)  # run_idx of the iterations kept (the first invocation of each process is discarded, as in T7)

GQA = dict(num_heads=32, num_kv_heads=8)
PROD = dict(
    num_heads=32,
    num_kv_heads=8,
    fidelity="HiFi4",
    exp_approx_mode=False,
    fp32_dest_acc=True,
    accum_dtype="float32",
    num_cores=64,
)
MLA = dict(
    head_dim=576,
    v_head_dim=512,
    num_kv_heads=1,
    num_heads=16,
    fidelity="HiFi4",
    input_dtype="bfloat16",
    exp_approx_mode=False,
    q_chunk=32,
    k_chunk=128,
)

# (label, counters tag, model kwargs on top of S 4096 / 110 cores / HiFi2 / bfp8, group)
CONFIGS = [
    ("causal q128 k128", "t21_causal_q128k128_zon_mp", dict(**GQA), "grid1"),
    ("causal q128 k256", "t21_causal_q128k256_zon_mp", dict(k_chunk=256, **GQA), "grid1"),
    ("causal q128 k512", "t21_causal_q128k512_zon_mp", dict(k_chunk=512, **GQA), "grid1"),
    ("causal q64 k128", "t21_causal_q64k128_zon_mp", dict(q_chunk=64, **GQA), "grid1"),
    ("causal q256 k128", "t21_causal_q256k128_zon_mp", dict(q_chunk=256, **GQA), "grid1"),
    ("causal q512 k128", "t21_causal_q512k128_zon_mp", dict(q_chunk=512, **GQA), "grid1"),
    ("causal q512 k512", "t21_causal_q512k512_zon_mp", dict(q_chunk=512, k_chunk=512, **GQA), "grid1"),
    ("non-causal q128 k128", "t21_noncausal_q128k128_zon_mp", dict(is_causal=False, **GQA), "grid1"),
    ("non-causal q128 k256", "t21_noncausal_q128k256_zon_mp", dict(is_causal=False, k_chunk=256, **GQA), "grid1"),
    ("non-causal q128 k512", "t21_noncausal_q128k512_zon_mp", dict(is_causal=False, k_chunk=512, **GQA), "grid1"),
    ("non-causal q64 k128", "t21_noncausal_q64k128_zon_mp", dict(is_causal=False, q_chunk=64, **GQA), "grid1"),
    ("non-causal q256 k128", "t21_noncausal_q256k128_zon_mp", dict(is_causal=False, q_chunk=256, **GQA), "grid1"),
    ("non-causal q512 k128", "t21_noncausal_q512k128_zon_mp", dict(is_causal=False, q_chunk=512, **GQA), "grid1"),
    (
        "non-causal q512 k512",
        "t21_noncausal_q512k512_zon_mp",
        dict(is_causal=False, q_chunk=512, k_chunk=512, **GQA),
        "grid1",
    ),
    ("causal q128 k128, unmodified kernel", "t01_causal_q128k128_mp", dict(**GQA), "unmod"),
    ("non-causal q128 k128, unmodified kernel", "t01_noncausal_q128k128_mp", dict(is_causal=False, **GQA), "unmod"),
    ("causal q128 k128, A6 exp stub", "t22_abl_a6_causal_q128k128_zon_mp", dict(**GQA), "a6"),
    (
        "cross 2k/8k nh16 q128 k128",
        "t23r_cross_2048_8192_nh16_zon_mp",
        dict(S=2048, kv_seq=8192, num_heads=16, is_causal=False),
        "cross",
    ),
    ("MLA nh16 S2048 q32 k128 HiFi4", "t23r_mla_nh16_S2048_zon_mp", dict(S=2048, **MLA), "mla"),
    (
        "production S4096 q256 k256 g64 HiFi4 fp32",
        "t24_prod_causal_S4096_q256k256_g64_zon_mp",
        dict(q_chunk=256, k_chunk=256, **PROD),
        "prod",
    ),
]
# Production points without a counter capture (zones on only): zone matmul time on the wall core is the proxy
# (bh/production_config.md section 3 table).
PROD_ZONE_ONLY = [
    (
        "production settings, 110 cores (no counters)",
        dict(q_chunk=256, k_chunk=256, **dict(PROD, num_cores=110)),
        dict(QK_MM=899695, PV_MM=849165, steps_wc=51, wall=3019747),
    ),
    (
        "A10 fp32 off, streaming kernel at HiFi4 (no counters)",
        dict(q_chunk=256, k_chunk=256, **dict(PROD, fp32_dest_acc=False, accum_dtype="bfloat16")),
        dict(QK_MM=1085750, PV_MM=1109280, steps_wc=68, wall=3025634),
    ),
]
DECODE_KW = dict(
    cache_len=16384,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    k_chunk=0,
    fidelity="HiFi2",
    input_dtype="bfloat16",
    accum_dtype="float32",
    paged=True,
    page_block_size=32,
    max_cores_per_head_batch=16,
    batch=32,
    num_cores=64,
    cur_pos=1024,
    kv_input_dtype="bfp8_b",
)


def mk(**kw):
    base = dict(S=4096, num_cores=110, fidelity="HiFi2", input_dtype="bfp8_b")
    base.update(kw)
    return M.SdpaConfig(arch=M.ArchConfig(), **base)


# ---------------------------------------------------------------------------------------------------------
# counters
# ---------------------------------------------------------------------------------------------------------
def load_counters(tag):
    """Per iteration (run_idx 1 and 2): mean over the active cores (FPU_COUNTER > 0) and the value on the
    wall-setting core of that iteration (from <tag>_runs.csv); STEP_N of that core when the run was zoned."""
    d = pd.read_csv(DD / f"{tag}_counters.csv")
    runs = pd.read_csv(DD / f"{tag}_runs.csv")
    d = d[d.counter.isin(C3)]
    cores_path = DD / f"{tag}_cores.csv"
    cores = pd.read_csv(cores_path) if cores_path.exists() else None
    out = {"tag": tag}
    means, walls, stepn, wall_cycles, wall_cores = [], [], [], [], []
    for _, r in runs.iterrows():
        if int(r.run_idx) not in ITERS:
            continue
        dr = d[d.run_id == r.run_id]
        fpu = dr[(dr.counter == "FPU_COUNTER") & (dr.value > 0)]
        active = set(zip(fpu.core_x, fpu.core_y))
        dr = dr[[(x, y) in active for x, y in zip(dr.core_x, dr.core_y)]]
        means.append(dr.groupby("counter").value.mean())
        w = dr[(dr.core_x == r.wall_core_x) & (dr.core_y == r.wall_core_y)].set_index("counter").value
        walls.append(w)
        wall_cycles.append(float(r.wall_dev_cycles))
        wall_cores.append((int(r.wall_core_x), int(r.wall_core_y)))
        if cores is not None and "STEP_N" in cores.columns:
            c = cores[
                (cores.run_id == r.run_id)
                & (cores.risc == "TRISC_1")
                & (cores.core_x == r.wall_core_x)
                & (cores.core_y == r.wall_core_y)
            ]
            stepn.append(float(c.STEP_N.iloc[0]) if len(c) else float("nan"))
        out["active_cores"] = len(active)
    mm = pd.concat(means, axis=1).mean(axis=1)
    ww = pd.concat(walls, axis=1).mean(axis=1)
    for c in C3:
        out[c + "_mean"] = float(mm[c])
        out[c + "_wall"] = float(ww[c])
    out["wall_mp_mean"] = float(np.mean(wall_cycles))
    out["wall_cores"] = str(wall_cores)
    out["STEP_N_wall"] = float(np.mean(stepn)) if stepn else float("nan")
    return out


# ---------------------------------------------------------------------------------------------------------
# work geometry (MEASURED rule: compute_streaming.hpp:2259-2266 narrows the last visited k chunk of a causal
# q chunk to target_active_Sk = q_start_tile + Sq_chunk_t - k_chunk * Sk_chunk_t; active_Sk drives the QK^T
# sub-block count (l.1276), the exp and the P.V matmul (l.1674, 1831-1857)).
# ---------------------------------------------------------------------------------------------------------
def geometry(cfg, r):
    """Per average core: steps, exp tiles, tile MACs (truncated and untruncated), q-tile steps, k-tile steps."""
    qct, kct = cfg.q_chunk // 32, cfg.k_chunk // 32
    v_head = cfg.v_head_dim or cfg.head_dim
    dct = M._ceil_div(cfg.head_dim, 32) + M._ceil_div(v_head, 32)
    Skv = cfg.kv_seq or cfg.S
    nq = M._ceil_div(cfg.S, cfg.q_chunk)
    K = M._ceil_div(Skv, cfg.k_chunk)
    Q = r.q_chunks_per_core
    if cfg.is_causal and Skv == cfg.S and not cfg.has_attn_mask and cfg.sliding_window == 0:
        visits = [min(K, M._ceil_div((i + 1) * qct, kct)) for i in range(nq)]
        active = [min(K * kct, (i + 1) * qct) for i in range(nq)]  # k tiles actually processed
    else:
        visits = [K] * nq
        active = [K * kct] * nq
    K_eff = sum(visits) / nq
    kt_per_chunk = sum(active) / nq  # truncated k tiles per q chunk
    steps = Q * K_eff
    return dict(
        qct=qct,
        kct=kct,
        dct=dct,
        Q=Q,
        K=K,
        K_eff=K_eff,
        kt_per_chunk=kt_per_chunk,
        steps=steps,
        exp_tiles=Q * qct * kt_per_chunk,
        exp_tiles_untrunc=steps * qct * kct,
        tile_macs=Q * qct * kt_per_chunk * dct,
        tile_macs_untrunc=steps * qct * kct * dct,
        qtile_steps=steps * qct,
        ktile_steps=Q * kt_per_chunk,
        qtiles=Q * qct,
        rescale_qtiles=Q * max(0.0, K_eff - 1.0) * qct,
        cpt=cfg.arch.cpt(cfg.fidelity),
        steps_wc=r.steps_wall_core,
        chunks_wc=r.q_chunks_wall_core,
    )


# ---------------------------------------------------------------------------------------------------------
# refit laws (coefficients filled by the fits below, then used by the patched model)
# ---------------------------------------------------------------------------------------------------------
REFIT = dict(
    f0=None,
    bq=None,  # FPU = tile_macs*cpt*(1+f0) + bq*cpt*qtile_steps
    delta=None,
    beta=None,
    gamma=None,
    alpha=None,
    recip=17.0,  # SFPU law
    overlap=dict(causal=(None, None), noncausal=(None, None)),  # (r_sat, p): hidden = r_sat*(1-(2/qct)^p)*FPU
    init=2900.0,
)


def fpu_law(g, cpt, f0, bq):
    return g["tile_macs"] * cpt * (1.0 + f0) + bq * cpt * g["qtile_steps"]


def sfpu_law(g, delta, beta, gamma, alpha, recip=17.0):
    return (
        delta * g["exp_tiles"]
        + beta * g["qtile_steps"]
        + gamma * g["ktile_steps"]
        + alpha * g["steps"]
        + recip * g["qtiles"]
    )


def hidden_law(qct, fpu, sfpu, r_sat, p):
    return min(sfpu, max(0.0, r_sat * (1.0 - (2.0 / qct) ** p)) * fpu)


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def main():
    rows = []
    for label, tag, kw, group in CONFIGS:
        cfg = mk(**kw)
        r = M.predict(cfg)
        g = geometry(cfg, r)
        c = load_counters(tag)
        row = dict(
            label=label,
            tag=tag,
            group=group,
            causal=bool(cfg.is_causal),
            q_chunk=cfg.q_chunk,
            k_chunk=cfg.k_chunk,
            fidelity=cfg.fidelity,
            cores=cfg.num_cores,
            **{k: v for k, v in c.items() if k != "tag"},
            m_fpu=r.fpu_cycles,
            m_sfpu=r.sfpu_cycles,
            m_union=r.math_active_cycles,
            m_floor=r.compute_latency_cycles,
            m_wall=r.wall_clock_cycles,
            m_overlap=r.overlap_frac,
            m_steps=r.inner_iters,
            m_steps_wc=r.steps_wall_core,
            m_fpu_matmul=r.fpu_matmul_cycles,
            m_fpu_overhead=r.fpu_overhead_cycles,
            m_sfpu_exp=r.sfpu_exp_cycles,
            m_sfpu_reduce=r.sfpu_reduce_cycles,
            m_sfpu_recip=r.sfpu_recip_cycles,
            m_sfpu_ovh=r.sfpu_overhead_cycles,
            kernel_path=r.kernel_path,
            **g,
        )
        for k, mk_ in [("FPU_COUNTER", "m_fpu"), ("SFPU_COUNTER", "m_sfpu"), ("MATH_COUNTER", "m_union")]:
            row[k + "_wall_scaled_model"] = row[mk_] * r.steps_wall_core / r.inner_iters
        rows.append(row)
    T = pd.DataFrame(rows).set_index("label")
    T["hidden"] = T.FPU_COUNTER_mean + T.SFPU_COUNTER_mean - T.MATH_COUNTER_mean
    T["o_meas"] = T.hidden / np.minimum(T.FPU_COUNTER_mean, T.SFPU_COUNTER_mean)
    T["r_meas"] = T.hidden / T.FPU_COUNTER_mean
    T["fpu_per_tilemac_trunc"] = T.FPU_COUNTER_mean / T.tile_macs
    T["fpu_per_tilemac_untrunc"] = T.FPU_COUNTER_mean / T.tile_macs_untrunc

    # ---------------- Table 1: measured vs current model -------------------------------------------------
    print("\n### T1 measured counters (mean core / wall core) vs current model (HEAD f1adecb)\n")
    hdr = (
        "| config | FPU meas mean | model FPU | err % | SFPU meas mean | model SFPU | err % | MATH meas mean | model union | err % | "
        "FPU wall | model FPU wall | MATH wall | model union wall | STEP_N wall | model steps wall | mp wall |"
    )
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for lab, t in T.iterrows():
        print(
            f"| {lab} | {t.FPU_COUNTER_mean:,.0f} | {t.m_fpu:,.0f} | {pct(t.m_fpu, t.FPU_COUNTER_mean):+.1f} | "
            f"{t.SFPU_COUNTER_mean:,.0f} | {t.m_sfpu:,.0f} | {pct(t.m_sfpu, t.SFPU_COUNTER_mean):+.1f} | "
            f"{t.MATH_COUNTER_mean:,.0f} | {t.m_union:,.0f} | {pct(t.m_union, t.MATH_COUNTER_mean):+.1f} | "
            f"{t.FPU_COUNTER_wall:,.0f} | {t.FPU_COUNTER_wall_scaled_model:,.0f} | {t.MATH_COUNTER_wall:,.0f} | "
            f"{t.MATH_COUNTER_wall_scaled_model:,.0f} | {t.STEP_N_wall:.0f} | {t.m_steps_wc:.0f} | {t.wall_mp_mean:,.0f} |"
        )

    # ---------------- geometry table -----------------------------------------------------------------------
    print("\n### G work per average core (truncated rule) and measured FPU per tile MAC\n")
    print(
        "| config | Q chunks/core | K_eff steps/chunk | active k tiles/chunk | steps | exp tiles | tile MACs (trunc) | tile MACs (untrunc) | "
        "FPU/tileMAC trunc | FPU/tileMAC untrunc | SFPU/exp tile | hidden | o_meas | r_meas = hidden/FPU |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, t in T.iterrows():
        print(
            f"| {lab} | {t.Q:.3f} | {t.K_eff:.3f} | {t.kt_per_chunk:.3f} | {t.steps:.1f} | {t.exp_tiles:,.1f} | {t.tile_macs:,.0f} | "
            f"{t.tile_macs_untrunc:,.0f} | {t.fpu_per_tilemac_trunc:.2f} | {t.fpu_per_tilemac_untrunc:.2f} | "
            f"{t.SFPU_COUNTER_mean / t.exp_tiles:.1f} | {t.hidden:,.0f} | {t.o_meas:.3f} | {t.r_meas:.3f} |"
        )

    # ---------------- FPU fit -------------------------------------------------------------------------------
    fit_set = T[(T.group == "grid1") | (T.group == "cross")]  # 15 HiFi2 streaming-kernel points
    y = fit_set.FPU_COUNTER_mean.values
    X = np.column_stack(
        [fit_set.tile_macs.values * fit_set.cpt.values, fit_set.qtile_steps.values * fit_set.cpt.values]
    )
    Wt = 1.0 / y
    coef, *_ = np.linalg.lstsq(X * Wt[:, None], y * Wt, rcond=None)
    A_rel, bq = coef
    f0 = A_rel - 1.0
    REFIT["f0"], REFIT["bq"] = float(f0), float(bq)
    X2 = np.column_stack(
        [fit_set.tile_macs_untrunc.values * fit_set.cpt.values, fit_set.qtile_steps.values * fit_set.cpt.values]
    )
    coef2, *_ = np.linalg.lstsq(X2 * Wt[:, None], y * Wt, rcond=None)
    X3 = np.column_stack([fit_set.tile_macs.values * fit_set.cpt.values, fit_set.steps.values * fit_set.cpt.values])
    coef3, *_ = np.linalg.lstsq(X3 * Wt[:, None], y * Wt, rcond=None)
    X4 = fit_set.tile_macs.values[:, None] * fit_set.cpt.values[:, None]
    coef4, *_ = np.linalg.lstsq(X4 * Wt[:, None], y * Wt, rcond=None)
    print("\n### FPU fit (15 HiFi2 points: 14 grid 1 + cross), FPU = tile_macs*cpt*(1+f0) + bq*cpt*qtile_steps")
    print(
        f"f0 = {f0:.4f}  (cpt*(1+f0) = {32 * (1 + f0):.2f} cycles per HiFi2 tile MAC), bq = {bq:.4f} tile-MAC equivalents per q-tile step "
        f"= {bq * 32:.1f} cycles per q tile per step at HiFi2"
    )
    print(
        f"alternatives: untruncated tiles -> f0 {coef2[0] - 1:.4f}, bq {coef2[1]:.4f}; per-step term -> f0 {coef3[0] - 1:.4f}, "
        f"per step {coef3[1] * 32:.1f} cyc; pure fraction -> f0 {coef4[0] - 1:.4f}"
    )
    T["fpu_refit"] = [fpu_law(t, t.cpt, f0, bq) for _, t in T.iterrows()]
    T["fpu_alt_untrunc"] = [
        t.tile_macs_untrunc * t.cpt * coef2[0] + coef2[1] * t.cpt * t.qtile_steps for _, t in T.iterrows()
    ]
    T["fpu_alt_perstep"] = [t.tile_macs * t.cpt * coef3[0] + coef3[1] * t.cpt * t.steps for _, t in T.iterrows()]
    T["fpu_alt_frac"] = [t.tile_macs * t.cpt * coef4[0] for _, t in T.iterrows()]
    print(
        "\n| config | FPU meas | refit | err % | untrunc alt err % | per-step alt err % | pure-fraction alt err % | current model err % |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for lab, t in T.iterrows():
        print(
            f"| {lab} | {t.FPU_COUNTER_mean:,.0f} | {t.fpu_refit:,.0f} | {pct(t.fpu_refit, t.FPU_COUNTER_mean):+.2f} | "
            f"{pct(t.fpu_alt_untrunc, t.FPU_COUNTER_mean):+.2f} | {pct(t.fpu_alt_perstep, t.FPU_COUNTER_mean):+.2f} | "
            f"{pct(t.fpu_alt_frac, t.FPU_COUNTER_mean):+.2f} | {pct(t.m_fpu, t.FPU_COUNTER_mean):+.2f} |"
        )
    print("\n### LLK overhead law: measured effective fraction FPU/(tile_macs*cpt) - 1 vs current law vs refit law")
    print(
        "| config | qct | kct | dct sum | measured f | current law | refit f0 + bq/(active kct x dct) | current FPU err % | refit FPU err % |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for lab, t in T.iterrows():
        a = M.ArchConfig()
        if t.group == "mla":
            cur = a.fpu_overhead_frac_mla
        else:
            cur = (
                a.fpu_overhead_frac
                + a.fpu_overhead_per_inv_dct * (1.0 / t.dct - 1.0 / 8.0)
                + a.fpu_overhead_per_inv_qct * (1.0 / t.qct - 0.25)
            )
        kct_act = t.kt_per_chunk / t.K_eff
        ref = f0 + bq / (kct_act * t.dct)
        meas = t.FPU_COUNTER_mean / (t.tile_macs * t.cpt) - 1.0
        print(
            f"| {lab} | {t.qct} | {t.kct} | {t.dct} | {meas:.3f} | {cur:.3f} | {ref:.3f} | {pct(t.m_fpu, t.FPU_COUNTER_mean):+.1f} | "
            f"{pct(t.fpu_refit, t.FPU_COUNTER_mean):+.1f} |"
        )

    # ---------------- SFPU fit -----------------------------------------------------------------------------
    y = fit_set.SFPU_COUNTER_mean.values - 17.0 * fit_set.qtiles.values
    X = np.column_stack(
        [fit_set.exp_tiles.values, fit_set.qtile_steps.values, fit_set.ktile_steps.values, fit_set.steps.values]
    )
    Wt = 1.0 / fit_set.SFPU_COUNTER_mean.values
    coef, *_ = np.linalg.lstsq(X * Wt[:, None], y * Wt, rcond=None)
    delta, beta, gamma, alpha = coef
    base = T.loc["causal q128 k128"]
    a6 = T.loc["causal q128 k128, A6 exp stub"]
    delta_a6 = (base.SFPU_COUNTER_mean - a6.SFPU_COUNTER_mean) / base.exp_tiles
    y2 = y - delta_a6 * fit_set.exp_tiles.values
    X2 = np.column_stack([fit_set.qtile_steps.values, fit_set.steps.values])  # gamma fixed at 0 (free fit gives -0.8)
    coef2, *_ = np.linalg.lstsq(X2 * Wt[:, None], y2 * Wt, rcond=None)
    X2g = np.column_stack([fit_set.qtile_steps.values, fit_set.ktile_steps.values, fit_set.steps.values])
    coef2g, *_ = np.linalg.lstsq(X2g * Wt[:, None], y2 * Wt, rcond=None)
    X3 = np.column_stack([fit_set.exp_tiles.values, fit_set.qtile_steps.values, fit_set.ktile_steps.values])
    coef3, *_ = np.linalg.lstsq(X3 * Wt[:, None], y * Wt, rcond=None)
    print(
        "\n### SFPU fit (15 HiFi2 approx-exp points), SFPU = delta*exp_tiles + beta*qtile_steps + gamma*ktile_steps + alpha*steps + 17*qtiles"
    )
    print(
        f"free fit: delta {delta:.2f} per exp tile, beta {beta:.1f} per q tile per step, gamma {gamma:.2f} per k tile per step, alpha {alpha:.1f} per step"
    )
    print(
        f"A6 direct: delta {delta_a6:.2f} per exp tile (SFPU {base.SFPU_COUNTER_mean:,.0f} - {a6.SFPU_COUNTER_mean:,.0f} over {base.exp_tiles:,.1f} exp tiles)"
    )
    print(f"constrained (delta = A6, gamma = 0): beta {coef2[0]:.1f}, alpha {coef2[1]:.1f}   [ADOPTED]")
    print(f"constrained (delta = A6, gamma free): beta {coef2g[0]:.1f}, gamma {coef2g[1]:.2f}, alpha {coef2g[2]:.1f}")
    print(f"3-parameter (alpha = 0): delta {coef3[0]:.2f}, beta {coef3[1]:.1f}, gamma {coef3[2]:.2f}")
    REFIT.update(delta=float(delta_a6), beta=float(coef2[0]), gamma=0.0, alpha=float(coef2[1]))
    REFIT["sfpu_a6_gamma_free"] = dict(beta=float(coef2g[0]), gamma=float(coef2g[1]), alpha=float(coef2g[2]))
    REFIT["sfpu_free_fit"] = dict(delta=float(delta), beta=float(beta), gamma=float(gamma), alpha=float(alpha))
    T["sfpu_refit"] = [
        sfpu_law(t, REFIT["delta"], REFIT["beta"], REFIT["gamma"], REFIT["alpha"]) for _, t in T.iterrows()
    ]
    T["sfpu_free"] = [sfpu_law(t, delta, beta, gamma, alpha) for _, t in T.iterrows()]
    print(
        "\n| config | SFPU meas | refit (A6 delta) | err % | free fit err % | current model err % | current non-exp per step | refit non-exp per step |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for lab, t in T.iterrows():
        exp_cyc = 88.2 if (t.fidelity == "HiFi2" and t.group not in ("mla", "prod")) else 77.9
        cur_ns = (t.m_sfpu - exp_cyc * t.exp_tiles_untrunc) / t.steps
        ref_ns = (REFIT["beta"] * t.qtile_steps + REFIT["gamma"] * t.ktile_steps + REFIT["alpha"] * t.steps) / t.steps
        print(
            f"| {lab} | {t.SFPU_COUNTER_mean:,.0f} | {t.sfpu_refit:,.0f} | {pct(t.sfpu_refit, t.SFPU_COUNTER_mean):+.2f} | "
            f"{pct(t.sfpu_free, t.SFPU_COUNTER_mean):+.2f} | {pct(t.m_sfpu, t.SFPU_COUNTER_mean):+.2f} | {cur_ns:,.0f} | {ref_ns:,.0f} |"
        )

    # ---------------- overlap -------------------------------------------------------------------------------
    print(
        "\n### Overlap: measured hidden = FPU + SFPU - MATH; o = hidden/min(FPU,SFPU) (the model's definition), r = hidden/FPU"
    )
    print(
        "| config | qct | kct | FPU | SFPU | MATH | hidden | o_meas | model o (1 - k/qct) | r_meas | hidden per step | FPU per step | SFPU per step |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, t in T.iterrows():
        print(
            f"| {lab} | {t.qct} | {t.kct} | {t.FPU_COUNTER_mean:,.0f} | {t.SFPU_COUNTER_mean:,.0f} | {t.MATH_COUNTER_mean:,.0f} | {t.hidden:,.0f} | "
            f"{t.o_meas:.3f} | {t.m_overlap:.3f} | {t.r_meas:.3f} | {t.hidden / t.steps:,.0f} | {t.FPU_COUNTER_mean / t.steps:,.0f} | "
            f"{t.SFPU_COUNTER_mean / t.steps:,.0f} |"
        )
    d_r = {}
    for reg, ztag, utag in [
        ("causal", "causal q128 k128", "causal q128 k128, unmodified kernel"),
        ("noncausal", "non-causal q128 k128", "non-causal q128 k128, unmodified kernel"),
    ]:
        d_r[reg] = float(T.loc[utag, "r_meas"] - T.loc[ztag, "r_meas"])
    print(
        f"\nzone effect on r at the anchors (unmodified minus zoned): causal {d_r['causal']:+.4f}, non-causal {d_r['noncausal']:+.4f}"
    )
    g1 = T[T.group == "grid1"].copy()
    g1["reg"] = np.where(g1.causal, "causal", "noncausal")
    g1["r_unmod"] = g1.r_meas + g1.reg.map(d_r)
    g1.loc[g1.qct == 2, "r_unmod"] = g1.loc[g1.qct == 2, "r_meas"]  # r(2) kept at the measured zero (mechanism)
    for reg, ztag, utag in [
        ("causal", "causal q128 k128", "causal q128 k128, unmodified kernel"),
        ("noncausal", "non-causal q128 k128", "non-causal q128 k128, unmodified kernel"),
    ]:
        g1.loc[ztag, "r_unmod"] = T.loc[utag, "r_meas"]
    g1["hidden_unmod"] = g1.r_unmod * g1.FPU_COUNTER_mean
    g1["math_unmod"] = g1.FPU_COUNTER_mean + g1.SFPU_COUNTER_mean - g1.hidden_unmod

    def union_err(pred_hidden, meas_math, fpu, sfpu):
        return 100.0 * ((fpu + sfpu - pred_hidden) / meas_math - 1.0)

    fits = {}
    for basis, mcol in [("zoned", "MATH_COUNTER_mean"), ("unmodified", "math_unmod")]:
        fits[basis] = {}
        for reg in ["causal", "noncausal"]:
            s = g1[g1.reg == reg]
            fpu, sfpu, math_m, qct = s.FPU_COUNTER_mean.values, s.SFPU_COUNTER_mean.values, s[mcol].values, s.qct.values
            best = None
            for k in np.arange(0.0, 8.0, 0.001):
                o = np.clip(1.0 - k / qct, 0.0, 0.95)
                e = union_err(o * np.minimum(fpu, sfpu), math_m, fpu, sfpu)
                sc = np.sqrt(np.mean(e**2))
                if best is None or sc < best[0]:
                    best = (sc, k, e)
            best2 = None
            for rs in np.arange(0.0, 0.8, 0.0005):
                h = np.array([hidden_law(q, f, sf, rs, 1.0) for q, f, sf in zip(qct, fpu, sfpu)])
                e = union_err(h, math_m, fpu, sfpu)
                sc = np.sqrt(np.mean(e**2))
                if best2 is None or sc < best2[0]:
                    best2 = (sc, rs, e)
            best3 = None
            for p in np.arange(0.5, 3.01, 0.05):
                for rs in np.arange(0.0, 0.8, 0.0005):
                    h = np.array([hidden_law(q, f, sf, rs, p) for q, f, sf in zip(qct, fpu, sfpu)])
                    e = union_err(h, math_m, fpu, sfpu)
                    sc = np.sqrt(np.mean(e**2))
                    if best3 is None or sc < best3[0]:
                        best3 = (sc, rs, p, e)
            best4 = None
            for rs in np.arange(0.0, 0.8, 0.0005):
                h = np.array([hidden_law(q, f, sf, rs, 2.0) for q, f, sf in zip(qct, fpu, sfpu)])
                e = union_err(h, math_m, fpu, sfpu)
                sc = np.sqrt(np.mean(e**2))
                if best4 is None or sc < best4[0]:
                    best4 = (sc, rs, e)
            cur_k = M.OVERLAP_K_CAUSAL if reg == "causal" else M.OVERLAP_K_NONCAUSAL
            o_cur = np.clip(1.0 - cur_k / qct, 0.0, 0.95)
            e_cur = union_err(o_cur * np.minimum(fpu, sfpu), math_m, fpu, sfpu)
            fits[basis][reg] = dict(
                F4_rsat=float(best4[1]),
                F4_err=dict(zip(s.index, best4[2])),
                F4_rms=float(best4[0]),
                F1_k=float(best[1]),
                F1_err=dict(zip(s.index, best[2])),
                F1_rms=float(best[0]),
                F2_rsat=float(best2[1]),
                F2_err=dict(zip(s.index, best2[2])),
                F2_rms=float(best2[0]),
                F3_rsat=float(best3[1]),
                F3_p=float(best3[2]),
                F3_err=dict(zip(s.index, best3[3])),
                F3_rms=float(best3[0]),
                cur_err=dict(zip(s.index, e_cur)),
                cur_rms=float(np.sqrt(np.mean(e_cur**2))),
            )
    for basis in fits:
        print(
            f"\n### Overlap fits on the {basis} basis (union error % per config = (FPU + SFPU - hidden_model)/MATH - 1)"
        )
        print("| regime | law | parameters | rms % | per-config errors |")
        print("|---|---|---|---|---|")
        for reg in ["causal", "noncausal"]:
            f = fits[basis][reg]
            cur_k = M.OVERLAP_K_CAUSAL if reg == "causal" else M.OVERLAP_K_NONCAUSAL

            def fmt(d):
                return ", ".join(
                    f"{k.replace('non-causal ', '').replace('causal ', '')}: {v:+.1f}" for k, v in d.items()
                )

            print(
                f"| {reg} | current 1 - k/qct on min(FPU,SFPU) | k {cur_k} | {f['cur_rms']:.1f} | {fmt(f['cur_err'])} |"
            )
            print(f"| {reg} | F1 refit k, same form | k {f['F1_k']:.2f} | {f['F1_rms']:.1f} | {fmt(f['F1_err'])} |")
            print(
                f"| {reg} | F2 hidden = r (1 - 2/qct) FPU, cap SFPU | r_sat {f['F2_rsat']:.3f} | {f['F2_rms']:.1f} | {fmt(f['F2_err'])} |"
            )
            print(
                f"| {reg} | F3 hidden = r (1 - (2/qct)^p) FPU, cap SFPU | r_sat {f['F3_rsat']:.3f}, p {f['F3_p']:.2f} | {f['F3_rms']:.1f} | {fmt(f['F3_err'])} |"
            )
            print(
                f"| {reg} | F4 hidden = r (1 - 4/qct^2) FPU, cap SFPU (p = 2 fixed) | r_sat {f['F4_rsat']:.3f} | {f['F4_rms']:.1f} | {fmt(f['F4_err'])} |"
            )
    for reg in ["causal", "noncausal"]:
        REFIT["overlap"][reg] = (fits["unmodified"][reg]["F4_rsat"], 2.0)  # ADOPTED: p fixed at 2, unmodified basis
    REFIT["overlap_zoned_basis"] = {reg: (fits["zoned"][reg]["F4_rsat"], 2.0) for reg in ["causal", "noncausal"]}
    REFIT["overlap_free_p"] = {
        reg: (fits["unmodified"][reg]["F3_rsat"], fits["unmodified"][reg]["F3_p"]) for reg in ["causal", "noncausal"]
    }
    REFIT["d_r_anchor"] = d_r

    def union_refit(t, fpu, sfpu):
        if t.kernel_path == "legacy" or t.group == "mla":
            return fpu + sfpu
        rs, p = REFIT["overlap"]["causal" if t.causal else "noncausal"]
        return fpu + sfpu - hidden_law(t.qct, fpu, sfpu, rs, p)

    T["union_refit"] = [union_refit(t, t.fpu_refit, t.sfpu_refit) for _, t in T.iterrows()]
    T["math_unmod_basis"] = [
        g1.loc[lab, "math_unmod"] if lab in g1.index else t.MATH_COUNTER_mean for lab, t in T.iterrows()
    ]
    cx = "cross 2k/8k nh16 q128 k128"  # zoned-only capture on the non-causal kernel: same shift as the non-causal anchor (INFERRED)
    T.loc[cx, "math_unmod_basis"] = T.loc[cx, "MATH_COUNTER_mean"] - d_r["noncausal"] * T.loc[cx, "FPU_COUNTER_mean"]
    print("\n### Union with the refit constants (FPU law + SFPU law + F3 overlap on the unmodified basis)")
    print(
        "| config | MATH meas (zoned) | MATH unmodified basis | current union | err vs zoned % | err vs unmod % | refit union | err vs zoned % | err vs unmod % |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for lab, t in T.iterrows():
        print(
            f"| {lab} | {t.MATH_COUNTER_mean:,.0f} | {t.math_unmod_basis:,.0f} | {t.m_union:,.0f} | {pct(t.m_union, t.MATH_COUNTER_mean):+.1f} | "
            f"{pct(t.m_union, t.math_unmod_basis):+.1f} | {t.union_refit:,.0f} | {pct(t.union_refit, t.MATH_COUNTER_mean):+.1f} | "
            f"{pct(t.union_refit, t.math_unmod_basis):+.1f} |"
        )

    for col in ["fpu_refit", "sfpu_refit", "union_refit"]:
        T[col + "_wall"] = T[col] * T.m_steps_wc / T.m_steps
    print(
        "\n### Wall-setting core: measured counters on the mp run's wall core vs current and refit model scaled by the core's steps"
    )
    print(
        "| config | steps wall core | FPU wall meas | current | err % | refit | err % | SFPU wall meas | refit | err % | MATH wall meas (zoned where zoned) | current union | err % | refit union | err % |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, t in T.iterrows():
        print(
            f"| {lab} | {t.m_steps_wc:.0f} | {t.FPU_COUNTER_wall:,.0f} | {t.FPU_COUNTER_wall_scaled_model:,.0f} | {pct(t.FPU_COUNTER_wall_scaled_model, t.FPU_COUNTER_wall):+.1f} | "
            f"{t.fpu_refit_wall:,.0f} | {pct(t.fpu_refit_wall, t.FPU_COUNTER_wall):+.1f} | {t.SFPU_COUNTER_wall:,.0f} | {t.sfpu_refit_wall:,.0f} | "
            f"{pct(t.sfpu_refit_wall, t.SFPU_COUNTER_wall):+.1f} | {t.MATH_COUNTER_wall:,.0f} | {t.MATH_COUNTER_wall_scaled_model:,.0f} | "
            f"{pct(t.MATH_COUNTER_wall_scaled_model, t.MATH_COUNTER_wall):+.1f} | {t.union_refit_wall:,.0f} | {pct(t.union_refit_wall, t.MATH_COUNTER_wall):+.1f} |"
        )
    g14 = T[T.group == "grid1"]
    print("\n### Summary over the 14 grid 1 configs: |error| mean / max, current vs refit")
    for name, cur, ref, meas in [
        ("FPU", "m_fpu", "fpu_refit", "FPU_COUNTER_mean"),
        ("SFPU", "m_sfpu", "sfpu_refit", "SFPU_COUNTER_mean"),
        ("union vs zoned MATH", "m_union", "union_refit", "MATH_COUNTER_mean"),
        ("union vs unmodified-basis MATH", "m_union", "union_refit", "math_unmod_basis"),
    ]:
        ec = np.abs(pct(g14[cur], g14[meas]))
        er = np.abs(pct(g14[ref], g14[meas]))
        print(f"{name}: current mean {ec.mean():.1f} max {ec.max():.1f}; refit mean {er.mean():.1f} max {er.max():.1f}")

    # ---------------- production points without counters (zone proxies) --------------------------------------
    print(
        "\n### Production points without counters: zone QK_MM + PV_MM on the wall core (MATH thread) vs model FPU scaled to the wall core"
    )
    print(
        "| point | zone QK_MM + PV_MM (wall core) | current model FPU (wall core) | err % | refit FPU (wall core) | err % | steps wall core |"
    )
    print("|---|---|---|---|---|---|---|")
    for label, kw, z in PROD_ZONE_ONLY:
        cfg = mk(**kw)
        r = M.predict(cfg)
        g = geometry(cfg, r)
        scale = z["steps_wc"] / g["steps"]
        cur = r.fpu_cycles * scale
        ref = fpu_law(g, g["cpt"], REFIT["f0"], REFIT["bq"]) * scale
        zone = z["QK_MM"] + z["PV_MM"]
        print(
            f"| {label} | {zone:,.0f} | {cur:,.0f} | {pct(cur, zone):+.1f} | {ref:,.0f} | {pct(ref, zone):+.1f} | {z['steps_wc']} |"
        )

    # ---------------- decode (supplementary) ---------------------------------------------------------------
    dc = load_counters("t28_decode_b32_g64_pos1024_mp")
    rd = M.predict_decode(arch=M.ArchConfig(), **DECODE_KW)
    print(
        "\n### Decode T2.8 b32 g64 pos1024 (supplementary; the decode floor is not a wall term, the wall is memory bound)"
    )
    print(
        f"measured mean core: FPU {dc['FPU_COUNTER_mean']:,.0f}, SFPU {dc['SFPU_COUNTER_mean']:,.0f}, MATH {dc['MATH_COUNTER_mean']:,.0f} "
        f"(FPU + SFPU = {dc['FPU_COUNTER_mean'] + dc['SFPU_COUNTER_mean']:,.0f}); mp wall {dc['wall_mp_mean']:,.0f}; active cores {dc['active_cores']}"
    )
    print(
        f"model predict_decode: FPU {rd.fpu_cycles:,}, SFPU {rd.sfpu_cycles:,}, union {rd.math_active_cycles:,}, wall {rd.wall_clock_cycles:,}, "
        f"k_chunk {rd.config_echo['k_chunk']}, heads per core {rd.config_echo['heads_per_core']}, active cores {rd.active_cores}, "
        f"k tiles per core {rd.k_eff:.0f}"
    )
    REFIT["decode"] = dict(
        meas=dict(
            FPU=dc["FPU_COUNTER_mean"],
            SFPU=dc["SFPU_COUNTER_mean"],
            MATH=dc["MATH_COUNTER_mean"],
            wall=dc["wall_mp_mean"],
        ),
        model=dict(FPU=rd.fpu_cycles, SFPU=rd.sfpu_cycles, union=rd.math_active_cycles, wall=rd.wall_clock_cycles),
    )

    # ---------------- patched model: walls with the refit adopted ---------------------------------------------
    orig_engine, orig_predict = M._engine_cycles, M.predict
    state = {}

    def engine_refit(
        r,
        *,
        Q,
        K_eff,
        K_eff_gt0,
        inner_iters,
        qct,
        kct,
        dct_qk,
        dct_v,
        cpt,
        exp_approx,
        arch,
        fpu_overhead_frac,
        sfpu_overhead_per_inner_iter,
        overlap_frac,
        mask_add_tiles_per_iter=0.0,
        sink_passes_per_q=0.0,
        combine_passes=0.0,
    ):
        cfg = state["cfg"]
        g = state["geom"]
        r.fpu_matmul_cycles = round(g["tile_macs"] * cpt)
        r.fpu_overhead_cycles = round(g["tile_macs"] * cpt * REFIT["f0"] + REFIT["bq"] * cpt * g["qtile_steps"])
        r.fpu_cycles = r.fpu_matmul_cycles + r.fpu_overhead_cycles
        r.sfpu_exp_cycles = round(REFIT["delta"] * g["exp_tiles"])
        r.sfpu_reduce_cycles = round(REFIT["gamma"] * g["ktile_steps"])
        r.sfpu_recip_cycles = round(REFIT["recip"] * g["qtiles"])
        r.sfpu_overhead_cycles = round(REFIT["beta"] * g["qtile_steps"] + REFIT["alpha"] * g["steps"])
        r.sfpu_extra_cycles = 0
        r.sfpu_cycles = r.sfpu_exp_cycles + r.sfpu_reduce_cycles + r.sfpu_recip_cycles + r.sfpu_overhead_cycles
        if r.kernel_path == "legacy" or r.is_mla:
            hidden = 0.0
        else:
            rs, p = REFIT["overlap"]["causal" if (cfg.is_causal or cfg.has_attn_mask) else "noncausal"]
            hidden = hidden_law(qct, r.fpu_cycles, r.sfpu_cycles, rs, p)
        r.overlap_frac = hidden / min(r.fpu_cycles, r.sfpu_cycles) if min(r.fpu_cycles, r.sfpu_cycles) else 0.0
        r.math_active_cycles = round(r.fpu_cycles + r.sfpu_cycles - hidden)
        r.serial_sum_cycles = r.fpu_cycles + r.sfpu_cycles

    def predict_refit(cfg, **kw):
        r0 = orig_predict(cfg, **kw)  # first pass: the factory split (q chunks per core)
        state["cfg"], state["geom"] = cfg, geometry(cfg, r0)
        cfg2 = replace(cfg, arch=replace(cfg.arch, init_overhead_cycles=REFIT["init"]))
        M._engine_cycles = engine_refit
        try:
            return orig_predict(cfg2, **kw)
        finally:
            M._engine_cycles = orig_engine

    print(
        "\n### Effect of adopting the refit on the anchor floor and the walls (patched model in memory; polaris untouched)"
    )
    print(
        "| config | measured wall (zones off) | current wall | err % | refit wall | err % | current union | refit union | current floor (+30000) | refit floor (+2900) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    walls = {}
    ds = pd.read_csv(ROOT / "data" / "t42_scripts" / "decomp_summary_t21_t22.csv", comment="#").set_index("tag")
    wall_meas = {
        lab: float(ds.loc[tag.replace("_zon_mp", ""), "wall_zoff_mean"])
        for lab, tag, kw, gr in CONFIGS
        if gr == "grid1"
    }
    wall_meas["cross 2k/8k nh16 q128 k128"] = 1649702.0
    wall_meas["MLA nh16 S2048 q32 k128 HiFi4"] = 9574958.0
    wall_meas["production S4096 q256 k256 g64 HiFi4 fp32"] = 3983028.0
    for label, tag, kw, group in CONFIGS:
        if group in ("unmod", "a6"):
            continue
        cfg = mk(**kw)
        r0 = M.predict(cfg)
        r1 = predict_refit(cfg)
        wm = wall_meas[label]
        walls[label] = dict(
            meas=wm,
            cur=r0.wall_clock_cycles,
            ref=r1.wall_clock_cycles,
            cur_union=r0.math_active_cycles,
            ref_union=r1.math_active_cycles,
            cur_floor=r0.compute_latency_cycles,
            ref_floor=r1.compute_latency_cycles,
            cur_fpu=r0.fpu_cycles,
            ref_fpu=r1.fpu_cycles,
            cur_sfpu=r0.sfpu_cycles,
            ref_sfpu=r1.sfpu_cycles,
            cur_comp={k: round(v) for k, v in r0.components.items()},
            ref_comp={k: round(v) for k, v in r1.components.items()},
        )
        print(
            f"| {label} | {wm:,.0f} | {r0.wall_clock_cycles:,} | {pct(r0.wall_clock_cycles, wm):+.1f} | {r1.wall_clock_cycles:,} | "
            f"{pct(r1.wall_clock_cycles, wm):+.1f} | {r0.math_active_cycles:,} | {r1.math_active_cycles:,} | {r0.compute_latency_cycles:,} | "
            f"{r1.compute_latency_cycles:,} |"
        )
    for label, kw, z in PROD_ZONE_ONLY:
        cfg = mk(**kw)
        r0 = M.predict(cfg)
        r1 = predict_refit(cfg)
        walls[label] = dict(
            meas=z["wall"],
            cur=r0.wall_clock_cycles,
            ref=r1.wall_clock_cycles,
            cur_union=r0.math_active_cycles,
            ref_union=r1.math_active_cycles,
            cur_floor=r0.compute_latency_cycles,
            ref_floor=r1.compute_latency_cycles,
            cur_fpu=r0.fpu_cycles,
            ref_fpu=r1.fpu_cycles,
            cur_sfpu=r0.sfpu_cycles,
            ref_sfpu=r1.sfpu_cycles,
            cur_comp={k: round(v) for k, v in r0.components.items()},
            ref_comp={k: round(v) for k, v in r1.components.items()},
        )
        print(
            f"| {label} | {z['wall']:,} | {r0.wall_clock_cycles:,} | {pct(r0.wall_clock_cycles, z['wall']):+.1f} | {r1.wall_clock_cycles:,} | "
            f"{pct(r1.wall_clock_cycles, z['wall']):+.1f} | {r0.math_active_cycles:,} | {r1.math_active_cycles:,} | {r0.compute_latency_cycles:,} | "
            f"{r1.compute_latency_cycles:,} |"
        )
    a0 = walls["causal q128 k128"]
    print(
        f"\nanchor: union {a0['cur_union']:,} -> {a0['ref_union']:,} ({pct(a0['ref_union'], a0['cur_union']):+.1f} %); "
        f"floor {a0['cur_floor']:,} -> {a0['ref_floor']:,}; wall {a0['cur']:,} -> {a0['ref']:,}"
    )
    print("anchor components current:", a0["cur_comp"])
    print("anchor components refit:  ", a0["ref_comp"])

    # ---------------- outputs -------------------------------------------------------------------------------
    T.to_csv(ROOT / "data" / "floor_verification_configs.csv")
    with open(ROOT / "data" / "floor_verification_fits.json", "w") as f:
        json.dump(dict(REFIT=REFIT, overlap_fits=fits, walls=walls), f, indent=1, default=float)
    print("\nwrote", ROOT / "data" / "floor_verification_configs.csv", "and floor_verification_fits.json")
    print("\nREFIT constants:", json.dumps({k: v for k, v in REFIT.items() if k != "decode"}, indent=1, default=float))
    return T, walls


if __name__ == "__main__":
    main()
