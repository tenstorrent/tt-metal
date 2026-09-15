#!/usr/bin/env python
"""Floor constants on the R1f unmodified-kernel perf counters (model/refit_r2_notes.md s2.1).

Reads data/bh_zones/r1f_counters_table.csv (mean over the active cores, run_idx 1 and 2 averaged) and fits:
the FPU law (f0, tile-MAC equivalents per q tile per step) and the SFPU law on the 14 grid 1 configs, the
overlap ratio r per regime on the 14 unmodified MATH counters (hidden = r (1 - 4 / qct^2) FPU capped at the
SFPU), LoFi and HiFi3 cycles per tile MAC from the fidelity captures at the anchor, the accurate exp cost per
tile on the streaming kernel (anchor capture) and on the fp32 DEST kernel (T2.4 g64 and R1f g110 production
captures), and reports the head_dim 64 and A10 checks. Read-only on data/; prints the tables.
    $POLARIS/.venv/bin/python model/refit_r2_floor_r1f.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# POLARIS is the polaris checkout beside handoff/ in the workspace (PORTABLE_CONTRACT.md).
POLARIS = Path(os.environ.get("POLARIS", ROOT.parents[1] / "polaris"))
sys.path.insert(0, str(POLARIS))
import ttsim.perf.roofline_sdpa as M  # noqa: E402

GQA = dict(num_heads=32, num_kv_heads=8)
PROD = dict(
    num_heads=32, num_kv_heads=8, fidelity="HiFi4", exp_approx_mode=False, fp32_dest_acc=True, accum_dtype="float32"
)
TAGS = {
    "r1f_causal_q128k128_zoff_mp": dict(**GQA),
    "r1f_causal_q128k256_zoff_mp": dict(k_chunk=256, **GQA),
    "r1f_causal_q128k512_zoff_mp": dict(k_chunk=512, **GQA),
    "r1f_causal_q64k128_zoff_mp": dict(q_chunk=64, **GQA),
    "r1f_causal_q256k128_zoff_mp": dict(q_chunk=256, **GQA),
    "r1f_causal_q512k128_zoff_mp": dict(q_chunk=512, **GQA),
    "r1f_causal_q512k512_zoff_mp": dict(q_chunk=512, k_chunk=512, **GQA),
    "r1f_noncausal_q128k128_zoff_mp": dict(is_causal=False, **GQA),
    "r1f_noncausal_q128k256_zoff_mp": dict(is_causal=False, k_chunk=256, **GQA),
    "r1f_noncausal_q128k512_zoff_mp": dict(is_causal=False, k_chunk=512, **GQA),
    "r1f_noncausal_q64k128_zoff_mp": dict(is_causal=False, q_chunk=64, **GQA),
    "r1f_noncausal_q256k128_zoff_mp": dict(is_causal=False, q_chunk=256, **GQA),
    "r1f_noncausal_q512k128_zoff_mp": dict(is_causal=False, q_chunk=512, **GQA),
    "r1f_noncausal_q512k512_zoff_mp": dict(is_causal=False, q_chunk=512, k_chunk=512, **GQA),
    "r1f_causal_q128k128_lofi_zoff_mp": dict(fidelity="LoFi", **GQA),
    "r1f_causal_q128k128_hifi3_zoff_mp": dict(fidelity="HiFi3", **GQA),
    "r1f_causal_q128k128_expaccurate_zoff_mp": dict(exp_approx_mode=False, **GQA),
    "r1f_prod_causal_S4096_q256k256_g110_zoff_mp": dict(q_chunk=256, k_chunk=256, num_cores=110, **PROD),
    "r1f_a10_causal_S4096_q256k256_g64_fp32off_zoff_mp": dict(
        q_chunk=256, k_chunk=256, num_cores=64, **dict(PROD, fp32_dest_acc=False, accum_dtype="bfloat16")
    ),
    "r1c_causal_S4096_q128k128_hd64_zoff_mp": dict(head_dim=64, **GQA),
}
# the T2.4 g64 production capture (zoned kernel; its MATH = FPU + SFPU exactly, the legacy path has no PACK exp)
T24_PROD = dict(
    tag="t24_prod_causal_S4096_q256k256_g64",
    kw=dict(q_chunk=256, k_chunk=256, num_cores=64, **PROD),
    FPU=2418688.0,
    SFPU=442996.0,
    MATH=2861684.0,
)


def cfg(**kw):
    base = dict(S=4096, num_cores=110, fidelity="HiFi2", input_dtype="bfp8_b")
    base.update(kw)
    return M.SdpaConfig(arch=M.ArchConfig(), **base)


def work(c, r):
    """Per mean core: steps, exp tiles, tile MACs, q tile steps, q tiles (the truncated diagonal rule)."""
    qct, kct = c.q_chunk // 32, c.k_chunk // 32
    dct = -(-c.head_dim // 32) + -(-(c.v_head_dim or c.head_dim) // 32)
    K = -(-c.S // c.k_chunk)
    if c.is_causal:
        visits, tiles = M._causal_visits(c.S, qct, kct, K)
    else:
        nq = -(-c.S // c.q_chunk)
        visits, tiles = [float(K)] * nq, [float(K * kct)] * nq
    Q = r.q_chunks_per_core
    steps = Q * sum(visits) / len(visits)
    exp_tiles = Q * qct * sum(tiles) / len(tiles)
    return dict(
        qct=qct,
        kct=kct,
        dct=dct,
        steps=steps,
        exp_tiles=exp_tiles,
        tile_macs=exp_tiles * dct,
        qtile_steps=steps * qct,
        qtiles=Q * qct,
    )


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def main():
    t = pd.read_csv(ROOT / "data" / "bh_zones" / "r1f_counters_table.csv", comment="#")
    t = (
        t[t.run_idx > 0]
        .groupby("tag")[
            ["FPU_COUNTER_mean_active", "SFPU_COUNTER_mean_active", "MATH_COUNTER_mean_active", "wall_cycles"]
        ]
        .mean()
    )
    a = M.ArchConfig()
    rows = {}
    for tag, kw in TAGS.items():
        c = cfg(**kw)
        r = M.predict(c)
        g = work(c, r)
        m = t.loc[tag]
        rows[tag] = dict(
            kw=kw,
            c=c,
            r=r,
            g=g,
            FPU=m.FPU_COUNTER_mean_active,
            SFPU=m.SFPU_COUNTER_mean_active,
            MATH=m.MATH_COUNTER_mean_active,
            wall=m.wall_cycles,
        )
    grid = [
        k
        for k in TAGS
        if k.startswith(("r1f_causal_q", "r1f_noncausal_q"))
        and "lofi" not in k
        and "hifi3" not in k
        and "expacc" not in k
    ]
    # ---- FPU law on the 14 grid 1 configs (HiFi2 approx exp)
    y = np.array([rows[k]["FPU"] for k in grid])
    X = np.array([[rows[k]["g"]["tile_macs"] * 32.0, rows[k]["g"]["qtile_steps"] * 32.0] for k in grid])
    w = 1.0 / y
    (A, bq), *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    f0 = A - 1.0
    print(f"FPU law (14 R1f grid 1 configs): f0 {f0:.4f}, tile MACs per q tile per step {bq:.3f}")
    # ---- SFPU law (delta fixed at the A6 70.0, gamma 0)
    y = np.array([rows[k]["SFPU"] - 70.0 * rows[k]["g"]["exp_tiles"] - 17.0 * rows[k]["g"]["qtiles"] for k in grid])
    X = np.array([[rows[k]["g"]["qtile_steps"], rows[k]["g"]["steps"]] for k in grid])
    ws = 1.0 / np.array([rows[k]["SFPU"] for k in grid])
    (beta, alpha), *_ = np.linalg.lstsq(X * ws[:, None], y * ws, rcond=None)
    print(f"SFPU law (delta 70.0 from A6): per q tile per step {beta:.1f}, per step {alpha:.1f}")
    # ---- overlap ratios on the unmodified MATH counters
    print(
        "\n| config | FPU | SFPU | MATH (unmodified) | hidden | hidden / FPU | hidden / SFPU | r law (1 - 4/qct^2) needs r_sat |"
    )
    print("|---|---|---|---|---|---|---|---|")
    fits = {}
    for reg, causal in (("causal", True), ("noncausal", False)):
        ks = [k for k in grid if rows[k]["c"].is_causal == causal]
        fpu = np.array([rows[k]["FPU"] for k in ks])
        sfpu = np.array([rows[k]["SFPU"] for k in ks])
        math = np.array([rows[k]["MATH"] for k in ks])
        qct = np.array([rows[k]["g"]["qct"] for k in ks])
        for k, f_, s_, m_, q_ in zip(ks, fpu, sfpu, math, qct):
            h = f_ + s_ - m_
            need = (h / f_) / (1 - 4 / q_**2) if q_ > 2 else float("nan")
            print(
                f"| {k.replace('r1f_', '').replace('_zoff_mp', '')} | {f_:,.0f} | {s_:,.0f} | {m_:,.0f} | {h:,.0f} | {h / f_:.3f} | {h / s_:.3f} | {need:.3f} |"
            )
        best = None
        for rs in np.arange(0.15, 0.6, 0.0005):
            hidden = np.minimum(sfpu, np.maximum(0.0, rs * (1 - 4 / qct**2)) * fpu)
            e = 100 * ((fpu + sfpu - hidden) / math - 1)
            sc = np.sqrt(np.mean(e**2))
            if best is None or sc < best[0]:
                best = (sc, rs, e)
        fits[reg] = best
        print(
            f"{reg}: r_sat {best[1]:.4f} (p = 2), rms {best[0]:.2f} percent, per config {np.round(best[2], 2).tolist()}"
        )
    # ---- fidelity: effective cycles per tile MAC from the anchor captures under the FPU law
    print("\n| fidelity | FPU counter | cycles per tile MAC = FPU / (tile MACs (1 + f0) + bq x q tile steps) |")
    print("|---|---|---|")
    for tag, name in (
        ("r1f_causal_q128k128_lofi_zoff_mp", "LoFi"),
        ("r1f_causal_q128k128_zoff_mp", "HiFi2"),
        ("r1f_causal_q128k128_hifi3_zoff_mp", "HiFi3"),
    ):
        g = rows[tag]["g"]
        cpt = rows[tag]["FPU"] / (
            g["tile_macs"] * (1 + a.fpu_overhead_frac) + a.fpu_overhead_tile_macs_per_qtile_step * g["qtile_steps"]
        )
        print(f"| {name} | {rows[tag]['FPU']:,.0f} | {cpt:.2f} |")
    for tag, name in (
        ("r1f_prod_causal_S4096_q256k256_g110_zoff_mp", "HiFi4 legacy g110"),
        ("r1f_a10_causal_S4096_q256k256_g64_fp32off_zoff_mp", "HiFi4 streaming A10 g64"),
        ("r1c_causal_S4096_q128k128_hd64_zoff_mp", "HiFi2 head_dim 64"),
    ):
        g = rows[tag]["g"]
        cpt = rows[tag]["FPU"] / (
            g["tile_macs"] * (1 + a.fpu_overhead_frac) + a.fpu_overhead_tile_macs_per_qtile_step * g["qtile_steps"]
        )
        print(f"| {name} | {rows[tag]['FPU']:,.0f} | {cpt:.2f} |")
    gp = work(cfg(**T24_PROD["kw"]), M.predict(cfg(**T24_PROD["kw"])))
    cpt = T24_PROD["FPU"] / (
        gp["tile_macs"] * (1 + a.fpu_overhead_frac) + a.fpu_overhead_tile_macs_per_qtile_step * gp["qtile_steps"]
    )
    print(f"| HiFi4 legacy g64 (T2.4) | {T24_PROD['FPU']:,.0f} | {cpt:.2f} |")
    # ---- accurate exp per tile: streaming kernel (anchor capture) and legacy kernel (production g64 and g110)
    base, acc = rows["r1f_causal_q128k128_zoff_mp"], rows["r1f_causal_q128k128_expaccurate_zoff_mp"]
    exp_acc_stream = 70.0 + (acc["SFPU"] - base["SFPU"]) / base["g"]["exp_tiles"]
    print(
        f"\naccurate exp, streaming kernel (anchor): SFPU {acc['SFPU']:,.0f} against {base['SFPU']:,.0f} approx over {base['g']['exp_tiles']:,.1f} exp tiles: {exp_acc_stream:.1f} cycles per tile"
    )
    a10 = rows["r1f_a10_causal_S4096_q256k256_g64_fp32off_zoff_mp"]
    g = a10["g"]
    nonexp = (
        a.sfpu_overhead_per_qtile_step * g["qtile_steps"] + a.sfpu_overhead_per_step * g["steps"] + 17.0 * g["qtiles"]
    )
    print(
        f"accurate exp, streaming kernel (A10 q256 k256 HiFi4): ({a10['SFPU']:,.0f} - non-exp {nonexp:,.0f}) / {g['exp_tiles']:,.0f} = {(a10['SFPU'] - nonexp) / g['exp_tiles']:.1f}"
    )
    for lab, S_, g_ in (
        ("production g64 (T2.4)", T24_PROD["SFPU"], gp),
        (
            "production g110 (R1f)",
            rows["r1f_prod_causal_S4096_q256k256_g110_zoff_mp"]["SFPU"],
            rows["r1f_prod_causal_S4096_q256k256_g110_zoff_mp"]["g"],
        ),
    ):
        nonexp = (
            a.sfpu_overhead_per_qtile_step * g_["qtile_steps"]
            + a.sfpu_overhead_per_step * g_["steps"]
            + 17.0 * g_["qtiles"]
        )
        print(
            f"accurate exp, legacy kernel ({lab}): ({S_:,.0f} - non-exp {nonexp:,.0f}) / {g_['exp_tiles']:,.0f} = {(S_ - nonexp) / g_['exp_tiles']:.1f}"
        )
    # ---- checks: MATH on the fp32 DEST kernel, A10 overlap, head_dim 64 overlap
    p110 = rows["r1f_prod_causal_S4096_q256k256_g110_zoff_mp"]
    print(
        f"\nproduction g110: MATH {p110['MATH']:,.0f} = FPU {p110['FPU']:,.0f} + SFPU {p110['SFPU']:,.0f} ({p110['FPU'] + p110['SFPU']:,.0f}); wall {p110['wall']:,.0f}"
    )
    hd = rows["r1c_causal_S4096_q128k128_hd64_zoff_mp"]
    print(
        f"head_dim 64: FPU {hd['FPU']:,.0f} (law {hd['g']['tile_macs'] * 32 * (1 + a.fpu_overhead_frac) + a.fpu_overhead_tile_macs_per_qtile_step * 32 * hd['g']['qtile_steps']:,.0f}), "
        f"SFPU {hd['SFPU']:,.0f}, MATH {hd['MATH']:,.0f}, hidden / FPU {(hd['FPU'] + hd['SFPU'] - hd['MATH']) / hd['FPU']:.3f}, hidden / SFPU {(hd['FPU'] + hd['SFPU'] - hd['MATH']) / hd['SFPU']:.3f}"
    )
    print(
        f"A10: FPU {a10['FPU']:,.0f}, SFPU {a10['SFPU']:,.0f}, MATH {a10['MATH']:,.0f}, hidden / FPU {(a10['FPU'] + a10['SFPU'] - a10['MATH']) / a10['FPU']:.3f}, hidden / SFPU {(a10['FPU'] + a10['SFPU'] - a10['MATH']) / a10['SFPU']:.3f}"
    )
    # ---- model against the counters with the module's current constants
    print("\n| config | FPU meas | model | err | SFPU meas | model | err | MATH meas | union | err |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag in TAGS:
        x = rows[tag]
        r = x["r"]
        print(
            f"| {tag.replace('_zoff_mp', '')} | {x['FPU']:,.0f} | {r.fpu_cycles:,} | {pct(r.fpu_cycles, x['FPU']):+.2f} | {x['SFPU']:,.0f} | {r.sfpu_cycles:,} | "
            f"{pct(r.sfpu_cycles, x['SFPU']):+.2f} | {x['MATH']:,.0f} | {r.math_active_cycles:,} | {pct(r.math_active_cycles, x['MATH']):+.2f} |"
        )
    return fits


if __name__ == "__main__":
    main()
