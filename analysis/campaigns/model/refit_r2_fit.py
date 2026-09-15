#!/usr/bin/env python
"""Refit and evaluation of the SDPA roofline on this campaign's walls (model/refit_r2_notes.md).

Fits (relative least squares on device walls): the legacy DEST round-trip pair on the T2.4 production walls,
the stream lane factor of windowed / chunked / MLA on the R1a and T2.3 walls, the provided-mask cost per tile
on the R1a masked walls, the sparse per-token and gather terms on three R1a sparse walls, the joint per-step
and per-tile-MAC terms on the two R1a joint walls, the non-paged decode rate on the R1b 110-core rows and the
MLA decode law on the R1b MLA rows. Then every wall of the campaign is priced and tabulated with its set.

Run with the polaris venv python from the polaris checkout on the branch:
    $POLARIS/.venv/bin/python model/refit_r2_fit.py [--fit]
--fit prints the fitted constants (the module is patched in memory); without it the module's constants are
evaluated as they are. Writes data/refit_r2_walls.csv.
"""
import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# POLARIS is the polaris checkout beside handoff/ in the workspace (PORTABLE_CONTRACT.md).
POLARIS = Path(os.environ.get("POLARIS", ROOT.parents[1] / "polaris"))
sys.path.insert(0, str(POLARIS))
import ttsim.perf.roofline_sdpa as M  # noqa: E402

CLK = 1350.0
GQA = dict(num_heads=32, num_kv_heads=8)
MLA = dict(
    head_dim=576, v_head_dim=512, num_kv_heads=1, fidelity="HiFi4", input_dtype="bfloat16", exp_approx_mode=False
)
PROD = dict(
    num_heads=32, num_kv_heads=8, fidelity="HiFi4", exp_approx_mode=False, fp32_dest_acc=True, accum_dtype="float32"
)
SPARSE = dict(
    S=2048,
    head_dim=576,
    v_head_dim=512,
    q_chunk=128,
    k_chunk=128,
    num_heads=32,
    num_kv_heads=1,
    is_sparse=True,
    is_causal=False,
    input_dtype="bfloat16",
    fidelity="HiFi4",
    exp_approx_mode=False,
)
JOINT = dict(
    S=4096 + 333,
    joint_seq=333,
    num_heads=24,
    is_causal=False,
    is_joint=True,
    q_chunk=128,
    k_chunk=512,
    input_dtype="bfloat16",
    fidelity="HiFi2",
    exp_approx_mode=True,
)


def cfg(**kw):
    base = dict(S=4096, num_cores=110)
    base.update(kw)
    arch = base.pop("arch", None)
    return M.SdpaConfig(arch=arch or M.ArchConfig(), **base)


# (label, regime group, block, set, config kwargs, measured cycles). set: fit | pred | holdout | repeat | fresh
WALLS = [
    # grid 1 (T2.1 zones-off walls, S4096 nh32 nkv8 bfp8 HiFi2, 110 cores)
    ("causal q128 k128 (anchor)", "causal", "T2.1", "fit", dict(**GQA), 2562642),
    ("causal q128 k256", "causal", "T2.1", "pred", dict(k_chunk=256, **GQA), 2596634.5),
    ("causal q128 k512", "causal", "T2.1", "pred", dict(k_chunk=512, **GQA), 2729800.5),
    ("causal q64 k128", "causal", "T2.1", "pred", dict(q_chunk=64, **GQA), 4932323),
    ("causal q256 k128", "causal", "T2.1", "fit", dict(q_chunk=256, **GQA), 1826179.5),
    ("causal q512 k128", "causal", "T2.1", "fit", dict(q_chunk=512, **GQA), 2445391),
    ("causal q512 k512", "causal", "T2.1", "fit", dict(q_chunk=512, k_chunk=512, **GQA), 1834463.5),
    ("non-causal q128 k128", "noncausal", "T2.1", "fit", dict(is_causal=False, **GQA), 2884626),
    ("non-causal q128 k256", "noncausal", "T2.1", "fit", dict(is_causal=False, k_chunk=256, **GQA), 3009305),
    ("non-causal q128 k512", "noncausal", "T2.1", "fit", dict(is_causal=False, k_chunk=512, **GQA), 3206662.5),
    ("non-causal q64 k128", "noncausal", "T2.1", "fit", dict(is_causal=False, q_chunk=64, **GQA), 5609143.5),
    ("non-causal q256 k128", "noncausal", "T2.1", "fit", dict(is_causal=False, q_chunk=256, **GQA), 2644797.5),
    ("non-causal q512 k128", "noncausal", "T2.1", "fit", dict(is_causal=False, q_chunk=512, **GQA), 3098890),
    (
        "non-causal q512 k512",
        "noncausal",
        "T2.1",
        "fit",
        dict(is_causal=False, q_chunk=512, k_chunk=512, **GQA),
        2248111,
    ),
    # T2.3 stream law points
    ("causal bf16 K/V", "causal", "T2.3", "fit", dict(kv_input_dtype="bfloat16", **GQA), 4419128),
    (
        "causal all bf16",
        "causal",
        "T2.3",
        "pred",
        dict(input_dtype="bfloat16", kv_input_dtype="bfloat16", **GQA),
        4456116,
    ),
    ("causal nkv32", "causal", "T2.3", "pred", dict(num_heads=32, num_kv_heads=32), 2563786),
    ("causal nkv1", "causal", "T2.3", "pred", dict(num_heads=32, num_kv_heads=1), 2565033),
    ("causal 64 cores", "causal", "T2.3", "fit", dict(num_cores=64, **GQA), 2962746),
    # T2.3r regimes
    ("causal S1024", "causal", "T2.3", "pred", dict(S=1024, **GQA), 258568),
    ("causal S16384", "causal", "T2.3", "pred", dict(S=16384, **GQA), 38629694),
    ("windowed S8192 W1024", "windowed", "T2.3", "fit", dict(S=8192, num_heads=16, sliding_window=1024), 1411340),
    (
        "chunked start4096 Sq2048",
        "chunked",
        "T2.3",
        "fit",
        dict(S=2048, kv_seq=6144, chunk_start_idx=4096, num_heads=16, paged=True, page_block_size=128),
        2026025,
    ),
    ("MLA nh16 S2048", "mla", "T2.3", "fit", dict(S=2048, q_chunk=32, num_heads=16, **MLA), 9574958),
    ("cross 2048/8192", "cross", "T2.3", "pred", dict(S=2048, kv_seq=8192, num_heads=16, is_causal=False), 1649702),
    (
        "masked S8192 d0.25",
        "masked",
        "T2.3",
        "pred",
        dict(S=8192, num_heads=16, is_causal=False, has_attn_mask=True),
        7717703,
    ),
    # R1a
    ("causal S2048", "causal", "R1a", "pred", dict(S=2048, **GQA), 754364),
    ("causal S8192", "causal", "R1a", "pred", dict(S=8192, **GQA), 10094706),
    ("non-causal S2048", "noncausal", "R1a", "pred", dict(S=2048, is_causal=False, **GQA), 733819),
    ("non-causal S8192", "noncausal", "R1a", "pred", dict(S=8192, is_causal=False, **GQA), 10875202),
    ("cross 1024/8192", "cross", "R1a", "pred", dict(S=1024, kv_seq=8192, num_heads=16, is_causal=False), 1107792),
    ("cross 4096/16384", "cross", "R1a", "pred", dict(S=4096, kv_seq=16384, num_heads=16, is_causal=False), 5462958),
    ("windowed S4096 W1024", "windowed", "R1a", "fit", dict(S=4096, num_heads=16, sliding_window=1024), 759909),
    ("windowed S8192 W2048", "windowed", "R1a", "fit", dict(S=8192, num_heads=16, sliding_window=2048), 2486860),
    ("windowed S8192 W4096", "windowed", "R1a", "fit", dict(S=8192, num_heads=16, sliding_window=4096), 4345654),
    (
        "masked S4096 d0.25",
        "masked",
        "R1a",
        "fit",
        dict(S=4096, num_heads=16, is_causal=False, has_attn_mask=True),
        1945668,
    ),
    (
        "masked S8192 d0.5",
        "masked",
        "R1a",
        "fit",
        dict(S=8192, num_heads=16, is_causal=False, has_attn_mask=True),
        7725170,
    ),
    (
        "chunked start2048 Sq2048",
        "chunked",
        "R1a",
        "fit",
        dict(S=2048, kv_seq=4096, chunk_start_idx=2048, num_heads=16, paged=True, page_block_size=128),
        1241686,
    ),
    (
        "chunked start4096 Sq1024",
        "chunked",
        "R1a",
        "fit",
        dict(S=1024, kv_seq=5120, chunk_start_idx=4096, num_heads=16, paged=True, page_block_size=128),
        686071,
    ),
    (
        "chunked start8192 Sq2048",
        "chunked",
        "R1a",
        "fit",
        dict(S=2048, kv_seq=10240, chunk_start_idx=8192, num_heads=16, paged=True, page_block_size=128),
        3593322,
    ),
    ("MLA nh16 S1024", "mla", "R1a", "fit", dict(S=1024, q_chunk=32, num_heads=16, **MLA), 2640221),
    ("MLA nh16 S4096", "mla", "R1a", "fit", dict(S=4096, q_chunk=32, num_heads=16, **MLA), 37137796),
    ("MLA nh32 S2048", "mla", "R1a", "fit", dict(S=2048, q_chunk=32, num_heads=32, **MLA), 19315356),
    ("sparse T8192 TOPK1024", "sparse", "R1a", "fit", dict(kv_seq=1024, **SPARSE), 8603386),
    ("sparse T8192 TOPK2048", "sparse", "R1a", "fit", dict(kv_seq=2048, **SPARSE), 16081570),
    ("sparse T16384 TOPK2048", "sparse", "R1a", "fit", dict(kv_seq=2048, **SPARSE), 16172522),
    ("sparse T32768 TOPK2048", "sparse", "R1a", "holdout", dict(kv_seq=2048, **SPARSE), 16155094),
    ("joint N4096 L333 d128", "joint", "R1a", "fit", dict(head_dim=128, **JOINT), 8578988),
    ("joint N4096 L333 d64", "joint", "R1a", "fit", dict(head_dim=64, **JOINT), 4532419),
    # production (T2.4, legacy path)
    (
        "production S1024 q64 g64",
        "production",
        "T2.4",
        "fit",
        dict(S=1024, q_chunk=64, k_chunk=64, num_cores=64, **PROD),
        468469,
    ),
    (
        "production S2048 q256 g64",
        "production",
        "T2.4",
        "pred",
        dict(S=2048, q_chunk=256, k_chunk=256, num_cores=64, **PROD),
        1072058,
    ),
    (
        "production S4096 q256 g64",
        "production",
        "T2.4",
        "fit",
        dict(S=4096, q_chunk=256, k_chunk=256, num_cores=64, **PROD),
        3983028,
    ),
    (
        "production S8192 q256 g64",
        "production",
        "T2.4",
        "pred",
        dict(S=8192, q_chunk=256, k_chunk=256, num_cores=64, **PROD),
        15392503,
    ),
    (
        "production S4096 q256 g110",
        "production",
        "T2.4",
        "pred",
        dict(S=4096, q_chunk=256, k_chunk=256, num_cores=110, **PROD),
        3019747,
    ),
    # R1d repeats
    ("anchor repeat start", "causal", "R1d", "repeat", dict(**GQA), 2559756),
    ("anchor repeat mid", "causal", "R1d", "repeat", dict(**GQA), 2561014),
    ("anchor repeat end", "causal", "R1d", "repeat", dict(**GQA), 2563848),
    (
        "production S4096 g64 repeat",
        "production",
        "R1d",
        "repeat",
        dict(S=4096, q_chunk=256, k_chunk=256, num_cores=64, **PROD),
        3982135,
    ),
    # R1c hold-outs (never fit)
    ("hold-out causal S2048 q64 k64", "causal", "R1c", "holdout", dict(S=2048, q_chunk=64, k_chunk=64, **GQA), 1286729),
    (
        "hold-out causal all bf16",
        "causal",
        "R1c",
        "holdout",
        dict(input_dtype="bfloat16", kv_input_dtype="bfloat16", **GQA),
        4458372,
    ),
    ("hold-out causal batch 2 nh16", "causal", "R1c", "holdout", dict(batch=2, num_heads=16, num_kv_heads=8), 2562633),
    ("causal head_dim 64", "causal", "R1c", "fit", dict(head_dim=64, **GQA), 1416986),
    ("hold-out causal HiFi4", "causal", "R1c", "holdout", dict(fidelity="HiFi4", **GQA), 2842621),
    ("hold-out causal LoFi", "causal", "R1c", "holdout", dict(fidelity="LoFi", **GQA), 2529248),
    ("hold-out causal MHA nh16", "causal", "R1c", "holdout", dict(num_heads=16, num_kv_heads=16), 1401709),
    ("hold-out causal S512 q64 k64", "causal", "R1c", "holdout", dict(S=512, q_chunk=64, k_chunk=64, **GQA), 132298),
    ("non-causal head_dim 64", "noncausal", "R1c", "fit", dict(head_dim=64, is_causal=False, **GQA), 2309290),
    # R1g head_dim 64 hold-outs (never fit)
    ("hold-out causal S2048 head_dim 64", "causal", "R1g", "holdout", dict(S=2048, head_dim=64, **GQA), 462832),
    (
        "hold-out non-causal S2048 head_dim 64",
        "noncausal",
        "R1g",
        "holdout",
        dict(S=2048, head_dim=64, is_causal=False, **GQA),
        585244,
    ),
    (
        "hold-out causal q256 k128 head_dim 64",
        "causal",
        "R1g",
        "holdout",
        dict(q_chunk=256, head_dim=64, **GQA),
        1499473,
    ),
    (
        "hold-out non-causal q128 k256 head_dim 64",
        "noncausal",
        "R1g",
        "holdout",
        dict(k_chunk=256, head_dim=64, is_causal=False, **GQA),
        1751182,
    ),
    # R1e fresh build (kernel-state check, not fit)
    ("fresh build joint d128", "joint", "R1e", "fresh", dict(head_dim=128, **JOINT), 8708672),
    ("fresh build sparse T8192 TOPK2048", "sparse", "R1e", "fresh", dict(kv_seq=2048, **SPARSE), 16119196),
]

# Decode: (label, block, set, kwargs, measured us)
T28 = dict(
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    k_chunk=0,
    fidelity="HiFi2",
    input_dtype="bfloat16",
    kv_input_dtype="bfp8_b",
    accum_dtype="float32",
    paged=True,
    page_block_size=32,
    max_cores_per_head_batch=16,
    cache_len=16384,
)
NP = dict(
    num_q_heads=32, num_kv_heads=8, head_dim=128, k_chunk=128, fidelity="HiFi2", input_dtype="bfloat16", num_cores=110
)
MLAD_NP = dict(
    num_q_heads=16,
    num_kv_heads=1,
    head_dim=576,
    v_head_dim=512,
    batch=8,
    k_chunk=128,
    fidelity="HiFi4",
    input_dtype="bfloat16",
    kv_input_dtype="bfloat16",
    num_cores=110,
    mla_v_read=True,
)
MLAD_P = dict(
    num_q_heads=128,
    num_kv_heads=1,
    head_dim=576,
    v_head_dim=512,
    k_chunk=128,
    fidelity="HiFi4",
    input_dtype="bfloat16",
    kv_input_dtype="bfp8_b",
    num_cores=110,
    paged=True,
    page_block_size=64,
    max_cores_per_head_batch=16,
    q_in_dram=False,
    q_shard_cores=64,
    cache_len=16384,
)
DECODE = [
    ("paged b32 g64 pos128", "T2.8", "fit", dict(T28, batch=32, num_cores=64, cur_pos=128), 73.8),
    ("paged b32 g64 pos512", "T2.8", "fit", dict(T28, batch=32, num_cores=64, cur_pos=512), 162.8),
    ("paged b32 g64 pos1024", "T2.8", "fit", dict(T28, batch=32, num_cores=64, cur_pos=1024), 284.0),
    ("paged b32 g64 pos2048", "T2.8", "fit", dict(T28, batch=32, num_cores=64, cur_pos=2048), 523.1),
    ("paged b32 g64 pos4096", "T2.8", "fit", dict(T28, batch=32, num_cores=64, cur_pos=4096), 1005.5),
    ("paged b32 g64 pos8192", "T2.8", "fit", dict(T28, batch=32, num_cores=64, cur_pos=8192), 1964.8),
    ("paged b8 g64 pos1024", "T2.8", "fit", dict(T28, batch=8, num_cores=64, cur_pos=1024), 78.8),
    ("paged b16 g64 pos1024", "T2.8", "fit", dict(T28, batch=16, num_cores=64, cur_pos=1024), 148.8),
    (
        "paged b32 g64 bf16 pos1024",
        "T2.8",
        "fit",
        dict(T28, batch=32, num_cores=64, cur_pos=1024, kv_input_dtype="bfloat16"),
        505.1,
    ),
    ("paged b32 g110 pos1024", "T2.8", "fit", dict(T28, batch=32, num_cores=110, cur_pos=1024), 257.2),
    ("paged b32 g110 pos4096", "T2.8", "fit", dict(T28, batch=32, num_cores=110, cur_pos=4096), 901.9),
    ("paged b32 g110 pos8192", "R1b", "pred", dict(T28, batch=32, num_cores=110, cur_pos=8192), 1759.3),
    (
        "non-paged b32 bfp8 cache1024",
        "R1b",
        "fit",
        dict(NP, batch=32, kv_input_dtype="bfp8_b", cache_len=1024, cur_pos=1023),
        229.1,
    ),
    (
        "non-paged b32 bfp8 cache4096",
        "R1b",
        "fit",
        dict(NP, batch=32, kv_input_dtype="bfp8_b", cache_len=4096, cur_pos=4095),
        877.2,
    ),
    (
        "non-paged b8 bf16 cache1024",
        "R1b",
        "fit",
        dict(NP, batch=8, kv_input_dtype="bfloat16", cache_len=1024, cur_pos=1023),
        105.6,
    ),
    (
        "non-paged b8 bf16 cache4096",
        "R1b",
        "fit",
        dict(NP, batch=8, kv_input_dtype="bfloat16", cache_len=4096, cur_pos=4095),
        397.3,
    ),
    ("MLA decode non-paged b8 nh16 cache1024", "R1b", "fit", dict(MLAD_NP, cache_len=1024, cur_pos=1023), 82.1),
    ("MLA decode non-paged b8 nh16 cache4096", "R1b", "fit", dict(MLAD_NP, cache_len=4096, cur_pos=4095), 243.5),
    ("MLA decode non-paged b8 nh16 cache8192", "R1b", "fit", dict(MLAD_NP, cache_len=8192, cur_pos=8191), 447.2),
    ("MLA decode paged b4 nh128 pos1024", "R1b", "fit", dict(MLAD_P, batch=4, cur_pos=1024), 91.8),
    ("MLA decode paged b4 nh128 pos4096", "R1b", "fit", dict(MLAD_P, batch=4, cur_pos=4096), 181.4),
    ("MLA decode paged b4 nh128 pos8192", "R1b", "fit", dict(MLAD_P, batch=4, cur_pos=8192), 295.3),
    ("MLA decode paged b8 nh128 pos1024", "R1b", "fit", dict(MLAD_P, batch=8, cur_pos=1024), 106.8),
    ("MLA decode paged b8 nh128 pos4096", "R1b", "fit", dict(MLAD_P, batch=8, cur_pos=4096), 287.9),
    ("MLA decode paged b8 nh128 pos8192", "R1b", "fit", dict(MLAD_P, batch=8, cur_pos=8192), 531.3),
]


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def wall(kw, arch=None):
    return M.predict(cfg(**kw, arch=arch)).wall_clock_cycles


def dwall_us(kw, arch=None):
    return M.predict_decode(arch=arch or M.ArchConfig(), **kw).wall_clock_cycles / CLK


def rel_lsq(rows, features, targets):
    """Relative least squares: minimise sum ((X beta - y) / y)^2."""
    X = np.array(features, dtype=float)
    y = np.array(targets, dtype=float)
    w = 1.0 / y
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return beta


def fit_scale(points, regime, table, base=None):
    """Stream lane factor of one regime by 1-D search on the relative residuals."""

    def err(s):
        t = replace(table[regime], stream_scale=s)
        a = replace(base or M.ArchConfig(), wall_terms={**table, regime: t})
        return sum(pct(wall(kw, a), meas) ** 2 for _, kw, meas in points)

    lo, hi = 0.5, 1.5
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if err(m1) < err(m2):
            hi = m2
        else:
            lo = m1
    return round((lo + hi) / 2, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    table = dict(M.WALL_TERMS_BH)
    base = M.ArchConfig()
    if args.fit:
        # 1. legacy pair: on the compute lane, (wall - init) / steps - floor - control - mask = L + D x windows
        rows = []
        for lab, g, b, st, kw, meas in WALLS:
            if b == "T2.4" and st == "fit":
                a = replace(base, legacy_step_cycles=0.0, dest_roundtrip_cycles=0.0)
                r = M.predict(cfg(**kw, arch=a))
                steps = r.q_chunks_per_core * r.k_eff
                qct, kct = kw["q_chunk"] // 32, kw["k_chunk"] // 32
                target = (
                    (meas - a.wall_fixed_cycles) / r.steps_wall_core
                    - r.components["compute_floor"] / steps
                    - (247 + 60 * qct)
                    - 57
                )
                rows.append((target, qct * kct / 4))
        A = np.array([[1.0, w] for _, w in rows])
        y = np.array([t for t, _ in rows])
        L, D = np.linalg.solve(A, y)
        print(f"legacy_step_cycles {L:.1f}  dest_roundtrip_cycles {D:.1f}")
        base = replace(base, legacy_step_cycles=round(L, 1), dest_roundtrip_cycles=round(D, 1))
        # 1b. head_dim term: the two R1c head_dim 64 walls. The causal one is stream bound, so its step fixes the
        # split of the 691 per k tile into a per-k-tile and a per-K/V-tile part; the non-causal one is PACK bound
        # and splits the per-q-k-tile PACK cost into a score part and a per-K/V-tile part (shared by both lanes).
        c_meas = [m for lab, g, b, st, kw, m in WALLS if lab == "causal head_dim 64"][0]
        n_meas = [m for lab, g, b, st, kw, m in WALLS if lab == "non-causal head_dim 64"][0]
        step_c = (c_meas - base.wall_fixed_cycles) / 165.0
        fixed_hd64 = (step_c - 4 * 4 * 1088 / base.kv_stream_rate_bpc[110]) / 4  # per k tile at dct_sum 4
        f1 = (691.0 - fixed_hd64) / 4.0
        f0 = 691.0 - 8.0 * f1
        step_n = (n_meas - base.wall_fixed_cycles) / 320.0
        pk_hd64 = (step_n - 1136.0 - 4 * 663.0) / 16.0  # per q-k tile at dct_sum 4
        pkd = (319.5 - pk_hd64) / 4.0
        print(f"kv_stream_fixed_per_ktile {f0:.1f}  kv_stream_fixed_per_kv_tile {f1:.2f}  pack_per_qk_dtile {pkd:.2f}")
        base = replace(base, kv_stream_fixed_per_ktile=round(f0, 1), kv_stream_fixed_per_kv_tile=round(f1, 2))
        for reg, pk in (("prefill_causal", 348.0), ("prefill_noncausal", 319.5), ("cross", 319.5), ("masked", 319.5)):
            table[reg] = replace(table[reg], pack_per_qktile=pk - 8 * round(pkd, 2), pack_per_qk_dtile=round(pkd, 2))
        # 2. stream scales
        for regime in ("windowed", "chunked", "mla"):
            pts = [
                (lab, kw, meas)
                for lab, g, b, st, kw, meas in WALLS
                if g == regime and st == "fit" and not (regime == "mla" and "S1024" in lab)
            ]
            s = fit_scale(pts, regime, table, base)
            table[regime] = replace(table[regime], stream_scale=s)
            print(f"{regime} stream_scale {s}")
        # 3. masked per tile: linear in mask_per_tile
        pts = [(lab, kw, meas) for lab, g, b, st, kw, meas in WALLS if g == "masked" and st == "fit"]
        feats, targ = [], []
        for lab, kw, meas in pts:
            t0 = replace(table["masked"], mask_per_tile=0.0)
            a0 = replace(base, wall_terms={**table, "masked": t0})
            r0 = M.predict(cfg(**kw, arch=a0))
            tiles_steps = r0.steps_wall_core * (kw.get("q_chunk", 128) // 32) * (kw.get("k_chunk", 128) // 32)
            feats.append([tiles_steps])
            targ.append(meas - r0.wall_clock_cycles)
        mpt = rel_lsq(pts, feats, targ)[0]
        table["masked"] = replace(table["masked"], mask_per_tile=round(mpt, 1))
        print(f"masked mask_per_tile {mpt:.1f}")
        # 4. sparse: per token = fe + TOPK x key_row_bytes / rate, linear in (fe, 1/rate) when gather bound
        pts = [(lab, kw, meas) for lab, g, b, st, kw, meas in WALLS if g == "sparse" and st == "fit"]
        feats, targ = [], []
        for lab, kw, meas in pts:
            t0 = replace(table["sparse"], fe_per_token=0.0, gather_rate_bpc=1e9)
            a0 = replace(base, wall_terms={**table, "sparse": t0})
            r0 = M.predict(cfg(**kw, arch=a0))
            tok = r0.q_chunks_wall_core
            key_bytes = 18 * 2048 / 32
            feats.append([tok, tok * kw["kv_seq"] * key_bytes])
            targ.append(meas - 2900.0)
        fe, inv_rate = rel_lsq(pts, feats, targ)
        table["sparse"] = replace(table["sparse"], fe_per_token=round(fe, 0), gather_rate_bpc=round(1.0 / inv_rate, 3))
        print(f"sparse fe_per_token {fe:.0f}  gather_rate_bpc {1.0 / inv_rate:.3f}")
        # 5. joint: per step = floor + control + c x tile MACs (the two walls leave no flat part), linear in c
        pts = [(lab, kw, meas) for lab, g, b, st, kw, meas in WALLS if g == "joint" and st == "fit"]
        feats, targ = [], []
        for lab, kw, meas in pts:
            t0 = replace(table["joint"], fe_flat_per_kchunk=0.0, fe_per_tile_mac=0.0)
            a0 = replace(base, wall_terms={**table, "joint": t0})
            r0 = M.predict(cfg(**kw, arch=a0))
            tm = (kw["q_chunk"] // 32) * (kw["k_chunk"] // 32) * 2 * (kw["head_dim"] // 32)
            feats.append([r0.steps_wall_core * tm])
            targ.append(meas - r0.wall_clock_cycles)
        fc = rel_lsq(pts, feats, targ)[0]
        table["joint"] = replace(table["joint"], fe_flat_per_kchunk=0.0, fe_per_tile_mac=round(fc, 1))
        print(f"joint fe_per_tile_mac {fc:.1f}")
        base = replace(base, wall_terms=table)
        # 6. non-paged decode rate at 110: wall = fixed + bytes x 1.35 / R, linear in 1/R with the paged fixed law
        pts = [(lab, kw, meas) for lab, b, st, kw, meas in DECODE if lab.startswith("non-paged") and st == "fit"]
        feats, targ = [], []
        for lab, kw, meas in pts:
            r0 = M.predict_decode(arch=base, **kw)
            feats.append([r0.dram_in_bytes * 1.35])
            targ.append(meas * CLK - r0.components["init"])
        inv_r = rel_lsq(pts, feats, targ)[0]
        base = replace(base, decode_kv_stream_gbps_nonpaged=round(1.0 / inv_r, 1))
        print(f"decode_kv_stream_gbps_nonpaged {1.0 / inv_r:.1f}")
        # 7. MLA decode: wall = F0 + h x slices + bytes x 1.35 / R, linear in (F0, h, 1/R)
        pts = [(lab, kw, meas) for lab, b, st, kw, meas in DECODE if lab.startswith("MLA decode") and st == "fit"]
        feats, targ = [], []
        for lab, kw, meas in pts:
            r0 = M.predict_decode(arch=base, **kw)
            feats.append([1.0, r0.config_echo["q_head_slices"], r0.dram_in_bytes * 1.35])
            targ.append(meas * CLK)
        F0, h, inv_r = rel_lsq(pts, feats, targ)
        base = replace(
            base,
            decode_fixed_overhead_cycles_mla=round(F0, 0),
            decode_fixed_per_qhead_slice_cycles_mla=round(h, 0),
            decode_kv_stream_gbps_mla=round(1.0 / inv_r, 1),
        )
        print(
            f"decode_fixed_overhead_cycles_mla {F0:.0f}  decode_fixed_per_qhead_slice_cycles_mla {h:.0f}  decode_kv_stream_gbps_mla {1.0 / inv_r:.1f}"
        )
    # evaluation table
    out = []
    print(f"\n{'label':40s} {'block':5s} {'set':7s} {'measured':>12s} {'model':>12s} {'err%':>7s}")
    for lab, g, b, st, kw, meas in WALLS:
        r = M.predict(cfg(**kw, arch=base))
        e = pct(r.wall_clock_cycles, meas)
        out.append(
            dict(
                kind="prefill",
                label=lab,
                regime=g,
                block=b,
                set=st,
                measured_cycles=meas,
                model_cycles=r.wall_clock_cycles,
                err_pct=round(e, 2),
                steps_wall_core=r.steps_wall_core,
                flags=";".join(r.low_confidence_reasons),
            )
        )
        print(f"{lab:40s} {b:5s} {st:7s} {meas:12,.0f} {r.wall_clock_cycles:12,d} {e:+7.2f}")
    print()
    for lab, b, st, kw, meas in DECODE:
        r = M.predict_decode(arch=base, **kw)
        us = r.wall_clock_cycles / CLK
        e = pct(us, meas)
        out.append(
            dict(
                kind="decode",
                label=lab,
                regime="decode",
                block=b,
                set=st,
                measured_cycles=round(meas * CLK),
                model_cycles=r.wall_clock_cycles,
                err_pct=round(e, 2),
                steps_wall_core="",
                flags=";".join(r.low_confidence_reasons),
            )
        )
        print(
            f"{lab:40s} {b:5s} {st:7s} {meas:12.1f} {us:12.1f} {e:+7.2f}  slices {r.config_echo.get('q_head_slices')} bytes {r.dram_in_bytes / 1e6:.1f} MB"
        )
    # summary by set
    print()
    for st in ("fit", "pred", "holdout", "repeat", "fresh"):
        e = np.array([o["err_pct"] for o in out if o["set"] == st])
        if st == "holdout":
            for blk in ("R1c", "R1g", "R1a"):
                eb = np.array([o["err_pct"] for o in out if o["set"] == st and o["block"] == blk])
                if len(eb):
                    print(
                        f"holdout {blk}: n {len(eb)} within5 {int((abs(eb) <= 5).sum())} within10 {int((abs(eb) <= 10).sum())} mean {eb.mean():+.2f} mabs {abs(eb).mean():.2f}"
                    )
        if len(e):
            worst = max((o for o in out if o["set"] == st), key=lambda o: abs(o["err_pct"]))
            print(
                f"{st:8s} n {len(e):3d} within5 {int((abs(e) <= 5).sum()):3d} within10 {int((abs(e) <= 10).sum()):3d} "
                f"mean {e.mean():+.2f} mabs {abs(e).mean():.2f} worst {worst['label']} {worst['err_pct']:+.2f}"
            )
    path = ROOT / "data" / "refit_r2_walls.csv"
    with open(path, "w", newline="") as f:
        f.write(
            "# PROVENANCE: every wall of the 2026-09-12 campaign (p100a, fw 19.9.0, tt-metal 72620d5 kernels; R1e rows on the fresh build "
            "f3fbc8984f0) priced by ttsim/perf/roofline_sdpa.py of branch mvlahovic/sdpa_revamp; measured = zones-off device wall "
            "(mean of invocations 1 and 2) or DEVICE KERNEL DURATION median for decode; model/refit_r2_fit.py\n"
        )
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("wrote", path)


if __name__ == "__main__":
    main()
