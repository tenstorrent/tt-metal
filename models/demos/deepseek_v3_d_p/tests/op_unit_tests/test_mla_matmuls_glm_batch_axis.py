# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hand-tuning the GLM-5.3 MLA matmuls for the batch-axis (one user per mesh column, TP=1) layout.

In batch-axis mode (``ttMLA(batch_axis=1)``) every chip runs the WHOLE attention matmul of its own user:
hidden, all 64 heads and every weight are local, so the per-chip shapes are the TP=4 shapes with the
TP-split dimension 4x wider. The TP-tuned table in mla_config.py overflows them, which is why batch-axis
mode fell back to TTNN defaults. These matmuls are chip-local (no CCL), so one chip measures them: the
test opens a 1x1 mesh.

  test_glm_batch_axis_mm_sweep -- every candidate (plus a "default" row reproducing today's batch-axis
                                  fallback) in one session, each preceded by a tracy signpost naming it
  test_glm_batch_axis_mm       -- the chosen config per matmul (what mla_config.py wires in)

Per-chip shapes (M = seq_len_local = 640 rows; tile counts M_t x K_t x N_t):
  q_a_proj             640 x 6144  x 2048            20 x 192 x  64
  q_b_proj             640 x 2048  x 16384           20 x  64 x 512
  kv_a_proj_with_mqa   640 x 6144  x 576             20 x 192 x  18
  wkv_b1      Z=64     640 x 192   x 512             20 x   6 x  16
  wkv_b2      Z=64     640 x 512   x 256             20 x  16 x   8
  o_proj               640 x 16384 x 6144            20 x 512 x 192
  indexer.wq_b         640 x 2048  x 4096            20 x  64 x 128
  indexer.wk           640 x 6144  x 128             20 x 192 x   4
  indexer.weights_proj 640 x 6144  x 32              20 x 192 x   1
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

PCC_REQUIRED = 0.99
GRID = (11, 10)  # 12x10 available; 11x10 for di/dt and throttling headroom (as mla_config.COMPUTE_GRID)
ITERS = 3  # first iteration compiles; the parser keeps the fastest

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
BF16 = ttnn.bfloat16
BF8 = ttnn.bfloat8_b

HIDDEN, HEADS, Q_LORA, KV_LORA = 6144, 64, 2048, 512
QK_NOPE, QK_ROPE, V_HEAD = 192, 64, 256
IDX_HEADS, IDX_HEAD_DIM = 32, 128
M = 640

# name -> (Z, K, N, in0 dtype, in1 dtype, out dtype). in0 dtypes are what batch-axis attention feeds each
# matmul: o_proj consumes the BF8 wkv_b2 output; indexer.weights_proj keeps its BF16 weight (#51005).
SHAPES = {
    "q_a_proj": (1, HIDDEN, Q_LORA, BF16, BF8, BF16),
    "q_b_proj": (1, Q_LORA, HEADS * (QK_NOPE + QK_ROPE), BF16, BF8, BF16),
    "kv_a_proj_with_mqa": (1, HIDDEN, KV_LORA + QK_ROPE, BF16, BF8, BF16),
    "wkv_b1": (HEADS, QK_NOPE, KV_LORA, BF16, BF8, BF16),
    "wkv_b2": (HEADS, KV_LORA, V_HEAD, BF16, BF8, BF8),
    "o_proj": (1, HEADS * V_HEAD, HIDDEN, BF8, BF8, BF16),
    "indexer.wq_b": (1, Q_LORA, IDX_HEADS * IDX_HEAD_DIM, BF16, BF8, BF16),
    "indexer.wk": (1, HIDDEN, IDX_HEAD_DIM, BF16, BF8, BF16),
    "indexer.weights_proj": (1, HIDDEN, IDX_HEADS, BF16, BF16, BF16),
    # Hadamard transforms of the indexer Q (all 32 heads) and K; the [128, 128] weight broadcasts over heads.
    "indexer.q_hadamard": (IDX_HEADS, IDX_HEAD_DIM, IDX_HEAD_DIM, BF16, BF16, BF8),
    "indexer.k_hadamard": (1, IDX_HEAD_DIM, IDX_HEAD_DIM, BF16, BF16, BF16),
}
BROADCAST_WEIGHT = {"indexer.q_hadamard", "indexer.k_hadamard"}


def _mc2d(ib, sh, sw, pcm, pcn):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=ib,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


def _reuse(ib, sh, sw, pcm, pcn):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=ib,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pcm,
        per_core_N=pcn,
    )


def _mc1d_batched_fallback(k_t, n_t):
    """Exactly what ttMLA._make_batched_mm_kwargs builds when no tuned config applies (batch-axis today)."""
    return _mc1d_batched_fallback_m(k_t, n_t, M)


def _mc1d_batched_fallback_m(k_t, n_t, m):
    m_t = m // ttnn.TILE_SIZE
    pcm = max(1, -(-m_t // (GRID[0] * GRID[1])))
    while m_t % pcm:
        pcm += 1
    sw = min(n_t, 8)
    while n_t % sw:
        sw -= 1
    sh = min(pcm, 8 // sw)
    while pcm % sh:
        sh -= 1
    ib = min(4, k_t)
    while k_t % ib:
        ib -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=ib,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pcm,
        per_core_N=n_t,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
    )


# (variant_id, matmul, program_config or None (= TTNN default), act mem, out mem)
SWEEP = [
    # --- today's batch-axis fallback: no program_config, DRAM in/out (1D mcast for the batched pair) ---
    ("q_a_proj__default", "q_a_proj", None, DRAM, DRAM),
    ("q_b_proj__default", "q_b_proj", None, DRAM, DRAM),
    ("kv_a_proj__default", "kv_a_proj_with_mqa", None, DRAM, DRAM),
    ("wkv_b1__default_1d", "wkv_b1", _mc1d_batched_fallback(6, 16), DRAM, DRAM),
    ("wkv_b2__default_1d", "wkv_b2", _mc1d_batched_fallback(16, 8), DRAM, DRAM),
    ("o_proj__default", "o_proj", None, DRAM, DRAM),
    ("wq_b__default", "indexer.wq_b", None, DRAM, DRAM),
    ("wk__default", "indexer.wk", None, DRAM, DRAM),
    ("weights_proj__default", "indexer.weights_proj", None, DRAM, DRAM),
    # --- q_a_proj: pc 2x6 -> 10 x ceil(64/6)=11 = 110 cores ---
    ("q_a_proj__ib8_1x6_dd", "q_a_proj", _mc2d(8, 1, 6, 2, 6), DRAM, DRAM),
    ("q_a_proj__ib8_1x6_dl", "q_a_proj", _mc2d(8, 1, 6, 2, 6), DRAM, L1),
    ("q_a_proj__ib16_1x6_dl", "q_a_proj", _mc2d(16, 1, 6, 2, 6), DRAM, L1),
    ("q_a_proj__ib24_1x6_dl", "q_a_proj", _mc2d(24, 1, 6, 2, 6), DRAM, L1),
    ("q_a_proj__ib32_1x6_dl", "q_a_proj", _mc2d(32, 1, 6, 2, 6), DRAM, L1),
    ("q_a_proj__ib16_2x3_dl", "q_a_proj", _mc2d(16, 2, 3, 2, 6), DRAM, L1),
    ("q_a_proj__ib16_1x6_ll", "q_a_proj", _mc2d(16, 1, 6, 2, 6), L1, L1),
    # --- q_b_proj: pc 2x48 -> 10 x ceil(512/48)=11 = 110 cores ---
    ("q_b_proj__ib4_1x8_dd", "q_b_proj", _mc2d(4, 1, 8, 2, 48), DRAM, DRAM),
    ("q_b_proj__ib8_1x8_dd", "q_b_proj", _mc2d(8, 1, 8, 2, 48), DRAM, DRAM),
    # out in L1 (21 MB) clashes with the CBs; act L1 / out DRAM is the L1 option left.
    ("q_b_proj__ib16_1x8_ld", "q_b_proj", _mc2d(16, 1, 8, 2, 48), L1, DRAM),
    ("q_b_proj__ib8_2x4_ld", "q_b_proj", _mc2d(8, 2, 4, 2, 48), L1, DRAM),
    ("q_b_proj__ib16_1x8_dd", "q_b_proj", _mc2d(16, 1, 8, 2, 48), DRAM, DRAM),
    ("q_b_proj__ib8_1x8_ld", "q_b_proj", _mc2d(8, 1, 8, 2, 48), L1, DRAM),
    # --- kv_a_proj_with_mqa: N_t=18 -> pc 2x2 -> 10 x 9 = 90 cores (max) ---
    ("kv_a_proj__ib8_1x2_dl", "kv_a_proj_with_mqa", _mc2d(8, 1, 2, 2, 2), DRAM, L1),
    ("kv_a_proj__ib16_1x2_dl", "kv_a_proj_with_mqa", _mc2d(16, 1, 2, 2, 2), DRAM, L1),
    ("kv_a_proj__ib24_2x2_dl", "kv_a_proj_with_mqa", _mc2d(24, 2, 2, 2, 2), DRAM, L1),
    ("kv_a_proj__ib32_2x2_dl", "kv_a_proj_with_mqa", _mc2d(32, 2, 2, 2, 2), DRAM, L1),
    ("kv_a_proj__ib16_1x2_ll", "kv_a_proj_with_mqa", _mc2d(16, 1, 2, 2, 2), L1, L1),
    # --- wkv_b1 (batched, Z*M_t = 1280): Reuse folds the batch onto the grid ---
    ("wkv_b1__r_pcm20_ib6_2x4_dd", "wkv_b1", _reuse(6, 2, 4, 20, 16), DRAM, DRAM),  # 64 cores
    # Reuse needs per_core_M | M_t (20) and per_core_N == N, so 1280 / per_core_M <= 110 cores leaves
    # only per_core_M = 20 (64 cores); pcm10 / pcm5 (128 / 256 blocks) probe the grid limit.
    ("wkv_b1__r_pcm10_ib6_2x4_dd", "wkv_b1", _reuse(6, 2, 4, 10, 16), DRAM, DRAM),
    ("wkv_b1__r_pcm5_ib6_1x8_dd", "wkv_b1", _reuse(6, 1, 8, 5, 16), DRAM, DRAM),
    ("wkv_b1__r_pcm20_ib6_2x4_dl", "wkv_b1", _reuse(6, 2, 4, 20, 16), DRAM, L1),
    ("wkv_b1__r_pcm20_ib6_2x4_ll", "wkv_b1", _reuse(6, 2, 4, 20, 16), L1, L1),
    ("wkv_b1__r_pcm20_ib3_1x8_ll", "wkv_b1", _reuse(3, 1, 8, 20, 16), L1, L1),
    ("wkv_b1__r_pcm20_ib2_4x2_ll", "wkv_b1", _reuse(2, 4, 2, 20, 16), L1, L1),
    # --- wkv_b2 (batched, Z*M_t = 1280) ---
    ("wkv_b2__r_pcm20_ib4_1x8_dd", "wkv_b2", _reuse(4, 1, 8, 20, 8), DRAM, DRAM),
    ("wkv_b2__r_pcm10_ib4_1x8_dd", "wkv_b2", _reuse(4, 1, 8, 10, 8), DRAM, DRAM),
    ("wkv_b2__r_pcm20_ib4_1x8_ll", "wkv_b2", _reuse(4, 1, 8, 20, 8), L1, L1),
    ("wkv_b2__r_pcm20_ib8_2x4_ll", "wkv_b2", _reuse(8, 2, 4, 20, 8), L1, L1),
    ("wkv_b2__r_pcm20_ib16_2x4_ll", "wkv_b2", _reuse(16, 2, 4, 20, 8), L1, L1),
    ("wkv_b2__r_pcm20_ib2_1x8_ll", "wkv_b2", _reuse(2, 1, 8, 20, 8), L1, L1),
    ("wkv_b2__r_pcm20_ib4_1x8_dl", "wkv_b2", _reuse(4, 1, 8, 20, 8), DRAM, L1),
    # --- o_proj: pc 2x18 -> 10 x ceil(192/18)=11 = 110 cores ---
    ("o_proj__ib8_1x6_dd", "o_proj", _mc2d(8, 1, 6, 2, 18), DRAM, DRAM),
    ("o_proj__ib16_1x6_dd", "o_proj", _mc2d(16, 1, 6, 2, 18), DRAM, DRAM),
    ("o_proj__ib16_1x6_dl", "o_proj", _mc2d(16, 1, 6, 2, 18), DRAM, L1),
    ("o_proj__ib32_1x6_dd", "o_proj", _mc2d(32, 1, 6, 2, 18), DRAM, DRAM),
    ("o_proj__ib16_2x3_dl", "o_proj", _mc2d(16, 2, 3, 2, 18), DRAM, L1),
    ("o_proj__ib16_1x6_ll", "o_proj", _mc2d(16, 1, 6, 2, 18), L1, L1),
    # --- indexer.wq_b: pc 2x12 -> 10 x ceil(128/12)=11 = 110 cores ---
    ("wq_b__ib8_1x6_dd", "indexer.wq_b", _mc2d(8, 1, 6, 2, 12), DRAM, DRAM),
    ("wq_b__ib8_1x6_ll", "indexer.wq_b", _mc2d(8, 1, 6, 2, 12), L1, L1),
    ("wq_b__ib16_1x6_ll", "indexer.wq_b", _mc2d(16, 1, 6, 2, 12), L1, L1),
    ("wq_b__ib8_2x3_ll", "indexer.wq_b", _mc2d(8, 2, 3, 2, 12), L1, L1),
    ("wq_b__ib8_1x6_dl", "indexer.wq_b", _mc2d(8, 1, 6, 2, 12), DRAM, L1),
    # --- indexer.wk: N_t=4 floors at 10 x 4 = 40 cores ---
    ("wk__ib8_dd", "indexer.wk", _mc2d(8, 1, 1, 2, 1), DRAM, DRAM),
    ("wk__ib24_dd", "indexer.wk", _mc2d(24, 1, 1, 2, 1), DRAM, DRAM),
    ("wk__ib48_dd", "indexer.wk", _mc2d(48, 1, 1, 2, 1), DRAM, DRAM),
    ("wk__ib24_ll", "indexer.wk", _mc2d(24, 1, 1, 2, 1), L1, L1),
    ("wk__ib24_dl", "indexer.wk", _mc2d(24, 1, 1, 2, 1), DRAM, L1),
    # --- indexer.weights_proj: N_t=1 floors at 10 cores ---
    ("weights_proj__ib8_dd", "indexer.weights_proj", _mc2d(8, 1, 1, 2, 1), DRAM, DRAM),
    ("weights_proj__ib24_dd", "indexer.weights_proj", _mc2d(24, 1, 1, 2, 1), DRAM, DRAM),
    ("weights_proj__ib48_dd", "indexer.weights_proj", _mc2d(48, 1, 1, 2, 1), DRAM, DRAM),
    ("weights_proj__ib24_ll", "indexer.weights_proj", _mc2d(24, 1, 1, 2, 1), L1, L1),
    ("weights_proj__ib24_dl", "indexer.weights_proj", _mc2d(24, 1, 1, 2, 1), DRAM, L1),
]


def _mc1d(ib, sh, sw, pcm, pcn, fuse_batch=True):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=ib,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pcm,
        per_core_N=pcn,
        fuse_batch=fuse_batch,
        mcast_in0=False,
    )


# Hadamard at 640 rows: the TP table's 640 / 160 entries as candidates, plus a few more.
SWEEP += [
    ("q_hadamard__default", "indexer.q_hadamard", None, DRAM, DRAM),
    ("q_hadamard__1d_pcm2_2x4", "indexer.q_hadamard", _mc1d(2, 2, 4, 2, 4), DRAM, DRAM),  # 320 blocks: overflow
    ("q_hadamard__1d_pcm8_2x4", "indexer.q_hadamard", _mc1d(4, 2, 4, 8, 4), DRAM, DRAM),  # 80 blocks
    ("q_hadamard__1d_pcm6_2x4", "indexer.q_hadamard", _mc1d(4, 2, 4, 6, 4), DRAM, DRAM),  # 107 blocks
    ("q_hadamard__1d_pcm8_4x2", "indexer.q_hadamard", _mc1d(4, 4, 2, 8, 4), DRAM, DRAM),
    ("k_hadamard__default", "indexer.k_hadamard", None, DRAM, DRAM),
    ("k_hadamard__tp640", "indexer.k_hadamard", _mc2d(4, 2, 1, 2, 1), DRAM, DRAM),  # the TP table's 640 entry
    ("k_hadamard__1d_pcm1", "indexer.k_hadamard", _mc1d(4, 1, 4, 1, 4, fuse_batch=False), DRAM, DRAM),
]

# 256 rows per chip (2k chunks): M_t = 8, so per_core_M = 1 puts 8 block rows on the 10-row grid.
M256 = 256
SWEEP_256 = [
    ("q_a_proj__default", "q_a_proj", None, DRAM, DRAM, M256),
    ("q_a_proj__ib16_1x6_dl", "q_a_proj", _mc2d(16, 1, 6, 1, 6), DRAM, L1, M256),  # 8 x 11 = 88 cores
    ("q_a_proj__ib24_1x6_dl", "q_a_proj", _mc2d(24, 1, 6, 1, 6), DRAM, L1, M256),
    ("q_a_proj__ib32_1x6_dl", "q_a_proj", _mc2d(32, 1, 6, 1, 6), DRAM, L1, M256),
    ("q_a_proj__ib16_1x8_dl", "q_a_proj", _mc2d(16, 1, 8, 1, 8), DRAM, L1, M256),  # 8 x 8 = 64 cores
    ("q_b_proj__default", "q_b_proj", None, DRAM, DRAM, M256),
    ("q_b_proj__ib8_1x8_ll", "q_b_proj", _mc2d(8, 1, 8, 1, 48), L1, L1, M256),  # 88 cores; out 8.4 MB
    ("q_b_proj__ib8_1x8_ld", "q_b_proj", _mc2d(8, 1, 8, 1, 48), L1, DRAM, M256),
    ("q_b_proj__ib16_1x8_ll", "q_b_proj", _mc2d(16, 1, 8, 1, 48), L1, L1, M256),
    ("q_b_proj__ib4_1x8_ll", "q_b_proj", _mc2d(4, 1, 8, 1, 48), L1, L1, M256),
    ("kv_a_proj__default", "kv_a_proj_with_mqa", None, DRAM, DRAM, M256),
    ("kv_a_proj__ib16_1x2_dl", "kv_a_proj_with_mqa", _mc2d(16, 1, 2, 1, 2), DRAM, L1, M256),  # 72 cores
    ("kv_a_proj__ib24_1x2_dl", "kv_a_proj_with_mqa", _mc2d(24, 1, 2, 1, 2), DRAM, L1, M256),
    ("kv_a_proj__ib32_1x2_dl", "kv_a_proj_with_mqa", _mc2d(32, 1, 2, 1, 2), DRAM, L1, M256),
    # Batched (Z*M_t = 512): Reuse needs per_core_M | 8 and blocks <= 110 -> per_core_M = 8 (64 cores).
    ("wkv_b1__default_1d", "wkv_b1", _mc1d_batched_fallback_m(6, 16, M256), DRAM, DRAM, M256),
    ("wkv_b1__r_pcm8_ib6_2x4_dd", "wkv_b1", _reuse(6, 2, 4, 8, 16), DRAM, DRAM, M256),
    ("wkv_b1__r_pcm8_ib3_1x8_dd", "wkv_b1", _reuse(3, 1, 8, 8, 16), DRAM, DRAM, M256),
    ("wkv_b1__r_pcm8_ib6_2x4_dl", "wkv_b1", _reuse(6, 2, 4, 8, 16), DRAM, L1, M256),  # out 16.8 MB
    ("wkv_b1__r_pcm8_ib6_2x4_ld", "wkv_b1", _reuse(6, 2, 4, 8, 16), L1, DRAM, M256),
    ("wkv_b2__default_1d", "wkv_b2", _mc1d_batched_fallback_m(16, 8, M256), DRAM, DRAM, M256),
    ("wkv_b2__r_pcm8_ib4_1x8_dd", "wkv_b2", _reuse(4, 1, 8, 8, 8), DRAM, DRAM, M256),
    ("wkv_b2__r_pcm8_ib8_2x4_dd", "wkv_b2", _reuse(8, 2, 4, 8, 8), DRAM, DRAM, M256),
    ("wkv_b2__r_pcm8_ib4_1x8_dl", "wkv_b2", _reuse(4, 1, 8, 8, 8), DRAM, L1, M256),
    ("wkv_b2__r_pcm8_ib4_1x8_ll", "wkv_b2", _reuse(4, 1, 8, 8, 8), L1, L1, M256),  # in 16.8 MB, out 4.5 MB
    ("o_proj__default", "o_proj", None, DRAM, DRAM, M256),
    ("o_proj__ib16_1x6_dl", "o_proj", _mc2d(16, 1, 6, 1, 18), DRAM, L1, M256),  # 88 cores
    ("o_proj__ib8_1x6_dl", "o_proj", _mc2d(8, 1, 6, 1, 18), DRAM, L1, M256),
    ("o_proj__ib32_1x6_dl", "o_proj", _mc2d(32, 1, 6, 1, 18), DRAM, L1, M256),
    ("o_proj__ib16_1x8_dl", "o_proj", _mc2d(16, 1, 8, 1, 24), DRAM, L1, M256),  # 64 cores
    ("o_proj__ib16_1x6_ll", "o_proj", _mc2d(16, 1, 6, 1, 18), L1, L1, M256),
    ("wq_b__default", "indexer.wq_b", None, DRAM, DRAM, M256),
    ("wq_b__ib8_1x6_ll", "indexer.wq_b", _mc2d(8, 1, 6, 1, 12), L1, L1, M256),  # 88 cores
    ("wq_b__ib16_1x6_ll", "indexer.wq_b", _mc2d(16, 1, 6, 1, 12), L1, L1, M256),
    ("wq_b__ib8_1x8_ll", "indexer.wq_b", _mc2d(8, 1, 8, 1, 16), L1, L1, M256),  # 64 cores
    ("wk__default", "indexer.wk", None, DRAM, DRAM, M256),
    ("wk__ib24_dd", "indexer.wk", _mc2d(24, 1, 1, 1, 1), DRAM, DRAM, M256),  # 32 cores (N_t=4)
    ("wk__ib48_dd", "indexer.wk", _mc2d(48, 1, 1, 1, 1), DRAM, DRAM, M256),
    ("weights_proj__default", "indexer.weights_proj", None, DRAM, DRAM, M256),
    ("weights_proj__ib8_dd", "indexer.weights_proj", _mc2d(8, 1, 1, 1, 1), DRAM, DRAM, M256),  # 8 cores
    ("weights_proj__ib24_dd", "indexer.weights_proj", _mc2d(24, 1, 1, 1, 1), DRAM, DRAM, M256),
    ("q_hadamard__default", "indexer.q_hadamard", None, DRAM, DRAM, M256),
    ("q_hadamard__1d_pcm4_2x4", "indexer.q_hadamard", _mc1d(4, 2, 4, 4, 4), DRAM, DRAM, M256),  # 64 blocks
    ("q_hadamard__1d_pcm3_1x4", "indexer.q_hadamard", _mc1d(4, 1, 4, 3, 4), DRAM, DRAM, M256),  # 86 blocks
    ("q_hadamard__1d_pcm4_4x2", "indexer.q_hadamard", _mc1d(4, 4, 2, 4, 4), DRAM, DRAM, M256),
    ("k_hadamard__default", "indexer.k_hadamard", None, DRAM, DRAM, M256),
    ("k_hadamard__2d_pcm1", "indexer.k_hadamard", _mc2d(4, 1, 1, 1, 1), DRAM, DRAM, M256),  # 8 x 4 = 32 cores
]


def _run(mesh_device, name, prog_config, act_mem, out_mem, variant_id=None, check_pcc=True, m=M):
    z, k, n, in0_dtype, in1_dtype, out_dtype = SHAPES[name]
    torch.manual_seed(42)
    a = torch.randn(1, z, m, k, dtype=torch.bfloat16)
    w = torch.randn(1, 1 if name in BROADCAST_WEIGHT else z, k, n, dtype=torch.bfloat16) * 0.02
    repl = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_a = ttnn.from_torch(
        a, device=mesh_device, dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, memory_config=act_mem, mesh_mapper=repl
    )
    tt_w = ttnn.from_torch(
        w, device=mesh_device, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM, mesh_mapper=repl
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    kwargs = {"program_config": prog_config} if prog_config is not None else {}
    if variant_id is not None:
        ttnn.tracy_message(f"`TT_SIGNPOST: {variant_id}`")
    for _ in range(ITERS):
        op = ttnn.matmul if name in BROADCAST_WEIGHT else ttnn.linear  # the indexer issues matmul for these
        out = op(tt_a, tt_w, memory_config=out_mem, dtype=out_dtype, compute_kernel_config=ckc, **kwargs)
        ttnn.synchronize_device(mesh_device)
        got = out
    if check_pcc:
        dev = ttnn.to_torch(got, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1]
        passing, pcc = comp_pcc(torch.matmul(a.float(), w.float()), dev.float(), PCC_REQUIRED)
        logger.info(f"[{variant_id or name}] PCC {pcc}")
        assert passing, f"{variant_id or name}: PCC {pcc} < {PCC_REQUIRED}"
    for t in (tt_a, tt_w, got):
        ttnn.deallocate(t)


@pytest.mark.parametrize("sweep_name", ["SWEEP", "SWEEP_256"], ids=["m640", "m256"])
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_glm_batch_axis_mm_sweep(mesh_device, sweep_name):
    """All variants in one device session, each preceded by a signpost naming it. A variant the op
    rejects (grid / L1 overflow) is logged and skipped rather than failing the sweep."""
    sweep = globals()[sweep_name]
    rejected, wrong = [], []
    for variant in sweep:
        variant_id, name, prog, act, out = variant[:5]
        m = variant[5] if len(variant) > 5 else M
        try:
            _run(mesh_device, name, prog, act, out, variant_id=variant_id, m=m)
        except RuntimeError as e:
            rejected.append(variant_id)
            logger.warning(f"[{variant_id}] rejected: {str(e).splitlines()[0][:200]}")
        except AssertionError as e:
            # Accepted by the op but numerically wrong -- e.g. a Reuse config with more blocks than the
            # grid has cores (wkv_b2 at per_core_M=10: 128 blocks on 110 cores) runs and returns garbage.
            wrong.append(variant_id)
            logger.error(f"[{variant_id}] WRONG OUTPUT: {e}")
    ttnn.ReadDeviceProfiler(mesh_device)
    logger.info(f"rejected variants: {rejected}")
    logger.info(f"wrong-output variants: {wrong}")


@pytest.mark.parametrize("m", [640, 256], ids=["m640", "m256"])
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("name", list(SHAPES.keys()), ids=list(SHAPES.keys()))
def test_glm_batch_axis_mm(mesh_device, name, m):
    """The configs batch-axis ttMLA / TtIndexer actually use, read from mla_config so this cannot drift."""
    from models.demos.deepseek_v3_d_p.tt.mla.mla_config import get_batch_axis_matmul_config

    cfg = get_batch_axis_matmul_config(name, m)
    assert cfg is not None, f"no batch-axis config for {name} at {m} rows"
    assert cfg["out_dtype"] == SHAPES[name][5], f"{name}: table out dtype {cfg['out_dtype']} != {SHAPES[name][5]}"
    _run(mesh_device, name, cfg["program_config"], cfg["act_mem_config"], cfg["out_mem_config"], variant_id=name, m=m)
