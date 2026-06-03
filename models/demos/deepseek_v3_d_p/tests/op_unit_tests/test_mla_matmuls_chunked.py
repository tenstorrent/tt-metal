# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Theoretical compute-duration model for the MLA matmuls under CHUNKED prefill (Kimi K2.6).

This file does NOT run any device op. It is a pure analytical model so the numbers can be
cross-checked against the tuning spreadsheet before any hand-tuned program_config is written.
See models/demos/deepseek_v3_d_p/tests/op_unit_tests/MLA_MATMUL_TUNING_CHUNKED.md.

Model (per the context doc):
    M_tiles = M / 32,  K_tiles = K / 32,  N_tiles = N / 32
    tiles   = M_tiles * K_tiles * N_tiles * batch
    cycles  = tiles * FIDELITY_CYCLES          (HiFi2 = 32)
    comp_ns = cycles / TOTAL_CORES / CLOCK_GHZ (blackhole = 1.35 GHz)
    -> NO causal /2 (that is SDPA-specific, not a plain matmul)
    DM bytes = (M*K*b0 + K*N*b1 + M*N*bout) * batch    (per dtype: bf16=2B, bf8_b=1B)
    dm_ns    = DM_bytes / DRAM_BW                       (only if DRAM_BW_GBPS is set)

Shapes are PER-CHIP (after SP/TP sharding) for the 1x4 test mesh (SP=1, TP=4):
chunk = 5k global / 8 SP = 640 tokens; on the 1x4 mesh SP=1 so global seq == per-chip M == 640.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

PCC_REQUIRED = 0.99

# ----------------------------------------------------------------------------------------------
# Model constants
# ----------------------------------------------------------------------------------------------
TILE = 32
FIDELITY_CYCLES = {"HiFi4": 64, "HiFi3": 48, "HiFi2": 32, "LoFi": 16}
FIDELITY = "HiFi2"  # matches mla.py default_compute_kernel_config

CLOCK_GHZ = 1.35  # blackhole; wormhole_b0 = 1.0
TOTAL_CORES = 110  # 11x10 grid (12x10 minus a row to dodge di/dt + throttling)

# Expected achievable utilization -> est_ns = comp_ns / TARGET_UTIL (e.g. 0.7 == 70% util).
TARGET_UTIL = 0.7

DTYPE_BYTES = {"bf16": 2, "bf8_b": 1}

# DRAM read-time model for fetching in0 + in1 (output write not counted here).
#   effective_bw = DRAM_BW_GBPS * DRAM_UTIL
#   read_ns      = (in0_bytes + in1_bytes) / effective_bw   (bytes / (GB/s) == ns)
DRAM_BW_GBPS = 512.0
DRAM_UTIL = 0.6
EFFECTIVE_DRAM_BW_GBPS = DRAM_BW_GBPS * DRAM_UTIL  # 307.2 GB/s

# ----------------------------------------------------------------------------------------------
# Kimi K2.6 dims (reference/kimi_k2_6_config.py) + chunked-prefill sharding
# ----------------------------------------------------------------------------------------------
EMB = 7168
NUM_HEADS = 64
Q_LORA = 1536
KV_LORA = 512
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
QK_HEAD = QK_NOPE + QK_ROPE  # 192

TP = 4
LOCAL_HEADS = NUM_HEADS // TP  # 16  -> batch dim of the per-head matmuls (after nlp head split)
CHUNK_M = 640  # per-chip query/kv chunk

# Each entry: per-chip M, K, N (already sharded), batch, dtypes, and TP mode (for reference only).
#   K-shard  -> in0 & in1 both split on contraction dim; output is a partial, all-reduced over TP.
#   N-shard  -> in1 split on output dim; output sharded on N.
#   head     -> batched per-head matmul; heads split over TP -> LOCAL_HEADS in the batch dim.
MATMULS = [
    {
        "name": "mm0  q_a_proj",
        "M": CHUNK_M,
        "K": EMB // TP,
        "N": Q_LORA,
        "batch": 1,
        "in0": "bf16",
        "in1": "bf8_b",
        "out": "bf16",
        "tp": "K-shard (all-reduce)",
    },
    {
        "name": "mm1  q_b_proj",
        "M": CHUNK_M,
        "K": Q_LORA,
        "N": (NUM_HEADS * QK_HEAD) // TP,
        "batch": 1,
        "in0": "bf16",
        "in1": "bf8_b",
        "out": "bf16",
        "tp": "N-shard",
    },
    {
        "name": "mm2  wkv_b1 (q absorb)",
        "M": CHUNK_M,
        "K": QK_NOPE,
        "N": KV_LORA,
        "batch": LOCAL_HEADS,
        "in0": "bf16",
        "in1": "bf8_b",
        "out": "bf16",
        "tp": "head-shard (batched)",
    },
    {
        "name": "mm3  kv_a_proj_with_mqa",
        "M": CHUNK_M,
        "K": EMB // TP,
        "N": KV_LORA + QK_ROPE,
        "batch": 1,
        "in0": "bf16",
        "in1": "bf8_b",
        "out": "bf16",
        "tp": "K-shard (all-reduce)",
    },
    {
        "name": "mm4  wkv_b2 (RELOCATED, post-SDPA)",
        "M": CHUNK_M,
        "K": KV_LORA,
        "N": V_HEAD,
        "batch": LOCAL_HEADS,
        "in0": "bf16",
        "in1": "bf8_b",
        "out": "bf8_b",
        "tp": "head-shard (batched)",
    },
    {
        "name": "mm5  o_proj",
        "M": CHUNK_M,
        "K": (NUM_HEADS * V_HEAD) // TP,
        "N": EMB,
        "batch": 1,
        "in0": "bf16",
        "in1": "bf8_b",
        "out": "bf16",
        "tp": "K-shard (all-reduce)",
    },
]


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def theoretical(mm, fidelity=FIDELITY, total_cores=TOTAL_CORES, clock_ghz=CLOCK_GHZ):
    M, K, N, batch = mm["M"], mm["K"], mm["N"], mm["batch"]
    for dim, val in (("M", M), ("K", K), ("N", N)):
        assert val % TILE == 0, f"{mm['name']}: {dim}={val} not tile-aligned ({TILE})"

    m_t, k_t, n_t = M // TILE, K // TILE, N // TILE
    tiles = m_t * k_t * n_t * batch
    cycles = tiles * FIDELITY_CYCLES[fidelity]
    comp_ns = cycles / total_cores / clock_ghz  # cycles/core / (cycles/ns)
    est_ns = comp_ns / TARGET_UTIL  # expected measured duration at TARGET_UTIL utilization

    in0_bytes = batch * M * K * DTYPE_BYTES[mm["in0"]]
    in1_bytes = batch * K * N * DTYPE_BYTES[mm["in1"]]
    read_bytes = in0_bytes + in1_bytes
    read_ns = read_bytes / EFFECTIVE_DRAM_BW_GBPS  # bytes / (GB/s) == ns

    return {
        "name": mm["name"],
        "M": M,
        "K": K,
        "N": N,
        "batch": batch,
        "m_t": m_t,
        "k_t": k_t,
        "n_t": n_t,
        "tiles": tiles,
        "cycles": cycles,
        "comp_ns": comp_ns,
        "est_ns": est_ns,
        "in0_bytes": in0_bytes,
        "in1_bytes": in1_bytes,
        "read_bytes": read_bytes,
        "read_ns": read_ns,
        # Like-for-like: read_ns is at DRAM_UTIL, so compare against est_ns (compute at TARGET_UTIL),
        # not the ideal comp_ns. With double-buffering the op runs at max(read_ns, est_ns).
        "bound": ("DM" if read_ns > est_ns else "compute"),
        "tp": mm["tp"],
    }


def format_table(rows):
    hdr = (
        f"{'matmul':<34} {'M':>5} {'K':>5} {'N':>5} {'b':>3} "
        f"{'comp_ns':>10} {'est_ns':>10} {'in0_MB':>7} {'in1_MB':>7} {'read_ns':>10} {'bound':>7}  tp-mode"
    )
    lines = [
        f"MLA chunked-prefill theoretical  (Kimi K2.6, chunk M={CHUNK_M}, TP={TP}, "
        f"{FIDELITY}={FIDELITY_CYCLES[FIDELITY]}cyc, cores={TOTAL_CORES}, clk={CLOCK_GHZ}GHz, util={TARGET_UTIL:.0%})",
        f"  est_ns  = comp_ns / {TARGET_UTIL}",
        f"  read_ns = (in0+in1 bytes) / ({DRAM_BW_GBPS:.0f} GB/s * {DRAM_UTIL:.0%} = {EFFECTIVE_DRAM_BW_GBPS:.1f} GB/s)",
        hdr,
        "-" * len(hdr),
    ]
    for r in rows:
        lines.append(
            f"{r['name']:<34} {r['M']:>5} {r['K']:>5} {r['N']:>5} {r['batch']:>3} "
            f"{r['comp_ns']:>10.1f} {r['est_ns']:>10.1f} "
            f"{r['in0_bytes'] / 1e6:>7.2f} {r['in1_bytes'] / 1e6:>7.2f} {r['read_ns']:>10.1f} "
            f"{r['bound']:>7}  {r['tp']}"
        )
    total = sum(r["comp_ns"] for r in rows)
    total_est = sum(r["est_ns"] for r in rows)
    total_read = sum(r["read_ns"] for r in rows)
    lines.append("-" * len(hdr))
    lines.append(
        f"{'TOTAL (sum of all matmuls)':<34} {'':>5} {'':>5} {'':>5} {'':>3} "
        f"{total:>10.1f} {total_est:>10.1f} {'':>7} {'':>7} {total_read:>10.1f}"
    )
    return "\n".join(lines)


def test_mla_mm_theoretical_durations():
    """Emit the per-matmul theoretical compute duration; asserts only tile-alignment."""
    rows = [theoretical(mm) for mm in MATMULS]
    logger.info("\n" + format_table(rows))
    for r in rows:
        assert r["comp_ns"] > 0


@pytest.mark.parametrize("mm", MATMULS, ids=[m["name"].split()[0] for m in MATMULS])
def test_single_matmul_theoretical(mm):
    """Per-matmul variant so a single mm can be inspected: `pytest ... -k mm4`."""
    r = theoretical(mm)
    logger.info(
        f"{r['name']}: M={r['M']} K={r['K']} N={r['N']} batch={r['batch']} | "
        f"tiles(m,k,n)=({r['m_t']},{r['k_t']},{r['n_t']}) total_tiles={r['tiles']} "
        f"cycles={r['cycles']} -> comp_ns={r['comp_ns']:.1f} "
        f"est_ns@{TARGET_UTIL:.0%}={r['est_ns']:.1f} | "
        f"read_ns={r['read_ns']:.1f} (in0={r['in0_bytes']/1e6:.2f}MB in1={r['in1_bytes']/1e6:.2f}MB) "
        f"({r['bound']}-bound)"
    )
    assert r["comp_ns"] > 0


# ==============================================================================================
# Device tests — run the real ttnn.linear with a hand-tuned program_config and check PCC.
#
# Mesh: 1x4 (SP=0 axis size 1, TP=1 axis size 4). SP=1 means seq is NOT sharded, so the global
# in0 seq dim == per-chip chunk M = CHUNK_M. TP=4 shards K/N/heads as per each matmul.
# Profile via:  python -m tracy -p -r -m "pytest <this_file>::test_mla_mm_device -k mm0"
# then read column 19 (DEVICE KERNEL DURATION [ns]) and compare to the theoretical row above.
# ==============================================================================================

GRID_11x10 = (11, 10)  # 110 cores (12x10 minus a row to dodge di/dt + throttling)

# --- mm0  q_a_proj (2D) ----------------------------------------------------------------------
# per-chip: M_t=20 (640/32), K_t=56 (1792/32), N_t=48 (1536/32).
# Cores: per_core_M=2 -> ceil(20/2)=10 M-cores; per_core_N=5 -> ceil(48/5)=10 N-cores => 100 cores.
# Subblock: h*w<=8, h|per_core_M, w|per_core_N -> h=1,w=5 (5 dst tiles). in0_block_w|K_t=56 -> 8.
# Measured ~21.8us @ ~53% util. ABLATION VERDICT: orchestration-bound (7.5us mcast/sync floor of
# the 22us op); weight read 1.76us, output write 1.43us, compute 0.82us, act read 0.31us -> no
# single resource dominates. memory_config/dtype/block tweaks don't help; ~53% is the practical ceiling.
prog_config_mm0 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=GRID_11x10,
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=5,
    per_core_M=2,
    per_core_N=5,
    transpose_mcast=False,
    fuse_batch=False,
    fused_activation=None,
)

# --- mm1  q_b_proj (2D, N-sharded) -----------------------------------------------------------
# per-chip: M_t=20, K_t=48 (1536/32), N_t=96 (3072/32 = 12288/4 N-shard). in0 NOT TP-sharded.
# Cores: per_core_M=2 -> 10 M-cores; per_core_N=9 -> ceil(96/9)=11 N-cores => 110 cores (max).
# SWEPT: in0_block_w=8 is the sweet spot (4/6 and 12/16/24 all worse); 110 cores beats 80/100c;
# subblock 1x3 (==2x1) beats 2x3 and 1x1. With in0+out L1 interleaved: 67.8% util (29.3us), and very
# stable (~60ns per-device spread vs ~600ns for DRAM-in). Ablation: ~32% orchestration floor +
# overlapped compute(~16us) ‖ DM(~19us, weight-read led). ~68% is near the practical ceiling here.
prog_config_mm1 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=GRID_11x10,
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=3,
    per_core_M=2,
    per_core_N=9,
    transpose_mcast=False,
    fuse_batch=False,
    fused_activation=None,
)

# --- mm5  o_proj (2D, K-shard contraction) ---------------------------------------------------
# per-chip: M_t=20, K_t=64 (2048/32 = 8192/4 K-shard), N_t=224 (7168/32). need_tp_sum (like mm0/mm3).
# Cores: per_core_M=2 -> 10 M-cores; per_core_N=21 -> ceil(224/21)=11 N-cores => 110 cores (max;
# 224 has no even divisor giving <=11 cores, so last N-core is ragged). Subblock 1x7 (7 dst tiles,
# w|21). in0_block_w|K_t=64 -> 8. Output is big (9.2MB) -> keep in L1 (DRAM write would cost ~30us).
# SWEPT: ~74.5% util (82.9us), already above 70% target and at its ceiling (bw8/bw32 & sb 1x7/2x3 all
# tie ~74.5%; 80c -> 57%). COMPUTE-DOMINANT regime: 61.8us compute amortizes the ~8-10us orch floor.
prog_config_mm5 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=GRID_11x10,
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=7,
    per_core_M=2,
    per_core_N=21,
    transpose_mcast=False,
    fuse_batch=False,
    fused_activation=None,
)

# --- mm3  kv_a_proj_with_mqa (2D, K-shard contraction) ---------------------------------------
# per-chip: M_t=20, K_t=56 (1792/32 = 7168/4 K-shard), N_t=18 (576/32 = 512+64). need_tp_sum
# (like mm0/mm5: in0 K-shard dim3 + in1 K-shard dim2 -> partial output all-reduced over TP).
# Cores: per_core_M=2 -> 10 M-cores; per_core_N=2 -> ceil(18/2)=9 N-cores => 90 cores. N_t=18 has
# no divisor giving 10-11 N-cores within grid_x=11, so 90 is this shape's core ceiling.
# Subblock: per_core_M=2, per_core_N=2 -> sb2x1 (h|2, w|2; sb1x2 ties, 2x2 worse).
# in0_block_w|K_t=56 -> 14 is the in0-L1 sweet spot (4 mcast rounds): bw14=11.38us beats bw8=12.55us
# by ~9%. NOTE: the earlier in0-DRAM sweep picked bw8; the L1 regime shifts the optimum to bw14
# (bigger payloads are cheap from L1, fewer rounds win). Always re-sweep in0_block_w after L1<->DRAM.
# in0+out L1 -> 11.38us (38.2% util, comp_ns 4344 / measured) vs 16.0us in0-DRAM; in0-L1 also
# stabilizes spread 578ns -> ~40ns (mm1 signature).
# ABLATION VERDICT (baseline 11.37us): ORCHESTRATION-BOUND once in0 is in L1. all-off floor = 5.58us
# (~49% of the op: mcast/semaphore/CB-sync/dispatch). Individual deltas all small/overlapped: act-read
# 0.77us, weight-read 0.03us (fully hidden), out-write 0.50us, compute 1.13us (5.3us of math exposes
# only 1.1us, rest hidden behind DM). The bound SHIFTED: in0-from-DRAM was DM-bound (act-read pole ->
# why in0->L1 bought 21%); in0-in-L1 leaves the orch floor. Blocks/dtype/memory can't beat 5.6us of
# pure mcast at 90c -- mm3 is at its practical ceiling.
# 90c is the 2D ceiling (N_t=18 -> ceil(18/2)=9 N-cores; pcn=1 -> 18 > grid_x=11). N-padding to force
# 100c/110c (test_mm3_sweep_pad) is flat-to-worse at every block-width: more compute cores can't move
# a DM-bound op (the ~8% gain first seen at 100c was purely the bw14 effect, reproduced clean at 90c).
prog_config_mm3 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=GRID_11x10,
    in0_block_w=14,
    out_subblock_h=2,
    out_subblock_w=1,
    per_core_M=2,
    per_core_N=2,
    transpose_mcast=False,
    fuse_batch=False,
    fused_activation=None,
)

# --- mm4  wkv_b2 (RELOCATED, post-SDPA; batched head-shard, 1D) -------------------------------
# TRUE batched matmul (NOT broadcast): post-SDPA in0 is the per-head SDPA output, so BOTH operands are
# batch=64 heads -> TP-shard dim1 -> 16/chip. in0 [1,16,640,512] (10.5MB, the DM long pole),
# in1 [1,16,512,128], out [1,16,640,128] bf8_b. (Pre-relocation it was in0 batch=1 shared latent;
# the move to per-head SDPA output is exactly what makes in0 explode and mm4 DM-bound.)
# per-chip tiles: M_t=20, K_t=16, N_t=4, batch=16.
# WHY NOT MultiCast1D: that batched path sets cores = ceil(Mt/pcm)*ceil(Nt/pcn) with Mt=PER-BATCH=20
# and LOOPS the 16 heads in-kernel (never on the grid); Nt=4 can't be split (pcn<4 -> wrong PCC). So it
# caps at 20 cores -> 16 tiny per-head matmuls run serially -> 121us (in0+out L1 = no change, NOT
# DM-bound, SERIALIZATION-bound). fuse_batch=True can't rescue it (FATALs 'get_batch_size(b)==1', only
# valid for shared weights).
# FIX = the older MatmulMultiCoreReuseProgramConfig (non-mcast): cores = (B*Mt/pcm)*(Nt/pcn) -- it FOLDS
# B*Mt=320, so batch finally maps to cores. Constraints: pcm | 320 AND pcm | per-batch M=20 -> pcm in
# {1,2,4,5,10,20}; pcn | Nt=4. pcm4/pcn4 = 80 cores (the max <=110). Works correctly on L1-interleaved
# in0/in1 (PCC 0.999, no sharding needed). Cores drive perf (80->26us, 64->30, 32->56, 16->86); small
# in0_block_w wins (bw2). Best: bw2/sb4x1/pcm4/pcn4 -> 26.2us = 4.6x faster than the 121us MultiCast1D.
# Absolute: 26.2us vs compute floor 4.4us@110c (unreachable) / 6.1us@80c (real cap) -> 80c is the cap
# because B*Mt=320, Nt=4 only reach 80 within the 110 grid; mm4 is now core-bound at its 80c ceiling.
prog_config_mm4 = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=GRID_11x10,
    in0_block_w=2,
    out_subblock_h=4,
    out_subblock_w=1,
    per_core_M=4,
    per_core_N=4,
)

# --- mm2  wkv_b1 (q absorb; batched head-shard) ----------------------------------------------
# Same shape family as mm4 (true batched, both operands batch=16 heads -> TP-shard dim1 -> 16/chip),
# transposed dims: in0 q_nope [1,16,640,128] bf16, in1 wkv_b1 weight [1,16,128,512] bf8_b,
# out [1,16,640,512] bf16 (out is the 10.5MB tensor here, the N=512 side). per-chip tiles: M_t=20,
# K_t=4, N_t=16, batch=16. Use MatmulMultiCoreReuse (folds B*Mt=320 onto the grid; MultiCast1D caps
# at Mt=20). CONSTRAINT: per_core_N must == Nt (N is never split), so cores = 320/pcm; pcm in
# {1,2,4,5,10,20} -> pcm4 = 80 cores (max <=110). Swept: 80c -> 39us, subblock/bw ~noise (bw2/sb2x4).
# out-L1 essential (+41%, the 10.5MB output write dominates); in0 placement free (2.62MB).
prog_config_mm2 = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=GRID_11x10,
    in0_block_w=2,
    out_subblock_h=2,
    out_subblock_w=4,
    per_core_M=4,
    per_core_N=16,
)

# Global shapes (the 1x4 mesh shards them per the flags). SP=1 so in0 seq == CHUNK_M.
DEVICE_MATMULS = [
    {
        "name": "mm0_q_a_proj",
        "in0_shape": (1, 1, CHUNK_M, EMB),  # [1,1,640,7168]
        "in1_shape": (1, 1, EMB, Q_LORA),  # [1,1,7168,1536]
        "in0_sp_sharded": True,
        "in0_tp_sharded": True,
        "in0_tp_shard_dim": 3,  # K-shard activation
        "in1_tp_sharded": True,
        "in1_tp_shard_dim": 2,  # K-shard weight -> partial output, all-reduced over TP
        "in0_dtype": ttnn.bfloat16,
        "in1_dtype": ttnn.bfloat8_b,
        "out_dtype": ttnn.bfloat16,
        "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,
        "program_config": prog_config_mm0,
    },
    {
        "name": "mm1_q_b_proj",
        "in0_shape": (1, 1, CHUNK_M, Q_LORA),  # [1,1,640,1536]
        "in1_shape": (1, 1, Q_LORA, NUM_HEADS * QK_HEAD),  # [1,1,1536,12288] (Kimi 64 heads)
        "in0_sp_sharded": True,
        "in0_tp_sharded": False,  # in0 (q_lora) replicated across TP
        "in0_tp_shard_dim": None,
        "in1_tp_sharded": True,
        "in1_tp_shard_dim": 3,  # N-shard weight -> output sharded on N (concat, no all-reduce)
        "in0_dtype": ttnn.bfloat16,
        "in1_dtype": ttnn.bfloat8_b,
        "out_dtype": ttnn.bfloat16,
        "act_mem_config": ttnn.L1_MEMORY_CONFIG,  # input + output L1 interleaved
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,
        "program_config": prog_config_mm1,
    },
    {
        "name": "mm3_kv_a_proj_with_mqa",
        "in0_shape": (1, 1, CHUNK_M, EMB),  # [1,1,640,7168]
        "in1_shape": (1, 1, EMB, KV_LORA + QK_ROPE),  # [1,1,7168,576]
        "in0_sp_sharded": True,
        "in0_tp_sharded": True,
        "in0_tp_shard_dim": 3,  # K-shard activation
        "in1_tp_sharded": True,
        "in1_tp_shard_dim": 2,  # K-shard weight -> partial output, all-reduced over TP
        "in0_dtype": ttnn.bfloat16,
        "in1_dtype": ttnn.bfloat8_b,
        "out_dtype": ttnn.bfloat16,
        "act_mem_config": ttnn.L1_MEMORY_CONFIG,  # DM-bound: in0+out L1 -> 11.38us vs 16.0us DRAM-in
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,
        "program_config": prog_config_mm3,
    },
    {
        "name": "mm5_o_proj",
        "in0_shape": (1, 1, CHUNK_M, NUM_HEADS * V_HEAD),  # [1,1,640,8192]
        "in1_shape": (1, 1, NUM_HEADS * V_HEAD, EMB),  # [1,1,8192,7168]
        "in0_sp_sharded": True,
        "in0_tp_sharded": True,
        "in0_tp_shard_dim": 3,  # K-shard activation
        "in1_tp_sharded": True,
        "in1_tp_shard_dim": 2,  # K-shard weight -> partial output, all-reduced over TP
        "in0_dtype": ttnn.bfloat16,
        "in1_dtype": ttnn.bfloat8_b,
        "out_dtype": ttnn.bfloat16,
        "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,  # L1-in tried, zero change (compute-dominant)
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,  # output L1 (DRAM write of 9.2MB would cost ~30us)
        "program_config": prog_config_mm5,
    },
    {
        "name": "mm4_wkv_b2",  # batched head-shard, 1D; appended last so mm5 stays index 3
        # RELOCATED post-SDPA: in0 is the PER-HEAD SDPA output (batch=heads), NOT the pre-SDPA shared
        # latent (batch=1). That is the whole point of the move -> in0 grows to 16*640*512 = 10.5MB,
        # making mm4 heavily DM-bound on the activation read (model read_ns 37.5us).
        "in0_shape": (1, NUM_HEADS, CHUNK_M, KV_LORA),  # [1,64,640,512] SDPA out, batch=64 heads
        "in1_shape": (1, NUM_HEADS, KV_LORA, V_HEAD),  # [1,64,512,128] per-head weight, batch=64
        "in0_sp_sharded": True,  # seq dim2 (SP=1 -> no-op)
        "in0_tp_sharded": True,
        "in0_tp_shard_dim": 1,  # head/batch shard -> 16 heads/chip
        "in1_tp_sharded": True,
        "in1_tp_shard_dim": 1,  # head/batch shard -> output sharded on dim1 (concat heads, no sum)
        "in0_dtype": ttnn.bfloat16,
        "in1_dtype": ttnn.bfloat8_b,
        "out_dtype": ttnn.bfloat8_b,
        # MEM SWEEP on the 80c config: in0->L1 saves 39% (43.2us in0-DRAM -> 26.3us in0-L1), out->L1
        # another ~8%. Opposite of the MultiCast1D case (in0-L1 was a no-op there) because that was
        # serialization-bound; at 80c the 10.5MB in0 DRAM read IS the bottleneck. L1/L1 optimal.
        "act_mem_config": ttnn.L1_MEMORY_CONFIG,  # in0 is 10.5MB SDPA out -> L1 (39% win at 80c)
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,
        "program_config": prog_config_mm4,  # MultiCoreReuse, 80 cores
    },
    {
        "name": "mm2_wkv_b1",  # batched head-shard, MultiCoreReuse; appended last (index 5)
        "in0_shape": (1, NUM_HEADS, CHUNK_M, QK_NOPE),  # [1,64,640,128] q_nope, batch=64 heads
        "in1_shape": (1, NUM_HEADS, QK_NOPE, KV_LORA),  # [1,64,128,512] per-head weight, batch=64
        "in0_sp_sharded": True,  # seq dim2 (SP=1 -> no-op)
        "in0_tp_sharded": True,
        "in0_tp_shard_dim": 1,  # head/batch shard -> 16 heads/chip
        "in1_tp_sharded": True,
        "in1_tp_shard_dim": 1,  # head/batch shard -> output sharded on dim1 (concat heads, no sum)
        "in0_dtype": ttnn.bfloat16,
        "in1_dtype": ttnn.bfloat8_b,
        "out_dtype": ttnn.bfloat16,
        # MEM SWEEP (80c): out->L1 is the big lever (+41%: 39.5us vs 55.7us out-DRAM) -- the 10.5MB
        # output WRITE is the bottleneck (mirror of mm4 where the big in0 READ was). in0 placement is
        # free (2.62MB, within noise) -> keep in0 in DRAM to free L1 for the big output.
        "act_mem_config": ttnn.DRAM_MEMORY_CONFIG,  # in0 only 2.62MB; DRAM is free here
        "out_mem_config": ttnn.L1_MEMORY_CONFIG,  # out is 10.5MB -> L1 (41% win)
        "program_config": prog_config_mm2,  # MultiCoreReuse, 80 cores
    },
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("cfg", DEVICE_MATMULS, ids=[m["name"] for m in DEVICE_MATMULS])
def test_mla_mm_device(mesh_device, cfg):
    _run_device_matmul(mesh_device, cfg)


def _run_device_matmul(mesh_device, cfg):
    torch.manual_seed(42)
    in0_x, in0_y, in0_z, in0_w = cfg["in0_shape"]
    in1_x, in1_y, in1_z, in1_w = cfg["in1_shape"]

    hidden_states = torch.randn(in0_x, in0_y, in0_z, in0_w, dtype=torch.bfloat16)
    weight = torch.randn(in1_x, in1_y, in1_z, in1_w, dtype=torch.bfloat16) * 0.02
    weight_ref = weight  # reference uses the UNPADDED weight

    # Optional N-padding: pad weight's N (dim3) with zeros so N_t rounds up and more N-cores come
    # online (the extra cores compute padded, sliced-off tiles). Used to test >ceil(N_t/pcn) splits.
    n_pad_to = cfg.get("n_pad_to")
    if n_pad_to and n_pad_to > in1_w:
        weight = torch.nn.functional.pad(weight, (0, n_pad_to - in1_w))
    eff_n = weight.shape[-1]

    sp_axis, tp_axis = 0, 1

    # in0 sharding: SP on dim2 (seq), TP on cfg["in0_tp_shard_dim"].
    in0_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    if cfg["in0_sp_sharded"] or cfg["in0_tp_sharded"]:
        shard_dims = [None, None]
        if cfg["in0_sp_sharded"]:
            shard_dims[sp_axis] = 2
        if cfg["in0_tp_sharded"]:
            shard_dims[tp_axis] = cfg["in0_tp_shard_dim"]
        in0_mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims)
    tt_input = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        dtype=cfg["in0_dtype"],
        layout=ttnn.TILE_LAYOUT,
        memory_config=cfg["act_mem_config"],
        mesh_mapper=in0_mesh_mapper,
    )

    in1_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    if cfg["in1_tp_sharded"]:
        shard_dims = [None, None]
        shard_dims[tp_axis] = cfg["in1_tp_shard_dim"]
        in1_mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims)
    tt_weight = ttnn.from_torch(
        weight,
        device=mesh_device,
        dtype=cfg["in1_dtype"],
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=in1_mesh_mapper,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    logger.info(f"{cfg['name']}: in0={list(tt_input.shape)} in1={list(tt_weight.shape)}")

    tt_output = ttnn.linear(
        tt_input,
        tt_weight,
        memory_config=cfg["out_mem_config"],
        compute_kernel_config=compute_kernel_config,
        dtype=cfg["out_dtype"],
        program_config=cfg["program_config"],
    )
    ttnn.synchronize_device(mesh_device)

    # For DM-kernel ablation runs the CBs hold garbage, so PCC is meaningless -> skip the check.
    if not cfg.get("check_pcc", True):
        logger.info(f"{cfg['name']}: PCC check skipped (ablation run)")
        return

    # Reference + concat back. K-shard contraction (in0 dim3 + in1 dim2) -> sum partials over TP.
    reference_output = torch.matmul(hidden_states, weight_ref)
    concat_dims = [None, None]
    if cfg["in0_sp_sharded"]:
        concat_dims[sp_axis] = 2
    need_tp_sum = (
        cfg["in0_tp_sharded"]
        and cfg["in0_tp_shard_dim"] == 3
        and cfg["in1_tp_sharded"]
        and cfg["in1_tp_shard_dim"] == 2
    )
    if cfg["in0_tp_sharded"] and cfg["in0_tp_shard_dim"] != 3:
        concat_dims[tp_axis] = cfg["in0_tp_shard_dim"]
    elif cfg["in1_tp_sharded"] and cfg["in1_tp_shard_dim"] == 3:
        concat_dims[tp_axis] = 3
    elif cfg["in1_tp_sharded"] and cfg["in1_tp_shard_dim"] == 1:
        # Batched head-shard (mm2/mm4): in0 batch=1 broadcast, in1 batch=heads sharded on dim1 ->
        # output inherits the head shard on dim1 (concat heads back over TP, no sum).
        concat_dims[tp_axis] = 1
    elif need_tp_sum:
        concat_dims[tp_axis] = 3

    tt_full = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape),
    )
    if need_tp_sum:
        tp = mesh_device.shape[tp_axis]
        tt_torch = tt_full.reshape(in0_x, in0_y, in0_z, tp, eff_n).sum(dim=3)
    else:
        tt_torch = tt_full
    if n_pad_to:  # drop the padded N columns before comparing
        tt_torch = tt_torch[..., :in1_w]

    passing, pcc = comp_pcc(reference_output, tt_torch, PCC_REQUIRED)
    logger.info(f"{cfg['name']} PCC={pcc} (required {PCC_REQUIRED})")
    assert passing, f"{cfg['name']} failed PCC {pcc} < {PCC_REQUIRED}"


# ----------------------------------------------------------------------------------------------
# mm0 program_config sweep — profile all variants in one tracy run, then read the MatmulDevice-
# Operation rows in order (4 device rows per variant) and compare DEVICE KERNEL DURATION.
#   python -m tracy -p -r -m "pytest <file>::test_mm0_sweep"
# ----------------------------------------------------------------------------------------------
def _mm0_2d(in0_block_w, osh, osw, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=GRID_11x10,
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


# label -> config. Cores = ceil(20/per_core_M) * ceil(48/per_core_N).
SWEEP_MM0 = [
    ("bw7_100c", _mm0_2d(7, 1, 5, 2, 5)),
    ("bw8_100c", _mm0_2d(8, 1, 5, 2, 5)),
    ("bw14_100c", _mm0_2d(14, 1, 5, 2, 5)),  # baseline (52% util)
    ("bw28_100c", _mm0_2d(28, 1, 5, 2, 5)),
    ("bw56_100c", _mm0_2d(56, 1, 5, 2, 5)),  # whole K in one block
    ("bw28_80c", _mm0_2d(28, 1, 6, 2, 6)),  # per_core_N=6 -> 8 N-cores, 80 cores
    ("bw28_50c", _mm0_2d(28, 1, 5, 4, 5)),  # per_core_M=4 -> 5 M-cores, 50 cores
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config", SWEEP_MM0, ids=[s[0] for s in SWEEP_MM0])
def test_mm0_sweep(mesh_device, label, program_config):
    base = DEVICE_MATMULS[0]
    cfg = {**base, "name": f"mm0_{label}", "program_config": program_config}
    _run_device_matmul(mesh_device, cfg)


# label -> (act_mem_config, out_mem_config), all on the best compute config (bw8, 100 cores).
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
SWEEP_MM0_MEM = [
    ("in0DRAM_outL1", DRAM, L1),  # baseline
    ("in0L1_outL1", L1, L1),
    ("in0L1_outDRAM", L1, DRAM),
    ("in0DRAM_outDRAM", DRAM, DRAM),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,act_mc,out_mc", SWEEP_MM0_MEM, ids=[s[0] for s in SWEEP_MM0_MEM])
def test_mm0_sweep_mem(mesh_device, label, act_mc, out_mc):
    base = DEVICE_MATMULS[0]
    cfg = {
        **base,
        "name": f"mm0_{label}",
        "program_config": _mm0_2d(8, 1, 5, 2, 5),
        "act_mem_config": act_mc,
        "out_mem_config": out_mc,
    }
    _run_device_matmul(mesh_device, cfg)


# ----------------------------------------------------------------------------------------------
# mm0 DM-kernel ablation reference run (PCC disabled). Use this fixed config, then comment out
# the NOC reads in the in0/in1 sender kernels and re-run to measure each transfer's contribution:
#   reader_bmm_tile_layout_in0_sender_padding.cpp       -> activation read
#   reader_bmm_tile_layout_in1_sender_writer_padding.cpp-> weight read
#   python -m tracy -p -r -m "pytest <file>::test_mm0_ablation"
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("cfg", DEVICE_MATMULS, ids=[m["name"] for m in DEVICE_MATMULS])
def test_mm_ablation(mesh_device, cfg):
    # PCC disabled so CB garbage is fine. Select one matmul (-k mm1), then comment NOC reads/writes
    # or matmul_block in the kernels (scripts/ablate_kernel.py) and re-run to measure each delta.
    _run_device_matmul(mesh_device, {**cfg, "check_pcc": False})


# mm1 program_config sweep. mm1: M_t=20, K_t=48, N_t=96. _mm0_2d(in0_block_w,osh,osw,pcm,pcn).
# Cores = ceil(20/pcm)*ceil(96/pcn). subblock osh|pcm, osw|pcn, osh*osw<=8.
SWEEP_MM1 = [
    ("bw8_110c_sb2x3", _mm0_2d(8, 2, 3, 2, 9)),  # current (pN9->11 N-cores =110)
    ("bw4_110c", _mm0_2d(4, 2, 3, 2, 9)),
    ("bw6_110c", _mm0_2d(6, 2, 3, 2, 9)),
    ("bw12_110c", _mm0_2d(12, 2, 3, 2, 9)),
    ("bw16_110c", _mm0_2d(16, 2, 3, 2, 9)),
    ("bw24_110c", _mm0_2d(24, 2, 3, 2, 9)),
    ("bw8_110c_sb1x3", _mm0_2d(8, 1, 3, 2, 9)),
    ("bw8_100c_sb1x5", _mm0_2d(8, 1, 5, 2, 10)),  # pN10 -> 100 cores
    ("bw8_80c_sb2x4", _mm0_2d(8, 2, 4, 2, 12)),  # pN12 -> 80 cores, fat 2x4=8 subblock
    ("bw12_80c_sb2x4", _mm0_2d(12, 2, 4, 2, 12)),
    # subblock micro-sweep on the winning bw8/110c (pN9 -> osw in {1,3}, osh in {1,2}):
    ("bw8_110c_sb1x1", _mm0_2d(8, 1, 1, 2, 9)),
    ("bw8_110c_sb2x1", _mm0_2d(8, 2, 1, 2, 9)),
    ("bw6_110c_sb1x3", _mm0_2d(6, 1, 3, 2, 9)),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config", SWEEP_MM1, ids=[s[0] for s in SWEEP_MM1])
def test_mm1_sweep(mesh_device, label, program_config):
    base = DEVICE_MATMULS[1]
    cfg = {**base, "name": f"mm1_{label}", "program_config": program_config}
    _run_device_matmul(mesh_device, cfg)


# mm5 program_config sweep. mm5: M_t=20, K_t=64, N_t=224. Cores = ceil(20/pcm)*ceil(224/pcn).
# pN=21 -> 11 N-cores (110, ragged); pN=28 -> 8 N-cores (80, even). subblock osh|pcm, osw|pcn, <=8.
SWEEP_MM5 = [
    ("bw8_110c_1x7", _mm0_2d(8, 1, 7, 2, 21)),  # current 74.5%
    ("bw4_110c_1x7", _mm0_2d(4, 1, 7, 2, 21)),
    ("bw16_110c_1x7", _mm0_2d(16, 1, 7, 2, 21)),
    ("bw32_110c_1x7", _mm0_2d(32, 1, 7, 2, 21)),
    ("bw8_110c_1x3", _mm0_2d(8, 1, 3, 2, 21)),
    ("bw8_110c_2x3", _mm0_2d(8, 2, 3, 2, 21)),
    ("bw8_110c_2x1", _mm0_2d(8, 2, 1, 2, 21)),
    ("bw16_110c_1x3", _mm0_2d(16, 1, 3, 2, 21)),
    ("bw8_80c_2x4", _mm0_2d(8, 2, 4, 2, 28)),  # pN28 -> 80 cores, fat 2x4=8 subblock
    ("bw8_80c_1x7", _mm0_2d(8, 1, 7, 2, 28)),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config", SWEEP_MM5, ids=[s[0] for s in SWEEP_MM5])
def test_mm5_sweep(mesh_device, label, program_config):
    base = DEVICE_MATMULS[3]  # mm5 is index 3 in DEVICE_MATMULS (after mm0, mm1, mm3)
    cfg = {**base, "name": f"mm5_{label}", "program_config": program_config}
    _run_device_matmul(mesh_device, cfg)


# mm3 program_config sweep. mm3: M_t=20, K_t=56, N_t=18. Cores = ceil(20/pcm)*ceil(18/pcn).
# pcm=2 -> 10 M-cores (max); pcn=2 -> 9 N-cores (90, ceiling); pcn=3 -> 6 N-cores (60). subblock
# osh|pcm, osw|pcn, osh*osw<=8. in0_block_w|K_t=56 -> {7,8,14,28,56}.
SWEEP_MM3 = [
    ("bw8_90c_2x2", _mm0_2d(8, 2, 2, 2, 2)),  # current
    ("bw7_90c_2x2", _mm0_2d(7, 2, 2, 2, 2)),
    ("bw14_90c_2x2", _mm0_2d(14, 2, 2, 2, 2)),
    ("bw28_90c_2x2", _mm0_2d(28, 2, 2, 2, 2)),
    ("bw56_90c_2x2", _mm0_2d(56, 2, 2, 2, 2)),  # whole K in one block
    ("bw8_90c_1x2", _mm0_2d(8, 1, 2, 2, 2)),
    ("bw8_90c_2x1", _mm0_2d(8, 2, 1, 2, 2)),
    ("bw8_90c_1x1", _mm0_2d(8, 1, 1, 2, 2)),
    ("bw8_60c_2x3", _mm0_2d(8, 2, 3, 2, 3)),  # pcn=3 -> 6 N-cores, 60 cores
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config", SWEEP_MM3, ids=[s[0] for s in SWEEP_MM3])
def test_mm3_sweep(mesh_device, label, program_config):
    base = DEVICE_MATMULS[2]  # mm3 is index 2 in DEVICE_MATMULS
    cfg = {**base, "name": f"mm3_{label}", "program_config": program_config}
    _run_device_matmul(mesh_device, cfg)


# mm3 memory-config sweep on the best compute config (bw8, 90c, sb2x1). DM-bound per model
# (read 10.8us vs est 6.2us), so in0-L1 may help; output is small (0.74MB) -> keep L1.
SWEEP_MM3_MEM = [
    ("in0DRAM_outL1", DRAM, L1),  # baseline
    ("in0L1_outL1", L1, L1),
    ("in0L1_outDRAM", L1, DRAM),
    ("in0DRAM_outDRAM", DRAM, DRAM),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,act_mc,out_mc", SWEEP_MM3_MEM, ids=[s[0] for s in SWEEP_MM3_MEM])
def test_mm3_sweep_mem(mesh_device, label, act_mc, out_mc):
    base = DEVICE_MATMULS[2]
    cfg = {
        **base,
        "name": f"mm3_{label}",
        "program_config": _mm0_2d(8, 2, 1, 2, 2),
        "act_mem_config": act_mc,
        "out_mem_config": out_mc,
    }
    _run_device_matmul(mesh_device, cfg)


# mm3 N-padding sweep: pad N_t 18 -> 20/22 so per_core_N=2 yields 10/11 N-cores (100/110 cores),
# the extra ragged cores computing padded (sliced-off) tiles. Tests whether more cores beats the
# clean 90c despite mm3 being DM-bound. All on the best in0+out-L1 path (set in DEVICE_MATMULS[2]).
# 100c: pad N->640 (N_t=20), pcn=2 -> 10 N-cores. 110c: pad N->704 (N_t=22), pcn=2 -> 11 N-cores.
# Sweep in0_block_w {7,8,14,28} and subblock {2x1,2x2,1x2} at each, vs the clean 90c baseline.
SWEEP_MM3_PAD = [
    # --- 90 cores (N_t=18, no pad): block-width sweep at in0-L1 to separate bw-effect from cores ---
    ("90c_bw8_2x1", _mm0_2d(8, 2, 1, 2, 2), None),
    ("90c_bw14_2x1", _mm0_2d(14, 2, 1, 2, 2), None),
    ("90c_bw28_2x1", _mm0_2d(28, 2, 1, 2, 2), None),
    ("90c_bw56_2x1", _mm0_2d(56, 2, 1, 2, 2), None),
    # --- 100 cores (N_t=20) ---
    ("100c_bw8_2x1", _mm0_2d(8, 2, 1, 2, 2), 640),
    ("100c_bw14_2x1", _mm0_2d(14, 2, 1, 2, 2), 640),
    ("100c_bw14_2x2", _mm0_2d(14, 2, 2, 2, 2), 640),
    ("100c_bw14_1x2", _mm0_2d(14, 1, 2, 2, 2), 640),
    ("100c_bw28_2x1", _mm0_2d(28, 2, 1, 2, 2), 640),
    ("100c_bw56_2x1", _mm0_2d(56, 2, 1, 2, 2), 640),
    # --- 110 cores (N_t=22) ---
    ("110c_bw8_2x1", _mm0_2d(8, 2, 1, 2, 2), 704),
    ("110c_bw14_2x1", _mm0_2d(14, 2, 1, 2, 2), 704),
    ("110c_bw28_2x1", _mm0_2d(28, 2, 1, 2, 2), 704),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config,n_pad_to", SWEEP_MM3_PAD, ids=[s[0] for s in SWEEP_MM3_PAD])
def test_mm3_sweep_pad(mesh_device, label, program_config, n_pad_to):
    base = DEVICE_MATMULS[2]  # mm3, in0+out L1
    cfg = {**base, "name": f"mm3_{label}", "program_config": program_config, "n_pad_to": n_pad_to}
    _run_device_matmul(mesh_device, cfg)


# mm4 MultiCast1D sweep (DEAD END, kept for the record): this batched path caps at Mt=20 cores because
# batch is looped in-kernel. _mm4_1d(bw,osh,osw,pcm,pcn); osh|pcm, osw|pcn, osh*osw<=8, bw|K_t=16.
def _mm4_1d(in0_block_w, osh, osw, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID_11x10,
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
    )


SWEEP_MM4 = [
    ("bw4_pcm1_pcn4", _mm4_1d(4, 1, 4, 1, 4)),  # baseline (119us)
    ("bw4_pcm2_pcn4", _mm4_1d(4, 2, 4, 2, 4)),
    ("bw4_pcm4_pcn4", _mm4_1d(4, 2, 4, 4, 4)),
    ("bw4_pcm5_pcn4", _mm4_1d(4, 1, 4, 5, 4)),
    ("bw4_pcm10_pcn4", _mm4_1d(4, 2, 4, 10, 4)),
    ("bw4_pcm20_pcn4", _mm4_1d(4, 2, 4, 20, 4)),
    ("bw8_pcm2_pcn4", _mm4_1d(8, 2, 4, 2, 4)),
    ("bw16_pcm2_pcn4", _mm4_1d(16, 2, 4, 2, 4)),
    ("bw16_pcm4_pcn4", _mm4_1d(16, 2, 4, 4, 4)),
    ("bw4_pcm2_pcn2", _mm4_1d(4, 2, 2, 2, 2)),
    ("bw4_pcm2_pcn1", _mm4_1d(4, 2, 1, 2, 1)),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config", SWEEP_MM4, ids=[s[0] for s in SWEEP_MM4])
def test_mm4_sweep(mesh_device, label, program_config):
    base = DEVICE_MATMULS[4]  # mm4 is appended last (index 4)
    cfg = {**base, "name": f"mm4_{label}", "program_config": program_config}
    _run_device_matmul(mesh_device, cfg)


# mm4 fuse_batch sweep. KEY: with fuse_batch=False, Mt=per-batch (20), batch=16 looped in-kernel ->
# cores=ceil(Mt/pcm)*ceil(Nt/pcn) caps at 20. With fuse_batch=True, the matmul folds batch into M
# (Mt -> B*Mt=320), so per_core_M sets cores ~= ceil(320/pcm). Tests whether that is also CORRECT
# (fuse_batch may broadcast a single in1 weight across the fused batch -> wrong for per-head weights).
def _mm4_1d_fb(in0_block_w, osh, osw, per_core_M, per_core_N, fuse_batch):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID_11x10,
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=fuse_batch,
        fused_activation=None,
        mcast_in0=False,
    )


SWEEP_MM4_FB = [
    ("fbF_pcm96", _mm4_1d_fb(4, 1, 4, 96, 4, False)),  # literal suggestion, fuse off
    ("fbT_pcm96", _mm4_1d_fb(4, 1, 4, 96, 4, True)),  # literal suggestion, fuse on
    ("fbT_pcm3", _mm4_1d_fb(4, 1, 4, 3, 4, True)),  # ~107 cores if cores=ceil(320/pcm)
    ("fbT_pcm4", _mm4_1d_fb(4, 1, 4, 4, 4, True)),  # 320/4=80 cores
    ("fbT_pcm5", _mm4_1d_fb(4, 1, 4, 5, 4, True)),  # 320/5=64
    ("fbT_pcm8", _mm4_1d_fb(4, 1, 4, 8, 4, True)),  # 320/8=40
    ("fbT_pcm16", _mm4_1d_fb(4, 1, 4, 16, 4, True)),  # 320/16=20
    ("fbT_pcm20", _mm4_1d_fb(4, 1, 4, 20, 4, True)),  # 320/20=16
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config", SWEEP_MM4_FB, ids=[s[0] for s in SWEEP_MM4_FB])
def test_mm4_sweep_fb(mesh_device, label, program_config):
    base = DEVICE_MATMULS[4]
    cfg = {**base, "name": f"mm4_{label}", "program_config": program_config, "check_pcc": False}
    _run_device_matmul(mesh_device, cfg)


# mm4 via the OLDER non-mcast MatmulMultiCoreReuseProgramConfig. KEY DIFFERENCE: cores =
# (B*Mt/per_core_M) * (Nt/per_core_N) -- it FOLDS B*Mt (=16*20=320), so batch finally counts toward
# cores (mcast_1d caps at Mt=20). Constraints: 320 % per_core_M == 0, Nt(4) % per_core_N == 0. Max
# core count <=110 is 80 (1280/(pcm*pcn), need pcm*pcn=16 -> pcm4/pcn4, pcm8/pcn2, pcm16/pcn1).
def _mm4_reuse(in0_block_w, osh, osw, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=GRID_11x10,
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


# EXTRA constraint (matmul_device_operation.cpp:1507): per_core_M must ALSO divide per-batch M=20,
# so per_core_M in {1,2,4,5,10,20} (pcm8/16 FATAL). cores=(320/pcm)*(4/pcn); pcm4/pcn4=80 is the max
# valid <=110. Block-width sweep on the 80c winner + a core-count ladder.
# EXTRA constraint: per_core_M must ALSO divide per-batch M=20 -> pcm in {1,2,4,5,10,20} (pcm8/16
# FATAL). 80c (pcm4/pcn4) is the max valid; cores drive perf monotonically (80->27us .. 16->86us);
# smaller in0_block_w wins (bw16 49us, bw8 32us, bw4 28us, bw2 27us). Fine-tune bw1/bw2 + subblock.
SWEEP_MM4_REUSE = [
    ("reuse_bw1_sb2x4", _mm4_reuse(1, 2, 4, 4, 4)),  # 80c, smallest block
    ("reuse_bw2_sb2x4", _mm4_reuse(2, 2, 4, 4, 4)),  # 80c (prev best 26.7us)
    ("reuse_bw2_sb1x4", _mm4_reuse(2, 1, 4, 4, 4)),
    ("reuse_bw2_sb4x2", _mm4_reuse(2, 4, 2, 4, 4)),
    ("reuse_bw2_sb4x1", _mm4_reuse(2, 4, 1, 4, 4)),
    ("reuse_bw4_sb2x4", _mm4_reuse(4, 2, 4, 4, 4)),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config", SWEEP_MM4_REUSE, ids=[s[0] for s in SWEEP_MM4_REUSE])
def test_mm4_reuse(mesh_device, label, program_config):
    base = DEVICE_MATMULS[4]
    cfg = {**base, "name": f"mm4_{label}", "program_config": program_config}
    _run_device_matmul(mesh_device, cfg)


# mm4 memory sweep on the FINAL MultiCoreReuse 80c config: confirm whether in0->L1 (10.5MB SDPA out)
# and out->L1 actually help, now that the op is 80-core-bound (not serialization-bound like MultiCast1D).
SWEEP_MM4_MEM = [
    ("in0DRAM_outL1", DRAM, L1),
    ("in0L1_outL1", L1, L1),  # current locked
    ("in0L1_outDRAM", L1, DRAM),
    ("in0DRAM_outDRAM", DRAM, DRAM),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,act_mc,out_mc", SWEEP_MM4_MEM, ids=[s[0] for s in SWEEP_MM4_MEM])
def test_mm4_sweep_mem(mesh_device, label, act_mc, out_mc):
    base = DEVICE_MATMULS[4]
    cfg = {**base, "name": f"mm4_{label}", "act_mem_config": act_mc, "out_mem_config": out_mc}
    _run_device_matmul(mesh_device, cfg)


# mm2 MultiCoreReuse sweep (same family as mm4). M_t=20, K_t=4, N_t=16, B*Mt=320. cores=(320/pcm)*
# (16/pcn); pcm in {1,2,4,5,10,20}, pcn|16. pcm4/pcn16=80c (max). bw|K_t=4 -> {1,2,4}. _mm4_reuse is
# generic (builds MatmulMultiCoreReuseProgramConfig).
SWEEP_MM2_REUSE = [
    ("pcm4_pcn16_bw2_sb2x4", _mm4_reuse(2, 2, 4, 4, 16)),  # 80c base
    ("pcm4_pcn16_bw1_sb2x4", _mm4_reuse(1, 2, 4, 4, 16)),  # 80c bw1
    ("pcm4_pcn16_bw4_sb2x4", _mm4_reuse(4, 2, 4, 4, 16)),  # 80c bw4
    ("pcm4_pcn16_bw2_sb1x8", _mm4_reuse(2, 1, 8, 4, 16)),  # 80c, fat 1x8
    ("pcm4_pcn16_bw2_sb4x2", _mm4_reuse(2, 4, 2, 4, 16)),  # 80c
    ("pcm4_pcn16_bw2_sb4x1", _mm4_reuse(2, 4, 1, 4, 16)),  # 80c
    ("pcm5_pcn16_bw2_sb1x8", _mm4_reuse(2, 1, 8, 5, 16)),  # 64c
    ("pcm10_pcn8_bw2_sb2x4", _mm4_reuse(2, 2, 4, 10, 8)),  # 64c
    ("pcm10_pcn16_bw2_sb2x4", _mm4_reuse(2, 2, 4, 10, 16)),  # 32c
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,program_config", SWEEP_MM2_REUSE, ids=[s[0] for s in SWEEP_MM2_REUSE])
def test_mm2_reuse(mesh_device, label, program_config):
    base = DEVICE_MATMULS[5]  # mm2 is appended last (index 5)
    cfg = {**base, "name": f"mm2_{label}", "program_config": program_config}
    _run_device_matmul(mesh_device, cfg)


# mm2 memory sweep on the 80c config: out is the 10.5MB tensor here (N=512 bf16), so out-L1 vs DRAM is
# the key lever (mirror of mm4 where in0 was big). in0 is only 2.62MB.
SWEEP_MM2_MEM = [
    ("in0L1_outL1", L1, L1),  # current locked
    ("in0L1_outDRAM", L1, DRAM),
    ("in0DRAM_outL1", DRAM, L1),
    ("in0DRAM_outDRAM", DRAM, DRAM),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("label,act_mc,out_mc", SWEEP_MM2_MEM, ids=[s[0] for s in SWEEP_MM2_MEM])
def test_mm2_sweep_mem(mesh_device, label, act_mc, out_mc):
    base = DEVICE_MATMULS[5]
    cfg = {**base, "name": f"mm2_{label}", "act_mem_config": act_mc, "out_mem_config": out_mc}
    _run_device_matmul(mesh_device, cfg)


if __name__ == "__main__":
    print(format_table([theoretical(mm) for mm in MATMULS]))
