# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b megakernel — the sparse_dsa (MLA) attention half of a GLM-5.3-Flash decode layer as persistent generic_op
programs on the 7a megakernel contract (doc/ttnn/07a-fused-streaming-layer/megakernel_plan.md: BRISC weight streamer
on the bank-adjacent DS readers, TRISC phase compute, NCRISC activation mover, point-to-point hand-off semaphores).

MkMlaA (one program, feeds the existing flash-MLA op):
  P1 mhc_pre                owners O (64) + root    x [1,1,T,16384] fp32 -> hn [1,1,T,4096] bf16, x_out, coef
  P2 q_a | kv_a             readers R               hn -> qk [1,1,T,2048] bf16 (DS, in place)
  P3 RMSNorm_1536(q_a)      readers R               fused into P5's in0 (every reader normalises the 48 q_a tiles)
  P3' RMSNorm_512(kv_a)     latent core L           -> latent [1,1,T,512] bf16
  P4 latent cache write     latent core L           row t -> the paged cache at cur_pos[t] (page table)
  P5 q_b                    readers R               -> q [1,1,T,H*256] bf16 (DS, in place)
  P6 W_uk                   readers R               head h on the reader that owns q_b's head h (grouped DS layout,
                                                    streamed from that reader's bank) -> q_lat [1,T,H,512] (head rows
                                                    scattered straight into the flash-MLA op's layout)
MkMlaB (one program, after the flash-MLA op):
  P7 W_uv                   readers R               o_lat [1,T,H,512] head rows gathered -> o [1,1,T,H*256]
  P8 o_proj                 readers R               o -> a [1,1,T,4096] (DS, in place)
  P9 mhc_post               owners O                a, x_out, coef -> s1 [T,1,4,4096] fp32 (stacked residual)

Hand-offs (semaphore ids): owners+root 0..3 mhc_pre; readers 4 HN (mhc_pre -> P2), 5 QA (P2 -> P5, all readers),
6 Q (P5 -> P6, all readers), 7 O (P7 -> P8, all readers); latent core 5 QA (P2 -> P3'); owners 8 A (P8 -> P9).
TP8 (shard_contract.md): every phase takes the chip's local head range; q_a | kv_a, the norms and the latent cache
stay replicated, o_proj's rows are the local heads' (a partial [T,4096] for Stage 14's fabric all-reduce after P8).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_post as _mpost
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as _mhc
from models.autoports.zai_org_glm_5_3_flash.tt.ds_mm import TILE_BYTES, _tiles_per_txn
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import _rt_args, rect_placement
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MK_READER_REV, reader_assignments

KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
TILE = 32
BF16, BF8, FP32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
SRC_TENSOR, SRC_NORM, SRC_ROWS = 0, 1, 2
DST_TENSOR, DST_ROWS = 0, 1
SKIP = 0xFFFFFFFF
# reader-core CB indices (7a's numbering for the shared ones)
CB_IN0, CB_PART, CB_OUT, CB_RING0 = 40, 41, 42, 43
CB_GIN, CB_INV, CB_X, CB_NW, CB_QN, CB_STG = 34, 35, 36, 37, 33, 30
CB_OUT32 = 39  # the fp32 output tiles of fp32-output reader phases (the DSA projection I1)
CB_RS = 38  # the row-scale tile of RS reader phases (kernels/mla/ra.cpp, rc.cpp: the folded q_b / I3)
PROBE_CB = 31
# latent-core CB indices
LCB_X, LCB_NW, LCB_INV, LCB_OUT, LCB_SCR = 0, 1, 2, 3, 4
# semaphores
SEM_HN, SEM_QA, SEM_Q, SEM_O, SEM_A, SEM_UK = 4, 5, 6, 7, 8, 9
SEM_RDY, SEM_DAT, SEM_OL = 10, 11, 12  # attention tree (owners), attention roots -> W_uv readers
SEM_DATO = 7  # attention tree: O tiles of a streamed partial landed (cumulative per message; owners)
SEM_INVH = 14  # mhc_pre's root -> every reader: the deferred 1/rms(h) tile landed (q_a | kv_a late wait; Stage 14's id)
# attention-phase CB indices on the owners (mhc_post: 0..7)
# (partials: stats CBs APA / APB / ARX / ASEND = m, l; O CBs APAO / APBO / ARXO / ASENDO; ADIAG = diagonal coefficient
# tiles of the FPU rescales, AID = the identity tile; kernels/mla/att_c.cpp)
ACB = dict(AQ=10, AK=11, AMASK=12, AINFO=13, AS=14, AM=15, AP=16, AE=17, APA=18, APB=19, ARX=20, ACOEF=21, ASEND=22,
           AOUT=23, ASCR=24, APAO=25, APBO=26, ARXO=27, ASENDO=28, ADIAG=29, AID=30)  # fmt: skip
ATT_NKMAX = 3  # row tiles per partial (kernels/mla/att_common.hpp NKMAX): ceil(65 / 32) for the 2051-row dense-exact
# range on 32 chunk cores; DSA batches of selected pools hold <= 3 tiles too
ACB1 = {k: v + 22 for k, v in ACB.items()}  # the attention CBs of the one-program half's owners: 32..52
# kernel revision: part of the compile signature so a long-lived process (serve.py) rebuilds after a kernel edit
# rev 63: the decode kernels address the paged pooled-key store (StackBackend._paged_pool_state, Stage 12's host side)
DSA_PAGED_POOLS = True
MLA_REV = 63  # 63: the paged pooled-key store on the sequence-sharded line (GLM53_DSA_PAGED_POOLS=1, doc/ttnn/12-fast-start/paged_pooled_keys.md rev 3: one [32, 128] page per local latent block, its 16 pools at rows 0..15): att_dsa_shard_dm0 I4 reads the top halves of pages ptab_local[t][2u], [2u + 1]; ix_dm1 writes pool q at row q % 16 of page ptab_local[slot][q / 16] (the local page table: an extra accessor + runtime arg MLA_DSA_PT_ARG); att_dsa_shard_dm0 / ix_dm1 changed; 62: the spec verify on the shard WITH token lanes (GLM53_MLA_DSA_BATON=1, dsa/spec_on_shard_design.md section 6): ix_dm1 MLA_DSA_ONE_SLOT_BATON -- the one slot's ring / pooled-key writes in row order across the lane index cores (BATON / DONE semaphores 10 / 11 on the index cores), each lane keeps its select / merge / attention, so the w8 verify keeps plain w8's lane partition; ix_dm1 changed; 61: the speculative verify on the sequence-sharded DSA (Stage 17 x Stage 14, dsa/spec_on_shard_design.md): MLA_DSA_ONE_SLOT with MLA_DSA_SHARD on one index core (att_lanes(t) = 1: t <= 4, or GLM53_MLA_DSA_LANES_AT="8:1" -- the plain step of the same width must run the same lanes: the per-lane chunk partition fixes the merge tie order and the split-K sums), every row scoring the launch slot's LOCAL pooled keys (att_dsa_shard_dm0.cpp); att_dsa_shard_dm0 changed; 60: the sequence-sharded DSA with TOKEN LANES at t > 4 (W: native B=32 / widths 8 / 16, t <= 32; shard_tp8_design.md section 7): per lane the shard coordinator on the lane's index core (local select over its C / L chunk cores, records at the token's x_ag slot, the global merge of its tokens on its merge helpers), the AG producer count t (C / L + 1) + 1 and its consumers = every lane index core, the RS consumers = every lane root, the lane-local mailbox / kept blocks / CB_ASEL (j = t / L < shard_tl(t) = 8 at t > 4; kept blocks / CB_ASEL pages of min(512, keys per core)), o_T from the shard roots, the latent core's batched write on the owner chip; review 2 (tp8b/review.md) items 1 (the merge scratch over the dead phase-A keys in CB_AKEY), 4 (fs_common.hpp PROBE_FLUSH), 5 (fs kernel text hashes probe.hpp), 7 (static / runtime asserts), 8 (no y_ag accessor in ix_dm0); att_dsa_shard_dm0-1 / att_c / ix_dm0-1 / lat_dm1 / fs_common / fs_send changed; 59: shard perf + hardening (shard TP8 review): the global merge on NCHIPS merge-helper owners (ix_dm0 coordinates the radix rounds, helper q holds chip q's list: the same union histograms / T_g / tie order as 58's serial merge), the TP8 q tiles assembled once per chip by ix_dm1 into q_asm (owners read 16 pages per push), x_rs 384 wide (36-page RS round 1), the RS consumers = the roots, opt-in shard probe marks (owner NCRISC / BRISC, index core, stream lanes rows 236..239); att_dsa_shard_dm0-1 / ix_dm0-1 / fs_send changed; 58: the sequence-sharded DSA split-K phase at TP8 on the 2x4 mesh (MLA_DSA_SHARD_LINE = 4: the q-tile assembly from 4 chips' 8-row pieces, the 8-list global merge on a per-core merge-scratch CB, the 2-round reduce-scatter with the round-2 scatter of 8-row slices + the own-then-column combine; ix_dm0 / lat_dm1 / att_dsa_shard_dm0-1 / att_c / fs_send / fs_read changed; the pair path keeps its formats); 57: the sequence-sharded DSA split-K phase (dsa/shard 6043dff8af, TP2, GLM53_DSA_SHARD) merged onto rev 56 (shared kernels att_c / lat_dm1 / ix_dm0-1 / owner_dm0-1 changed: fresh JIT); 56: the DSA-capable speculative verify (Stage 17; GLM53_MLA_ATT_VERIFY with dsa=: MLA_DSA_ONE_SLOT on the index core and the owner readers -- every row is the launch's slot s0, pool q's ring rows at ring tile rows 4 (q - q0) .., kernels/mla/ix_dm1.cpp; t <= 4); 55: the owner NCRISC composite compiles mhc_pre's "dz" (GLM53_MHC_MODE=dz: its NoC headers at global scope, kernels/mla/owner_dm1.cpp); 54: PROBE_ROWS 224 -> 240 (rev 53's per-lane index-core probe rows 220..235 overflowed the probe tensor: card0 u72 hang, probe builds only); 53: review fixes (NKMAX floor without lanes, hist returned, last_mcast recorded, a probe row pair per lane index core, NOC0 assert for the multicast, comments); 52: the I5 coordinators' histogram sum with register accumulators (dsa_common.hpp sum_hist; GLM53_MLA_DSA_HSUM=0 = the old loop); 51: per-round I5 probe marks in ix_dm0 (probe builds only); 50: the DSA I5 broadcast as multicasts onto exact owner rectangles (ix_dm0 MLA_DSA_MCAST: message multicast + SEL_SEM set to the running count; GLM53_MLA_DSA_MCAST, opt-in); 49: GLM53_MLA_DSA_NKMAX (the DSA program's attention batch at t > 4: dense rows and S1; default 3 = rev 48); 48: DSA S1 gather in runs of consecutive selected pools (att_dsa_dm0 MLA_DSA_RUNS: one read of 128 k B per face; GLM53_MLA_DSA_RUNS, default on) + GLM53_MLA_DSA_QFIRST (W_uk before I3, opt-in); 47: one DSA index core per token lane at t > 4 (the lane cores run ix_dm0 / ix_c / ix_dm1 on their tokens; GLM53_MLA_DSA_IX_LANES, default on); 46: row-granular DSA index writes (ix_dm1 MLA_DSA_IX_ROWS: ring rows 0..3 + one pooled-key row per token instead of 12-tile RMWs; GLM53_MLA_DSA_IX_ROWS, default on); 45: index-core probe marks (slots 220 / 221: ix_dm0 / ix_dm1, + ix_dm1 'rows done'); 44: GLM53_MLA_DSA_SB default 1 (card0 u38: SB 4 gives no gain, I4 is DRAM-bound on the 64 MB of replicated pooled keys at 1M); 43: the lane coordinator's cur_pos scratch 64-B aligned (a DRAM read target); 42: GLM53_MLA_DSA_SB default 2 (the owners' DSA score CBs 112 -> 56 KB: 14b's served-stack L1 floor); 41: DSA score batches always take SB CB tiles (the last padded; a short batch broke the CB batch grid); 40: DSA token lanes at t > 4 (one select coordinator core per lane, dsa_coord.cpp; GLM53_MLA_DSA_LANES, default 8); 39: o_T with DSA (att_dsa_dm1.cpp MLA_ATT_OT; GLM53_MLA_DSA_OT, default on); 38: dsa_state slot base s0 (host only: the launch's rows = slots s0.. of a [S, ..] store, dsa_slot_addrs); 37: DSA I4 scoring in batches of GLM53_MLA_DSA_SB (default 4) pooled-key tiles; 36: DSA radix select on compacted candidates (GLM53_MLA_DSA_COMPACT, default on); 35: o_T + NKMAX 8 + bf16 O by default at every t > 4 (card0 t=16 515 -> 479, t=8 413 -> 386 us); 34: o_T at 4 < t < 32 (opt-in GLM53_MLA_OT_SMALL=1; W_uv zeroes rows >= T); 33: o_T + NKMAX 8 + bf16 O partials by default at t = 32 (card0 vtp8 729.3 -> 595.2 us); 32: GLM53_MLA_ATT_NKMAX (row tiles per attention batch, t > 4); 31: GLM53_MLA_ATT_FID_QK (the attention's bf16 matmuls' fidelity); 30: o_T (t = 32: the attention roots write W_uv's token-row in0, GLM53_MLA_OT); 29: gather / scatter one address per row pair, lanes 4 at t = 2..4; 25: t <= 32 (flat residual / per-tile mhc_post, 32-row DSA mailbox); 26: token lanes (t > 4);
# 27: batched latent cache write (t > 4); 28: staging batch of the row gather / scatter (t > 4)
PROBE_ROWS = 240  # owners 0..63, root 64, readers 65 + 2i (streamer) / 66 + 2i (mover), latent 130 / 131, owner NCRISC  # 132 + k, q-norm cores 196 + 2i / 197 + 2i (i < 12), index cores 220 + 2l / 221 + 2l (l < 8 lanes, rev 53)  # rev 54: was 224 - rev 53's per-lane index-core rows (up to 235) wrote past it and hung card0 (u72); keep <= MK_PROBE_LROWS
SEM_QL, SEM_LAT = 13, 14  # one-program half: W_uk readers -> attention owners, latent core -> attention owners
SEM_QR = 15  # one-program half: the q-norm core -> the q_b readers
POST_CB0_ONE = 21  # mhc_post's CBs on the owners of the one-program half (mhc_pre's K role: 0..13, probe 31)
# DSA (kernels/mla/dsa_common.hpp, ix_*.cpp, att_dsa_*.cpp): owner CBs after the attention's 32..52, owner semaphores
# (index core -> owners, I3 readers -> owners, coordinator -> owners), index-core semaphores (I1 readers -> index core,
# group-0 owners -> coordinator); the mailbox / histogram layout of dsa_common.hpp
DCB = dict(AQI=53, AWS=54, APK=55, ASR=56, ASC=57, ASEL=58, AKEY=59)
SEM_SEL, SEM_IXR, SEM_QI = 4, 5, 6
SEM_IX, SEM_H = 4, 5
DSA_TOPK, DSA_KPOOL, DSA_DENSE_MAX = 512, 4, 2051
# THE DENSE WINDOW (GLM53_DSA_DENSE_WINDOW, doc/ttnn/14-production-tp8-megakernel/dsa/dense_window_design.md): the
# replicated [S x DSA_WIN_BLOCKS, 1, 64, 512] bf16 store of rows 0 .. DENSE_MAX - 1 per slot (the cache's paged layout)
DSA_WIN_BLOCKS = -(-DSA_DENSE_MAX // 64)


def _win_defs(shard) -> list:
    """the dense window's defines (the reader / writer / compute of the split-K phase and the latent core)"""
    return (
        [("MLA_DSA_DENSE_WINDOW", "1"), ("MLA_DSA_WIN_BLOCKS", str(DSA_WIN_BLOCKS))]
        if shard is not None and shard.get("window") is not None
        else []
    )


DSA_ROUNDS = 8  # kernels/mla/dsa_common.hpp ROUNDS: 32 / DIGIT_BITS radix rounds (+ the tie round: DSA_MSGS below)
DSA_T_MAX, DSA_MSGS, DSA_MAX_CHUNKS = 4, 9, 64
DSA_MBOX_BYTES = DSA_T_MAX * DSA_MSGS * 16 + DSA_T_MAX * DSA_MAX_CHUNKS * 4
IX = dict(Y=0, C=1, ROW=2, W=3, RING=4, POOL=5, INFO=6, SCR=7, T0=8, CSCR=14)
MCAST_MAX = 16  # rev 50: owner rectangles per I5 coordinator (ix_dm0's runtime args are padded to this)
DSA_POS_BYTES = 128  # kernels/mla/dsa_common.hpp POS_BYTES: a cur_pos page in L1 scratch (<= 32 int32)
# the sequence-sharded split-K variant (kernels/mla/dsa_shard.hpp, tt/dsa_shard.py, tt/tp_dsa_shard.py): the owners' bf16
# payload CBs of a cross-chip partial (O hi / O lo / stats), the reduce-scatter consumer semaphore on the owners, the
# all-gather consumer semaphore on the index core, the all-gather lane's wait semaphore on the lane cores
SCB = {}  # (the first form's bf16 payload CBs 60-62: gone -- the cross-chip partial rides the tree's own fp32 CBs)
SEM_RSC, SEM_AGC, SEM_FSW, SEM_FSA = 9, 6, 0, 1  # (SEM_FSA: the AR lane released the link, on the stream lane)
SEM_BATON, SEM_DONE = 10, 11  # rev 62: the verify's lane baton on the index cores (ix_dm1 MLA_DSA_ONE_SLOT_BATON)
SEM_QASM = 7  # rev 59, TP8: the index core's q assembly (ix_dm1) -> its coordinator (ix_dm0), tokens done
SHARD_T_MAX, SHARD_CAND_MAX, SHARD_REC_BYTES, SHARD_REC_HDR, SHARD_RS_CHUNKS = 32, 512, 16, 16, 3
# rev 60 (W): tokens per token lane at t > 4 (dsa_shard.hpp TL_MAX; t <= 4: 4): the owners' mailbox slots, kept blocks
# and CB_ASEL pages, the lane index core's T_g probe words -- a t-row launch needs att_lanes(t) >= t / shard_tl(t) lanes
SHARD_TL_MAX = 8
# TP8 (the 2x4 mesh; kernels/mla/dsa_shard.hpp LINE = 4): the compact AG chunk (q piece + records) and the RS round-2
# slice chunk, pages per 32-row chunk of their [., 320] bf16 tensors
SHARD_AG8_PPC, SHARD_AG8_QPAGES, SHARD_RS2_PPC = 10, 4, 10
SHARD_RS_PPC = 12  # rev 59: x_rs / y_rs pages per chunk ([., 384] bf16): 3 chunks = 36 pages = the fp32 RS-1 message
SHARD_MERGE_MSGS = 11  # rev 59: dsa_shard.hpp MERGE_MSGS (start + 8 rounds + 2 tie-quota slots per token)
SHARD_MERGE_SCR = 5 * 2048 + 2 * 512 * 4 + 64 * 4  # dsa_shard.hpp MERGE_SCR_BYTES (the helpers' CB_AKEY scratch)
SHARD_PROBE_LANE0 = (
    236  # rev 59: probe rows of the shard stream lanes (fs_send.cpp): 236 + i, i < 4 (inside PROBE_ROWS)
)


def _shard_header_check():
    """the host mirrors of kernels/mla/dsa_shard.hpp are parsed from it and asserted (program lesson: a stale Python
    SYNC_WORDS vs fs_common.hpp tripped the watcher on d0, 2026-10-22)"""
    from models.autoports.zai_org_glm_5_3_flash.tt import shard_plan as SP

    h = SP.HDR
    got = dict(T_MAX=SHARD_T_MAX, REC_BYTES=SHARD_REC_BYTES, REC_HDR=SHARD_REC_HDR, RS_CHUNKS=SHARD_RS_CHUNKS,
               AG8_PPC=SHARD_AG8_PPC, AG8_QPAGES=SHARD_AG8_QPAGES, RS2_PPC=SHARD_RS2_PPC, RS_PPC=SHARD_RS_PPC,
               TL_MAX=SHARD_TL_MAX)  # fmt: skip
    bad = {k: (v, h.get(k)) for k, v in got.items() if h.get(k) != v}
    assert not bad, f"mk_mla.py SHARD_* vs kernels/mla/dsa_shard.hpp: {bad}"
    assert SP.merge_msgs() == SHARD_MERGE_MSGS and SP.merge_scr_bytes() == SHARD_MERGE_SCR, "dsa_shard.hpp merge layout"
    # every probe row a shard kernel flushes stays inside the probe tensor and its stack-layer block (kernels/mk/probe.hpp
    # MK_PROBE_LROWS): an out-of-bounds probe row hung card0 (u72)
    lrows = SP.header_constants_define("mk/probe.hpp", "MK_PROBE_LROWS")
    assert SHARD_PROBE_LANE0 + 4 <= PROBE_ROWS <= lrows, (SHARD_PROBE_LANE0, PROBE_ROWS, lrows)


def _shard_line_defs(shard) -> list:
    """the MLA_DSA_SHARD_LINE define of the 2x4 build (kernels/mla/dsa_shard.hpp LINE: the chips of a mesh row; the
    pair builds without it, LINE = 1)"""
    line = int((shard or {}).get("line", 1))
    assert line in (1, 4), line
    return [("MLA_DSA_SHARD_LINE", str(line))] if line > 1 else []


def stg_slots(t: int) -> int:
    """face rows per staging batch of the readers' row gather / scatter (kernels/mla/ra.cpp STG_SLOTS; 128 B each):
    32 up to t = 4 (those programs unchanged), GLM53_MLA_STG_SLOTS (default 128) above -- the t = 32 W_uv gather was 16
    barrier round trips per K block at 32 (vtp8 probe 2026-10-20: W_uv 180 us at t = 32 vs 7.5 at t = 1)"""
    if t <= DSA_T_MAX:
        return 32
    return max(32, int(os.environ.get("GLM53_MLA_STG_SLOTS", "128")))


def dsa_tmax(t: int) -> int:
    """rows the DSA mailbox of a t-row launch serves (kernels/mla/dsa_common.hpp T_MAX = the MLA_DSA_TMAX define): 4
    up to t = 4 (the t <= 4 programs unchanged: 1.6 KB per owner), 32 above (12.8 KB per owner)"""
    return DSA_T_MAX if t <= DSA_T_MAX else TILE


def _dsa_lanes_at(t: int) -> int:
    """rev 61: GLM53_MLA_DSA_LANES_AT ("8:1,16:2"; default none): a cap on the DSA token lanes of the t-row builds only
    (the reserved coordinators / lane cores stay GLM53_MLA_DSA_LANES'); 1 at t = 8 runs the sharded w8 verify and its
    plain w8 step on one lane (dsa/spec_on_shard_design.md section 3)"""
    for item in filter(None, os.environ.get("GLM53_MLA_DSA_LANES_AT", "").split(",")):
        k, v = (int(x) for x in item.split(":"))
        if k == t:
            if v < 1 or v & (v - 1):
                raise ValueError(f"GLM53_MLA_DSA_LANES_AT: {item}: lanes must be a power of two >= 1")
            return v
    return 1 << 30


def shard_tl(t: int) -> int:
    """rev 60: the lane-local token slots of a sharded t-row build (dsa_shard.hpp TL = mladsa::T_MAX, the MLA_DSA_TMAX
    define of the shard build): 4 up to t = 4 (rev 59's layout), SHARD_TL_MAX above (token lanes: t / L tokens each)"""
    return DSA_T_MAX if t <= DSA_T_MAX else SHARD_TL_MAX


def dsa_wdtype():
    """the replicated indexer projections' stream format (w_x, wq_b / its q_a-norm fold): bf16 (default) or bfp8 with
    GLM53_MLA_DSA_WBF8=1 -- 16.8 MB -> 8.9 MB per chip per sparse_dsa layer (the MLA half itself streams 23.4 MB); the
    selection's numerics are gated vs the fp32 oracle / the select emulation"""
    return BF8 if os.environ.get("GLM53_MLA_DSA_WBF8", "0") == "1" else BF16


def dsa_slot_addrs(st: dict, t: int) -> tuple:
    """(pools, ring) addresses of slot st["s0"] (default 0) of a DSA state (Stage 14 rev 38, the persistent stack: ONE
    store of S slots per sparse_dsa layer, which the prefill writes too, and a launch of t rows on slots s0 .. s0 + t - 1).
    The kernels address slot t of the launch as page t pps of the interleaved DRAM tensor (pools: Tp / 32 x 4 tiles
    per slot, ring: 8), pages round-robin over the banks from bank 0: when s0 pps is a multiple of the bank count,
    slot s0 + t is page t of the same tensor at base + s0 pps / banks aligned pages."""
    s0 = int(st.get("s0", 0))
    slots = int(st.get("slots", int(st["ring"].shape[0])))
    if st.get("one_slot"):  # rev 56: a speculative verify's rows are one user's consecutive positions -> slot s0 only
        t = 1
    if s0 < 0 or s0 + t > slots:
        raise ValueError(f"DSA state: rows {s0}..{s0 + t - 1} outside its {slots} slots")
    ps = int(st.get("pool_slots", slots))  # GLM53_DSA_STORE_SEATS: rows s0 + t >= ps must be idle (stack_state)
    if s0 >= ps:
        raise ValueError(f"DSA state: slot {s0} outside the pooled-key store's {ps} seats")
    out = []
    for key, pps in (("pools", st["tp"] // TILE * (DsaIndexer.DIM // TILE)), ("ring", 2 * DsaIndexer.DIM // TILE)):
        ten = st[key]
        base = int(ten.buffer_address())
        if key == "pools" and st.get("paged"):  # rev 63: block-addressed through the page table (no slot base)
            out.append(base)
            continue
        if s0:
            dram = ten.memory_config().buffer_type == ttnn.BufferType.DRAM
            if ten.dtype != BF16 or ten.layout != ttnn.TILE_LAYOUT or not dram:
                raise ValueError(f"DSA state {key}: bf16 TILE DRAM interleaved expected")
            banks = int(ttnn.get_memory_view(ten.device(), ttnn.BufferType.DRAM).num_banks)
            if s0 * pps % banks:
                raise ValueError(f"DSA state {key}: slot {s0} starts at page {s0 * pps}, not a multiple of {banks}")
            base += s0 * pps // banks * TILE * TILE * 2
        out.append(base)
    return tuple(out)


def dsa_mbox_bytes(t: int, shard: bool = False) -> int:
    """the owners' mailbox (kernels/mla/dsa_common.hpp MBOX_BYTES; with the shard variant dsa_shard.hpp MBOX_SHARD_BYTES:
    (quota, offset) pairs in the tie message, the global message and the global quotas per token)"""
    tm = shard_tl(t) if shard else dsa_tmax(t)  # rev 60: the shard mailbox holds the lane-local tokens
    if shard:  # (rev 59: + the merge helpers' messages, dsa_shard.hpp MERGE_OFF / MERGE_MSGS)
        return (tm * DSA_MSGS * 16 + tm * DSA_MAX_CHUNKS * 4 * 2 + tm * 16 + tm * DSA_MAX_CHUNKS * 4
                + tm * SHARD_MERGE_MSGS * 16)  # fmt: skip
    return tm * DSA_MSGS * 16 + tm * DSA_MAX_CHUNKS * 4


def shard_coord_words(chunks: int, C: int) -> int:
    """uint32 words of an index core's histogram / scratch tensor under the shard variant (ix_dm0.cpp; chunks = the
    chunk cores it coordinates: the lane's with token lanes): the chunk cores' histogram slots, at least MAX_CHUNKS words
    (rev 59: the merge helpers' slots 0 .. C - 1 and helper `rank`'s per-core quotas in words 0..63 too), the 64-B
    aligned count header (+ the T_g probe words 8 + j), the (quota, offset) pairs. (rev 58 kept the serial merge's
    candidate pages / keys in a 45-KB CB on this core; rev 59's helpers hold them in their CB_AKEY)"""
    return max(chunks * 16, DSA_MAX_CHUNKS) + 32 + DSA_MAX_CHUNKS * 2


_SRC = {}


def mla_src() -> str:
    """hash of every kernel source the MLA programs compile (kernels/{mla,mhc_pre,mhc_post,mk,fabric_ar}): the MLA_SRC
    define, so a long-lived server rebuilds a FILE_PATH kernel exactly when a source changes (MLA_REV alone needed a
    manual bump), keyed by the files' mtimes"""
    import glob
    import hashlib

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
    files = sorted(
        g for d in ("mla", "mhc_pre", "mhc_post", "mk", "fabric_ar") for g in glob.glob(os.path.join(base, d, "*"))
    )
    key = tuple((f, os.path.getmtime(f)) for f in files)
    if _SRC.get("key") != key:
        h = hashlib.sha1()
        for f in files:
            with open(f, "rb") as fh:
                h.update(fh.read())
        _SRC.update(key=key, value=h.hexdigest()[:12])
    return _SRC["value"]


def _a64(n: int) -> int:
    """bytes a NoC page read of an n-byte DRAM row-major page lands: the page rounded up to the 64-B DRAM alignment (a
    132-B page-table row reads 192 B; a scratch CB sized to the page overflows)"""
    return -(-max(n, 1) // 64) * 64


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _host0(t) -> torch.Tensor:
    """host copy of a device tensor; on a multi-chip mesh device 0's (the replicated tensors are equal everywhere)"""
    dev = t.device()
    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
        t = ttnn.get_device_tensors(t)[0]
    return ttnn.to_torch(t).float()


def _upload(device, w, dtype):
    """host weight -> TILE tensor in DRAM: a torch tensor = the same on every chip (replicated); a list = rank r's block
    on mesh device r (ShardTensorToMesh on dim 0: one buffer address on every chip, so ONE captured program serves every
    chip of a TP mesh, tt/tp_mla.py)"""
    if isinstance(w, (list, tuple)):
        if len(w) > 1:
            return ttnn.from_torch(torch.cat([b.contiguous() for b in w], dim=0), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=DRAM,
                                   mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))  # fmt: skip
        w = w[0]
    return ttnn.from_torch(w.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def _norm_weight(device, w: torch.Tensor):
    """[n] -> fp32 TILE [32, n] DRAM with row 0 = w (page c row 0 = w[32c : 32c + 32]: the SFPU row broadcast)"""
    n = w.numel()
    t = torch.zeros(TILE, n)
    t[0] = w.float().reshape(-1)
    return ttnn.from_torch(t, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


class GroupedWeight:
    """A per-head weight W [H, KG*32, NG*32] (W_uk: [64, 256, 512], W_uv: [64, 512, 256]) re-laid out once per layer as
    the virtual DS matrix of a grouped reader phase: reader r (= bank * readers_per_bank + index) owns virtual columns
    [r NR, (r + 1) NR) and streams its sub-matmuls' [KG, NR] blocks from ITS bank (width-sharded over the banks, so
    reader_w.cpp streams it like any DS weight). Whole heads per reader when H * NG / R >= NG (NH heads, NR = NG),
    else NG / NR readers per head (NH = 1). Per reader: h0 (first head) and n0 (first output column tile in the
    head-major output space h * NG + c). `w` may be a list of per-rank weights (TP mesh: rank r's heads on chip r)."""

    def __init__(self, device, w, *, readers_per_bank: int, role: str, dtype=BF16,
                 math_fidelity=ttnn.MathFidelity.HiFi4):  # fmt: skip
        ws = list(w) if isinstance(w, (list, tuple)) else [w]
        h, k, n = (int(v) for v in ws[0].shape)
        banks = device.dram_grid_size().x
        r_all = banks * readers_per_bank
        kg, ng = k // TILE, n // TILE
        cols = h * ng // r_all
        if h * ng % r_all:
            raise ValueError(f"{role}: {h} heads x {ng} tiles over {r_all} readers")
        if cols >= ng:
            if cols % ng:
                raise ValueError(f"{role}: {cols} columns per reader, head width {ng}")
            nh, nr = cols // ng, ng
            self.head_of = lambda r, j: (r * nh + j, 0)
        else:
            if ng % cols:
                raise ValueError(f"{role}: head width {ng} not a multiple of {cols} columns per reader")
            nh, nr = 1, cols
            per = ng // cols
            self.head_of = lambda r, j: (r // per, (r % per) * cols)
        wvs = []
        for w_ in ws:
            wv = torch.zeros(nh * k, r_all * nr * TILE, dtype=torch.bfloat16)
            wb = w_.to(torch.bfloat16)
            for r in range(r_all):
                for j in range(nh):
                    hh, c0 = self.head_of(r, j)
                    wv[j * k : (j + 1) * k, r * nr * TILE : (r + 1) * nr * TILE] = wb[
                        hh, :, c0 * TILE : (c0 + nr) * TILE
                    ]
            wvs.append(wv)
        t = _upload(device, wvs, dtype)
        shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (nh * k, readers_per_bank * nr * TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.weight = ttnn.to_memory_config(
            t, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)
        )
        ttnn.deallocate(t)
        self.role, self.k, self.n_pad = role, nh * k, r_all * nr * TILE
        self.heads, self.nh, self.kg, self.ng, self.nr = h, nh, kg, ng, nr
        self.readers_per_bank, self.math_fidelity = readers_per_bank, math_fidelity
        self.host = w  # for the record / CPU checks

    def reader_geo(self, bank: int, idx: int) -> tuple:
        """(h0, n0) of reader (bank, idx)"""
        r = bank * self.readers_per_bank + idx
        h0, c0 = self.head_of(r, 0)
        return h0, h0 * self.ng + c0


class DSWeight:
    """A dense weight W [K, N] width-sharded over the DRAM banks in DSLinear's layout (N zero-padded to whole tiles per
    bank and reader), streamed in place by a reader phase; no matmul program of its own (the DSA projections, the folded
    q_b). `w` may be a list of per-rank weights (TP mesh: rank r's block on chip r)."""

    def __init__(self, device, w, *, role: str, readers: int = 1, dtype=BF16,
                 math_fidelity=ttnn.MathFidelity.HiFi4):  # fmt: skip
        ws = list(w) if isinstance(w, (list, tuple)) else [w]
        k, n = (int(v) for v in ws[0].shape)
        banks = device.dram_grid_size().x
        n_pad = -(-n // (TILE * banks * readers)) * TILE * banks * readers
        wps = []
        for w_ in ws:
            wp = torch.zeros(k, n_pad, dtype=torch.bfloat16)
            wp[:, :n] = w_.to(torch.bfloat16)
            wps.append(wp)
        t = _upload(device, wps, dtype)
        shard = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (k, n_pad // banks),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.weight = ttnn.to_memory_config(
            t, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)
        )
        ttnn.deallocate(t)
        self.role, self.k, self.n, self.n_pad = role, k, n, n_pad
        self.ckc = type("ckc", (), {"math_fidelity": math_fidelity})()


class DsaIndexer:
    """The DSA indexer of one MLA/DSA layer for the in-program phases (5d m4_phase_spec I1..I5, kernels/mla/
    dsa_common.hpp): w_x = [wk | index_kpool_compress_gate | weights_proj] [4096, 288] (I1: hn -> y fp32), wq_b [1536,
    4096] (I3: q_r -> q_idx rows [1, T, 32, 128]), the index core's 13 fp32 constant tiles (LayerNorm_128 weight / bias in
    row 0, ape rows 0..3, the ring-row mask: rows 4..31 = -1e30). Replicated per chip under TP (shard_contract.md)."""

    HEADS, DIM = 32, 128

    def __init__(self, device, source, layer: int):
        h = lambda n: source.host(layer, f"self_attn.indexer.{n}").float()
        self.host = dict(wk=h("wk.weight"), gate=h("index_kpool_compress_gate"), wp=h("weights_proj.weight"),
                         wq_b=h("wq_b.weight"), kn_w=h("k_norm.weight").reshape(-1),
                         kn_b=h("k_norm.bias").reshape(-1), ape=h("index_kpool_compress_ape").reshape(4, -1))  # fmt: skip
        hw = self.host
        wdt = dsa_wdtype()
        self.w_x = DSWeight(device, torch.cat([hw["wk"], hw["gate"], hw["wp"]], dim=1), role="dsa_wx", dtype=wdt)
        self.wq_b = DSWeight(device, hw["wq_b"], role="dsa_wqb", dtype=wdt)
        c = torch.zeros(TILE, 13 * TILE)
        c[0, 0:128] = hw["kn_w"]
        c[0, 128:256] = hw["kn_b"]
        c[0:4, 256:384] = hw["ape"]
        c[4:, 384:416] = -1.0e30
        self.const = ttnn.from_torch(c.reshape(1, 1, TILE, -1), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=DRAM)  # fmt: skip

    @staticmethod
    def pool_rows(page_tokens: int) -> int:
        return -(-(page_tokens // DSA_KPOOL) // TILE) * TILE

    def state(self, device, slots: int, page_tokens: int, pools=None, ring=None) -> dict:
        """per-slot DSA state: pools [S, 1, Tp, 128] bf16 (pooled-key row q = pool q), ring [S, 1, 32, 256] bf16 (rows 0..3
        = [k | g] of the current pool's tokens); zero or the given host tensors"""
        tp = self.pool_rows(page_tokens)
        put = lambda v: ttnn.from_torch(v.to(torch.bfloat16), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device,
                                        memory_config=DRAM)  # fmt: skip
        return dict(pools=put(pools if pools is not None else torch.zeros(slots, 1, tp, self.DIM)),
                    ring=put(ring if ring is not None else torch.zeros(slots, 1, TILE, 2 * self.DIM)), tp=tp)  # fmt: skip

    def stack_state(self, device, slots: int, page_tokens: int, pool_slots: int | None = None) -> dict:
        """the persistent stack's per-layer DSA state (rev 38), zeroed on the device: pools as Stage 16's pooled-key store
        [S Tp / 32, 1, 32, 128] bf16 (the same pages as [S, 1, Tp, 128]: slot s = pages s Tp / 32 x 4 ..; the prefill's
        paged_fill_cache writes it through its slot-major pool table), ring [S, 1, 32, 256]; a launch picks its slots
        with s0 (dsa_slot_addrs). pool_slots (GLM53_DSA_STORE_SEATS): the pools hold only the first pool_slots <= S slots
        (the seats); rows past them must stay idle (cur_pos < 0: ix_dm1 / att_dsa(_shard)_dm0 never address their
        pools), the ring keeps all S rows"""
        tp = self.pool_rows(page_tokens)
        ps = slots if pool_slots is None else int(pool_slots)
        assert 0 < ps <= slots, (ps, slots)
        z = lambda shape: ttnn.zeros(
            shape, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # noqa: E731  # fmt: skip
        return dict(pools=z((ps * tp // TILE, 1, TILE, self.DIM)), ring=z((slots, 1, TILE, 2 * self.DIM)), tp=tp,
                    slots=slots, pool_slots=ps, s0=0)  # fmt: skip


@dataclass
class RPhase:
    """One reader phase (kernels/mla/mla_common.hpp)."""

    w: object  # DSLinear (dense) or GroupedWeight
    kb: int
    src: int = SRC_TENSOR
    dst: int = DST_TENSOR
    wait: tuple | None = None  # (semaphore id, producer count)
    signal: tuple | None = None  # (semaphore id, [(x, y) consumer NoC coordinates])
    fid: object = None  # MathFidelity (None: the DSLinear's / GroupedWeight's)
    rows_heads: int = 0  # SRC_ROWS / DST_ROWS: heads of the [1, T, H, D] rows tensor
    norm: tuple | None = None  # SRC_NORM: (weight tensor (fp32 row 0), eps, KX tiles)
    ring_kb: int = 0  # this phase's ring budget (KB); 0 = MlaReaders' default for the format
    out_fp32: bool = False  # fp32 output tiles (CB_OUT32) instead of bf16
    rs: object = None  # row-scale tensor (1 fp32 tile, column 0 = per-token scale) applied to the output rows
    late: tuple | None = (
        None  # (semaphore id, producer count) of the row-scale tile, waited for before the first output
    )


class HeadShard:
    """TP over heads (shard_contract.md): the per-chip view of an OptimizedMLA for rank r of tp — heads [r H_r, (r + 1)
    H_r) with q_b's columns, W_uk / W_uv and o_proj's rows of those heads (re-packed from the cached values: tile-aligned
    slices, lossless); q_a | kv_a, the norms and the latent cache replicated. Exposes the attributes the MLA
    megakernel programs read, so `MkMla(device, mhc_args, HeadShard(attn, r, 8))` is chip r's TP8 program (its o_proj
    output a = the partial for the fabric all-reduce)."""

    def __init__(self, attn, r: int, tp: int):
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear

        dev = attn.device
        hh = attn.h
        if hh % tp:
            raise ValueError(f"{hh} heads over TP {tp}")
        hr = hh // tp
        h0 = r * hr
        self.device, self.ocfg, self.fcfg, self.rank, self.tp = dev, attn.ocfg, attn.fcfg, r, tp
        self.h, self.q_lora, self.kv_lora, self.dqk, self.dv = hr, attn.q_lora, attn.kv_lora, attn.dqk, attn.dv
        self.eps, self.scale, self.block_size = attn.eps, attn.scale, attn.block_size
        self.ds_qa_kva, self.qa_norm, self.kva_norm = attn.ds_qa_kva, attn.qa_norm, attn.kva_norm
        host = lambda t: ttnn.to_torch(t).float()
        put = lambda w, dt: ttnn.from_torch(w.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                            memory_config=DRAM)  # fmt: skip
        qb = host(attn.ds_qb.weight).reshape(attn.q_lora, -1)[:, : attn.ds_qb.n]
        qb = qb[:, h0 * attn.dqk : (h0 + hr) * attn.dqk]
        wo = host(attn.ds_o.weight).reshape(hh * attn.dv, -1)[h0 * attn.dv : (h0 + hr) * attn.dv, : attn.ds_o.n]
        self.ds_qb = DSLinear(dev, put(qb, attn.ds_qb.weight.dtype), role="mla_qb", cfg=attn.ocfg, ckc=attn.ds_qb.ckc,
                              dtype=BF16)  # fmt: skip
        self.ds_o = DSLinear(dev, put(wo, attn.ds_o.weight.dtype), role="mla_o", cfg=attn.ocfg, ckc=attn.ds_o.ckc,
                             dtype=BF16)  # fmt: skip
        self.w_uk = put(host(attn.w_uk).reshape(hh, attn.dqk, attn.kv_lora)[h0 : h0 + hr], BF16)
        self.w_uv = put(host(attn.w_uv).reshape(hh, attn.kv_lora, attn.dv)[h0 : h0 + hr], BF16)


class MlaReaders:
    """The reader cores' part of an MLA program: reader_w.cpp (BRISC, the dense weight streamer, unchanged) + ra.cpp
    (NCRISC) + rc.cpp (TRISC) with one phase table."""

    def __init__(self, device, phases: list, *, readers_per_bank: int, placement_name: str, lookahead: bool = False):
        """lookahead: the weight streamer kernels/mla/rw.cpp (the next phase's ring fills while the current one waits)
        instead of 7a's kernels/mk/reader_w.cpp (same tables, same layout)."""
        self.device, self.phases = device, phases
        self.lookahead = lookahead
        self.banks = device.dram_grid_size().x
        self.rpb = readers_per_bank
        self.assign = mla_reader_assignments(device, readers_per_bank, placement_name)
        self.cores = merged_ranges([(c.x, c.y) for c, _, _ in self.assign])
        self.geo, rings = [], {}
        for p in phases:
            w = p.w
            if isinstance(w, GroupedWeight):
                assert w.readers_per_bank == self.rpb, (w.role, w.readers_per_bank, self.rpb)
                nh, kg, ng, nr, rp = w.nh, w.kg, w.ng, w.nr, self.rpb
                shard_w = self.rpb * nr
                fid = p.fid or w.math_fidelity
            else:  # DSLinear: dense, read in place
                kt = w.k // TILE
                shard_w = w.n_pad // TILE // self.banks
                rp = max(r for r in range(1, self.rpb + 1) if shard_w % r == 0)
                nr = shard_w // rp
                nh, kg, ng = 1, kt, nr
                fid = p.fid or w.ckc.math_fidelity
            if kg % p.kb:
                raise ValueError(f"{w.role}: K block {p.kb} does not divide {kg}")
            if p.src == SRC_NORM:
                kg = p.norm[2]
                assert nh == 1 and kg == w.k // TILE, (w.role, kg)
            tb = TILE_BYTES[w.weight.dtype]
            key = (w.weight.dtype, p.kb * nr)
            if key not in rings:
                rings[key] = CB_RING0 + len(rings)
            hpt = -(-p.rows_heads // TILE) if p.rows_heads else 0
            self.geo.append(dict(nb=nh * kg // p.kb, kb=p.kb, nr=nr, nh=nh, kg=kg, ng=ng, tb=tb, rp=rp,
                                 tpt=_tiles_per_txn(nr, tb), shard_w=shard_w, ring=rings[key], fid=fid, hpt=hpt,
                                 subw=max(s for s in (4, 2, 1) if nr % s == 0)))  # fmt: skip
        self.rings = {cb: (dt, block) for (dt, block), cb in rings.items()}
        default_kb = {BF8: int(os.environ.get("GLM53_MLA_RING8_KB", "280")),
                      BF16: int(os.environ.get("GLM53_MLA_RING16_KB", "192"))}  # fmt: skip
        self.ring_slots = {}
        for p, g in zip(phases, self.geo):
            dt, block = self.rings[g["ring"]]
            budget = p.ring_kb or default_kb.get(dt, 192)
            slots = max(2, budget * 1024 // (block * TILE_BYTES[dt]))
            self.ring_slots[g["ring"]] = max(self.ring_slots.get(g["ring"], 2), slots)
        kbs = [g["kb"] for p, g in zip(phases, self.geo) if p.src != SRC_ROWS]
        self.in0_cap = 2 * max(kbs) if kbs else 2
        for p, g in zip(phases, self.geo):
            if p.src != SRC_ROWS and self.in0_cap % g["kb"]:
                raise ValueError(f"{p.w.role}: K block {g['kb']} does not divide the in0 CB ({self.in0_cap} tiles)")
        gins = [g["kb"] for p, g in zip(phases, self.geo) if p.src == SRC_ROWS]
        self.gin_cap = 2 * max(gins) if gins else 0
        norms = [(p, g) for p, g in zip(phases, self.geo) if p.src == SRC_NORM]
        assert len(norms) <= 1, "one SRC_NORM phase per program (CB_X / CB_NW sizing)"
        self.norm = norms[0] if norms else None
        self.nr_max = max(g["nr"] for g in self.geo)
        f32 = [g["nr"] for p, g in zip(phases, self.geo) if p.out_fp32]
        self.out32_cap = max(f32) if f32 else 0
        self.zero_mask = 0  # rev 34: bit p = phase p's pulls zero rows >= T (set by MkMla.readers for o_T at t < 32)

    def noc_coords(self):
        return [(int(w.x), int(w.y)) for w in (self.device.worker_core_from_logical_core(c) for c, _, _ in self.assign)]

    def l1_bytes(self) -> int:
        rings = sum(self.ring_slots[cb] * block * TILE_BYTES[dt] for cb, (dt, block) in self.rings.items())
        extra = 0
        if self.norm is not None:
            g = self.norm[1]
            extra = g["kg"] * 2048 + 2 * g["kb"] * (4096 + 2048) + 4096
        return (rings + self.in0_cap * 2048 + self.gin_cap * 2048 + self.nr_max * (4096 + 2048) + extra
                + self.out32_cap * 4096)  # fmt: skip

    def parts(self, xs: list, ys: list, *, t: int, mk_probe: tuple | None = None):
        """(kernels, cbs) of the reader cores for per-phase inputs xs / outputs ys (tensors)."""
        nph = len(self.phases)
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        pr = lambda i, off: [mk_probe[0], mk_probe[1] + 2 * i + off] if mk_probe is not None else []
        # weight streamer (kernels/mk/reader_w.cpp, unchanged)
        w_ct = [nph]
        for g in self.geo:
            w_ct += [g["nb"], g["kb"], g["nr"], g["tb"], g["tpt"], g["shard_w"], g["ring"], self.ring_slots[g["ring"]],
                     g["rp"]]  # fmt: skip
        w_rt = {}
        for i, (c, bank, idx) in enumerate(self.assign):
            w_rt[(c.x, c.y)] = [bank, idx, bank & 3] + [p.w.weight.buffer_address() for p in self.phases] + pr(i, 0)
        # activation mover
        a_ct = [nph, CB_IN0, CB_X, CB_NW, CB_GIN, self.in0_cap, max(self.gin_cap, 1), CB_STG]
        tail, offs = [], []
        base = 8 + 21 * nph
        for p, x, y in zip(self.phases, xs, ys):
            xo = base + len(tail)
            tail += acc(x)
            yo = base + len(tail)
            tail += acc(y)
            no = ro = 0
            if p.src == SRC_NORM:
                no = base + len(tail)
                tail += acc(p.norm[0])
            if p.rs is not None:
                ro = base + len(tail)
                tail += acc(p.rs)
            offs.append((xo, yo, no, ro))
        for p, g, (xo, yo, no, ro) in zip(self.phases, self.geo, offs):
            ws, wc = p.wait if p.wait else (0, 0)
            ss, cons = p.signal if p.signal else (0, [])
            ls, lc = p.late if p.late else (0, 0)
            a_ct += [p.src, p.dst, g["nb"], g["kb"], g["nr"], g["nh"], g["kg"], g["ng"], ws, wc, ss,
                     1 if cons is None else len(cons), xo, yo, CB_OUT32 if p.out_fp32 else CB_OUT, g["hpt"], no,
                     int(p.rs is not None), ls, lc, ro]  # fmt: skip
        a_ct += tail
        n_rt = 8 * nph
        cons_off, flat = [], []
        self_sig = {}  # phase -> runtime-arg index of the reader's own (x, y) (signal = (sem, None): the reader itself)
        for i, p in enumerate(self.phases):
            cons = p.signal[1] if p.signal else []
            cons_off.append(n_rt + len(flat))
            if p.signal and cons is None:
                self_sig[i] = n_rt + len(flat)
                flat += [0, 0]
            else:
                flat += [v for xy in cons for v in xy]
        a_rt, c_rt = {}, {}
        for i, (c, bank, idx) in enumerate(self.assign):
            args, comp = [], []
            for p, g, x, y, co in zip(self.phases, self.geo, xs, ys, cons_off):
                if isinstance(p.w, GroupedWeight):
                    h0, n0 = p.w.reader_geo(bank, idx)
                else:
                    h0, n0 = 0, (bank * g["shard_w"] + idx * g["nr"] if idx < g["rp"] else SKIP)
                nw = p.norm[0].buffer_address() if p.src == SRC_NORM else 0
                args += [x.buffer_address(), y.buffer_address(), n0, h0, nw, t, co,
                         p.rs.buffer_address() if p.rs is not None else 0]  # fmt: skip
                comp.append(int(n0 != SKIP))
            mine = list(flat)
            if self_sig:
                w = self.device.worker_core_from_logical_core(c)
                for off in self_sig.values():
                    mine[off - n_rt : off - n_rt + 2] = [int(w.x), int(w.y)]
            a_rt[(c.x, c.y)] = args + mine + pr(i, 1)
            c_rt[(c.x, c.y)] = comp
        # compute
        c_ct = [nph, CB_IN0, CB_PART, CB_X, CB_NW, CB_INV, CB_GIN, self.nr_max, CB_QN, CB_OUT32, self.out32_cap, CB_RS]
        for p, g in zip(self.phases, self.geo):
            eps, inv_n = (p.norm[1], 1.0 / (p.norm[2] * TILE)) if p.src == SRC_NORM else (0.0, 0.0)
            c_ct += [p.src, g["nb"], g["kb"], g["nr"], g["nh"], g["kg"], g["subw"], g["ring"], int(g["fid"].value),
                     CB_OUT32 if p.out_fp32 else CB_OUT, f32_bits(inv_n), f32_bits(eps), int(p.rs is not None)]  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (CB_PART, CB_INV, CB_NW, CB_RS):
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        rev = [("MLA_REV", str(MLA_REV)), ("MLA_SRC", "0x" + mla_src()[:8]), ("MK_READER_REV", str(MK_READER_REV))]
        pw = pa = []
        if mk_probe is not None:
            pw = [("MK_PROBE_ARG", str(3 + nph)), ("MK_PROBE_CB", str(PROBE_CB))]
            pa = [("MK_PROBE_ARG", str(n_rt + len(flat))), ("MK_PROBE_CB", str(PROBE_CB))]
        if self.gin_cap:
            pa = list(pa) + [("MLA_GIN_ZERO", "1")]
        if self.zero_mask:  # rev 34: phases whose token-row input has unwritten rows >= T (ra.cpp zero_rows_ge)
            pa = list(pa) + [("MLA_PULL_ZMASK", str(self.zero_mask))]
        if os.environ.get("GLM53_MLA_ROW_PULL", "0") == "1":  # in0 pulls of the T valid rows only (ra.cpp pull_block)
            pa = list(pa) + [("MLA_ROW_PULL", "1")]
        stg = stg_slots(t)
        if stg != 32:  # the row gather / scatter batch (ra.cpp STG_SLOTS)
            pa = list(pa) + [("MLA_STG_SLOTS", str(stg))]
        # Stage 14's knob: the weight stream starts late so mhc_pre's x / fn reads go first (default 10 us for the MLA
        # programs: mhc_pre ends ~20 us earlier, measured 5 / 10 / 15 / 20 us -> 10 best; "0" disables)
        w_delay = float(os.environ.get("GLM53_MK_W_DELAY_US", "10"))
        if w_delay > 0:
            pw = list(pw) + [("MK_W_DELAY_CYCLES", str(int(w_delay * 1350)))]
        kd = lambda src, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores, compile_time_args=ct, runtime_args=_rt_args(rt), defines=list(defs) + rev,
            config=cfg,
        )  # fmt: skip
        streamer = "mla/rw.cpp" if self.lookahead else "mk/reader_w.cpp"
        kernels = [
            kd(streamer, w_ct, w_rt, pw, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            kd("mla/ra.cpp", a_ct, a_rt, pa, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
            kd("mla/rc.cpp", c_ct, c_rt, [], cc),
        ]
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(
            total_size=page * n, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # fmt: skip
        cbs = [cb(CB_IN0, BF16, 2048, self.in0_cap), cb(CB_PART, FP32, 4096, self.nr_max),
               cb(CB_OUT, BF16, 2048, self.nr_max), cb(CB_STG, ttnn.uint32, stg * 128 + 64, 1)]  # fmt: skip
        if self.gin_cap:
            cbs.append(cb(CB_GIN, BF16, 2048, self.gin_cap))
        if self.out32_cap:
            cbs.append(cb(CB_OUT32, FP32, 4096, self.out32_cap))
        if any(p.rs is not None for p in self.phases):
            cbs.append(cb(CB_RS, FP32, 4096, 1))
        if self.norm is not None:
            g = self.norm[1]
            cbs += [cb(CB_X, BF16, 2048, g["kg"]), cb(CB_NW, FP32, 4096, 2 * g["kb"]), cb(CB_INV, FP32, 4096, 1),
                    cb(CB_QN, BF16, 2048, 2 * g["kb"])]  # fmt: skip
        for ring, (dt, block) in self.rings.items():
            cbs.append(cb(ring, dt, TILE_BYTES[dt], self.ring_slots[ring] * block))
        if mk_probe is not None:
            cbs.append(cb(PROBE_CB, ttnn.uint32, 128, 1))
        return kernels, cbs


def compact_leaders(cells: list) -> list:
    """mhc_pre's K order (core k = 8 cx + cy; the 8 leaders cy = 0 run their own kernel group) over 64 owner cells with
    the fewest core ranges: leaders = the 8 contiguous cells of one column (or the last 8 cells) that minimise the leader
    + member range count (merged_ranges), members = the other 56 in placement order"""
    xy = [(int(c.x), int(c.y)) for c in cells]
    nr = lambda sub: len(list(merged_ranges(sub).ranges()))
    cands = [list(range(56, 64))]
    for x in sorted({p[0] for p in xy}):
        col = sorted((p[1], i) for i, p in enumerate(xy) if p[0] == x)
        for j in range(len(col) - 7):
            run = col[j : j + 8]
            if run[-1][0] - run[0][0] == 7:
                cands.append([i for _, i in run])
    best = None
    for lead in cands:
        ls = set(lead)
        mem = [i for i in range(64) if i not in ls]
        cost = nr([xy[i] for i in lead]) + nr([xy[i] for i in mem])
        if best is None or cost < best[0]:
            best = (cost, lead, mem)
    _, lead, mem = best
    return [cells[lead[q // 8]] if q % 8 == 0 else cells[mem[7 * (q // 8) + q % 8 - 1]] for q in range(64)]


def mla_reader_assignments(device, rpb: int, placement_name: str) -> list:
    """[(logical core, bank, index in bank)] of the MLA readers. "SPREAD": ttnn's nearest reader per bank (MESH_ROW_PLUS_X,
    1 per bank), then per bank, round robin, the nearest free core whose row no reader of that bank uses yet (rows
    reused only when none is left) — cycle probes: a bank's 3 ttnn-placed readers (one row next to the bank) finish
    o_proj 50-80 us after its 4th reader placed elsewhere (shared NoC row links). Anything else: 7a's
    reader_assignments (ttnn's placement + nearest free cores beyond 3 per bank)."""
    if not placement_name.startswith("SPREAD"):
        return reader_assignments(device, rpb, placement_name)
    base = reader_assignments(device, 1, "MESH_ROW_PLUS_X")
    grid = device.compute_with_storage_grid_size()
    taken = {(int(c.x), int(c.y)) for c, _, _ in base}
    free = [(x, y) for x in range(int(grid.x)) for y in range(int(grid.y)) if (x, y) not in taken]
    mine = {int(b): [(int(c.x), int(c.y))] for c, b, _ in base}
    out = [(c, int(b), 0) for c, b, _ in base]
    for idx in range(1, rpb):
        for b in sorted(mine):
            ax, ay = mine[b][0]
            rows = {y for _, y in mine[b]}
            cost = lambda xy: abs(xy[0] - ax) + abs(xy[1] - ay) + (8 if xy[1] in rows else 0)
            best = min(free, key=cost)
            free.remove(best)
            mine[b].append(best)
            out.append((ttnn.CoreCoord(*best), b, idx))
    return out


def _free_cores(device, taken: set) -> list:
    grid = device.compute_with_storage_grid_size()
    return [(x, y) for x in range(int(grid.x)) for y in range(int(grid.y)) if (x, y) not in taken]


def _probe_tensor(device):
    return ttnn.from_torch(torch.zeros(PROBE_ROWS, 16, dtype=torch.int32), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)  # fmt: skip


class _MlaBase:
    def __init__(self, device, attn, *, readers_per_bank: int | None, probe: bool):
        self.device, self.attn = device, attn
        self.rpb = int(readers_per_bank or os.environ.get("GLM53_MLA_RPB", "4"))
        # the DS readers' placement (ttnn DramReaderPlacement: MESH_ROW_PLUS_X (the decoder's), ROW_PLUS_X, NOC_HOPS, or
        # SPREAD); GLM53_MLA_PLACEMENT overrides, default ROW_PLUS_X: the fastest in the placement sweep (T = 1, single /
        # steady / in-program us: ROW_PLUS_X 557.8 / 523.3 / 457.2, MESH_ROW_PLUS_X 564.3 / 524.2 / 462.1, NOC_HOPS
        # 565.8 / 524.1 / 472.1, SPREAD 582.0 / 548.2 / 513.7; doc/ttnn/07b-mla-megakernel/perf.jsonl)
        self.placement_name = os.environ.get("GLM53_MLA_PLACEMENT") or "ROW_PLUS_X"
        self.heads = attn.h
        self.probe = _probe_tensor(device) if probe else None
        self.persist = {}
        self.calls = 0

    def _place(self, readers: MlaReaders):
        taken = {(int(c.x), int(c.y)) for c, _, _ in readers.assign}
        place = rect_placement(self.device, taken)
        taken |= {(int(c.x), int(c.y)) for c in place["k"]} | {(int(place["root"].x), int(place["root"].y))}
        return place, _free_cores(self.device, taken)

    def _mk(self, shape, dt, mem=L1):
        return ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, self.device, mem)


class MkMlaA(_MlaBase):
    """P1 mhc_pre -> P2 q_a|kv_a -> P3 norms -> P4 latent cache write -> P5 q_b -> P6 W_uk in ONE program."""

    def __init__(self, device, mhc_args: dict, attn, *, readers_per_bank: int | None = None, probe: bool = False,
                 kb_qa: int = 8, kb_qb: int = 4, kb_uk: int = 2, uk_fidelity=None, rings: tuple | None = None):  # fmt: skip
        super().__init__(device, attn, readers_per_bank=readers_per_bank, probe=probe)
        # ring budgets (KB per reader) of the q_a, q_b and W_uk rings (GLM53_MLA_RINGS_A="qa,qb,uk")
        self.rings = rings or tuple(int(v) for v in os.environ.get("GLM53_MLA_RINGS_A", "140,210,128").split(","))
        host = _host0
        self.w_uk = GroupedWeight(device, host(attn.w_uk).reshape(attn.h, attn.dqk, attn.kv_lora),
                                  readers_per_bank=self.rpb, role="mla_w_uk")  # fmt: skip
        self.nw_q = _norm_weight(device, host(attn.qa_norm).reshape(-1)[: attn.q_lora])
        self.nw_kv = _norm_weight(device, host(attn.kva_norm).reshape(-1)[: attn.kv_lora])
        self.kx_q, self.kx_kv = attn.q_lora // TILE, attn.kv_lora // TILE
        self.kb = (kb_qa, kb_qb, kb_uk)
        self.uk_fidelity = uk_fidelity
        # placement from a phase-less reader table (the same assignment for any phase list)
        probe_readers = mla_reader_assignments(device, self.rpb, self.placement_name)
        taken = {(int(c.x), int(c.y)) for c, _, _ in probe_readers}
        self.place = rect_placement(device, taken)
        taken |= {(int(c.x), int(c.y)) for c in self.place["k"]} | {(int(self.place["root"].x),
                                                                       int(self.place["root"].y))}  # fmt: skip
        self.lat_core = ttnn.CoreCoord(*_free_cores(device, taken)[0])
        self.lat_noc = device.worker_core_from_logical_core(self.lat_core)
        self.mhc = _mhc.MhcPre(device, **mhc_args, placement=self.place)

    def readers(self, t: int) -> MlaReaders:
        attn = self.attn
        kb_qa, kb_qb, kb_uk = self.kb
        ra, rb, rc = self.rings
        # consumers are filled after the assignment is known (all readers + the latent core)
        phases = [
            RPhase(attn.ds_qa_kva, kb_qa, wait=(SEM_HN, 64), ring_kb=ra),
            RPhase(attn.ds_qb, kb_qb, src=SRC_NORM, norm=(self.nw_q, attn.eps, self.kx_q), ring_kb=rb),
            RPhase(self.w_uk, kb_uk, dst=DST_ROWS, rows_heads=attn.h, fid=self.uk_fidelity, ring_kb=rc),
        ]
        r = MlaReaders(self.device, phases, readers_per_bank=self.rpb, placement_name=self.placement_name)
        rnoc = r.noc_coords()
        n = len(rnoc)
        phases[0].signal = (SEM_QA, rnoc + [(int(self.lat_noc.x), int(self.lat_noc.y))])
        phases[1].wait, phases[1].signal = (SEM_QA, n), (SEM_Q, rnoc)
        phases[2].wait, phases[2].signal = (SEM_Q, n), (SEM_UK, [(int(self.lat_noc.x), int(self.lat_noc.y))])
        return r

    def __call__(self, x, cache, page_table, cur_pos, *, t: int):
        """x [1,1,t,16384] fp32 (the layer input) -> dict(hn, post, comb, x_out, coef, qk, q, q_lat, latent); the latent
        cache (paged, DRAM) gets row t at cur_pos[t]."""
        if not 1 <= t <= 4:
            raise ValueError(f"MkMlaA: t={t} (the stacked residual serves t <= 4)")
        attn, mhc, dev = self.attn, self.mhc, self.device
        mk = self._mk
        hn = mk((1, 1, t, mhc.hidden), BF16)
        post, comb = mk((t, 1, 4, 1), FP32), mk((t, 1, 4, 4), FP32)
        x_out = mk((t, 1, 4, mhc.hidden), FP32)
        coef = mk((1, 1, 32, 32), FP32)
        qk = mk((1, 1, t, attn.ds_qa_kva.n_pad), BF16)
        q = mk((1, 1, t, attn.ds_qb.n_pad), BF16)
        q_lat_l1 = mk((1, t, attn.h, attn.kv_lora), BF16)  # the readers' face-row scatter target
        q_lat = mk((1, t, attn.h, attn.kv_lora), BF16, DRAM)  # the flash-MLA op takes an interleaved Q from DRAM only
        latent = mk((1, 1, t, attn.kv_lora), BF16)
        rd = self.readers(t)
        pa = self.probe.buffer_address() if self.probe is not None else None
        desc = mhc.descriptor(x, hn, post, comb, x_out, coef, t=t, x_mode=_mhc.X_FLAT, xout_mode=_mhc.XOUT_STACKED,
                              pc_mode=_mhc.PC_STACKED, signal=(SEM_HN, rd.noc_coords()),
                              mk_probe=(pa, 0) if pa is not None else None)  # fmt: skip
        r_kernels, r_cbs = rd.parts([hn, qk, q], [qk, q, q_lat_l1], t=t,
                                    mk_probe=(pa, 65) if pa is not None else None)  # fmt: skip
        # latent core
        lc = ttnn.CoreRangeSet([ttnn.CoreRange(self.lat_core, self.lat_core)])
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        lxy = (int(self.lat_core.x), int(self.lat_core.y))
        rev = [("MLA_REV", str(MLA_REV)), ("MLA_SRC", "0x" + mla_src()[:8])]
        p0 = [("MK_PROBE_ARG", "3"), ("MK_PROBE_CB", str(PROBE_CB))] if pa is not None else []
        p1 = [("MK_PROBE_ARG", "10"), ("MK_PROBE_CB", str(PROBE_CB))] if pa is not None else []
        lk = lambda src, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "mla", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=lc, compile_time_args=ct, runtime_args=_rt_args({lxy: rt}), defines=defs + rev, config=cfg,
        )  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb_ in (LCB_NW, LCB_INV):
            modes[cb_] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        n_r = len(rd.assign)
        pt_row_bytes = int(page_table.shape[-1]) * 4
        lat_kernels = [
            lk("lat_dm0.cpp", [LCB_X, LCB_NW, self.kx_kv, self.kx_q, SEM_QA, n_r, self.kx_kv] + acc(qk)
               + acc(self.nw_kv),
               [qk.buffer_address(), self.nw_kv.buffer_address(), 0] + ([pa, 130] if pa is not None else []), p0,
               dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            lk("lat_c.cpp", [LCB_X, LCB_NW, LCB_INV, LCB_OUT, self.kx_kv, f32_bits(1.0 / attn.kv_lora),
                             f32_bits(attn.eps), self.kx_kv], [0], [], cc),
            lk("lat_dm1.cpp", [LCB_OUT, LCB_SCR, self.kx_kv, attn.block_size, SEM_UK, n_r, 0, 0, self.kx_kv] + acc(latent)
               + acc(cache)
               + acc(page_table) + acc(cur_pos) + acc(q_lat_l1) + acc(q_lat),
               [latent.buffer_address(), cache.buffer_address(), page_table.buffer_address(), cur_pos.buffer_address(),
                t, pt_row_bytes, q_lat_l1.buffer_address(), q_lat.buffer_address(), t * (-(-attn.h // TILE))
                * (attn.kv_lora // TILE), 0] + ([pa, 131] if pa is not None else []), p1,
               dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
        ]  # fmt: skip
        lcb = lambda i, dt, page, n: ttnn.CBDescriptor(
            total_size=page * n, core_ranges=lc,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # fmt: skip
        scr_bytes = self.kx_kv * 2048 + 256 + _a64(pt_row_bytes)
        lat_cbs = [lcb(LCB_X, BF16, 2048, self.kx_kv), lcb(LCB_NW, FP32, 4096, self.kx_kv), lcb(LCB_INV, FP32, 4096, 1),
                   lcb(LCB_OUT, BF16, 2048, self.kx_kv), lcb(LCB_SCR, ttnn.uint32, scr_bytes, 1)]  # fmt: skip
        if pa is not None:
            lat_cbs.append(lcb(PROBE_CB, ttnn.uint32, 128, 1))
            desc.cbs = list(desc.cbs) + [ttnn.CBDescriptor(total_size=128, core_ranges=mhc.all, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=PROBE_CB, data_format=ttnn.uint32, page_size=128)])]  # fmt: skip
        desc.kernels = list(desc.kernels) + r_kernels + lat_kernels
        desc.cbs = list(desc.cbs) + r_cbs + lat_cbs
        desc.semaphores = list(desc.semaphores) + [
            ttnn.SemaphoreDescriptor(id=SEM_HN, core_ranges=rd.cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_QA, core_ranges=rd.cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_Q, core_ranges=rd.cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_QA, core_ranges=lc, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_UK, core_ranges=lc, initial_value=0),
        ]
        io = [x, mhc.fn, mhc.w, mhc.sb, mhc.dbg, post, comb, coef, x_out, hn, qk, q, q_lat_l1, q_lat, latent]
        io += [attn.ds_qa_kva.weight, attn.ds_qb.weight, self.w_uk.weight, self.nw_q, self.nw_kv, cache, page_table,
               cur_pos]  # fmt: skip
        if self.probe is not None:
            io.append(self.probe)
        ttnn.generic_op(io, desc)
        self.calls += 1
        self.last_readers = rd
        return dict(hn=hn, post=post, comb=comb, x_out=x_out, coef=coef, qk=qk, q=q, q_lat=q_lat, latent=latent)


class MkMlaB(_MlaBase):
    """P7 W_uv -> P8 o_proj -> P9 mhc_post in ONE program (after the flash-MLA op)."""

    POST_CB0 = 0

    def __init__(self, device, attn, *, readers_per_bank: int | None = None, probe: bool = False, kb_uv: int = 4,
                 kb_o: int = 16, uv_fidelity=None, rings: tuple | None = None, attention: bool = False):  # fmt: skip
        """attention=True: the owners run the attention phase first (kernels/mla/att_*.cpp, flash-decoding over the
        latent cache, o_lat written to L1) and then mhc_post; the W_uv readers wait for the attention roots — no
        dispatched flash-MLA op between the programs."""
        super().__init__(device, attn, readers_per_bank=readers_per_bank, probe=probe)
        self.attention = attention
        # ring budgets (KB per reader) of the W_uv and o_proj rings (GLM53_MLA_RINGS_B="uv,o")
        self.rings = rings or tuple(int(v) for v in os.environ.get("GLM53_MLA_RINGS_B", "192,420").split(","))
        host = _host0
        self.w_uv = GroupedWeight(device, host(attn.w_uv).reshape(attn.h, attn.kv_lora, attn.dv),
                                  readers_per_bank=self.rpb, role="mla_w_uv")  # fmt: skip
        self.kb = (kb_uv, kb_o)
        self.uv_fidelity = uv_fidelity
        probe_readers = mla_reader_assignments(device, self.rpb, self.placement_name)
        self.place = rect_placement(device, {(int(c.x), int(c.y)) for c, _, _ in probe_readers})
        self.post = _mpost.MhcPost(device, cores=self.place["k"], cb_base=self.POST_CB0)
        self.owner_noc = [(int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c)
                                                         for c in self.place["k"])]  # fmt: skip
        self.groups = -(-attn.h // TILE)  # head tile rows (G); chunk cores per group C = 64 / G
        self.chunks = len(self.place["k"]) // self.groups
        assert self.chunks & (self.chunks - 1) == 0, self.chunks
        self.mask = mask_bank(device)  # the attention's additive masks + the identity tile

    def readers(self, t: int) -> MlaReaders:
        attn = self.attn
        kb_uv, kb_o = self.kb
        ra, rb = self.rings
        phases = [
            RPhase(self.w_uv, kb_uv, src=SRC_ROWS, rows_heads=attn.h, fid=self.uv_fidelity, ring_kb=ra,
                   wait=(SEM_OL, self.groups) if self.attention else None),
            RPhase(attn.ds_o, kb_o, signal=(SEM_A, self.owner_noc), ring_kb=rb),
        ]  # fmt: skip
        r = MlaReaders(self.device, phases, readers_per_bank=self.rpb, placement_name=self.placement_name)
        rnoc = r.noc_coords()
        phases[0].signal = (SEM_O, rnoc)
        phases[1].wait = (SEM_O, len(rnoc))
        return r

    def __call__(self, o_lat, x_out, coef, *, t: int, cache=None, page_table=None, cur_pos=None, q_lat=None):
        """o_lat [1,t,H,512] (the flash-MLA op's output, L1), x_out / coef (MkMlaA's) -> dict(o, a, s1); attention
        mode: q_lat [1,t,H,512] (MkMlaA's) + the cache / page table / cur_pos instead of o_lat (-> also o_lat)."""
        attn = self.attn
        mk = self._mk
        if self.attention:
            o_lat = mk((1, t, attn.h, attn.kv_lora), BF16)
        o = mk((1, 1, t, attn.h * attn.dv), BF16)
        a = mk((1, 1, t, attn.ds_o.n_pad), BF16)
        s1 = self.post.output(t, _mpost.OUT_STACKED)
        rd = self.readers(t)
        pa = self.probe.buffer_address() if self.probe is not None else None
        r_kernels, r_cbs = rd.parts([o_lat, o], [o, a], t=t, mk_probe=(pa, 65) if pa is not None else None)
        roles, post_cbs, post_io = self.post.parts(a, x_out, coef, s1, t=t, x_mode=_mpost.X_STACKED,
                                                   out_mode=_mpost.OUT_STACKED, wait=(SEM_A, len(rd.assign)))  # fmt: skip
        suffix = "_p" if self.post.last_packed else ""
        sems = [ttnn.SemaphoreDescriptor(id=SEM_O, core_ranges=rd.cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_A, core_ranges=self.post.cores, initial_value=0)]  # fmt: skip
        att_cbs, att_io = [], []
        if not self.attention:
            kernels = [
                ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "mhc_post", f"{name}{suffix}.cpp"),
                                      source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                      core_ranges=self.post.cores, compile_time_args=ct, runtime_args=_rt_args(rt),
                                      defines=defs, config=cfg)
                for name, (ct, rt, defs, cfg) in roles.items()
            ]  # fmt: skip
        else:
            assert self.post.last_packed, "the composite owner kernels include the packed mhc_post kernels"
            kernels, att_cbs = self._owner_kernels(roles, t, q_lat, cache, page_table, cur_pos, o_lat, rd, pa)
            sems += [ttnn.SemaphoreDescriptor(id=SEM_RDY, core_ranges=self.post.cores, initial_value=0),
                     ttnn.SemaphoreDescriptor(id=SEM_DAT, core_ranges=self.post.cores, initial_value=0),
                     ttnn.SemaphoreDescriptor(id=SEM_DATO, core_ranges=self.post.cores, initial_value=0),
                     ttnn.SemaphoreDescriptor(id=SEM_OL, core_ranges=rd.cores, initial_value=0)]  # fmt: skip
            att_io = [q_lat, cache, page_table, cur_pos, self.mask]
        kernels = kernels + r_kernels
        io = [o_lat, o, a, attn.ds_o.weight, self.w_uv.weight] + att_io
        io += [t_ for t_ in post_io if all(t_ is not u for u in io)]
        if self.probe is not None:
            io.append(self.probe)
            if self.attention:
                att_cbs.append(ttnn.CBDescriptor(total_size=128, core_ranges=self.post.cores, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=PROBE_CB, data_format=ttnn.uint32, page_size=128)]))  # fmt: skip
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=post_cbs + r_cbs + att_cbs))
        self.calls += 1
        self.last_readers = rd
        return dict(o=o, a=a, s1=s1, o_lat=o_lat)

    def _owner_kernels(self, roles, t, q_lat, cache, page_table, cur_pos, o_lat, rd, pa):
        """The owners' composite kernels (kernels/mla/owner_b_*.cpp): the attention phase's args + mhc_post's,
        MK_PP_CT / MK_PP_RT = mhc_post's bases."""
        att = attention_args(self.attn, self.place["k"], self.owner_noc, self.groups, self.chunks, t, q_lat, cache,
                             page_table, cur_pos, o_lat, self.mask, rd.noc_coords(), ACB, (0, 0, 0, 0))  # fmt: skip
        cores = [(int(c.x), int(c.y)) for c in self.place["k"]]
        files = dict(reader="owner_b_dm0.cpp", writer="owner_b_dm1.cpp", compute="owner_b_c.cpp")
        kernels = []
        for name, (pct, prt, pdefs, pcfg) in roles.items():
            parts = [att[name], (list(pct), {xy: [int(v) for v in prt[xy]] for xy in cores})]
            ct, rt, bases = concat_args(parts, cores)
            defs = list(pdefs) + [("MK_PP_CT", str(bases[1][0])), ("MK_PP_RT", str(bases[1][1])),
                                  ("MLA_REV", str(MLA_REV)), ("MLA_SRC", "0x" + mla_src()[:8])]  # fmt: skip
            if pa is not None and name != "compute":
                n_all = len(next(iter(rt.values())))
                k_of = {xy: k for k, xy in enumerate(cores)}
                rt = {xy: v + [pa, (0 if name == "reader" else 132) + k_of[xy]] for xy, v in rt.items()}
                defs += [("MK_PROBE_ARG", str(n_all)), ("MK_PROBE_CB", str(PROBE_CB))]
            cfg = attention_compute_config(pcfg, ACB) if name == "compute" else pcfg
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "mla", files[name]),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.post.cores,
                compile_time_args=ct, runtime_args=_rt_args(rt), defines=defs, config=cfg))  # fmt: skip
        return kernels, attention_cbs(self.post.cores, ACB, page_table)


class MkMla(_MlaBase):
    """The whole sparse_dsa attention half as ONE program (megakernel_plan.md contract):
    owners O (64)  : mhc_pre (K role) -> attention (head group k // C, row chunk k % C) -> mhc_post
    root           : mhc_pre's root
    readers R      : q_a|kv_a -> q_b (RMSNorm fused) -> W_uk (head rows -> q_lat, L1) -> W_uv (head rows of o_lat)
                     -> o_proj (-> a, signals mhc_post); ONE weight stream over the five phases (rings: q_a; bfp8 64-tile
                     q_b + o_proj; bf16 32-tile W_uk + W_uv), so W_uv / o_proj weights prefetch during the attention
    latent core L  : RMSNorm_512(kv_a) -> the paged cache write -> signals the attention owners (SEM_LAT)
    Hand-offs: SEM_HN (owners -> R), SEM_QA (R -> R, L), SEM_Q (R -> R), SEM_QL (W_uk R -> owners), SEM_LAT (L ->
    owners), SEM_RDY / SEM_DAT (the attention tree), SEM_OL (attention roots -> R), SEM_O (R -> R), SEM_A (o_proj R ->
    owners). No dispatched op inside the half."""

    def __init__(self, device, mhc_args: dict, attn, *, readers_per_bank: int | None = None, probe: bool = False,
                 kb: tuple | None = None, rings: tuple | None = None, lookahead: bool | None = None,
                 dsa: DsaIndexer | None = None, dsa_kb: tuple = (32, 8), ukv_bf8: bool | None = None,
                 dsa_shard: bool = False):  # fmt: skip
        """ukv_bf8 (GLM53_MLA_UKV_BF8, default 0): W_uk / W_uv as bfloat8_b (HiFi2, as the DS projections) in K blocks
        of 64-tile ring slots, so q_b, W_uk, W_uv and o_proj stream through ONE bfp8 ring (no bf16 ring: its L1 goes to
        the shared ring, which prefetches o_proj during the attention) and the 25 MB of bf16 per-head weights halve.
        dsa: the layer's DsaIndexer -> the DSA phases (5d m4_phase_spec I1..I5 + S1): reader phases I1 (hn @ w_x, fp32
        -> the index core) after q_a and I3 (q_r @ wq_b -> q_idx rows -> the owners) after q_b, the index core (LayerNorm,
        ring row, pooled key, head weights; then the select coordinator), and the owners' sparse attention path for
        tokens with p + 1 > 2051 (kernels/mla/att_dsa_*.cpp); every token's ring row and pooled key are written."""
        super().__init__(device, attn, readers_per_bank=readers_per_bank, probe=probe)
        self.dsa, self.dsa_kb = dsa, dsa_kb
        # dsa_shard (tt/tp_dsa_shard.py): the sequence-sharded split-K variant -- the call takes shard= (the exchange
        # tensors / lanes), the owner, index and latent kernels compile their MLA_DSA_SHARD paths, W_uk also signals
        # the latent core (it copies q_lat into the all-gather block)
        self.dsa_shard = bool(dsa_shard) and dsa is not None
        self.shard_slots = {}
        if self.dsa_shard:
            _shard_header_check()
        # the lookahead weight streamer (kernels/mla/rw.cpp); GLM53_MLA_LOOKAHEAD=0 -> 7a's reader_w.cpp
        self.lookahead = lookahead if lookahead is not None else os.environ.get("GLM53_MLA_LOOKAHEAD", "1") == "1"
        # ring budgets (KB per reader) of the phases' rings: q_a, q_b, W_uk / W_uv (bf16), o_proj (GLM53_MLA_RINGS=
        # "qa,qb,b16,o"); phases whose (format, block) agree share one ring (single chip: q_b and o_proj, 64-tile bfp8
        # blocks; W_uk and W_uv, 32-tile bf16 blocks) and the ring gets the larger budget
        # defaults = the measured best (chip 5, traced, T = 1: steady 444.4 us vs 523.3 with bf16 W_uk / W_uv and
        # rings 140 / 480 / 128 / 480): bfp8 W_uk / W_uv through ONE 720-KB bfp8 ring shared with q_b and o_proj
        self.ukv_bf8 = ukv_bf8 if ukv_bf8 is not None else os.environ.get("GLM53_MLA_UKV_BF8", "1") == "1"
        # (with DSA the I1 / I3 phases keep a 128-KB bf16 ring: the shared ring stays at 600 KB, a 720-KB one clashes
        # with the step's L1 tensors)
        ring_default = (
            ("140,600,128,600" if dsa is not None else "140,720,0,720") if self.ukv_bf8 else "140,480,128,480"
        )
        self.rings = rings or tuple(int(v) for v in os.environ.get("GLM53_MLA_RINGS", ring_default).split(","))
        host = _host0
        # TP mesh (tt/tp_mla.py MeshHeadShard): host_weights = per-rank host slices (w_uk, w_uv, qb), chip r's on chip r
        hw = getattr(attn, "host_weights", None)
        ukv = dict(dtype=BF8, math_fidelity=ttnn.MathFidelity.HiFi2) if self.ukv_bf8 else {}
        self.w_uk = GroupedWeight(device, hw["w_uk"] if hw else host(attn.w_uk).reshape(attn.h, attn.dqk, attn.kv_lora),
                                  readers_per_bank=self.rpb, role="mla_w_uk", **ukv)  # fmt: skip
        self.w_uv = GroupedWeight(device, hw["w_uv"] if hw else host(attn.w_uv).reshape(attn.h, attn.kv_lora, attn.dv),
                                  readers_per_bank=self.rpb, role="mla_w_uv", **ukv)  # fmt: skip
        self.nw_q = _norm_weight(device, host(attn.qa_norm).reshape(-1)[: attn.q_lora])
        self.nw_kv = _norm_weight(device, host(attn.kva_norm).reshape(-1)[: attn.kv_lora])
        self.kx_q, self.kx_kv = attn.q_lora // TILE, attn.kv_lora // TILE
        # K blocks (q_a, q_b, W_uk, W_uv, o_proj); bfp8 W_uk / W_uv: 64-tile blocks (4 x 16, 8 x 8) = q_b's / o_proj's
        self.kb = kb or ((8, 4, 4, 8, 16) if self.ukv_bf8 else (8, 4, 2, 4, 16))
        probe_readers = mla_reader_assignments(device, self.rpb, self.placement_name)
        taken = {(int(c.x), int(c.y)) for c, _, _ in probe_readers}
        self.place = rect_placement(device, taken)
        # owner order (GLM53_MLA_KORDER, default leaders_last): mhc_pre's K core k = 8 cx + cy has its column leader at
        # cy = 0, and the leaders run their own kernel group; rect_placement's column-major cells put every 8th cell in
        # it, which splits both groups into many core ranges (the owner composites are ~65 KB of TRISC code each,
        # multicast once per range at every launch). leaders_last gives the 8 leaders the last 8 cells and the 56
        # members the first ones (point-to-point messaging only, so any k -> core map is valid).
        if os.environ.get("GLM53_MLA_KORDER", "leaders_last") == "leaders_last":
            self.place["k"] = compact_leaders(list(self.place["k"]))
        taken |= {(int(c.x), int(c.y)) for c in self.place["k"]} | {(int(self.place["root"].x),
                                                                       int(self.place["root"].y))}  # fmt: skip
        free = _free_cores(device, taken)
        self.lat_core = ttnn.CoreCoord(*free[0])
        self.lat_noc = device.worker_core_from_logical_core(self.lat_core)
        # q_b's RMSNorm_1536 FOLDED (GLM53_MLA_QFOLD, default 1): q = (q_a @ (diag(w) W_qb)) * 1/rms(q_a) — the norm
        # weight w goes into q_b's weight rows once (host), q_b runs on the raw q_a as soon as every reader's q_a tiles
        # landed, and its output rows are scaled by 1/rms from ONE q-norm core (inv only) just before they are packed;
        # the DSA I3 phase (q_r @ wq_b) folds the same way. Unfolded: q_r on NQ q-norm cores (GLM53_MLA_QN, default 4).
        self.qfold = os.environ.get("GLM53_MLA_QFOLD", "1") == "1"
        # mhc_pre's 1/rms(h) DEFERRED (GLM53_MLA_DEFER, default 0; Stage 14's glue trim): hn = h * w is published before
        # mhc_pre's round 2, so q_a | kv_a starts that much earlier; its output rows (and the DSA I1 rows) are scaled by
        # the root's 1/rms(h) (RS phases, late wait before the first output). The scale cannot be left to the downstream
        # norms: rms(h) is ~4e-3 on this model, and the RMSNorm eps (1e-6) then dominates c^2 mean(x^2) (measured on
        # device without the RS: latent x0.040, q_lat x0.019 -- PCC 0.99999, max abs errors 1.2 / 6.1).
        self.defer = os.environ.get("GLM53_MLA_DEFER", "1") == "1"
        nq = 1 if self.qfold else int(os.environ.get("GLM53_MLA_QN", "4"))
        if self.qfold:
            w_q = host(attn.qa_norm).reshape(-1)[: attn.q_lora]
            wqb = hw["qb"] if hw else host(attn.ds_qb.weight).reshape(attn.q_lora, -1)[:, : attn.ds_qb.n]
            wf = [w_q[:, None] * b for b in wqb] if isinstance(wqb, list) else w_q[:, None] * wqb
            self.qb_fold = DSWeight(device, wf, role="mla_qb_fold", dtype=attn.ds_qb.weight.dtype,
                                    math_fidelity=attn.ds_qb.ckc.math_fidelity)  # fmt: skip
            if dsa is not None:
                self.ix_qb_fold = DSWeight(device, w_q[:, None] * dsa.host["wq_b"], role="dsa_wqb_fold",
                                           dtype=dsa.w_x.weight.dtype)  # fmt: skip
        if self.kx_q % nq:
            raise ValueError(f"MkMla: {nq} q-norm cores do not divide {self.kx_q} tiles")
        self.qn_cores = [ttnn.CoreCoord(*free[1 + i]) for i in range(nq)]
        self.qn_nocs = [device.worker_core_from_logical_core(c) for c in self.qn_cores]
        self.ix_core = ttnn.CoreCoord(*free[1 + nq])  # DSA: the index core (I2 + the I5 coordinator)
        self.ix_noc = device.worker_core_from_logical_core(self.ix_core)
        # every core a role of this program may take (TP mesh: the fabric lanes go elsewhere, tt/tp_mla.py)
        self.used = set(taken) | {tuple(c) for c in free[: 2 + nq]}
        self.mhc = _mhc.MhcPre(device, **mhc_args, placement=self.place)
        self.post = _mpost.MhcPost(device, cores=self.place["k"], cb_base=POST_CB0_ONE)
        self.owner_noc = [(int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c)
                                                         for c in self.place["k"])]  # fmt: skip
        self.groups = -(-attn.h // TILE)
        self.chunks = len(self.place["k"]) // self.groups
        C = self.chunks
        assert C & (C - 1) == 0 and self.groups * C == len(self.place["k"]), (self.groups, C)
        self.mask = mask_bank(device)
        # rev 40: DSA token lanes at t > 4 -- the radix select of a lane's sparse tokens runs over the lane's C / L chunk
        # cores with the lane's own coordinator core (kernels/mla/dsa_coord.cpp), so L tokens select concurrently;
        # GLM53_MLA_DSA_LANES (default 8; 1 = rev 39's one token at a time on every chunk core) coordinator cores
        # are reserved here, a power of two <= C / 2
        self.co_cores = []
        if (
            dsa is not None and not self.dsa_shard
        ):  # (the shard variant, rev 60: tt/tp_dsa_shard.py reserves its lane cores after its stream lanes -- the t <= 4
            # placement unchanged -- with reserve_lane_cores)
            v, p = int(os.environ.get("GLM53_MLA_DSA_LANES", "8")), 1
            while p * 2 <= min(v, C // 2, len(free) - (2 + nq)):
                p *= 2
            if p > 1:
                self.co_cores = [ttnn.CoreCoord(*free[2 + nq + i]) for i in range(p)]
                self.used |= {tuple(free[2 + nq + i]) for i in range(p)}

    def reserve_lane_cores(self, free: list, helpers: int) -> list:
        """rev 60 (W, dsa_shard): the token lanes' index cores of the sharded split-K phase at t > 4, from `free` (the
        cores no role of this program -- nor the caller's fabric / stream lanes -- takes): GLM53_MLA_DSA_LANES (default
        8) rounded down to a power of two <= C / helpers (every lane holds the 2 LINE merge helpers) and <= len(free);
        -> the cores taken (1 lane: none)"""
        assert self.dsa_shard and not self.co_cores, "reserve_lane_cores: the shard variant, once"
        v, p = int(os.environ.get("GLM53_MLA_DSA_LANES", "8")), 1
        while p * 2 <= min(v, self.chunks // max(1, helpers), len(free)):
            p *= 2
        if p > 1:
            self.co_cores = [ttnn.CoreCoord(*free[i]) for i in range(p)]
            self.used |= {tuple(free[i]) for i in range(p)}
        return [tuple(free[i]) for i in range(p)] if p > 1 else []

    def readers(self, t: int) -> MlaReaders:
        attn = self.attn
        kb_qa, kb_qb, kb_uk, kb_uv, kb_o = self.kb
        r_qa, r_qb, r_b16, r_o = self.rings
        # bfp8 W_uk / W_uv share the bfp8 ring (its budget = the larger of q_b's / o_proj's); the DSA projections keep a
        # bf16 ring (budget r_b16, 128 KB when 0)
        r_ukv = max(r_qb, r_o) if self.ukv_bf8 else r_b16
        r_b16 = r_b16 or 128
        ph = dict(
            qa=RPhase(attn.ds_qa_kva, kb_qa, wait=(SEM_HN, 64), ring_kb=r_qa, late=(SEM_INVH, 1) if self.defer else None,
                      signal=(SEM_QA, [(int(n.x), int(n.y)) for n in [self.lat_noc] + self.qn_nocs])),
            qb=(RPhase(self.qb_fold, kb_qb, ring_kb=r_qb, late=(SEM_QR, 1)) if self.qfold
                else RPhase(attn.ds_qb, kb_qb, wait=(SEM_QR, len(self.qn_cores)), ring_kb=r_qb)),
            uk=RPhase(self.w_uk, kb_uk, dst=DST_ROWS, rows_heads=attn.h, ring_kb=r_ukv,
                      signal=(SEM_QL, self.owner_noc + ([(int(self.lat_noc.x), int(self.lat_noc.y))] if self.dsa_shard
                                                         else []))),
            uv=(RPhase(self.w_uv, kb_uv, ring_kb=r_ukv, wait=(SEM_OL, self.groups * self.att_lanes(t))) if self.ot(t)
                else RPhase(self.w_uv, kb_uv, src=SRC_ROWS, rows_heads=attn.h, ring_kb=r_ukv,
                      wait=(SEM_OL, self.groups * self.att_lanes(t)))),  # every (group, lane) root signals once
            o=RPhase(attn.ds_o, kb_o, ring_kb=r_o, signal=(SEM_A, self.owner_noc)),
        )  # fmt: skip
        order = ["qa", "qb", "uk", "uv", "o"]
        if self.dsa is not None:
            # (deferred 1/rms(h): q_a | kv_a waited for it already -- the tile is in DRAM, no second wait); rev 47: every
            # lane index core
            ph["i1"] = RPhase(self.dsa.w_x, self.dsa_kb[0], ring_kb=r_b16, out_fp32=True,
                              signal=(SEM_IX, self._ix_nocs(t)))  # fmt: skip
            ph["i3"] = RPhase(self.ix_qb_fold if self.qfold else self.dsa.wq_b, self.dsa_kb[1], dst=DST_ROWS,
                              rows_heads=DsaIndexer.HEADS, ring_kb=r_b16, signal=(SEM_QI, self.owner_noc))  # fmt: skip
            # rev 48 GLM53_MLA_DSA_QFIRST=1: W_uk before I3 (q_lat reaches the owners ~60 us sooner at t = 32 -- the dense
            # regime's wait, card0 u45; the sparse regime's first I4 then waits for I3's q_ix instead)
            qfirst = os.environ.get("GLM53_MLA_DSA_QFIRST", "0") == "1"
            order = ["qa", "i1", "qb", "uk", "i3", "uv", "o"] if qfirst else ["qa", "i1", "qb", "i3", "uk", "uv", "o"]
        phases = [ph[k] for k in order]
        r = MlaReaders(self.device, phases, readers_per_bank=self.rpb, placement_name=self.placement_name,
                       lookahead=self.lookahead)  # fmt: skip
        idx = {k: i for i, k in enumerate(order)}
        if self.dsa is not None:
            assert r.geo[idx["i3"]]["ng"] == DsaIndexer.DIM // TILE, "I3: one index head per reader column group"
        rnoc = r.noc_coords()
        n = len(rnoc)
        # W_uk's input q tiles: when every reader's q_b columns are exactly its W_uk heads' (whole heads per reader,
        # the same bank / index order), each reader waits for its OWN q_b output only (no all-reader barrier)
        g_qb = r.geo[idx["qb"]]
        aligned = (
            self.w_uk.nr == self.w_uk.ng
            and g_qb["rp"] == self.rpb
            and all(
                b * g_qb["shard_w"] + i * g_qb["nr"] == self.w_uk.reader_geo(b, i)[0] * self.w_uk.kg
                and g_qb["nr"] == self.w_uk.nh * self.w_uk.kg
                for _, b, i in r.assign
            )
        )
        if self.qfold:  # q_b waits for every reader's q_a tiles itself (no q_r core in between)
            ph["qa"].signal = (SEM_QA, [(int(c.x), int(c.y)) for c in [self.lat_noc] + self.qn_nocs] + rnoc)
            ph["qb"].wait = (SEM_QA, n)
        ph["qb"].signal = (SEM_Q, None) if aligned else (SEM_Q, rnoc)
        ph["uk"].wait = (SEM_Q, 1 if aligned else n)
        ph["uv"].signal = (SEM_O, rnoc)
        if self.ot(t) and t < TILE:  # rev 34: o_T's rows >= T are never written -> W_uv's pulls zero them
            r.zero_mask = 1 << idx["uv"]
        ph["o"].wait = (SEM_O, n)
        self.q_aligned = aligned
        self.phase_order = order
        return r

    def __call__(
        self, x, cache, page_table, cur_pos, *, t: int, dsa_state: dict | None = None, keep=None, stack: bool = False,
        shard: dict | None = None
    ):  # fmt: skip
        """x [1,1,t,16384] fp32 (the layer input), t <= 32 -> dict(s1 (t <= 4: [t,1,4,4096] fp32 stacked; t > 4: flat
        [1,1,t,16384], 7a's residual_modes), + the intermediates
        hn, x_out, coef, qk, q, q_lat, latent, o_lat, o, a [+ DSA: y_ix, q_ix, wsc]); the latent cache gets row t at
        cur_pos[t]; with DSA, slot t's ring row and pooled key too (dsa_state = DsaIndexer.state(...)).
        keep (e.g. ("s1",)): only those outputs are returned, every other intermediate is freed right after the enqueue
        (the layer path: at T = 4 they are ~8 MB of L1, which would lower the next program's CB ceiling).
        stack=True (Stage 14b, the persistent step): x = the persistent residual (t <= 4 STACKED [t,1,4,4096], t > 4
        flat [1,1,t,16384] fp32) — P1 reads it (no residual copy) and P6 takes it as the attention residual; s1 as usual
        (the FFN half's input). t > 4 (Stage 14, native t): every token row runs in this ONE launch — the latent core
        writes row t's cache row at cur_pos[t] through row t of the page table, the owners attend row t over its own
        sequence, DSA (dsa=) selects per row (the 32-row mailbox, dsa_tmax)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import residual_modes

        rm = residual_modes(t)  # t <= 4: the stacked residual (packed mhc_post); t > 4: flat (per-tile), t <= 32
        attn, mhc = self.attn, self.mhc
        if cache.dtype != BF16 or int(cache.shape[-1]) != attn.kv_lora:
            raise ValueError(f"MkMla: the latent cache must be bf16 [.., {attn.kv_lora}] (got {cache.dtype})")
        cap = int(page_table.shape[-1]) * attn.block_size
        if self.dsa is None and self.att_lanes(t) == 1 and cap > self.chunks * TILE * ATT_NKMAX:
            raise ValueError(f"MkMla: cache capacity {cap} rows > {self.chunks * TILE * ATT_NKMAX} (dense attention "
                             f"CBs: {ATT_NKMAX} row tiles per chunk core); longer contexts need dsa=")  # fmt: skip
        if self.dsa is not None and dsa_state is None:
            raise ValueError("MkMla(dsa=...): dsa_state required")
        if self.dsa is not None and os.environ.get("GLM53_MLA_ATT_VERIFY") == "1":
            # rev 56 (Stage 17's speculative verify, tt/spec_mtp.py): the t rows are consecutive positions of the user in
            # slot s0 -> one slot (MLA_DSA_ONE_SLOT); one index core in row order (t > 4: token lanes only on the select
            # coordinators, MLA_DSA_COORD_EXT)
            if t > 8:
                raise ValueError(f"MkMla(dsa=...) verify: t <= 8 (got {t})")
            dsa_state = dict(dsa_state, one_slot=True)
        # (ix_lanes: a one-slot verify keeps ONE index core -- its rows' ring / pooled-key writes in row order)
        self._one_slot = bool(dsa_state is not None and dsa_state.get("one_slot"))
        self._paged_now = bool(dsa_state is not None and dsa_state.get("paged"))
        if self._paged_now and (shard is None or os.environ.get("GLM53_MLA_DSA_IX_ROWS", "1") != "1"):
            raise ValueError("MkMla: the paged pooled-key store (rev 63) needs the shard and GLM53_MLA_DSA_IX_ROWS=1")
        if self.dsa_shard != (shard is not None):
            raise ValueError("MkMla(dsa_shard=True) takes shard= (tt/tp_dsa_shard.py) and only then")
        if shard is not None:
            self.shard_lanes_check(t, int(shard.get("line", 1)))
        if shard is not None and dsa_state is not None and dsa_state.get("window") is not None:
            # the dense window (the caller's replicated store): the split-K phase's all-dense pass + the latent write
            win = dsa_state["window"]
            assert int(win.shape[-2]) == 64 and int(win.shape[0]) % DSA_WIN_BLOCKS == 0, tuple(win.shape)
            s0 = int(dsa_state.get("s0", 0))
            pps = DSA_WIN_BLOCKS * (64 // TILE) * (512 // TILE)  # window tiles per slot (bf16 2 KiB pages)
            banks = int(ttnn.get_memory_view(win.device(), ttnn.BufferType.DRAM).num_banks)
            assert (s0 * pps) % banks == 0, (s0, pps, banks)
            shard = dict(shard, window=win, window_a=int(win.buffer_address()) + s0 * pps // banks * 2048)
        self._shard_now = shard
        if shard is not None:
            # rev 59 hardening: only THIS build's rank / peer slots (entries from an earlier call -- another t, other arg
            # widths -- would patch wrong words, and the read-back audit would pass on the word just written)
            self.shard_slots = {}
        if shard is not None and self._one_slot and self.att_lanes(t) != 1 and not self._baton(t):
            # rev 61: the verify's rows write one slot's ring / pooled keys in row order on ONE index core, and the shard
            # runs one index core per token lane -> one lane (t <= 4, or GLM53_MLA_DSA_LANES_AT="t:1" for this width AND
            # its plain step: dsa/spec_on_shard_design.md section 3)
            raise ValueError(f"MkMla(dsa_shard=True) verify at t={t}: one token lane (att_lanes {self.att_lanes(t)}; "
                             f'GLM53_MLA_DSA_LANES_AT="{t}:1")')  # fmt: skip
        if cur_pos.layout != ttnn.ROW_MAJOR_LAYOUT or _a64(4 * int(cur_pos.shape[-1])) > DSA_POS_BYTES:
            raise ValueError(f"MkMla: cur_pos must be row-major int32, <= 32 entries per page ({cur_pos.shape})")
        mk = self._mk
        hn = mk((1, 1, t, mhc.hidden), BF16)
        post, comb = (mk(sh, FP32) for sh in rm["pc_shapes"])
        # P6's attention residual: t <= 4 P1's stacked copy of x (stack: x itself); t > 4 x itself (flat, no copy)
        x_out = mk((t, 1, 4, mhc.hidden), FP32) if rm["stacked"] and not stack else None
        coef = mk((1, 1, 32, 32), FP32)
        qk = mk((1, 1, t, attn.ds_qa_kva.n_pad), BF16)
        q = mk((1, 1, t, attn.ds_qb.n_pad), BF16)
        q_r = None if self.qfold else mk((1, 1, t, attn.q_lora), BF16)
        # one-tile hand-offs in DRAM: an L1-interleaved tensor takes a page of the L1 ceiling on EVERY core
        qinv = mk((1, 1, TILE, TILE), FP32, DRAM) if self.qfold else None  # 1 / rms(q_a), column 0 (the folded phases)
        invh = mk((1, 1, TILE, TILE), FP32, DRAM) if self.defer else None  # mhc_pre's 1 / rms(h) (deferred; DSA I1)
        q_lat = mk((1, t, attn.h, attn.kv_lora), BF16)
        latent = mk((1, 1, t, attn.kv_lora), BF16)
        o_lat = mk((1, t, attn.h, attn.kv_lora), BF16)
        o_t = mk((1, 1, t, attn.h * attn.kv_lora), BF16) if self.ot(t) else None  # rev 30: W_uv's token-row in0
        o = mk((1, 1, t, attn.h * attn.dv), BF16)
        a = mk((1, 1, t, attn.ds_o.n_pad), BF16)
        s1 = self.post.output(t, rm["s1_out"], out_memory_config=self.s1_memory(t))
        rd = self.readers(t)
        n_r = len(rd.assign)
        pa = self.probe.buffer_address() if self.probe is not None else None
        defer = None
        if self.defer:  # the root writes 1/rms(h) to invh and signals every reader (SEM_INVH: q_a | kv_a's late wait)
            defer = (invh, SEM_INVH, rd.noc_coords())
        roles, mhc_cbs, mhc_sems = mhc.parts(x, hn, post, comb, x_out, coef, t=t,
                                             x_mode=rm["s1_x"] if stack else _mhc.X_FLAT,
                                             xout_mode=_mhc.XOUT_NONE if stack else rm["xout"],
                                             pc_mode=rm["pc"],
                                             signal=(SEM_HN, rd.noc_coords()), defer=defer,
                                             mk_probe=(pa, 0) if pa is not None else None)  # fmt: skip
        proles, post_cbs, post_io = self.post.parts(a, x_out if x_out is not None else x, coef, s1, t=t,
                                                    x_mode=rm["post_x"], out_mode=rm["s1_out"], wait=(SEM_A, n_r),
                                                    packed=rm["stacked"])  # fmt: skip
        assert self.post.last_packed == rm["stacked"]
        if not rm["stacked"]:  # the owner composites' per-tile mhc_post kernels (kernels/mla/owner_*.cpp)
            for role in proles.values():
                role[2].append(("MLA_POST_TILE", "1"))
        xs = dict(qa=hn, qb=qk if self.qfold else q_r, uk=q, uv=o_lat if o_t is None else o_t, o=o)
        if self.qfold:
            for k_, p_ in zip(self.phase_order, rd.phases):
                if k_ in ("qb", "i3"):
                    p_.rs = qinv
        if self.defer:
            rd.phases[self.phase_order.index("qa")].rs = invh
            if self.dsa is not None:  # I1 reads invh without a wait: every I1 reader must have waited in q_a | kv_a
                iq, i1 = self.phase_order.index("qa"), self.phase_order.index("i1")
                assert rd.geo[i1]["rp"] <= rd.geo[iq]["rp"], "an I1 reader that sits q_a | kv_a out"
                rd.phases[i1].rs = invh
        ys = dict(qa=qk, qb=q, uk=q_lat, uv=o, o=a)
        dsa_args, extra = None, {}
        if self.dsa is not None:
            y_ix = mk((1, 1, t, self.dsa.w_x.n_pad), FP32, DRAM)  # the DSA hand-offs: read once, kept off the L1 top
            q_ix = mk(
                (1, t, DsaIndexer.HEADS, DsaIndexer.DIM), BF16
            )  # L1: I3's 32-B face-row scatter (validated to L1)
            wsc = mk((1, 1, TILE, TILE), FP32, DRAM)
            kc = mhc.kcores
            mbox = _sharded_u32(self.device, kc, len(self.place["k"]), dsa_mbox_bytes(t, shard is not None) // 4)
            ixc = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self._ix_cores(t)])
            nl = self.att_lanes(t)
            if shard is not None:  # the coordinators' merge scratch (ix_dm0.cpp MLA_DSA_SHARD; rev 60: one per lane)
                hist = _sharded_u32(self.device, ixc, len(self._ix_cores(t)),
                                    shard_coord_words(self.chunks // nl, int(shard["C"])))  # fmt: skip
            elif self.ix_lanes(t) > 1:  # rev 47: each lane index core holds its C / L histogram slots (CB_CSCR scratch)
                hist = _sharded_u32(self.device, ixc, nl, (self.chunks // nl) * 16)
            elif nl > 1:  # rev 40: each lane's coordinator holds its C / L histogram slots + 512 B of scratch
                coc = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.co_cores[:nl]])
                hist = _sharded_u32(
                    self.device, coc, nl, (self.chunks // nl) * 16 + 128 + 16
                )  # + 64 B: aligned scratch
            else:
                hist = _sharded_u32(self.device, ixc, 1, self.chunks * 16)
            xs.update(i1=hn, i3=qk if self.qfold else q_r)
            ys.update(i1=y_ix, i3=q_ix)
            pools_a, ring_a = dsa_slot_addrs(dsa_state, t)
            dsa_args = dict(n_r=n_r, tp_tiles=dsa_state["tp"] // TILE, q_ix=q_ix, wsc=wsc, pools=dsa_state["pools"],
                            pools_a=pools_a, ring_a=ring_a,
                            ix_noc=(int(self.ix_noc.x), int(self.ix_noc.y)), hist=hist, mbox=mbox,
                            coord_noc=[(int(w.x), int(w.y)) for w in (self.device.worker_core_from_logical_core(c)
                                                                       for c in self.co_cores[:nl])] if nl > 1 else None,
                            shard=shard)  # fmt: skip
            # (mbox: the I5 messages, a harness diagnostic; rev 53, review S5: hist too, so TPMla's cleanup frees both)
            extra = dict(y_ix=y_ix, q_ix=q_ix, wsc=wsc, mbox=mbox, hist=hist)
            # mailbox shard row of each owner (height-sharded over kcores, row-major): its I5 messages per token
            self.mbox_rows = {(int(c.x), int(c.y)): i for i, c in enumerate(ttnn.corerange_to_cores(kc, row_wise=True))}
            # rev 60: the index cores in the histogram tensor's shard-row order (a harness reads lane l's T_g probe)
            self.ix_rows = [(int(c.x), int(c.y)) for c in ttnn.corerange_to_cores(ixc, row_wise=True)]
        att = attention_args(attn, self.place["k"], self.owner_noc, self.groups, self.chunks, t, q_lat, cache,
                             page_table, cur_pos, o_lat, self.mask, rd.noc_coords(), ACB1,
                             (SEM_QL, n_r, SEM_LAT, 1), dsa=dsa_args, lanes=self.att_lanes(t), o_t=o_t)  # fmt: skip
        self._dsa_k = self._dsa_keys(dsa_state, t)  # (rev 36: the owners' key capacity, MLA_DSA_KEYS)
        kernels = self._owner_kernels(roles, proles, att, pa, t)
        for r_ in roles:  # mhc_pre's root kernels, unchanged
            if ("MHC_ROOT", "1") in r_["defines"]:
                kernels.append(ttnn.KernelDescriptor(
                    kernel_source=os.path.join(KDIR, "mhc_pre", r_["name"]),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=r_["cores"],
                    compile_time_args=r_["ct"], runtime_args=_rt_args(r_["rt"]), defines=r_["defines"],
                    config=r_["config"]))  # fmt: skip
        r_kernels, r_cbs = rd.parts([xs[k] for k in self.phase_order], [ys[k] for k in self.phase_order], t=t,
                                    mk_probe=(pa, 65) if pa is not None else None)  # fmt: skip
        lat_kernels, lat_cbs = self._norm_core(
            self.lat_core,
            self.kx_kv,
            self.kx_q,
            self.nw_kv,
            qk,
            latent,
            cache,
            page_table,
            cur_pos,
            t,
            SEM_LAT,
            self.owner_noc,
            pa,
            (130, 131),
            n_r,
            shard=dict(shard, q_lat=q_lat, ql_tiles=t * -(-attn.h // TILE) * (attn.kv_lora // TILE)) if shard else None,
        )
        qn_kernels, qn_cbs = self._norm_core(self.qn_cores, self.kx_q, 0, self.nw_q, qk,
                                             qinv if self.qfold else q_r, None, page_table, cur_pos, t, SEM_QR,
                                             rd.noc_coords(), pa, (196, 197), n_r, inv_only=self.qfold)  # fmt: skip
        lat_kernels, lat_cbs = lat_kernels + qn_kernels, lat_cbs + qn_cbs
        kcores = mhc.kcores
        att_cbs = attention_cbs(kcores, ACB1, page_table, dsa_keys=self._dsa_k, dsa_cand=self.dsa_compact(),
                                dsa_sb=self.dsa_sb(), nk=self.att_nkmax(t),
                                obf16=self.att_obf16(t), shard_t=-(-t // self.att_lanes(t)) if shard is not None else 0,
                                shard_tl=shard_tl(t))  # fmt: skip
        ix_sems = []
        if self.dsa is not None:
            ix_kernels, ix_cbs = self._index_core(t, y_ix, wsc, cur_pos, dsa_state, hist, mbox, n_r, pa=pa, shard=shard,
                                                  page_table=page_table)  # fmt: skip
            lat_kernels, lat_cbs = lat_kernels + ix_kernels, lat_cbs + ix_cbs
            ix_sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=ixc, initial_value=0) for i in (SEM_IX, SEM_H)]
            if self.att_lanes(t) > 1 and self.ix_lanes(t) == 1:  # rev 40: the lanes' select coordinators
                co_kernels, coc = self._coord_kernels(t, cur_pos, hist, mbox)
                lat_kernels = lat_kernels + co_kernels
                ix_sems.append(ttnn.SemaphoreDescriptor(id=SEM_H, core_ranges=coc, initial_value=0))
            ix_sems += [ttnn.SemaphoreDescriptor(id=i, core_ranges=kcores, initial_value=0)
                        for i in (SEM_SEL, SEM_IXR, SEM_QI)]  # fmt: skip
            if shard is not None:  # the reduce-scatter consumer (owners), the all-gather consumer (index core)
                ix_sems += [ttnn.SemaphoreDescriptor(id=SEM_RSC, core_ranges=kcores, initial_value=0),
                            ttnn.SemaphoreDescriptor(id=SEM_AGC, core_ranges=ixc, initial_value=0)]  # fmt: skip
                if int(shard.get("line", 1)) > 1:  # rev 59: ix_dm1's q assembly -> ix_dm0
                    ix_sems.append(ttnn.SemaphoreDescriptor(id=SEM_QASM, core_ranges=ixc, initial_value=0))
                if self._baton(t):  # rev 62: the verify's lane baton (ix_dm1.cpp MLA_DSA_ONE_SLOT_BATON)
                    ix_sems += [ttnn.SemaphoreDescriptor(id=i, core_ranges=ixc, initial_value=0)
                                for i in (SEM_BATON, SEM_DONE)]  # fmt: skip
        probe_cbs = []
        if pa is not None:
            probe_cbs = [ttnn.CBDescriptor(total_size=128, core_ranges=mhc.all, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=PROBE_CB, data_format=ttnn.uint32, page_size=128)])]  # fmt: skip
        lc = ttnn.CoreRangeSet([ttnn.CoreRange(self.lat_core, self.lat_core)])
        qc = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.qn_cores])
        sems = list(mhc_sems) + [ttnn.SemaphoreDescriptor(id=i, core_ranges=rd.cores, initial_value=0)
                                 for i in (SEM_HN, SEM_QA, SEM_Q, SEM_O, SEM_OL)]  # fmt: skip
        sems += [ttnn.SemaphoreDescriptor(id=i, core_ranges=kcores, initial_value=0)
                 for i in (SEM_QL, SEM_LAT, SEM_RDY, SEM_DAT, SEM_A, SEM_DATO)]  # fmt: skip
        sems += [ttnn.SemaphoreDescriptor(id=SEM_QA, core_ranges=lc, initial_value=0),
                 ttnn.SemaphoreDescriptor(id=SEM_QA, core_ranges=qc, initial_value=0),
                 ttnn.SemaphoreDescriptor(id=SEM_QR, core_ranges=rd.cores, initial_value=0)]  # fmt: skip
        if shard is not None:  # W_uk's readers signal the latent core too (its q_lat copy into the all-gather block)
            sems.append(ttnn.SemaphoreDescriptor(id=SEM_QL, core_ranges=lc, initial_value=0))
        if self.defer:
            sems.append(ttnn.SemaphoreDescriptor(id=SEM_INVH, core_ranges=rd.cores, initial_value=0))
        sems += ix_sems
        io = [x, mhc.fn, mhc.w, mhc.sb, mhc.dbg, post, comb, coef] + ([x_out] if x_out is not None else [])
        io += [hn, qk, q, qinv if self.qfold else q_r, q_lat, latent, o_lat, o, a, s1]
        io += [o_t] if o_t is not None else []
        if self.dsa is not None:
            io += [y_ix, q_ix, wsc, mbox, hist, self.dsa.w_x.weight, self.dsa.wq_b.weight, self.dsa.const,
                   dsa_state["pools"], dsa_state["ring"]]  # fmt: skip
        if shard is not None:
            io += [shard["x_ag"], shard["y_ag"], shard["x_rs"], shard["y_rs"]]
            io += [shard["x_rs2"], shard["y_rs2"], shard["q_asm"]] if int(shard.get("line", 1)) > 1 else []
            io += [shard["window"]] if shard.get("window") is not None else []
        if self.qfold:
            io += [self.qb_fold.weight] + ([self.ix_qb_fold.weight] if self.dsa is not None else [])
        if self.defer:
            io.append(invh)
        io += [attn.ds_qa_kva.weight, attn.ds_qb.weight, self.w_uk.weight, self.w_uv.weight, attn.ds_o.weight,
               self.nw_q, self.nw_kv, cache, page_table, cur_pos, self.mask]  # fmt: skip
        io += [t_ for t_ in post_io if all(t_ is not u for u in io)]
        if self.probe is not None:
            io.append(self.probe)
        self.last_dispatch = dispatch_stats(kernels + r_kernels + lat_kernels)
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels + r_kernels + lat_kernels, semaphores=sems,
                                                   cbs=list(mhc_cbs) + att_cbs + post_cbs + r_cbs + lat_cbs
                                                   + probe_cbs))  # fmt: skip
        self.calls += 1
        self.last_readers = rd
        out = dict(s1=s1, hn=hn, x_out=x_out, coef=coef, qk=qk, q=q, q_lat=q_lat, latent=latent, o_lat=o_lat, o=o,
                   a=a, **extra)  # fmt: skip
        out.update(dict(qinv=qinv) if self.qfold else dict(q_r=q_r))
        if o_t is not None:  # (in the dict so keep= frees it, as invh)
            out["o_t"] = o_t
        if self.defer:  # the deferred 1/rms(h) tile goes with the other intermediates (keep: freed): left out, the
            # program cache's reference kept it alive -> every later trace capture left one buffer per layer alive
            # under the earlier traces (Stage 14b 2026-10-20 21:12Z, m5b_trace_leak)
            out["invh"] = invh
        out.update(post=post, comb=comb)
        if keep is not None:
            # (mbox / hist are outputs since revs 51 / 53: freed here unless kept - no second explicit deallocate)
            for k in [k for k in out if k not in keep]:
                v = out.pop(k)
                if v is not None:
                    ttnn.deallocate(v)
        else:
            out.pop("post"), out.pop("comb")
        return out

    def att_lanes(self, t: int) -> int:
        """token lanes of the attention phase (kernels/mla/att_common.hpp LANES; attention_args): 1 at t <= 4; with DSA
        at t > 4 (rev 40) up to the reserved select coordinators (GLM53_MLA_DSA_LANES); dense t > 4 GLM53_MLA_ATT_LANES (default 0 = auto: C / 4
        lanes of 4 chunk cores), as a power of two <= min(t, C): the tokens of different lanes run concurrently. TP2
        card0 2026-10-20 (unwatched, traced replay, layer 3): t = 32 sequential 2436 us, 8 lanes 1101, 16 lanes 1025,
        32 lanes 1022; t = 16: 8 lanes 721, 16 lanes 699"""
        if self.dsa is not None:  # rev 40: t > 4 lanes with their own select coordinators (the reserved cores)
            p, cap = 1, _dsa_lanes_at(t)
            while t > DSA_T_MAX and p * 2 <= min(len(self.co_cores), t, cap):
                p *= 2
            return p
        if t <= DSA_T_MAX:  # GLM53_MLA_ATT_LANES_T4 = P (t <= 4, default 0 = off): P lanes of C / P cores (t = 1: one
            # lane works, a log2(C / P)-level tree instead of log2(C); t = 2..4: tokens in parallel)
            # default 4 at t = 2..4 (vtp8 2026-10-20: t = 4 362.7 us vs 491.4 sequential; 16 lanes 422.0); t = 1: 1
            # (4 lanes 300.3 vs 298.2: neutral)
            v = int(os.environ.get("GLM53_MLA_ATT_LANES_T4", "4" if t >= 2 else "0"))
            if v <= 1:
                return 1
        else:
            v = int(os.environ.get("GLM53_MLA_ATT_LANES", "0")) or max(1, self.chunks // 4)
        p = 1
        cap = self.chunks if t <= DSA_T_MAX else min(t, self.chunks)  # t <= 4: lanes may exceed t (narrower trees)
        while p * 2 <= min(v, cap):
            p *= 2
        return p

    def shard_lanes_check(self, t: int, line: int = 1) -> None:
        """rev 60 (W, shard_tp8_design.md section 7): the sharded split-K phase at t rows -- t <= SHARD_T_MAX, one head
        group, and with token lanes (t > 4: att_lanes(t) = L, the lane cores tt/tp_dsa_shard.py reserves) at most
        shard_tl(t) tokens per lane, one index core per lane (the shard coordinator: GLM53_MLA_DSA_IX_LANES=1), and the
        2 LINE merge helpers among the lane's chunk cores (TP8: L <= C / 8); raises otherwise"""
        nl = self.att_lanes(t)
        why = []
        if t > SHARD_T_MAX:
            why.append(f"t <= {SHARD_T_MAX}")
        if self.groups != 1:
            why.append(f"one head group (groups {self.groups})")
        if -(-t // nl) > shard_tl(t):
            why.append(
                f"<= {shard_tl(t)} tokens per token lane ({nl} lanes: GLM53_MLA_DSA_LANES / the reserved lane cores)"
            )
        if nl > 1 and self.ix_lanes(t) != nl:
            why.append("one index core per token lane (GLM53_MLA_DSA_IX_LANES=1)")
        if self.chunks // nl < 2 * line:
            why.append(f"{2 * line} merge helpers among the lane's {self.chunks // nl} chunk cores")
        if why:
            raise ValueError(f"MkMla(dsa_shard=True) at t={t}: {'; '.join(why)}")

    def att_nkmax(self, t: int) -> int:
        """row tiles per attention batch (att_common.hpp NKMAX; CB_AK holds one batch of K tiles). t > 4 dense:
        GLM53_MLA_ATT_NKMAX (default 3). Each batch pays a fixed cost: the diag rescale of the fp32 O, 16 O packs, the
        row reductions and the stats. The rev 29 vtp8 fit (t = 1 vs t = 32 probe spans) puts it at ~5-7 us, paid 12
        times per core at t = 32 with 3 row tiles per batch. L1 per owner core: CB_AK 32 KB + CB_AS / CB_AE 4 KB each +
        CB_AP 2 KB per row tile."""
        if t <= DSA_T_MAX:
            return ATT_NKMAX
        if (
            self.dsa is not None
        ):  # rev 49 GLM53_MLA_DSA_NKMAX (t > 4; default 3 = rev 48): the dense rows' and S1's batches
            v = max(1, min(16, int(os.environ.get("GLM53_MLA_DSA_NKMAX", str(ATT_NKMAX)))))
            # rev 53 (review S1): without token lanes (GLM53_MLA_DSA_LANES=1) the dense rows of a core can exceed a batch
            # below 3 row tiles (att_c.cpp pad(CB_AS, NKMAX - nk) would underflow): keep the rev <= 48 floor there
            return max(v, ATT_NKMAX) if self.att_lanes(t) == 1 else v
        # default 8 at t > 4: rev 33 at t = 32 (card0 2026-10-21 01:09Z vtp8 with o_T: NKMAX 3 / 4 / 6 / 8 = 625.5 / 611.0
        # / 598.6 / 595.2 us); rev 35 at t = 8 / 16 (01:39Z: t = 8 413.3 -> 384.7 with bf16 O, t = 16 neutral 522.6 ->
        # 515.9; watched TP2 numerics pass at t = 8 / 16 / 32)
        default = 8
        return max(1, min(16, int(os.environ.get("GLM53_MLA_ATT_NKMAX", str(default)))))

    def att_obf16(self, t: int) -> bool:
        """bf16 O partials (GLM53_MLA_ATT_OBF16, 7b rev 22): the default since rev 33 wherever NKMAX > 3, i.e. t = 32.
        NKMAX 8 adds ~160 KB of owner CBs; the bf16 O frees 128 KB. 14b's native B = 32 stack mock: NKMAX 8 with fp32 O
        overflows the owners' L1 by 37 KB, with bf16 O it fits. Watched TP2 numerics (01:08Z): o_lat vs f64 rel L2
        0.0039 vs 0.0048 fp32. Timing is neutral at NKMAX 3: 625.0 vs 625.5 us."""
        v = os.environ.get("GLM53_MLA_ATT_OBF16")
        return v == "1" if v is not None else self.att_nkmax(t) > ATT_NKMAX

    def ot(self, t: int) -> bool:
        """rev 30 o_T (dense attention, t = 32; default on since rev 33, GLM53_MLA_OT=0 restores the gather; card0 vtp8
        2026-10-21: MLA half 729.3 -> 625.5 us, W_uv 127.5 -> 4.7 us, W_uv output bit-identical): the attention roots write each head's o rows into a
        [1, 1, T, H * 512] token-row tensor (att_dm1.cpp MLA_ATT_OT) and W_uv pulls its in0 as whole tiles (SRC_TENSOR)
        instead of gathering face rows (rev 29 vtp8 t = 32: W_uv 127.5 us, 4 readers per head gathering the same 1024
        rows). 4 < t < 32 (rev 34, opt-in GLM53_MLA_OT_SMALL=1): o_T's rows >= T are never written, so W_uv's pulls zero
        them after the tile reads (ra.cpp MLA_PULL_ZMASK)"""
        if t <= DSA_T_MAX or os.environ.get("GLM53_MLA_OT", "1") != "1":
            return False
        if self.dsa is not None and os.environ.get("GLM53_MLA_DSA_OT", "1") != "1":  # rev 39: the DSA roots too
            return False
        if t != TILE and os.environ.get("GLM53_MLA_OT_SMALL", "1") != "1":  # rev 35: on by default (34: opt-in)
            return False
        return self.attn.h <= TILE or self.attn.h % TILE == 0

    def s1_memory(self, t: int):
        """s1's memory: L1 (t <= 4 always; the stacked 512 KB / row); t > 4 the flat [1,1,t,16384] fp32 (2 MB) goes to
        DRAM with GLM53_MLA_S1_DRAM=1 (the downstream MkMoE reads it through its accessor either way)"""
        if t > DSA_T_MAX and os.environ.get("GLM53_MLA_S1_DRAM", "0") == "1":
            return DRAM
        return L1

    def dsa_sb(self) -> int:
        """rev 37 GLM53_MLA_DSA_SB (default 1 since rev 44, 1..4): pooled-key tiles per I4 batch. The owner BRISC reads SB tiles (4 x SB
        pages) per barrier, and the compute scores them in one DST acquire (SB fp32 slots). The scores are bitwise as at
        1. Probe 2026-10-21 at p = 1,048,574: I4 294 us (128 tiles per owner, one tile in flight). Rev 42: default 2 --
        the owners' CB_APK / ASR / ASC take 4 SB x 2 KB / SB x 4 KB / 2 SB x 4 KB = 112 KB at 4, which ended the t = 32 DSA
        program's static CBs 2.9 KB (4.7 KB at narrow w = 1) above 14b's served stack L1 floor on the owners (int16 mock
        2026-10-21 05:22Z); 2 frees 56 KB per owner. Rev 44: default 1 -- card0 u38 (2026-10-21 08:11Z, TP8 shard at
        p = 1,048,574, 2,000 replays): SB 4 996.9 us vs SB 1 969.5 us, I4 289 vs 295 us -- the scoring is DRAM-bound on the
        64 MB of replicated pooled keys (~220 GB/s), so batching the DST work does not help; 1 frees another 56 KB."""
        if self.dsa is None:
            return 1
        return max(1, min(4, int(os.environ.get("GLM53_MLA_DSA_SB", "1"))))

    def dsa_compact(self) -> bool:
        """rev 36 GLM53_MLA_DSA_COMPACT (default on): the owners' radix rounds scan the candidates that still match the
        prefix, compacted after every round, instead of every key. The selection is unchanged. card0 2026-10-21 probe at
        p = 1,048,574 (TP8 head shard): I5 select 559 us of the 1,161-us step (scoring 294)."""
        return self.dsa is not None and os.environ.get("GLM53_MLA_DSA_COMPACT", "1") == "1"

    def _dsa_keys(self, st, t: int = 1) -> int:
        """keys one owner holds per sparse token (CB_AKEY): its pooled-key tiles at the slot's capacity x 32 (rev 40:
        over the token lane's C / L chunk cores)"""
        if self.dsa is None:
            return 0
        return -(-(st["tp"] // TILE) // (self.chunks // self.att_lanes(t))) * TILE

    def _coord_kernels(self, t, cur_pos, hist, mbox):
        """rev 40: the select coordinator of each token lane (kernels/mla/dsa_coord.cpp, BRISC of the reserved core l):
        lane l's sparse tokens, its C / L group-0 chunk cores' histograms (SEM_H on its core), broadcasts to the lane's
        owners of every head group -> (kernels, the cores' CoreRangeSet)"""
        nl = self.att_lanes(t)
        cl = self.chunks // nl
        cores = self.co_cores[:nl]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        acc = list(ttnn.TensorAccessorArgs(cur_pos).get_compile_time_args())
        rt = {}
        for lane, c in enumerate(cores):
            own = [v for g in range(self.groups) for k in range(cl)
                   for v in self.owner_noc[g * self.chunks + lane * cl + k]]  # fmt: skip
            rt[(int(c.x), int(c.y))] = [cur_pos.buffer_address(), t, hist.buffer_address(), mbox.buffer_address(),
                                        lane] + own  # fmt: skip
        defs = [("MLA_REV", str(MLA_REV)), ("MLA_SRC", "0x" + mla_src()[:8]), ("MLA_ATT_LANES", str(nl))]
        if os.environ.get("GLM53_MLA_DSA_HSUM", "1") == "0":
            defs.append(("MLA_DSA_HSUM_OLD", "1"))
        if dsa_tmax(t) != DSA_T_MAX:
            defs.append(("MLA_DSA_TMAX", str(dsa_tmax(t))))
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "mla", "dsa_coord.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=[SEM_H, SEM_SEL, cl, self.groups * cl] + acc, runtime_args=_rt_args(rt),
            defines=defs, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                    noc=ttnn.NOC.RISCV_0_default),
        )  # fmt: skip
        return [kd], crs

    def ix_lanes(self, t: int) -> int:
        """rev 47 (GLM53_MLA_DSA_IX_LANES, default on): with DSA token lanes, one INDEX CORE per lane -- the reserved lane
        cores run the whole index core (ix_dm0 + the lane's select coordinator, ix_c, ix_dm1) on the lane's tokens, so the
        per-token ring / pooled-key work (~20 us per token, the t = 32 critical path, card0 u42 / u44) runs L-wide.
        1: one index core (+ rev 40's dsa_coord coordinators when att_lanes(t) > 1)."""
        nl = self.att_lanes(t)
        if getattr(self, "_one_slot", False) and not self._baton(t):  # rev 56: a verify's rows share one slot: one
            return 1  # index core (rev 62: the shard's lane baton keeps the lanes)
        return nl if nl > 1 and os.environ.get("GLM53_MLA_DSA_IX_LANES", "1") == "1" else 1

    def _owner_rects(self, idx: list) -> list:
        """rev 50: owners idx (place["k"] indices) as [(x0, y0, x1, y1, n)] virtual-NoC rectangles that cover exactly
        those cores (merged logical ranges whose virtual bounding box holds exactly their cores -- else split into rows,
        else single cores: n = 1 = a unicast in ix_dm0), for ix_dm0's multicast broadcast"""
        v = lambda x, y: self.device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))  # noqa: E731
        cells = [(int(self.place["k"][i].x), int(self.place["k"][i].y)) for i in idx]

        def exact(x0, y0, x1, y1):
            a, b = v(x0, y0), v(x1, y1)
            n = (x1 - x0 + 1) * (y1 - y0 + 1)
            if b.x < a.x or b.y < a.y or (b.x - a.x + 1) * (b.y - a.y + 1) != n:
                return None
            inside = all(a.x <= v(x, y).x <= b.x and a.y <= v(x, y).y <= b.y
                         for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))  # fmt: skip
            return (int(a.x), int(a.y), int(b.x), int(b.y), n) if inside else None

        out = []
        for r in merged_ranges(cells).ranges():
            x0, y0, x1, y1 = int(r.start.x), int(r.start.y), int(r.end.x), int(r.end.y)
            e = exact(x0, y0, x1, y1)
            if e is not None:
                out.append(e)
                continue
            for y in range(y0, y1 + 1):
                e = exact(x0, y, x1, y)
                out += [e] if e is not None else [exact(x, y, x, y) for x in range(x0, x1 + 1)]
        assert sum(r[4] for r in out) == len(cells), (out, cells)
        return out

    def _baton(self, t: int) -> bool:
        """rev 62: the one-slot verify on the shard with token lanes (GLM53_MLA_DSA_BATON=1): ix_dm1's lane baton"""
        return (bool(getattr(self, "_one_slot", False)) and getattr(self, "_shard_now", None) is not None
                and os.environ.get("GLM53_MLA_DSA_BATON", "0") == "1" and self.att_lanes(t) > 1)  # fmt: skip

    def _ix_cores(self, t: int) -> list:
        nx = self.ix_lanes(t)
        return list(self.co_cores[:nx]) if nx > 1 else [self.ix_core]

    def _ix_nocs(self, t: int) -> list:
        return [(int(w.x), int(w.y)) for w in (self.device.worker_core_from_logical_core(c) for c in self._ix_cores(t))]

    def _index_core(self, t, y_ix, wsc, cur_pos, st, hist, mbox, n_r, pa=None, shard=None, page_table=None):
        """The DSA index core(s) (kernels/mla/ix_*.cpp): LayerNorm_128(k) -> slot t's ring row, the pooled key, wsc -> the
        owners (SEM_IXR); its BRISC then coordinates the owners' top-512 radix select (SEM_H / SEM_SEL, mailbox). Rev 47:
        ix_lanes(t) > 1 -> one per token lane (its tokens, its owners, its histogram slots)."""
        cores = self._ix_cores(t)
        nx = len(cores)
        cl = self.chunks // nx
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        rev = [("MLA_REV", str(MLA_REV)), ("MLA_SRC", "0x" + mla_src()[:8])]
        tm = (
            shard_tl(t) if shard is not None else dsa_tmax(t)
        )  # (rev 60: the shard mailbox is lane-local, dsa_shard.hpp TL)
        if tm != DSA_T_MAX:
            rev.append(("MLA_DSA_TMAX", str(tm)))
        if nx > 1:  # rev 47: the lane index cores (att_common.hpp LANES: token stride)
            rev.append(("MLA_ATT_LANES", str(nx)))
        elif self.att_lanes(t) > 1:  # rev 40: the lanes' own coordinators run the select (dsa_coord.cpp)
            rev.append(("MLA_DSA_COORD_EXT", "1"))
        if os.environ.get("GLM53_MLA_DSA_IX_ROWS", "1") == "1":  # rev 46 (card0 u42: 20 us per row of full-tile RMWs)
            rev.append(("MLA_DSA_IX_ROWS", "1"))
        baton = bool(st.get("one_slot")) and shard is not None and self._baton(t)
        if st.get("one_slot"):  # rev 56: a speculative verify (needs the row-granular writes on one index core)
            if ("MLA_DSA_IX_ROWS", "1") not in rev or (nx > 1 and not baton):
                raise ValueError(
                    "MkMla(dsa=...) verify: GLM53_MLA_DSA_IX_ROWS=1 and one index core (or the lane baton)"
                )
            rev.append(("MLA_DSA_ONE_SLOT", "1"))
            if baton:  # rev 62: the lanes' ring / pooled-key writes in row order (ix_dm1.cpp)
                assert nx == self.att_lanes(t) > 1, (nx, self.att_lanes(t))
                rev += [("MLA_DSA_ONE_SLOT_BATON", "1"), ("MLA_DSA_BATON_SEM", str(SEM_BATON)),
                        ("MLA_DSA_DONE_SEM", str(SEM_DONE))]  # fmt: skip
        if os.environ.get("GLM53_MLA_DSA_HSUM", "1") == "0":  # rev 52: the coordinator's old histogram sum (A/B)
            rev.append(("MLA_DSA_HSUM_OLD", "1"))
        if shard is not None:  # kernels/mla/dsa_shard.hpp: the coordinator merges the gathered candidates
            assert nx == self.att_lanes(t), "shard: one index core per token lane (rev 60)"
            if os.environ.get("GLM53_MLA_DSA_MCAST", "0") == "1":  # its args sit where the shard block is (SH0 = MC0)
                raise ValueError("MkMla(dsa_shard=True): GLM53_MLA_DSA_MCAST is not supported (ix_dm0 arg layout)")
            rev.append(("MLA_DSA_SHARD", "1"))
            rev += _shard_line_defs(shard)
        ring, pools, const = st["ring"], st["pools"], self.dsa.const
        paged = bool(st.get("paged"))
        if paged:  # rev 63 (ix_dm1.cpp MLA_DSA_PAGED_POOLS): pool q -> its block's page through the LOCAL page table
            assert shard is not None and page_table is not None, "paged pooled keys: the shard's local page table"
            rev.append(("MLA_DSA_PAGED_POOLS", "1"))
        lane = lambda l: [l] if nx > 1 else []  # noqa: E731
        rt0, rtc, rt1 = {}, {}, {}
        n_own = 0
        idx_of = lambda l: (
            [g * self.chunks + l * cl + k for g in range(self.groups) for k in range(cl)]
            if nx > 1
            else list(range(len(self.owner_noc)))
        )  # noqa: E731  # fmt: skip
        # rev 50 (GLM53_MLA_DSA_MCAST): the I5 broadcast as multicasts onto exact owner rectangles (when ix_dm0 is the
        # coordinator); off when any coordinator's owners need more than MCAST_MAX rectangles
        mc = None
        if os.environ.get("GLM53_MLA_DSA_MCAST", "0") == "1" and (nx > 1 or self.att_lanes(t) == 1):
            mc = [self._owner_rects(idx_of(l)) for l in range(nx)]
            if max(len(r) for r in mc) > MCAST_MAX:
                mc = None
        if mc is not None:
            rev.append(("MLA_DSA_MCAST", "1"))
        # rev 53 (review S2): what actually ran (the knob drops silently past MCAST_MAX rectangles)
        self.last_mcast = None if mc is None else [len(r) for r in mc]
        mcast = lambda l: (
            [] if mc is None else [len(mc[l])] + [v for r in mc[l] for v in r] + [0] * (5 * (MCAST_MAX - len(mc[l])))
        )  # noqa: E731  # fmt: skip
        # rev 54: every probe row a kernel flushes must lie inside the probe tensor (rev 53 overflowed it: card0 u72 hang)
        assert pa is None or 221 + 2 * (len(cores) - 1) < PROBE_ROWS, (len(cores), PROBE_ROWS)
        for l, c in enumerate(cores):
            xy = (int(c.x), int(c.y))
            own = [v for i in idx_of(l) for v in self.owner_noc[i]]
            n_own = len(own) // 2
            rt0[xy] = [y_ix.buffer_address(), const.buffer_address(), cur_pos.buffer_address(), t, hist.buffer_address(),
                       mbox.buffer_address()] + own + lane(l) + mcast(l)  # fmt: skip
            rtc[xy] = [t] + lane(l)
            rt1[xy] = [*dsa_slot_addrs(st, t)[::-1], cur_pos.buffer_address(), wsc.buffer_address(), t] + own + lane(l)
            if shard is not None:
                # ix_dm0 SH0 = 6 + 2 NOWN: rank, C, x_ag, y_ag, the AG lane + its wait semaphore, the AG consumer
                # semaphore [, TP8: the q-assembly semaphore]; ix_dm1 after its owner list: rank, C [, TP8: y_ag, q_asm,
                # the AG consumer + q-assembly semaphores]; then the probe pair (rev 59: the shard block first). The rank
                # slots are patched per chip by tt/tp_dsa_shard.py
                tp8 = int(shard.get("line", 1)) > 1
                self.shard_slots.setdefault("ix_dm0.cpp", []).append((len(rt0[xy]), None, {xy}, [(4, 5)]))
                rt0[xy] += [0, int(shard["C"]), shard["x_ag"].buffer_address(), shard["y_ag"].buffer_address(),
                            *shard["ag_lane"], SEM_FSW, SEM_AGC] + ([SEM_QASM] if tp8 else [])  # fmt: skip
                bt = []  # rev 62: the next lane's index core, then every lane's (DONE), at offsets 2 / 6 from the rank
                if baton:
                    nocs = self._ix_nocs(t)
                    bt = list(nocs[(l + 1) % nx]) + [v for xy_ in nocs for v in xy_]
                b0 = 6 if tp8 else 2
                pairs = [(b0 + 2 * k, b0 + 2 * k + 1) for k in range(len(bt) // 2)]
                self.shard_slots.setdefault("ix_dm1.cpp", []).append((len(rt1[xy]), None, {xy}, pairs))
                rt1[xy] += [0, int(shard["C"])] + ([shard["y_ag"].buffer_address(), shard["q_asm"].buffer_address(),
                                                    SEM_AGC, SEM_QASM] if tp8 else []) + bt  # fmt: skip
            if paged:  # rev 63: the local page table's address after the shard / baton blocks (MLA_DSA_PT_ARG)
                pt_arg = len(rt1[xy])
                rt1[xy] += [page_table.buffer_address()]
            sh0, sh1 = len(rt0[xy]), len(rt1[xy])  # (the probe pair's index: after every other arg)
            rt0[xy] += [pa, 220 + 2 * l] if pa is not None else []
            rt1[xy] += [pa, 221 + 2 * l] if pa is not None else []  # (rev 53, review S4: a row pair per lane core)
        lk = lambda src, ct, rt, cfg, defs=(): ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "mla", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct, runtime_args=_rt_args(rt), defines=rev + list(defs), config=cfg,
        )  # fmt: skip
        # rev 45: the index core's probe marks (probe rows 220 / 221 -- every lane core writes them; probe.hpp args after
        # the owner list and the lane)
        nl_ = 1 if nx > 1 else 0
        nm_ = 0 if mc is None else 1 + 5 * MCAST_MAX
        p0 = [("MK_PROBE_ARG", str(sh0)), ("MK_PROBE_CB", str(PROBE_CB))] if pa is not None else []
        p1 = [("MK_PROBE_ARG", str(sh1)), ("MK_PROBE_CB", str(PROBE_CB))] if pa is not None else []
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb_ in [IX["Y"], IX["C"]] + list(range(IX["T0"], IX["T0"] + 6)):
            modes[cb_] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kernels = [
            lk("ix_dm0.cpp", [SEM_IX, n_r, dsa_tmax(t), SEM_H, SEM_SEL, cl, n_own] + acc(y_ix) + acc(const) + acc(cur_pos)
               + (acc(shard["x_ag"]) if shard is not None else []),  # (rev 60, review 2 item 8: no y_ag accessor)
               rt0, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default), p0),
            lk("ix_c.cpp", [dsa_tmax(t), f32_bits(1.0 / 128), f32_bits(1.0e-6), f32_bits(2.0**-6)], rtc, cc),
            lk("ix_dm1.cpp", [st["tp"] // TILE, SEM_IXR, n_own] + acc(ring) + acc(pools) + acc(cur_pos) + acc(wsc)
               + (acc(shard["y_ag"]) + acc(shard["q_asm"]) if shard is not None and int(shard.get("line", 1)) > 1 else [])
               + (acc(page_table) if paged else []),
               rt1, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
               p1 + ([("MLA_DSA_PT_ARG", str(pt_arg))] if paged else [])),
        ]  # fmt: skip
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(
            total_size=page * n, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # fmt: skip
        t0 = IX["T0"]
        cbs = [cb(IX["Y"], FP32, 4096, 9), cb(IX["C"], FP32, 4096, 13), cb(IX["ROW"], BF16, 2048, 8),
               cb(IX["W"], FP32, 4096, 1), cb(IX["RING"], BF16, 2048, 8), cb(IX["POOL"], BF16, 2048, 4),
               cb(IX["INFO"], ttnn.uint32, DSA_POS_BYTES, 1), cb(IX["SCR"], ttnn.uint32, 4 * 2048 + 128, 1),
               cb(t0, FP32, 4096, 1), cb(t0 + 1, FP32, 4096, 4), cb(t0 + 2, FP32, 4096, 1), cb(t0 + 3, FP32, 4096, 4),
               cb(t0 + 4, FP32, 4096, 4), cb(t0 + 5, FP32, 4096, 4), cb(IX["CSCR"], ttnn.uint32, 512, 1)]  # fmt: skip
        if pa is not None:  # ix_dm0 flushes its marks at [0, 64), ix_dm1 at [64, 128)
            cbs.append(cb(PROBE_CB, ttnn.uint32, 128, 1))
        return kernels, cbs

    def _owner_kernels(self, roles, proles, att, pa, t: int) -> list:
        """mhc_pre's K-role kernels (stream on every K core; sync / compute per leader / member group) composed with
        the attention phase and mhc_post (kernels/mla/owner_{dm0,dm1,c}.cpp)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import _cores_of, _role

        k_of = {(int(c.x), int(c.y)): k for k, c in enumerate(self.place["k"])}
        specs = [(_role(roles, "stream.cpp", "MHC_K"), "reader", "owner_dm0.cpp")]
        for lead in (True, False):
            sr = _role(roles, "sync.cpp", "MHC_LEADER") if lead else _role(roles, "sync.cpp", "MHC_K", "MHC_LEADER")
            cr = (_role(roles, "compute.cpp", "MHC_LEADER") if lead
                  else _role(roles, "compute.cpp", "MHC_K", "MHC_LEADER"))  # fmt: skip
            specs += [(sr, "writer", "owner_dm1.cpp"), (cr, "compute", "owner_c.cpp")]
        kernels = []
        for mrole, name, src in specs:
            mine = sorted(_cores_of(mrole["cores"]))
            pct, prt, pdefs, pcfg = proles[name]
            act, art = att[name]
            parts = [(list(mrole["ct"]), {xy: list(mrole["rt"].get(xy, [])) for xy in mine}),
                     (act, {xy: art[xy] for xy in mine}),
                     (list(pct), {xy: [int(v) for v in prt[xy]] for xy in mine})]  # fmt: skip
            ct, rt, bases = concat_args(parts, mine)
            defs = list(mrole["defines"]) + [d for d in pdefs if d not in mrole["defines"]]
            defs += [("MK_PA_CT", str(bases[1][0])), ("MK_PA_RT", str(bases[1][1])), ("MK_PP_CT", str(bases[2][0])),
                     ("MK_PP_RT", str(bases[2][1])), ("MLA_REV", str(MLA_REV)), ("MLA_SRC", "0x" + mla_src()[:8])]  # fmt: skip
            if self.dsa is not None:
                defs.append(("MLA_DSA", "1"))
                # the 32-row mailbox (kernels/mla/dsa_common.hpp T_MAX; rev 60: the shard mailbox is lane-local, its
                # TL = shard_tl(t) lane tokens: dsa_shard.hpp)
                tm = shard_tl(t) if getattr(self, "_shard_now", None) is not None else dsa_tmax(t)
                if tm != DSA_T_MAX:
                    defs.append(("MLA_DSA_TMAX", str(tm)))
                if (
                    getattr(self, "_shard_now", None) is not None
                ):  # kernels/mla/dsa_shard.hpp (att_dsa_shard_*.cpp, att_c.cpp)
                    defs.append(("MLA_DSA_SHARD", "1"))
                    defs += _shard_line_defs(self._shard_now)
                    defs += _win_defs(self._shard_now)
                    if (
                        name != "compute"
                    ):  # the rank / peer-block slots of this role's attention args (the hook patches them)
                        # per KERNEL GROUP (the leaders' and the members' composites have different mhc_pre arg
                        # widths -> different attention bases: d0 card0 2026-10-22, one slot per source patched the
                        # leaders' wrong word): (rank slot, peer slot, this group's cores, the NoC-target offsets)
                        ra = bases[1][1]
                        w0 = att.get("writer_shard0", 0)
                        tp8 = int(self._shard_now.get("line", 1)) > 1  # + the RS round-2 lane (15, 16)
                        ent = ((ra + 11, ra + 13, set(mine), []) if name == "reader"
                               else (ra + w0, ra + w0 + 2, set(mine), [(6, 7), (9, 10)] + ([(15, 16)] if tp8 else [])))  # fmt: skip
                        self.shard_slots.setdefault(src, []).append(ent)
                    if name == "writer" and not self.dsa_compact():
                        raise ValueError("shard v0 needs the compacted candidates (GLM53_MLA_DSA_COMPACT=1)")
                if name == "writer" and self.dsa_compact():  # rev 36 (att_dsa_dm1.cpp: the candidates after the keys)
                    defs += [("MLA_DSA_COMPACT", "1"), ("MLA_DSA_KEYS", str(self._dsa_k))]
                if self.dsa_sb() > 1:  # rev 37: the reader (att_dsa_dm0) and the compute (att_c score) batch together
                    defs.append(("MLA_DSA_SB", str(self.dsa_sb())))
                if name == "reader" and os.environ.get("GLM53_MLA_DSA_RUNS", "1") == "1":  # rev 48: S1 gather runs
                    defs.append(("MLA_DSA_RUNS", "1"))
                if name == "reader" and getattr(self, "_paged_now", False):  # rev 63: block-addressed pooled keys
                    defs.append(("MLA_DSA_PAGED_POOLS", "1"))
            if self.att_lanes(t) > 1:  # token lanes (kernels/mla/att_common.hpp)
                defs.append(("MLA_ATT_LANES", str(self.att_lanes(t))))
            if name == "reader" and os.environ.get("GLM53_MLA_ATT_VERIFY") == "1":
                # Stage 17 speculative verify (set around a verify / MTP catch-up launch only; tt/spec_mtp.py): the rows
                # are consecutive positions of one user -> the latent hand-off before any tile >= row 0's (att_dm0.cpp);
                # with DSA (rev 56) every row scores the launch's slot (att_dsa_dm0.cpp waits for the whole latent write
                # and for every row's index writes before its first sparse row)
                assert t <= 8, ("MLA_ATT_VERIFY: t <= 8", t)  # (t > 4: the lanes path, causal per lane core)
                defs.append(("MLA_DSA_ONE_SLOT", "1") if self.dsa is not None else ("MLA_ATT_VERIFY", "1"))
            if self.att_nkmax(t) != ATT_NKMAX:  # rev 32: every owner kernel sizes its batches by it (att_common.hpp)
                defs.append(("MLA_ATT_NKMAX", str(self.att_nkmax(t))))
            if name == "writer" and self.ot(t):  # rev 30: the roots write o_T (att_dm1.cpp; real heads per group)
                defs.append(("MLA_ATT_OT", str(min(TILE, self.attn.h))))
            if name == "compute":  # O tiles per DST acquire in the attention compute (att_c.cpp OB: 1, 2 or 4)
                defs.append(("MLA_ATT_OB", os.environ.get("GLM53_MLA_ATT_OB", "4")))
                fq = os.environ.get("GLM53_MLA_ATT_FID_QK")  # rev 31: S = Q K^T / O += E V only (HiFi2 / LoFi / ...)
                if fq:
                    defs.append(("MLA_ATT_FID_QK", str(int(getattr(ttnn.MathFidelity, fq).value))))
            if pa is not None and name == "writer":  # the owner NCRISC's marks -> slot 132 + k
                n_all = len(next(iter(rt.values())))
                rt = {xy: v + [pa, 132 + k_of[xy]] for xy, v in rt.items()}
                defs += [("MK_PROBE_ARG", str(n_all)), ("MK_PROBE_CB", str(PROBE_CB))]
            cfg = mrole["config"]
            if name == "compute":
                cfg = attention_compute_config(mrole["config"], ACB1)
                modes = list(cfg.unpack_to_dest_mode)
                pmodes = list(pcfg.unpack_to_dest_mode)
                modes = [a_ if a_ != ttnn.UnpackToDestMode.Default else b_ for a_, b_ in zip(modes, pmodes)]
                cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "mla", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=mrole["cores"], compile_time_args=ct, runtime_args=_rt_args(rt), defines=defs,
                config=cfg))  # fmt: skip
        return kernels

    def _norm_core(self, cores, kx, koff, nw, qk, out, cache, page_table, cur_pos, t, sig_sem, consumers, pa, prows,
                   n_r, inv_only: bool = False, shard: dict | None = None):  # fmt: skip
        """Norm cores (kernels/mla/lat_*.cpp), ONE kernel group: out = RMSNorm(qk tiles [koff, koff + kx)) (bf16), [+ the
        paged cache write when `cache` is given], then sig_sem on every consumer. The latent core (kv_a, cache, -> the
        attention owners) and the q-norm cores (q_a -> q_r, -> the q_b readers): core i writes output tiles
        [i on, (i + 1) on) (on = kx / len(cores); the slice offset is a runtime arg, the row scale from all kx tiles).
        Probe rows of core i: prows + 2 i."""
        attn = self.attn
        if isinstance(cores, ttnn.CoreCoord):
            cores = [cores]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        xys = [(int(c.x), int(c.y)) for c in cores]
        on = 0 if inv_only else kx // len(cores)  # 0: the inv-only q-norm core (one fp32 tile out)
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        rev = [("MLA_REV", str(MLA_REV)), ("MLA_SRC", "0x" + mla_src()[:8])]
        flat = [v for c in consumers for v in c]
        p0 = [("MK_PROBE_ARG", "3"), ("MK_PROBE_CB", str(PROBE_CB))] if pa is not None else []
        p1 = [("MK_PROBE_ARG", str(10 + len(flat))), ("MK_PROBE_CB", str(PROBE_CB))] if pa is not None else []
        probe = lambda i, r: [pa, r + 2 * i] if pa is not None else []  # core i: rows prows + 2 i
        lk = lambda src, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "mla", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=ct, runtime_args=_rt_args(rt), defines=defs + rev, config=cfg,
        )  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb_ in (LCB_NW, LCB_INV):
            modes[cb_] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        pt_row_bytes = int(page_table.shape[-1]) * 4
        block = attn.block_size if cache is not None else 0
        # the latent core's batched cache write (kernels/mla/lat_dm1.cpp MLA_LAT_NB, t > 4: GLM53_MLA_LAT_NB, default 4)
        nb = int(os.environ.get("GLM53_MLA_LAT_NB", "4")) if cache is not None and t > DSA_T_MAX else 1
        nb = max(1, min(nb, t))
        lat_nb = [("MLA_LAT_NB", str(nb))] if nb > 1 else []
        ct_cache = cache if cache is not None else out  # an accessor of some tensor (unused without a cache)
        # the shard variant (kernels/mla/dsa_shard.hpp): the latent core copies q_lat (L1) into the all-gather block
        # x_ag once W_uk's readers signalled (SEM_QL, n_r), writes the cache row on the owner chip only, then
        # increments the AG lane's wait semaphore; its args after the consumer pairs: rank, C, the lane, its semaphore
        ql_sem, ql_n = (SEM_QL, n_r) if shard else (SEM_UK, 0)
        ql_src, ql_dst = (shard["q_lat"], shard["x_ag"]) if shard else (out, out)
        ql_tiles = int(shard["ql_tiles"]) if shard else 0
        sh_rt = [0, int(shard["C"]), *shard["ag_lane"], SEM_FSW] if shard else []
        sh_defs = ([("MLA_DSA_SHARD", "1")] + _shard_line_defs(shard) + _win_defs(shard)) if shard else []
        win = shard.get("window") if shard else None
        if win is not None:  # lat_dm1.cpp: the window's address after the shard block, its accessor after q_lat's
            sh_rt = sh_rt + [shard["window_a"]]
        if shard:  # (rev 60: t > 4 too -- lat_dm1.cpp's batched write skips the rows other chips own)
            # rev 59: the probe pair after the shard block (lat_dm1.cpp reads the shard args at 10 + 2 NCONS)
            p1 = (
                [("MK_PROBE_ARG", str(10 + len(flat) + len(sh_rt))), ("MK_PROBE_CB", str(PROBE_CB))]
                if pa is not None
                else []
            )
            self.shard_slots.setdefault("lat_dm1.cpp", []).append(
                (10 + 2 * len(consumers), None, set(xys), [(2, 3)])
            )  # (the q-norm cores run the
            # same source without the shard block: the hook patches the latent core only)
        kernels = [
            lk("lat_dm0.cpp", [LCB_X, LCB_NW, kx, koff, SEM_QA, n_r, on] + acc(qk) + acc(nw),
               {xy: [qk.buffer_address(), nw.buffer_address(), i * on] + probe(i, prows[0])
                for i, xy in enumerate(xys)}, p0,
               dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            lk("lat_c.cpp", [LCB_X, LCB_NW, LCB_INV, LCB_OUT, kx, f32_bits(1.0 / (kx * TILE)), f32_bits(attn.eps), on],
               {xy: [i * on] for i, xy in enumerate(xys)}, [], cc),
            lk("lat_dm1.cpp", [LCB_OUT, LCB_SCR, kx, block, ql_sem, ql_n, sig_sem, len(consumers), on] + acc(out)
               + acc(ct_cache) + acc(page_table) + acc(cur_pos) + acc(ql_src) + acc(ql_dst) + (acc(win) if win is not None else []),
               {xy: [out.buffer_address(), ct_cache.buffer_address(), page_table.buffer_address(),
                     cur_pos.buffer_address(), t, pt_row_bytes, ql_src.buffer_address(), ql_dst.buffer_address(), ql_tiles,
                     i * on] + flat + sh_rt + probe(i, prows[1]) for i, xy in enumerate(xys)}, p1 + lat_nb + sh_defs,
               dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
        ]  # fmt: skip
        lcb = lambda i, dt, page, n: ttnn.CBDescriptor(
            total_size=page * n, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # fmt: skip
        scr_bytes = (nb * kx * 2048 + 256 + (t if nb > 1 else 1) * _a64(pt_row_bytes)) if cache is not None else 64
        cbs = [lcb(LCB_X, BF16, 2048, kx), lcb(LCB_NW, FP32, 4096, max(on, 1)), lcb(LCB_INV, FP32, 4096, 1),
               lcb(LCB_OUT, FP32, 4096, 1) if inv_only else lcb(LCB_OUT, BF16, 2048, on),
               lcb(LCB_SCR, ttnn.uint32, scr_bytes, 1)]  # fmt: skip
        if pa is not None:
            cbs.append(lcb(PROBE_CB, ttnn.uint32, 128, 1))
        return kernels, cbs


def concat_args(parts: list, cores: list):
    """Phase-composed kernel args: parts = [(ct, {core: rt})] in phase order -> (ct, {core: rt}, [(ct base, rt base)])
    with every phase's runtime args padded to one width (the next phase's base is a compile-time define)."""
    ct, rt, bases = [], {xy: [] for xy in cores}, []
    for pct, prt in parts:
        bases.append((len(ct), len(rt[cores[0]])))
        ct += list(pct)
        width = max(len(prt.get(xy, [])) for xy in cores)
        for xy in cores:
            v = [int(a) for a in prt.get(xy, [])]
            rt[xy] += v + [0] * (width - len(v))
    return ct, rt, bases


def attention_args(
    attn,
    owners,
    owner_noc,
    G,
    C,
    t,
    q_lat,
    cache,
    page_table,
    cur_pos,
    o_lat,
    mask,
    reader_noc,
    acb: dict,
    waits: tuple,
    dsa: dict | None = None,
    lanes: int = 1,
    o_t=None,
) -> dict:
    """(dsa["shard"] (tt/tp_dsa_shard.py): the split-K blocks of att_dsa_shard_dm0 / dm1 / att_c, after every other arg).
    The attention phase's args per RISC role: {"reader" (att_dm0) / "writer" (att_dm1) / "compute" (att_c):
    (ct, {(x, y): rt})} for owners [k] = head group k // C, row chunk k % C. waits = (q_lat sem, count, latent-write
    sem, count) of the one-program half ((0, 0, 0, 0): both already in place). dsa (MkMla with a DsaIndexer): the
    att_dsa_*.cpp args (n_r, tp_tiles, q_ix, wsc, pools, ix_noc, hist, mbox). lanes > 1 (the
    MLA_ATT_LANES define; DSA too since rev 40, one select coordinator per lane): the C owners of a group form `lanes` token lanes of C / lanes chunk cores (lane l = row chunks
    [l C / lanes, (l + 1) C / lanes), tokens l, l + lanes, ...), each with its own partial tree; the kernels get the
    lane-local chunk index / count / tree depth and the lane as their last runtime arg."""
    acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())
    assert lanes >= 1 and C % lanes == 0 and lanes & (lanes - 1) == 0, (C, lanes)
    assert not (dsa and lanes > 1 and not dsa.get("coord_noc")), "DSA lanes: one coordinator per lane (rev 40)"
    G_C, C = C, C // lanes  # C from here on: chunk cores per token (the lane's tree)
    LV = C.bit_length() - 1
    hpt = -(-attn.h // TILE)
    cores = [(int(c.x), int(c.y)) for c in owners]
    dm0_ct = [acb["AQ"], acb["AK"], acb["AMASK"], acb["AINFO"], acb["ASCR"], C, hpt, attn.block_size] + list(waits)
    dm0_ct += [acb["AID"]]
    if dsa:
        dm0_ct += [DCB["AQI"], DCB["AWS"], DCB["APK"], DCB["ASEL"], SEM_IXR, SEM_QI, dsa["n_r"], dsa["tp_tiles"]]
    dm0_ct += acc(q_lat) + acc(cache) + acc(page_table) + acc(cur_pos) + acc(mask)
    shard = dsa.get("shard") if dsa else None
    tp8 = bool(shard) and int(shard.get("line", 1)) > 1  # kernels/mla/dsa_shard.hpp LINE > 1: the 2x4 exchange
    # the owners' q source of the shard passes: y_ag (TP2: the peer's whole block) / q_asm (TP8, rev 59: assembled once
    # per chip by the index core)
    q_src = (shard["q_asm"] if tp8 else shard["y_ag"]) if shard else None
    if dsa:
        dm0_ct += acc(dsa["q_ix"]) + acc(dsa["wsc"]) + acc(dsa["pools"])
    if shard:  # (rev 60: token lanes and o_T too; one head group per chip)
        assert G == 1, "shard: one head group per chip"
        dm0_ct += acc(q_src)
        if shard.get("window") is not None:  # the dense window's accessor after the q source's
            dm0_ct += acc(shard["window"])
    dm1_ct = [acb["ARX"], acb["ASEND"], acb["AOUT"], C, LV, hpt, SEM_RDY, SEM_DAT, SEM_OL, len(reader_noc)]
    dm1_ct += [acb["ARXO"], acb["ASENDO"], SEM_DATO]
    if dsa:
        dm1_ct += [DCB["ASC"], DCB["ASEL"], DCB["AKEY"], SEM_SEL, SEM_H]
    if shard:
        dm1_ct += [SEM_RSC]
    dm1_ct += acc(o_lat) + (acc(cur_pos) if dsa else [])
    dm1_ct += acc(o_t) if o_t is not None else []  # rev 30 (MLA_ATT_OT): after o_lat's (DSA, rev 39: cur_pos's) args
    if shard:  # (rev 59: + y_ag for the merge helpers)
        dm1_ct += acc(shard["x_ag"]) + acc(shard["x_rs"]) + acc(shard["y_rs"]) + acc(shard["y_ag"])
        dm1_ct += (acc(shard["x_rs2"]) + acc(shard["y_rs2"])) if tp8 else []
    c_ct = [acb[k] for k in ("AQ", "AK", "AMASK", "AINFO", "AS", "AM", "AP", "AE", "APA", "APB", "ARX", "ACOEF", "ASEND",
                             "AOUT")]  # fmt: skip
    c_ct += [C, LV, f32_bits(attn.scale), f32_bits(-1.0e30)]
    c_ct += [acb[k] for k in ("APAO", "APBO", "ARXO", "ASENDO", "ADIAG", "AID")]
    if dsa:
        c_ct += [DCB[k] for k in ("AQI", "AWS", "APK", "ASR", "ASC")]
        dsa_rt0 = [dsa["q_ix"].buffer_address(), dsa["wsc"].buffer_address(), dsa["pools_a"]]
        if shard:  # rank, C, peer block, the q source (rank / peer patched per chip by TPMlaShard)
            dsa_rt0 += [0, int(shard["C"]), 1, q_src.buffer_address()]
        dsa_rt1 = [cur_pos.buffer_address(), *dsa["ix_noc"], dsa["hist"].buffer_address(), dsa["mbox"].buffer_address()]
    dm0_rt, dm1_rt, c_rt = {}, {}, {}
    flat_r = [v for xy in reader_noc for v in xy]
    lane_arg = lambda lane: [lane] if lanes > 1 else []
    for k, xy in enumerate(cores):
        g, cf = divmod(k, G_C)
        lane, c = divmod(cf, C)
        base = g * G_C + lane * C  # the lane's chunk core 0
        dm0_rt[xy] = [q_lat.buffer_address(), cache.buffer_address(), page_table.buffer_address(),
                      cur_pos.buffer_address(), mask.buffer_address(), g, c, t] + (dsa_rt0 if dsa else [])  # fmt: skip
        dm0_rt[xy] += lane_arg(lane)
        if shard and shard.get("window") is not None:  # the dense window after the lane (att_dsa_shard_dm0.cpp)
            dm0_rt[xy] += [shard["window_a"]]  # slot s0's base (the launch's rows are slots s0 ..)
        lv = []
        for L in range(LV):
            if c % (2 << L) == 0 and c + (1 << L) < C:
                lv += list(owner_noc[base + c + (1 << L)])
            elif c % (2 << L) == (1 << L):
                lv += list(owner_noc[base + c - (1 << L)])
            else:
                lv += [0, 0]
        dm1_rt[xy] = [o_lat.buffer_address(), g, c, t] + lv + (flat_r if c == 0 else [0] * len(flat_r))
        if dsa and lanes > 1:  # rev 40: the lane's coordinator (its histogram slots at the same address)
            dsa_rt1 = [cur_pos.buffer_address(), *dsa["coord_noc"][lane], dsa["hist"].buffer_address(),
                       dsa["mbox"].buffer_address()]  # fmt: skip
        dm1_rt[xy] += (dsa_rt1 if dsa else []) + lane_arg(lane) + ([o_t.buffer_address()] if o_t is not None else [])
        if (
            shard
        ):  # rank, C, peer block, x_ag, x_rs, y_rs, the AG lane + its wait semaphore, the RS lane + its wait semaphore
            writer_shard0 = len(dm1_rt[xy])
            dm1_rt[xy] += [0, int(shard["C"]), 1, shard["x_ag"].buffer_address(), shard["x_rs"].buffer_address(),
                           shard["y_rs"].buffer_address(), *shard["ag_lane"], SEM_FSW, *shard["rs_lane"], SEM_FSW,
                           shard["y_ag"].buffer_address()]  # fmt: skip
            if tp8:  # x_rs2, y_rs2, the RS round-2 (scatter) lane + its wait semaphore
                dm1_rt[xy] += [shard["x_rs2"].buffer_address(), shard["y_rs2"].buffer_address(), *shard["rs2_lane"],
                               SEM_FSW]  # fmt: skip
        c_rt[xy] = [c, t] + lane_arg(lane)
    out = dict(reader=(dm0_ct, dm0_rt), writer=(dm1_ct, dm1_rt), compute=(c_ct, c_rt))
    if shard:
        out["writer_shard0"] = writer_shard0
    return out


def dispatch_stats(kernels) -> dict:
    """what a launch of these kernels dispatches: kernel groups, core ranges, cores, runtime-arg words (per-core)"""
    ranges = cores = words = 0
    for k in kernels:
        rs = list(k.core_ranges.ranges())
        ranges += len(rs)
        for r in rs:
            for x in range(r.start.x, r.end.x + 1):
                for y in range(r.start.y, r.end.y + 1):
                    cores += 1
                    try:
                        words += len(k.runtime_args[x][y])
                    except Exception:  # noqa: BLE001
                        pass
    per = [(os.path.basename(str(k.kernel_source)), len(list(k.core_ranges.ranges()))) for k in kernels]
    return dict(kernel_groups=len(kernels), core_ranges=ranges, kernel_core_pairs=cores, rt_words=words, per=per)


def mask_bank(device):
    """[33 x 32, 32] fp32: tiles 0..31 = the additive masks (tile lim: columns > lim -> -1e30), tile 32 = the identity
    (kernels/mla/att_c.cpp's diagonal coefficient tiles)"""
    col = torch.arange(TILE, dtype=torch.float32)
    mask = torch.cat([torch.where(col > lim, -1.0e30, 0.0).repeat(TILE, 1) for lim in range(TILE)] + [torch.eye(TILE)])
    return ttnn.from_torch(mask, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def attention_compute_config(base_cfg, acb: dict):
    """the owners' compute config: HiFi4 / fp32 DST, the base config's unpack modes + the attention's fp32 CBs.
    GLM53_MLA_ATT_FIDELITY (HiFi2 / HiFi3 / LoFi; default HiFi4): the whole owner kernel's matmul fidelity -- the t = 32
    attention is compute-bound (vtp8 probe 2026-10-20: 282 us); numerics-gated (o_lat vs float64)"""
    fid = getattr(ttnn.MathFidelity, os.environ.get("GLM53_MLA_ATT_FIDELITY", "HiFi4"))
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True)
    modes = list(base_cfg.unpack_to_dest_mode)
    for key in ("AMASK", "AS", "AM", "AE", "APA", "APB", "ARX", "ACOEF", "AID"):  # O / ADIAG: matmul operands
        modes[acb[key]] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
    return cfg


def _sharded_u32(device, cores, n_cores: int, words: int):
    """an uninitialised uint32 L1 tensor of `words` per core, height-sharded over `cores` (the same L1 address on
    every core: the DSA mailbox / histogram slots)"""
    spec = ttnn.ShardSpec(cores, (1, words), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    return ttnn.empty((n_cores, words), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mc)


def attention_cbs(cores, acb: dict, page_table, dsa_keys: int = 0, dsa_cand: bool = False, dsa_sb: int = 1,
                  nk: int = ATT_NKMAX,
                  obf16: bool | None = None, shard_t: int = 0, shard_tl: int = DSA_T_MAX) -> list:  # fmt: skip
    """the owners' attention CBs (+ the DSA ones when dsa_keys > 0: the keys one owner holds per sparse token; shard_t > 0:
    the split-K variant -- shard_t = the tokens of one token lane (rev 60; t at t <= 4): shard_t CB_ASEL pages of
    (3 + min(512, keys)) words + 52 B (a core selects <= its kept pools), in CB_AKEY the region U = max(the keys +
    candidates, the merge helpers' scratch + 64: overlaid, review 2 item 1), shard_tl kept (key, pool) blocks of
    min(512, keys) pairs and the record staging; att_dsa_shard_dm1.cpp UB / KW / kept / stg)"""
    cb = lambda i, dt, page, n: ttnn.CBDescriptor(
        total_size=page * n, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
    )  # fmt: skip
    pt_bytes = int(page_table.shape[-1]) * 4
    # the O partials (matmul operands of the rescales, the tree's messages): fp32, or bf16 (GLM53_MLA_ATT_OBF16=1: half
    # the bytes per tree level; the operands are tf32 in the FPU either way)
    if obf16 is None:
        obf16 = os.environ.get("GLM53_MLA_ATT_OBF16", "0") == "1"
    odt, ob = (BF16, 2048) if obf16 else (FP32, 4096)
    return [cb(acb["AQ"], BF16, 2048, 16), cb(acb["AK"], BF16, 2048, nk * 16), cb(acb["AMASK"], FP32, 4096, 1),
            cb(acb["AINFO"], ttnn.uint32, 64, 1), cb(acb["AS"], FP32, 4096, nk), cb(acb["AM"], FP32, 4096, 1),
            cb(acb["AP"], BF16, 2048, nk), cb(acb["AE"], FP32, 4096, nk), cb(acb["APA"], FP32, 4096, 2),
            cb(acb["APB"], FP32, 4096, 2), cb(acb["ARX"], FP32, 4096, 2), cb(acb["ACOEF"], FP32, 4096, 2),
            cb(acb["ASEND"], FP32, 4096, 2), cb(acb["AOUT"], BF16, 2048, 16), cb(acb["APAO"], odt, ob, 16),
            cb(acb["APBO"], odt, ob, 16), cb(acb["ARXO"], odt, ob, 16), cb(acb["ASENDO"], odt, ob, 16),
            cb(acb["ADIAG"], FP32, 4096, 2), cb(acb["AID"], FP32, 4096, 1),
            cb(acb["ASCR"], ttnn.uint32, 256 + _a64(pt_bytes), 1)] + ([
            cb(DCB["AQI"], BF16, 2048, 4), cb(DCB["AWS"], FP32, 4096, 1), cb(DCB["APK"], BF16, 2048, 8 * dsa_sb),
            cb(DCB["ASR"], FP32, 4096, dsa_sb), cb(DCB["ASC"], FP32, 4096, 2 * dsa_sb),
            cb(DCB["ASEL"], ttnn.uint32, (3 + (min(DSA_TOPK, dsa_keys) if shard_t else DSA_TOPK)) * 4 + 52,
               max(1, shard_t)),
            cb(DCB["AKEY"], ttnn.uint32, _akey_bytes(dsa_keys, dsa_cand, shard_t, shard_tl), 1)]
           if dsa_keys else [])  # fmt: skip


def _akey_bytes(dsa_keys: int, dsa_cand: bool, shard_t: int, shard_tl: int = DSA_T_MAX) -> int:
    """CB_AKEY bytes per owner: [0, 64) histogram, the cur_pos page, the keys (+ the compacted candidates); with the shard
    variant (shard_t > 0; rev 60) the region U = max(keys + candidates, SHARD_MERGE_SCR + 64) (the merge helpers'
    scratch over the dead phase-A keys), shard_tl kept blocks of KW = min(512, keys) (key, pool) pairs, the 64-B
    aligned record staging (TB + 64) + 64 slack (att_dsa_shard_dm1.cpp UB / KW / kept / stg / mstg)"""
    keyb = 4 * dsa_keys * (2 if dsa_cand else 1)
    if not shard_t:
        return 64 + DSA_POS_BYTES + keyb
    assert dsa_cand and 1 <= shard_t <= shard_tl <= SHARD_TL_MAX, (dsa_cand, shard_t, shard_tl)
    kw = min(SHARD_CAND_MAX, dsa_keys)
    return (64 + DSA_POS_BYTES + max(keyb, SHARD_MERGE_SCR + 64) + shard_tl * 2 * kw * 4
            + 64 + 2048 + 64 + 64)  # fmt: skip


def attention_tree(p, t0: int, chunks: int, groups: int, mhz: float = 1350.0) -> dict:
    """per tree level of the attention combine (owner NCRISC rows 132 + k, rev 20 marks of token 0: start, per received
    level (child stats in, child O in), [partial ready to send], end): medians over the receivers / senders, us"""
    import statistics

    lv = chunks.bit_length() - 1
    recv_s, recv_o, send = ([[] for _ in range(lv)] for _ in range(3))
    for k in range(groups * chunks):
        c = k % chunks
        m = [(int(v) - t0) / mhz for v in p[132 + k] if int(v) > 0]
        i = 1
        for L in range(lv):
            if c % (2 << L) == 0:
                if i + 1 < len(m):
                    recv_s[L].append(m[i])
                    recv_o[L].append(m[i + 1])
                i += 2
            elif c % (2 << L) == (1 << L):
                if i < len(m):
                    send[L].append(m[i])
                break
    med = lambda v: round(statistics.median(v), 1) if v else None  # noqa: E731
    return dict(send_ready=[med(v) for v in send], child_stats_in=[med(v) for v in recv_s],
                child_o_in=[med(v) for v in recv_o])  # fmt: skip


def decode_probe(probe_host, n_readers: int, *, mhz: float = 1350.0, chunks: int = 0, groups: int = 0) -> dict:
    """The probe tensor [PROBE_ROWS, 16] (host) -> per-role marks in us after the earliest mark: per mark index the
    median (and max) over the cores of the role that recorded it, with the core count: owners (mhc_pre stream, rows
    0..63), root (64), readers' streamer (65 + 2i: start, phase ends), readers' mover (66 + 2i: start, after wait /
    after signal per phase), latent core dm0 / dm1 (130 / 131), owner NCRISC (132 + k: mhc_pre sync, attention start /
    end, mhc_post), q-norm core dm0 / dm1 (196 / 197)."""
    import statistics

    p = probe_host.reshape(-1, 16).long() & 0xFFFFFFFF
    nz = p[p > 0]
    t0 = int(nz.min()) if nz.numel() else 0
    us = lambda v: round((int(v) - t0) / mhz, 1)

    def role(rows, fn):
        cols = [[int(v) for v in p[r] if int(v) > 0] for r in rows if r < p.shape[0]]
        cols = [c for c in cols if c]
        n = max((len(c) for c in cols), default=0)
        return [us(fn([c[i] for c in cols if len(c) > i])) for i in range(n)]

    med = lambda rows: role(rows, statistics.median)
    mx = lambda rows: role(rows, max)
    rd_w, rd_a = range(65, 65 + 2 * n_readers, 2), range(66, 66 + 2 * n_readers, 2)
    return dict(owner=med(range(64)), owner_max=mx(range(64)), root=med([64]), reader_w=med(rd_w),
                reader_w_max=mx(rd_w), reader_a=med(rd_a), reader_a_max=mx(rd_a), latent_dm0=med([130]),
                latent_dm1=med([131]), owner_dm1=med(range(132, 196)), owner_dm1_max=mx(range(132, 196)),
                qnorm_dm0=med(range(196, 220, 2)), qnorm_dm1=med(range(197, 220, 2)),
                qnorm_dm1_max=mx(range(197, 220, 2)), ix_dm0=med(range(220, 236, 2)), ix_dm1=med(range(221, 236, 2)),
                ix_dm0_max=mx(range(220, 236, 2)), ix_dm1_max=mx(range(221, 236, 2)),
                **({"att_tree": attention_tree(p, t0, chunks, groups, mhz)} if chunks else {}))  # fmt: skip
