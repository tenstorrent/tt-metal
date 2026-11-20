# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention half of a Hy4-preview decoder layer for 8 token rows on the 8x4 mesh (PLAN.md has the plan, the
cores, the L1 arena and the wait graph; this builds it). Two programs under one trace:

    P1   root chip, 16 site cores   iHC pass (fused/ihc_site): post of the sublayer before (optional) + pre, the
                                    input RMSNorm on the same cores, x [8, 6144] BF16 (normalised) to the sender
         X1                         the root's sender -> every chip's pre hub (fused/layer_common/transport.py)
         every chip, pre group      fused/projections pre units: q_a's 2 tiles and their lane sums, the pool tile,
                                    the gate of the chip's 2 heads (kept on the chip for P2)
         Xa                         q_a tiles -> every chip's query hub x buffer; statistic tile -> its lane landing
         every chip, query group    the chips' lane sums added (Float32), then fused/projections query units: the
                                    absorbed queries of the chip's 2 heads, [2, 8 users, 576] BF16
         X3                         -> the landing core of the chip's head group on every chip
    P2   every chip, attention      fused/sparse_mla with LayerIo: every attention core assembles its query pages
                                    from its head group's landing, the rows come from DRAM by id, the final partial
                                    (O, max, sum) per (user, head group) stays at a fixed address
         X4                         column stage: the coordinator pulls the 8 users' partials of one head group into
                                    one of two staging slots, the sender sends them to the chip of that mesh row in
                                    the chip's column; a merge core per user merges what lands in arrival order; row
                                    stage: the column results of the 4 chips of a row; sink and division; the rows of
                                    the chip's 2 heads -> the post hub
         every chip, post group     fused/projections post units: value side of kv_b, gate, o_proj partial
         X5                         the transport's reduce to the root; F lands on the root's site cores

State (2026-11-20): both programs run on all 32 chips on the board, every multi-chip rung after a parse-only run of
its senders (tests/test_p1.py, test_p2.py), and together with checkpoint weights of layers 0 to 5 (tests/
test_layer.py): F against HF's attn_out PCC 0.9988 to 0.9999. PLAN.md section 12 has the numbers and the logs.
Not built: the latent append on the owner chips (the host writes the new token's row), the full-indexer layer (scan,
X2, cut: the selection table comes from the host), the post-mix of this half's F (the next program's leading pass).

Pieces, used through their interfaces:
  fused/projections  one ProjProgram per group for the whole mesh (cores=, hub=, rect=, hub_buffers=, weights=): its
                     units and hub in one rectangle of cores, its hub buffers ranges of this piece's arena, its
                     weight stream one mesh tensor with a shard per chip, the gate units their own done group. Its
                     hub kernel is replaced by a wrapper that includes proj_hub.cpp. Every chip gets a pool unit
                     (zero weights where the chip has no pool column), so the 32 programs differ in runtime
                     arguments only.
  fused/ihc_site     IhcSite.program(norm_out=...) on the root, as fused/layer_mlp has it.
  fused/collective   its sender kernel through layer_common.sender (explicit ops).

Precision: the iHC pass and the norm are fp32 (SFPU); x travels as BF16; the projections' matmuls are what
``fidelity`` says ("custom_mm": LoFi, the board-proven default of fused/projections; "recommended": hifi_mm at the
schedule that takes the operands whole) on BFP8 weights (layout.stored); the lane sums are Float32.
"""

from __future__ import annotations

import contextlib
import dataclasses

from pathlib import Path

import numpy as np
import torch

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.ihc_site import layout as ihc_layout
from models.experimental.hy4_preview.fused.ihc_site.op import IhcSite, NormOut, SiteRef
from models.experimental.hy4_preview.fused.kv_store import device as kv_device
from models.experimental.hy4_preview.fused.kv_store import frames as kv_frames
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.layer_common.sender import Sender, mesh_branches, unicast
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.op import LayerIo, SparseMla
from models.experimental.hy4_preview.reference.ihc import IHCConfig

KERNEL_DIR = Path(__file__).parent / "kernels"
HIDDEN = L.HIDDEN
ROW_TILE = P.ROW_TILE  # bytes of an [8, 32] BF16 tile
PAYLOAD = 2 * L.ROWS * HIDDEN  # x [8, 6144] BF16
CHIPS = L.CHIPS

# ---- cores (PLAN.md section 3) -------------------------------------------------------------------------------------
SENDER = (6, 8)
REDUCE = [(6, 4), (6, 5), (6, 6), (6, 7)]
PRE_RECT, PRE_HUB = ((0, 0), (3, 3)), (3, 3)
QUERY_RECT, QUERY_HUB = ((7, 0), (10, 3)), (10, 3)
MERGE = [(5, y) for y in range(8)]  # merge core of user y; landing core of query group y (P2 reads it)
SITE = dict(num_cores=16, origin=(5, 0), per_row=2, cb_base=32)
POST_RECT, POST_HUB = ((0, 8), (11, 9)), (5, 8)
POST_CORES = [(x, y) for y in (8, 9) for x in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)]  # 4 value units, 16 o_proj units
POST_O_CT = 12

# ---- the arena: one L1 tensor over the grid, each core uses its copy for its role (PLAN.md section 4) --------------
ARENA_BYTES = 307_200
# sender core
A_STAGING = 0  # the transport's staging (X1 on the root in P1, X5 in P2)
A_XA = 98_304  # q_a 2 tiles | pool tile | - | the statistic tile at + 2048, the first quarter of a zero Float32 page
A_X3 = 106_496  # the absorbed queries, 36 tiles
# reduce cores
A_LANDING1, A_LANDING2 = 0, 98_304
# pre hub
A_PRE_X, A_PRE_OUT, A_PRE_STAT, A_PRE_META, A_PRE_STATUS = 0, 99_328, 111_104, 112_128, 113_664
# query hub
A_Q_X, A_Q_OUT, A_Q_META, A_Q_STATUS, A_Q_LANES = 0, 33_792, 52_224, 53_760, 55_168
# merge cores
A_M_QUERY = 0  # 4 chips x 36 tiles: the absorbed queries of the head group, [chip][head][tile]
A_M_COL, A_M_ROW = 73_728, 143_360  # one source's 8 partials each: the column partner's, the row partner's
# sender core in P2: two column staging slots and the row staging behind the transport's staging
A_COL, A_ROW_STAGE = 98_304, 237_568
# post hub
A_POST_X, A_POST_OUT, A_POST_META, A_POST_STATUS = 0, 24_576, 122_880, 124_416
# attention cores: the final partial of (user = S-block, head group), [max / sum tile | 16 O tiles]
A_ATT_FINAL = 0
PART = 17 * ROW_TILE  # 8,704
GROUP_BYTES = L.ROWS * PART  # 69,632: the 8 users' partials of one head group
NONE = 0xFFFF
# circular buffers of a merge core (kernels/attn_merge_*.cpp)
M_ACC_O, M_ACC_MS, M_SEND_MS, M_SEND_O, M_SINK, M_FINAL, M_IN, M_T0_MS, M_T0_O, M_T1_MS, M_T1_O = range(11)
VIRT = (6, 9)  # the landing core of the sends to "virtual" partners (one chip with host-written partners)
STAT_PAGE = 4096  # a chip's statistic tile travels as the first quarter of a Float32 [32, 32] page
QUERY_OUT = L.HEADS_PER_CHIP * P.LAT_TILES * ROW_TILE  # 18,432

# ---- this piece's words (kernels/attn_common.hpp) -------------------------------------------------------------------
W_QA, W_LANES, W_POOL, W_QARR, W_COLARR, W_ROWARR, W_QGO, W_COLMASK, W_ROWMASK, W_LOCAL, W_CTX, W_SERVED = range(12)
W_ASTATUS, W_ARUNS, W_CGROUP, W_CROW = 18, 19, 20, 28
A_WORDS = 32
# zero after a run that completed (the served word keeps the last launch's bits: the coordinator zeroes it)
REST_WORDS = list(range(11)) + list(range(12, 19)) + list(range(20, 29))
CB_LAND, CB_SUM = 24, 25  # on the query hub core
# the latent append (PLAN.md section 17): the append core, its two arena ranges (the pool tiles' landing; the BF16
# rows it finished, kept for tests) and its buffers
APPEND = (11, 0)
A_AP_RAW, A_AP_LAT = 16_384, 32_768
AP_TILES = L.LATENT // L.TILE  # 18 pool tiles: 16 of k_pass, 2 of k_rope
(
    AP_RAW,
    AP_W,
    AP_CS,
    AP_MEAN,
    AP_ONE,
    AP_SQ,
    AP_STAT,
    AP_INV,
    AP_TMP,
    AP_LAT,
    AP_KEEP,
    AP_ROPE,
    AP_TILE,
    AP_ABS,
    AP_SCALE,
    AP_INVS,
    AP_E4M3,
    AP_ROWS,
    AP_POS,
    AP_FRAMES,
) = range(20)
KV_STORE_KERNELS = Path(__file__).parent.parent / "kv_store" / "kernels"
# the index-key append of a full-indexer layer (PLAN.md section 20): a core of its own beside the append core (its
# kernels run beside the latent row's, not behind them), its arena ranges (the 4 key tiles' landing; kept for tests:
# the BF16 keys before the Hadamard basis, 4 small tiles, and behind them the 4 packed BFP8 tiles) and its buffers
KEY = (11, 1)
A_KEY_RAW, A_KEY_KEEP = 16_384, 20_480
# (attn_key_write.cpp's keep area: the keys before the Hadamard basis | the packed tiles | 16 words | the blocks as
# read, 8 slots | the blocks as DRAM holds them after the pass, 8 slots)
K_ROT, K_PACKED, K_INFO, K_BEFORE, K_AFTER = 0, 2048, 6400, 6464, 6464 + 8 * 4352
# the key core's status word (W_ASTATUS) when two token rows of a pass were given the same stripe row of one block
# (attn_key_write.cpp): this code | u | v << 8. The pass completes; token row v's key is the one stored.
ST_KEY_SAME_ROW = 0xDE << 24
KEY_TILES = L.IDX_DIM // L.TILE  # 4
KEY_TILE_BYTES, KEY_BLOCK_BYTES = 1088, 4 * 1088  # a BFP8 tile; the 32-row block of a key store (4 tiles)
(
    KY_RAW,
    KY_SUM,
    KY_WB,
    KY_CS,
    KY_H,
    KY_MEAN,
    KY_STAT,
    KY_CEN,
    KY_SQ,
    KY_VAR,
    KY_INV,
    KY_TMP,
    KY_NW,
    KY_K,
    KY_ROT,
    KY_KEEP,
    KY_BFP,
    KY_BLOCKS,
    KY_POS,
    KY_FRAMES,
) = range(20)

# the index query of a full-indexer layer (PLAN.md section 21.3): 4 idx cores beside the append and the key core.
# Every chip's 4 index-query tiles (the pre group's output tiles 19..22: its index head's query for the token rows)
# land on the first of them with one more Xa op, chip 22's pool tile (the head weights) with its pool op; the 8
# data-movement processors of the 4 cores read one token row's pieces each from that landing into the scan's 4
# tiles and write them and the row's 32 multipliers into the scan's own DRAM tensors (kernels/attn_idx_dm.cpp).
IDX_CORES = [(11, 2), (11, 3), (11, 4), (11, 5)]
IDX = IDX_CORES[0]  # the landing core
W_IDX, W_IDXW, W_IDXGO = 0, 2, 10  # an idx core's words: chips' tiles landed, the head-weight tile, go to a processor
XA_IDX_TILE = 12  # where a chip's 4 index-query tiles sit in Xa's staging, in row tiles from its start
IDX_QUERY_BYTES = 4 * 2048  # a token row's query for the scan: 4 standard BF16 tiles
ST_IDX, ST_IDXW, ST_IDXGO = 0xDF << 24, 0xE0 << 24, 0xE1 << 24  # the idx core's status codes (attn_idx_dm.cpp)

# ---- 16 token rows a pass (PLAN.md section 19): two sub-passes of 8 with a row mask ---------------------------------
REDUCE16 = [(6, y) for y in range(8)]  # 8 reduce cores: a part of the doubled payload is what a part of 8 rows was
M_CTL = 11  # merge cores: the number of chains, data-movement kernel -> merge kernel (attn_merge16_*.cpp)
W_HAND = 32  # the sender core's hand-over words (collective_sender.cpp HY4_COLLECTIVE_HANDOVERS), 4 per op
WORDS16 = W_HAND + 4 * 16


class Layout:
    """Row-dependent sizes and the arena offsets of a layer with ``rows`` token rows a pass (8: the module's
    constants, which the assertions below hold it to; 16: PLAN.md section 19). ROW_TILE, PART and GROUP_BYTES stay
    what they are at 16 rows: a partial is [8 heads, 32] tiles per token row, a group's payload the partials of the
    8 rows of ONE sub-pass.

    16 rows, by role: the sender core has X5's staging (twice the bytes), the two column staging slots (used in
    turn by both sub-passes) and a row staging PER sub-pass; a merge core has the group's query landing (16-row
    tiles) and a column and a row landing slot PER sub-pass (written by other chips: nothing orders a partner's
    second payload behind this chip's use of the first); the reduce cores keep their landings' sizes because there
    are twice as many of them; the hubs' buffers hold [16, 32] tiles."""

    def __init__(self, rows, cap=None):
        """cap: the rows the OFFSETS are laid out for (default: rows). An 8-row program on a 16-row layout
        (rows=8, cap=16) has its own sizes at the 16-row offsets, so the two program sets of a layer share the arena
        and every persistent tensor (the serving loop picks the set by the pass's live rows)."""
        cap = rows if cap is None else cap
        assert rows in (L.ROWS, 2 * L.ROWS) and cap in (rows, 2 * L.ROWS), (rows, cap)
        self.rows, self.cap, self.subpasses = rows, cap, rows // L.ROWS
        self.row_tile = 2 * L.TILE * rows
        self.payload = 2 * rows * HIDDEN
        self.query_out = L.HEADS_PER_CHIP * P.LAT_TILES * self.row_tile
        self.reduce = REDUCE if rows == L.ROWS else REDUCE16
        t, k = 2 * L.TILE * cap, cap // L.ROWS
        payload, query_out = 2 * cap * HIDDEN, L.HEADS_PER_CHIP * P.LAT_TILES * t
        # sender core (P1: Xa's and X3's staging behind X1's; P2: X5's staging, the column slots, the row staging)
        self.staging = 0
        self.xa = payload
        self.x3 = self.xa + 8192 * k
        self.col = payload
        self.row_stage = self.col + 2 * GROUP_BYTES
        sender = self.row_stage + k * GROUP_BYTES
        # reduce cores (a part is payload / len(reduce) = 24,576 B at 8 and at 16 rows)
        self.landing1, self.landing2 = 0, 98_304
        reduce = self.landing2 + 8 * self.payload // len(self.reduce)
        # merge cores
        self.m_query = 0
        self.m_col = 4 * query_out
        self.m_row = self.m_col + k * GROUP_BYTES
        merge = self.m_row + k * GROUP_BYTES
        # pre hub
        self.pre_x = 0
        self.pre_out = P.PRE_X_TILES * t
        self.pre_stat = self.pre_out + (P.PRE_IDX + P.IDX_TILES) * t
        self.pre_meta = self.pre_stat + 2 * t
        self.pre_status = self.pre_meta + 1536
        # query hub
        self.q_x = 0
        self.q_out = P.QUERY_X_TILES * t
        self.q_meta = self.q_out + L.HEADS_PER_CHIP * P.LAT_TILES * t
        self.q_status = self.q_meta + 1536
        self.q_lanes = self.q_status + 1408
        query = self.q_lanes + CHIPS * STAT_PAGE
        # post hub
        self.post_x = 0
        self.post_out = P.POST_X_TILES * t
        self.post_meta = self.post_out + (HIDDEN // L.TILE) * t
        self.post_status = self.post_meta + 1536
        post = self.post_status + 4 * P.STATUS_WORDS
        # attention cores: one final partial per sub-pass
        self.att_final = 0
        # idx cores (a full layer with the index query): on the first the chips' index-query tiles [chip][tile] and
        # the head-weight tile; on each its two token rows' queries (4 standard tiles each) and their multiplier
        # words (64 B each)
        self.idx_land = 32_768
        self.idx_w = self.idx_land + CHIPS * P.IDX_TILES * t
        self.idx_work = -(-(self.idx_w + t) // 4096) * 4096
        self.idx_mult = self.idx_work + 2 * IDX_QUERY_BYTES
        self.idx_end = self.idx_mult + 4096
        self.roles = dict(
            sender=sender,
            reduce=reduce,
            merge=merge,
            query=query,
            post=post,
            pre=self.pre_status + 4 * P.STATUS_WORDS,
            attention=k * PART,
        )
        if cap == L.ROWS:
            self.arena = ARENA_BYTES
            assert (self.col, self.row_stage, sender) == (A_COL, A_ROW_STAGE, ARENA_BYTES)
            assert (self.xa, self.x3) == (A_XA, A_X3)
            assert (self.m_col, self.m_row, self.query_out) == (A_M_COL, A_M_ROW, QUERY_OUT)
            assert (self.pre_out, self.pre_stat, self.pre_meta, self.pre_status) == (
                A_PRE_OUT,
                A_PRE_STAT,
                A_PRE_META,
                A_PRE_STATUS,
            )
            assert (self.q_out, self.q_meta, self.q_status, self.q_lanes) == (A_Q_OUT, A_Q_META, A_Q_STATUS, A_Q_LANES)
            assert (self.post_out, self.post_meta, self.post_status) == (A_POST_OUT, A_POST_META, A_POST_STATUS)
            assert self.payload == PAYLOAD and self.landing2 == A_LANDING2 and reduce <= ARENA_BYTES
        else:
            self.arena = -(-max(self.roles.values()) // 4096) * 4096
        assert k * PART <= self.idx_land and self.idx_end <= self.arena, (self.idx_end, self.arena)


def is_mock() -> bool:
    return tp.is_mock()


def rect_cores(rect):
    (x0, y0), (x1, y1) = rect
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def rect_set(rect):
    (x0, y0), (x1, y1) = rect
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def bf16_bytes(values: torch.Tensor) -> np.ndarray:
    """BF16 values (any shape, row-major) -> their bytes."""
    return values.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()


def bytes_bf16(raw, shape=None) -> torch.Tensor:
    t = torch.from_numpy(np.frombuffer(bytes(raw), dtype=np.int16).copy()).view(torch.bfloat16).float()
    return t if shape is None else t.reshape(shape)


def bytes_f32(raw) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(bytes(raw), dtype=np.float32).copy())


def _cb_roles(kernel):
    """(circular buffers the kernel pushes into, those it pops from) for the kernels of this piece and of
    fused/layer_common, from the kernel's own arguments; None for a kernel of another piece (its buffers are that
    piece's own business and are declared on its own cores)."""
    name = Path(str(kernel.kernel_source)).name
    ct = [int(v) for v in kernel.compile_time_args]
    named = {k: int(v) for k, v in kernel.named_compile_time_args}
    if name in ("attn_merge_dm.cpp", "attn_merge_compute.cpp", "attn_merge16_dm.cpp", "attn_merge16_compute.cpp"):
        n_col, n_row = ct[0], ct[1]
        acc, send, sink, final, cb_in = (ct[2], ct[3]), (ct[4], ct[5]), ct[6], ct[7], ct[8]
        local = send if (n_col == 0 and n_row > 0) else acc
        if name == "attn_merge_dm.cpp":
            return {*local, sink, cb_in}, {final}
        if name == "attn_merge16_dm.cpp":  # (and the number of chains for the merge kernel)
            return {*local, sink, cb_in, ct[9]}, {final}
        if name == "attn_merge16_compute.cpp":
            push, pop = _cb_roles(_Renamed(kernel, "attn_merge_compute.cpp"))
            return push, pop | {ct[14]}
        t = [(ct[9], ct[10]), (ct[11], ct[12])]
        push, pop, run, k = {final}, {sink}, local, 0
        for i in range(n_col):
            to_send = n_row > 0 and i + 1 == n_col
            cur = send if to_send else t[k]
            pop |= {cb_in, *run}
            push |= set(cur)
            run, k = cur, k if to_send else k ^ 1
        for _ in range(n_row):
            pop |= {cb_in, *run}
            push |= set(t[k])
            run, k = t[k], k ^ 1
        return push, pop | set(run)
    if name == "attn_query_hub.cpp":
        return {named["cb_land"]}, {named["cb_sum"]}
    if name == "attn_append_dm.cpp":
        return set(ct[:5]), set()
    if name == "attn_append_write.cpp":
        return set(), set(ct[:4])
    if name == "attn_append_compute.cpp":  # raw, w, cs, mean, one in; e4m3, scale, rope, keep out; the rest its own
        own = set(ct[5:10]) | {ct[12], ct[13], ct[15]}
        return own | {ct[10], ct[11], ct[14], ct[16]}, set(ct[:5]) | own
    if name == "attn_key_dm.cpp":
        return set(ct[:6]), set()
    if name == "attn_key_write.cpp":
        return set(), set(ct[:2])
    if name == "attn_key_compute.cpp":  # raw, sum, wb, cs, h, mean in; keep, bfp out; the rest its own
        own = set(ct[6:15])
        return own | {ct[15], ct[16]}, set(ct[:6]) | own
    if name == "attn_lane_sum.cpp":
        return {ct[1]}, {ct[0]}
    if name in ("reduce_dm.cpp", "reduce_sum.cpp"):
        s1, total, s2, f, root = ct[0], ct[1], ct[2], ct[3], ct[7]
        dm = ({s1} | ({s2} if root else set()), {total} | ({f} if root else set()))
        return dm if name == "reduce_dm.cpp" else (dm[1], dm[0])
    if name.startswith("attn_") or name in ("collective_sender.cpp", "xfer_hub.cpp"):
        return set(), set()  # (the hub wrappers push nothing of their own: proj_hub.cpp only takes addresses)
    return None


def noc_audit(desc):
    """Which data-movement processor of which cores runs which kernel on which NoC index, from a chip's built
    program descriptor: [(kernel, processor 0 | 1, NoC 0 | 1, number of cores)]. Asserts that no two processors of
    a core issue their transactions on the same NoC index (two RISCs with transactions outstanding on one NoC share
    its command buffers and its counters of issued and acknowledged transactions)."""
    per_core, rows = {}, []
    for kernel in desc.kernels:
        cfg = kernel.config
        if isinstance(cfg, ttnn.ComputeConfigDescriptor):
            continue
        if isinstance(cfg, ttnn.ReaderConfigDescriptor):  # (tt_metal ReaderDataMovementConfig: RISCV_1 on NOC_0)
            proc, noc = 1, 0
        elif isinstance(cfg, ttnn.WriterConfigDescriptor):  # (WriterDataMovementConfig: RISCV_0 on NOC_1)
            proc, noc = 0, 1
        else:
            # (by name: these enums do not compare equal to their module constants from Python; NOC_0 prints as
            # RISCV_0_default, NOC_1 as RISCV_1_default)
            proc_name, noc_name = str(cfg.processor), str(cfg.noc)
            assert proc_name.endswith(("RISCV_0", "RISCV_1")) and noc_name.endswith(
                ("RISCV_0_default", "RISCV_1_default", "NOC_0", "NOC_1")
            ), (proc_name, noc_name)
            proc = 0 if proc_name.endswith("RISCV_0") else 1
            noc = 0 if noc_name.endswith(("RISCV_0_default", "NOC_0")) else 1
        name = Path(str(kernel.kernel_source)).name
        cores = [(c.x, c.y) for c in ttnn.corerange_to_cores(kernel.core_ranges)]
        rows.append((name, proc, noc, len(cores)))
        for core in cores:
            per_core.setdefault(core, []).append((proc, noc, name))
    for core, uses in per_core.items():
        assert len({p for p, _, _ in uses}) == len(uses), f"two kernels on one processor of core {core}: {uses}"
        assert len({n for _, n, _ in uses}) == len(uses), f"two processors of core {core} on one NoC index: {uses}"
    return rows


class _Renamed:
    """A kernel descriptor's arguments under another source name (for _cb_roles)."""

    def __init__(self, kernel, name):
        self.kernel_source, self.compile_time_args = name, kernel.compile_time_args
        self.named_compile_time_args = kernel.named_compile_time_args


def cb_audit(desc):
    """Every circular buffer has exactly ONE producer processor and ONE consumer processor (PLAN.md section 14): the
    pack thread and a data-movement kernel each keep their own count of what they pushed, and the pack thread
    publishes its own, so a second producer makes tiles vanish for the consumer, whose wait then never ends.
    Returns the table [(core, buffer, producers, consumers)] over the kernels whose buffers this piece knows
    (_cb_roles) and raises on a buffer with two producers or two consumers."""
    table = {}
    for kernel in desc.kernels:
        roles = _cb_roles(kernel)
        if roles is None:
            continue
        who = Path(str(kernel.kernel_source)).name
        for kind, ids in zip(("push", "pop"), roles):
            for core in ttnn.corerange_to_cores(kernel.core_ranges):
                for cb in ids:
                    table.setdefault(((core.x, core.y), cb), {"push": [], "pop": []})[kind].append(who)
    rows = [(core, cb, v["push"], v["pop"]) for (core, cb), v in sorted(table.items())]
    bad = [row for row in rows if len(row[2]) > 1 or len(row[3]) > 1]
    assert not bad, f"circular buffers with two producers or two consumers: {bad[:6]}"
    return rows


class LayerSet:
    """What ONE layer adds to an AttnLayer (AttnLayer.layer_set()): its three projection groups, its sink (the peer
    values; with the tensor form also the tensor), its norm weight, its site index and its slot in the KV storage,
    and the descriptors built for them. ``p1(layer=...)`` / ``p2(layer=...)`` build and run with a set without the
    caller touching the AttnLayer's attributes."""

    # (key_wb, key_rank: a full layer's key LayerNorm tensor and its rank among the key store's layer slots)
    FIELDS = (
        "pre",
        "query",
        "post",
        "sink",
        "sink_peer",
        "gamma",
        "table",
        "site_index",
        "cache_layer",
        "kv_norm",
        "key_wb",
        "key_rank",
    )

    def __init__(self, **kw):
        assert set(kw) == set(self.FIELDS), sorted(set(kw) ^ set(self.FIELDS))
        self.__dict__.update(kw)
        self.programs = {}


class AttnLayer:
    """The programs and their persistent state. ``chips``: the chips that take part (default all 32; fewer only
    for bring-up: what the absent chips would have sent is then written into the landings by the host)."""

    def __init__(
        self,
        mesh_device,
        root=(0, 0),
        chips=None,
        full=False,
        timing=False,
        wait_limit=30_000_000,
        fmt="bfp8",
        fidelity="custom_mm",
        proj_defines=(),
        parse_only=False,
        virtual=None,
        watch_x3=False,
        merge_fp32=True,
        sink_args=False,
        append=None,
        append_keep=True,
        reader_frames=False,
        latent_scale=None,
        rows=L.ROWS,
        p2_alone=False,
        host=None,
        key_dry=False,
        index_query=False,
        fixed_order=False,
    ):
        """parse_only: every chip's sender is built parse-only (layer_common Sender(parse_page=...)): it writes its
        argument table as it read it to arena offset A_PARSE on the sender core and sends nothing; ``parsed`` then
        holds what each chip's page must be. The other kernels run into their bounded waits."""
        self.parse_only, self.parsed, self.hand_overs = parse_only, {}, {}
        # watch_x3 (with timing, and ONLY when P1 runs alone): a measurement kernel on the landing cores of X3 stamps
        # the last arrival and takes the arrival counts, which are P2's relays' to take when P2 follows
        self.watch_x3 = watch_x3
        # merge_fp32: X4's merge kernel with a Float32 DEST (kernels/attn_merge_compute.cpp says why; the default:
        # ctx then drifts -0.05 % per merge against a float64 merge of the same partials instead of +0.19 %)
        self.merge_fp32 = merge_fp32
        # sink_args: the sink peer reaches the merge cores as 8 runtime arguments of attn_merge_dm.cpp (the chip's
        # head group's max values; the kernel builds the peer in a program buffer) instead of an L1 tensor per layer
        self.sink_args = sink_args
        # fixed_order (PLAN.md section 19.14): X4's merge cores take their partners' partials in index order whatever
        # order they land in (attn_merge_dm.cpp / attn_merge16_dm.cpp under HY4A_FIXED_ORDER): a traced step's merges
        # are then one sequence on every replay. Default: arrival order.
        self.fixed_order = bool(fixed_order)
        # append: None (the host writes the new token's cache row before the step), or the rounding of the row's
        # e4m3 pack when P1 writes it on the owner chips: "truncate" (the prefill stack's pack, bit for bit) or
        # "nearest" (PLAN.md section 17). append_keep: the finished BF16 rows stay in the append core's arena.
        assert append in (None, "truncate", "nearest"), append
        self.append, self.append_keep = append, append_keep
        # reader_frames: sparse_mla's row reader takes its addresses from fused/kv_store's frame tables (identity
        # tables on today's store) instead of the slot arithmetic
        self.reader_frames = reader_frames
        # latent_scale: ONE scalar c on the FP8 part of every stored row at read, folded into the weights that read it
        # (set_weights: the absorption w_k for the scores' content columns, w_v on the value side). It undoes the
        # mean of the truncating cast's bias (1.0430: PLAN.md sections 12a, 17); every row must then carry that bias,
        # so the append stays "truncate".
        self.latent_scale = None if latent_scale is None else float(latent_scale)
        assert not (self.latent_scale is not None and append == "nearest"), "a 'nearest' row would be over-corrected"
        self.feeds = (
            {}
        )  # (program, coord) -> [(compute kernel, buffer, tiles it waits for, tiles pushed in every outcome)]
        # virtual: (mesh rows, mesh columns) of partners that do not take part, for ONE chip without a fabric
        # connection: P2 then runs its whole X4 as if they did. Their partials are written into the landings by the
        # host with their arrival counts (p2_virtual_prefill), and what the chip would send them (the column ops,
        # the row op) goes through the sender as own copies into the arena of core VIRT, counted in its words.
        self.virtual = virtual
        assert virtual is None or (chips is not None and len(chips) == 1), "virtual partners: one chip"
        # full: a full-indexer layer. Built so far (PLAN.md section 20): the pre group with the layer's index units
        # and, with ``append``, the index KEY of the step's own position written into the key store by P1 (set_keys).
        # Not built: the index query's way to the scan (section 21): the selection is the caller's.
        self.mesh = mesh_device
        self.topo = tp.Topology(tuple(mesh_device.shape), root, chips)
        self.full, self.timing, self.wait_limit, self.fmt = full, timing, int(wait_limit), fmt
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        # rows: token rows a pass. 16 (PLAN.md section 19; P2 only so far): two sub-passes of 8 with a row mask
        # (set_mask), [16, 32] tiles in the projection groups (which then need a fidelity other than "custom_mm")
        # p2_alone (measurement only): P2's relays do not wait for the queries' arrival counts, so P2 can be launched
        # again and again (a trace of P2 without P1) on the queries and gates a prefill left
        self.p2_alone = p2_alone
        # host: another AttnLayer (of more rows) on the same mesh whose persistent tensors this one uses: its arena
        # (at the host's offsets), its words, its site. This layer is then the host's second program set (8 rows on
        # a 16-row layer); it owns only its transport's words, its rope tiles and its groups' go words.
        self.host = host
        assert host is None or (
            host.mesh is mesh_device and host.rows >= rows and rows == L.ROWS
        ), "a host has the larger layout"
        self.rows, self.lay = rows, Layout(rows, None if host is None else host.rows)
        assert rows == L.ROWS or (fidelity != "custom_mm" and append is None), "16 rows: P2 only, hifi_mm units"
        if host is None:
            self.n_words = A_WORDS if rows == L.ROWS else WORDS16
            self.arena = tp.core_buffer(mesh_device, tp.grid_set(mesh_device), self.lay.arena, ttnn.uint32, width=256)
            self.words = tp.core_buffer(
                mesh_device, tp.grid_set(mesh_device), 4 * self.n_words, ttnn.uint32, width=self.n_words
            )
        else:
            self.n_words, self.arena, self.words = host.n_words, host.arena, host.words
        # cos | sin tiles of the rows' positions (a program set's own: a host write per pass, in its rows' tiles)
        self.rope = tp.core_buffer(mesh_device, [QUERY_HUB], 2 * self.lay.row_tile)
        # the pass's row mask (16 rows): one word, the same on every core; bit r = token row r is live
        self.mask = (
            None if rows == L.ROWS else tp.core_buffer(mesh_device, tp.grid_set(mesh_device), 64, ttnn.uint32, width=16)
        )
        self.parse = tp.core_buffer(mesh_device, [SENDER], Sender.PARSE_PAGE_BYTES, ttnn.uint32) if parse_only else None
        ttnn.synchronize_device(mesh_device)
        self.arena_address = tp.address(self.arena)
        self.words_address = tp.address(self.words)
        self.t = tp.Transport(
            mesh_device,
            self.topo,
            PRE_HUB,
            SENDER,
            self.lay.reduce,
            self.lay.payload,
            wait_limit=wait_limit,
            timing=timing,
            arena=(
                self.arena,
                {"staging": self.lay.staging, "landing1": self.lay.landing1, "landing2": self.lay.landing2},
            ),
            x_address=self.at(A_PRE_X),
        )
        self.defines = self.t.defines + [("HY4A_WORDS", f"{self.words_address}u")]
        if self.mask is not None:
            self.defines += [("HY4A_ROWS16", "1"), ("HY4A_MASK", f"{tp.address(self.mask)}u")]
        self.include_paths = [str(KERNEL_DIR), str(tp.KERNEL_DIR), str(cl.KERNEL_DIR), str(P.KERNEL_DIR)]

        self.cfg = IHCConfig(dim=HIDDEN)
        if host is None:
            self.site = IhcSite(mesh_device, self.cfg, f_dtype=ttnn.bfloat16, **SITE)
            self.xn = self.site.empty_rows(ttnn.bfloat16)
            self.y = self.site.empty_rows(ttnn.float32)
            self.site_f = self.site.empty_rows(ttnn.bfloat16)  # the site's own F rows (8 rows)
            self.streams = self.site.streams_to_device(torch.zeros(L.ROWS, 4, HIDDEN))
        else:
            self.site, self.xn, self.y, self.site_f, self.streams = (
                host.site,
                host.xn,
                host.y,
                host.site_f,
                host.streams,
            )
        self.f = self.site_f
        if rows != L.ROWS:
            # F at 16 rows is twice the site's row tensor (the site at 16 rows is rung (c)): a buffer of its own on
            # the site's cores, a core's share of the payload each, in the reduce's order
            (ox, oy), per_row = SITE["origin"], SITE["per_row"]
            f_cores = rect_cores(((ox, oy), (ox + per_row - 1, oy + SITE["num_cores"] // per_row - 1)))
            assert f_cores == [(k.x, k.y) for k in self.site.cores], "F: the site's cores in row-major order"
            self.f = tp.core_buffer(mesh_device, rect_set((f_cores[0], f_cores[-1])), self.lay.payload // len(f_cores))
        self.gamma = self.table = None
        self.site_index = 0

        self.proj_kw = dict(fmt=fmt, defines=list(proj_defines) + ([("HY4_TIMING", "1")] if timing else []))
        if fidelity != "custom_mm":
            self.proj_kw["fidelity"] = fidelity
        self.pre = self.query = self.post = None
        self.sm = self.kv = self.sel = self.q = self.out = None
        self.keep_out = False
        # the sink peer, [max / sum tile | O = 0], per merge core (not with sink_args)
        self.sink = None if sink_args else tp.core_buffer(mesh_device, MERGE, PART)
        self.sink_peer = None
        self.cache_layer = 0
        self.kv_norm = self.cs = self.positions = self.frames = None
        self.kv_norm_eps = 1e-6
        if append is not None:
            dram = lambda shape, dtype, layout: ttnn.from_torch(
                torch.zeros(shape, dtype=torch.float32 if dtype == ttnn.bfloat16 else torch.int32),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.cs = dram((1, 1, 32, 4 * L.TILE), ttnn.bfloat16, ttnn.TILE_LAYOUT)  # cos 0, cos 1, sin 0, sin 1
            self.positions = dram((1, 1, 1, 16), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self.frames = kv_device.FrameTables(mesh_device, [APPEND], rows=L.ROWS)
        self.keys = self.key_wb = self.key_h = self.key_frames = self.key_geometry = None
        # key_dry (bring-up): the key core reads and keeps the blocks its rows would go to and writes nothing
        self.key_dry = key_dry
        self.key_rank, self.key_eps = 0, 1e-5
        # index_query (a full layer; PLAN.md section 21.3): P1 writes the scan's query. The pre hub gives the pre
        # group the step's cos / sin tiles (so the index-query tiles' rope dims are valid) and hands the chip's 4
        # index-query tiles to the sender; one more Xa op per chip and chip 22's pool op (the head weights) land on
        # the first idx core; the 8 data-movement processors of the 4 idx cores write the 8 token rows' query tiles into
        # ``Scan.qt`` and their 32 multipliers into the step pages (set_scan) before the chip's P1 ends.
        self.index_query = bool(index_query)
        assert not self.index_query or (full and rows == L.ROWS), "the index query: a full layer, 8 token rows"
        self.scan_qt = self.scan_step = None
        # idx_expect (bring-up only): the arrivals of index-query tiles the idx core waits for, instead of the chips
        # that take part (one more than can come: its wait runs out, the status path of attn_idx_dm.cpp)
        self.idx_expect = None
        if append is not None and full:
            from models.experimental.hy4_preview.tt import frames as tt_frames

            self.key_frames = kv_device.FrameTables(mesh_device, [KEY], rows=L.ROWS)
            # the Hadamard matrix the keys are stored in (tt/frames.py: BF16 entries, as the prefill indexer loads it)
            self.key_h = ttnn.from_torch(
                tt_frames.hadamard(L.IDX_DIM).reshape(1, 1, L.IDX_DIM, L.IDX_DIM),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self._check_cores()
        self._programs = {}
        self._shadow = None
        ttnn.synchronize_device(mesh_device)

    # ------------------------------------------------------------------------------------------ construction
    def at(self, offset) -> int:
        """The L1 address of arena offset ``offset`` (on whichever core uses it)."""
        assert 0 <= offset < self.lay.arena and offset % 64 == 0, offset
        return self.arena_address + offset

    def aword(self, index) -> int:
        return self.words_address + 4 * index

    @staticmethod
    def _share(w):
        """A chip's share with a pool column on every chip (zero where it has none), so that the 32 chips' unit
        lists are the same."""
        w = dict(w)
        w.setdefault("pool", torch.zeros(HIDDEN, L.TILE))
        return w

    def _group(self, build, units_of, lw, rect, hub, offsets, cores=None, stream=None, **kw):
        """A projection group for the whole mesh: its units on the first cores of ``rect`` (the hub aside), its hub
        buffers at arena ``offsets`` on ``hub``, its stream a mesh tensor with a shard per chip."""
        fmt = self.fmt
        if stream is not None:  # (stream: the weight stream another program set of the layer already holds)
            weights = stream
        else:
            shards, rows = [], 0
            streams = []
            for chip in range(self.mesh.get_num_devices()):
                streams.append(P.unit_stream(units_of(self.chip_share(lw, chip)), fmt))
                rows = max(rows, streams[-1].shape[0])
            for one in streams:
                if one.shape[0] < rows:
                    one = torch.cat((one, torch.zeros(rows - one.shape[0], one.shape[1])), dim=0)
                shards.append(P.host_shard(one, fmt))
            weights = P.mesh_stream(self.mesh, shards)
        share = self.chip_share(lw, 0)
        n_units = len(units_of(share))
        cores = ([c for c in rect_cores(rect) if c != tuple(hub)] if cores is None else list(cores))[:n_units]
        assert len(cores) == n_units, (len(cores), n_units)
        (x0, y0), (x1, y1) = rect
        prog = build(
            self.mesh,
            share,
            cores=cores,
            hub=tuple(hub),
            rect=(x0, y0, x1, y1),
            hub_buffers={k: self.at(v) for k, v in offsets.items()},
            weights=weights,
            **self.proj_kw,
            **kw,
        )
        need = prog.hub_buffer_bytes()
        spans = sorted(
            (offsets[k], offsets[k] + need[k], k) for k in need if not (k == "stat" and offsets[k] == offsets["out"])
        )
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), f"hub buffers overlap: {spans}"
        prog.offsets = dict(offsets)
        return prog

    def _check_cores(self):
        site = {(c.x, c.y) for c in self.site.cores}
        roles = {
            "pre": set(rect_cores(PRE_RECT)),
            "query": set(rect_cores(QUERY_RECT)),
            "sender": {SENDER},
            "site": site,
        }
        names = list(roles)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                assert not (roles[a] & roles[b]), f"core roles overlap: {a} and {b}: {roles[a] & roles[b]}"

    # ------------------------------------------------------------------------------------------ weights
    def chip_share(self, lw, chip):
        return self._share(L.chip_weights(lw, chip))

    def scaled(self, lw: L.LayerWeights) -> L.LayerWeights:
        """lw as the groups' streams hold it: with latent_scale folded into w_k and w_v (lw itself otherwise)."""
        if self.latent_scale is None:
            return lw
        return dataclasses.replace(lw, w_k=lw.w_k * self.latent_scale, w_v=lw.w_v * self.latent_scale)

    def set_weights(self, lw: L.LayerWeights, free=True):
        """The layer's projections (layout.layer_weights WITHOUT the input norm's weight folded in: x arrives
        normalised) as one mesh stream per group, a shard per chip, packed on the host: a stream holds
        layout.stored(w, fmt), which the tests' "exact arithmetic on the stored weights" is taken on.
        free=False: the groups set before stay allocated (a LayerSet taken with layer_set() owns them)."""
        assert lw.full == self.full
        # A second program set (host=) reads the streams its host's groups hold ONLY when the host was given this
        # very weights object (the same layer, the same latent_scale); any other object gets streams of its own.
        # streams_from records the weights object the streams this set reads were packed from: a test compares it
        # with the weights its reference uses (a set that ran on other weights than it was given must not pass).
        was_shared = getattr(self, "shared_streams", False)
        self.shared_streams = (
            self.host is not None
            and self.host.pre is not None
            and getattr(self.host, "lw", None) is lw
            and self.host.latent_scale == self.latent_scale
        )
        self.lw = lw
        self.streams_from = self.host.streams_from if self.shared_streams else lw
        assert self.streams_from is lw, "a program set's streams must come from the weights object it was given"
        lw = self.scaled(lw)
        for prog in (self.pre, self.query, self.post):
            if prog is not None and free:
                if not was_shared:
                    ttnn.deallocate(prog.weights)
                ttnn.deallocate(prog.go)
        lay = self.lay
        rows_kw = {} if self.rows == L.ROWS else dict(rows=self.rows)
        # (a second program set reads the streams the host's groups hold: the same weights, the same stream format)
        held = (lambda name: getattr(self.host, name).weights) if self.shared_streams else (lambda name: None)
        self.pre = self._group(
            P.pre_program,
            lambda w: P.pre_units(w, gate_done=1),
            lw,
            PRE_RECT,
            PRE_HUB,
            dict(x=lay.pre_x, out=lay.pre_out, stat=lay.pre_stat, meta=lay.pre_meta, status=lay.pre_status),
            gate_done=1,
            stream=held("pre"),
            **rows_kw,
        )
        self.query = self._group(
            P.query_program,
            P.query_units,
            lw,
            QUERY_RECT,
            QUERY_HUB,
            dict(x=lay.q_x, out=lay.q_out, stat=lay.q_out, meta=lay.q_meta, status=lay.q_status),
            stream=held("query"),
            **rows_kw,
        )
        self.post = self._group(
            P.post_program,
            lambda w: P.post_units(w, o_ct=POST_O_CT),
            lw,
            POST_RECT,
            POST_HUB,
            dict(x=lay.post_x, out=lay.post_out, stat=lay.post_out, meta=lay.post_meta, status=lay.post_status),
            cores=POST_CORES,
            o_ct=POST_O_CT,
            stream=held("post"),
            **rows_kw,
        )
        self._programs = {}
        if self.append is not None:  # the layer's kv norm weight goes where the latent row is finished
            from models.experimental.hy4_preview.reference.attention import AttentionConfig

            if free and self.kv_norm is not None:
                ttnn.deallocate(self.kv_norm)
            self.set_kv_norm(lw.kv_norm, AttentionConfig().norm_eps)
            if self.full:  # the key LayerNorm's weight and bias go where the key row is finished
                from models.experimental.hy4_preview.reference.indexer import IndexerConfig

                if free and self.key_wb is not None:
                    ttnn.deallocate(self.key_wb)
                self.set_key_norm(lw.key_norm_w, lw.key_norm_b, IndexerConfig().norm_eps)
        ttnn.synchronize_device(self.mesh)

    def set_cache(self, cfg: SL.SparseMlaConfig, kv, sel, layer, keep_out=False):
        """The KV storage (the prefill stack's packed rows, one mesh tensor), the selection table (uint32
        [1, 1, 1 + users, topk] per chip: sparse_mla's contract) and the layer's index in the storage's slots.
        keep_out: the attention cores also write their partials to the ``out`` tensor (tests)."""
        if self.sm is None or self.sm.cfg != cfg:
            self.sm = SparseMla(cfg, self.mesh, k_first=True, frames=True if self.reader_frames else None)
            dram = lambda n, w: ttnn.from_torch(
                torch.zeros(1, 1, n, w),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.q = dram(cfg.users * cfg.head_groups, self.sm.q_page_values)  # not read in a layer program
            self.out = dram(cfg.users * cfg.head_groups, self.sm.out_page_values)  # status pages (and partials)
        self.kv, self.sel, self.cache_layer, self.keep_out = kv, sel, int(layer), keep_out
        self._programs = {}
        if self.append is not None:
            self.set_frames()

    def set_sink(self, sink, new=False):
        """sink [64]: the layer's per-head sink logit. The merge cores of a chip get the peer of the chip's head
        group (its mesh row): max = BF16(sink / scale), sum = 1, O = 0. new (tensor form): a tensor of its own
        for this sink (the one set before belongs to a LayerSet)."""
        if new and not self.sink_args:
            self.sink = tp.core_buffer(self.mesh, MERGE, PART)
            self._programs = {k: v for k, v in self._programs.items() if k[0] != "p2"}
        scale = self.sm.cfg.scale if self.sm is not None else 1.0 / 16.0
        peer = (sink.float() / scale).to(torch.bfloat16).float()
        self.sink_peer = peer
        if self.sink_args:  # (runtime arguments of the merge cores: the descriptors hold them)
            self._programs = {k: v for k, v in self._programs.items() if k[0] != "p2"}
            return
        shards = []
        for chip in range(self.mesh.get_num_devices()):
            g = chip // 4
            ms = torch.zeros(L.ROWS, L.TILE)
            ms[:, 0], ms[:, 1] = peer[8 * g : 8 * g + 8], 1.0
            page = torch.zeros(PART // ROW_TILE, ROW_TILE // 2)
            page[0] = SL.pack_tiny(ms)
            shards.append(
                ttnn.from_torch(page.repeat(len(MERGE), 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            )
        ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, self.mesh.shape), self.sink)

    # ------------------------------------------------------------------------------------------ a layer's set
    def layer_set(self) -> LayerSet:
        """The set that is current (what set_weights / set_sink / set_site / set_cache put there), as an object. Give
        it to p1 / p2 / run_p1 / run_p2 as ``layer``; set the next layer with set_weights(lw, free=False)."""
        return LayerSet(**{k: getattr(self, k) for k in LayerSet.FIELDS})

    @contextlib.contextmanager
    def _bound(self, layer):
        """The layer's set in place of the current one while a descriptor is built."""
        keep = {k: getattr(self, k) for k in LayerSet.FIELDS}
        keep_programs = self._programs
        try:
            for k in LayerSet.FIELDS:
                setattr(self, k, getattr(layer, k))
            self._programs = layer.programs
            yield
        finally:
            for k, v in keep.items():
                setattr(self, k, v)
            self._programs = keep_programs

    def release_set(self, layer):
        """Free a set's own tensors (its groups' streams and go words, its norm weight, its sink tensor)."""
        tensors = [layer.gamma] + ([layer.sink] if layer.sink is not self.sink else [])
        tensors += [layer.key_wb] if layer.key_wb is not self.key_wb else []
        for prog in (layer.pre, layer.query, layer.post):
            if prog is not None and prog not in (self.pre, self.query, self.post):
                tensors += [prog.weights, prog.go]
        for tensor in tensors:
            if tensor is not None and tensor is not self.gamma:
                ttnn.deallocate(tensor)
        layer.programs = {}

    def set_site(self, sites, site_index, gamma):
        """``sites`` [(fn, base, scale), ...]: the iHC sites in the DRAM table (``site_index``: this layer's
        attention site), ``gamma``: the input norm's weight [6144]."""
        self.site_index = site_index
        self.table = self.site.site_table(sites)
        self.gamma = self.site.gamma_to_device(gamma)
        self._programs = {}
        ttnn.synchronize_device(self.mesh)

    # ------------------------------------------------------------------------------------------ program P1
    def noc_word(self, core) -> int:
        return self.t.noc_word(core)

    def _all_routes(self, coord):
        """From ``coord`` to every other chip that takes part: the four mesh branches, or unicasts on a part of
        the mesh."""
        topo = self.topo
        rows, cols = topo.mesh_shape
        if topo.whole_mesh and rows > 1 and cols > 1:
            return mesh_branches(topo.mesh_shape, coord)
        others = [c for c in topo.chips if c != tuple(coord)]
        assert len(others) <= cl.PACKET_HEADERS_PER_RISC
        return [unicast(c) for c in others]

    def _norm_out(self):
        sender, staging, ready, bit = self.t.x_source_args()
        return NormOut(
            gamma=self.gamma,
            xn=self.xn,
            sender=sender,
            staging=staging,
            ready=ready,
            ready_bit=bit,
            gather=self.t.word_address(tp.W_GATHER),
            status=self.t.word_address(tp.W_STATUS),
            wait_limit=self.t.wait_limit,
            stamps=self.t.word_address(tp.W_STAMP) if self.timing else 0,
        )

    def _wrap_hub(self, prog, desc, source, extra_rt, named_ct=(), defines=()):
        hub = desc.kernels[0]
        assert str(hub.kernel_source).endswith("proj_hub.cpp"), hub.kernel_source
        hub.kernel_source = str(KERNEL_DIR / source)
        hub.defines = list(hub.defines) + self.defines + list(defines)
        hub.compiler_include_paths = self.include_paths
        if named_ct:
            hub.named_compile_time_args = list(named_ct)
        a = prog.hub_addr
        hub.runtime_args[prog.hub.x][prog.hub.y] = [a["x"], a["meta"], a["status"], prog.go.buffer_address()] + list(
            extra_rt
        )

    def _p1_chip(self, coord, post):
        lay = self.lay
        topo, t = self.topo, self.t
        coord = tuple(coord)
        chip, root = topo.chip(coord), coord == topo.root
        present = len(topo.chips)
        ready = t.word_address(tp.W_READY)
        sender = Sender(
            self.mesh,
            coord,
            SENDER,
            ready,
            self.wait_limit,
            parse_page=tp.address(self.parse) if self.parse_only else None,
        )
        if root:
            t.broadcast_op(sender)
        routes = self._all_routes(coord)
        op_qa = sender.op(
            self.at(lay.xa),
            2 * ROW_TILE,
            2 * ROW_TILE,
            QUERY_HUB,
            self.at(lay.q_x) + 2 * chip * ROW_TILE,
            self.aword(W_QA),
            routes,
            own=True,
        )
        op_stat = sender.op(
            self.at(lay.xa) + 2048,
            STAT_PAGE,
            STAT_PAGE,
            QUERY_HUB,
            self.at(lay.q_lanes) + chip * STAT_PAGE,
            self.aword(W_LANES),
            routes,
            own=True,
        )
        op_x3 = sender.op(
            self.at(lay.x3),
            QUERY_OUT,
            min(t.chunk, QUERY_OUT),
            MERGE[chip // 4],
            self.at(lay.m_query) + (chip % 4) * QUERY_OUT,
            self.aword(W_QARR),
            routes,
            own=True,
        )
        xa_bits = Sender.bit(op_qa) | Sender.bit(op_stat)
        # the chip's pool tile: a tile column of the raw latent row to the append core of every chip, or (full
        # layers, chips 18..21) of the raw index key to the key core of every chip. (Chip 22's tile, the head
        # weights, has no op yet: PLAN.md section 21.)
        pool = L.pool_of(chip, self.full) if self.append is not None else None
        has_pool = pool is not None and pool[0] in ("kv", "key")
        if has_pool:
            core, landing = (APPEND, A_AP_RAW) if pool[0] == "kv" else (KEY, A_KEY_RAW)
            op_pool = sender.op(
                self.at(lay.xa) + 2 * ROW_TILE,
                ROW_TILE,
                ROW_TILE,
                core,
                self.at(landing) + pool[1] * ROW_TILE,
                self.aword(W_POOL),
                routes,
                own=True,
            )
            xa_bits |= Sender.bit(op_pool)
        if self.index_query:
            # chip 22's pool tile, the head weights [token rows, 32 heads], and every chip's 4 index-query tiles: to
            # the idx core of every chip
            if (L.pool_of(chip, True) or ("", 0))[0] == "head_w":
                assert not has_pool
                has_pool = True
                op_w = sender.op(
                    self.at(lay.xa) + 2 * ROW_TILE,
                    ROW_TILE,
                    ROW_TILE,
                    IDX,
                    self.at(lay.idx_w),
                    self.aword(W_IDXW),
                    routes,
                    own=True,
                )
                xa_bits |= Sender.bit(op_w)
            idx_bytes = P.IDX_TILES * ROW_TILE
            op_idx = sender.op(
                self.at(lay.xa) + XA_IDX_TILE * ROW_TILE,
                idx_bytes,
                idx_bytes,
                IDX,
                self.at(lay.idx_land) + chip * idx_bytes,
                self.aword(W_IDX),
                routes,
                own=True,
            )
            xa_bits |= Sender.bit(op_idx)

        desc = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
        io = []
        # the pre group
        d, _ = self.pre.descriptor()
        self._wrap_hub(
            self.pre,
            d,
            "attn_pre_hub.cpp",
            [
                self.pre.done_units(0),
                self.pre.done_address(0),
                self.at(lay.pre_out),
                self.at(lay.pre_stat),
                self.noc_word(SENDER),
                self.at(lay.xa),
                ready,
                xa_bits,
                int(has_pool),
            ]
            + ([self.noc_word(QUERY_HUB), self.rope.buffer_address(), P.PRE_HIDDEN_TILES] if self.index_query else []),
            defines=[("HY4A_PRE_IDX", str(P.PRE_IDX)), ("HY4A_XA_IDX_TILE", str(XA_IDX_TILE))]
            if self.index_query
            else (),
        )
        desc.kernels = list(desc.kernels) + list(d.kernels)
        desc.cbs = list(desc.cbs) + list(d.cbs)
        io += [self.pre.weights, self.pre.go]
        # the query group, and the lane sum on its hub core
        d, _ = self.query.descriptor()
        self._wrap_hub(
            self.query,
            d,
            "attn_query_hub.cpp",
            [
                self.query.done_units(0),
                self.query.done_address(0),
                self.at(lay.q_out),
                QUERY_OUT,
                present,
                self.rope.buffer_address(),
                P.Q_A_TILES,
                self.noc_word(SENDER),
                self.at(lay.x3),
                ready,
                Sender.bit(op_x3),
            ],
            named_ct=[("cb_land", CB_LAND), ("cb_sum", CB_SUM), ("land_pages", CHIPS)],
        )
        hub_set = tp.core_set([QUERY_HUB])
        land = ttnn.cb_descriptor_from_sharded_tensor(
            CB_LAND, ttnn.get_device_tensors(self.arena)[chip], address_offset=lay.q_lanes, total_size=CHIPS * STAT_PAGE
        )
        land.core_ranges = hub_set
        f32 = lambda cb: [ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.float32, page_size=STAT_PAGE)]
        land.format_descriptors = f32(CB_LAND)
        total = ttnn.CBDescriptor(total_size=STAT_PAGE, core_ranges=hub_set, format_descriptors=f32(CB_SUM))
        compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB_LAND] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode = modes
        lane_sum = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "attn_lane_sum.cpp"),
            core_ranges=hub_set,
            compile_time_args=[CB_LAND, CB_SUM, CHIPS],
            config=compute,
            compiler_include_paths=self.include_paths,
        )
        self.feeds[("p1", coord)] = [("attn_lane_sum.cpp", "lane landing", CHIPS, CHIPS)]
        desc.kernels = list(desc.kernels) + list(d.kernels) + [lane_sum]
        desc.cbs = list(desc.cbs) + list(d.cbs) + [land, total]
        io += [self.query.weights, self.query.go]
        if self.append is not None:
            k, c, a_io = self._append_chip(coord)
            desc.kernels = list(desc.kernels) + k
            desc.cbs = list(desc.cbs) + c
            io += a_io
            # (the append's buffers hold fixed tile counts; the feeder pushes all of them whether or not its wait
            # for the pool tiles ran out)
            self.feeds[("p1", coord)] += [
                (
                    "attn_append_compute.cpp",
                    "pool tiles (18) and the swapped rope tiles (2)",
                    AP_TILES + 2,
                    AP_TILES + 2,
                ),
                ("attn_append_compute.cpp", "kv norm weight", 16, 16),
                ("attn_append_compute.cpp", "cos / sin", 4, 4),
                ("attn_append_compute.cpp", "the two reduce scalers", 2, 2),
            ]
            if self.full:
                k, c, a_io = self._key_chip(coord)
                desc.kernels = list(desc.kernels) + k
                desc.cbs = list(desc.cbs) + c
                io += a_io
                # (as the append's: fixed tile counts, pushed whether or not the wait for the key tiles ran out)
                self.feeds[("p1", coord)] += [
                    (
                        "attn_key_compute.cpp",
                        "key tiles (4) and the rope tiles' twins (2)",
                        KEY_TILES + 2,
                        KEY_TILES + 2,
                    ),
                    ("attn_key_compute.cpp", "key tiles for the mean", KEY_TILES, KEY_TILES),
                    ("attn_key_compute.cpp", "LayerNorm weight | bias", 12, 12),
                    ("attn_key_compute.cpp", "cos / sin", 4, 4),
                    ("attn_key_compute.cpp", "Hadamard matrix", 16, 16),
                    ("attn_key_compute.cpp", "the reduce scaler", 1, 1),
                ]
        if self.index_query:
            k, a_io = self._idx_chip(coord)
            desc.kernels = list(desc.kernels) + k
            io += a_io
        if root:
            s_io, s_desc = self.site.program(
                self.streams,
                f=self.f if post else None,
                params=SiteRef(self.table, self.site_index),
                y=self.y,
                norm_out=self._norm_out(),
            )
            desc.kernels = list(desc.kernels) + list(s_desc.kernels)
            desc.cbs = list(desc.cbs) + list(s_desc.cbs)
            desc.semaphores = list(desc.semaphores) + list(s_desc.semaphores)
            io += s_io
        if self.timing and self.watch_x3 and not root:  # measurement: the last landing of each head group's queries
            rt = ttnn.RuntimeArgs()
            for g, (x, y) in enumerate(MERGE):
                rt[x][y] = [sum(1 for c in topo.chips if topo.chip(c) // 4 == g)]
            desc.kernels = list(desc.kernels) + [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "attn_watch.cpp"),
                    core_ranges=rect_set((MERGE[0], MERGE[-1])),
                    runtime_args=rt,
                    defines=self.defines,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                    compiler_include_paths=self.include_paths,
                )
            ]
        sender.finish(desc)
        self.parsed[("p1", coord)] = sender.parsed
        return desc, io

    def _append_chip(self, coord):
        """(kernels, circular buffers, io tensors) of the latent append on the chip's append core (PLAN.md 17)."""
        assert self.kv_norm is not None and self.sm is not None, "set_kv_norm and set_cache first"
        topo, sm = self.topo, self.sm
        chip = topo.chip(coord)
        cores = tp.core_set([APPEND])
        x, y = APPEND
        bf16, f32 = 2 * L.TILE * L.TILE, 4 * L.TILE * L.TILE
        cb = lambda index, dtype, page, pages: ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )
        cbs = [
            cb(AP_RAW, ttnn.bfloat16, bf16, AP_TILES + 2),
            cb(AP_W, ttnn.bfloat16, bf16, 16),
            cb(AP_CS, ttnn.bfloat16, bf16, 4),
            cb(AP_MEAN, ttnn.bfloat16, bf16, 1),
            cb(AP_ONE, ttnn.float32, f32, 1),
            cb(AP_SQ, ttnn.float32, f32, 1),
            cb(AP_STAT, ttnn.float32, f32, 1),
            cb(AP_INV, ttnn.float32, f32, 1),
            cb(AP_TMP, ttnn.float32, f32, 4),
            cb(AP_LAT, ttnn.bfloat16, bf16, 16),
            cb(AP_KEEP, ttnn.bfloat16, bf16, 16),
            cb(AP_ROPE, ttnn.bfloat16, bf16, 2),
            cb(AP_TILE, ttnn.float32, f32, 4),
            cb(AP_ABS, ttnn.float32, f32, 4),
            cb(AP_SCALE, ttnn.float32, f32, 4),
            cb(AP_INVS, ttnn.float32, f32, 1),
            cb(AP_E4M3, ttnn.fp8_e4m3, L.TILE * L.TILE, 16),
            cb(AP_ROWS, ttnn.uint32, sm.cfg.row_stride, L.ROWS),
            cb(AP_POS, ttnn.uint32, 64, 1),
        ]
        frames = ttnn.cb_descriptor_from_sharded_tensor(AP_FRAMES, ttnn.get_device_tensors(self.frames.tensor)[chip])
        cbs.append(frames)
        n_pool = sum(1 for c in topo.chips if (L.pool_of(topo.chip(c), self.full) or ("", 0))[0] == "kv")
        dev = lambda tensor: ttnn.get_device_tensors(tensor)[chip].buffer_address()
        feed = ttnn.RuntimeArgs()
        feed[x][y] = [n_pool, self.at(A_AP_RAW), dev(self.kv_norm), dev(self.cs)]
        write = ttnn.RuntimeArgs()
        write[x][y] = [
            chip,
            dev(self.positions),
            dev(self.kv),
            self.append_geometry().slot_bytes(self.cache_layer),
            self.at(A_AP_LAT) if self.append_keep else 0,
        ]
        eps = int.from_bytes(np.float32(self.kv_norm_eps).tobytes(), "little")
        kernels = [
            self._dm(
                "attn_append_dm.cpp",
                cores,
                feed,
                ttnn.DataMovementProcessor.RISCV_0,
                ttnn.NOC.NOC_0,
                ct=[AP_RAW, AP_W, AP_CS, AP_MEAN, AP_ONE],
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "attn_append_write.cpp"),
                core_ranges=cores,
                compile_time_args=[AP_E4M3, AP_SCALE, AP_ROPE, AP_KEEP, AP_ROWS, AP_POS, AP_FRAMES],
                runtime_args=write,
                defines=self.defines,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
                compiler_include_paths=self.include_paths + [str(KV_STORE_KERNELS)],
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "attn_append_compute.cpp"),
                core_ranges=cores,
                compile_time_args=[
                    AP_RAW,
                    AP_W,
                    AP_CS,
                    AP_MEAN,
                    AP_ONE,
                    AP_SQ,
                    AP_STAT,
                    AP_INV,
                    AP_TMP,
                    AP_LAT,
                    AP_KEEP,
                    AP_ROPE,
                    AP_TILE,
                    AP_ABS,
                    AP_SCALE,
                    AP_INVS,
                    AP_E4M3,
                    eps,
                ],
                defines=[("HY4A_FP8_NEAREST", "1")] if self.append == "nearest" else [],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
                compiler_include_paths=self.include_paths,
            ),
        ]
        return kernels, cbs, [self.kv_norm, self.cs, self.positions, self.kv, self.frames.tensor]

    def _key_chip(self, coord):
        """(kernels, circular buffers, io tensors) of the index-key append on the chip's key core (PLAN.md 20)."""
        assert self.key_wb is not None and self.keys is not None, "set_weights (a full layer's) and set_keys first"
        topo = self.topo
        chip = topo.chip(coord)
        cores = tp.core_set([KEY])
        x, y = KEY
        bf16, f32 = 2 * L.TILE * L.TILE, 4 * L.TILE * L.TILE
        cb = lambda index, dtype, page, pages: ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )
        all_tiles = KEY_TILES + 2  # the key's 4 tiles and the twins of its 2 rope tiles
        cbs = [
            cb(KY_RAW, ttnn.bfloat16, bf16, all_tiles),
            cb(KY_SUM, ttnn.bfloat16, bf16, KEY_TILES),
            cb(KY_WB, ttnn.bfloat16, bf16, 2 * all_tiles),
            cb(KY_CS, ttnn.bfloat16, bf16, 4),
            cb(KY_H, ttnn.bfloat16, bf16, KEY_TILES * KEY_TILES),
            cb(KY_MEAN, ttnn.bfloat16, bf16, 1),
            cb(KY_STAT, ttnn.float32, f32, 1),
            cb(KY_CEN, ttnn.float32, f32, all_tiles),
            cb(KY_SQ, ttnn.float32, f32, KEY_TILES),
            cb(KY_VAR, ttnn.float32, f32, 1),
            cb(KY_INV, ttnn.float32, f32, 1),
            cb(KY_TMP, ttnn.float32, f32, all_tiles),
            cb(KY_NW, ttnn.float32, f32, all_tiles),
            cb(KY_K, ttnn.bfloat16, bf16, all_tiles),
            cb(KY_ROT, ttnn.bfloat16, bf16, KEY_TILES),
            cb(KY_KEEP, ttnn.bfloat16, bf16, KEY_TILES),
            cb(KY_BFP, ttnn.bfloat8_b, KEY_TILE_BYTES, KEY_TILES),
            cb(KY_BLOCKS, ttnn.uint32, KEY_BLOCK_BYTES, L.ROWS),
            cb(KY_POS, ttnn.uint32, 64, 1),
        ]
        cbs.append(
            ttnn.cb_descriptor_from_sharded_tensor(KY_FRAMES, ttnn.get_device_tensors(self.key_frames.tensor)[chip])
        )
        n_key = sum(1 for c in topo.chips if (L.pool_of(topo.chip(c), True) or ("", 0))[0] == "key")
        dev = lambda tensor: ttnn.get_device_tensors(tensor)[chip].buffer_address()
        feed = ttnn.RuntimeArgs()
        feed[x][y] = [n_key, self.at(A_KEY_RAW), dev(self.key_wb), dev(self.cs), dev(self.key_h)]
        write = ttnn.RuntimeArgs()
        write[x][y] = [
            chip,
            dev(self.positions),
            dev(self.keys),
            self.key_geometry.slot_bytes(self.key_rank),
            self.at(A_KEY_KEEP) if self.append_keep else 0,
        ]
        eps = int.from_bytes(np.float32(self.key_eps).tobytes(), "little")
        kernels = [
            self._dm(
                "attn_key_dm.cpp",
                cores,
                feed,
                ttnn.DataMovementProcessor.RISCV_0,
                ttnn.NOC.NOC_0,
                ct=[KY_RAW, KY_SUM, KY_WB, KY_CS, KY_H, KY_MEAN],
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "attn_key_write.cpp"),
                core_ranges=cores,
                compile_time_args=[KY_BFP, KY_KEEP, KY_BLOCKS, KY_POS, KY_FRAMES],
                runtime_args=write,
                defines=self.defines + ([("HY4A_KEY_DRY", "1")] if self.key_dry else []),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
                compiler_include_paths=self.include_paths + [str(KV_STORE_KERNELS)],
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "attn_key_compute.cpp"),
                core_ranges=cores,
                compile_time_args=[
                    KY_RAW,
                    KY_SUM,
                    KY_WB,
                    KY_CS,
                    KY_H,
                    KY_MEAN,
                    KY_STAT,
                    KY_CEN,
                    KY_SQ,
                    KY_VAR,
                    KY_INV,
                    KY_TMP,
                    KY_NW,
                    KY_K,
                    KY_ROT,
                    KY_KEEP,
                    KY_BFP,
                    eps,
                ],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
                compiler_include_paths=self.include_paths,
            ),
        ]
        return kernels, cbs, [self.key_wb, self.key_h, self.cs, self.positions, self.keys, self.key_frames.tensor]

    def _idx_chip(self, coord):
        """(kernels, io tensors) of the index query's assembly on the chip's idx cores (PLAN.md 21.3): the same
        kernel on both data-movement processors of the 4 cores, one token row per processor; processor 0 of the
        first core waits for the arrivals and releases the others. No circular buffer and no compute kernel."""
        assert self.scan_qt is not None, "set_scan first"
        topo, lay = self.topo, self.lay
        chip = topo.chip(coord)
        assert len(IDX_CORES) * 2 == L.ROWS and [c[0] for c in IDX_CORES] == [IDX[0]] * len(IDX_CORES)
        cores = rect_set((IDX_CORES[0], IDX_CORES[-1]))
        n_idx = len(topo.chips) if self.idx_expect is None else int(self.idx_expect)
        n_w = sum(1 for c in topo.chips if (L.pool_of(topo.chip(c), True) or ("", 0))[0] == "head_w")
        dev = lambda tensor: ttnn.get_device_tensors(tensor)[chip].buffer_address()
        assert (
            dev(self.scan_qt) % 64 == 0 and dev(self.scan_step) % 64 == 0
        ), "NoC writes: source and destination congruent mod 16"
        nocs = [self.noc_word(c) for c in IDX_CORES]
        kernels = []
        for part, (proc, noc) in enumerate(
            (
                (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            )
        ):
            rt = ttnn.RuntimeArgs()
            for k, (x, y) in enumerate(IDX_CORES):
                rt[x][y] = [
                    2 * k,
                    int(k == 0),
                    n_idx,
                    n_w,
                    nocs[0],
                    self.at(lay.idx_land),
                    self.at(lay.idx_w),
                    self.at(lay.idx_work),
                    self.at(lay.idx_mult),
                    dev(self.scan_qt),
                    dev(self.scan_step),
                ] + nocs
            kernels.append(self._dm("attn_idx_dm.cpp", cores, rt, proc, noc, ct=[part, self.rows]))
        return kernels, [self.scan_qt, self.scan_step]

    def read_parsed(self, coord, n):
        """The first ``n`` words of a chip's sender parse page."""
        ttnn.synchronize_device(self.mesh)
        page = ttnn.to_torch(ttnn.get_device_tensors(self.parse)[self.topo.chip(coord)]).to(torch.int64) & 0xFFFFFFFF
        return [int(v) for v in page.flatten()[:n]]

    def p1(self, post=False, layer=None):
        """(io tensors, MeshProgramDescriptor) of P1. post: the root's leading iHC pass also does the post-mix of
        the F in the site's F tensor (the sublayer before). layer: a LayerSet (default: the current set)."""
        if layer is not None:
            with self._bound(layer):
                return self.p1(post)
        assert self.rows == L.ROWS, "P1 at 16 rows is not built (PLAN.md section 19, rung (c))"
        key = ("p1", post)
        if key not in self._programs:
            assert self.pre is not None and self.table is not None, "set_weights and set_site first"
            mesh_desc = ttnn.MeshProgramDescriptor()
            io = [self.arena, self.words, self.rope] + self.t.io() + ([self.parse] if self.parse_only else [])
            for coord in self.topo.chips:
                desc, chip_io = self._p1_chip(coord, post)
                io += chip_io
                mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
            unique = []
            for tensor in io + [self.f]:
                if not any(tensor is u for u in unique):
                    unique.append(tensor)
            self._programs[key] = (unique, mesh_desc)
        return self._programs[key]

    def run_p1(self, post=False, layer=None):
        io, desc = self.p1(post, layer)
        (ttnn.experimental.prepare_generic_op if is_mock() else ttnn.generic_op)(io, desc)

    # ------------------------------------------------------------------------------------------ program P2
    def _tiny_cb(self, cores, index, pages):
        tile = ttnn.Tile((L.ROWS, L.TILE))
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=index, data_format=ttnn.bfloat16, page_size=ROW_TILE, tile=ttnn.TileDescriptor(tile)
        )
        return ttnn.CBDescriptor(total_size=ROW_TILE * pages, core_ranges=cores, format_descriptors=[fmt])

    def _dm(self, source, cores, rt, processor, noc, ct=(), defines=()):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / source),
            core_ranges=cores,
            compile_time_args=list(ct),
            runtime_args=rt,
            defines=self.defines + list(defines),
            config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            compiler_include_paths=self.include_paths,
        )

    def p2_sources(self, coord):
        """Who lands on the chip at ``coord`` in X4: (mesh rows of its column partners, mesh columns of its row
        partners), of the chips that take part."""
        r, c = coord
        if self.virtual is not None:
            return list(self.virtual[0]), list(self.virtual[1])
        rows = [g for g in range(self.topo.mesh_shape[0]) if g != r and (g, c) in self.topo.chips]
        cols = [k for k in range(self.topo.mesh_shape[1]) if k != c and (r, k) in self.topo.chips]
        return rows, cols

    def _p2_chip(self, coord, stage):
        topo, t, sm = self.topo, self.t, self.sm
        coord = tuple(coord)
        r, c = coord
        chip, root = topo.chip(coord), coord == topo.root
        ready = t.word_address(tp.W_READY)
        rows, cols = self.p2_sources(coord)
        arena = ttnn.get_device_tensors(self.arena)[chip]
        att = lambda b, g: self.noc_word(sm.blocks[b].attention[g])
        merge_words = [self.noc_word(m) for m in MERGE]
        merge_set = rect_set((MERGE[0], MERGE[-1]))
        b0, b1 = ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1
        desc = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
        # 16 rows (PLAN.md section 19): the attention queues the 16 rows on the 8 S-blocks and leaves two final
        # partials per attention core; X4, the merges and ctx run sub-pass by sub-pass (the 16-row kernels and the
        # sender's hand-over words); post and X5 take [16, 32] tiles once
        lay, r16 = self.lay, self.rows != L.ROWS

        # sparse attention: queries from the landings, partials at a fixed address
        lio = LayerIo(
            arena,
            lay.att_final,
            self.aword(W_QGO),
            self.at(lay.m_query),
            MERGE,
            self.aword(W_LOCAL),
            SENDER,
            self.aword(W_CGROUP),
            own_group=r,
            keep_out=self.keep_out,
            mask=tp.address(self.mask) if r16 else None,
        )
        d = sm.attend_program(self.kv, self.sel, self.q, self.out, self.cache_layer, layer_io=lio, chip=chip)
        desc.kernels = list(desc.kernels) + list(d.kernels)
        desc.cbs = list(desc.cbs) + list(d.cbs)
        # relays on the landing cores
        rt = ttnn.RuntimeArgs()
        for i, (x, y) in enumerate(MERGE):
            q_need = 0 if self.p2_alone else sum(1 for cc in topo.chips if topo.chip(cc) // 4 == i)
            head = [q_need, int(stage == "all" and i in rows), int(stage == "all" and i in cols), i]
            rt[x][y] = head + [att(b, i) for b in range(len(sm.blocks))] + merge_words
        relay = "attn_relay16.cpp" if r16 else "attn_relay.cpp"
        desc.kernels = list(desc.kernels) + [self._dm(relay, merge_set, rt, b0, ttnn.NOC.NOC_0)]
        io = [self.kv, self.sel, self.q, self.out] + sm.io_extra() + ([self.mask] if r16 else [])
        if stage != "all":
            return desc, io

        sender = Sender(
            self.mesh,
            coord,
            SENDER,
            ready,
            self.wait_limit,
            served=self.aword(W_SERVED),
            parse_page=tp.address(self.parse) if self.parse_only else None,
            handovers=self.aword(W_HAND) if r16 else None,
        )
        if not root:
            t.reduce_ops(sender, coord)
        chunk = min(t.chunk, GROUP_BYTES)
        col_ops = {}
        virtual = self.virtual is not None
        # (16 rows: an op is handed over once per sub-pass, with its landing slot and its staging slot in the
        # hand-over words: the op's own addresses are those of slot 0)
        hand = dict(handovers=lay.subpasses) if r16 else {}
        for g in rows:  # this chip's partials of head group g: to the chip of mesh row g in its column
            slot = self.at(lay.col) + (0 if r16 else (len(col_ops) % 2) * GROUP_BYTES)
            if virtual:
                col_ops[g] = sender.op(
                    slot, GROUP_BYTES, chunk, VIRT, self.at(0), self.aword(W_COLARR), [], own=True, **hand
                )
            else:
                col_ops[g] = sender.op(
                    slot,
                    GROUP_BYTES,
                    chunk,
                    MERGE[r],
                    self.at(lay.m_col),
                    self.aword(W_COLARR),
                    [unicast((g, c))],
                    own=False,
                    **hand,
                )
        row_op = NONE
        if cols and virtual:
            row_op = sender.op(
                self.at(lay.row_stage),
                GROUP_BYTES,
                chunk,
                VIRT,
                self.at(lay.subpasses * GROUP_BYTES),
                self.aword(W_ROWARR),
                [],
                own=True,
                **hand,
            )
        elif cols:
            whole = topo.whole_mesh and topo.mesh_shape[1] > 1
            routes = mesh_branches(topo.mesh_shape, coord)[:2] if whole else [unicast((r, k)) for k in cols]
            row_op = sender.op(
                self.at(lay.row_stage),
                GROUP_BYTES,
                chunk,
                MERGE[c],
                self.at(lay.m_row),
                self.aword(W_ROWARR),
                routes,
                own=False,
                **hand,
            )
        # the coordinator on the sender core's second processor
        sx, sy = SENDER
        rt = ttnn.RuntimeArgs()
        rt[sx][sy] = (
            [r, self.at(lay.att_final), PART, self.at(lay.col), GROUP_BYTES, ready, row_op]
            + ([self.aword(W_HAND)] if r16 else [])
            + [col_ops.get(g, NONE) for g in range(8)]
            + [att(u, 0) for u in range(L.ROWS)]
        )
        coord_kernel = "attn_coord16.cpp" if r16 else "attn_coord.cpp"
        desc.kernels = list(desc.kernels) + [self._dm(coord_kernel, tp.core_set([SENDER]), rt, b1, ttnn.NOC.NOC_1)]
        # the merge cores
        ct = [len(rows), len(cols), M_ACC_MS, M_ACC_O, M_SEND_MS, M_SEND_O, M_SINK, M_FINAL, M_IN]
        col_mask, row_mask = sum(1 << g for g in rows), sum(1 << k for k in cols)
        sink_words = []
        if self.sink_args:  # the peer's max of the chip's 8 heads (its mesh row's group) as BF16 bits
            assert self.sink_peer is not None, "set_sink first"
            bits = self.sink_peer[8 * r : 8 * r + 8].to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
            sink_words = [int(v) for v in bits]
        rt = ttnn.RuntimeArgs()
        for u, (x, y) in enumerate(MERGE):
            rt[x][y] = (
                [
                    u,
                    c,
                    col_mask,
                    row_mask,
                    att(u, r),
                    self.at(lay.att_final),
                    self.at(lay.m_col),
                    self.at(lay.m_row),
                    self.noc_word(SENDER),
                    self.at(lay.row_stage),
                    self.noc_word(POST_HUB),
                    self.at(lay.post_x),
                ]
                + merge_words
                + sink_words
            )
        # what the merge kernel's unpack side waits for against what its data-movement kernel pushes in EVERY
        # outcome (feed_audit): cb_in gets exactly n_col + n_row partials (attn_merge_dm.cpp pull_stage tops up to
        # its count when a bounded wait runs out), the own partial and the sink one each
        tiles = PART // ROW_TILE
        assert bin(col_mask).count("1") == ct[0] and bin(row_mask).count("1") == ct[1], (col_mask, row_mask, ct[:2])
        # (a merge takes 1 max / sum tile and num_blocks x block_size = 16 O tiles of each operand and leaves as
        # many: 2 blocks of 8 with a 16-bit DEST, 4 of 4 with a Float32 DEST; every buffer below holds whole partials)
        blocks, block = (4, 4) if self.merge_fp32 else (2, 8)
        per_partial = 1 + blocks * block
        assert per_partial == tiles
        self.feeds[("p2", coord)] = [
            (
                "attn_merge_compute.cpp",
                "cb_in (column and row partials)",
                per_partial * (ct[0] + ct[1]),
                tiles * (len(rows) + len(cols)),
            ),
            ("attn_merge_compute.cpp", f"own partial ({blocks} blocks of {block} O tiles)", per_partial, tiles),
            ("attn_merge_compute.cpp", "sink peer", per_partial, tiles),
        ]
        if self.sink_args:
            sink = self._tiny_cb(merge_set, M_SINK, PART // ROW_TILE)
        else:
            sink = ttnn.cb_descriptor_from_sharded_tensor(M_SINK, ttnn.get_device_tensors(self.sink)[chip])
            sink.core_ranges = merge_set
            sink.format_descriptors = self._tiny_cb(merge_set, M_SINK, PART // ROW_TILE).format_descriptors
        scale = int.from_bytes(np.float32(sm.cfg.scale).tobytes(), "little")
        desc.kernels = list(desc.kernels) + [
            self._dm(
                "attn_merge16_dm.cpp" if r16 else "attn_merge_dm.cpp",
                merge_set,
                rt,
                b1,
                ttnn.NOC.NOC_1,
                ct=ct + ([M_CTL] if r16 else []),
                defines=([("HY4A_SINK_ARGS", "1")] if self.sink_args else [])
                + ([("HY4A_FIXED_ORDER", "1")] if self.fixed_order else []),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / ("attn_merge16_compute.cpp" if r16 else "attn_merge_compute.cpp")),
                core_ranges=merge_set,
                compile_time_args=ct + [M_T0_MS, M_T0_O, M_T1_MS, M_T1_O, scale] + ([M_CTL] if r16 else []),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=self.merge_fp32, math_approx_mode=False
                ),
                compiler_include_paths=self.include_paths,
            ),
        ]
        desc.cbs = list(desc.cbs) + [
            self._tiny_cb(merge_set, M_ACC_O, 16),
            self._tiny_cb(merge_set, M_ACC_MS, 1),
            self._tiny_cb(merge_set, M_SEND_MS, 1),
            self._tiny_cb(merge_set, M_SEND_O, 16),
            self._tiny_cb(merge_set, M_FINAL, 16),
            self._tiny_cb(merge_set, M_IN, 2 * (PART // ROW_TILE)),
            self._tiny_cb(merge_set, M_T0_MS, 1),
            self._tiny_cb(merge_set, M_T0_O, 16),
            self._tiny_cb(merge_set, M_T1_MS, 1),
            self._tiny_cb(merge_set, M_T1_O, 16),
            sink,
        ]
        if r16:  # one word from the data-movement kernel to the merge kernel: how many chains
            fmt = ttnn.CBFormatDescriptor(buffer_index=M_CTL, data_format=ttnn.bfloat16, page_size=64)
            desc.cbs = list(desc.cbs) + [
                ttnn.CBDescriptor(total_size=64, core_ranges=merge_set, format_descriptors=[fmt])
            ]
        # the post group
        d, _ = self.post.descriptor()
        self._wrap_hub(
            self.post,
            d,
            "attn_post_hub.cpp",
            [
                self.post.done_units(0),
                self.post.done_address(0),
                self.at(lay.post_out),
                len(MERGE),
                self.noc_word(PRE_HUB),
                self.at(lay.pre_out) + P.PRE_GATE * lay.row_tile,
                P.PRE_GATE_TILES * lay.row_tile,
                L.HEADS_PER_CHIP * P.POST_CTX_TILES,
            ]
            + t.feed_args(coord),
        )
        desc.kernels = list(desc.kernels) + list(d.kernels)
        desc.cbs = list(desc.cbs) + list(d.cbs)
        io += ([] if self.sink_args else [self.sink]) + [self.post.weights, self.post.go]
        if topo.is_head(coord):
            targets = None
            if root:
                site = [(k.x, k.y) for k in self.site.cores]
                per = len(site) // t.parts
                targets = [site[k * per : (k + 1) * per] for k in range(t.parts)]
            t.add_reduce(desc, coord, targets, self.f.buffer_address())
        if sender.ops:  # (the root alone has nothing to send)
            sender.finish(desc)
            self.parsed[("p2", coord)] = sender.parsed
            self.hand_overs[coord] = list(sender.op_handovers)  # per op: how often it is handed over in a pass
        return desc, io

    def p2(self, stage="all", layer=None):
        """(io tensors, MeshProgramDescriptor) of P2. stage "attention": the sparse attention and the relays only
        (bring-up: the partials stay on the attention cores and their counts on the merge and sender cores).
        layer: a LayerSet (default: the current set)."""
        if layer is not None:
            with self._bound(layer):
                return self.p2(stage)
        key = ("p2", stage)
        if key not in self._programs:
            assert self.post is not None and self.sm is not None, "set_weights and set_cache first"
            mesh_desc = ttnn.MeshProgramDescriptor()
            io = [self.arena, self.words] + self.t.io() + ([self.parse] if self.parse_only else [])
            for coord in self.topo.chips:
                desc, chip_io = self._p2_chip(coord, stage)
                io += chip_io
                mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
            unique = []
            for tensor in io + [self.f]:
                if not any(tensor is u for u in unique):
                    unique.append(tensor)
            self._programs[key] = (unique, mesh_desc)
        return self._programs[key]

    def run_p2(self, stage="all", layer=None):
        io, desc = self.p2(stage, layer)
        (ttnn.experimental.prepare_generic_op if is_mock() else ttnn.generic_op)(io, desc)

    # ------------------------------------------------------------------------------------------ host side
    def _replicated(self, data, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(data, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def set_mask(self, mask=0xFFFF):
        """The pass's row mask (16 rows): bit r = token row r is live. A host write of one word per core; a row that
        is not live must also have a zero count in the selection table's count row."""
        assert self.mask is not None and 0 <= mask < (1 << self.rows)
        words = torch.zeros(self.grid[0] * self.grid[1], 16, dtype=torch.int32)
        words[:, 0] = mask
        ttnn.synchronize_device(self.mesh)
        ttnn.copy_host_to_device_tensor(self._replicated(words, ttnn.uint32), self.mask)
        ttnn.synchronize_device(self.mesh)

    def set_streams(self, x):
        """[8, 4, 6144] -> the root's streams (written to every chip: only the root's are used)."""
        pages = ihc_layout.streams_to_pages(x).reshape(1, 1, 32, -1)
        ttnn.copy_host_to_device_tensor(self._replicated(pages, ttnn.float32, ttnn.TILE_LAYOUT), self.streams)

    def set_f(self, f):
        pages = ihc_layout.rows_to_pages(f).reshape(1, 1, 32, -1)
        ttnn.copy_host_to_device_tensor(self._replicated(pages, ttnn.bfloat16, ttnn.TILE_LAYOUT), self.f)

    def set_rope(self, cos, sin):
        """The interleaved RoPE tables of the 8 rows' positions ([8, 64] each, tt/frames.py rope_tables)."""
        tiles = L.to_row_tiles(torch.cat((L.rope_tile(cos), L.rope_tile(sin)), dim=1))
        ttnn.copy_host_to_device_tensor(self._replicated(tiles, ttnn.bfloat16), self.rope)
        if self.append is not None:  # the append's own copy: plain [32, 32] tiles, rows 0..7 the token rows
            cs = torch.zeros(1, 1, 32, 4 * L.TILE)
            cs[0, 0, : L.ROWS] = torch.cat((cos, sin), dim=1).float()
            ttnn.copy_host_to_device_tensor(self._replicated(cs, ttnn.bfloat16, ttnn.TILE_LAYOUT), self.cs)

    # ------------------------------------------------------------------------------------------ the latent append
    def set_kv_norm(self, weight, eps=1e-6):
        """The layer's kv_a_layernorm weight [512] (a DRAM tensor of 16 tiles, the weight on every row) and eps."""
        assert self.append is not None and tuple(weight.shape) == (L.KV_LORA,)
        tiles = weight.float().reshape(1, 1, 1, L.KV_LORA).expand(1, 1, 32, L.KV_LORA).contiguous()
        self.kv_norm = ttnn.from_torch(
            tiles, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.kv_norm_eps = float(eps)
        self._programs = {k: v for k, v in self._programs.items() if k[0] != "p1"}

    def set_positions(self, positions):
        """positions [8]: the token rows' sequence positions of this step (0xFFFFFFFF: an idle row). In place."""
        value = torch.zeros(16, dtype=torch.int64)
        value[: L.ROWS] = torch.as_tensor(positions, dtype=torch.int64) & 0xFFFFFFFF
        value = torch.where(value >= 1 << 31, value - (1 << 32), value).to(torch.int32).reshape(1, 1, 1, 16)
        ttnn.copy_host_to_device_tensor(self._replicated(value, ttnn.uint32), self.positions)

    def append_geometry(self):
        """fused/kv_store's geometry of the store set_cache gave: slot-major, ``rows_per_slot / 256`` frames a slot."""
        cfg = self.sm.cfg
        assert cfg.rows_per_slot % kv_frames.FRAME_ROWS == 0, "the frame table addresses frames of 256 stripe rows"
        kind = kv_frames.Kind("latent", cfg.num_layers, cfg.row_bytes, cfg.shard_rows * cfg.row_stride, cfg.row_stride)
        return kv_frames.Geometry.contiguous(kind, cfg.rows_per_slot // kv_frames.FRAME_ROWS)

    def set_frames(self, tables=None):
        """The frame tables of the 8 token rows (``tables``: 8 x 128 frame bases; default: the identity tables of
        the slot-major store, under which the append writes where sparse_mla's reader reads)."""
        if tables is None:
            g = self.append_geometry()
            tables = [g.identity_table(u) for u in range(L.ROWS)]
        self.frames.write([v for t in tables for v in t])

    # ------------------------------------------------------------------------------------------ the index-key append
    def set_key_norm(self, weight, bias, eps=1e-5):
        """The layer's key LayerNorm weight and bias [128] in the key's column order (tt/frames.py indexer_weights:
        [rope 64, interleaved | pass 64]) and eps, as a DRAM tensor of 12 tiles, every row the same: weight | bias,
        6 tiles each: the key's 4, then those of the rope tiles' twins (the key core normalises the rope tiles a
        second time with the members of every pair exchanged; twin column j takes sign(j) w[j ^ 1] and
        sign(j) b[j ^ 1], sign = -1 on even columns, so the normalised twin is (-k[2 i + 1], k[2 i]), the second
        operand of the rotation)."""
        assert self.append is not None and self.full and tuple(weight.shape) == tuple(bias.shape) == (L.IDX_DIM,)
        swap = torch.arange(L.ROPE) ^ 1
        sign = torch.where(torch.arange(L.ROPE) % 2 == 0, -1.0, 1.0)
        twin = lambda v: v.float()[: L.ROPE][swap] * sign
        row = torch.cat((weight.float(), twin(weight), bias.float(), twin(bias)))
        tiles = row.reshape(1, 1, 1, -1).expand(1, 1, 32, row.numel()).contiguous()
        self.key_wb = ttnn.from_torch(
            tiles, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.key_eps = float(eps)
        self._programs = {k: v for k, v in self._programs.items() if k[0] != "p1"}

    def set_keys(self, keys, rank, layers, geometry=None, tables=None):
        """The index-key store (the prefill stack's tensor or indexer_scan.keys_to_device's: per chip
        [slots, 1, stripe rows, 128] BFP8 tiles, a 32-row block of 4 tiles per DRAM bank in turn), this layer's
        ``rank`` among the store's ``layers`` layer slots a user, and the frame tables of the 8 token rows (default:
        the identity tables of the slot-major store, slot = user * layers + rank; ``geometry``: fused/kv_store's,
        default slot-major)."""
        assert self.full and self.append is not None, "the key append is P1's on a full layer with append="
        rows = keys.shape[2]
        assert rows % kv_frames.FRAME_ROWS == 0, "the frame table addresses frames of 256 stripe rows"
        kind = kv_frames.Kind("index", int(layers), L.IDX_DIM, KEY_BLOCK_BYTES, 0)
        self.key_geometry = (
            kv_frames.Geometry.contiguous(kind, rows // kv_frames.FRAME_ROWS) if geometry is None else geometry
        )
        self.keys, self.key_rank = keys, int(rank)
        assert 0 <= self.key_rank < self.key_geometry.kind.layers
        self.set_key_frames(tables)
        self._programs = {k: v for k, v in self._programs.items() if k[0] != "p1"}

    def set_key_rank(self, rank):
        """The current layer's rank among the key store's layer slots (set_keys gave the store; a LayerSet taken
        with layer_set() keeps its own rank and its own key LayerNorm tensor)."""
        assert self.key_geometry is not None and 0 <= int(rank) < self.key_geometry.kind.layers
        self.key_rank = int(rank)
        self._programs = {k: v for k, v in self._programs.items() if k[0] != "p1"}

    def set_key_frames(self, tables=None):
        if tables is None:
            tables = [self.key_geometry.identity_table(u) for u in range(L.ROWS)]
        self.key_frames.write([v for t in tables for v in t])

    # ------------------------------------------------------------------------------------------ the index query
    def set_scan(self, scan):
        """The scan whose query P1 writes (fused/indexer_scan Scan with ScanConfig(query_rows=True), 8 users = the 8
        token rows): its query tensor (per user 4 BF16 tiles [32 heads x 32 dims]) and its step tensor (a 128-byte
        page per user; P1 writes words 16..31, the 32 head-sum multipliers = the head weights, unfolded; words 4, 5,
        the context length and the centre, stay the host's)."""
        assert self.index_query, "AttnLayer(index_query=True)"
        assert scan.cfg.query_rows and scan.cfg.users == L.ROWS and scan.cfg.selectors == "l1", scan.cfg
        self.scan_qt, self.scan_step = scan.qt, scan.step
        self._programs = {k: v for k, v in self._programs.items() if k[0] != "p1"}

    def idx_prefill(self, coord, query, head_w=None):
        """What an ABSENT chip ``coord`` would have landed on the idx cores with Xa, as arena updates for every chip
        that takes part: its index head's query for the token rows [8, 128] (BF16 values, the pre group's tiles
        19..22) and, for chip 22, its pool tile, the head weights [8, 32]."""
        chip = self.topo.chip(coord)
        items = [(self.lay.idx_land + chip * P.IDX_TILES * ROW_TILE, bf16_bytes(L.to_row_tiles(query)))]
        if head_w is not None:
            assert L.pool_of(chip, True)[0] == "head_w"
            items.append((self.lay.idx_w, bf16_bytes(L.to_row_tiles(head_w))))
        return {(c, IDX): list(items) for c in self.topo.chips}

    def read_idx(self, coord, image=None):
        """ "landed": the 32 chips' index-query tiles as they landed on the chip's idx core, [32 heads, 8 token rows,
        128] (the device's frame: RoPE on the leading 64 dims, Hadamard basis); "head_w": the head-weight tile as it
        landed, [8 token rows, 32 heads]; "own": the chip's own tiles in its pre hub's output buffer, [8, 128]."""
        image = self.read_arena(coord) if image is None else image
        tiles = lambda core, off, n: L.from_row_tiles(
            bytes_bf16(self.arena_bytes(image, core, off, n * ROW_TILE), (n, ROW_TILE // 2))
        )
        landed = tiles(IDX, self.lay.idx_land, CHIPS * P.IDX_TILES).reshape(L.ROWS, CHIPS, L.IDX_DIM).permute(1, 0, 2)
        return {
            "landed": landed.contiguous(),
            "head_w": tiles(IDX, self.lay.idx_w, 1),
            "own": tiles(PRE_HUB, self.lay.pre_out + P.PRE_IDX * ROW_TILE, P.IDX_TILES),
        }

    def key_prefill(self, coord, tile):
        """What an ABSENT chip ``coord`` (one of chips 18..21 of a full layer) would have landed with Xa's pool op,
        as arena updates for every chip that takes part: its tile of the raw key [8, 32] (BF16 values)."""
        pool = L.pool_of(self.topo.chip(coord), True)
        assert pool is not None and pool[0] == "key"
        item = (A_KEY_RAW + pool[1] * ROW_TILE, bf16_bytes(L.to_row_tiles(tile)))
        return {(c, KEY): [item] for c in self.topo.chips}

    def read_key(self, coord, image=None):
        """ "raw": the key tiles as they landed, [8, 128] (wk(x) in the device's column order); "rot": the keys the
        key core finished before the Hadamard basis, BF16 [8, 128] (LayerNorm, rotation: what its matmul saw);
        "packed": the 4 BFP8 tiles it packed, bytes [4352] (the token rows are tile rows 0..7); "where": per token
        row the block it was written to on this chip as (DRAM bank, byte offset in the bank) or None; "groups": the
        blocks of the pass, each {"where", "rows": its token rows, "before": the block's bytes as the writer read
        them, "after": as DRAM holds them after the pass (read again by the writer after its writes drained)}.
        (append_keep)"""
        image = self.read_arena(coord) if image is None else image
        tiles = lambda off: L.from_row_tiles(
            bytes_bf16(self.arena_bytes(image, KEY, off, KEY_TILES * ROW_TILE), (KEY_TILES, ROW_TILE // 2))
        )
        raw = lambda off, n: np.frombuffer(self.arena_bytes(image, KEY, A_KEY_KEEP + off, n), dtype=np.uint8).copy()
        info = raw(K_INFO, 64).view(np.uint32)
        where = [None if int(w) == 0xFFFFFFFF else (int(w) & 15, int(w) & ~15) for w in info[1 : 1 + L.ROWS]]
        groups = []
        for u, w in enumerate(where):  # (attn_key_rows.hpp group_rows: a group per distinct block, in token-row order)
            if w is None:
                continue
            for g in groups:
                if g["where"] == w:
                    g["rows"].append(u)
                    break
            else:
                n = len(groups)
                groups.append(
                    {
                        "where": w,
                        "rows": [u],
                        "before": raw(K_BEFORE + n * KEY_BLOCK_BYTES, KEY_BLOCK_BYTES),
                        "after": raw(K_AFTER + n * KEY_BLOCK_BYTES, KEY_BLOCK_BYTES),
                    }
                )
        assert int(info[0]) == len(groups), (int(info[0]), len(groups))
        return {
            "raw": tiles(A_KEY_RAW),
            "rot": tiles(A_KEY_KEEP + K_ROT),
            "packed": raw(K_PACKED, KEY_BLOCK_BYTES),
            "where": where,
            "groups": groups,
        }

    def pool_prefill(self, coord, tile):
        """What an ABSENT chip ``coord`` (one of chips 0..17) would have landed with Xa's pool op, as arena updates
        for every chip that takes part: its pool tile [8, 32] (BF16 values)."""
        chip = self.topo.chip(coord)
        assert (L.pool_of(chip, self.full) or ("", 0))[0] == "kv"
        item = (A_AP_RAW + chip * ROW_TILE, bf16_bytes(L.to_row_tiles(tile)))
        return {(c, APPEND): [item] for c in self.topo.chips}

    def read_append(self, coord, image=None):
        """ "raw": the pool tiles as they landed, [8, 576] (kv_a(x): k_pass | k_rope before the norm and the
        rotation); "rows": the rows the append core finished, BF16 [8, 576] (k_pass after the kv norm | k_rope after
        the rotation): what its cast saw and what it stored as the RoPE part (append_keep)."""
        image = self.read_arena(coord) if image is None else image
        tiles = lambda off: L.from_row_tiles(
            bytes_bf16(self.arena_bytes(image, APPEND, off, AP_TILES * ROW_TILE), (AP_TILES, ROW_TILE // 2))
        )
        return {"raw": tiles(A_AP_RAW), "rows": tiles(A_AP_LAT)}

    def _core_index(self, core) -> int:
        return core[1] * self.grid[0] + core[0]

    def read_arena(self, coord) -> np.ndarray:
        """A chip's arena as bytes [cores, ARENA_BYTES]; core (x, y) is row y * grid_x + x."""
        ttnn.synchronize_device(self.mesh)
        t = ttnn.to_torch(ttnn.get_device_tensors(self.arena)[self.topo.chip(coord)])
        return (
            t.to(torch.int32)
            .numpy()
            .astype(np.uint32)
            .reshape(self.grid[0] * self.grid[1], self.lay.arena // 4)
            .view(np.uint8)
        )

    def arena_bytes(self, image, core, offset, nbytes) -> bytes:
        return image[self._core_index(core), offset : offset + nbytes].tobytes()

    def write_arena(self, updates):
        """updates: {(coord, core): [(offset, bytes), ...]}. Overwrites the whole arena of every chip (ranges not
        named keep what earlier calls of this function wrote, zero otherwise): a host write, mesh idle, for
        bring-up rungs only."""
        chips = self.mesh.get_num_devices()
        cores = self.grid[0] * self.grid[1]
        if self._shadow is None:
            self._shadow = {}
        for (coord, core), items in updates.items():
            image = self._shadow.setdefault(self.topo.chip(coord), np.zeros((cores, self.lay.arena), dtype=np.uint8))
            for offset, raw in items:
                image[self._core_index(core), offset : offset + len(raw)] = np.frombuffer(bytes(raw), dtype=np.uint8)
        zero = None
        shards = []
        for chip in range(chips):
            if chip in self._shadow:
                data = torch.from_numpy(self._shadow[chip].view(np.int32).reshape(-1, 256).copy())
            else:
                zero = torch.zeros(cores * self.lay.arena // 1024, 256, dtype=torch.int32) if zero is None else zero
                data = zero
            shards.append(ttnn.from_torch(data, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT))
        ttnn.synchronize_device(self.mesh)
        ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, self.mesh.shape), self.arena)
        ttnn.synchronize_device(self.mesh)

    def read_words(self) -> dict:
        ttnn.synchronize_device(self.mesh)
        tensors = ttnn.get_device_tensors(self.words)
        return {
            c: ttnn.to_torch(tensors[self.topo.chip(c)]).reshape(-1, self.n_words).to(torch.int64) & 0xFFFFFFFF
            for c in self.topo.chips
        }

    def status(self, p1_only=False) -> dict:
        """Everything that is not at rest after a run: the transport's words and this piece's words of every chip
        that takes part. Empty when the run completed. p1_only: P1 ran alone, so the arrivals of X3 are still
        counted on the landing cores (P2's relays take them): each must be the number of chips of its head group
        that take part."""
        out = dict(self.t.status())
        landed = {}
        for c in self.topo.chips:
            g = self.topo.chip(c) // 4
            landed[g] = landed.get(g, 0) + 1
        for coord, words in self.read_words().items():
            for i in range(words.shape[0]):
                core = (i % self.grid[0], i // self.grid[0])
                want = {W_QARR: landed.get(MERGE.index(core), 0)} if p1_only and core in MERGE else {}
                bad = {w: hex(int(words[i][w])) for w in REST_WORDS if int(words[i][w]) != want.get(w, 0)}
                if bad:
                    out[(coord, core, "attn")] = bad
        return out

    def reset_words(self):
        """This piece's words to zero (after P1 ran alone: the X3 arrivals nobody took)."""
        zeros = torch.zeros(self.grid[0] * self.grid[1], self.n_words, dtype=torch.int32)
        ttnn.synchronize_device(self.mesh)
        ttnn.copy_host_to_device_tensor(self._replicated(zeros, ttnn.uint32), self.words)
        ttnn.synchronize_device(self.mesh)

    def proj_status(self, image, prog):
        """A group's status words (uint32) from a chip's arena image."""
        raw = self.arena_bytes(image, (prog.hub.x, prog.hub.y), prog.offsets["status"], 4 * P.STATUS_WORDS)
        return [int(v) for v in np.frombuffer(raw, dtype=np.uint32)]

    def reset(self):
        """After a run that left a status: every word to zero (transport, this piece, the groups' go words)."""
        self.t.reset()
        zeros = torch.zeros(self.grid[0] * self.grid[1], self.n_words, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(self._replicated(zeros, ttnn.uint32), self.words)
        for prog, rect in ((self.pre, PRE_RECT), (self.query, QUERY_RECT), (self.post, POST_RECT)):
            (x0, y0), (x1, y1) = rect
            go = torch.zeros((x1 - x0 + 1) * (y1 - y0 + 1), 16, dtype=torch.int32)
            ttnn.copy_host_to_device_tensor(self._replicated(go, ttnn.uint32), prog.go)
        ttnn.synchronize_device(self.mesh)

    def read_root(self):
        """What the root's site holds: y [8, 6144] fp32, 1 / RMS [8], x [8, 6144] (the BF16 it broadcast), F."""
        root = self.topo.chip(self.topo.root)
        if self.rows != L.ROWS:  # (16 rows, P2 only: F [16, 6144] from its own buffer, [16, 32] tiles in core order)
            raw = ttnn.to_torch(ttnn.get_device_tensors(self.f)[root]).float()
            return {"f": L.from_row_tiles(raw.reshape(-1, self.lay.row_tile // 2))}
        m = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[root]).float()[0, 0]
        rows = lambda t: ihc_layout.pages_to_rows(m(t))
        stat = ihc_layout.stat_from_page(m(self.site.stat)[:, :32])
        return {"y": rows(self.y), "inv_rms": stat, "x": rows(self.xn), "f": rows(self.f)}

    def read_streams(self):
        root = self.topo.chip(self.topo.root)
        return ihc_layout.pages_to_streams(ttnn.to_torch(ttnn.get_device_tensors(self.streams)[root]).float()[0, 0])

    def read_p1(self, coord, image=None):
        """A chip's P1 buffers from its arena: x as it arrived [8, 6144]; the pre group's output (q_a [8, 64], pool
        [8, 32], gate [8, 512]) and lane sums [8, 8]; the query hub's x buffer (q_a whole [8, 2048]) and the lane
        landing [32 chips, 8, 8]; the query group's output [2 heads, 8, 576]; the head group's query landing on the
        merge core [4 chips, 2 heads, 8, 576]."""
        image = self.read_arena(coord) if image is None else image
        tiles = lambda core, off, n: L.from_row_tiles(
            bytes_bf16(self.arena_bytes(image, core, off, n * ROW_TILE), (n, ROW_TILE // 2))
        )
        lay = self.lay
        pre_out = tiles(PRE_HUB, lay.pre_out, P.PRE_IDX)
        stat = bytes_f32(self.arena_bytes(image, PRE_HUB, lay.pre_stat, 2 * ROW_TILE)).reshape(2, L.ROWS, 16)
        lanes = bytes_f32(self.arena_bytes(image, QUERY_HUB, lay.q_lanes, CHIPS * STAT_PAGE)).reshape(CHIPS, 4, 16, 16)
        q_out = tiles(QUERY_HUB, lay.q_out, L.HEADS_PER_CHIP * P.LAT_TILES)
        group = self.topo.chip(coord) // 4
        landing = tiles(MERGE[group], A_M_QUERY, 4 * L.HEADS_PER_CHIP * P.LAT_TILES)
        return {
            "x": tiles(PRE_HUB, A_PRE_X, P.PRE_HIDDEN_TILES),
            "q_a": pre_out[:, : 2 * L.TILE],
            "pool": pre_out[:, P.PRE_POOL * L.TILE : P.PRE_GATE * L.TILE],
            "gate": pre_out[:, P.PRE_GATE * L.TILE : P.PRE_IDX * L.TILE],
            "lanes": stat[0, :, 0::2],
            "q_a_all": tiles(QUERY_HUB, A_Q_X, P.Q_A_TILES),
            "lanes_all": lanes[:, 0, : L.ROWS, 0::2],
            "q_lat": q_out.reshape(L.ROWS, L.HEADS_PER_CHIP, L.LATENT).permute(1, 0, 2),
            "q_landing": landing.reshape(L.ROWS, 4, L.HEADS_PER_CHIP, L.LATENT).permute(1, 2, 0, 3),
            "status": {"pre": self.proj_status(image, self.pre), "query": self.proj_status(image, self.query)},
        }

    def xa_prefill(self, coord, q_a, lanes):
        """What an ABSENT chip ``coord`` would have landed in Xa, as arena updates for every chip that takes part:
        its q_a tiles [8, 64] (BF16 values) and its lane sums [8, 8] (fp32)."""
        chip = self.topo.chip(coord)
        page = torch.zeros(4, 16, 16)
        page[0, : L.ROWS, 0::2] = lanes.float()
        items = [
            (self.lay.q_x + 2 * chip * ROW_TILE, bf16_bytes(L.to_row_tiles(q_a))),
            (self.lay.q_lanes + chip * STAT_PAGE, page.numpy().astype(np.float32).tobytes()),
        ]
        return {(c, QUERY_HUB): list(items) for c in self.topo.chips}

    def write_words(self, values):
        """values: {(coord, core): {word index: value}} of this piece's words; every other word becomes zero. A
        host write (bring-up: the arrivals a program that did not run would have counted)."""
        cores = self.grid[0] * self.grid[1]
        shards = []
        for chip in range(self.mesh.get_num_devices()):
            data = torch.zeros(cores, self.n_words, dtype=torch.int32)
            for (coord, core), words in values.items():
                if self.topo.chip(coord) == chip:
                    for index, value in words.items():
                        data[self._core_index(core), index] = value
            shards.append(ttnn.from_torch(data, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT))
        ttnn.synchronize_device(self.mesh)
        ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, self.mesh.shape), self.words)
        ttnn.synchronize_device(self.mesh)

    def p2_prefill(self, q_lat, gates):
        """What P1 leaves for P2 on every chip that takes part, as (arena updates, word values), for a P2 that runs
        alone: q_lat [64 heads, 8 users, 576] the absorbed queries of all heads (as the query units pack them:
        [8 users, 32] tiles per head) on the 8 landing cores, with the arrival counts the relays wait for; gates
        {coord: [8, 512]} the chip's gate in its pre hub's output buffer."""
        updates, words = {}, {}
        for coord in self.topo.chips:
            for g, core in enumerate(MERGE):
                raw = b"".join(bf16_bytes(L.to_row_tiles(q_lat[h])) for h in range(8 * g, 8 * g + 8))
                updates[(coord, core)] = [(self.lay.m_query, raw)]
                if not self.p2_alone:  # (p2_alone: the relays wait for no arrival count and take none)
                    words[(coord, core)] = {W_QARR: sum(1 for cc in self.topo.chips if self.topo.chip(cc) // 4 == g)}
            gate = (self.lay.pre_out + P.PRE_GATE * self.lay.row_tile, bf16_bytes(L.to_row_tiles(gates[coord])))
            updates[(coord, PRE_HUB)] = [gate]
        return updates, words

    @staticmethod
    def partial_bytes(o, m, s):
        """One partial (O [8 heads, 512], max [8], sum [8]) as it travels: [max / sum tile | 16 O tiles], 8,704 B."""
        ms = torch.zeros(L.ROWS, L.TILE)
        ms[:, 0], ms[:, 1] = m, s
        return bf16_bytes(SL.pack_tiny(ms)) + bf16_bytes(SL.pack_tiny(o))

    def p2_virtual_prefill(self, col_partials, row_partials, mask=None):
        """The landings of the virtual partners of the one chip that takes part, as (arena updates, word values):
        col_partials {mesh row: [per user (O [8, 512], max [8], sum [8])]}, row_partials {mesh column: the same}:
        what each partner would have landed for this chip's head group. 16 rows: per sub-pass with a live row
        (``mask``) the partials of its live rows, in row order, in the sub-pass's slot of the landing, and as many
        arrivals counted as there are such sub-passes."""
        (coord,) = self.topo.chips
        mask = (1 << self.rows) - 1 if mask is None else mask
        updates, words = {}, {}
        for table, offset, word in ((col_partials, self.lay.m_col, W_COLARR), (row_partials, self.lay.m_row, W_ROWARR)):
            for index, users in table.items():
                arrivals = 0
                for p in range(self.lay.subpasses):
                    live = [u for u in range(L.ROWS * p, L.ROWS * (p + 1)) if (mask >> u) & 1]
                    if not live:
                        continue
                    raw = b"".join(self.partial_bytes(*users[u]) for u in live)
                    updates.setdefault((coord, MERGE[index]), []).append((offset + p * GROUP_BYTES, raw))
                    arrivals += 1
                if arrivals:
                    words.setdefault((coord, MERGE[index]), {})[word] = arrivals
        return updates, words

    def read_p2(self, coord, image=None):
        """A chip's P2 buffers from its arena: the attention cores' final partials (O [8 users, 64 heads, 512], max
        and sum [8, 64]); ctx of the chip's 2 heads in the post hub's x buffer [2, 8 users, 512] and the gate behind
        it [8, 512]; the chip's partial of the sublayer output [8, 6144]; the post group's status words."""
        image = self.read_arena(coord) if image is None else image
        tiles = lambda core, off, n: L.from_row_tiles(
            bytes_bf16(self.arena_bytes(image, core, off, n * ROW_TILE), (n, ROW_TILE // 2))
        )
        users, groups, lay = self.rows, 8, self.lay
        # (the hubs' buffers hold [rows, 32] tiles; a partial is [8 heads, 32] tiles whatever the rows, and the one of
        # user u is on S-block u % 8, the second of a core's two behind the first)
        hub = lambda off, n: L.from_row_tiles(
            bytes_bf16(self.arena_bytes(image, POST_HUB, off, n * lay.row_tile), (n, lay.row_tile // 2))
        )
        o = torch.zeros(users, 64, L.KV_LORA)
        m, sm_ = torch.zeros(users, 64), torch.zeros(users, 64)
        for u in range(users):
            for g in range(groups):
                core = self.sm.blocks[u % groups].attention[g]
                part = tiles(core, lay.att_final + (u // groups) * PART, PART // ROW_TILE)  # [8 heads, 17 * 32]
                m[u, 8 * g : 8 * g + 8], sm_[u, 8 * g : 8 * g + 8] = part[:, 0], part[:, 1]
                o[u, 8 * g : 8 * g + 8] = part[:, L.TILE :]
        x = hub(lay.post_x, P.POST_X_TILES)
        n_ctx = L.HEADS_PER_CHIP * L.KV_LORA
        return {
            "o": o,
            "max": m,
            "sum": sm_,
            "ctx": x[:, :n_ctx].reshape(users, L.HEADS_PER_CHIP, L.KV_LORA).permute(1, 0, 2),
            "gate": x[:, n_ctx:],
            "out": hub(lay.post_out, HIDDEN // L.TILE),
            "status": self.proj_status(image, self.post),
        }

    def release(self):
        """Free this layer's L1 and weight tensors (a second layer in the same process does not fit beside them)."""
        ttnn.synchronize_device(self.mesh)
        tensors = [self.sink, self.parse, self.t.words, self.gamma, self.q, self.out, self.mask, self.rope]
        tensors += [self.kv_norm, self.cs, self.positions, self.frames.tensor if self.frames is not None else None]
        tensors += [self.key_wb, self.key_h, self.key_frames.tensor if self.key_frames is not None else None]
        tensors += [self.f] if self.f is not self.site_f else []
        if self.host is None:  # (a second program set leaves its host's tensors and the streams it reads alone)
            tensors += [self.arena, self.words, self.xn, self.y, self.site_f, self.streams]
            tensors += [self.site.coef, self.site.stat]
        for prog in (self.pre, self.query, self.post):
            if prog is not None:
                tensors += [prog.go] + ([] if getattr(self, "shared_streams", False) else [prog.weights])
        for tensor in tensors:
            if tensor is not None:
                ttnn.deallocate(tensor)
        self._programs = {}
        if self.sm is not None:
            # sparse_mla's 6 global semaphores are L1 buffers (384 B) that live as long as their objects: left
            # behind they pin the address space and a larger layer built afterwards does not fit
            import gc

            self.sm.semaphores = None
            self.sm = None
            gc.collect()

    def feed_audit(self, program, coord):
        """Per compute kernel of this piece on the chip: the tiles its unpack side waits for on a buffer and the
        tiles its data-movement kernel pushes there in every outcome, a bounded wait running out included. They
        must be equal: a compute kernel left waiting is not recovered by the operation timeout. (The pieces' own
        compute kernels are fed by their own data-movement kernels, which push after their bounded waits whatever
        these gave: sm_krecv.cpp, sm_attn_io.cpp's receive slots and query pages, proj_writer.cpp, reduce_dm.cpp.)"""
        rows = self.feeds[(program, tuple(coord))]
        bad = [row for row in rows if row[2] != row[3]]
        assert not bad, f"a compute kernel waits for tiles nobody is certain to push: {bad}"
        return rows

    def l1_bytes(self) -> int:
        return ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1).total_bytes_allocated_per_bank
