# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``sparse_sdpa_tiled``: block-shared sparse attention for the prefill slab, one ``generic_op`` program.

The slab's QSA attention today expands every query's 512 selected blocks to 2048 token ids (five ops over a
``[rows, 2080]`` uint32 tensor per layer), pads the six query heads to 32 with a zero V half, and runs
``ttnn.transformer.sparse_sdpa``, which gathers each query's own 2048 K/V rows: 51.5 GB of cache traffic per
2048-row slab.  Consecutive queries select overlapping blocks (a tile of 16 queries needs 1.0-3.5x one query's
blocks, not 16x), so this program works per tile of ``TQ`` consecutive queries x all local heads on one core: the
reader builds the tile's block union on the RISC (a seed block per row first, then the ids ascending) and one
membership word per union slot, the union's K/V rows stream once per tile in chunks of ``CB`` blocks through the
dual-NoC transaction-id ring gather of ``sparse_sdpa``, the writer turns the membership words into an additive bf16
mask band per chunk, and compute runs ``sparse_sdpa``'s streaming flash loop with the band added before the running
max.  The attended set per query is exactly today's (the selected complete blocks' tokens plus the tail of its own
incomplete block); the visiting order and the chunk size differ, so the class is tolerance against
``sparse_sdpa`` (bit-exact run to run).  docs/PREFILL.md (the block-shared attention section) and docs/NUMERICS.md carry the
numbers and the class.

Inputs (per device): ``q`` ``[1, H, S, Dq]`` bf16 ROW_MAJOR (the local heads' query rows, Dq = 256); ``kv`` the packed
``[1, 1, T, W]`` bf16 ROW_MAJOR cache (1 KB rows ``[V | K]``, W = 512); ``block_ids`` ``[1, 1, S, IDS]`` uint32
ROW_MAJOR (row j: the top-k blocks in descending score, the first ``min(IDS, complete_j)`` of them valid, where
``complete_j = (p_j + 1) // block_tokens``); ``positions`` ``[1, 1, 1, S]`` uint32 ROW_MAJOR (``p_j`` = the absolute
position of row j).  Output ``[1, H, S, v_dim]`` bf16 ROW_MAJOR.  ``reference_fp32`` is the exact-arithmetic oracle,
``tile_plan`` the host image of the reader's union / membership passes and the writer's bands (the kernel's spec in
code), ``expand_today`` the host form of the slab's expansion for the comparative column, and
``sparse_sdpa_tiled_composed`` today's device chain on the same inputs.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

import ttnn

from .. import program as fp
from ..registry import COMPONENT, FusedKernel, register

NAME = "sparse_sdpa_tiled"
READER = fp.kernel_source(NAME, "reader.cpp")
WRITER = fp.kernel_source(NAME, "writer.cpp")
COMPUTE = fp.kernel_source(NAME, "compute.cpp")

BF16, FP32, U32 = ttnn.bfloat16, ttnn.float32, ttnn.uint32
TILE = fp.TILE
TILE_BF16, TILE_FP32 = fp.TILE_BYTES[BF16], fp.TILE_BYTES[FP32]
MASKED_INDEX = 0xFFFFFFFF  # sparse_sdpa's sentinel (today's expansion)
MASK_FLOOR = -3.3895313892515355e38  # the most negative finite bf16 (0xFF7F): the band's hidden value
MSG_WORDS = 4  # kernels/common.h sst::MSG_WORDS
MSG_PAGE_BYTES = MSG_WORDS * 4
# The usable L1 for the program's circular buffers on Blackhole: 1497 KiB (MEM_MAX_KERNEL_SIZE) less the kernel
# config region (~69 KB): 1427 KB per the design note; keep 27 KB for the kernel binaries' own use.
L1_BUDGET_BYTES = 1400 * 1024

# CB ids (the single source; the kernels take them as named compile-time args)
CB_Q_RM, CB_Q_IN, CB_K_RM, CB_K_IN, CB_SCALE, CB_COL_IDENTITY, CB_MASK_BAND, CB_QK_IM = range(8)
CB_MAX_A, CB_MAX_B, CB_SUM_A, CB_SUM_B, CB_OUT_A, CB_OUT_B, CB_CORR, CB_OUT_IM, CB_OUT_RM, CB_RECIP = range(8, 18)
CB_CTRL, CB_KREQ, CB_KACK, CB_TILECTL, CB_IDS, CB_POS, CB_BITMAP, CB_SEEDMAP, CB_RANK, CB_UNION, CB_MEMBER = range(
    18, 29
)
CB_BAND_TEMPLATE = 29


@dataclass(frozen=True)
class Config:
    """The launch parameters (all hashed into the program).  ``tile_queries`` x heads rows per core (a multiple
    of 32), ``chunk_blocks`` blocks per streamed chunk (``chunk_blocks * block_tokens`` a multiple of 32, at least
    ``tile_queries`` so chunk 0 holds every row's seed), the K / V column offsets and the V width inside the packed
    row (multiples of 32), the gather ring depth and the writer's share of each chunk's rows (``split_num /
    split_den``), the running-state format, the compute config (the slab's: HiFi4, exact correction exp, fp32
    DEST) and the seed prefix (off only for the negative test)."""

    tile_queries: int = 16
    chunk_blocks: int = 64
    block_tokens: int = 4
    k_offset: int = 256
    v_offset: int = 0
    v_dim: int = 256
    ring_depth: int = 8
    split_num: int = 3
    split_den: int = 8
    # bf16 running state (the chain's format class) by default: measured no farther from the fp32 oracle than the
    # chain in either format (the approximate probability exp shapes the row); fp32 state tightens the normalization
    # alone (ones-V column 0.4 % vs 1.2 %) at +176 KB of L1, and stays a switch (2026-09-25).
    fp32_state: bool = False
    fidelity: Any = ttnn.MathFidelity.HiFi4
    approx: bool = False
    fp32_dest: bool = True
    seed: bool = True
    # Row-major K/V chunk slots in L1 (the reader runs this many chunks ahead of compute; 2 fits with bf16 state).
    k_slots: int = 1
    # Study builds only: 1 = chunk-0 scores after the band, 2 = raw scores, 3 = probabilities leave as the output;
    # 4 = compute skips the math (the gather / union / band path alone), 5 = the gather skips its NoC reads (the
    # compute path alone, on whatever the K/V slots hold), 6 = both (the union / band / handshake floor).
    debug_stage: int = 0

    @property
    def k_chunk(self) -> int:
        return self.chunk_blocks * self.block_tokens

    @property
    def skt(self) -> int:
        return self.k_chunk // TILE

    def union_max(self, ids: int) -> int:
        """Slots the union can hold: every valid id of the tile plus the distinct diagonal blocks of its span (never
        among a row's valid ids; ``tile_queries`` consecutive positions span at most ``ceil(TQ / BT) + 1`` of them),
        rounded up to whole chunks (the reader clears the padding slots of the last chunk)."""

        bound = self.tile_queries * ids + -(-self.tile_queries // self.block_tokens) + 1
        return -(-bound // self.chunk_blocks) * self.chunk_blocks

    def dst_size(self) -> int:
        return 4 if self.fp32_dest else 8

    def query_subblock(self, sqt: int) -> int:
        for d in range(min(sqt, self.dst_size()), 0, -1):
            if sqt % d == 0:
                return d
        return 1


DEFAULT = Config()


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def geometry(q, kv, block_ids, positions, config: Config = DEFAULT) -> dict[str, int]:
    """The program's integer geometry from the tensors; raises on any shape, dtype or layout the kernels were not
    written for (``admits`` is this check as a predicate)."""

    qs, kvs, ids_s, ps = _shape(q), _shape(kv), _shape(block_ids), _shape(positions)
    if len(qs) != 4 or qs[0] != 1 or q.dtype != BF16 or q.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"q must be [1, H, S, Dq] bf16 ROW_MAJOR, got {q.layout} {q.dtype} {qs}")
    if len(kvs) != 4 or kvs[:2] != (1, 1) or kv.dtype != BF16 or kv.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"kv must be [1, 1, T, W] bf16 ROW_MAJOR, got {kv.layout} {kv.dtype} {kvs}")
    _, H, S, Dq = qs
    T, W = kvs[2], kvs[3]
    if ids_s[:3] != (1, 1, S) or len(ids_s) != 4 or block_ids.dtype != U32 or block_ids.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(
            f"block_ids must be [1, 1, {S}, IDS] uint32 ROW_MAJOR, got {block_ids.layout} {block_ids.dtype} {ids_s}"
        )
    if ps != (1, 1, 1, S) or positions.dtype != U32 or positions.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(
            f"positions must be [1, 1, 1, {S}] uint32 ROW_MAJOR, got {positions.layout} {positions.dtype} {ps}"
        )
    IDS = ids_s[3]
    return _geometry_of(H, S, T, W, Dq, IDS, config)


def _geometry_of(H: int, S: int, T: int, W: int, Dq: int, IDS: int, config: Config) -> dict[str, int]:
    TQ, CB, BT = config.tile_queries, config.chunk_blocks, config.block_tokens
    if not 1 <= TQ <= 32 or (TQ * H) % TILE:
        raise ValueError(f"tile_queries {TQ} x {H} heads must be whole tile rows (a multiple of {TILE}) with TQ <= 32")
    if 32 % TQ or TQ % 16:
        # 32 rows of a band tile hold whole copies of the TQ patterns; the positions read (TQ x 4 bytes at t0 x 4)
        # keeps the 64-byte DRAM phase only when TQ is a multiple of 16
        raise ValueError(f"tile_queries must be 16 or 32, got {TQ}")
    if S % TQ:
        raise ValueError(f"S {S} must be a multiple of tile_queries {TQ}")
    if BT != 4:
        raise ValueError(f"block_tokens must be 4 (the band builder writes 4-token blocks), got {BT}")
    if (CB * BT) % TILE or CB < TQ:
        raise ValueError(f"chunk_blocks {CB}: {CB} x {BT} must be a multiple of {TILE} and CB >= tile_queries {TQ}")
    if T % BT:
        raise ValueError(f"the cache length {T} must be a multiple of block_tokens {BT}")
    if Dq % TILE or W % TILE or config.k_offset % TILE or config.v_offset % TILE or config.v_dim % TILE:
        raise ValueError("head dims, the packed row width, the K / V offsets and v_dim must be multiples of 32")
    if config.k_offset + Dq > W or config.v_offset + config.v_dim > W:
        raise ValueError(
            f"K [{config.k_offset}, +{Dq}) and V [{config.v_offset}, +{config.v_dim}) must lie inside the {W}-wide row"
        )
    if IDS % 4 or IDS < 1 or (IDS * 4) % 16:
        raise ValueError(f"the block-id row must be whole 16-byte pages, got {IDS} ids")
    if not 1 <= config.ring_depth <= 15 or not 0 <= config.split_num <= config.split_den or config.split_den < 1:
        raise ValueError("ring_depth in 1..15, 0 <= split_num <= split_den")
    if not 1 <= config.k_slots <= 4:
        raise ValueError(f"k_slots in 1..4, got {config.k_slots}")
    R = TQ * H
    Sqt = R // TILE
    return {
        "H": H,
        "S": S,
        "T": T,
        "W": W,
        "Dq": Dq,
        "IDS": IDS,
        "R": R,
        "Sqt": Sqt,
        "DQt": Dq // TILE,
        "DHt": W // TILE,
        "vDHt": config.v_dim // TILE,
        "Skt": config.skt,
        "k_chunk": config.k_chunk,
        "max_blocks": T // BT,
        "words": -(-(T // BT) // 32),
        "U_max": config.union_max(IDS),
        "num_tiles": S // TQ,
        "qsb": config.query_subblock(Sqt),
    }


def admits_shapes(H: int, S: int, T: int, W: int, Dq: int, IDS: int, config: Config = DEFAULT) -> bool:
    """The geometry contract on the shapes alone (the model decides before it builds its tensors): the tile rows are
    whole tiles, S divides into tiles, the chunk holds every seed, the offsets lie in the row, the circular buffers
    fit L1."""

    try:
        g = _geometry_of(H, S, T, W, Dq, IDS, config)
    except ValueError:
        return False
    return l1_bytes(g, config) <= L1_BUDGET_BYTES


FIDELITY_ENV = "QWEN38_SPARSE_SDPA_TILED_FIDELITY"
FIDELITIES = ("hifi2", "hifi4")
DEFAULT_FIDELITY = "hifi2"
# The served config: the op family's HiFi2 / approximate correction exp / bf16-DEST compute config on the bf16
# running state.  Chosen on the 4-chip line (2026-09-26, 32k, the 31,716-token prompt): closer to the control than the
# slab's HiFi4 / fp32-DEST form (KL 0.0028 vs 0.0059 mean, top-1 160/160 vs 159/160) and faster (4.65 -> 4.14 ms per
# layer at 28k on an 11x10 die).
HIFI2 = replace(DEFAULT, fidelity=ttnn.MathFidelity.HiFi2, approx=True, fp32_dest=False)


def model_config(environ: Mapping[str, str] | None = None) -> Config:
    """The served base config (``HIFI2``); ``QWEN38_SPARSE_SDPA_TILED_FIDELITY=hifi4`` restores the slab's HiFi4 /
    exact correction exp / fp32-DEST compute config (``DEFAULT``, the kernel's validation config).  Resolved once by
    the model at construction."""

    environ = os.environ if environ is None else environ
    raw = environ.get(FIDELITY_ENV, DEFAULT_FIDELITY).strip().lower()
    if raw not in FIDELITIES:
        raise ValueError(f"{FIDELITY_ENV} must be one of {FIDELITIES}, got {raw!r}")
    return HIFI2 if raw == "hifi2" else DEFAULT


def config_for_grid(S: int, cores: int, base: Config = DEFAULT) -> Config:
    """``base`` when its 16-query tiles fit the grid one per core (128 tiles on the 130-core p150 grid), else 32-query
    tiles with 32-block chunks (the L1 table's config B): a smaller grid would carry two 16-query tiles on some cores
    and set the layer time by them (measured 5.5 vs 4.8 ms per layer at 28k on a 110-core die)."""

    if S % base.tile_queries == 0 and S // base.tile_queries <= cores:
        return base
    return replace(base, tile_queries=32, chunk_blocks=min(base.chunk_blocks, 32))


def admits(q, kv, block_ids, positions, *, config: Config = DEFAULT, **_ignored) -> bool:
    """The geometry contract on the tensors plus the L1 budget (what ``build`` would refuse)."""

    try:
        g = geometry(q, kv, block_ids, positions, config)
    except (ValueError, AttributeError, TypeError):
        return False
    return l1_bytes(g, config) <= L1_BUDGET_BYTES


def cb_table(g: dict[str, int], config: Config) -> list[tuple[int, Any, int, int]]:
    """``(index, format, page_bytes, pages)`` per circular buffer of the program (the L1 table of the design note)."""

    state = FP32 if config.fp32_state else BF16
    state_tile = TILE_FP32 if config.fp32_state else TILE_BF16
    Sqt, Skt, DQt, DHt, vDHt = g["Sqt"], g["Skt"], g["DQt"], g["DHt"], g["vDHt"]

    def aligned(bytes_: int) -> int:
        return -(-bytes_ // 16) * 16

    return [
        (CB_Q_RM, BF16, g["Dq"] * 2, g["R"]),
        (CB_Q_IN, BF16, TILE_BF16, Sqt * DQt),
        (CB_K_RM, BF16, g["W"] * 2, g["k_chunk"] * config.k_slots),
        (CB_K_IN, BF16, TILE_BF16, Skt * DHt),
        (CB_SCALE, BF16, TILE_BF16, 1),
        (CB_COL_IDENTITY, BF16, TILE_BF16, 1),
        (CB_MASK_BAND, BF16, TILE_BF16, 2 * Skt),
        (CB_BAND_TEMPLATE, BF16, Skt * TILE_BF16, 1),
        # exactly Sqt x Skt pages: the held-write-pointer scheme rewinds to the same base every group, so a second
        # buffer would desynchronize the pack base from the read pointer
        (CB_QK_IM, state, state_tile, Sqt * Skt),
        (CB_MAX_A, state, state_tile, Sqt),
        (CB_MAX_B, state, state_tile, Sqt),
        (CB_SUM_A, state, state_tile, Sqt),
        (CB_SUM_B, state, state_tile, Sqt),
        (CB_OUT_A, state, state_tile, Sqt * vDHt),
        (CB_OUT_B, state, state_tile, Sqt * vDHt),
        (CB_CORR, state, state_tile, Sqt),
        (CB_OUT_IM, BF16, TILE_BF16, Sqt * vDHt),
        (CB_OUT_RM, BF16, TILE_BF16, Sqt * vDHt),
        (CB_RECIP, state, state_tile, 1),
        (CB_CTRL, U32, MSG_PAGE_BYTES, 2),
        (CB_KREQ, U32, MSG_PAGE_BYTES, 2),
        (CB_KACK, U32, MSG_PAGE_BYTES, 2),
        (CB_TILECTL, U32, MSG_PAGE_BYTES, 2),
        (CB_IDS, U32, config.tile_queries * g["IDS"] * 4, 1),
        (CB_POS, U32, aligned(config.tile_queries * 4), 1),
        (CB_BITMAP, U32, aligned(g["words"] * 4), 1),
        (CB_SEEDMAP, U32, aligned(g["words"] * 4), 1),
        (CB_RANK, U32, aligned(g["words"] * 2), 1),
        (CB_UNION, U32, g["U_max"] * 4, 1),
        (CB_MEMBER, U32, g["U_max"] * 4, 1),
    ]


def l1_bytes(g: dict[str, int], config: Config) -> int:
    return sum(page * pages for _, _, page, pages in cb_table(g, config))


def build(q, kv, block_ids, positions, out, *, scale: float, config: Config = DEFAULT):
    """The program descriptor over the compute grid (tiles split contiguously, several per core when the grid is
    smaller than the tile count) and the io list; ``out`` is the pre-allocated ``[1, H, S, v_dim]`` bf16 ROW_MAJOR
    output."""

    g = geometry(q, kv, block_ids, positions, config)
    if _shape(out) != (1, g["H"], g["S"], config.v_dim) or out.dtype != BF16 or out.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(
            f"out must be [1, {g['H']}, {g['S']}, {config.v_dim}] bf16 ROW_MAJOR, got {out.layout} {out.dtype} {_shape(out)}"
        )
    if l1_bytes(g, config) > L1_BUDGET_BYTES:
        raise ValueError(
            f"the program's circular buffers take {l1_bytes(g, config)} bytes, over the {L1_BUDGET_BYTES} L1 budget"
        )
    mesh = q.device()
    work = fp.split_work(g["num_tiles"], mesh)
    cores = fp.core_rectangle(work, mesh)
    cbs = [fp.cb_descriptor(index, fmt, page, pages, cores) for index, fmt, page, pages in cb_table(g, config)]
    shared = {
        "TQ": config.tile_queries,
        "H": g["H"],
        "S": g["S"],
        "CB": config.chunk_blocks,
        "BT": config.block_tokens,
        "K_ROW_BYTES": g["W"] * 2,
        "RING_DEPTH": config.ring_depth,
        "DEBUG_STAGE": int(config.debug_stage),
    }
    reader_named = {
        **shared,
        "MAX_BLOCKS": g["max_blocks"],
        "U_MAX": g["U_max"],
        "IDS": g["IDS"],
        "Q_ROW_BYTES": g["Dq"] * 2,
        "SPLIT_NUM": config.split_num,
        "SPLIT_DEN": config.split_den,
        "SEED": int(config.seed),
        "CB_Q_RM": CB_Q_RM,
        "CB_K_RM": CB_K_RM,
        "CB_IDS": CB_IDS,
        "CB_POS": CB_POS,
        "CB_BITMAP": CB_BITMAP,
        "CB_SEEDMAP": CB_SEEDMAP,
        "CB_RANK": CB_RANK,
        "CB_UNION": CB_UNION,
        "CB_MEMBER": CB_MEMBER,
        "CB_CTRL": CB_CTRL,
        "CB_KREQ": CB_KREQ,
        "CB_KACK": CB_KACK,
        "CB_TILECTL": CB_TILECTL,
    }
    writer_named = {
        **shared,
        "SKT": g["Skt"],
        "SQT": g["Sqt"],
        "VDHT": g["vDHt"],
        "OUT_ROW_BYTES": config.v_dim * 2,
        "CB_OUT_RM": CB_OUT_RM,
        "CB_SCALE": CB_SCALE,
        "CB_COL_IDENTITY": CB_COL_IDENTITY,
        "CB_MASK_BAND": CB_MASK_BAND,
        "CB_BAND_TEMPLATE": CB_BAND_TEMPLATE,
        "CB_POS": CB_POS,
        "CB_UNION": CB_UNION,
        "CB_MEMBER": CB_MEMBER,
        "CB_KREQ": CB_KREQ,
        "CB_KACK": CB_KACK,
        "CB_TILECTL": CB_TILECTL,
    }
    compute_named = {
        "TQ": config.tile_queries,
        "H": g["H"],
        "SKT": g["Skt"],
        "DQT": g["DQt"],
        "DHT": g["DHt"],
        "VDHT": g["vDHt"],
        "K_OFF_T": config.k_offset // TILE,
        "V_OFF_T": config.v_offset // TILE,
        "SCALE": _bits(scale),
        "QSB": g["qsb"],
        "MATH_APPROX": int(config.approx),
        "DEBUG_STAGE": int(config.debug_stage),
        "CB_Q_RM": CB_Q_RM,
        "CB_Q_IN": CB_Q_IN,
        "CB_K_RM": CB_K_RM,
        "CB_K_IN": CB_K_IN,
        "CB_SCALE": CB_SCALE,
        "CB_COL_IDENTITY": CB_COL_IDENTITY,
        "CB_MASK_BAND": CB_MASK_BAND,
        "CB_QK_IM": CB_QK_IM,
        "CB_MAX_A": CB_MAX_A,
        "CB_MAX_B": CB_MAX_B,
        "CB_SUM_A": CB_SUM_A,
        "CB_SUM_B": CB_SUM_B,
        "CB_OUT_A": CB_OUT_A,
        "CB_OUT_B": CB_OUT_B,
        "CB_CORR": CB_CORR,
        "CB_OUT_IM": CB_OUT_IM,
        "CB_OUT_RM": CB_OUT_RM,
        "CB_RECIP": CB_RECIP,
        "CB_CTRL": CB_CTRL,
    }
    reader_cta = [
        *fp.accessor_args(q),
        *fp.accessor_args(kv),
        *fp.accessor_args(block_ids),
        *fp.accessor_args(positions),
    ]
    writer_cta = [*fp.accessor_args(out), *fp.accessor_args(kv)]
    addresses = (q.buffer_address(), kv.buffer_address(), block_ids.buffer_address(), positions.buffer_address())
    reader = fp.reader_kernel(
        READER, cores, reader_cta, [(w.core, [*addresses, w.start, w.count]) for w in work], named=reader_named
    )
    writer = fp.writer_kernel(
        WRITER,
        cores,
        writer_cta,
        [(w.core, [out.buffer_address(), kv.buffer_address(), w.start, w.count]) for w in work],
        named=writer_named,
    )
    compute = fp.compute_kernel(
        COMPUTE,
        cores,
        [],
        [(w.core, [w.count]) for w in work],
        named=compute_named,
        fidelity=config.fidelity,
        fp32_dest=config.fp32_dest,
        approx=config.approx,
    )
    return [q, kv, block_ids, positions, out], fp.program_descriptor([reader, writer, compute], cbs=cbs)


def sparse_sdpa_tiled(q, kv, block_ids, positions, *, scale: float | None = None, config: Config = DEFAULT, out=None):
    """The block-shared attention of the slab's query rows: ``[1, H, S, v_dim]`` bf16 ROW_MAJOR."""

    g = geometry(q, kv, block_ids, positions, config)
    scale = g["Dq"] ** -0.5 if scale is None else float(scale)
    if out is None:
        out = fp.allocate((1, g["H"], g["S"], config.v_dim), BF16, ttnn.ROW_MAJOR_LAYOUT, q.device())
    io, descriptor = build(q, kv, block_ids, positions, out, scale=scale, config=config)
    tokens = g["IDS"] * config.block_tokens  # the selected tokens of one row
    # Q, the block ids and the positions in, the attention out; the K/V rows at the union's floor (one row's selection
    # per tile: the measured union ratios over it, 1.03-3.45 at TQ 16, are in PREFILL.md); the FLOPs of every row's
    # own selected tokens (Q K^T and P V; the union's extra cells are the kernel's overhead, not the model's work)
    meta = fp.program_meta(
        NAME,
        "flash",
        g["S"],
        reads=(q, block_ids, positions),
        writes=(out,),
        partial=((kv, g["num_tiles"] * tokens * g["W"] * 2),),
        flops=2 * g["H"] * g["S"] * tokens * (g["Dq"] + config.v_dim),
        cores=len(fp.split_work(g["num_tiles"], q.device())),
    )
    return fp.run_program(io, descriptor, meta=meta)


# --------------------------------------------------------------------------- the host oracle and the emulation


def complete_blocks(positions: torch.Tensor, block_tokens: int = 4) -> torch.Tensor:
    """``(p + 1) // block_tokens`` per row: the blocks fully in the causal past of position ``p``."""

    return (positions.to(torch.int64) + 1) // block_tokens


def attended_mask(block_ids: torch.Tensor, positions: torch.Tensor, T: int, block_tokens: int = 4) -> torch.Tensor:
    """``[S, T]`` bool: the tokens query row j attends = the tokens of ``block_ids[j, :min(IDS, complete_j)]`` that
    lie below ``complete_j`` (the positional rule; ids at or past ``complete_j`` are masked slots) plus the tail
    ``block_tokens * complete_j .. p_j`` of its own incomplete block.  Today's expansion (``expand_today``) selects
    the same tokens; the identity test pins it."""

    ids = block_ids.reshape(block_ids.shape[-2], block_ids.shape[-1]).to(torch.int64)
    pos = positions.reshape(-1).to(torch.int64)
    S, IDS = ids.shape
    complete = complete_blocks(pos, block_tokens)
    n = complete.clamp(max=IDS)
    slot = torch.arange(IDS).unsqueeze(0)
    valid = (slot < n.unsqueeze(1)) & (ids < complete.unsqueeze(1)) & (ids * block_tokens < T)
    mask = torch.zeros(S, T, dtype=torch.bool)
    rows = torch.arange(S).unsqueeze(1).expand(S, IDS)[valid]
    blocks = ids[valid]
    for o in range(block_tokens):
        mask[rows, blocks * block_tokens + o] = True
    token = torch.arange(T).unsqueeze(0)
    tail = (token >= (complete * block_tokens).unsqueeze(1)) & (token <= pos.unsqueeze(1))
    return mask | tail


def reference_fp32(
    q, kv, block_ids, positions, *, scale: float | None = None, config: Config = DEFAULT
) -> torch.Tensor:
    """Exact-arithmetic (fp32) attention over ``attended_mask`` on host tensors: ``q`` ``[1, H, S, Dq]``, ``kv``
    ``[1, 1, T, W]`` (any float dtype; computed in fp32 on the values given, so pass the bf16-rounded inputs to
    measure the kernel rather than the input quantization), ``block_ids`` ``[1, 1, S, IDS]``, ``positions`` ``[S]``
    or ``[1, 1, 1, S]``.  Returns ``[1, H, S, v_dim]`` fp32."""

    qf = q.reshape(q.shape[1], q.shape[2], q.shape[3]).float()
    kvf = kv.reshape(kv.shape[-2], kv.shape[-1]).float()
    H, S, Dq = qf.shape
    T = kvf.shape[0]
    k = kvf[:, config.k_offset : config.k_offset + Dq]
    v = kvf[:, config.v_offset : config.v_offset + config.v_dim]
    scale = Dq**-0.5 if scale is None else float(scale)
    mask = attended_mask(block_ids, positions, T, config.block_tokens)
    scores = torch.einsum("hsd,td->hst", qf, k) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hst,td->hsd", probs, v).unsqueeze(0)


def expand_today(
    block_ids: torch.Tensor, positions: torch.Tensor, *, capacity: int = 2080, block_tokens: int = 4
) -> torch.Tensor:
    """The slab's expansion of the block ids to ``sparse_sdpa``'s token rows, ``[S, capacity]`` int64 (uint32 values;
    ``MASKED_INDEX`` = sentinel): the first ``4 * min(IDS, complete_j)`` slots hold the selected blocks' tokens in
    selection order, then the tail tokens ``4 * complete_j .. p_j``, then sentinels (``qsa.derive_qsa_chunk_inputs``'s
    ``row_keep_bits`` / ``row_fill`` on ``[ids << 2, repeat 4, + offsets | sentinel pad]``)."""

    ids = block_ids.reshape(block_ids.shape[-2], block_ids.shape[-1]).to(torch.int64)
    pos = positions.reshape(-1).to(torch.int64)
    S, IDS = ids.shape
    budget = IDS * block_tokens
    if capacity < budget:
        raise ValueError(f"capacity {capacity} below the expanded budget {budget}")
    complete = complete_blocks(pos, block_tokens)
    selected = complete.clamp(max=IDS)
    lo = selected * block_tokens
    tail = (pos + 1) % block_tokens
    hi = lo + tail
    expanded = (ids * block_tokens).repeat_interleave(block_tokens, dim=1) + torch.arange(block_tokens).repeat(IDS)
    template = torch.cat([expanded, torch.full((S, capacity - budget), MASKED_INDEX, dtype=torch.int64)], dim=1)
    slot = torch.arange(capacity).unsqueeze(0)
    kept = torch.where(slot < lo.unsqueeze(1), template, torch.zeros_like(template))
    tail_ids = slot + ((complete - selected) * block_tokens).unsqueeze(1)
    tail_bits = (slot >= lo.unsqueeze(1)) & (slot < hi.unsqueeze(1))
    fill = torch.where(tail_bits, tail_ids, torch.zeros_like(template))
    fill = fill | torch.where(
        slot >= hi.unsqueeze(1), torch.full_like(template, MASKED_INDEX), torch.zeros_like(template)
    )
    return kept | fill


@dataclass(frozen=True)
class TilePlan:
    """The reader's result for one tile: the seeds (in slot order), the union (block id per slot), the membership
    word per slot, and the per-row geometry."""

    t0: int
    seeds: list[int]
    union: list[int]
    member: list[int]
    complete: list[int]
    tail: list[int]

    @property
    def U(self) -> int:
        return len(self.union)

    def n_chunks(self, chunk_blocks: int) -> int:
        return -(-self.U // chunk_blocks)


def tile_plan(block_ids: torch.Tensor, positions: torch.Tensor, tile: int, config: Config = DEFAULT) -> TilePlan:
    """The host image of the reader's three passes for ``tile`` (rows ``tile * TQ ..``): the seed of every row (its
    first valid id, else its diagonal block) emitted first in ascending order, then every other marked block
    ascending (valid ids and the diagonal blocks with a tail), membership bit q on the valid ids of row q."""

    ids = block_ids.reshape(block_ids.shape[-2], block_ids.shape[-1]).to(torch.int64)
    pos = positions.reshape(-1).to(torch.int64)
    TQ, BT = config.tile_queries, config.block_tokens
    t0 = tile * TQ
    complete, tail, seeds, marked, valid_rows = [], [], set(), set(), []
    for qi in range(TQ):
        p = int(pos[t0 + qi])
        c = (p + 1) // BT
        t = (p + 1) % BT
        complete.append(c)
        tail.append(t)
        row = [int(b) for b in ids[t0 + qi, : min(c, ids.shape[1])].tolist() if int(b) < c]
        valid_rows.append(row)
        marked.update(row)
        if t > 0:
            marked.add(c)
        if config.seed:
            seed = row[0] if row else c
            seeds.add(seed)
            marked.add(seed)
    seed_list = sorted(seeds)
    union = seed_list + sorted(marked - seeds)
    slot_of = {b: s for s, b in enumerate(union)}
    member = [0] * len(union)
    for qi, row in enumerate(valid_rows):
        for b in row:
            member[slot_of[b]] |= 1 << qi
    return TilePlan(t0=t0, seeds=seed_list, union=union, member=member, complete=complete, tail=tail)


def band(plan: TilePlan, chunk: int, config: Config = DEFAULT) -> torch.Tensor:
    """The writer's mask band of ``chunk``: ``[32, CB * BT]`` float32 of 0 / ``MASK_FLOOR`` (row i = query ``i % TQ``)."""

    TQ, CB, BT = config.tile_queries, config.chunk_blocks, config.block_tokens
    out = torch.full((32, CB * BT), MASK_FLOOR, dtype=torch.float32)
    for s in range(CB):
        slot = chunk * CB + s
        if slot >= plan.U:
            continue
        b, m = plan.union[slot], plan.member[slot]
        for qi in range(TQ):
            if (m >> qi) & 1:
                visible = BT
            elif b == plan.complete[qi]:
                visible = plan.tail[qi]
            else:
                continue
            for r in range(qi, 32, TQ):
                out[r, s * BT : s * BT + visible] = 0.0
    return out


def tile_attended_from_bands(plan: TilePlan, config: Config = DEFAULT, T: int | None = None) -> torch.Tensor:
    """``[TQ, T]`` bool: the tokens the bands leave visible per query of the tile (the kernel's attended set)."""

    TQ, CB, BT = config.tile_queries, config.chunk_blocks, config.block_tokens
    T = T if T is not None else (max(plan.union) + 1) * BT
    mask = torch.zeros(TQ, T, dtype=torch.bool)
    for c in range(plan.n_chunks(CB)):
        b = band(plan, c, config)
        for s in range(CB):
            slot = c * CB + s
            if slot >= plan.U:
                break
            block = plan.union[slot]
            for qi in range(TQ):
                for j in range(BT):
                    if b[qi, s * BT + j] == 0.0 and block * BT + j < T:
                        mask[qi, block * BT + j] = True
    return mask


# --------------------------------------------------------------------------- today's chain on the same inputs


def sparse_sdpa_tiled_composed(
    q, kv, block_ids, positions, *, scale: float | None = None, config: Config = DEFAULT, out=None
):
    """Today's slab attention on the same inputs: the expansion of the block ids to ``sparse_sdpa``'s token rows
    (the ops of ``qsa._sparse_indices_slab`` on device: shift, repeat, offsets, keep / tail / sentinel from the
    positions), the zero V half, the pad to 32 heads, ``ttnn.transformer.sparse_sdpa`` at the slab's compute
    config, the slice back to the local heads.  Returns ``[1, H, S, v_dim]`` bf16 ROW_MAJOR."""

    g = geometry(q, kv, block_ids, positions, config)
    if config.k_offset != config.v_dim or config.v_offset != 0 or g["W"] != 2 * config.v_dim or g["Dq"] != config.v_dim:
        raise ValueError("the composed chain is today's [V | K] packed row with the query's zero V half")
    dram = ttnn.DRAM_MEMORY_CONFIG
    mesh = q.device()
    S, IDS, H, BT = g["S"], g["IDS"], g["H"], config.block_tokens
    budget = IDS * BT
    capacity = -(-(budget + 7) // TILE) * TILE  # sparse_sdpa's SPARSE_INDEX_CAPACITY rule at 4 tokens per block
    scale = g["Dq"] ** -0.5 if scale is None else float(scale)
    # The chain's constant operands (the model keeps them in its chunk constants): uploaded once per process and
    # shape, so the chain is trace-safe (a host upload inside a capture is refused).
    constants = _composed_constants(mesh, S, IDS, H, capacity, config)
    slots, ones, offsets, pad, zero_half = (constants[k] for k in ("slots", "ones", "offsets", "pad", "zero_half"))

    # The positional geometry per row (uint32 ops on device, as derive_qsa_chunk_inputs): kept slots, tail, sentinels.
    pos_col = ttnn.reshape(positions, (1, 1, S, 1))
    context = ttnn.add(pos_col, 1, memory_config=dram)
    complete = ttnn.bitwise_right_shift(context, 2, memory_config=dram)
    selected = ttnn.minimum(complete, IDS, memory_config=dram)
    lo = ttnn.bitwise_left_shift(selected, 2, memory_config=dram)
    tail_count = ttnn.bitwise_and(context, BT - 1, memory_config=dram)
    hi = ttnn.add(lo, tail_count, memory_config=dram)
    before_lo = ttnn.lt(slots, lo, dtype=U32, memory_config=dram)
    keep_bits = ttnn.multiply(before_lo, ones, memory_config=dram)
    skipped = ttnn.subtract(complete, selected, memory_config=dram)
    tail_shift = ttnn.bitwise_left_shift(skipped, 2, memory_config=dram)
    tail_ids = ttnn.add(slots, tail_shift, memory_config=dram)
    from_lo = ttnn.ge(slots, lo, dtype=U32, memory_config=dram)
    before_hi = ttnn.lt(slots, hi, dtype=U32, memory_config=dram)
    tail_bits = ttnn.multiply(from_lo, before_hi, memory_config=dram)
    tail_fill = ttnn.multiply(tail_ids, tail_bits, memory_config=dram)
    from_hi = ttnn.ge(slots, hi, dtype=U32, memory_config=dram)
    sentinel_bits = ttnn.multiply(from_hi, ones, memory_config=dram)
    fill = ttnn.bitwise_or(tail_fill, sentinel_bits, memory_config=dram)
    # The expansion of the ids.
    starts = ttnn.bitwise_left_shift(block_ids, 2, memory_config=dram)
    repeated = ttnn.repeat_interleave(starts, repeats=BT, dim=3, memory_config=dram)
    expanded = ttnn.add(repeated, offsets, memory_config=dram)
    template = ttnn.concat([expanded, pad], dim=3, memory_config=dram)
    kept = ttnn.bitwise_and(template, keep_bits, memory_config=dram)
    sparse_indices = ttnn.bitwise_or(kept, fill, memory_config=dram)
    # The padded query rows: [zeros(v_dim) | Q] per head, 32 heads.
    q_packed = ttnn.concat([zero_half, q], dim=3, memory_config=dram)
    q_padded = ttnn.pad(q_packed, [(0, 0), (0, 32 - H), (0, 0), (0, 0)], 0.0, memory_config=dram)
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=config.fidelity,
        math_approx_mode=config.approx,
        fp32_dest_acc_en=config.fp32_dest,
        packer_l1_acc=False,
    )
    sparse_output = ttnn.transformer.sparse_sdpa(
        q_padded,
        kv,
        sparse_indices,
        config.v_dim,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=scale,
        k_chunk_size=TILE,
        compute_kernel_config=compute_config,
    )
    local = ttnn.slice(sparse_output, (0, 0, 0, 0), (1, H, S, config.v_dim), memory_config=dram)
    for t in (
        pos_col,
        context,
        complete,
        selected,
        lo,
        tail_count,
        hi,
        before_lo,
        keep_bits,
        skipped,
        tail_shift,
        tail_ids,
        from_lo,
        before_hi,
        tail_bits,
        tail_fill,
        from_hi,
        sentinel_bits,
        fill,
        starts,
        repeated,
        expanded,
        template,
        kept,
        sparse_indices,
        q_packed,
        q_padded,
        sparse_output,
    ):
        ttnn.deallocate(t)
    if out is not None:
        # signature parity with the fused form (the registry's dispatch forwards the same kwargs): the chain's result
        # is the tensor returned; a caller's pre-allocated output is released
        ttnn.deallocate(out)
    return local


_COMPOSED_CONSTANTS: dict[tuple, dict[str, Any]] = {}


def _composed_constants(mesh, S: int, IDS: int, H: int, capacity: int, config: Config) -> dict[str, Any]:
    """The composed chain's constant operands per (device, shape), uploaded once and kept for the process."""

    mesh_key = mesh.id() if hasattr(mesh, "id") and callable(mesh.id) else id(mesh)
    key = (mesh_key, S, IDS, H, capacity, config.v_dim, config.block_tokens)
    if key not in _COMPOSED_CONSTANTS:
        dram = ttnn.DRAM_MEMORY_CONFIG
        BT = config.block_tokens
        budget = IDS * BT

        def upload(host, dtype):
            return ttnn.from_torch(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram)

        _COMPOSED_CONSTANTS[key] = {
            "slots": upload(
                torch.arange(capacity, dtype=torch.int32)
                .reshape(1, 1, 1, capacity)
                .expand(1, 1, S, capacity)
                .contiguous(),
                U32,
            ),
            "ones": upload(torch.full((1, 1, S, capacity), -1, dtype=torch.int32), U32),
            "offsets": upload(
                torch.arange(BT, dtype=torch.int32)
                .repeat(IDS)
                .reshape(1, 1, 1, budget)
                .expand(1, 1, S, budget)
                .contiguous(),
                U32,
            ),
            "pad": upload(torch.full((1, 1, S, capacity - budget), -1, dtype=torch.int32), U32),
            "zero_half": upload(torch.zeros(1, H, S, config.v_dim, dtype=torch.bfloat16), BF16),
        }
    return _COMPOSED_CONSTANTS[key]


# --------------------------------------------------------------------------- metrics for the gates


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1])


def ulp_histogram(new: torch.Tensor, old: torch.Tensor) -> dict[str, Any]:
    """|new - old| in bf16 ulps of the old value (``qsa_indexer_diet_microtest``'s bins)."""

    new, old = new.flatten().float(), old.flatten().float()
    delta = (new - old).abs()
    exponent = torch.floor(torch.log2(old.abs().clamp(min=torch.finfo(torch.float32).tiny)))
    ulps = delta / torch.exp2(exponent - 7)
    bins = {
        "0": ulps == 0,
        "1": (ulps > 0) & (ulps <= 1),
        "2-4": (ulps > 1) & (ulps <= 4),
        "5-16": (ulps > 4) & (ulps <= 16),
        ">16": ulps > 16,
    }
    return {
        "ulp_histogram": {k: int(v.sum()) for k, v in bins.items()},
        "max_ulp": float(ulps.max()),
        "max_abs": float(delta.max()),
    }


def row_rel_err(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per (head, row) ||out - ref|| / ||ref|| over the head dim: ``[H, S]``."""

    o, r = (
        out.reshape(out.shape[-3], out.shape[-2], out.shape[-1]).float(),
        ref.reshape(ref.shape[-3], ref.shape[-2], ref.shape[-1]).float(),
    )
    return (o - r).norm(dim=-1) / r.norm(dim=-1).clamp(min=1e-30)


def row_gain(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per (head, row) least-squares scale of out against ref (1.0 = no gain error): ``[H, S]``."""

    o, r = (
        out.reshape(out.shape[-3], out.shape[-2], out.shape[-1]).float(),
        ref.reshape(ref.shape[-3], ref.shape[-2], ref.shape[-1]).float(),
    )
    return (o * r).sum(-1) / (r * r).sum(-1).clamp(min=1e-30)


register(
    FusedKernel(
        name=NAME,
        replaces=(
            "the slab's QSA attention: the block-id expansion (shift, repeat, offsets, keep, tail | sentinel), the zero "
            "V half, the 6 -> 32 head pad, sparse_sdpa over 2048 gathered rows per query, the head slice (10 programs/layer)"
        ),
        tolerance=COMPONENT,
        fused=sparse_sdpa_tiled,
        composed=sparse_sdpa_tiled_composed,
        gate=None,
        # The line gate of 2026-09-26 (4x p150, 32k, `--prefill-slab 2048`, arm 9 = this kernel at HiFi2 against the
        # chain): acceptance pins 12/12 and the acceptance columns identical; the 31,716-token long part identical
        # through the first slab, from position 8128 (cross-slab attention) KL control -> arm mean 0.0028 / max 0.084,
        # top-1 vs the control 160/160, top-1 vs HF 0.9875 (= the control's); TTFT 15.92 -> 14.25 s.  On one die the
        # kernel is no farther from the fp32 oracle than the chain on captured selections (docs/NUMERICS.md).
        component_proof="line gate 2026-09-26: KL 0.0028 mean / 0.084 max vs the control, top-1 160/160, vs HF 0.9875 = control",
        admits=admits,
    )
)

__all__ = [
    "Config",
    "DEFAULT",
    "MASK_FLOOR",
    "MASKED_INDEX",
    "TilePlan",
    "admits",
    "admits_shapes",
    "attended_mask",
    "band",
    "build",
    "cb_table",
    "complete_blocks",
    "config_for_grid",
    "model_config",
    "FIDELITY_ENV",
    "HIFI2",
    "DEFAULT_FIDELITY",
    "expand_today",
    "geometry",
    "l1_bytes",
    "pcc",
    "reference_fp32",
    "row_gain",
    "row_rel_err",
    "sparse_sdpa_tiled",
    "sparse_sdpa_tiled_composed",
    "tile_attended_from_bands",
    "tile_plan",
    "ulp_histogram",
]
