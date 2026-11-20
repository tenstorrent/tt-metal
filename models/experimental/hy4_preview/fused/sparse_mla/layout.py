# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Geometry and host-side goldens of the sparse-MLA piece (piece 3 of the decode megakernel): pure torch, no device.

One chip, 8 users. Each user's query (64 heads, absorbed into the latent's space) attends over the rows of this
chip's stripe that the selection gave it; the chip returns a partial result per head, and the partials of the 32
chips and the sink are merged afterwards by log-sum-exp (another piece).

Partial contract (what the cross-chip merge gets from a chip, per user and head h), with
``l[h, t] = q_lat[h] . row[t]`` the UNSCALED logit over this chip's valid rows t and ``a = scaling = 1/16``::

    m[h] = max_t l[h, t]
    s[h] = sum_t exp(a * (l[h, t] - m[h]))
    O[h] = sum_t exp(a * (l[h, t] - m[h])) * row[t][:512]            [512], unnormalised, latent space

A user with no valid row on the chip returns ``m = EMPTY_MAX, s = 0, O = 0``. The sink is NOT in a chip's partial:
it is one more merge peer ``(m = sink[h] / a, s = 1, O = 0)``, entering the total once. The merge of peers c is::

    M[h]   = max_c m_c[h]
    ctx[h] = sum_c exp(a * (m_c - M)) O_c[h] / sum_c exp(a * (m_c - M)) s_c[h]

and ``ctx`` is reference/attention_decode.py ``absorbed_stages(...)["ctx"]``; ``kv_b`` (W_v), the gate and o_proj
follow (other pieces).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

TILE = 32
# max of a user that has no valid row on the chip: finite (so that two empty peers merge to 0 * exp(0), not NaN)
# and below any real logit. The value is the bf16 0xF149, about -9.95e29.
EMPTY_MAX_BF16 = 0xF149
EMPTY_MAX = struct.unpack("<f", struct.pack("<I", EMPTY_MAX_BF16 << 16))[0]


@dataclass(frozen=True)
class SparseMlaConfig:
    """Sizes of the piece. Defaults are tencent/Hy4-preview on the prefill stack's packed KV storage."""

    rows_per_slot: int  # rows of one (user, layer) slot on a chip: max context / 32
    users: int = 8
    heads: int = 64
    latent: int = 512
    rope: int = 64
    topk: int = 2048
    num_layers: int = 78  # slot = user * num_layers + layer
    scale: float = 1.0 / 16.0  # (192 + 64) ** -0.5
    chunk_rows: int = 128  # rows per attention chunk
    groups: int = 8  # S-blocks: `converters` row readers / converters and heads / heads_per_core attention cores
    # Converter v of an S-block takes slabs v, v + converters, ... of each of its chunks. 2 halves the conversion
    # time of a chunk, but converter 1 sits one row below the first row of its attention cores, so its NOC0
    # multicast (which only goes east and south) wraps around the chip through the other S-blocks' rows: slower
    # than 1 as soon as several S-blocks send (measured). It needs a sender north-west or, on NOC1, south-east.
    converters: int = 1
    heads_per_core: int = 8
    shard_rows: int = 32  # rows per DRAM shard of the KV storage
    scale_block: int = 128  # latent values per FP8 scale

    def __post_init__(self):
        assert self.chunk_rows % TILE == 0 and self.chunk_rows >= 2 * TILE
        assert self.topk % self.chunk_rows == 0
        assert self.groups == 8, "the merge tree is written for 8 S-blocks"
        assert self.heads % self.heads_per_core == 0
        assert self.latent % self.scale_block == 0 and self.latent % TILE == 0 and self.rope % TILE == 0

    @property
    def width(self) -> int:
        return self.latent + self.rope

    @property
    def dht(self) -> int:
        return self.width // TILE

    @property
    def vdht(self) -> int:
        return self.latent // TILE

    @property
    def scale_blocks(self) -> int:
        return self.latent // self.scale_block

    @property
    def row_bytes(self) -> int:
        """Packed row: FP8 latent | FP32 scales | BF16 rope."""
        return self.latent + 4 * self.scale_blocks + 2 * self.rope

    @property
    def row_stride(self) -> int:
        """Bytes a packed row takes in DRAM (page padded to the 64-byte alignment)."""
        return -(-self.row_bytes // 64) * 64

    @property
    def slabs(self) -> int:
        return self.chunk_rows // TILE

    @property
    def head_groups(self) -> int:
        return self.heads // self.heads_per_core

    def slot(self, user: int, layer: int) -> int:
        return user * self.num_layers + layer

    def chunks(self, n: int) -> int:
        return -(-n // self.chunk_rows)

    def block_of(self, user: int, chunk: int) -> int:
        """The S-block that serves chunk ``chunk`` of ``user``."""
        return (user + chunk) % self.groups


# ---------------------------------------------------------------------------------------------------------------
# Where a packed row is in DRAM (mirrors kernels/sm_common.hpp kv_row_*)
# ---------------------------------------------------------------------------------------------------------------
def kv_row_location(cfg: SparseMlaConfig, slot: int, row: int, banks: int = 8) -> tuple[int, int]:
    """(DRAM bank, byte offset inside the tensor's allocation in that bank) of row ``row`` of ``slot``."""
    page = slot * cfg.rows_per_slot + row
    shard = page // cfg.shard_rows
    return shard % banks, ((shard // banks) * cfg.shard_rows + page % cfg.shard_rows) * cfg.row_stride


# ---------------------------------------------------------------------------------------------------------------
# Core layout on a Blackhole worker grid (logical coordinates)
# ---------------------------------------------------------------------------------------------------------------
X0_LEFT, X0_RIGHT = 0, 7  # first logical column of the two halves of a Blackhole worker grid (7 + 5 columns)


@dataclass(frozen=True)
class SBlock:
    readers: tuple[tuple[int, int], ...]  # the row reader / converter cores, converter v at readers[v]
    attention: tuple[tuple[int, int], ...]  # head group g -> core
    rect: tuple[tuple[int, int], tuple[int, int]]  # the attention cores' rectangle (multicast target)


def sblocks(cfg: SparseMlaConfig) -> list[SBlock]:
    """8 S-blocks, each 2 rows of one half of the worker grid: the reader(s) in the half's first column (the DRAM
    banks are at the left edge of each half: bank -> nearest worker is (0, y) for banks 0..3, (7, y) for 4..7),
    reader v in the block's row v, and the 4x2 attention cores next to them. kernels/sm_common.hpp
    role_from_position is the inverse of this function. The programs span rows 0..7 of columns 0..4 and 7..11 as
    four rectangles (two reader columns, two attention blocks); a reader-column core without a reader returns
    at once."""
    assert cfg.head_groups == 8 and cfg.groups == 8 and cfg.converters in (1, 2)
    out = []
    for b in range(cfg.groups):
        x0 = X0_LEFT if b < 4 else X0_RIGHT
        y0 = 2 * (b % 4)
        att = tuple((x0 + 1 + g % 4, y0 + g // 4) for g in range(8))
        readers = tuple((x0, y0 + v) for v in range(cfg.converters))
        out.append(SBlock(readers, att, ((x0 + 1, y0), (x0 + 4, y0 + 1))))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Tiny tiles: [8 rows, 32 columns] BF16 = two [8, 16] faces, row-major inside a face
# ---------------------------------------------------------------------------------------------------------------
def pack_tiny(x: torch.Tensor) -> torch.Tensor:
    """[R, W] (R = tile rows, W a multiple of 32) -> [W / 32 tiles * R * 32] in tile memory order."""
    r, w = x.shape
    assert w % TILE == 0
    return x.reshape(r, w // TILE, 2, 16).permute(1, 2, 0, 3).reshape(-1)


def unpack_tiny(flat: torch.Tensor, rows: int) -> torch.Tensor:
    """Inverse of pack_tiny: [tiles * rows * 32] -> [rows, tiles * 32]."""
    tiles = flat.numel() // (rows * TILE)
    return flat.reshape(tiles, 2, rows, 16).permute(2, 0, 1, 3).reshape(rows, tiles * TILE)


# ---------------------------------------------------------------------------------------------------------------
# Selection table
# ---------------------------------------------------------------------------------------------------------------
def selection_table(cfg: SparseMlaConfig, lists: list[torch.Tensor]) -> torch.Tensor:
    """Per-user id lists (chip-local rows) -> int32 [1 + users, topk]: row 0 the counts, row 1 + u the ids."""
    assert len(lists) == cfg.users
    table = torch.zeros(1 + cfg.users, cfg.topk, dtype=torch.int32)
    for u, ids in enumerate(lists):
        n = int(ids.numel())
        assert n <= cfg.topk and (n == 0 or (int(ids.min()) >= 0 and int(ids.max()) < cfg.rows_per_slot))
        table[0, u] = n
        table[1 + u, :n] = ids.to(torch.int32)
    return table


def selection_table_frames(
    cfg: SparseMlaConfig, lists: list[torch.Tensor], tables, chip=None, positions=None
) -> torch.Tensor:
    """selection_table for a store read through fused/kv_store's FRAME TABLES (SparseMla(frames=...), a paged pool):
    an id is a STRIPE ROW of the chip in the row's conversation (frame = id >> 8, row of the frame = id & 255), not
    a row of a slot, so it is not bounded by ``rows_per_slot``. ``tables``: per token row its frame table (128 frame
    bases, kv_store UNMAPPED where the conversation has no frame). Validated, since a reader sent to an unmapped
    frame reads through a garbage address: at most top-k ids a row, every id distinct, non-negative and in a frame
    its row's table maps. With ``chip`` and ``positions`` (the token rows' own positions; None for an idle row, which
    must have no ids): every id a position at or below the row's own, and the frame of the row's own position
    mapped (the append writes there).
    Returns int32 [1 + users, topk]: row 0 the counts, row 1 + u the ids."""
    from models.experimental.hy4_preview.fused.kv_store import frames as kv_frames

    assert len(lists) == cfg.users and len(tables) == cfg.users, (len(lists), len(tables), cfg.users)
    assert (chip is None) == (positions is None) and (positions is None or len(positions) == cfg.users)
    table = torch.zeros(1 + cfg.users, cfg.topk, dtype=torch.int32)
    for u, ids in enumerate(lists):
        row_table = list(tables[u])
        assert len(row_table) == kv_frames.TABLE_FRAMES, (u, len(row_table))
        n = int(ids.numel())
        assert n <= cfg.topk, f"row {u}: {n} ids, top-k {cfg.topk}"
        own = None if positions is None else positions[u]
        if positions is not None and own is None:
            assert n == 0, f"row {u}: an idle row with {n} ids"
        if own is not None:
            frame = kv_frames.stripe_row(int(own)) >> 8
            assert row_table[frame] != kv_frames.UNMAPPED, f"row {u}: the frame of its own position {own} is not mapped"
        if n:
            ids = ids.to(torch.int64)
            assert int(ids.min()) >= 0 and len(set(ids.tolist())) == n, f"row {u}: negative or repeated ids"
            frames = sorted(set((ids >> 8).tolist()))
            unmapped = [f for f in frames if f >= kv_frames.TABLE_FRAMES or row_table[f] == kv_frames.UNMAPPED]
            assert not unmapped, f"row {u}: ids in frames {unmapped} that its table does not map"
            if own is not None:
                last = max(kv_frames.position_of(int(chip), int(r)) for r in ids.tolist())
                assert last <= int(own), f"row {u} chip {chip}: a position {last} beyond its own {own}"
            table[1 + u, :n] = ids.to(torch.int32)
        table[0, u] = n
    return table


# ---------------------------------------------------------------------------------------------------------------
# Goldens
# ---------------------------------------------------------------------------------------------------------------
def partial(q_lat: torch.Tensor, rows: torch.Tensor, cfg: SparseMlaConfig):
    """The partial result of one chip for one user. q_lat [H, width], rows [n, width] (n may be 0).
    Returns (O [H, latent], m [H], s [H]) in float64."""
    q, k = q_lat.double(), rows.double()
    if k.shape[0] == 0:
        h = q.shape[0]
        return (
            torch.zeros(h, cfg.latent, dtype=torch.float64),
            torch.full((h,), EMPTY_MAX, dtype=torch.float64),
            torch.zeros(h, dtype=torch.float64),
        )
    logits = q @ k.t()
    m = logits.max(dim=-1).values
    p = torch.exp(cfg.scale * (logits - m[:, None]))
    return p @ k[:, : cfg.latent], m, p.sum(-1)


def merge(partials, sink: torch.Tensor | None, cfg: SparseMlaConfig) -> torch.Tensor:
    """Log-sum-exp merge of partials [(O, m, s), ...] and, when given, the sink [H] (a logit): ctx [H, latent]."""
    o = torch.stack([p[0].double() for p in partials])
    m = torch.stack([p[1].double() for p in partials])
    s = torch.stack([p[2].double() for p in partials])
    if sink is not None:
        o = torch.cat([o, torch.zeros_like(o[:1])])
        m = torch.cat([m, (sink.double() / cfg.scale)[None]])
        s = torch.cat([s, torch.ones_like(s[:1])])
    w = torch.exp(cfg.scale * (m - m.max(dim=0).values))  # [peers, H]
    return (w[:, :, None] * o).sum(0) / (w * s).sum(0)[:, None]


def reference_ctx(q_lat: torch.Tensor, rows: torch.Tensor, sink: torch.Tensor, cfg: SparseMlaConfig) -> torch.Tensor:
    """reference/attention_decode.py absorbed_stages' ctx for rows that are all valid: [H, latent] float64."""
    logits = (q_lat.double() @ rows.double().t()) * cfg.scale
    combined = torch.cat([logits, sink.double().reshape(-1, 1)], dim=-1)
    probs = torch.softmax(combined - combined.max(dim=-1, keepdim=True).values, dim=-1)[:, :-1]
    return probs @ rows.double()[:, : cfg.latent]


# ---------------------------------------------------------------------------------------------------------------
# The merge tree of one user over the S-blocks that hold its chunks (mirrors kernels/sm_common.hpp)
# ---------------------------------------------------------------------------------------------------------------
MERGE_STEPS = 3


def active_blocks(cfg: SparseMlaConfig, n: int) -> int:
    return min(cfg.chunks(n), cfg.groups)


def receives(r: int, n_act: int, s: int) -> bool:
    """Does relative block r (= (block - user) % groups) receive a partial at step s?"""
    return r < n_act and r % (2 << s) == 0 and r + (1 << s) < n_act


def send_step(r: int) -> int:
    """The step at which relative block r > 0 sends its partial to r - 2**step."""
    return (r & -r).bit_length() - 1
