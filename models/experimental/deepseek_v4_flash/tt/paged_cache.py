# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side paged KV book-keeping for multi-session decode (vLLM / tt-transformers style).

Every attention layer reads its KV through a *pool* of fixed-size blocks
``[num_blocks, 1, block_size, Dh]`` in DRAM (``Dh`` = ``head_dim``), plus a ``page_table``
``[B, logical_blocks]`` mapping a slot's logical KV index to a physical block (``B`` = users
decoded per step, row ``u`` holding slot ``u``'s mapping). Sessions therefore never need
contiguous storage and -- the point of the design -- the two device buffers a captured trace
addresses (the pool and the page-table tensor) are the same for every session: switching
sessions only rewrites the *contents* of the page-table rows.

Layers are grouped by ``layer_type``, because the logical KV axis differs per type:

* ``sliding_attention`` -- the axis is just the ``sliding_window`` ring, so a session needs
  ``sliding_window / block_size`` blocks however long it runs. The ops wrap absolute positions
  into that capacity via ``cache_position_modulo`` (see :attr:`PagedGroup.position_modulo` and
  ``tests/ttnn/unit_tests/operations/sdpa/test_bounded_sliding_kv_cache.py``); without it every
  position past the window collapses onto physical block 0 and silently corrupts another
  session's cache.
* ``compressed_sparse_attention`` / ``heavily_compressed_attention`` -- the axis is
  ``[sliding ring | one entry per closed compressor window]``, addressed by *already-wrapped*
  indices (``pos % window`` for the ring, ``window + w`` for window ``w``), so no modulo
  applies. The ring blocks are allocated when the session opens, the compressed blocks as
  windows close -- which is what lets several sessions share one pool sized for a total token
  budget rather than ``sessions x max_context``.

Physical block ``0`` (:data:`ZERO_BLOCK`) is all-zero and never handed out: the unmapped tail
of a page-table row points at it, so a kernel that reads past the valid region (chunked SDPA
rounds up to ``k_chunk_size``) sees zeros instead of another session's tokens.

This is all pure host state; the device side (pool allocation, page-table tensors, the
attention plumbing) lives in ``model.py`` / ``attention.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

import ttnn

#: Physical block reserved as all-zero filler for unmapped page-table entries.
ZERO_BLOCK = 0


class PagedCacheFull(RuntimeError):
    """The shared block pool ran out of blocks."""


@dataclass(frozen=True)
class PagedGroup:
    """Block geometry for one layer type's logical KV axis.

    ``max_seq`` is the per-session context capacity (the longest absolute position plus
    one any session may decode); it fixes the page-table row width ``[logical_blocks]``
    and hence the shapes baked into the captured traces.

    ``block_size`` counts *rows of the layer's KV axis*, not tokens: a compressor
    group's row is one pooled entry per ``compress_rate`` tokens, so a block of the
    same row count spans ``compress_rate`` times more context.
    """

    layer_type: str
    block_size: int
    sliding_window: int
    max_seq: int
    compress_rate: int | None  # None for sliding-only layers

    def __post_init__(self) -> None:
        """Reject geometry the paged ops cannot bake into a trace.

        Both paged ops derive their per-block stride in tiles, so ``block_size`` has to be
        a positive multiple of ``ttnn.TILE_SIZE``; a sliding-only group additionally needs
        ``cache_position_modulo == sliding_window`` to be a whole number of blocks, and a
        compressor group needs ``max_seq`` to be a whole number of ``compress_rate`` windows.
        Every number here ends up in a traced shape: the ``[num_blocks, 1, block_size, Dh]``
        pool and the ``[B, logical_blocks]`` page table.
        """
        if self.block_size <= 0 or self.block_size % ttnn.TILE_SIZE:
            raise ValueError(f"block_size {self.block_size} must be a positive multiple of {ttnn.TILE_SIZE}")
        if self.compress_rate is None:
            if self.sliding_window % self.block_size:
                raise ValueError(
                    f"sliding_window {self.sliding_window} must be a multiple of block_size {self.block_size}"
                )
        elif self.max_seq % self.compress_rate:
            raise ValueError(
                f"max_seq {self.max_seq} must be a multiple of compress_rate {self.compress_rate} "
                f"for layer type {self.layer_type}"
            )

    def axis_rows_for(self, max_seq: int) -> int:
        """Rows of the layer's logical KV axis used at context length ``max_seq`` (rows, not
        blocks) -- the valid prefix of the ``[1, 1, 1, kv_len]`` additive mask: the whole sliding
        ring, plus one row per compressed entry for a compressor group. The argument is what
        varies -- the properties below pass ``self.max_seq``."""
        entries = 0 if self.compress_rate is None else max_seq // self.compress_rate
        return self.sliding_window + entries

    def logical_blocks_for(self, max_seq: int) -> int:
        """Page-table row width ``[logical_blocks]``, in blocks, at context length ``max_seq``."""
        return math.ceil(self.axis_rows_for(max_seq) / self.block_size)

    def kv_len_for(self, max_seq: int) -> int:
        """Addressable KV-axis length in rows at context length ``max_seq``: the width the
        additive mask ``[1, 1, 1, kv_len]`` must have.

        Rounded up to whole blocks, because the paged ops address whole blocks: the tail
        past ``axis_rows_for(max_seq)`` is unmapped (zero block) and every mask marks it
        invalid, so it is never read as data.
        """
        return self.logical_blocks_for(max_seq) * self.block_size

    @property
    def axis_rows(self) -> int:
        """``axis_rows_for(max_seq)``: the rows of the logical KV axis this group uses, i.e. the
        valid prefix of the ``[1, 1, 1, kv_len]`` additive mask."""
        return self.axis_rows_for(self.max_seq)

    @property
    def ring_blocks(self) -> int:
        """Blocks covering the sliding ring, at logical block indices ``[0, ring_blocks)``.
        Allocated when a session opens.

        The ring is a *prefix* of the axis rather than a block-aligned region: when a
        block is wider than the window (a CSA block spans 1024 rows against a 128-row
        ring) the first block holds the ring and the earliest compressed entries both.
        """
        return math.ceil(self.sliding_window / self.block_size)

    @property
    def compressed_blocks(self) -> int:
        """Blocks past the ring's, i.e. the length of the row tail ``row[ring_blocks:]``: 0 for
        a sliding-only group, otherwise the compressed entries' blocks, handed out one at a
        time as windows close."""
        return self.logical_blocks - self.ring_blocks

    @property
    def logical_blocks(self) -> int:
        """``logical_blocks_for(max_seq)`` == the layer's logical KV axis in blocks, i.e. the
        page-table row width ``[logical_blocks]``."""
        return self.logical_blocks_for(self.max_seq)

    @property
    def kv_len(self) -> int:
        """``kv_len_for(max_seq)``: the addressable KV-axis rows of this group, i.e. the width
        of the additive mask ``[1, 1, 1, kv_len]``."""
        return self.kv_len_for(self.max_seq)

    @property
    def position_modulo(self) -> int | None:
        """``cache_position_modulo`` for the paged ops, or ``None`` when the indices are
        already wrapped by the caller (the compressor layers' combined axis)."""
        return self.sliding_window if self.compress_rate is None else None

    def compressed_blocks_for(self, pos: int) -> int:
        """Blocks past the ring's a session needs once it has decoded through ``pos``.

        A window closes every ``compress_rate`` tokens, so ``pos`` has produced
        ``(pos + 1) // compress_rate`` entries, which sit at axis rows
        ``[sliding_window, sliding_window + entries)``. The count is taken over the
        whole axis and the ring's blocks subtracted, because the block holding the
        ring's tail may already cover the first entries.
        """
        if self.compress_rate is None:
            return 0
        rows = self.sliding_window + (pos + 1) // self.compress_rate
        blocks = min(math.ceil(rows / self.block_size), self.logical_blocks)
        return blocks - self.ring_blocks


def build_groups(
    layer_types,
    compress_rates: dict,
    sliding_window: int,
    max_seq: int,
    block_size: int,
) -> dict[str, PagedGroup]:
    """One :class:`PagedGroup` per distinct layer type in ``layer_types``, in first-seen
    order.

    ``compress_rates`` maps a layer type to the rows it pools per entry, ``sliding_window``
    and ``max_seq`` are axis lengths in rows, and ``block_size`` is the one row count every
    group uses (the model's own layout), so every group's page-table row is ``[logical_blocks]``
    and all layers share one block geometry. ``layer_type -> PagedGroup``.
    """
    return {
        lt: PagedGroup(
            layer_type=lt,
            block_size=block_size,
            sliding_window=sliding_window,
            max_seq=max_seq,
            compress_rate=None if lt == "sliding_attention" else compress_rates[lt],
        )
        for lt in dict.fromkeys(layer_types)
    }


def round_context(max_seq: int, compress_rates, block_size: int) -> int:
    """Round a requested context length ``max_seq`` up to a length every group can tile.

    Each compressor's entry count (``max_seq / compress_rate``) has to be a whole
    number of blocks, so the context must be a multiple of ``compress_rate *
    block_size`` for every rate -- and of the tile, for the dense buffers and the additive
    mask ``[1, 1, 1, kv_len]``, all of which are ``max_seq``- or ``kv_len``-wide. Returns that
    length in rows (never below 1); it does not change ``PagedGroup.max_seq``.
    """
    rates = [int(cr) for cr in compress_rates]
    step = math.lcm(ttnn.TILE_SIZE, *[cr * block_size for cr in rates]) if rates else ttnn.TILE_SIZE
    return math.ceil(max(max_seq, 1) / step) * step


def plan_pool_blocks(groups: dict[str, PagedGroup], max_sessions: int, total_tokens: int) -> dict[str, int]:
    """Size each group's pool in blocks: ``group -> num_blocks``, where ``num_blocks`` is the
    leading dim of that layer type's ``[num_blocks, 1, block_size, Dh]`` pool tensor.

    The pool holds the zero block, every session's ring, plus compressed blocks for a
    *shared* budget of ``total_tokens`` across all sessions. Sliding groups are bounded by
    construction (one ring per session, whatever the context length), so only the compressor
    groups scale with the token budget, and both are capped at ``1 + max_sessions *
    logical_blocks`` -- a session's page-table row cannot address more than that.
    """
    plan = {}
    for name, g in groups.items():
        blocks = 1 + max_sessions * g.ring_blocks
        if g.compress_rate is not None:
            # +max_sessions: each session's final, partially filled block.
            blocks += math.ceil(total_tokens / g.compress_rate / g.block_size) + max_sessions
            blocks = min(blocks, 1 + max_sessions * g.logical_blocks)
        plan[name] = blocks
    return plan


@dataclass(frozen=True)
class PagedLayerView:
    """The device handles one layer reads and writes its KV through.

    Both are *persistent* buffers: a captured trace bakes in their addresses, and a
    session switch only rewrites the contents of ``page_table`` (see
    :meth:`DeepSeekV4Model.activate_sessions`), so one capture serves every session.
    """

    pool: ttnn.Tensor  # [num_blocks, 1, block_size, Dh] bf16 TILE DRAM
    page_table: ttnn.Tensor  # [B, logical_blocks] INT32 ROW_MAJOR, row u = slot u's mapping
    position_modulo: Optional[int]  # set only for the bounded sliding ring


class _BlockPool:
    """Free-list allocator over physical block ids ``[1, num_blocks)``."""

    def __init__(self, num_blocks: int):
        """``num_blocks`` counts ``ZERO_BLOCK`` too; a pool needs at least one allocatable
        block, so a smaller pool is a configuration error rather than an empty free list."""
        if num_blocks < 2:
            raise ValueError(f"pool needs at least 2 blocks (one is the zero block), got {num_blocks}")
        self.num_blocks = num_blocks
        self._free = list(range(1, num_blocks))

    @property
    def free_blocks(self) -> int:
        """Blocks still allocatable (``num_blocks - 1`` at most, the zero block never being
        one of them)."""
        return len(self._free)

    def alloc(self, n: int) -> list[int]:
        """Take ``n`` physical block ids as a list ``[n]``, all or nothing:
        :class:`PagedCacheFull` when fewer than ``n`` are free."""
        if n > len(self._free):
            raise PagedCacheFull(f"need {n} blocks, {len(self._free)} free of {self.num_blocks - 1}")
        return [self._free.pop() for _ in range(n)]

    def free(self, ids) -> None:
        """Return the physical block ids in ``ids`` to the free list. ``ZERO_BLOCK`` is never
        passed -- callers filter it out -- so block 0 never becomes allocatable."""
        self._free.extend(ids)


@dataclass
class _Session:
    """One open session: its id plus the host-side page rows every device table is written
    from."""

    sid: int
    #: group -> logical block index -> physical block (``ZERO_BLOCK`` where unmapped).
    rows: dict[str, list[int]] = field(default_factory=dict)


class PagedKVManager:
    """Shared block pools plus per-session page-table rows.

    One pool (and one free list) per layer-type group; every layer in a group shares
    the same logical->physical mapping, so a group needs a single page-table row per
    session however many layers it has (each layer still owns its own
    ``[num_blocks, 1, block_size, Dh]`` pool tensor on device -- the mapping is what is
    shared, not the data).
    """

    def __init__(self, groups: dict[str, PagedGroup], pool_blocks: dict[str, int]):
        """``pool_blocks`` gives each group's pool size in blocks (see
        :func:`plan_pool_blocks`), one :class:`_BlockPool` per group."""
        self.groups = groups
        self.pools = {name: _BlockPool(pool_blocks[name]) for name in groups}
        self._sessions: dict[int, _Session] = {}
        self._next_sid = 0

    # -- sessions ------------------------------------------------------------- #
    def open_session(self) -> int:
        """Allocate a session's ring blocks (its compressed blocks are handed out by
        :meth:`ensure_capacity` as windows close) and return its id, an ``int`` unique for
        the life of the manager.

        Every group gets a fresh row of ``[logical_blocks]`` entries, the ring prefix
        filled in and the rest left at ``ZERO_BLOCK``. A pool that cannot take another
        ring raises :class:`PagedCacheFull` with any partial open rolled back.
        """
        sid = self._next_sid
        session = _Session(sid)
        try:
            for name, g in self.groups.items():
                row = [ZERO_BLOCK] * g.logical_blocks
                for i, block in enumerate(self.pools[name].alloc(g.ring_blocks)):
                    row[i] = block
                session.rows[name] = row
        except PagedCacheFull:
            for name, row in session.rows.items():  # roll back a partial open
                self.pools[name].free([b for b in row if b != ZERO_BLOCK])
            raise
        self._sessions[sid] = session
        self._next_sid += 1
        return sid

    def close_session(self, sid: int) -> None:
        """Drop session ``sid`` and return every block it holds -- ring and compressed alike
        -- to its group's pool. ``sid`` has to be open (the lookup is a plain ``dict.pop``)."""
        session = self._sessions.pop(sid)
        for name, row in session.rows.items():
            self.pools[name].free([b for b in row if b != ZERO_BLOCK])

    def reset_session(self, sid: int) -> None:
        """Rewind a session to position 0: release its compressed blocks (its ring
        blocks are kept, and re-zeroed by the caller) so another session can use them.

        The row's tail ``row[ring_blocks:]`` -- ``[compressed_blocks]`` entries -- goes back
        to ``ZERO_BLOCK``, so the next :meth:`ensure_capacity` re-allocates from scratch.
        """
        session = self._sessions[sid]
        for name, g in self.groups.items():
            row = session.rows[name]
            tail = row[g.ring_blocks :]
            self.pools[name].free([b for b in tail if b != ZERO_BLOCK])
            row[g.ring_blocks :] = [ZERO_BLOCK] * g.compressed_blocks

    def has_session(self, sid: int) -> bool:
        """Whether ``sid`` is currently open."""
        return sid in self._sessions

    # -- capacity / page tables ----------------------------------------------- #
    def ensure_capacity(self, sid: int, pos: int) -> list[str]:
        """Make sure ``sid`` has blocks for every logical row a step at ``pos`` touches.

        Returns the groups whose page-table row ``[logical_blocks]`` changed (so the caller
        can refresh just those tensors on device). Raises :class:`PagedCacheFull` if ``pos``
        is past ``max_seq`` or a pool is exhausted; that aborts the step, but unlike
        :meth:`open_session` the groups already grown earlier in the loop keep their new
        blocks (and the ``changed`` list is dropped, so the caller must not publish a page
        table from a failed call).
        """
        session = self._sessions[sid]
        changed: list[str] = []
        for name, g in self.groups.items():
            if pos >= g.max_seq:
                raise PagedCacheFull(f"session {sid} position {pos} exceeds capacity {g.max_seq}")
            row = session.rows[name]
            need = g.compressed_blocks_for(pos)
            have = sum(1 for b in row[g.ring_blocks :] if b != ZERO_BLOCK)
            if need <= have:
                continue
            blocks = self.pools[name].alloc(need - have)
            for i, block in enumerate(blocks, start=have):
                row[g.ring_blocks + i] = block
            changed.append(name)
        return changed

    def page_row(self, sid: int, group: str) -> torch.Tensor:
        """``[1, logical_blocks]`` INT32 page table for one session and group.

        The caller concatenates one row per resident session into the device table's
        ``[B, logical_blocks]`` contents.
        """
        return torch.tensor(self._sessions[sid].rows[group], dtype=torch.int32).reshape(1, -1)

    # -- reporting ------------------------------------------------------------ #
    def usage(self) -> dict[str, tuple[int, int]]:
        """``group -> (blocks in use, blocks in pool)``, for status output; the pool size is
        the ``num_blocks`` of each layer's ``[num_blocks, 1, block_size, Dh]`` tensor."""
        return {name: (pool.num_blocks - pool.free_blocks, pool.num_blocks) for name, pool in self.pools.items()}

    def tokens_left(self) -> int:
        """Tokens the *tightest* compressor group can still admit across all sessions:
        ``free_blocks * compress_rate * block_size``, i.e. the context those blocks can hold.

        Sliding groups never run out (their allocation is bounded by the window), so
        the binding constraint is always a compressor group; with no compressor group
        the pool is effectively unbounded.
        """
        limits = [
            self.pools[name].free_blocks * g.compress_rate * g.block_size
            for name, g in self.groups.items()
            if g.compress_rate is not None
        ]
        return min(limits) if limits else 2**31
