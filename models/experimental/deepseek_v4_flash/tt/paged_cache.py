# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side paged KV book-keeping for multi-session decode (vLLM / tt-transformers style).

Every attention layer reads its KV through a *pool* of fixed-size blocks,
``[num_blocks, 1, block_size, head_dim]`` in DRAM, plus a ``page_table`` row that
maps the layer's logical KV index to a physical block. Sessions therefore never
need contiguous storage and — crucially for the traced decode path — the device
buffers a captured trace addresses (the pool and the page-table tensor) stay the
same for every session: switching sessions only rewrites the *contents* of the
page-table row.

Layers are grouped by ``layer_type``, because the logical KV axis differs per type:

* ``sliding_attention`` — the axis is just the ``sliding_window`` ring, so a session
  only ever needs ``sliding_window / block_size`` blocks however long it runs. The
  ops wrap absolute positions into that capacity via ``cache_position_modulo``
  (see :attr:`PagedGroup.position_modulo` and
  ``tests/ttnn/unit_tests/operations/sdpa/test_bounded_sliding_kv_cache.py``);
  without it every position past the window collapses onto physical block 0 and
  silently corrupts another session's cache.
* ``compressed_sparse_attention`` / ``heavily_compressed_attention`` — the axis is
  ``[sliding ring | one entry per closed compressor window]`` (see
  ``_StaticLayerCache``), addressed by *already-wrapped* indices (``pos % window``
  for the ring, ``window + w`` for window ``w``), so no modulo applies. The ring
  blocks are allocated when the session opens; the compressed blocks are handed
  out as windows close, which is what lets several sessions share one pool sized
  for a total token budget rather than ``sessions x max_context``.

Physical block ``0`` is reserved as an all-zero block and is never handed out: the
unmapped tail of a page-table row points at it, so a kernel that reads past the
valid region (chunked SDPA rounds up to ``k_chunk_size``) sees zeros instead of
another session's tokens.

All of this is pure host state; the device side (pool allocation, page-table
tensors, the attention plumbing) lives in ``model.py`` / ``attention.py``.
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

    ``max_seq`` is the per-session context capacity (the longest absolute position
    plus one any session may decode), which fixes the page-table row width and
    hence the shapes baked into the captured traces.
    """

    layer_type: str
    block_size: int
    sliding_window: int
    max_seq: int
    compress_rate: int | None  # None for sliding-only layers

    def __post_init__(self) -> None:
        if self.sliding_window % self.block_size:
            raise ValueError(f"sliding_window {self.sliding_window} must be a multiple of block_size {self.block_size}")
        if self.compress_rate is not None:
            entries = self.max_seq // self.compress_rate
            if self.max_seq % self.compress_rate or entries % self.block_size:
                raise ValueError(
                    f"max_seq {self.max_seq} must be a multiple of compress_rate {self.compress_rate} "
                    f"* block_size {self.block_size} for layer type {self.layer_type}"
                )

    @property
    def ring_blocks(self) -> int:
        """Blocks holding the sliding ring, at logical block indices ``[0, ring_blocks)``."""
        return self.sliding_window // self.block_size

    @property
    def compressed_blocks(self) -> int:
        """Blocks holding the compressed entries (0 for sliding-only layers)."""
        if self.compress_rate is None:
            return 0
        return self.max_seq // self.compress_rate // self.block_size

    @property
    def logical_blocks(self) -> int:
        """Page-table row width == the layer's logical KV axis in blocks."""
        return self.ring_blocks + self.compressed_blocks

    @property
    def kv_len(self) -> int:
        """Logical KV axis length in tokens (the width the additive mask must have)."""
        return self.logical_blocks * self.block_size

    @property
    def position_modulo(self) -> int | None:
        """``cache_position_modulo`` for the paged ops, or ``None`` when indices are
        already wrapped by the caller (the compressor layers' combined axis)."""
        return self.sliding_window if self.compress_rate is None else None

    def compressed_blocks_for(self, pos: int) -> int:
        """Compressed blocks a session needs once it has decoded through ``pos``.

        A window closes every ``compress_rate`` tokens, so ``pos`` has produced
        ``(pos + 1) // compress_rate`` entries; they occupy that many rows of the
        compressed region, rounded up to whole blocks.
        """
        if self.compress_rate is None:
            return 0
        entries = (pos + 1) // self.compress_rate
        return min(math.ceil(entries / self.block_size), self.compressed_blocks)


def build_groups(
    layer_types,
    compress_rates: dict,
    sliding_window: int,
    max_seq: int,
    block_size: int,
) -> dict[str, PagedGroup]:
    """One :class:`PagedGroup` per distinct layer type present in ``layer_types``."""
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
    """Round a requested context up to a length every group's geometry can tile.

    Each compressor's entry count (``max_seq / compress_rate``) has to be a whole
    number of blocks, so the context must be a multiple of ``compress_rate *
    block_size`` for every rate -- and of the tile, for the dense buffers and masks.
    """
    rates = [int(cr) for cr in compress_rates]
    step = math.lcm(ttnn.TILE_SIZE, *[cr * block_size for cr in rates]) if rates else ttnn.TILE_SIZE
    return math.ceil(max(max_seq, 1) / step) * step


def plan_pool_blocks(groups: dict[str, PagedGroup], max_sessions: int, total_tokens: int) -> dict[str, int]:
    """Size each group's pool: every session's ring, plus compressed blocks for a
    *shared* budget of ``total_tokens`` across all sessions, plus the zero block.

    Sliding groups are bounded by construction (one ring per session, whatever the
    context length), so only the compressor groups scale with the token budget.
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
    :meth:`DeepSeekV4Model.activate_session`), so one capture serves every session.
    """

    pool: ttnn.Tensor  # [num_blocks, 1, block_size, head_dim]
    page_table: ttnn.Tensor  # [1, logical_blocks] INT32, holding the active session's row
    position_modulo: Optional[int]  # set only for the bounded sliding ring


class _BlockPool:
    """Free-list allocator over physical block ids ``[1, num_blocks)``."""

    def __init__(self, num_blocks: int):
        if num_blocks < 2:
            raise ValueError(f"pool needs at least 2 blocks (one is the zero block), got {num_blocks}")
        self.num_blocks = num_blocks
        self._free = list(range(1, num_blocks))

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> list[int]:
        if n > len(self._free):
            raise PagedCacheFull(f"need {n} blocks, {len(self._free)} free of {self.num_blocks - 1}")
        return [self._free.pop() for _ in range(n)]

    def free(self, ids) -> None:
        self._free.extend(ids)


@dataclass
class _Session:
    sid: int
    #: group -> logical block index -> physical block (``ZERO_BLOCK`` where unmapped).
    rows: dict[str, list[int]] = field(default_factory=dict)


class PagedKVManager:
    """Shared block pools plus per-session page-table rows.

    One pool (and one free list) per layer-type group; every layer in a group shares
    the same logical->physical mapping, so a group needs a single page-table row per
    session however many layers it has (each layer still owns its own pool tensor
    on device -- the mapping is what is shared, not the data).
    """

    def __init__(self, groups: dict[str, PagedGroup], pool_blocks: dict[str, int]):
        self.groups = groups
        self.pools = {name: _BlockPool(pool_blocks[name]) for name in groups}
        self._sessions: dict[int, _Session] = {}
        self._next_sid = 0

    # -- sessions ------------------------------------------------------------- #
    def open_session(self) -> int:
        """Allocate a session's ring blocks (its compressed blocks are handed out by
        :meth:`ensure_capacity` as windows close) and return its id."""
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
        session = self._sessions.pop(sid)
        for name, row in session.rows.items():
            self.pools[name].free([b for b in row if b != ZERO_BLOCK])

    def reset_session(self, sid: int) -> None:
        """Rewind a session to position 0: release its compressed blocks (its ring
        blocks are kept, and re-zeroed by the caller) so another session can use them."""
        session = self._sessions[sid]
        for name, g in self.groups.items():
            row = session.rows[name]
            tail = row[g.ring_blocks :]
            self.pools[name].free([b for b in tail if b != ZERO_BLOCK])
            row[g.ring_blocks :] = [ZERO_BLOCK] * g.compressed_blocks

    @property
    def session_ids(self) -> list[int]:
        return sorted(self._sessions)

    def has_session(self, sid: int) -> bool:
        return sid in self._sessions

    # -- capacity / page tables ----------------------------------------------- #
    def ensure_capacity(self, sid: int, pos: int) -> list[str]:
        """Make sure ``sid`` has blocks for every logical row a step at ``pos`` touches.

        Returns the groups whose row changed (so the caller can refresh just those
        page-table tensors on device). Raises :class:`PagedCacheFull` if the pool is
        exhausted -- the session is left exactly as it was.
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
        """``[1, logical_blocks]`` INT32 page table for one session and group."""
        return torch.tensor(self._sessions[sid].rows[group], dtype=torch.int32).reshape(1, -1)

    # -- reporting ------------------------------------------------------------ #
    def usage(self) -> dict[str, tuple[int, int]]:
        """group -> (blocks in use, pool size), for status output."""
        return {name: (pool.num_blocks - pool.free_blocks, pool.num_blocks) for name, pool in self.pools.items()}

    def tokens_left(self) -> int:
        """Tokens the *tightest* compressor group can still admit across all sessions.

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
