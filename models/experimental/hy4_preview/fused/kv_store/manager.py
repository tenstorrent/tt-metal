# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side manager of the paged KV store (KV_STORE.md 0, 4 option 3, 6, 7): plain Python, no ttnn.

The store is one pool of FRAMES of 8192 tokens. Frame f is, on every chip, slots ``f * 78 + layer`` of the latent
pool tensor ``[frames * 78, 1, 256, 656]`` and slots ``f * 21 + rank`` of the index-key pool tensor
``[frames * 21, 1, 256, 128]`` (``frames.Geometry.pool``): one frame number names both, on all 32 chips. A
conversation is an ordered list of frames: its positions ``[k * 8192, (k + 1) * 8192)`` live in ``frames[k]``.

What the manager owns:
  * the pool: which frame is free, which conversation holds it (never two);
  * the conversations: frames, how many positions are reserved, how many prompt positions are valid ("cached"),
    whether a request is in flight (pinned), the order of last use;
  * the budget: an allocation that does not fit evicts the least recently used conversations that have no request
    in flight, whole, until it fits; when what could be evicted is not enough nothing is evicted and the caller
    waits (``reserve`` returns False);
  * the binding of the decode rows (8 or 16 per pass) to conversations, and the tables the kernels read: per row
    ``TABLE_FRAMES`` entries, the frame number or, for a tensor geometry, the frame base in bytes
    (``frames.Geometry.frame_base``), ``UNMAPPED`` where the conversation has no frame;
  * the frame pairs a workbench prefill needs copied in and out (``kv_copy``).

It does not own time (the order of use is the order of the calls) and it does not talk to the device: the engine
writes ``table(...)`` to the table tensors when ``table_version`` changed and runs the copies it is handed.
"""

from __future__ import annotations

import hashlib
import heapq
from array import array
from dataclasses import dataclass, field
from typing import Callable, Hashable, Optional

from models.experimental.hy4_preview.fused.kv_store.frames import (
    FRAME_TOKENS,
    INDEX,
    KINDS,
    LATENT,
    PERIOD,
    TABLE_FRAMES,
    UNMAPPED,
    Geometry,
    Kind,
)


class FramePool:
    """Free frames of the pool, lowest number first (so a conversation that grows while nothing else allocates
    gets a run, and the used part of the pool stays at its start)."""

    def __init__(self, num_frames: int):
        assert num_frames > 0
        self.num_frames = num_frames
        self._free = list(range(num_frames))  # a sorted list is a heap
        self._is_free = bytearray([1]) * num_frames

    @property
    def free(self) -> int:
        return len(self._free)

    @property
    def used(self) -> int:
        return self.num_frames - len(self._free)

    def take(self, n: int) -> list[int]:
        assert 0 <= n <= len(self._free), f"{n} frames asked, {len(self._free)} free"
        out = [heapq.heappop(self._free) for _ in range(n)]
        for f in out:
            self._is_free[f] = 0
        return out

    def give(self, frames) -> None:
        for f in frames:
            assert not self._is_free[f], f"frame {f} freed twice"
            self._is_free[f] = 1
            heapq.heappush(self._free, f)

    def is_free(self, frame: int) -> bool:
        return bool(self._is_free[frame])


@dataclass
class Conversation:
    key: Hashable
    frames: list = field(default_factory=list)  # frames[k] holds positions [k * F, (k + 1) * F)
    tokens: int = 0  # positions reserved (prompt + what decode may append)
    cached: int = 0  # prompt positions that are valid in the store: what the next request can reuse
    pins: int = 0  # requests in flight
    row: Optional[int] = None  # the decode row it is bound to
    evicted_cached: int = 0  # valid positions an eviction dropped since the conversation was last served
    last_use: int = 0  # the store's clock when a request of it last began or ended; 0: never served
    seq: int = 0  # order of creation (breaks ties of last_use)


@dataclass
class Grant:
    """What ``begin`` gives a request."""

    keep: int  # prompt positions already valid: the prefill restarts at keep // 4096 * 4096
    restart: int
    frames: list  # the conversation's frames after the allocation
    new_frames: list  # the ones this request added
    reprefill: int  # prompt positions that were valid before an eviction and have to be prefilled again


@dataclass
class Stats:
    allocations: int = 0  # reserve calls that took at least one frame
    frames_allocated: int = 0
    frames_freed: int = 0  # by eviction and by shrinking
    evictions: int = 0  # conversations evicted
    evicted_frames: int = 0
    evicted_tokens: int = 0  # valid positions dropped
    reprefill_tokens: int = 0  # of those, positions a later request needed again
    waits: int = 0  # reserve calls that could not be served
    peak_frames: int = 0
    peak_conversations: int = 0


class KvStore:
    def __init__(
        self,
        num_frames: int,
        rows: int = 8,
        frame_tokens: int = FRAME_TOKENS,
        table_frames: int = TABLE_FRAMES,
        on_evict: Optional[Callable[[Conversation], None]] = None,
    ):
        assert frame_tokens % FRAME_TOKENS == 0, "a frame is a multiple of 8192 tokens (KV_STORE.md 1.3)"
        self.frame_tokens = frame_tokens
        self.table_frames = table_frames
        self.rows = rows
        self.pool = FramePool(num_frames)
        self.owner: list = [None] * num_frames  # frame -> key
        self.conversations: dict = {}  # key -> Conversation
        self._clock = 0
        self._created = 0
        self.binding: list = [None] * rows  # decode row -> key
        self.table_version = 0  # counts changes of what table() returns
        self.on_evict = on_evict
        self.stats = Stats()

    @classmethod
    def for_budget(cls, tokens: float, **kw) -> "KvStore":
        """A pool of at most ``tokens`` tokens."""
        frame_tokens = kw.get("frame_tokens", FRAME_TOKENS)
        return cls(int(tokens) // frame_tokens, **kw)

    # ---- sizes -------------------------------------------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return self.pool.num_frames

    @property
    def capacity_tokens(self) -> int:
        return self.pool.num_frames * self.frame_tokens

    @property
    def used_tokens(self) -> int:
        """Tokens of the budget in use: whole frames."""
        return self.pool.used * self.frame_tokens

    def frames_for(self, tokens: int) -> int:
        return -(-tokens // self.frame_tokens)

    def pool_shape(self, kind: Kind) -> tuple:
        """Per-chip shape of the pool tensor of ``kind``; what the prefill stack's allocator makes for
        ``seq_len = 8192 * k, num_users = num_frames, bank_aligned = True`` (KV_STORE.md 4)."""
        rows = self.frame_tokens // 32
        return (self.pool.num_frames * kind.layers, 1, rows, kind.width)

    def pool_bytes_per_chip(self) -> int:
        return self.pool.num_frames * (self.frame_tokens // FRAME_TOKENS) * sum(k.frame_bytes_per_chip for k in KINDS)

    # ---- conversations -------------------------------------------------------------------------------------------
    def conversation(self, key) -> Optional[Conversation]:
        return self.conversations.get(key)

    def cached(self, key) -> int:
        conv = self.conversations.get(key)
        return conv.cached if conv is not None else 0

    def touch(self, key) -> None:
        self._clock += 1
        self.conversations[key].last_use = self._clock

    def lru(self) -> list:
        """Conversations that hold frames, least recently used first (never served ones first, oldest first)."""
        return sorted((c for c in self.conversations.values() if c.frames), key=lambda c: (c.last_use, c.seq))

    def _evictable(self, conv: Conversation, requester: Conversation) -> bool:
        return conv is not requester and conv.pins == 0 and conv.row is None and bool(conv.frames)

    def _evict(self, conv: Conversation) -> None:
        assert conv.pins == 0 and conv.row is None
        n = len(conv.frames)
        for f in conv.frames:
            self.owner[f] = None
        self.pool.give(conv.frames)
        self.stats.evictions += 1
        self.stats.evicted_frames += n
        self.stats.frames_freed += n
        self.stats.evicted_tokens += conv.cached
        conv.evicted_cached = max(conv.evicted_cached, conv.cached)
        conv.frames, conv.tokens, conv.cached = [], 0, 0
        if self.on_evict is not None:
            self.on_evict(conv)

    def evict(self, key) -> None:
        """Drop a conversation's frames (the caller knows it ended, or wants the room)."""
        conv = self.conversations[key]
        if conv.frames:
            self._evict(conv)

    def forget(self, key) -> None:
        """Remove a conversation that holds nothing from the books."""
        conv = self.conversations[key]
        assert not conv.frames and conv.pins == 0 and conv.row is None
        del self.conversations[key]

    def reserve(self, key, tokens: int, evict: bool = True) -> bool:
        """Make the conversation hold the frames of ``tokens`` positions. Grows by allocation (evicting least
        recently used idle conversations when the pool is short and ``evict``), shrinks by freeing tail frames.
        False: it does not fit now (nothing was evicted); call again when a request ended."""
        need = self.frames_for(tokens)
        if need > self.pool.num_frames or need > self.table_frames:
            raise ValueError(
                f"{tokens} tokens: {need} frames, the pool has {self.pool.num_frames}, a table {self.table_frames}"
            )
        conv = self.conversations.get(key)
        if conv is None:
            self._created += 1
            conv = self.conversations[key] = Conversation(key, seq=self._created)
        have = len(conv.frames)
        if need < have:
            tail = conv.frames[need:]
            del conv.frames[need:]
            for f in tail:
                self.owner[f] = None
            self.pool.give(tail)
            self.stats.frames_freed += len(tail)
            conv.cached = min(conv.cached, need * self.frame_tokens)
            self._changed(conv)
        elif need > have:
            extra = need - have
            if self.pool.free < extra:
                victims, room = [], self.pool.free
                if evict:
                    for other in self.lru():
                        if room >= extra:
                            break
                        if self._evictable(other, conv):
                            victims.append(other)
                            room += len(other.frames)
                if room < extra:
                    self.stats.waits += 1
                    return False
                for other in victims:
                    self._evict(other)
            new = self.pool.take(extra)
            for f in new:
                assert self.owner[f] is None
                self.owner[f] = key
            conv.frames.extend(new)
            self.stats.allocations += 1
            self.stats.frames_allocated += extra
            self.stats.peak_frames = max(self.stats.peak_frames, self.pool.used)
            self.stats.peak_conversations = max(
                self.stats.peak_conversations, sum(1 for c in self.conversations.values() if c.frames)
            )
            self._changed(conv)
        conv.tokens = tokens
        return True

    def pin(self, key) -> None:
        self.conversations[key].pins += 1

    def unpin(self, key) -> None:
        conv = self.conversations[key]
        assert conv.pins > 0
        conv.pins -= 1

    def begin(self, key, tokens: int, shared_prefix: Optional[int] = None, evict: bool = True) -> Optional[Grant]:
        """A request for the conversation: ``tokens`` = prompt + the most it may generate, ``shared_prefix`` = how
        many prompt positions equal the conversation's previous prompt (None: all that is cached). Reserves, pins,
        and says where the prefill restarts. None: no room now, nothing changed."""
        before = list(self.conversations[key].frames) if key in self.conversations else []
        if not self.reserve(key, tokens, evict=evict):
            return None
        conv = self.conversations[key]
        conv.pins += 1
        self.touch(key)
        want = conv.cached if shared_prefix is None else shared_prefix
        keep = min(conv.cached, want)
        reprefill = max(min(want, conv.evicted_cached) - keep, 0)
        self.stats.reprefill_tokens += reprefill
        conv.evicted_cached = 0
        conv.cached = keep  # what follows is overwritten by this request's prefill and decode
        return Grant(
            keep=keep,
            restart=keep // PERIOD * PERIOD,
            frames=list(conv.frames),
            new_frames=conv.frames[len(before) :] if conv.frames[: len(before)] == before else list(conv.frames),
            reprefill=reprefill,
        )

    def end(self, key, cached: int, trim: bool = False) -> None:
        """The request returned: ``cached`` prompt positions are valid for the next one (the AgentX client's next
        prompt continues the PROMPT, not the generated tokens). ``trim``: give back the frames beyond them."""
        conv = self.conversations[key]
        self.unpin(key)
        assert cached <= len(conv.frames) * self.frame_tokens
        conv.cached = cached
        self.touch(key)
        if trim and conv.pins == 0:
            self.reserve(key, cached)

    def fork(self, src_key, dst_key, tokens: int) -> Optional[list]:
        """A new conversation whose first ``tokens`` positions equal ``src``'s: allocate its frames and return the
        (source frame, destination frame) pairs to copy inside the pool (``kv_copy``); frames are never shared.
        None: no room. The caller pins the source while the copy runs."""
        src = self.conversations[src_key]
        tokens = min(tokens, src.cached) // PERIOD * PERIOD
        assert dst_key not in self.conversations or not self.conversations[dst_key].frames
        src.pins += 1
        try:
            ok = self.reserve(dst_key, tokens)
        finally:
            src.pins -= 1
        if not ok:
            return None
        dst = self.conversations[dst_key]
        dst.cached = tokens
        return list(zip(src.frames[: len(dst.frames)], dst.frames))

    # ---- decode rows -----------------------------------------------------------------------------------------------
    def _changed(self, conv: Conversation) -> None:
        if conv.row is not None:
            self.table_version += 1

    def bind(self, row: int, key) -> None:
        """Decode row ``row`` serves the conversation from the next pass on."""
        conv = self.conversations[key]
        assert self.binding[row] is None, f"row {row} is bound to {self.binding[row]!r}"
        assert conv.row is None, f"{key!r} is already on row {conv.row}"
        self.binding[row], conv.row = key, row
        self.table_version += 1

    def unbind(self, row: int) -> None:
        key = self.binding[row]
        if key is not None:
            self.conversations[key].row = None
            self.binding[row] = None
            self.table_version += 1

    def free_row(self) -> Optional[int]:
        for row, key in enumerate(self.binding):
            if key is None:
                return row
        return None

    def frame_ids(self, row: int) -> list:
        """The frames of the conversation on ``row`` (``table_frames`` entries, UNMAPPED where it has none)."""
        key = self.binding[row]
        frames = self.conversations[key].frames if key is not None else []
        return list(frames) + [UNMAPPED] * (self.table_frames - len(frames))

    def table(self, geometry: Geometry, rows: Optional[range] = None) -> array:
        """The table of a kind's pool tensor as a flat uint32 array, ``[rows, table_frames]`` row-major: entry
        (row, k) = byte offset, inside a DRAM bank, of the layer-0 shard of the frame that holds the row's stripe
        rows [256 k, 256 k + 256) (kernels/kv_frames.hpp). One array serves every chip, bank and layer."""
        out = array("I")
        for row in range(self.rows) if rows is None else rows:
            out.extend(UNMAPPED if f == UNMAPPED else geometry.frame_base(f) for f in self.frame_ids(row))
        return out

    def tables(self) -> dict:
        """{"latent": ..., "index": ...}: the two tables of the pool tensors."""
        return {kind.name: self.table(Geometry.pool(kind)) for kind in (LATENT, INDEX)}

    def mapped_tokens(self, row: int) -> int:
        """Positions the row's conversation has frames for: an append beyond them needs ``reserve`` first."""
        key = self.binding[row]
        return len(self.conversations[key].frames) * self.frame_tokens if key is not None else 0

    # ---- workbench prefill (KV_STORE.md 4, 3b) ---------------------------------------------------------------------
    def copy_in(self, key, restart: int) -> list:
        """(pool frame, workbench frame) pairs that put the prefix [0, restart) of the conversation into the
        workbench slot before a prefill that restarts at ``restart``."""
        conv = self.conversations[key]
        n = self.frames_for(restart)
        assert n <= len(conv.frames)
        return [(conv.frames[k], k) for k in range(n)]

    def copy_out(self, key, start: int, end: int) -> list:
        """(workbench frame, pool frame) pairs that bring the rows [start, end) a prefill wrote back to the pool.
        A frame the prefill only part-wrote is whole in the workbench (its start was copied in), so whole frames
        are copied."""
        conv = self.conversations[key]
        first, last = start // self.frame_tokens, self.frames_for(end)
        assert last <= len(conv.frames), f"{end} tokens are beyond the {len(conv.frames)} frames reserved"
        return [(k, conv.frames[k]) for k in range(first, last)]

    # ---- invariants ----------------------------------------------------------------------------------------------
    def check(self) -> None:
        """Every frame is free or owned by exactly one conversation; the books agree; tables show bound rows."""
        seen = {}
        for key, conv in self.conversations.items():
            assert len(conv.frames) == self.frames_for(conv.tokens), (key, len(conv.frames), conv.tokens)
            assert conv.cached <= len(conv.frames) * self.frame_tokens
            for f in conv.frames:
                assert f not in seen, f"frame {f} is held by {seen[f]!r} and {key!r}"
                seen[f] = key
                assert self.owner[f] == key and not self.pool.is_free(f)
            assert (conv.row is None) or self.binding[conv.row] == key
        assert len(seen) == self.pool.used <= self.pool.num_frames
        assert sum(1 for o in self.owner if o is not None) == len(seen)
        in_tables = set()
        for row, key in enumerate(self.binding):
            ids = self.frame_ids(row)
            if key is None:
                assert all(f == UNMAPPED for f in ids)
                continue
            conv = self.conversations[key]
            assert conv.row == row and ids[: len(conv.frames)] == conv.frames
            assert all(f == UNMAPPED for f in ids[len(conv.frames) :])
            assert not in_tables & set(conv.frames), f"row {row}: a frame is in two rows' tables"
            in_tables |= set(conv.frames)


class PrefixIndex:
    """Which conversation holds the longest prefix of a prompt, at the grain a prefill can restart on (4096
    tokens): a chain of block digests, ``digest[k] = H(digest[k - 1], tokens[4096 k : 4096 (k + 1)])``. The engine
    registers a conversation's prompt when a request ends and asks with the next prompt. Forks (two conversations
    that share a start) are told apart by how much of the stored conversation the match covers: the engine
    continues a conversation only when the match reaches its last whole block, otherwise it starts a new one
    (``KvStore.fork`` copies the shared frames)."""

    def __init__(self, block: int = PERIOD):
        self.block = block
        self._owner = {}  # chain digest -> (key, blocks)
        self._chains = {}  # key -> list of digests

    def chain(self, tokens) -> list:
        out, h = [], b""
        for k in range(len(tokens) // self.block):
            m = hashlib.blake2b(h, digest_size=16)
            m.update(array("I", tokens[k * self.block : (k + 1) * self.block]).tobytes())
            h = m.digest()
            out.append(h)
        return out

    def register(self, key, tokens) -> None:
        self.remove(key)
        chain = self.chain(tokens)
        self._chains[key] = chain
        for blocks, h in enumerate(chain, start=1):
            self._owner[h] = (key, blocks)  # the latest conversation to store a prefix answers for it

    def remove(self, key) -> None:
        for h in self._chains.pop(key, ()):
            if self._owner.get(h, (None,))[0] == key:
                del self._owner[h]

    def match(self, tokens) -> tuple:
        """(key, shared tokens, whole blocks the conversation has stored) of the longest stored prefix, or
        (None, 0, 0)."""
        best = (None, 0, 0)
        for h in self.chain(tokens):
            hit = self._owner.get(h)
            if hit is None:
                break
            best = (hit[0], hit[1] * self.block, len(self._chains[hit[0]]))
        return best
