# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-side prefix reuse for the GLM-5.3 Flash runner (tt-blaze G53F-0311). Opt-in: ``GLM53F_PREFIX_REUSE=1``.

What the device already keeps. The runner prefills one prompt at a time, in the engine's chunks ``[kC, (k+1)C)``
(``C`` = PREFILL_CHUNK_SIZE, every chunk full but the prompt's last). After a prompt of ``L`` tokens:

* every DSA layer's latent cache rows and pooled-key index hold ``[0, L)`` (the model's own caches; a later prompt
  overwrites only the rows it computes),
* the export slabs hold the ring-format pages of ``[0, L)`` (what the KV Manager copies to the decode ring),
* the KDA carries (``kda_batched``) hold the state at ``L``.

What retention adds: the prompt's token ids and copies of the KDA carries at chunk boundaries (a device-DRAM pool of
``GLM53F_PREFIX_CKPTS`` slots, allocated before the warm-up).

A later prompt still arrives from position 0 (the engine is unchanged; it does not know about the reuse). Its chunk
``[s, e)`` is SKIPPED -- no compute, no DSA export, every layer acknowledged at once -- while every earlier chunk of the
prompt was skipped, its token ids equal the retained ids ``[s, e)`` and ``e <= L``: the slabs already hold ``[s, e)``
byte for byte (the same tokens through the same chunk partition), so the KV Manager ships the right pages. The only
state the KV Manager reads that a skip does not leave right is the KDA window, which it ships ONCE, after the burst's
last chunk. So a skipped chunk that is (or may be) the burst's last must write the KDA carries at ``e`` into the
export slabs, which needs those carries: a checkpoint at ``e``, or the live carries when ``e == L``.

Which chunk is last: the prefill worker marks it (tt-d-gen ``device.prefill.mark_last_chunk``): every chunk header's
slot word carries bit 29 ("this header is marked") and, on a burst's last chunk, bit 30. Per chunk, so nothing has to be
learned or trusted across requests (an aborted request ends without a last chunk; a restarted engine may not mark).
An unmarked header may be the last chunk, so such a chunk is skipped only when its end carries are held.

The first chunk that is not skipped resumes the model at its start ``s``: the KDA carries are restored from the
checkpoint at ``s``, or rebuilt from the nearest checkpoint ``b < s`` by recomputing ``[b, s)`` in full chunks without
export or acks (the slabs already hold those pages; the DSA rows are rewritten with the same values), and every
decoder's ``next_start`` is set to ``s``. Resumed chunks start where a cold prefill's start, so every computed chunk is
the same program on the same inputs: the result is meant to be bit-identical to a cold prefill of the whole prompt
(MEASURED on device, see the device check in ``tests/prefix_reuse``).

Checkpoints: one per chunk boundary while the pool has room; when it is full, the oldest boundary that is not a
multiple of the smallest power-of-two stride (in chunks) that leaves a candidate is evicted, and the two newest are
never evicted. That keeps a roughly even spread over the retained prompt plus its tail (the resume point of a next
turn that extends it). Nothing here is fitted to a workload.

Several retained prompts (phase 2, ``GLM53F_PREFIX_HOST_GB`` > 0; 0 = the above exactly). The device still holds one
prompt. Every full chunk the runner computes also goes into a host-RAM ``ChunkStore``: its DSA latent rows and pooled
keys (what the export reads back anyway), keyed by the chain hash of the token ids ``[0, chunk end)``, so chunks that
several prompts share are stored once. A chunk that the device does not hold but the store does is RESTORED instead of
computed: its rows go back into every chip's caches through the model's own fills, its slab pages are packed again from
them (the export's own functions), and if it may be the burst's last, its end carries come from the store. The KDA
carries of a prompt the device leaves are kept for its two newest boundaries (``GLM53F_PREFIX_SPILL_TIPS``): their
device checkpoints are read into the store -- after the burst's last chunk when headers are marked, at once otherwise --
so a later turn of that conversation resumes at its tip. Eviction is LRU over conversations: every chunk a prompt uses
takes that prompt's stamp, and the least recently used LEAF chunk (no stored chunk extends it) goes first, so a shared
prefix outlives the prompts that extend it. Nothing is fitted to a workload; the only knob is the RAM budget.

Everything device-side is a callback, so this module has no ttnn import and is tested on the host
(``tests/test_prefix_reuse.py``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

MARKED_FLAG = 1 << 29  # tt-d-gen device.prefill.mark_last_chunk: set in EVERY chunk header's slot word
LAST_CHUNK_FLAG = 1 << 30  # ... and this one on a burst's last chunk


def split_slot_word(word: int) -> tuple[int, Optional[bool]]:
    """(slot, last) from the header's slot word: last is True / False on a marked header, None on an unmarked one
    (an engine without mark_last_chunk: any chunk may be the burst's last). Negative words (sentinels) pass through."""
    word = int(word)
    if word < 0 or not word & MARKED_FLAG:
        return word, None
    return word & ~(MARKED_FLAG | LAST_CHUNK_FLAG), bool(word & LAST_CHUNK_FLAG)


def enabled() -> bool:
    return os.environ.get("GLM53F_PREFIX_REUSE", "0") == "1"


CHAIN_SEED = b"glm53f-prefix-chain-v1"


def chain_hash(parent: bytes, ids: torch.Tensor) -> bytes:
    """The chain hash after one more chunk: blake2b-128 of the parent hash and the chunk's int32 ids, so equal hashes
    mean equal ids from position 0 (stored entries also keep their own ids and are compared exactly)."""
    h = hashlib.blake2b(parent, digest_size=16)
    h.update(ids.to(torch.int32).contiguous().numpy().tobytes())
    return h.digest()


@dataclass
class StoreEntry:
    h: bytes
    parent: bytes
    start: int
    end: int
    tokens: torch.Tensor
    rows: object
    nbytes: int
    carries: object = None
    carries_nbytes: int = 0
    children: int = 0
    stamp: int = 0


class ChunkStore:
    """Host-RAM chunks of retained prompts: chain hash at the chunk's end -> its rows (opaque to this module) and, for
    a prompt's tip boundaries, the KDA carries there. LRU over conversations: ``touch`` stamps every chunk a prompt
    uses; eviction takes the oldest-stamped LEAF (no stored child) first and never a chunk of the prompt in flight."""

    def __init__(self, budget_bytes: int, log: Callable[[str], None] = print):
        self.budget = int(budget_bytes)
        self.log = log
        self.entries: dict[bytes, StoreEntry] = {}
        self.bytes = 0
        self.evicted = 0
        self.inserted = 0

    def get(self, h: Optional[bytes]) -> Optional[StoreEntry]:
        return None if h is None else self.entries.get(h)

    def touch(self, h: Optional[bytes], stamp: int) -> None:
        e = self.get(h)
        if e is not None:
            e.stamp = max(e.stamp, stamp)

    def put(self, h, parent, start, end, tokens, rows, nbytes, stamp) -> None:
        if h in self.entries:
            self.touch(h, stamp)
            return
        e = StoreEntry(h, parent, int(start), int(end), tokens.to(torch.int32).clone(), rows, int(nbytes), stamp=stamp)
        self.entries[h] = e
        if parent in self.entries:
            self.entries[parent].children += 1
        self.bytes += e.nbytes
        self.inserted += 1
        self.evict(stamp)

    def attach_carries(self, h, carries, nbytes, stamp) -> bool:
        e = self.get(h)
        if e is None or e.carries is not None:
            return False
        e.carries, e.carries_nbytes = carries, int(nbytes)
        self.bytes += e.carries_nbytes
        self.evict(stamp)
        return True

    def evict(self, protect_stamp: int) -> None:
        """Drop least-recently-used leaves until the store fits its budget; chunks stamped ``protect_stamp`` (the
        prompt in flight) stay even if that leaves the store over budget."""
        while self.bytes > self.budget:
            leaves = [e for e in self.entries.values() if e.children == 0 and e.stamp < protect_stamp]
            if not leaves:
                return
            v = min(leaves, key=lambda e: (e.stamp, -e.start))
            del self.entries[v.h]
            if v.parent in self.entries:
                self.entries[v.parent].children -= 1
            self.bytes -= v.nbytes + v.carries_nbytes
            self.evicted += 1


class CheckpointPool:
    """Boundary position -> checkpoint slot, at most ``n_slots`` (see the module docstring for the eviction rule)."""

    def __init__(self, n_slots: int, chunk: int):
        self.n_slots = int(n_slots)
        self.chunk = int(chunk)
        self.by_pos: dict[int, int] = {}
        self.free = list(range(self.n_slots))
        self.evictions = 0

    def held(self, pos: int) -> Optional[int]:
        return self.by_pos.get(int(pos))

    def nearest_at_or_below(self, pos: int) -> tuple[int, Optional[int]]:
        """(b, slot) of the highest checkpoint at or below ``pos``; (0, None) when there is none (b = 0 needs none)."""
        below = [p for p in self.by_pos if p <= pos]
        if not below:
            return 0, None
        b = max(below)
        return b, self.by_pos[b]

    def drop_above(self, pos: int) -> None:
        for p in [p for p in self.by_pos if p > pos]:
            self.free.append(self.by_pos.pop(p))

    def clear(self) -> None:
        self.drop_above(-1)

    def admit(self, pos: int) -> Optional[int]:
        """A slot for a new boundary ``pos`` (above every held one), evicting one if the pool is full."""
        pos = int(pos)
        if self.n_slots <= 0:
            return None
        if pos in self.by_pos:
            return self.by_pos[pos]
        if not self.free and not self.by_pos:
            return None  # every slot is pinned for a spill (phase 2)
        if not self.free:
            held = sorted(self.by_pos)
            # the newest is protected while another slot can go (spill-pinned slots can leave it the only one)
            protected = {held[-1]} if self.n_slots >= 3 and len(held) > 1 else set()
            stride = self.chunk
            while True:
                cands = [p for p in held if p not in protected and p % stride]
                if cands:
                    victim = min(cands)
                    break
                if (
                    stride > held[-1]
                ):  # every candidate is a multiple of a stride past them all: impossible unless empty
                    victim = min(p for p in held if p not in protected)
                    break
                stride *= 2
            self.free.append(self.by_pos.pop(victim))
            self.evictions += 1
        slot = self.free.pop()
        self.by_pos[pos] = slot
        return slot


@dataclass
class ReuseStats:
    prompts: int = 0
    resumed_prompts: int = 0
    skipped_chunks: int = 0
    skipped_tokens: int = 0
    silent_chunks: int = 0
    computed_chunks: int = 0
    checkpoints_saved: int = 0
    kda_slab_writes: int = 0
    marked_chunks: int = 0
    restored_chunks: int = 0
    restored_tokens: int = 0
    spills: int = 0
    host_carries_loaded: int = 0
    extra: dict = field(default_factory=dict)


class PrefixRetention:
    """The retained prompt and the skip / resume decisions. Device actions are callbacks:

    ``save(slot)``             live KDA carries -> checkpoint slot
    ``restore(slot)``          checkpoint slot -> live KDA carries
    ``export_carries(slot)``   carries of checkpoint ``slot`` (None = the live ones) -> the KDA export slabs (+ sync)
    ``set_next_start(pos)``    every decoder's chunked-prefill cursor
    ``compute_silent(ids, start)``  one chunk through the model, no export, no acks (the carries then hold its end)

    Phase 2 (``store`` given):
    ``restore_rows(rows, start)``  a stored chunk's rows -> the device caches + the export slab pages of [start, end)
    ``spill(slot) -> (carries, nbytes)``  checkpoint slot -> host carries;  ``load_carries(carries)``  -> live carries
    """

    def __init__(
        self,
        *,
        chunk: int,
        n_ckpts: int,
        save: Callable[[int], None],
        restore: Callable[[int], None],
        export_carries: Callable[[Optional[int]], None],
        set_next_start: Callable[[int], None],
        compute_silent: Callable[[torch.Tensor, int], None],
        log: Callable[[str], None] = print,
        store: Optional[ChunkStore] = None,
        restore_rows: Optional[Callable[[object, int], None]] = None,
        spill: Optional[Callable[[int], tuple]] = None,
        load_carries: Optional[Callable[[object], None]] = None,
        spill_tips: int = 2,
    ):
        self.chunk = int(chunk)
        self.pool = CheckpointPool(n_ckpts, chunk)
        self._save, self._restore, self._export = save, restore, export_carries
        self._set_next_start, self._compute_silent = set_next_start, compute_silent
        self.log = log
        self.tokens = torch.zeros(0, dtype=torch.int32)
        self.length = 0  # rows [0, length) of the caches / slabs hold tokens[:length]; the live carries are at length
        # the in-flight prompt: every chunk so far skipped (matching), and where its next chunk must start (cursor)
        self.matching = False
        self.cursor = 0
        self.prompt_resumed = False
        self.last_resume: Optional[dict] = None  # set by a chunk that resumed a matched prompt (for the caller's log)
        self.stats = ReuseStats()
        self.live_at: Optional[int] = None  # the live carries are the device chain's state at this position
        # phase 2: the host store and the in-flight prompt's chain hashes (position -> hash of ids [0, position))
        self.store = store
        self._restore_rows, self._spill, self._load_carries = restore_rows, spill, load_carries
        self.spill_tips = int(spill_tips)
        self.stamp = 0
        self.p_pos = 0
        self.p_hash: Optional[bytes] = None
        self.p_hashes: dict[int, bytes] = {}
        self._cur: Optional[tuple] = None  # (start, end, parent hash, hash) of the chunk being processed
        self.pending_spills: list[tuple[bytes, int]] = []  # (chain hash, pinned slot) read after the burst

    # ------------------------------------------------------------------ state
    def reset(self, why: str) -> None:
        self.tokens = torch.zeros(0, dtype=torch.int32)
        self.length = 0
        self.live_at = None
        self.pool.clear()
        for _h, slot in self.pending_spills:  # not read: what the device holds can no longer be vouched for
            self.pool.free.append(slot)
        self.pending_spills = []
        self.matching = False
        self.cursor = 0
        self.log(f"[glm53f-prefix] retention reset ({why})")

    def carries_at(self, pos: int):
        """("live", None) when the live carries are at ``pos``, ("ckpt", slot) for a checkpoint, else None."""
        if self.live_at is not None and pos == self.live_at:
            return ("live", None)
        slot = self.pool.held(pos)
        return None if slot is None else ("ckpt", slot)

    # ------------------------------------------------------------------ per chunk
    def begin_chunk(self, ids: torch.Tensor, start: int, end: int, last: Optional[bool]) -> bool:
        """Called before a non-empty chunk ``[start, end)``; ``last`` from ``split_slot_word`` (None = unmarked header).
        True = skipped (the caller only acknowledges it); False = compute it (the model is resumed at ``start`` if the
        prompt matched so far)."""
        if last is not None:
            self.stats.marked_chunks += 1
        if start == 0:
            self._new_prompt()
        ids = ids.to(torch.int32)
        self._chain_step(ids, start, end)
        if self.matching and start == self.cursor and end <= self.length and torch.equal(ids, self.tokens[start:end]):
            # a ragged chunk is always the burst's last; a full one is when marked so, or may be when unmarked
            may_be_last = last is None or last or (end - start) < self.chunk
            if not may_be_last:
                self._skipped(start, end)
                return True
            at = self.carries_at(end)
            if at is not None:
                self._export(at[1])
                self.stats.kda_slab_writes += 1
                self._skipped(start, end)
                return True
            # a possible last chunk whose end carries are not held: compute it (resumed at start below)
        elif self.store is not None and self.matching and start == self.cursor and self._try_restore(ids, last):
            return True
        self._prepare_compute(start, last)
        return False

    def note_empty(self) -> None:
        """An EMPTY chunk (a burst declined before compute): nothing computed or skipped, but it ends a request."""
        self.matching = False

    def end_compute(self, ids: torch.Tensor, start: int, end: int) -> None:
        """After the model computed ``[start, end)`` (the live carries are at ``end``): retain it."""
        self._record(ids, start, end)
        self.stats.computed_chunks += 1

    def wants_rows(self, start: int, end: int) -> bool:
        """Phase 2: should the caller hand this computed chunk's rows to ``store_rows`` (a full chunk of a hashed chain
        the store does not hold yet)?"""
        c = self._cur
        return (
            self.store is not None
            and c is not None
            and c[0] == start
            and c[1] == end
            and c[3] is not None
            and self.store.get(c[3]) is None
        )

    def store_rows(self, ids: torch.Tensor, start: int, end: int, rows, nbytes: int) -> None:
        c = self._cur
        if not self.wants_rows(start, end):
            return
        self.store.put(c[3], c[2], start, end, ids, rows, nbytes, self.stamp)

    def flush_spills(self) -> None:
        """Read the pinned checkpoints of prompts the device left into the store (after a burst's last chunk)."""
        pend, self.pending_spills = self.pending_spills, []
        for h, slot in pend:
            if self.store is not None and self.store.get(h) is not None and self.store.get(h).carries is None:
                carries, nbytes = self._spill(slot)
                if self.store.attach_carries(h, carries, nbytes, self.stamp):
                    self.stats.spills += 1
            self.pool.free.append(slot)

    def invalidate(self, why: str) -> None:
        """A chunk failed part-way: nothing on the device can be vouched for."""
        self.reset(why)

    # ------------------------------------------------------------------ internals
    def _new_prompt(self) -> None:
        self.stats.prompts += 1
        self.stamp += 1
        self.p_pos, self.p_hash, self.p_hashes = 0, CHAIN_SEED, {0: CHAIN_SEED}
        self.matching = self.length > 0
        self.cursor = 0
        self.prompt_resumed = False

    def _chain_step(self, ids: torch.Tensor, start: int, end: int) -> None:
        """Phase 2: the in-flight prompt's chain hash at ``end`` (full chunks only; a ragged chunk ends the chain)."""
        self._cur = None
        if self.store is None:
            return
        if self.p_hash is None or start != self.p_pos:
            self.p_hash = None  # not contiguous with what this prompt sent so far: no chain from here on
            return
        h = chain_hash(self.p_hash, ids) if end - start == self.chunk else None
        self._cur = (start, end, self.p_hash, h)
        self.store.touch(h, self.stamp)
        self.p_pos, self.p_hash = end, h
        if h is not None:
            self.p_hashes[end] = h

    def _try_restore(self, ids: torch.Tensor, last: Optional[bool]) -> bool:
        """Phase 2: the device does not hold this chunk but the store does: restore it instead of computing it."""
        c = self._cur
        if c is None or c[3] is None:
            return False
        start, end, _parent, h = c
        ent = self.store.get(h)
        if ent is None or ent.start != start or not torch.equal(ent.tokens, ids):
            return False
        may_be_last = last is None or last
        if may_be_last and ent.carries is None:
            return False  # its end carries would be needed and are not stored: compute it
        self._leaving(start, last)
        self._truncate(start)
        self._restore_rows(ent.rows, start)
        self._record_rows(ids, start, end)
        if may_be_last:
            self._load_carries(ent.carries)
            self.live_at = end
            self._export(None)
            self.stats.kda_slab_writes += 1
            self.stats.host_carries_loaded += 1
        self.stats.restored_chunks += 1
        self.stats.restored_tokens += end - start
        self._skipped(start, end)
        return True

    def _leaving(self, pos: int, last: Optional[bool]) -> None:
        """Phase 2: the device chain is about to be overwritten from ``pos``: keep the carries of its newest
        boundaries above ``pos`` (the next turn of that conversation resumes there) by pinning their checkpoint slots
        for a spill into the store -- after the burst's last chunk on a marked header, at once on an unmarked one."""
        if self.store is None or pos >= self.length or self.spill_tips <= 0:
            return
        above = sorted((p for p in self.pool.by_pos if p > pos), reverse=True)[: self.spill_tips]
        if not above:
            return
        h = CHAIN_SEED
        hashes = {}
        for x in range(0, max(above), self.chunk):
            h = chain_hash(h, self.tokens[x : x + self.chunk])
            hashes[x + self.chunk] = h
        for p in above:
            ent = self.store.get(hashes[p])
            if ent is None or ent.carries is not None:
                continue
            self.pending_spills.append((hashes[p], self.pool.by_pos.pop(p)))
        if last is None:
            self.flush_spills()

    def _skipped(self, start: int, end: int) -> None:
        self.cursor = end
        self.stats.skipped_chunks += 1
        self.stats.skipped_tokens += end - start
        if not self.prompt_resumed:
            self.prompt_resumed = True
            self.stats.resumed_prompts += 1

    def _prepare_compute(self, start: int, last: Optional[bool] = None) -> None:
        resume = self.matching and start > 0 and start == self.cursor
        self.matching = False
        self.last_resume = None
        if resume or start == 0:
            self._leaving(start, last)
        if not resume:
            if start == 0:
                # a new prompt computed from scratch (the model zeroes the KDA carries at start 0): only [0, 0) stays
                self._truncate(0)
            elif start != self.length:
                # not the continuation of the prompt the device holds (the engine's tt-d-gen #430 re-send): the model's
                # own contiguity check decides; nothing retained can be vouched for past this point
                self.reset(f"chunk start {start} is neither 0 nor the retained end {self.length}")
            return
        at = self.carries_at(start)
        self.last_resume = {"start": start, "how": "live" if at is not None and at[0] == "live" else "checkpoint"}
        host = self._host_carries(start) if at is None else None
        if host is not None:
            self._load_carries(host)
            self.stats.host_carries_loaded += 1
            self.live_at = start
            self._set_next_start(start)
            self.length = start
            self.last_resume = {"start": start, "how": "host carries"}
        elif at is not None and at[0] == "live":
            self._set_next_start(start)  # the live carries are already at start (the retained prompt ended there)
        elif at is not None:
            self._restore(at[1])
            self.live_at = start
            self._set_next_start(start)
            self.length = start
        else:
            b, slot = self.pool.nearest_at_or_below(start)
            hb = max((p for p in self.p_hashes if b < p <= start and self._host_carries(p) is not None), default=None)
            if hb is not None:  # phase 2: a stored tip of this chain is nearer than any device checkpoint
                b, slot = hb, None
                self._load_carries(self._host_carries(hb))
                self.stats.host_carries_loaded += 1
            elif slot is not None:
                self._restore(slot)
            self.live_at = b if b else None
            self._set_next_start(b)
            self.length = b
            n = 0
            for x in range(b, start, self.chunk):
                y = min(start, x + self.chunk)
                self._compute_silent(self.tokens[x:y].to(torch.int64), x)
                self._record(self.tokens[x:y], x, y)
                n += 1
            self.stats.silent_chunks += n
            self.last_resume = {"start": start, "how": f"rebuilt from {b}", "silent": n}
            self.log(f"[glm53f-prefix] rebuilt the KDA carries at {start} from {b} ({n} chunk(s), no export)")
        self._truncate(start)

    def _host_carries(self, pos: int):
        """Phase 2: stored KDA carries of the in-flight prompt's chain at boundary ``pos``."""
        if self.store is None:
            return None
        ent = self.store.get(self.p_hashes.get(pos))
        return None if ent is None else ent.carries

    def _truncate(self, pos: int) -> None:
        """The chunk about to be computed overwrites everything from ``pos`` on."""
        self.length = min(self.length, pos)
        if self.live_at is not None and self.live_at > pos:
            self.live_at = None
        self.pool.drop_above(pos)

    def _record_rows(self, ids: torch.Tensor, start: int, end: int) -> None:
        """The device chain now holds ``ids`` at [start, end) (rows written; the live carries untouched)."""
        ids = ids.to(torch.int32).flatten()
        assert int(ids.numel()) == end - start, (ids.numel(), start, end)
        if int(self.tokens.numel()) < end:
            grown = torch.zeros(max(end, 2 * int(self.tokens.numel())), dtype=torch.int32)
            grown[: int(self.tokens.numel())] = self.tokens
            self.tokens = grown
        self.tokens[start:end] = ids
        self.length = end
        self.pool.drop_above(start)

    def _record(self, ids: torch.Tensor, start: int, end: int) -> None:
        self._record_rows(ids, start, end)
        self.live_at = end
        if end % self.chunk == 0:
            slot = self.pool.admit(end)
            if slot is not None:
                self._save(slot)
                self.stats.checkpoints_saved += 1

    def summary(self) -> str:
        s = self.stats
        return (
            f"retained {self.length} tokens, {len(self.pool.by_pos)} checkpoint(s) at "
            f"{sorted(p // self.chunk for p in self.pool.by_pos)} chunk(s); prompts {s.prompts}, resumed {s.resumed_prompts}, "
            f"skipped {s.skipped_chunks} chunk(s) / {s.skipped_tokens} tokens, silent {s.silent_chunks}, computed "
            f"{s.computed_chunks}, checkpoints saved {s.checkpoints_saved} (evicted {self.pool.evictions}), KDA slab "
            f"writes {s.kda_slab_writes}, marked headers {s.marked_chunks}"
            + (
                f"; store {len(self.store.entries)} chunk(s) {self.store.bytes / 2**30:.2f} GiB (evicted "
                f"{self.store.evicted}), restored {s.restored_chunks} chunk(s), spills {s.spills}, host carries "
                f"{s.host_carries_loaded}"
                if self.store is not None
                else ""
            )
        )
