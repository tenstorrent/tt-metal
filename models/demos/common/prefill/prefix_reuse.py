# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-side prefix reuse, the model-independent half. H1: one retained prompt per slot, on the device. H2: a host-RAM
chunk store of several retained prompts. Ported from the GLM-5.3 Flash runner (tt-blaze G53F-0311 / G53F-0256 / G53F-0315,
``models/demos/glm53_flash/tt/runners/prefix_reuse.py`` on ``sdawle/glm53f_fix5_runner``) for DeepSeek-V4.1-Flash
(``tt/v41/prefix_reuse.py`` holds the device half). torch only: no ttnn import, every device action is a callback, so the
decisions are tested on the host (``models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse``).

What the device already keeps. The runner prefills one prompt at a time per slot, in the engine's chunks ``[kC, (k+1)C)``
(``C`` = PREFILL_CHUNK_SIZE, every chunk full but the prompt's last), and a later prompt overwrites only the rows it
computes. So after a prompt of ``L`` tokens the model's position-indexed caches hold ``[0, L)`` and the export holds the
rows the KV Manager ships for ``[0, L)``. What is NOT position-indexed (GLM: the KDA carries; V4.1: the 128-row window
carries of every attention layer) is held only at ``L``; retention adds copies of it at chunk boundaries (a device-DRAM pool
of ``n_ckpts`` checkpoint slots) and the prompt's token ids.

A later prompt still arrives from position 0 (the engine is unchanged). Its chunk ``[s, e)`` is SKIPPED -- no compute, every
layer acknowledged at once -- while every earlier chunk of the prompt was skipped, its ids equal the retained ids
``[s, e)`` and ``e <= L``. A skipped chunk that is (or may be) the burst's last must leave the export as a cold prefill
of ``[0, e)`` would: ``export_carries(slot, e, final)`` writes the non-positional state at ``e`` into the export (from a
checkpoint at ``e``, or the live state when ``e == L``); so such a chunk is skipped only when that state is held.
``final`` = known to be the last (a marked last header or a ragged chunk); ``export_carries`` returns True when it also
CLEARED the export past ``e`` (V4.1: the zero a cold prefill's chunk 0 does), and the retention then forgets ``[e, L)``.
``skip_partial_last=False`` (V4.1) never skips a partial (ragged) chunk: it is computed, resumed at its start, because the
export's state at a partial end needs more than the carries at ``e`` (V4.1's window ring also holds the previous boundary's).

Which chunk is last: the prefill worker marks it (tt-d-gen ``device.prefill.mark_last_chunk``): every chunk header's slot
word carries bit 29 ("this header is marked") and, on a burst's last chunk, bit 30 (``split_slot_word``). Per header, so
nothing is learned across requests (an aborted request ends without a last chunk; a restarted engine may not mark). An
unmarked header may be the last chunk.

The first chunk that is not skipped RESUMES the model at its start ``s``: its non-positional state is restored from the
checkpoint at ``s`` (or is live), or rebuilt from the nearest checkpoint ``b < s`` by recomputing ``[b, s)`` in full chunks
without export or acks (``compute_silent``: the export already holds those rows, the model's rows are rewritten with the
same values); ``set_next_start(pos)`` puts the model's counters at ``pos``; ``on_resume(s)`` then fixes the export for a
resume at ``s`` (V4.1: window rings at ``s``, rows past ``s`` cleared). Resumed chunks start where a cold prefill's
start, so every computed chunk is the same program on the same inputs: meant to be bit-identical to a cold prefill.

Checkpoints: one per chunk boundary while the pool has room; when it is full, the oldest boundary that is not a multiple
of the smallest power-of-two stride (in chunks) that leaves a candidate is evicted, and the two newest are never evicted.
Nothing here is fitted to a workload.

H2, several retained prompts (``store`` given; None = the above exactly, call for call). The device still holds one prompt
per slot. Every FULL chunk the runner computes also goes into a host-RAM ``ChunkStore``: its position-indexed rows (what a
restore must write back; opaque to this module), keyed by the chain hash of the token ids ``[0, chunk end)``, so chunks
that several prompts share are stored once. While a prompt matches (every chunk so far skipped or restored), a chunk that
the device does not hold but the store does is RESTORED instead of computed: ``restore_rows(rows, ids, start)`` writes its
rows back through the model's own writers and repacks its export rows; if it may be the burst's last, its END carries must
be stored too (``load_carries`` + ``export_carries``), else it is computed. A ragged chunk is never stored, so a partial last
chunk is always computed (H1's V4.1 rule). The carries of a prompt the device leaves are kept for its ``spill_tips``
newest boundaries: their device checkpoints are read into the store (``spill``) -- after the burst's last chunk when headers
are marked, at once otherwise -- so a later turn of that conversation resumes at its tip (``load_carries``). Eviction is
LRU over conversations: every chunk a prompt uses takes that prompt's stamp, and the least recently used LEAF chunk (no
stored chunk extends it) goes first, so a shared prefix outlives the prompts that extend it; the chunks of the prompt in
flight are never evicted. Nothing is fitted to a workload; the knobs are the RAM budget and the spill count.

H2's memory (G53F-0315: GLM's per-chunk heap tensors made a 64 GiB budget a 130 GB brk heap): the store's bytes live in two
FIXED arenas of equal-size slots (``SlotArena``: one anonymous private mapping each, page-aligned, committed by the kernel on
first write, never resized), one for chunk rows and a small one for tip carries. The store's resident size is at most the two
arenas, whatever the eviction history; a host-memory floor (``reserve_bytes``: MemAvailable) evicts and hands slot pages back
to the OS (madvise DONTNEED), and ``malloc_trim`` returns the transient read buffers' heap.
"""

from __future__ import annotations

import ctypes
import hashlib
import mmap
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import torch

MARKED_FLAG = 1 << 29  # tt-d-gen device.prefill.mark_last_chunk: set in EVERY chunk header's slot word
LAST_CHUNK_FLAG = 1 << 30  # ... and this one on a burst's last chunk
_SENTINEL_BIT = 1 << 31  # the runner's shutdown / warm-up words (0xFFFFFFFF-style): never split


def split_slot_word(word: int) -> tuple[int, Optional[bool]]:
    """(slot, last) from a chunk header's slot word: last is True / False on a marked header, None on an unmarked one (an
    engine without mark_last_chunk: any chunk may be the burst's last). Negative words and words with bit 31 set (the
    runner's sentinels) pass through untouched; bit 30 means nothing without bit 29."""
    word = int(word)
    if word < 0 or word & _SENTINEL_BIT or not word & MARKED_FLAG:
        return word, None
    return word & ~(MARKED_FLAG | LAST_CHUNK_FLAG), bool(word & LAST_CHUNK_FLAG)


# ------------------------------------------------------------------------------------------------ H2: chain hash, memory
DEFAULT_CHAIN_SEED = b"prefill-prefix-chain-v1"


def chain_hash(parent: bytes, ids: torch.Tensor) -> bytes:
    """The chain hash after one more chunk: blake2b-128 of the parent hash and the chunk's int32 ids, so equal hashes mean
    equal ids from position 0 (stored entries also keep their own ids and are compared exactly)."""
    h = hashlib.blake2b(parent, digest_size=16)
    h.update(ids.to(torch.int32).contiguous().numpy().tobytes())
    return h.digest()


def mem_available() -> Optional[int]:
    """The host's MemAvailable in bytes (None where /proc/meminfo is unreadable)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:  # not glibc: no trim; the arenas still bound the store's own bytes
    _LIBC = None

# Bytes read through transient host tensors (a capture's reads, a spill's) between two malloc_trim(0) calls: the arenas never
# touch the heap, but the reads land there first (G53F-0256 11-09: freed holes stay resident until a trim).
TRIM_EVERY_BYTES = 2 << 30
PAGE = mmap.PAGESIZE
_ALIGN = 64  # each part of a slot starts on a 64-byte boundary (any element size divides it)
# MAP_NORESERVE: no swap reservation for the arena's untouched pages (Python 3.10's mmap module lacks the constant)
_MAP_NORESERVE = getattr(mmap, "MAP_NORESERVE", 0x4000 if sys.platform.startswith("linux") else 0)


def _round_up(n: int, a: int) -> int:
    return -(-int(n) // a) * a


def packed_bytes(parts) -> int:
    """The bytes ``SlotArena.pack`` needs for ``parts`` (each part 64-byte aligned)."""
    off = 0
    for p in parts:
        off = _round_up(off, _ALIGN) + int(p.numel()) * p.element_size()
    return off


class SlotArena:
    """``n_slots`` equal byte slots in ONE anonymous private mapping: page-aligned, committed by the kernel on first write
    (the resident size is the slots ever written, at most ``n_slots x slot_bytes``), a slot's pages handed back to the OS
    with madvise(DONTNEED) (``release``). Fixed for the process's life: no allocator, no fragmentation (G53F-0315)."""

    def __init__(self, n_slots: int, slot_bytes: int):
        self.n_slots = int(n_slots)
        self.slot_bytes = _round_up(max(1, int(slot_bytes)), PAGE)
        self._mm = mmap.mmap(
            -1, max(1, self.n_slots) * self.slot_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_NORESERVE
        )
        self._buf = torch.frombuffer(self._mm, dtype=torch.uint8)
        self.free = list(range(self.n_slots - 1, -1, -1))  # pop() hands out slot 0 first
        self.written: set = set()  # slots whose pages may be resident (written since their last release)
        self.releases = 0

    @property
    def size_bytes(self) -> int:
        return self.n_slots * self.slot_bytes

    @property
    def resident_bytes(self) -> int:
        """An upper bound of the arena's resident pages: the slots written since their last release."""
        return len(self.written) * self.slot_bytes

    @property
    def used(self) -> int:
        return self.n_slots - len(self.free)

    def alloc(self) -> Optional[int]:
        return self.free.pop() if self.free else None

    def put_back(self, k: int, release: bool = False) -> int:
        """Slot ``k`` is free again; ``release`` also hands its pages to the OS. Returns the resident bytes released."""
        freed = self.release(k) if release else 0
        self.free.append(int(k))
        return freed

    def release(self, k: int) -> int:
        if k not in self.written:
            return 0
        self._mm.madvise(mmap.MADV_DONTNEED, int(k) * self.slot_bytes, self.slot_bytes)
        self.written.discard(k)
        self.releases += 1
        return self.slot_bytes

    def release_free(self, need: int) -> int:
        """Hand back the pages of free slots that still hold some (normal eviction keeps them for reuse) until ``need``
        bytes are released. Returns the bytes released."""
        freed = 0
        for k in list(self.free):
            if freed >= need:
                break
            freed += self.release(k)
        return freed

    def pack(self, k: int, parts) -> tuple:
        """Copy ``parts`` (host tensors) into slot ``k``; returns their layout ((offset, shape, dtype), ...)."""
        assert packed_bytes(parts) <= self.slot_bytes, (packed_bytes(parts), self.slot_bytes)
        base, off, layout = int(k) * self.slot_bytes, 0, []
        for p in parts:
            p = p.contiguous()
            off = _round_up(off, _ALIGN)
            n = int(p.numel()) * p.element_size()
            self._buf[base + off : base + off + n].copy_(p.reshape(-1).view(torch.uint8))
            layout.append((off, tuple(p.shape), p.dtype))
            off += n
        self.written.add(int(k))
        return tuple(layout)

    def unpack(self, k: int, layout) -> list:
        """Views (no copy) of the parts in slot ``k``: valid until the slot is reused."""
        base, out = int(k) * self.slot_bytes, []
        for off, shape, dtype in layout:
            n = int(torch.Size(shape).numel()) * torch.empty(0, dtype=dtype).element_size()
            out.append(self._buf[base + off : base + off + n].view(dtype).view(shape))
        return out


@dataclass
class StoreEntry:
    h: bytes
    parent: bytes
    start: int
    end: int
    tokens: torch.Tensor
    slot: int
    layout: tuple
    nbytes: int
    tip: Optional[int] = None  # tip-arena slot of the carries at ``end`` (None: not stored)
    tip_layout: tuple = ()
    children: int = 0  # stored entries whose parent is this one
    stamp: int = 0

    @property
    def has_carries(self) -> bool:
        return self.tip is not None


@dataclass
class StoreStats:
    inserted: int = 0
    evicted: int = 0
    refused: int = 0  # a chunk not stored: no slot could be freed (every one belongs to the prompt in flight)
    pressure_evictions: int = 0  # evicted for the host-memory floor, not the budget
    released_bytes: int = 0  # slot pages handed back to the OS (madvise)
    tips_attached: int = 0
    tip_drops: int = 0  # carries dropped from an older entry to make room in the tip arena
    tips_refused: int = 0
    trims: int = 0


class ChunkStore:
    """Host-RAM chunks of retained prompts: chain hash at the chunk's end -> its rows (a slot of the chunk arena) and, for
    a prompt's tip boundaries, the carries there (a slot of the tip arena). LRU over conversations: ``touch`` stamps every
    chunk a prompt uses; a slot is freed by evicting the oldest-stamped LEAF (no stored child), never a chunk of the prompt
    in flight (stamped ``protect``); if none can go, the chunk is not stored. A full tip arena drops the oldest-stamped
    entry's carries (the entry stays). The arenas are made at the first entry (their slot size is the first record's), with
    ``budget_bytes`` split: ``tip_budget_bytes`` (default 1/16) for tips, the rest for chunk rows."""

    def __init__(
        self,
        budget_bytes: int,
        *,
        tip_budget_bytes: Optional[int] = None,
        log: Callable[[str], None] = print,
        reserve_bytes: int = 0,
        available: Callable[[], Optional[int]] = mem_available,
        tag: str = "prefix-store",
    ):
        self.budget = max(0, int(budget_bytes))
        tips = self.budget // 16 if tip_budget_bytes is None else int(tip_budget_bytes)
        self.tip_budget = max(0, min(tips, self.budget))
        self.chunk_budget = self.budget - self.tip_budget
        # Host-memory floor (0 = off): while the host's MemAvailable is below it, the store evicts and releases pages by the
        # deficit, whatever took the memory (G53F-0256 11-09)
        self.reserve = max(0, int(reserve_bytes))
        self.available = available
        self.log, self.tag = log, tag
        self.entries: dict[bytes, StoreEntry] = {}
        self.arena: Optional[SlotArena] = None
        self.tips: Optional[SlotArena] = None
        self.no_chunks: Optional[str] = None  # why the chunk arena could not be made (the store then stores nothing)
        self.no_tips: Optional[str] = None
        self.stats = StoreStats()
        self.read_since_trim = 0

    # ---- reads
    def get(self, h: Optional[bytes]) -> Optional[StoreEntry]:
        return None if h is None else self.entries.get(h)

    def touch(self, h: Optional[bytes], stamp: int) -> None:
        e = self.get(h)
        if e is not None:
            e.stamp = max(e.stamp, int(stamp))

    def rows(self, e: StoreEntry) -> list:
        return self.arena.unpack(e.slot, e.layout)

    def carries(self, e: Optional[StoreEntry]):
        return None if e is None or e.tip is None else self.tips.unpack(e.tip, e.tip_layout)

    @property
    def bytes(self) -> int:
        """The slots in use (entries' rows + their carries), in arena bytes."""
        n = len(self.entries) * self.arena.slot_bytes if self.arena is not None else 0
        return n + (self.tips.used * self.tips.slot_bytes if self.tips is not None else 0)

    @property
    def resident_bytes(self) -> int:
        return sum(a.resident_bytes for a in (self.arena, self.tips) if a is not None)

    @property
    def capacity(self) -> int:
        """Chunk slots (0 before the first entry made the arena)."""
        return self.arena.n_slots if self.arena is not None else 0

    # ---- writes
    def _make(self, budget: int, nbytes: int, what: str) -> tuple[Optional[SlotArena], Optional[str]]:
        slot = _round_up(nbytes, PAGE)
        n = int(budget) // slot
        if n <= 0:
            why = f"{what} budget {budget / 2**20:.1f} MiB < one {slot / 2**20:.2f} MiB slot"
            self.log(f"[{self.tag}] {what} arena NOT made: {why}")
            return None, why
        a = SlotArena(n, slot)
        self.log(
            f"[{self.tag}] {what} arena: {n} slot(s) x {slot / 2**20:.3f} MiB = {a.size_bytes / 2**30:.2f} GiB (one fixed "
            f"anonymous mapping, pages committed on first write)"
        )
        return a, None

    def put(self, h, parent, start, end, tokens, parts, stamp) -> bool:
        """Store a chunk's rows (``parts``: host tensors) under ``h``. False = not stored (no room / the store is off)."""
        if h in self.entries:
            self.touch(h, stamp)
            return True
        nbytes = packed_bytes(parts)
        if self.arena is None:
            if self.no_chunks is not None:
                return False
            self.arena, self.no_chunks = self._make(self.chunk_budget, nbytes, "chunk")
            if self.arena is None:
                return False
        if nbytes > self.arena.slot_bytes:
            self.stats.refused += 1
            self.log(f"[{self.tag}] chunk [{start}, {end}) NOT stored: {nbytes} B > the {self.arena.slot_bytes} B slot")
            return False
        self._press(stamp)
        k = self._free_slot(stamp)
        if k is None:
            self.stats.refused += 1
            return False
        layout = self.arena.pack(k, parts)
        e = StoreEntry(
            h, parent, int(start), int(end), tokens.to(torch.int32).clone(), k, layout, nbytes, stamp=int(stamp)
        )
        e.children = sum(1 for x in self.entries.values() if x.parent == h)
        self.entries[h] = e
        if parent in self.entries and parent != h:
            self.entries[parent].children += 1
        self.stats.inserted += 1
        self._read(nbytes)
        return True

    def attach_carries(self, h, parts, stamp) -> bool:
        """The carries at entry ``h``'s end (``parts``: host tensors) into the tip arena. False = not attached."""
        e = self.get(h)
        if e is None or e.tip is not None:
            return False
        nbytes = packed_bytes(parts)
        if self.tips is None:
            if self.no_tips is not None:
                return False
            self.tips, self.no_tips = self._make(self.tip_budget, nbytes, "tip")
            if self.tips is None:
                return False
        if nbytes > self.tips.slot_bytes:
            self.stats.tips_refused += 1
            return False
        self._press(stamp)
        if self.get(h) is None:  # the floor evicted it
            return False
        k = self.tips.alloc()
        if k is None:
            holders = [x for x in self.entries.values() if x.tip is not None and x.stamp < stamp]
            if holders:
                v = min(holders, key=lambda x: (x.stamp, -x.start))
                self.tips.put_back(v.tip)
                v.tip, v.tip_layout = None, ()
                self.stats.tip_drops += 1
                k = self.tips.alloc()
        if k is None:
            self.stats.tips_refused += 1
            return False
        e.tip_layout = self.tips.pack(k, parts)
        e.tip = k
        self.stats.tips_attached += 1
        self._read(nbytes)
        return True

    def _leaves(self, protect: int) -> list:
        return [x for x in self.entries.values() if x.children == 0 and x.stamp < protect]

    def _drop(self, v: StoreEntry, release: bool = False) -> int:
        """Evict entry ``v`` (its rows and its carries). Returns the resident bytes released (``release``)."""
        del self.entries[v.h]
        if v.parent in self.entries and v.parent != v.h:
            self.entries[v.parent].children -= 1
        freed = self.arena.put_back(v.slot, release)
        if v.tip is not None:
            freed += self.tips.put_back(v.tip, release)
        self.stats.evicted += 1
        return freed

    def _free_slot(self, protect: int) -> Optional[int]:
        k = self.arena.alloc()
        while k is None:
            leaves = self._leaves(protect)
            if not leaves:
                return None
            self._drop(min(leaves, key=lambda x: (x.stamp, -x.start)))
            k = self.arena.alloc()
        return k

    def _press(self, protect: int) -> None:
        """The host-memory floor: while MemAvailable is under ``reserve``, release free slots' pages, then evict LRU leaves
        (their pages released) until the deficit is covered, and trim the heap so the freed pages reach MemAvailable."""
        if self.reserve <= 0:
            return
        avail = self.available()
        if avail is None or avail >= self.reserve:
            return
        deficit, freed = self.reserve - int(avail), 0
        for a in (self.arena, self.tips):
            if a is not None and freed < deficit:
                freed += a.release_free(deficit - freed)
        while freed < deficit:
            leaves = self._leaves(protect)
            if not leaves:
                break
            freed += self._drop(min(leaves, key=lambda x: (x.stamp, -x.start)), release=True)
            self.stats.pressure_evictions += 1
        self.stats.released_bytes += freed
        self.trim()

    def _read(self, nbytes: int) -> None:
        self.read_since_trim += int(nbytes)
        if self.read_since_trim >= TRIM_EVERY_BYTES:
            self.trim()

    def trim(self) -> None:
        """Give the heap pages of freed transient read buffers back to the OS."""
        self.read_since_trim = 0
        if _LIBC is not None:
            _LIBC.malloc_trim(0)
            self.stats.trims += 1

    def summary(self) -> str:
        s = self.stats
        tips = sum(1 for e in self.entries.values() if e.tip is not None)
        return (
            f"store {len(self.entries)}/{self.capacity} chunk(s) + {tips}/{self.tips.n_slots if self.tips else 0} tip(s), "
            f"{self.bytes / 2**30:.2f} GiB in use, {self.resident_bytes / 2**30:.2f} GiB resident (budget "
            f"{self.budget / 2**30:.2f}); inserted {s.inserted}, evicted {s.evicted} (pressure {s.pressure_evictions}, "
            f"released {s.released_bytes / 2**30:.2f} GiB), refused {s.refused}, tips {s.tips_attached} (dropped "
            f"{s.tip_drops}, refused {s.tips_refused}), trims {s.trims}"
        )


# ----------------------------------------------------------------------------------------------------- H1: checkpoints
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
            return None  # H2: every slot is pinned for a spill
        if not self.free:
            held = sorted(self.by_pos)
            # the newest is protected while another can go (H2's spill-pinned slots can leave it the only one)
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
    carry_exports: int = 0
    marked_chunks: int = 0
    invalidations: int = 0
    # H2
    restored_chunks: int = 0
    restored_tokens: int = 0
    stored_chunks: int = 0
    spills: int = 0
    host_carries_loaded: int = 0


class PrefixRetention:
    """The retained prompt of ONE slot and the skip / resume decisions. Device actions are callbacks:

    ``save(slot)``                     live non-positional state -> checkpoint slot
    ``restore(slot)``                  checkpoint slot -> live state
    ``export_carries(slot, end, final) -> bool``  the state of checkpoint ``slot`` (None = the live one) into the export
                                       for a burst ending at ``end``; True = it also cleared the export past ``end``
    ``set_next_start(pos)``            the model's counters (and whatever is derived from them) at ``pos``
    ``compute_silent(ids, start)``     one chunk through the model, no export, no acks (the live state then holds its end)
    ``on_resume(start)``               (optional) after the live state is at a resume point ``start``: fix the export

    H2 (``store`` given):
    ``restore_rows(rows, ids, start)`` a stored chunk's rows -> the device's caches + the export's rows of [start, end)
    ``spill(slot) -> parts``           checkpoint slot -> host carries (host tensors)
    ``load_carries(parts)``            host carries -> the live state
    """

    def __init__(
        self,
        *,
        chunk: int,
        n_ckpts: int,
        save: Callable[[int], None],
        restore: Callable[[int], None],
        export_carries: Callable[[Optional[int], int, bool], bool],
        set_next_start: Callable[[int], None],
        compute_silent: Callable[[torch.Tensor, int], None],
        on_resume: Optional[Callable[[int], None]] = None,
        log: Callable[[str], None] = print,
        tag: str = "prefix",
        skip_partial_last: bool = True,
        store: Optional[ChunkStore] = None,
        restore_rows: Optional[Callable[[list, torch.Tensor, int], None]] = None,
        spill: Optional[Callable[[int], list]] = None,
        load_carries: Optional[Callable[[list], None]] = None,
        spill_tips: int = 2,
        chain_seed: bytes = DEFAULT_CHAIN_SEED,
    ):
        self.chunk = int(chunk)
        # False: a matching PARTIAL chunk (always a burst's last) is computed, resumed at its start, never skipped -- for an
        # export whose state at a partial end is more than one checkpoint can hold (V4.1's window ring, device check 11-11)
        self.skip_partial_last = bool(skip_partial_last)
        self.pool = CheckpointPool(n_ckpts, chunk)
        self._save, self._restore, self._export = save, restore, export_carries
        self._set_next_start, self._compute_silent, self._on_resume = set_next_start, compute_silent, on_resume
        self.log, self.tag = log, tag
        self.tokens = torch.zeros(0, dtype=torch.int32)
        self.length = 0  # positions [0, length) of the device caches / export hold tokens[:length]
        self.matching = False  # the in-flight prompt: every chunk so far skipped / restored (matching)
        self.cursor = 0  # ... and where its next chunk must start
        self.prompt_resumed = False
        self.last_resume: Optional[dict] = None  # set by a chunk that resumed a matched prompt (for the caller's log)
        self.last_restored = False  # the last begin_chunk restored its chunk from the store (H2)
        self.live_at: Optional[int] = None  # the live state is the device chain's state at this position
        self.stats = ReuseStats()
        # H2: the host store and the in-flight prompt's chain hashes (position -> hash of ids [0, position))
        self.store = store
        if store is not None:
            assert restore_rows is not None and spill is not None and load_carries is not None
        self._restore_rows, self._spill, self._load_carries = restore_rows, spill, load_carries
        self.spill_tips = int(spill_tips)
        self.chain_seed = bytes(chain_seed)
        self.stamp = 0
        self.p_pos = 0
        self.p_hash: Optional[bytes] = None
        self.p_hashes: dict[int, bytes] = {}
        self._cur: Optional[tuple] = None  # (start, end, parent hash, hash) of the chunk being processed
        self.pending_spills: list[tuple[bytes, int]] = []  # (chain hash, pinned checkpoint slot) read after the burst

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
        self.log(f"[{self.tag}] retention reset ({why})")

    def carries_at(self, pos: int):
        """("live", None) when the live state is at ``pos``, ("ckpt", slot) for a checkpoint, else None."""
        if self.live_at is not None and pos == self.live_at:
            return ("live", None)
        slot = self.pool.held(pos)
        return None if slot is None else ("ckpt", slot)

    # ------------------------------------------------------------------ per chunk
    def begin_chunk(self, ids: torch.Tensor, start: int, end: int, last: Optional[bool]) -> bool:
        """Called before a non-empty chunk ``[start, end)``; ``last`` from ``split_slot_word`` (None = unmarked header).
        True = skipped or (H2) restored (the caller only acknowledges it); False = compute it (the model is resumed at
        ``start`` first if the prompt matched so far)."""
        start, end = int(start), int(end)
        self.last_restored = False
        if last is not None:
            self.stats.marked_chunks += 1
        if start == 0:
            self._new_prompt()
        ids = ids.to(torch.int32).flatten()
        self._chain_step(ids, start, end)
        partial = (end - start) < self.chunk  # a partial (ragged) chunk is always the burst's last
        if (
            self.matching
            and start == self.cursor
            and end <= self.length
            and (self.skip_partial_last or not partial)
            and torch.equal(ids, self.tokens[start:end])
        ):
            final = bool(last) or partial
            may_be_last = final or last is None
            if not may_be_last:
                self._skipped(start, end)
                return True
            at = self.carries_at(end)
            if at is not None:
                cleared = self._export(at[1], end, final)
                self.stats.carry_exports += 1
                self._skipped(start, end)
                if cleared and end < self.length:
                    self._leaving(end, last)  # H2: the older prompt's tips past `end` go to the store first
                    self._truncate(end)  # the export past `end` is gone: nothing past it can be shipped from here
                return True
            # a possible last chunk whose end state is not held: compute it (resumed at start below)
        elif self.store is not None and self.matching and start == self.cursor and self._try_restore(ids, last):
            return True
        self._prepare_compute(start, last)
        return False

    def note_empty(self) -> None:
        """An EMPTY chunk (a burst declined before compute): nothing computed or skipped, but it ends a request."""
        self.matching = False

    def end_compute(self, ids: torch.Tensor, start: int, end: int) -> None:
        """After the model computed ``[start, end)`` (the live state is at ``end``): retain it."""
        self._record(ids, int(start), int(end))
        self.stats.computed_chunks += 1

    def invalidate(self, why: str) -> None:
        """A chunk failed part-way, or the prompt cannot be vouched for (an image span): nothing on the device is retained.
        The host store keeps its chunks (they are keyed by their ids, not by what the device holds)."""
        self.stats.invalidations += 1
        self.reset(why)

    # ------------------------------------------------------------------ H2: the host store
    def wants_rows(self, start: int, end: int) -> bool:
        """Should the caller read this computed chunk's rows into ``store_rows`` (a full chunk of a hashed chain the store
        does not hold yet)?"""
        c = self._cur
        return (
            self.store is not None
            and c is not None
            and c[0] == int(start)
            and c[1] == int(end)
            and c[3] is not None
            and self.store.get(c[3]) is None
        )

    def store_rows(self, ids: torch.Tensor, start: int, end: int, parts) -> bool:
        if not self.wants_rows(start, end):
            return False
        c = self._cur
        ok = self.store.put(c[3], c[2], int(start), int(end), ids.to(torch.int32).flatten(), parts, self.stamp)
        self.stats.stored_chunks += int(ok)
        return ok

    def flush_spills(self) -> int:
        """Read the pinned checkpoints of the prompts the device left into the store (after a burst's last chunk).
        Returns the number of checkpoints read."""
        pend, self.pending_spills = self.pending_spills, []
        n = 0
        for h, slot in pend:
            e = self.store.get(h) if self.store is not None else None
            if e is not None and not e.has_carries:
                parts = self._spill(slot)
                n += 1
                if self.store.attach_carries(h, parts, self.stamp):
                    self.stats.spills += 1
            self.pool.free.append(slot)
        return n

    def _chain_step(self, ids: torch.Tensor, start: int, end: int) -> None:
        """H2: the in-flight prompt's chain hash at ``end`` (full chunks only; a ragged chunk ends the chain)."""
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
        """H2: the device does not hold this chunk but the store does: restore it instead of computing it."""
        c = self._cur
        if c is None or c[3] is None:
            return False
        start, end, _parent, h = c
        ent = self.store.get(h)
        if ent is None or ent.start != start or ent.end != end or not torch.equal(ent.tokens, ids):
            return False
        final = bool(last)  # a stored chunk is always full: only the header says it is the last
        may_be_last = last is None or final
        if may_be_last and not ent.has_carries:
            return False  # its end carries would be needed and are not stored: compute it
        self._leaving(start, last)
        self._truncate(start)
        self._restore_rows(self.store.rows(ent), ids, start)
        self._record_rows(ids, start, end)
        if may_be_last:
            self._load_carries(self.store.carries(ent))
            self.stats.host_carries_loaded += 1
            self.live_at = end
            self._export(None, end, final)  # length == end: nothing past it to forget
            self.stats.carry_exports += 1
        self.stats.restored_chunks += 1
        self.stats.restored_tokens += end - start
        self.last_restored = True
        self._skipped(start, end)
        return True

    def _leaving(self, pos: int, last: Optional[bool]) -> None:
        """H2: the device chain is about to be overwritten from ``pos``: keep the carries of its newest boundaries above
        ``pos`` (the next turn of that conversation resumes there) by pinning their checkpoint slots for a spill into the
        store -- after the burst's last chunk on a marked header, at once on an unmarked one."""
        if self.store is None or pos >= self.length or self.spill_tips <= 0:
            return
        above = sorted((p for p in self.pool.by_pos if p > pos), reverse=True)[: self.spill_tips]
        if not above:
            return
        h = self.chain_seed
        hashes = {}
        for x in range(0, max(above), self.chunk):
            h = chain_hash(h, self.tokens[x : x + self.chunk])
            hashes[x + self.chunk] = h
        for p in above:
            ent = self.store.get(hashes[p])
            if ent is None or ent.has_carries:
                continue
            self.pending_spills.append((hashes[p], self.pool.by_pos.pop(p)))
        if last is None:
            self.flush_spills()

    def _host_carries(self, pos: int):
        """H2: stored carries of the in-flight prompt's chain at boundary ``pos``."""
        if self.store is None:
            return None
        return self.store.carries(self.store.get(self.p_hashes.get(pos)))

    # ------------------------------------------------------------------ internals
    def _new_prompt(self) -> None:
        self.stats.prompts += 1
        self.stamp += 1
        self.p_pos, self.p_hash, self.p_hashes = 0, self.chain_seed, {0: self.chain_seed}
        # H2: a prompt may match the store even when the device retains nothing
        self.matching = self.length > 0 or self.store is not None
        self.cursor = 0
        self.prompt_resumed = False

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
                self._truncate(0)  # a new prompt computed from scratch: only [0, 0) stays
            elif start != self.length:
                # not the continuation of the prompt the device holds (tt-d-gen #430's re-send): the model's own contiguity
                # check decides; nothing retained can be vouched for past this point
                self.reset(f"chunk start {start} is neither 0 nor the retained end {self.length}")
            return
        at = self.carries_at(start)
        host = self._host_carries(start) if at is None else None
        if host is not None:
            self._load_carries(host)
            self.stats.host_carries_loaded += 1
            self.live_at = start
            self._set_next_start(start)
            self.length = start
            self.last_resume = {"start": start, "how": "host carries"}
        elif at is not None and at[0] == "live":
            self._set_next_start(start)  # the live state is already at start (the retained prompt ended there)
            self.last_resume = {"start": start, "how": "live"}
        elif at is not None:
            self._restore(at[1])
            self.live_at = start
            self._set_next_start(start)
            self.length = start
            self.last_resume = {"start": start, "how": "checkpoint"}
        else:
            b, slot = self.pool.nearest_at_or_below(start)
            hb = max((p for p in self.p_hashes if b < p <= start and self._host_carries(p) is not None), default=None)
            if hb is not None:  # H2: a stored tip of this chain is nearer than any device checkpoint
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
            self.log(f"[{self.tag}] rebuilt the carries at {start} from {b} ({n} chunk(s), no export)")
        self._truncate(start)
        if self._on_resume is not None:
            self._on_resume(start)

    def _truncate(self, pos: int) -> None:
        """Everything from ``pos`` on is about to be overwritten (or was cleared)."""
        self.length = min(self.length, pos)
        if self.live_at is not None and self.live_at > pos:
            self.live_at = None
        self.pool.drop_above(pos)

    def _record_rows(self, ids: torch.Tensor, start: int, end: int) -> None:
        """The device chain now holds ``ids`` at [start, end) (rows written; the live state untouched)."""
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
            f"{s.computed_chunks}, checkpoints saved {s.checkpoints_saved} (evicted {self.pool.evictions}), carry exports "
            f"{s.carry_exports}, marked headers {s.marked_chunks}, invalidations {s.invalidations}"
            + (
                f"; restored {s.restored_chunks} chunk(s) / {s.restored_tokens} tokens, stored {s.stored_chunks}, spills "
                f"{s.spills}, host carries {s.host_carries_loaded}; {self.store.summary()}"
                if self.store is not None
                else ""
            )
        )
