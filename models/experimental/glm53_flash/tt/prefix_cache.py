# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix cache of GLM-5.3-Flash's per-request state (Stage 9 srv/prefixcache; doc/ttnn/09-vllm/srv/prefixcache/README.md).

vLLM's automatic prefix caching shares the MLA/DSA latent PAGES of a common prefix (its block pool hashes 64-token
blocks and refcounts them). This model also carries state that is not paged: every KDA layer's recurrent state (S, conv
rows) and, in the sparse regime, every DSA layer's pooled indexer keys and ring -- one copy per state SLOT, overwritten
as the request advances. A prefix hit therefore needs a SNAPSHOT of that state taken exactly at the hit's boundary,
copied into the attaching request's slot. This module is the pure-python bookkeeping of those snapshots (host-mockable:
no device, no torch):

  key        = chained sha256 over BOUNDARY-token blocks of the prefix's token ids (vLLM's own scheme at 128 instead of
               64) salted with the serving signature (model / image / code / knob set): a hit never serves another
               model's or image's state.
  entry      = one snapshot: n_tokens (the boundary = the resume position), the page ids of the prefix (shared
               read-only through the page table; vLLM's pool owns them in serving, the store pins them in the
               generator-owned path), where it lives (device tier: a snapshot pool entry = a virtual slot
               max_batch + k; host tier: the backend's export blob), bytes, LRU tick, readers.
  register   : after a prefill chunk ending on a boundary (the snapshot was written inside the prefill trace) or when a
               finished request's trailing decode snapshot gets its key.
  attach     : the longest cached boundary <= the new prompt's hit -> the entry to restore from and the resume
               position; the suffix is prefilled as a continuation segment.

Two snapshot classes: P (prefill-boundary: bitwise the state an uncached prefill of the same tokens leaves) and G
(generation: the state of the live request as it crossed a boundary while decoding; bitwise the continuation of that
generation, PCC-equal to a fresh prefill of the same tokens). Multi-turn chat resumes from G.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

BOUNDARY = 128  # snapshot / resume granularity: the TP prefill's continuation alignment (SDPA q chunk)
PAGE = 64  # latent page (vLLM block)
DEVICE_ENTRIES_ENV = "GLM53_PREFIX_DEVICE_ENTRIES"  # device snapshot pool entries (0: no device tier)
HOST_BYTES_ENV = "GLM53_PREFIX_HOST_BYTES"  # host tier budget in bytes (0: no host tier)
ENABLED_ENV = "GLM53_PREFIX_CACHE"  # "1" (default): the adapter declares supports_prefix_caching and runs the store


def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "1") == "1"


# ------------------------------------------------------------------------------------------------- bytes model
@dataclass(frozen=True)
class SnapshotBytes:
    """What one snapshot holds, from the model's state shapes (StackBackend: S [64, 128, 128] fp32 and conv C
    [3, 24576] bf16 per KDA layer, sharded over the TP mesh; DSA pooled-key store pages of 32 pools x 128 bf16 per 128
    tokens and a ring of 8 tiles per DSA layer, replicated per chip; latent pages [64, 512] bf16 per MLA layer,
    replicated per chip, shared through the page table and NOT part of the snapshot)."""

    kda_layers: int = 34
    dsa_layers: int = 11
    tp: int = 8
    s_bytes: int = 64 * 128 * 128 * 4
    c_bytes: int = 3 * 24576 * 2
    pool_page_bytes: int = 32 * 128 * 2
    ring_bytes: int = 8 * 2048
    latent_page_bytes: int = 64 * 512 * 2

    @property
    def kda_total(self) -> int:
        """KDA state of one slot, all chips together"""
        return self.kda_layers * (self.s_bytes + self.c_bytes)

    @property
    def kda_per_chip(self) -> int:
        return -(-self.kda_total // self.tp)

    def dsa_per_chip(self, context: int, sparse: bool) -> int:
        """DSA pooled keys + ring of one slot (only the sparse regime, context > 2,051, keeps them)"""
        if not sparse:
            return 0
        pages = -(-int(context) // BOUNDARY)
        return self.dsa_layers * (pages * self.pool_page_bytes + self.ring_bytes)

    def entry_per_chip(self, context: int = 2048, sparse: bool = False) -> int:
        return self.kda_per_chip + self.dsa_per_chip(context, sparse)

    def pages_per_chip(self, n_tokens: int) -> int:
        """the shared latent pages of an n-token prefix (vLLM-owned, counted for the budget only)"""
        return -(-int(n_tokens) // PAGE) * self.dsa_layers * self.latent_page_bytes


# ---------------------------------------------------------------------------------------------------- hashing
def signature(**facts) -> str:
    """The serving signature a key is salted with: whatever identifies the numerics of the state (model name, image pin,
    code sha, knob set, TP). Different signature -> different keys -> no cross-model / cross-image hits."""
    h = hashlib.sha256()
    for k in sorted(facts):
        h.update(f"{k}={facts[k]}\0".encode())
    return h.hexdigest()[:16]


def prefix_hashes(token_ids: Sequence[int], sig: str, boundary: int = BOUNDARY) -> list[str]:
    """Chained block hashes: hashes[j] = key of the first (j + 1) * boundary tokens (only complete blocks)."""
    out, prev = [], sig.encode()
    n = len(token_ids) // boundary
    for j in range(n):
        blk = token_ids[j * boundary : (j + 1) * boundary]
        h = hashlib.sha256(prev)
        h.update(",".join(str(int(t)) for t in blk).encode())
        prev = h.digest()
        out.append(prev.hex())
    return out


def prefix_key(token_ids: Sequence[int], n_tokens: int, sig: str, boundary: int = BOUNDARY) -> str:
    if n_tokens <= 0 or n_tokens % boundary or n_tokens > len(token_ids):
        raise ValueError(f"prefix key needs a positive multiple of {boundary} tokens within the prompt ({n_tokens})")
    return prefix_hashes(token_ids[:n_tokens], sig, boundary)[-1]


# ------------------------------------------------------------------------------------------------------ store
@dataclass
class Entry:
    key: str
    n_tokens: int  # the boundary: a request attaching here resumes at position n_tokens
    pages: tuple  # latent page ids of tokens [0, n_tokens) (the page-table row prefix), shared read-only
    kind: str  # "P" (prefill boundary) or "G" (generation boundary)
    tier: str  # "device" | "host"
    device_entry: Optional[int] = None  # snapshot pool entry (virtual slot max_batch + entry) when tier == "device"
    host_blob: object = None  # the backend's export blob when tier == "host"
    bytes: int = 0
    readers: int = 0  # live slots attached to it (an entry with readers is never evicted from the device tier)
    hits: int = 0
    tick: int = 0  # LRU
    created: float = field(default_factory=time.time)


class PrefixStore:
    """Snapshot registry + device/host tiers + page pins. The backend work (copies) is the caller's; this records what
    exists where and decides evictions. All methods are host-side bookkeeping only."""

    def __init__(self, *, max_batch: int, device_entries: int, host_bytes: int = 0, sig: str = "", entry_bytes: int = 0,
                 boundary: int = BOUNDARY):  # fmt: skip
        self.max_batch, self.boundary, self.sig = int(max_batch), int(boundary), sig
        self.device_entries = int(device_entries)
        self.host_bytes = int(host_bytes)
        self.entry_bytes = int(entry_bytes)
        self.entries: OrderedDict[str, Entry] = OrderedDict()  # key -> entry (insertion order: a tick refreshes)
        self.free_device: list[int] = list(range(self.device_entries))
        self.device_used: dict[int, str] = {}  # device entry -> key (or a trailing slot tag "slot:<g>")
        self.pins: dict[int, int] = {}  # page id -> number of entries holding it
        self.attached: dict[int, str] = {}  # slot -> key it reads (readers accounting)
        self.trailing: dict[int, tuple[int, int]] = {}  # slot -> (device entry, position) of its decode snapshot
        self.host_used = 0
        self._tick = 0
        self.stats = {k: 0 for k in ("registered", "hits", "misses", "attached", "evicted_device", "evicted_host",
                                     "spilled", "invalidated", "trailing_snapshots", "trailing_committed",
                                     "tokens_saved", "tokens_prefilled")}  # fmt: skip

    # ---- keys
    def key(self, token_ids: Sequence[int], n_tokens: int) -> str:
        return prefix_key(token_ids, n_tokens, self.sig, self.boundary)

    def hashes(self, token_ids: Sequence[int]) -> list[str]:
        return prefix_hashes(token_ids, self.sig, self.boundary)

    def boundaries(self, n_tokens: int) -> list[int]:
        """the snapshot positions of an n-token prompt: every boundary below it (a prompt that is itself a multiple of
        the boundary snapshots at n - boundary: its last token must be prefilled to produce logits)"""
        return list(range(self.boundary, n_tokens, self.boundary))

    def last_boundary(self, n_tokens: int) -> int:
        """the boundary a prompt's prefill ends on before its tail (0: none)"""
        return max(n_tokens - 1, 0) // self.boundary * self.boundary

    # ---- lookup
    def longest_cached(self, token_ids: Sequence[int], max_tokens: Optional[int] = None) -> tuple[int, Optional[Entry]]:
        """(resume position, entry) of the longest cached boundary <= max_tokens (default: len - 1) of the prompt;
        (0, None) when nothing matches. Partial matches count: the suffix is prefilled from the resume position."""
        n = len(token_ids) - 1 if max_tokens is None else min(int(max_tokens), len(token_ids) - 1)
        n = max(n, 0) // self.boundary * self.boundary
        hs = self.hashes(token_ids[:n])
        for j in range(len(hs) - 1, -1, -1):
            e = self.entries.get(hs[j])
            if e is not None:
                return (j + 1) * self.boundary, e
        return 0, None

    def has(self, key: str) -> bool:
        return key in self.entries

    # ---- device tier
    def reserve_device(self) -> Optional[int]:
        """A free device entry, evicting the LRU unreferenced one if needed (None: no device tier / all busy)."""
        if self.device_entries == 0:
            return None
        if self.free_device:
            return self.free_device.pop(0)
        victim = next((e for e in self.entries.values() if e.tier == "device" and e.readers == 0), None)
        if victim is None:
            return None
        if not self._spill_victim(victim):
            self.evict(victim.key)
        return self.free_device.pop(0)

    #: host-tier spill hook (Stage 12 host KV tier): entry -> (host blob, bytes), or None to refuse (the entry is
    #: then dropped as before). The adapter exports the entry's virtual slot (backend.export_snapshot).
    spill_fn: Optional[Callable[["Entry"], Optional[tuple]]] = None

    def _spill_victim(self, victim: "Entry") -> bool:
        """With a host tier and a spill hook: move the LRU device entry to the host tier instead of dropping it."""
        if self.host_bytes <= 0 or self.spill_fn is None:
            return False
        try:
            res = self.spill_fn(victim)
        except Exception:  # noqa: BLE001  (an export failure must not fail the request: drop as before)
            self.stats["spill_failed"] = self.stats.get("spill_failed", 0) + 1
            return False
        if res is None:
            self.stats["spill_refused"] = self.stats.get("spill_refused", 0) + 1
            return False
        blob, nbytes = res
        try:
            self.spill(victim.key, blob, nbytes)
        except ValueError:  # the host tier cannot take it (too small / full of entries with readers)
            self.stats["spill_failed"] = self.stats.get("spill_failed", 0) + 1
            return False
        return True

    def unreserve(self, entry: int) -> None:
        """a reserved device entry that was never written (the prefill was refused)"""
        if entry not in self.device_used and entry not in self.free_device and 0 <= entry < self.device_entries:
            self.free_device.append(int(entry))

    def register(self, key: str, *, n_tokens: int, pages: Iterable[int], kind: str, device_entry: Optional[int] = None,
                 host_blob=None, bytes: Optional[int] = None) -> Entry:  # fmt: skip
        """Record a snapshot that now exists (device entry written, or a host blob). Re-registering a key replaces the
        older snapshot of the same prefix (same tokens -> same state up to the P/G class; the newer one stays)."""
        if n_tokens <= 0 or n_tokens % self.boundary:
            raise ValueError(f"a snapshot sits on a multiple of {self.boundary} tokens, not {n_tokens}")
        pages = tuple(int(p) for p in pages)
        if len(pages) < -(-n_tokens // PAGE):
            raise ValueError(f"{len(pages)} pages cannot cover {n_tokens} tokens")
        pages = pages[: -(-n_tokens // PAGE)]
        if key in self.entries:
            self.evict(key)
        if device_entry is not None and host_blob is not None:
            raise ValueError("one tier per entry")
        if device_entry is None and host_blob is None:
            raise ValueError("a snapshot lives on the device or on the host")
        e = Entry(key=key, n_tokens=int(n_tokens), pages=pages, kind=kind,
                  tier="device" if device_entry is not None else "host", device_entry=device_entry,
                  host_blob=host_blob, bytes=self.entry_bytes if bytes is None else int(bytes))  # fmt: skip
        if e.tier == "device":
            if device_entry in self.device_used or not 0 <= device_entry < self.device_entries:
                raise ValueError(f"device entry {device_entry} is not reserved for this snapshot")
            self.device_used[device_entry] = key
        else:
            self._host_admit(e.bytes)
            self.host_used += e.bytes
        for p in pages:
            self.pins[p] = self.pins.get(p, 0) + 1
        self._tick += 1
        e.tick = self._tick
        self.entries[key] = e
        self.stats["registered"] += 1
        return e

    def _host_admit(self, need: int) -> None:
        if self.host_bytes <= 0:
            raise ValueError("no host tier")
        while self.host_used + need > self.host_bytes:
            victim = next((e for e in self.entries.values() if e.tier == "host" and e.readers == 0), None)
            if victim is None:
                raise ValueError("host tier full of entries with readers")
            self.evict(victim.key)

    def attach(self, key: str, slot: int) -> tuple[int, Entry]:
        """Slot `slot` (a fresh request) will take the entry's state: returns (resume position, entry). The caller
        copies the state (device entry -> slot inside the prefill trace, or imports the host blob) and maps the pages.
        """
        e = self.entries.get(key)
        if e is None:
            self.stats["misses"] += 1
            raise KeyError(key)
        if slot in self.attached:
            self.detach(slot)
        e.readers += 1
        e.hits += 1
        self._tick += 1
        e.tick = self._tick
        self.entries.move_to_end(key)
        self.attached[slot] = key
        self.stats["hits"] += 1
        self.stats["attached"] += 1
        self.stats["tokens_saved"] += e.n_tokens
        return e.n_tokens, e

    def detach(self, slot: int) -> None:
        key = self.attached.pop(slot, None)
        if key is not None and key in self.entries:
            self.entries[key].readers -= 1

    def evict(self, key: str) -> Entry:
        e = self.entries.pop(key)
        if e.readers:
            # a reader keeps the entry's pages and tier bytes accounted until it detaches; the registry forgets it
            for s, k in list(self.attached.items()):
                if k == key:
                    self.attached.pop(s)
        if e.tier == "device":
            self.device_used.pop(e.device_entry, None)
            self.free_device.append(e.device_entry)
            self.stats["evicted_device"] += 1
        else:
            self.host_used -= e.bytes
            self.stats["evicted_host"] += 1
        for p in e.pages:
            n = self.pins.get(p, 0) - 1
            if n <= 0:
                self.pins.pop(p, None)
            else:
                self.pins[p] = n
        return e

    def spill(self, key: str, host_blob, bytes: Optional[int] = None) -> Entry:
        """Move a device entry to the host tier (the caller exported the blob first)."""
        e = self.entries[key]
        if e.tier != "device":
            raise ValueError(f"{key} is not on the device")
        if e.readers:
            raise ValueError(f"{key} has {e.readers} readers")
        need = self.entry_bytes if bytes is None else int(bytes)
        if need > self.host_bytes:
            raise ValueError(f"entry of {need} bytes exceeds the host tier ({self.host_bytes})")
        self._host_admit(need)  # first: a refusal leaves the entry on the device, untouched
        self.device_used.pop(e.device_entry, None)
        self.free_device.append(e.device_entry)
        e.tier, e.device_entry, e.host_blob = "host", None, host_blob
        e.bytes = need
        self.host_used += e.bytes
        self.stats["spilled"] += 1
        return e

    def promote(self, key: str, device_entry: int) -> Entry:
        """Move a host entry back to the device (the caller imported the blob into the entry's virtual slot)."""
        e = self.entries[key]
        if e.tier != "host":
            raise ValueError(f"{key} is not on the host")
        self.host_used -= e.bytes
        e.tier, e.host_blob, e.device_entry = "device", None, int(device_entry)
        self.device_used[e.device_entry] = key
        return e

    def spill_candidates(self) -> list[Entry]:
        """device entries without readers, LRU first (what to export to the host tier when the device tier is tight)"""
        return [e for e in self.entries.values() if e.tier == "device" and e.readers == 0]

    def invalidate(self, sig: Optional[str] = None) -> int:
        """Drop every entry (model / image / knob change: new signature). Trailing snapshots of live slots stay (they
        are re-keyed with the new signature at release)."""
        n = len(self.entries)
        for key in list(self.entries):
            self.evict(key)
        if sig is not None:
            self.sig = sig
        self.stats["invalidated"] += n
        return n

    # ---- trailing (generation) snapshots: one device entry per live slot, overwritten at every boundary crossing
    def trailing_entry(self, slot: int, position: int) -> Optional[int]:
        """The device entry the slot's decode snapshot at `position` goes to (its existing one, else a reserved one;
        None: no device entry available). Records the position."""
        if position <= 0 or position % self.boundary:
            raise ValueError(f"a trailing snapshot sits on a multiple of {self.boundary} tokens, not {position}")
        cur = self.trailing.get(slot)
        if cur is not None:
            self.trailing[slot] = (cur[0], int(position))
            self.stats["trailing_snapshots"] += 1
            return cur[0]
        ent = self.reserve_device()
        if ent is None:
            return None
        self.device_used[ent] = f"slot:{slot}"
        self.trailing[slot] = (ent, int(position))
        self.stats["trailing_snapshots"] += 1
        return ent

    def trailing_position(self, slot: int) -> int:
        cur = self.trailing.get(slot)
        return cur[1] if cur else 0

    def commit_trailing(self, slot: int, token_ids: Sequence[int], pages: Iterable[int]) -> Optional[Entry]:
        """The request in `slot` finished: its trailing snapshot (at position p) becomes the G entry keyed by its first p
        tokens. Returns the entry (None: the slot had no trailing snapshot)."""
        cur = self.trailing.pop(slot, None)
        if cur is None:
            return None
        ent, pos = cur
        self.device_used.pop(ent, None)
        if pos > len(token_ids):
            self.free_device.append(ent)
            return None
        key = self.key(token_ids, pos)
        e = self.register(key, n_tokens=pos, pages=pages, kind="G", device_entry=ent)
        self.stats["trailing_committed"] += 1
        return e

    def drop_trailing(self, slot: int) -> None:
        cur = self.trailing.pop(slot, None)
        if cur is not None:
            self.device_used.pop(cur[0], None)
            self.free_device.append(cur[0])

    def move_slot(self, src: int, dst: int) -> None:
        """the generator moved a live request between slots (compaction): its attachments follow"""
        if src in self.attached:
            self.attached[dst] = self.attached.pop(src)
        if src in self.trailing:
            self.trailing[dst] = self.trailing.pop(src)

    def release_slot(self, slot: int) -> None:
        """the slot stops holding a live request (finished / preempted): detach; a trailing snapshot not committed by
        then is dropped (commit_trailing first when the token ids are known)"""
        self.detach(slot)
        self.drop_trailing(slot)

    # ---- pages
    def pinned(self, page: int) -> bool:
        """a page some entry's prefix reads: the generator-owned path must not reuse it for another request"""
        return self.pins.get(int(page), 0) > 0

    # ---- reporting
    def report(self) -> dict:
        return dict(entries=len(self.entries), device_used=len(self.device_used), device_entries=self.device_entries,
                    host_used=self.host_used, host_bytes=self.host_bytes, pinned_pages=len(self.pins),
                    trailing=len(self.trailing), attached_slots=len(self.attached), **self.stats)  # fmt: skip


# ----------------------------------------------------------------------------------------------- TTFT ladder
@dataclass(frozen=True)
class PrefillCosts:
    """Measured prefill replay times (ms) by bucket on the served TP8 construction (2026-10-22: 2,048-token replay 441
    ms; 32K = 16 chunked 2,048 replays 4.69 s; the 128 / 512 buckets 202 / 259 ms, DECISIONS 03:40Z int25 MR), the
    host hand-off per chunk (22 ms, 03:40Z) and the state attach / snapshot costs (estimates until measured: the
    in-trace state read adds ~1.5 ms per replay; a host-dispatched SlotOps copy ~50 ms; a host-tier import ~450 ms)."""

    bucket_ms: tuple = ((128, 202.0), (512, 259.0), (2048, 441.0))
    chunk_tokens: int = 2048
    host_ms_per_chunk: float = 22.0
    attach_in_trace_ms: float = 1.5
    attach_dma_ms: float = 50.0
    attach_host_tier_ms: float = 450.0

    def replay_ms(self, n_tokens: int) -> float:
        """the replays of an n-token segment: whole chunks + the bucket of the tail"""
        full, tail = divmod(int(n_tokens), self.chunk_tokens)
        ms = full * dict(self.bucket_ms)[self.chunk_tokens]
        if tail:
            ms += next(v for b, v in self.bucket_ms if tail <= b)
        return ms

    def chunks(self, n_tokens: int) -> int:
        return -(-int(n_tokens) // self.chunk_tokens)

    def ttft_ms(self, n_tokens: int, cached: int = 0, attach: str = "in_trace") -> float:
        suffix = int(n_tokens) - int(cached)
        if suffix <= 0:
            raise ValueError("at least the last token is prefilled")
        ms = self.replay_ms(suffix) + self.chunks(suffix) * self.host_ms_per_chunk
        if cached:
            ms += {"in_trace": self.attach_in_trace_ms, "dma": self.attach_dma_ms, "host": self.attach_host_tier_ms}[
                attach
            ]
        return ms


def ttft_ladder(contexts=(128, 2048, 32768), costs: PrefillCosts = PrefillCosts(), turn_tail: int = 96) -> list[dict]:
    """The README's table: cold prefill of the context, cached prefix + 128-token suffix (a shared system prompt / a
    repeated prompt whose last boundary is cached), full-turn resume (the previous turn's trailing G snapshot plus the
    unaligned tail of the generation and the new question: `turn_tail` tokens past the boundary)."""
    rows = []
    for n in contexts:
        cold = costs.ttft_ms(n)
        b = max(n - 1, 0) // BOUNDARY * BOUNDARY
        shared = costs.ttft_ms(n, cached=b) if b else cold
        resume = costs.ttft_ms(n + turn_tail, cached=n // BOUNDARY * BOUNDARY)
        rows.append(dict(context=n, cold_ms=cold, cached_prefix=b, cached_prefix_suffix_ms=shared,
                         full_turn_resume_ms=resume, resume_suffix=n + turn_tail - n // BOUNDARY * BOUNDARY))  # fmt: skip
    return rows
