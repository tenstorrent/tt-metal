# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Automatic prefix caching (APC) of the Gated-DeltaNet recurrent state.

vLLM's APC only manages the paged attention KV blocks. The 48 GDN layers carry a fixed-size recurrent state
(rec_state [1,Nv_tp,Dk,Dv] fp32 + the cross-chunk conv_carry [1,3,C] bf16) that is NOT paged, so resuming a prefill at
position `pos` needs the GDN state AT `pos`. GdnPrefixStateCache keeps a bounded LRU of such snapshots, keyed by
(pos, digest of tokens[:pos]), in device DRAM (device-to-device copies of the B=1 prefill scratch, no host round trip).
The model (Qwen36Model.prefill_paged_slots) decides WHERE to snapshot and where to resume; see the README.

conv_states (the decode conv window) need no snapshot: capture_state=True rebuilds all K of them from the last chunk's
conv_new_state at the end of every GDN layer's prefill.
"""

import bisect
import hashlib
import os
from collections import OrderedDict

import numpy as np
import torch
from loguru import logger

import ttnn


def alloc_zeros_like(t, mesh):
    """Fresh zero tensor with the shape / dtype / TILE layout / DRAM / replicated mesh mapping of the GDN state `t`
    (the reset_state allocation recipe: ReplicateTensorToMesh, per-device contents diverge once written)."""
    tdt = torch.float32 if t.dtype == ttnn.float32 else torch.bfloat16
    return ttnn.from_torch(
        torch.zeros(*[int(d) for d in t.shape], dtype=tdt),
        dtype=t.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _tokens_np(tokens):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.reshape(-1).cpu().numpy()
    return np.ascontiguousarray(np.asarray(tokens, dtype=np.int32).reshape(-1))


SPEC_PREFIX_ALIGN = 128


def spec_prefix_cache_enabled():
    """QWEN36_SPEC_PREFIX_CACHE (default 1): prefix caching for the MTP spec class; 0 restores the no-prefix behaviour."""
    return os.environ.get("QWEN36_SPEC_PREFIX_CACHE", "1") == "1"


def spec_resume_cap(start_pos, prompt_len, align=SPEC_PREFIX_ALIGN, min_tail=3):
    """Largest resume position s a spec prefill may use for a vLLM prefix hit of `start_pos` cached tokens.

    s is a multiple of `align`, s <= floor((start_pos-1)/align)*align and prompt_len - s >= min_tail. The extra -1 (vs O's
    floor(start_pos/align)): the drafter KV at slot s-1 fuses token s; if s sits exactly at the end of the cached span
    that slot holds the OLD request's token s (stale), so one whole block more is recomputed. 0 = no resume."""
    start_pos = int(start_pos)
    if start_pos <= 0:
        return 0
    s = ((start_pos - 1) // align) * align
    while s > 0 and prompt_len - s < min_tail:
        s -= align
    return max(s, 0)


def spec_learn_positions(start_pos, prompt_len, align=SPEC_PREFIX_ALIGN, min_tail=3):
    """Snapshot positions worth taking when vLLM holds the KV up to start_pos: the resume cap itself (what a request
    with the same start_pos can use) and floor(start_pos/align)*align (usable by longer hits), both <= prompt_len-min_tail.
    """
    out = set()
    h = (max(int(start_pos), 0) // align) * align
    while h > 0 and prompt_len - h < min_tail:
        h -= align
    if h > 0:
        out.add(h)
    c = spec_resume_cap(start_pos, prompt_len, align, min_tail)
    if c > 0:
        out.add(c)
    return out


def prefix_stride_low_prio():
    """QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO (default 1): pure stride snapshots are evicted first; 0 = pure LRU."""
    return os.environ.get("QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO", "1") == "1"


class SpecPoints(list):
    """Sorted save points; ``low`` = the subset that is a pure stride point (low cache priority)."""

    low = frozenset()


def spec_save_points(
    prompt_len, s, learn, chunk=2048, min_len=8192, stride=8192, align=SPEC_PREFIX_ALIGN, end_point=True
):
    """Sorted segment split positions (snapshot after tokens[:p]) for a spec prefill resuming at s, p in (s, L-3]:
    the learn positions, the end-of-prompt position for L >= min_len, else the last full-chunk boundary, and every
    `stride` tokens from s (chunk-aligned strides only). Same policy as the plain O path (model._plan_prefix_prefill).
    ``end_point`` False (a non-final chunked-prefill chunk resumed exactly, Option B): no end-of-prompt / last-full-chunk
    point (the caller snapshots the chunk end itself); learn positions and strides stay. The result is a SpecPoints whose
    ``low`` holds the pure stride points (in the stride range only: not a learn / end / last-chunk point).
    """
    pts = {p for p in learn if s < p <= prompt_len - 3}
    b_end = ((prompt_len - 3) // align) * align
    if not end_point:
        pass
    elif prompt_len >= min_len:
        if b_end > max(s, max(pts, default=0)):
            pts.add(b_end)
    elif not pts:
        p = s + ((prompt_len - s) // chunk) * chunk
        if s < p <= prompt_len - 3:
            pts.add(p)
    out = SpecPoints()
    if stride > 0 and stride % chunk == 0:
        strided = set(range(s + stride, prompt_len - 2, stride))
        out.low = frozenset(strided - pts)
        pts.update(strided)
    out.extend(sorted(pts))
    return out


class GdnPrefixStateCache:
    """LRU cache of GDN (rec_state, conv_carry) snapshots at 128-aligned prompt positions.

    Allocates `num_slots` snapshot sets (one rec/conv_carry pair per GDN layer) at construction, i.e. BEFORE any
    prefill/decode trace is captured (allocating behind captured traces is unsafe). save/restore copy between a slot and
    the GDN state currently bound on the layers (the persistent B=1 prefill scratch inside prefill_paged_slots)."""

    def __init__(self, model, num_slots=None, name="GDN prefix-state cache"):
        self.model = model
        self.name = name
        self.num_slots = int(num_slots if num_slots is not None else os.environ.get("QWEN36_PREFIX_CACHE_SLOTS", "32"))
        assert self.num_slots >= 1
        self.debug = os.environ.get("QWEN36_PREFIX_CACHE_DEBUG", "0") == "1"
        # The B=1 scratch (and its init buffers) must exist first: the slots mirror its shapes.
        model._ensure_gdn_prefill_scratch()
        self._dns = [dn for dn, *_ in model._gdn_prefill_scratch]
        self._slots = []  # slot -> [(rec, conv_carry)] per GDN layer
        for _ in range(self.num_slots):
            self._slots.append([self._alloc_pair(dn) for dn in self._dns])
        self._free = list(range(self.num_slots))
        self._lru = OrderedDict()  # (pos, digest) -> slot, least recently used first
        self._by_pos = {}  # pos -> {digest, ...}
        self._positions = []  # sorted positions present
        self.stats = dict(lookups=0, hits=0, tokens_skipped=0, saves=0, evictions=0, evictions_low=0)
        logger.info(f"{self.name}: {self.num_slots} slots x {len(self._dns)} GDN layers allocated.")

    def _alloc_pair(self, dn):
        scratch = next(s for s in self.model._gdn_prefill_scratch if s[0] is dn)
        return alloc_zeros_like(scratch[1], dn.mesh), alloc_zeros_like(scratch[3], dn.mesh)

    @staticmethod
    def _digest(buf, pos):
        return hashlib.blake2b(buf[:pos].tobytes(), digest_size=16).digest()

    def lookup(self, tokens, max_pos):
        """Largest stored pos <= max_pos whose digest matches tokens[:pos] -> (pos, slot), else (0, None)."""
        self.stats["lookups"] += 1
        buf = _tokens_np(tokens)
        cands = self._positions[: bisect.bisect_right(self._positions, min(int(max_pos), buf.shape[0]))]
        # One running hash over the ascending candidates (copy() at each) == blake2b(tokens[:pos]) per candidate.
        hasher = hashlib.blake2b(digest_size=16)
        found, prev = {}, 0
        for pos in cands:
            hasher.update(buf[prev:pos].tobytes())
            prev = pos
            dig = hasher.copy().digest()
            if dig in self._by_pos[pos]:
                found[pos] = dig
        if not found:
            return 0, None
        pos = max(found)
        key = (pos, found[pos])
        self._lru.move_to_end(key)
        self._low().discard(key)  # a hit proves value
        self.stats["hits"] += 1
        self.stats["tokens_skipped"] += pos
        return pos, self._lru[key]

    # --- pinning (MTP chunked prefill Option B): a chunk-end snapshot is kept until its request's next chunk consumed it ---
    _STAT_KEYS = ("pinned_saves", "skipped_saves", "exact_resumes", "refusals")

    def _pins(self):
        """owner (engine row) -> key of its pinned snapshot, insertion ordered (lazy: caches built without __init__ work)."""
        return self.__dict__.setdefault("_pin_map", OrderedDict())

    def _low(self):
        """Keys saved at low priority (pure stride snapshots, evicted first), lazy like _pins."""
        return self.__dict__.setdefault("_low_keys", set())

    def bump(self, name, n=1):
        self.stats[name] = self.stats.get(name, 0) + n

    def max_pinned_limit(self):
        """Pinned entries never exceed this (the engine sets it to its row count B: one live chunk per row); at most
        num_slots-1 so one slot stays evictable."""
        lim = self.__dict__.get("max_pinned", None)
        return max(1, min(int(lim) if lim is not None else self.num_slots - 1, self.num_slots - 1))

    def pin(self, tokens, pos, owner):
        """Pin the snapshot (pos, digest(tokens[:pos])) for ``owner``; replaces the owner's earlier pin. Returns False when the
        snapshot is not in the cache. Over the cap the oldest pin of another owner is dropped (logged)."""
        key = (int(pos), self._digest(_tokens_np(tokens), int(pos)))
        if key not in self._lru:
            return False
        pins = self._pins()
        pins.pop(owner, None)
        while len(pins) >= self.max_pinned_limit():
            old_owner, old_key = next(iter(pins.items()))
            del pins[old_owner]
            logger.warning(
                f"{self.name}: pin cap {self.max_pinned_limit()} reached: unpinned the snapshot at {old_key[0]} of row {old_owner}"
            )
        pins[owner] = key
        self.bump("pinned_saves")
        return True

    def unpin(self, owner):
        """Unpin ``owner``'s snapshot (consumed by its next chunk, or the request left). Returns True if one was pinned."""
        return self._pins().pop(owner, None) is not None

    def remap_owners(self, new_of):
        """Device slot remap: owner row u becomes new_of[u]."""
        pins = self._pins()
        moved = [(new_of[o], k) for o, k in pins.items()]
        pins.clear()
        pins.update(moved)

    def pinned_count(self):
        return len(self._pins())

    def summary(self):
        """Compact counters line (INFO, once per 1000 prefills from the engine)."""
        st = self.stats
        return (
            f"pinned={self.pinned_count()}/{self.max_pinned_limit()} pinned_saves={st.get('pinned_saves', 0)} "
            f"skipped_saves={st.get('skipped_saves', 0)} evictions={st.get('evictions', 0)} "
            f"evictions_low={st.get('evictions_low', 0)} low={len(self._low())} "
            f"exact_resumes={st.get('exact_resumes', 0)} refusals={st.get('refusals', 0)} "
            f"saves={st.get('saves', 0)} entries={len(self._lru)}/{self.num_slots}"
        )

    def _evict_victim(self):
        """Least recently used UNPINNED low-priority key (QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO), else the LRU UNPINNED
        key, or None."""
        pinned = set(self._pins().values())
        low = self._low()
        if low and prefix_stride_low_prio():
            for k in self._lru:
                if k in low and k not in pinned:
                    return k
        for k in self._lru:
            if k not in pinned:
                return k
        return None

    def save(self, tokens, pos, low_prio=False):
        """Snapshot the currently bound GDN state (the state AFTER tokens[:pos]) into a slot. ``low_prio``: a new key is
        evicted before normal ones (an existing normal key is never downgraded; a normal save upgrades a low key). Eviction
        skips pinned snapshots; when no slot is free and none is evictable the save is skipped (counted, debug log).
        Returns True when the snapshot is in the cache afterwards."""
        buf = _tokens_np(tokens)
        key = (int(pos), self._digest(buf, int(pos)))
        if key in self._lru:
            self._lru.move_to_end(key)
            if not low_prio:
                self._low().discard(key)
            return True
        if self._free:
            slot = self._free.pop()
        else:
            old_key = self._evict_victim()
            if old_key is None:
                self.bump("skipped_saves")
                logger.debug(f"{self.name}: every slot is pinned: save at {key[0]} skipped")
                return False
            slot = self._lru.pop(old_key)
            self._drop_pos(old_key)
            if old_key in self._low():
                self._low().discard(old_key)
                self.bump("evictions_low")
            self.stats["evictions"] += 1
        for dn, (rec, carry) in zip(self._dns, self._slots[slot]):
            ttnn.copy(dn.rec_state, rec)
            ttnn.copy(dn.conv_carry, carry)
        self._lru[key] = slot
        if low_prio:
            self._low().add(key)
        if key[0] not in self._by_pos:
            self._by_pos[key[0]] = set()
            bisect.insort(self._positions, key[0])
        self._by_pos[key[0]].add(key[1])
        self.stats["saves"] += 1
        return True

    def _drop_pos(self, key):
        pos, dig = key
        self._by_pos[pos].discard(dig)
        if not self._by_pos[pos]:
            del self._by_pos[pos]
            self._positions.pop(bisect.bisect_left(self._positions, pos))

    def restore(self, slot):
        """Copy a slot's state into the currently bound GDN state (rec_state + conv_carry of every GDN layer)."""
        for dn, (rec, carry) in zip(self._dns, self._slots[slot]):
            ttnn.copy(rec, dn.rec_state)
            ttnn.copy(carry, dn.conv_carry)

    def log(self, prompt_len, start_pos, hit_pos, saved):
        if self.debug:
            logger.info(
                f"[prefix-cache] len={prompt_len} vllm_start={start_pos} gdn_hit={hit_pos} saved={saved} "
                f"entries={len(self._lru)}/{self.num_slots} stats={self.stats}"
            )
