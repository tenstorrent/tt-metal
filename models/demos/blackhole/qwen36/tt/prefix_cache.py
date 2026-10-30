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

Opt-in policy knobs (all default OFF = the plain LRU + low-priority strides above):
  * QWEN36_PREFIX_CACHE_VALUE_EVICT=1: value-aware eviction. End-of-prompt checkpoints and snapshots hit
    >= QWEN36_PREFIX_CACHE_PROTECT_HITS times are evicted only after every other unpinned entry (at most
    QWEN36_PREFIX_CACHE_PROTECT_FRAC of the slots are protected; older protected entries beyond that count as normal).
  * QWEN36_PREFIX_HOST_SLOTS=N > 0: a host-RAM spill tier. An evicted snapshot (pos >= QWEN36_PREFIX_HOST_SPILL_MIN_POS,
    not a pure stride point unless QWEN36_PREFIX_HOST_SPILL_LOW=1) is read to host (ttnn.from_device: the exact device
    bytes, every device of the mesh) before its slot is reused; a later hit writes it back (copy_host_to_device_tensor
    into one staging slot allocated at construction, then the same device copy as a device hit). A restore is a byte
    copy, so a host hit resumes from exactly the state a device hit would have. Host LRU of N entries.
  * QWEN36_PREFIX_END_AT_CAP=1: the spec end-of-prompt checkpoint sits at the resume cap of the next request that
    extends the prompt (spec_followup_cap) instead of floor((L-3)/128)*128, which is one block above that cap whenever
    L mod 128 is in [3, 63]. Never more save points than today, so no extra prefill split.
  * QWEN36_PREFIX_CACHE_FAST_ALLOC=1: the slots (and the host-tier staging set) are allocated on device, uninitialized,
    instead of from uploaded host zeros (25 ms per slot at TP8, i.e. 6.5 s of start-up for 256 slots; 1.3 ms on
    device). Slot contents are never read before a save writes them.
"""

import bisect
import hashlib
import os
import time
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


def prefix_cache_fast_alloc():
    """QWEN36_PREFIX_CACHE_FAST_ALLOC (default "0" = today's slots: host zeros built and uploaded tensor by tensor,
    25 ms per slot at TP8 on the Galaxy, 6.5 s for 256 slots): "1" allocates the snapshot slots and the host-tier
    staging set directly on device, uninitialized (alloc_empty_like; 1.3 ms per slot). Safe because a slot is only read
    after save() copied a state into it (lookup returns saved keys only; the staging set is written by restore right
    before its copy). Read once per cache, at construction (before any trace capture, as today)."""
    return os.environ.get("QWEN36_PREFIX_CACHE_FAST_ALLOC", "0") == "1"


def alloc_empty_like(t, mesh):
    """Uninitialized DRAM-interleaved tensor on every device of `mesh` with the shape / dtype / TILE layout of the GDN
    state `t`: the same spec alloc_zeros_like produces (from_torch's default memory config), without the host zeros and
    the per-device upload."""
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([int(d) for d in t.shape]), t.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
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


# The vLLM KV block size the spec engine serves with (qwen36_vllm._BLOCK_SIZE; its spec gate refuses any other): vLLM's
# start_pos (num_computed_tokens of a prefix hit) is a multiple of it.
SPEC_VLLM_BLOCK = 64


def prefix_end_at_cap():
    """QWEN36_PREFIX_END_AT_CAP (default "0" = today's end-of-prompt checkpoint at floor((L-3)/128)*128): "1" saves it
    at spec_followup_cap(L), the resume cap of the next request that extends the prompt. Spec planner only
    (Qwen36Model.plan_spec_prefix); read per prefill."""
    return os.environ.get("QWEN36_PREFIX_END_AT_CAP", "0") == "1"


def spec_followup_cap(prompt_len, block=SPEC_VLLM_BLOCK, align=SPEC_PREFIX_ALIGN):
    """spec_resume_cap of a later request that extends this ``prompt_len``-token prompt (an exact-prefix follow-up, e.g.
    the next chat turn): vLLM reports the prompt's full KV blocks as cached, start_pos = floor(L/block)*block, and the
    follow-up resumes at most at floor((start_pos-1)/align)*align (its min-tail walk never triggers: >= 64 tokens
    follow).

    The default end checkpoint floor((L-3)/align)*align is the same position for L mod 128 in {0, 1, 2} and [64, 127],
    and one block ABOVE this cap for L mod 128 in [3, 63] (start_pos = 128q, cap = 128(q-1), checkpoint 128q), so about
    half of the exact-prefix follow-ups miss it and resume from an older snapshot. Here every one of them finds it. A
    follow-up that matched further (its cap is higher) still resumes here, at most 128 tokens below its cap; it
    recomputes the matched decode tokens anyway (they are never snapshotted). <= L - 3 for L >= 128; 0 below 128."""
    sp = (int(prompt_len) // int(block)) * int(block)
    return ((sp - 1) // align) * align if sp > 0 else 0


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def prefix_value_evict():
    """QWEN36_PREFIX_CACHE_VALUE_EVICT (default 0): 1 protects end-of-prompt checkpoints and multi-hit snapshots from
    eviction (see the module docstring)."""
    return os.environ.get("QWEN36_PREFIX_CACHE_VALUE_EVICT", "0") == "1"


def host_spill_min_pos():
    """QWEN36_PREFIX_HOST_SPILL_MIN_POS (default 2048): shorter snapshots are cheaper to recompute than to spill."""
    return _env_int("QWEN36_PREFIX_HOST_SPILL_MIN_POS", 2048)


def host_min_gain():
    """QWEN36_PREFIX_HOST_MIN_GAIN (default 1024): a host snapshot is used only when it resumes at least this many tokens
    later than the best device snapshot (a host restore costs a PCIe write of the whole snapshot)."""
    return _env_int("QWEN36_PREFIX_HOST_MIN_GAIN", 1024)


class HostSnap:
    """Lookup handle of a snapshot in the host tier: key (pos, digest) + per GDN layer (rec, conv_carry) host tensors.
    The handle keeps the tensors alive even if the host LRU drops the entry before the restore."""

    __slots__ = ("key", "tensors")

    def __init__(self, key, tensors):
        self.key, self.tensors = key, tensors


def prefix_stride_low_prio():
    """QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO (default 1): pure stride snapshots are evicted first; 0 = pure LRU."""
    return os.environ.get("QWEN36_PREFIX_CACHE_STRIDE_LOW_PRIO", "1") == "1"


class SpecPoints(list):
    """Sorted save points; ``low`` = the subset that is a pure stride point (low cache priority); ``end`` = the
    end-of-prompt checkpoint (or the last-full-chunk point that stands in for it), protected under value-aware eviction."""

    low = frozenset()
    end = frozenset()


def spec_save_points(
    prompt_len,
    s,
    learn,
    chunk=2048,
    min_len=8192,
    stride=8192,
    align=SPEC_PREFIX_ALIGN,
    end_point=True,
    end_block=None,
):
    """Sorted segment split positions (snapshot after tokens[:p]) for a spec prefill resuming at s, p in (s, L-3]:
    the learn positions, the end-of-prompt position for L >= min_len, else the last full-chunk boundary, and every
    `stride` tokens from s (chunk-aligned strides only). Same policy as the plain O path (model._plan_prefix_prefill).
    ``end_point`` False (a non-final chunked-prefill chunk resumed exactly, Option B): no end-of-prompt / last-full-chunk
    point (the caller snapshots the chunk end itself); learn positions and strides stay. The result is a SpecPoints whose
    ``low`` holds the pure stride points (in the stride range only: not a learn / end / last-chunk point).
    ``end_block`` (QWEN36_PREFIX_END_AT_CAP: the vLLM block size; None = today's rule): the end-of-prompt position is
    spec_followup_cap(prompt_len, end_block) instead of floor((prompt_len-3)/align)*align, so the next request extending
    the prompt finds it at its resume cap. The move never adds a split: a stride point at today's end position (it stood
    in for the end point there) is dropped when the moved end point is taken, one block below it.
    """
    pts = {p for p in learn if s < p <= prompt_len - 3}
    b_end = ((prompt_len - 3) // align) * align
    moved_from = None
    if end_block is not None:
        e = spec_followup_cap(prompt_len, end_block, align)
        if e != b_end:
            moved_from = b_end
        b_end = e
    end = set()
    if not end_point:
        pass
    elif prompt_len >= min_len:
        if b_end in pts:
            end.add(b_end)  # already a learn point: it is still the end-of-prompt checkpoint
        elif b_end > max(s, max(pts, default=0)):
            pts.add(b_end)
            end.add(b_end)
    elif not pts:
        p = s + ((prompt_len - s) // chunk) * chunk
        if s < p <= prompt_len - 3:
            pts.add(p)
            end.add(p)
    out = SpecPoints()
    out.end = frozenset(end)
    if stride > 0 and stride % chunk == 0:
        strided = set(range(s + stride, prompt_len - 2, stride))
        if moved_from is not None and end:
            strided.discard(moved_from)  # QWEN36_PREFIX_END_AT_CAP: the end point one block lower replaces it
        out.low = frozenset(strided - pts)
        pts.update(strided)
    out.extend(sorted(pts))
    return out


class GdnPrefixStateCache:
    """LRU cache of GDN (rec_state, conv_carry) snapshots at 128-aligned prompt positions.

    Allocates `num_slots` snapshot sets (one rec/conv_carry pair per GDN layer) at construction, i.e. BEFORE any
    prefill/decode trace is captured (allocating behind captured traces is unsafe). save/restore copy between a slot and
    the GDN state currently bound on the layers (the persistent B=1 prefill scratch inside prefill_paged_slots)."""

    def __init__(self, model, num_slots=None, name="GDN prefix-state cache", host_slots=None):
        self.model = model
        self.name = name
        self.num_slots = int(num_slots if num_slots is not None else os.environ.get("QWEN36_PREFIX_CACHE_SLOTS", "32"))
        assert self.num_slots >= 1
        self.debug = os.environ.get("QWEN36_PREFIX_CACHE_DEBUG", "0") == "1"
        # The B=1 scratch (and its init buffers) must exist first: the slots mirror its shapes.
        model._ensure_gdn_prefill_scratch()
        self._dns = [dn for dn, *_ in model._gdn_prefill_scratch]
        self.fast_alloc = prefix_cache_fast_alloc()  # QWEN36_PREFIX_CACHE_FAST_ALLOC
        t_alloc = time.perf_counter()
        self._slots = []  # slot -> [(rec, conv_carry)] per GDN layer
        for _ in range(self.num_slots):
            self._slots.append([self._alloc_pair(dn) for dn in self._dns])
        self._free = list(range(self.num_slots))
        self._lru = OrderedDict()  # (pos, digest) -> slot, least recently used first
        self._by_pos = {}  # pos -> {digest, ...}
        self._positions = []  # sorted positions present
        self.stats = dict(lookups=0, hits=0, tokens_skipped=0, saves=0, evictions=0, evictions_low=0)
        if self.fast_alloc:
            logger.info(
                f"{self.name}: {self.num_slots} slots x {len(self._dns)} GDN layers allocated on device "
                f"(QWEN36_PREFIX_CACHE_FAST_ALLOC=1, uninitialized) in {time.perf_counter() - t_alloc:.2f} s."
            )
        else:
            logger.info(f"{self.name}: {self.num_slots} slots x {len(self._dns)} GDN layers allocated.")
        # Host spill tier (QWEN36_PREFIX_HOST_SLOTS, default 0 = off): one device staging set, allocated here (before any
        # trace capture), receives a host snapshot; restore then copies staging -> bound state like a device hit.
        self.host_slots = int(host_slots if host_slots is not None else os.environ.get("QWEN36_PREFIX_HOST_SLOTS", "0"))
        self._host = OrderedDict()  # (pos, digest) -> [(host rec, host conv_carry)] per GDN layer, LRU first
        self._staging = None
        if self.host_slots > 0:
            self._staging = [self._alloc_pair(dn) for dn in self._dns]
            mb = self._snapshot_mb()
            logger.info(
                f"{self.name}: host spill tier {self.host_slots} slots x ~{mb:.0f} MB = ~{self.host_slots * mb / 1024:.1f} GB "
                f"host RAM at most (spill pos >= {host_spill_min_pos()}, min gain {host_min_gain()} tokens)"
            )

    def _snapshot_mb(self):
        """Host bytes of one snapshot (every device of the mesh), from the slot shapes; an estimate (tile padding)."""
        try:
            tot = 0
            for rec, carry in self._slots[0]:
                for t in (rec, carry):
                    n = 1
                    for d in t.shape:
                        n *= int(d)
                    tot += n * (4 if t.dtype == ttnn.float32 else 2)
            return tot * int(self._dns[0].mesh.get_num_devices()) / 2**20
        except Exception:  # noqa: BLE001 - logging aid only
            return 0.0

    def _alloc_pair(self, dn):
        scratch = next(s for s in self.model._gdn_prefill_scratch if s[0] is dn)
        alloc = alloc_empty_like if self.__dict__.get("fast_alloc", False) else alloc_zeros_like
        return alloc(scratch[1], dn.mesh), alloc(scratch[3], dn.mesh)

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
        host = self._host_tier()
        pos = max((p for p in found if (p, found[p]) in self._lru), default=0)
        hpos = max((p for p in found if (p, found[p]) in host), default=0)
        if hpos > pos and hpos - pos >= host_min_gain():
            pos = hpos
        if pos == 0:
            return 0, None
        key = (pos, found[pos])
        if key in host:
            host.move_to_end(key)
            handle = HostSnap(key, host[key])
            self.bump("host_hits")
        else:
            self._lru.move_to_end(key)
            self._low().discard(key)  # a hit proves value
            hits = self._hit_counts()
            hits[key] = hits.get(key, 0) + 1
            handle = self._lru[key]
        self.stats["hits"] += 1
        self.stats["tokens_skipped"] += pos
        return pos, handle

    # --- pinning (MTP chunked prefill Option B): a chunk-end snapshot is kept until its request's next chunk consumed it ---
    _STAT_KEYS = ("pinned_saves", "skipped_saves", "exact_resumes", "refusals")

    def _pins(self):
        """owner (engine row) -> key of its pinned snapshot, insertion ordered (lazy: caches built without __init__ work)."""
        return self.__dict__.setdefault("_pin_map", OrderedDict())

    def _low(self):
        """Keys saved at low priority (pure stride snapshots, evicted first), lazy like _pins."""
        return self.__dict__.setdefault("_low_keys", set())

    def _end(self):
        """Keys saved as end-of-prompt checkpoints (protected under QWEN36_PREFIX_CACHE_VALUE_EVICT), lazy like _pins."""
        return self.__dict__.setdefault("_end_keys", set())

    def _hit_counts(self):
        """Device key -> lookup hits since it was saved (multi-hit = shared prefix), lazy like _pins."""
        return self.__dict__.setdefault("_hit_map", {})

    def _host_tier(self):
        return self.__dict__.setdefault("_host", OrderedDict())

    def _host_cap(self):
        return int(self.__dict__.get("host_slots", 0) or 0)

    def _protected(self, pinned):
        """Unpinned device keys protected from eviction (value-aware policy): end-of-prompt checkpoints and keys hit
        >= QWEN36_PREFIX_CACHE_PROTECT_HITS times, at most QWEN36_PREFIX_CACHE_PROTECT_FRAC * num_slots of them (the most
        recently used ones)."""
        need = _env_int("QWEN36_PREFIX_CACHE_PROTECT_HITS", 2)
        cap = int(self.num_slots * float(os.environ.get("QWEN36_PREFIX_CACHE_PROTECT_FRAC", "0.5")))
        end, hits = self._end(), self._hit_counts()
        prot = [k for k in self._lru if k not in pinned and (k in end or hits.get(k, 0) >= need)]
        return set(prot[len(prot) - cap :]) if cap > 0 else set()

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
        out = (
            f"pinned={self.pinned_count()}/{self.max_pinned_limit()} pinned_saves={st.get('pinned_saves', 0)} "
            f"skipped_saves={st.get('skipped_saves', 0)} evictions={st.get('evictions', 0)} "
            f"evictions_low={st.get('evictions_low', 0)} low={len(self._low())} "
            f"exact_resumes={st.get('exact_resumes', 0)} refusals={st.get('refusals', 0)} "
            f"saves={st.get('saves', 0)} entries={len(self._lru)}/{self.num_slots}"
        )
        vc = st.get("vllm_cached_tokens", 0)
        if st.get("resume_prefills", 0):
            hidden = st.get("hidden_recompute_tokens", 0)
            out += (
                f" | resume: prefills={st.get('resume_prefills', 0)} vllm_cached_tok={vc} "
                f"gdn_resumed_tok={st.get('gdn_resumed_tokens', 0)} hidden_recompute_tok={hidden} "
                f"({100.0 * hidden / max(vc, 1):.1f}% of vLLM-cached) below_cap_tok={st.get('below_cap_tokens', 0)} "
                f"from_device={st.get('resume_device', 0)} from_host={st.get('resume_host', 0)} "
                f"from_0={st.get('resume_none', 0)}"
            )
        if self._host_cap() > 0:
            n_sp, n_rs = st.get("spills", 0), st.get("host_restores", 0)
            out += (
                f" | host: entries={len(self._host_tier())}/{self._host_cap()} spills={n_sp} "
                f"spill_ms_avg={st.get('spill_ms', 0.0) / max(n_sp, 1):.1f} host_hits={st.get('host_hits', 0)} "
                f"restore_ms_avg={st.get('host_restore_ms', 0.0) / max(n_rs, 1):.1f} "
                f"host_evictions={st.get('host_evictions', 0)}"
            )
        if prefix_value_evict():
            out += f" | protected={len(self._protected(set(self._pins().values())))}"
        return out

    def record_resume(self, start_pos, cap, s, src_slot):
        """Account one prefill that vLLM reported ``start_pos`` cached tokens for (true resume metrics): the GDN resume
        position s actually used, the tokens recomputed although vLLM counts them as cached (start_pos - s), the part of
        that below the MTP-safe cap (cap - s: a missing snapshot, not structural), and the tier the state came from."""
        start_pos, cap, s = int(start_pos), int(cap), int(s)
        if start_pos <= 0:
            return
        self.bump("resume_prefills")
        self.bump("vllm_cached_tokens", start_pos)
        self.bump("gdn_resumed_tokens", s)
        self.bump("hidden_recompute_tokens", max(start_pos - s, 0))
        self.bump("below_cap_tokens", max(cap - s, 0))
        tier = "none" if s <= 0 else ("host" if isinstance(src_slot, HostSnap) else "device")
        self.bump(f"resume_{tier}")
        if self.debug:
            logger.info(
                f"[prefix-cache] resume vllm_start={start_pos} cap={cap} gdn_resume={s} tier={tier} "
                f"hidden_recompute={max(start_pos - s, 0)} below_cap={max(cap - s, 0)}"
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
        if prefix_value_evict():
            prot = self._protected(pinned)
            for k in self._lru:
                if k not in pinned and k not in prot:
                    return k
        for k in self._lru:
            if k not in pinned:
                return k
        return None

    def save(self, tokens, pos, low_prio=False, protect=False, copy_fn=None):
        """Snapshot the currently bound GDN state (the state AFTER tokens[:pos]) into a slot. ``low_prio``: a new key is
        evicted before normal ones (an existing normal key is never downgraded; a normal save upgrades a low key).
        ``protect``: an end-of-prompt checkpoint (kept longer under QWEN36_PREFIX_CACHE_VALUE_EVICT). Eviction
        skips pinned snapshots; when no slot is free and none is evictable the save is skipped (counted, debug log). The
        evicted snapshot moves to the host tier when it is on and the snapshot qualifies (_spill_ok).
        ``copy_fn(dn, rec_dst, carry_dst)`` (QWEN36_PREFILL_INTRACE_SNAP; None = copy the bound state): writes the
        state after tokens[:pos] of GDN layer dn into the slot's tensors from another source, an in-replay snapshot
        (Qwen36Model.gdn_snap_copy_fn). Same keys, eviction and accounting either way.
        Returns True when the snapshot is in the cache afterwards."""
        buf = _tokens_np(tokens)
        key = (int(pos), self._digest(buf, int(pos)))
        if key in self._lru:
            self._lru.move_to_end(key)
            if not low_prio:
                self._low().discard(key)
            if protect:
                self._end().add(key)
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
            spilled = False
            if self._spill_ok(old_key):
                try:
                    self._spill(old_key, slot)  # reads the slot BEFORE the copies below overwrite it; index entry kept
                    spilled = True
                except Exception as e:  # noqa: BLE001 - a failed spill only loses the snapshot (as without the tier)
                    self.bump("spill_errors")
                    logger.warning(f"{self.name}: host spill of the snapshot at {old_key[0]} failed: {e!r}")
            if not spilled:
                self._drop_pos(old_key)
            if old_key in self._low():
                self._low().discard(old_key)
                self.bump("evictions_low")
            self._end().discard(old_key)
            self._hit_counts().pop(old_key, None)
            self.stats["evictions"] += 1
        # the same state may sit in the host tier: it moves back to this device slot (one tier per key)
        self._host_tier().pop(key, None)
        for dn, (rec, carry) in zip(self._dns, self._slots[slot]):
            if copy_fn is not None:
                copy_fn(dn, rec, carry)
                continue
            ttnn.copy(dn.rec_state, rec)
            ttnn.copy(dn.conv_carry, carry)
        self._lru[key] = slot
        if low_prio:
            self._low().add(key)
        if protect:
            self._end().add(key)
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

    def _spill_ok(self, key):
        if self._host_cap() <= 0 or key[0] < host_spill_min_pos():
            return False
        return key not in self._low() or os.environ.get("QWEN36_PREFIX_HOST_SPILL_LOW", "0") == "1"

    def _spill(self, key, slot):
        """Copy device slot ``slot`` (snapshot ``key``) to host (blocking ttnn.from_device: waits for the queued work,
        then reads every device's bytes) and insert it into the host LRU; over capacity the LRU host entry is dropped."""
        t0 = time.perf_counter()
        snap = [(ttnn.from_device(rec), ttnn.from_device(carry)) for rec, carry in self._slots[slot]]
        host = self._host_tier()
        host[key] = snap
        host.move_to_end(key)
        while len(host) > self._host_cap():
            old, _ = host.popitem(last=False)
            if old not in self._lru:
                self._drop_pos(old)
            self.bump("host_evictions")
        self.bump("spills")
        self.bump("spill_ms", (time.perf_counter() - t0) * 1e3)

    def restore(self, slot):
        """Copy a snapshot into the currently bound GDN state (rec_state + conv_carry of every GDN layer). ``slot``: a
        device slot index, or a HostSnap (host tier hit: host -> staging slot (PCIe write), then the device copy)."""
        if isinstance(slot, HostSnap):
            t0 = time.perf_counter()
            for (srec, scarry), (hrec, hcarry) in zip(self._staging, slot.tensors):
                ttnn.copy_host_to_device_tensor(hrec, srec)
                ttnn.copy_host_to_device_tensor(hcarry, scarry)
            pairs = self._staging
            self.bump("host_restores")
            self.bump("host_restore_ms", (time.perf_counter() - t0) * 1e3)
        else:
            pairs = self._slots[slot]
        for dn, (rec, carry) in zip(self._dns, pairs):
            ttnn.copy(rec, dn.rec_state)
            ttnn.copy(carry, dn.conv_carry)

    def log(self, prompt_len, start_pos, hit_pos, saved):
        if self.debug:
            logger.info(
                f"[prefix-cache] len={prompt_len} vllm_start={start_pos} gdn_hit={hit_pos} saved={saved} "
                f"entries={len(self._lru)}/{self.num_slots} stats={self.stats}"
            )
