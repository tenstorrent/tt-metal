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


class GdnPrefixStateCache:
    """LRU cache of GDN (rec_state, conv_carry) snapshots at 128-aligned prompt positions.

    Allocates `num_slots` snapshot sets (one rec/conv_carry pair per GDN layer) at construction, i.e. BEFORE any
    prefill/decode trace is captured (allocating behind captured traces is unsafe). save/restore copy between a slot and
    the GDN state currently bound on the layers (the persistent B=1 prefill scratch inside prefill_paged_slots)."""

    def __init__(self, model, num_slots=None):
        self.model = model
        self.num_slots = int(num_slots if num_slots is not None else os.environ.get("QWEN36_PREFIX_CACHE_SLOTS", "64"))
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
        self.stats = dict(lookups=0, hits=0, tokens_skipped=0, saves=0, evictions=0)
        logger.info(f"GDN prefix-state cache: {self.num_slots} slots x {len(self._dns)} GDN layers allocated.")

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
        self.stats["hits"] += 1
        self.stats["tokens_skipped"] += pos
        return pos, self._lru[key]

    def save(self, tokens, pos):
        """Snapshot the currently bound GDN state (the state AFTER tokens[:pos]) into a slot."""
        buf = _tokens_np(tokens)
        key = (int(pos), self._digest(buf, int(pos)))
        if key in self._lru:
            self._lru.move_to_end(key)
            return
        if self._free:
            slot = self._free.pop()
        else:
            old_key, slot = self._lru.popitem(last=False)
            self._drop_pos(old_key)
            self.stats["evictions"] += 1
        for dn, (rec, carry) in zip(self._dns, self._slots[slot]):
            ttnn.copy(dn.rec_state, rec)
            ttnn.copy(dn.conv_carry, carry)
        self._lru[key] = slot
        if key[0] not in self._by_pos:
            self._by_pos[key[0]] = set()
            bisect.insort(self._positions, key[0])
        self._by_pos[key[0]].add(key[1])
        self.stats["saves"] += 1

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
