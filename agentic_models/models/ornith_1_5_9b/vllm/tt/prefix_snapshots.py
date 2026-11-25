# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN state snapshots that make vLLM automatic prefix caching exact for this hybrid model.

vLLM caches only the paged KV of the 8 full-attention layers. The 24 Gated-DeltaNet layers keep a per-slot conv +
recurrent state that KV pages cannot rebuild (the documented ``recurrent_state`` exclusion). So:

1. Every final prompt prefill is split at its last 32-token boundary A: prefill [start, A), copy the slot's GDN state
   into a preallocated pool entry keyed by hash(tokens[:A]), then prefill [A, L).
2. The scheduler's prefix hit is trimmed to the largest position P <= hit that has a snapshot whose key matches the
   request's tokens (0 if none): the model is only ever asked to resume where an exact state exists.
3. A resumed request restores the entry into its slot and continues at P.

An agent conversation (turn n+1 = turn n's prompt + answer + tool output) then recomputes only the new tokens. One
entry per conversation: a turn's snapshot supersedes the one it resumed from; otherwise LRU eviction.
4. Shared prefixes (one system prompt, different continuations): when a hit H is trimmed for lack of a snapshot, H is
   recorded as a junction; the next prefill that covers it splits there too and keeps a shared snapshot at H (never
   superseded by a conversation), so later requests with that prefix resume at H.

The scheduler (EngineCore) and the model run in one process (uniproc executor), so both use this module's REGISTRY.
Design follows the Qwen3.8 TT prefix-cache release (machine C, 2026-10); here the copies reuse the per-slot
read/write programs BatchState already warms, so no program compiles while decode traces are held.
"""

import hashlib
import os
from collections import Counter, OrderedDict

import numpy as np

BLOCK = 32
ENABLED = os.environ.get("ORNITH_PREFIX_CACHE", "1") == "1"
# ROUND 6: prompts shorter than this get no end-of-prompt snapshot (shared-prefix junction snapshots are unchanged). The snapshot costs a second prefill
# call (~30 ms) on every request, while recomputing a short prefix on a later hit costs < ~150 ms; hits on such a
# prefix trim to the next shorter snapshot (or 0) exactly as after an eviction. 0 snapshots every prompt (old behaviour).
SNAPSHOT_MIN_TOKENS = int(os.environ.get("ORNITH_SNAPSHOT_MIN_TOKENS", "2048"))
CAPACITY = int(os.environ.get("ORNITH_PREFIX_SNAPSHOTS", "64"))


def prefix_key(token_ids, length):
    return hashlib.blake2b(np.asarray(token_ids[:length], dtype=np.int64).tobytes(), digest_size=16).digest()


class SnapshotRegistry:
    """token-prefix key -> (position, pool entry), LRU order; one entry per conversation."""

    def __init__(self, capacity=CAPACITY):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.positions = {}  # position -> number of entries there (lookup walks positions, not entries)
        self.free = list(range(capacity))
        self.shared = set()  # keys of junction snapshots: never superseded by a conversation's next turn
        self.pins = Counter()  # key -> hits granted by the scheduler and not yet restored (never dropped meanwhile)
        self.junctions = OrderedDict()  # prefix key -> position of a trimmed shared-prefix hit, awaiting a snapshot
        self.stats = dict(
            created=0,
            restored=0,
            evicted=0,
            superseded=0,
            missing=0,
            trimmed_tokens=0,
            granted_tokens=0,
            junctions=0,
            shared_created=0,
        )

    def lookup(self, token_ids, limit):
        """Largest snapshot position <= limit whose prefix matches token_ids: (position, key) or (0, None)."""
        for position in sorted((p for p in self.positions if p <= limit), reverse=True):
            key = prefix_key(token_ids, position)
            if key in self.entries:
                return position, key
        return 0, None

    def get(self, key):
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
        return entry

    def _drop(self, key):
        self.shared.discard(key)
        position, index = self.entries.pop(key)
        self.positions[position] -= 1
        if not self.positions[position]:
            del self.positions[position]
        self.free.append(index)

    def add_junction(self, key, position):
        if key not in self.entries and key not in self.junctions:
            self.junctions[key] = position
            self.stats["junctions"] += 1
            while len(self.junctions) > 4 * self.capacity:
                self.junctions.popitem(last=False)

    def junction_in(self, token_ids, start, end):
        """Largest recorded junction position in (start, end) whose prefix matches token_ids, else None."""
        for position in sorted({p for p in self.junctions.values() if start < p < end}, reverse=True):
            key = prefix_key(token_ids, position)
            if self.junctions.get(key) == position:
                return position, key
        return None

    def put(self, key, position, supersedes=None, shared=False):
        """Reserve the pool entry that will hold the state at `position`."""
        if key in self.entries:
            self._drop(key)
        self.junctions.pop(key, None)
        if (
            supersedes is not None
            and supersedes != key
            and supersedes in self.entries
            and supersedes not in self.shared
            and not self.pins[supersedes]
        ):
            self._drop(supersedes)
            self.stats["superseded"] += 1
        if not self.free:
            victim = next((k for k in self.entries if not self.pins[k]), next(iter(self.entries)))
            self._drop(victim)
            self.stats["evicted"] += 1
        index = self.free.pop()
        self.entries[key] = (position, index)
        self.positions[position] = self.positions.get(position, 0) + 1
        self.stats["created"] += 1
        if shared:
            self.shared.add(key)
            self.stats["shared_created"] += 1
        return index


REGISTRY = SnapshotRegistry()


class SnapshotPool:
    """Device storage: per pool entry and GDN layer, one conv [1,1,3,2048] RM BF16 and one recurrent
    [1,8,128,128] FP32 tensor, shaped exactly like BatchState's zero-state scratch so the warmed per-slot
    read/write programs serve save and restore."""

    def __init__(self, model, cache, batch_state, capacity=CAPACITY):
        import ttnn

        self.model, self.cache = model, cache
        self.layers = sorted(batch_state.layer_scratch)
        self.entries = [
            {
                i: (
                    ttnn.zeros_like(batch_state.layer_scratch[i].zero_conv),
                    ttnn.zeros_like(batch_state.layer_scratch[i].zero_recurrent),
                )
                for i in self.layers
            }
            for _ in range(capacity)
        ]

    def save(self, slot, index):
        for i in self.layers:
            layer, state = self.model.layers[i], self.cache[i][slot]
            conv, recurrent = self.entries[index][i]
            import ttnn

            ttnn.copy(layer.read_linear_state(state, "conv"), conv)
            ttnn.copy(layer.read_linear_state(state, "recurrent"), recurrent)

    def restore(self, index, slot):
        for i in self.layers:
            layer, state = self.model.layers[i], self.cache[i][slot]
            conv, recurrent = self.entries[index][i]
            layer.write_linear_state(state, "conv", conv)
            layer.write_linear_state(state, "recurrent", recurrent)

    def warm(self, slot=0, index=0):
        """Compile the copy programs before any trace capture (state of `slot` is preserved)."""
        self.save(slot, index)
        self.restore(index, slot)


def install_hit_trim(log=print):
    """Trim vLLM's attention-only prefix hit to the last GDN snapshot position (or 0)."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager.get_computed_blocks, "_ornith_trim", False):
        return
    try:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    except ImportError:
        config = None
    if config is not None and getattr(config.cache_config, "mamba_cache_mode", "none") != "none":
        # vLLM's own hybrid "align" mode would split prefills for MambaSpec pages this model does not use.
        log(f"Ornith prefix caching: mamba_cache_mode {config.cache_config.mamba_cache_mode} -> none")
        config.cache_config.mamba_cache_mode = "none"
    original = KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request):
        blocks, hit, boundary = original(self, request)
        if hit <= 0:
            return blocks, hit, boundary
        position, key = REGISTRY.lookup(request.all_token_ids, hit)
        if key is not None:
            REGISTRY.pins[key] += 1  # released by the model when it restores (or misses) this snapshot
        REGISTRY.stats["trimmed_tokens"] += hit - position
        REGISTRY.stats["granted_tokens"] += position
        if position == hit:
            return blocks, hit, boundary
        if hit - position >= 8 * BLOCK:
            # A cached shared prefix without GDN state: snapshot it when this request prefills past it.
            REGISTRY.add_junction(prefix_key(request.all_token_ids, hit), hit)
        # vLLM 0.31 records hit statistics from the returned (trimmed) count in record_prefix_cache_stats.
        if position == 0:
            return self.empty_kv_cache_blocks, 0, 0
        keep = position // BLOCK
        return self.create_kv_cache_blocks(tuple(list(group[:keep]) for group in blocks.blocks)), position, 0

    get_computed_blocks._ornith_trim = True
    KVCacheManager.get_computed_blocks = get_computed_blocks
    log(f"Ornith prefix caching: GDN snapshot hit trim installed (capacity {REGISTRY.capacity})")
