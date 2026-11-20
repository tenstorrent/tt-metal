# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""S1 prefix resume: the device pool of saved prefill sliding-history tails (bench/resident/s1prefix/S1_DESIGN.md
§4.3, ADDENDUM_1, ADDENDUM_2 §1; LEAD_DECISIONS_S1P Q1-Q3).

The PB prefill carries each sliding layer's last 1,152 K/V rows (the window plus one page) from one chunk to the next
in `CacheState.prefill_history` (prefill_paged.py:232-243, 646-648), outside vLLM's blocks. A prefix-cache hit at c
needs that tail as of c, so the scheduler (vllm_tt_plugin.prefix_resume, the tail directory it owns) orders:
- store: after a row's chunk ends at e (a canonical run, e on the 8,192 grid), copy the row's tail into slot k;
- restore: before a row resumes at c, hand the generator slot k's tail as the row's history.
A slot is K/V of every sliding layer at the tail's shape, allocated once (K x 235.9 MB per chip at the shipped
geometry: 50 x 2 x 4 x 1,152 x 256 x 2 B). Each slot records (generation, end, digest of tokens[:end]); a restore
whose order does not match the slot's record raises TailRestoreMismatch (a scheduler / adapter disagreement is a bug).
"""

import hashlib
from collections import Counter

import torch

TAIL_ROWS = 1152  # the window plus one page (prefill_paged.py:236-238: window + PAGE_SIZE)


class TailRestoreMismatch(RuntimeError):
    """A restore order does not match what its slot holds."""


def token_digest(tokens, end):
    """sha256 of the first `end` token ids of one prompt row (a 1-D tensor or sequence)."""
    ids = torch.as_tensor(tokens)[:end].to(torch.int64).contiguous()
    if ids.numel() != end:
        raise ValueError(f"The row holds {ids.numel()} tokens, fewer than the tail's end {end}")
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def device_alloc(mesh_device):
    """The shipped allocator: one BF16 DRAM tensor per K or V of a sliding layer, at the tail's per-device shape."""
    import ttnn

    def alloc(layer):
        return ttnn.zeros(
            [1, layer.kv_heads, TAIL_ROWS, layer.head_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return alloc


def device_copy(src, dst):
    import ttnn

    ttnn.copy(src, dst)


class TailPool:
    def __init__(self, slots, layers, *, alloc, copy):
        if slots < 0:
            raise ValueError(f"A tail pool needs >= 0 slots, got {slots}")
        self.slots = slots
        self.sliding = [bool(layer.sliding) for layer in layers]
        self.tensors = [[(alloc(layer), alloc(layer)) if layer.sliding else None for layer in layers]
                        for _ in range(slots)]
        self.records = [None] * slots  # (generation, end, digest)
        self.copy = copy
        self.counters = Counter()

    def _slot(self, slot):
        if not 0 <= slot < self.slots:
            raise ValueError(f"Tail slot {slot} is outside the pool's {self.slots} slots")

    def store(self, slot, generation, end, digest, layers):
        """Copy a row's history (one (k, v) per sliding layer, None per full layer) into `slot`."""
        self._slot(slot)
        if len(layers) != len(self.sliding):
            raise ValueError(f"A history of {len(layers)} layers for a pool of {len(self.sliding)}")
        for i, (tail, sliding) in enumerate(zip(layers, self.sliding)):
            if not sliding:
                if tail is not None:
                    raise ValueError(f"Full layer {i} carries a history")
                continue
            if tail is None or len(tail) != 2:
                raise ValueError(f"Sliding layer {i} has no (k, v) history to store")
            for src, dst in zip(tail, self.tensors[slot][i]):
                if tuple(src.shape) != tuple(dst.shape):
                    raise ValueError(f"Sliding layer {i}: a tail of shape {tuple(src.shape)} for a slot of "
                                     f"{tuple(dst.shape)} (a tail at e >= {TAIL_ROWS} holds {TAIL_ROWS} rows)")
                self.copy(src, dst)
        self.records[slot] = (generation, end, digest)
        self.counters["stores"] += 1

    def restore(self, slot, generation, end, digest):
        """The history to hand a row resuming at `end` from `slot` (the slot's own tensors: the generator slices
        and concatenates them into the row's next tail, and never writes them)."""
        self._slot(slot)
        if self.records[slot] != (generation, end, digest):
            self.counters["mismatches"] += 1
            raise TailRestoreMismatch(f"Restore of slot {slot} as (generation {generation}, end {end}, digest "
                                      f"{digest[:12]}) but the slot holds {self._describe(self.records[slot])}")
        self.counters["restores"] += 1
        return [self.tensors[slot][i] if sliding else None for i, sliding in enumerate(self.sliding)]

    @staticmethod
    def _describe(record):
        if record is None:
            return "nothing"
        generation, end, digest = record
        return f"(generation {generation}, end {end}, digest {digest[:12]})"
