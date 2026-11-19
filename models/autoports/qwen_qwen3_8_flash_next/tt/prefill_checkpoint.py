# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact device checkpoints for continuation of an odd prefill tail.

An odd exclusive end E has a singleton at even position E-1. Preserve its
pre-token recurrent state while committing the ordinary n1 forward normally.
An immediate continuation can replay that token in a multi-row prefill chunk,
without changing either the n1 FP32 or n>1 BF16 decoder arithmetic policy.

Snapshots exclude KV, compressed index keys, page tables and sampler state.
The even-position singleton cannot finish a four-token compressed-index group;
replay overwrites its K/V row before causal attention reads it. Page ownership
always comes from the current validated scheduler submission.
"""
from dataclasses import dataclass

from .slot_state import validate_remap


def compact_state(cache):
    result = []
    for index, layer in enumerate(cache.layers):
        for field in ("conv", "recurrent", "ple_conv", "token_history"):
            tensor = getattr(layer, field, None)
            if tensor is not None:
                result.append((f"layer{index}.{field}", tensor))
        if layer.attention is not None:
            for field in ("raw_history", "position"):
                result.append((f"layer{index}.attention.{field}", getattr(layer.attention, field)))
    return tuple(result)


@dataclass
class _Checkpoint:
    tensors: tuple
    end: int | None = None
    token: int | None = None


class PrefillCheckpoints:
    """Persistent request-owned snapshots, indexed by current physical slot.

    Snapshot buffers are used only by transition copies outside model traces.
    A slot permutation moves buffer references and metadata together; every
    allocation remains alive, with unchanged addresses, until generator close.
    No extra device scratch is required for checkpoint permutation.
    """

    def __init__(self, caches, mesh=None, *, allocate=None, copy_tensor=None):
        if allocate is None or copy_tensor is None:
            import ttnn

            if allocate is None:
                allocate = lambda value: ttnn.empty(
                    value.shape,
                    dtype=value.dtype,
                    layout=value.layout,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if copy_tensor is None:
                copy_tensor = ttnn.copy
        self.copy_tensor = copy_tensor
        self.records = [
            _Checkpoint(tuple((name, allocate(value)) for name, value in compact_state(cache))) for cache in caches
        ]

    def _copy(self, source, destination):
        for (left, src), (right, dst) in zip(source, destination, strict=True):
            if left != right:
                raise ValueError("Checkpoint/cache state signatures differ")
            self.copy_tensor(src, dst)

    def warmup(self, caches):
        for cache, record in zip(caches, self.records, strict=True):
            self._copy(compact_state(cache), record.tensors)
            self._copy(record.tensors, compact_state(cache))

    def can_restore(self, slot, start, token, cache_position):
        record = self.records[slot]
        return bool(
            start > 0 and start % 2 == 1 and record.end == start and cache_position == start and record.token == token
        )

    def save(self, slot, cache, *, end, token):
        if end <= 0 or end % 2 != 1:
            raise ValueError("Only a final singleton at an even absolute position is checkpointable")
        record = self.records[slot]
        self._copy(compact_state(cache), record.tensors)
        record.end, record.token = int(end), int(token)

    def restore(self, slot, cache):
        record = self.records[slot]
        if record.end is None:
            raise ValueError("No valid prefill checkpoint")
        self._copy(record.tensors, compact_state(cache))
        cache.position = record.end - 1

    def invalidate(self, slots):
        for slot in slots:
            self.records[slot].end = self.records[slot].token = None

    def remap(self, remap):
        remap = validate_remap(remap, len(self.records))
        self.records = [self.records[source] for source in remap]
