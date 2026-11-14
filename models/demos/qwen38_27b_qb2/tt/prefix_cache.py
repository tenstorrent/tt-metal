# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device snapshots of the linear-attention state for vLLM prefix caching."""

import hashlib
import os
from collections import OrderedDict

import torch

import ttnn

PAGE_SIZE = 32
# Prefill SDPA shrinks its 128-token chunks to the alignment of the start position.
ALIGNMENT = 128
_ELEMENT_BYTES = {"FLOAT32": 4, "BFLOAT16": 2}


def _env_int(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be a non-negative decimal integer")
    return int(value)


def prefix_snapshot_settings():
    capacity = _env_int("QWEN_PREFIX_SNAPSHOTS", 0)
    minimum = _env_int("QWEN_PREFIX_SNAPSHOT_MIN", 4096)
    margin = _env_int("QWEN_PREFIX_SNAPSHOT_MARGIN", 16)
    if minimum < PAGE_SIZE:
        raise ValueError(f"QWEN_PREFIX_SNAPSHOT_MIN must be at least {PAGE_SIZE}")
    return capacity, minimum, max(1, margin)


def _token_bytes(row, begin, end):
    return row[begin:end].to(torch.int32).contiguous().numpy().tobytes()


class PrefixSnapshots:
    def __init__(self, generator, capacity, *, minimum, margin):
        if capacity < 1:
            raise ValueError("A snapshot pool needs at least one entry")
        self.generator = generator
        self.capacity, self.minimum, self.margin = capacity, minimum, margin
        self.batch_size = generator.cache.batch_size
        self.linear = [i for i, state in enumerate(generator.cache.layers) if state.recurrent is not None]
        self.entries = [
            [
                {name: self._allocate(getattr(generator.cache.layers[i], name)) for name in ("conv", "recurrent")}
                for i in self.linear
            ]
            for _ in range(capacity)
        ]
        self.keys = OrderedDict()
        self.free = list(range(capacity - 1, -1, -1))
        # Compile the copy programs before any trace capture.
        self._copy_in(0, 0)
        self._copy_out(0, 0)

    def _allocate(self, tensor):
        shape = [1, *list(tensor.shape)[1:]]
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), tensor.dtype, tensor.layout, self.generator.mesh, ttnn.DRAM_MEMORY_CONFIG
        )

    def bytes_per_entry(self):
        total = 0
        for buffers in self.entries[0]:
            for tensor in buffers.values():
                count = 1
                for size in tensor.shape:
                    count *= size
                total += count * _ELEMENT_BYTES[str(tensor.dtype).rsplit(".", 1)[-1]]
        return total

    def mark(self, end, begin):
        position = (end - self.margin) // ALIGNMENT * ALIGNMENT
        return position if position >= self.minimum and position > begin else None

    @staticmethod
    def key(row, position):
        return position, hashlib.blake2b(_token_bytes(row, 0, position), digest_size=16).digest()

    def __contains__(self, key):
        return key in self.keys

    def find(self, row, limit):
        positions = sorted({position for position, _ in self.keys if position <= limit})
        digest = hashlib.blake2b(digest_size=16)
        best, previous = None, 0
        for position in positions:
            digest.update(_token_bytes(row, previous, position))
            previous = position
            key = (position, digest.digest())
            if key in self.keys:
                best = key
        if best is None:
            return None
        self.keys.move_to_end(best)
        return best[0], self.keys[best]

    def demote(self, position, index):
        for key, value in self.keys.items():
            if value == index and key[0] == position:
                self.keys.move_to_end(key, last=False)
                return

    def save(self, slot, key):
        if key in self.keys:
            self.keys.move_to_end(key)
            return self.keys[key]
        if self.free:
            index = self.free.pop()
        else:
            _, index = self.keys.popitem(last=False)
            self.generator.counters["prefix_snapshot_evictions"] += 1
        self._copy_in(slot, index)
        self.keys[key] = index
        self.generator.counters["prefix_snapshot_saves"] += 1
        return index

    def restore(self, slot, index):
        self._copy_out(slot, index)
        self.generator.counters["prefix_snapshot_restores"] += 1

    def _copy_in(self, slot, index):
        for layer, buffers in zip(self.linear, self.entries[index]):
            state = self.generator.cache.layers[layer]
            for name, buffer in buffers.items():
                ttnn.copy(getattr(state, name)[slot : slot + 1], buffer)

    def _copy_out(self, slot, index):
        for layer, buffers in zip(self.linear, self.entries[index]):
            state = self.generator.cache.layers[layer]
            for name, buffer in buffers.items():
                tensor = getattr(state, name)
                if self.batch_size == 1:
                    ttnn.copy(buffer, tensor)
                    continue
                parts = [tensor[row : row + 1] for row in range(self.batch_size)]
                parts[slot] = buffer
                ttnn.copy(ttnn.concat(parts, dim=0), tensor)
