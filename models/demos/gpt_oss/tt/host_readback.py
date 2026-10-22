# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class DecodeHostRows:
    tensor: ttnn.Tensor
    batch_per_row: int
    row_start: int
    sample_rows: tuple[int, ...]
    device_tensor: ttnn.Tensor


class DecodeHostReadback:
    """Prepare bounded row-major reads before any model trace exists.

    Slice bounds are part of the program-cache key. Use aligned power-of-two
    ranges, clipped at the batch boundary, and compile every range at model
    construction. Share one persistent destination per output height. At batch
    32 this needs 62 programs and buffers totaling 31 rows per device.

    Slices and CPU reads use command queue 0. A later slice cannot overwrite a
    shared destination before an earlier read on that queue finishes. Each read
    allocates its own host tensor; callers must wait on its event before use.
    """

    def __init__(self, mesh_device, batch_per_row, vocab_per_device, num_rows):
        self.batch_per_row = batch_per_row
        self.num_rows = num_rows
        self.shape = (1, 1, batch_per_row, vocab_per_device)
        self.buffers = {}
        self.ranges = set()
        width = 1
        while width < batch_per_row:
            for start in range(0, batch_per_row, width):
                self.ranges.add((start, min(start + width, batch_per_row)))
            width *= 2
        if not self.ranges:
            return

        allocation = dict(
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        source = ttnn.empty(self.shape, **allocation)
        for start, end in sorted(self.ranges):
            height = end - start
            if height not in self.buffers:
                self.buffers[height] = ttnn.empty((1, 1, height, vocab_per_device), **allocation)
            # Warm the preallocated-output variant: it has a different cache key
            # from a slice that allocates its own destination.
            ttnn.slice(
                source,
                (0, 0, start, 0),
                (1, 1, end, vocab_per_device),
                output_tensor=self.buffers[height],
            )
        ttnn.synchronize_device(mesh_device)
        source.deallocate()

    def read(self, logits, sample_rows, blocking=True):
        capacity = self.batch_per_row * self.num_rows
        if (
            not sample_rows
            or len(set(sample_rows)) != len(sample_rows)
            or any(row < 0 or row >= capacity for row in sample_rows)
        ):
            raise ValueError(f"Invalid selective readback rows {sample_rows} for capacity {capacity}")
        # Refuse a different spec instead of compiling another slice variant
        # behind a live trace. GPT-OSS untilize returns BF16 DRAM logits.
        if (
            tuple(logits.shape) != self.shape
            or tuple(logits.padded_shape) != self.shape
            or logits.dtype != ttnn.bfloat16
            or logits.layout != ttnn.ROW_MAJOR_LAYOUT
            or logits.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError(f"Selective readback requires row-major BF16 DRAM logits with shape {self.shape}")

        local_rows = [row % self.batch_per_row for row in sample_rows]
        low, high = min(local_rows), max(local_rows)
        width = 1 << (low ^ high).bit_length()
        start = (low // width) * width
        end = min(start + width, self.batch_per_row)
        if start != 0 or end != self.batch_per_row:
            if (start, end) not in self.ranges:
                raise ValueError(f"Readback range {(start, end)} was not prepared before trace capture")
            logits = ttnn.slice(
                logits,
                (0, 0, start, 0),
                (1, 1, end, self.shape[-1]),
                output_tensor=self.buffers[end - start],
            )
        return DecodeHostRows(
            logits.cpu(blocking=blocking, cq_id=0), self.batch_per_row, start, tuple(sample_rows), logits
        )
