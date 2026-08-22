# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace-stable runtime inputs for Laguna resumed prefill.

The values in these tensors change for every request, but their shapes and device
buffers are fixed during prefill warmup.  Keeping the absolute RoPE positions and
chunked-SDPA starts as data prevents arbitrary prefix lengths from becoming TTNN
program-cache keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PrefillRuntimeOffsets:
    """Persistent device inputs and outputs for one bucketed prefill geometry.

    ``position_ids`` and ``chunk_start_idxs`` are refreshed in place before each
    request. ``rope_outputs`` contains preallocated ``(cos, sin)`` output pairs,
    keyed by attention kind, so indexed RoPE can be hoisted once per chunk and
    reused by every layer without allocating under the resident decode trace.
    """

    bucket_len: int
    chunk_offsets: tuple[int, ...]
    chunk_lengths: tuple[int, ...]
    position_ids: tuple[Any, ...]
    chunk_start_idxs: tuple[Any, ...]
    rope_outputs: Mapping[str, tuple[tuple[Any, Any], ...]]

    def __post_init__(self):
        n = len(self.chunk_lengths)
        if self.bucket_len <= 0:
            raise ValueError(f"prefill bucket length must be positive, got {self.bucket_len}")
        if not n:
            raise ValueError("prefill runtime must contain at least one chunk")
        if not (
            len(self.chunk_offsets) == n
            and len(self.position_ids) == n
            and len(self.chunk_start_idxs) == n
        ):
            raise ValueError("prefill runtime chunk metadata and tensors must have equal lengths")
        if self.chunk_offsets[0] != 0:
            raise ValueError(f"prefill runtime first chunk must start at zero, got {self.chunk_offsets[0]}")
        expected = 0
        for i, (offset, length) in enumerate(zip(self.chunk_offsets, self.chunk_lengths)):
            if length <= 0:
                raise ValueError(f"prefill runtime chunk {i} has non-positive length {length}")
            if offset != expected:
                raise ValueError(
                    f"prefill runtime chunk {i} starts at {offset}, expected contiguous offset {expected}"
                )
            expected += length
        if expected != self.bucket_len:
            raise ValueError(
                f"prefill runtime chunks cover {expected} tokens, expected bucket length {self.bucket_len}"
            )
        for kind, outputs in self.rope_outputs.items():
            if len(outputs) != n:
                raise ValueError(
                    f"prefill runtime RoPE kind {kind!r} has {len(outputs)} chunks, expected {n}"
                )


def prefill_chunk_plan(bucket_len: int, *, pipe_threshold: int, outer_chunk: int, block_size: int):
    """Return contiguous ``(relative_start, length)`` chunks for one warmed bucket.

    Single-shot buckets deliberately remain one chunk. Pipelined chunk starts are
    block aligned so the same plan is valid for paged fill and chunked SDPA.
    """

    bucket_len = int(bucket_len)
    pipe_threshold = int(pipe_threshold)
    outer_chunk = int(outer_chunk)
    block_size = int(block_size)
    if bucket_len <= 0:
        raise ValueError(f"prefill bucket length must be positive, got {bucket_len}")
    if pipe_threshold <= 0:
        raise ValueError(f"prefill pipe threshold must be positive, got {pipe_threshold}")
    if block_size <= 0:
        raise ValueError(f"prefill KV block size must be positive, got {block_size}")
    if bucket_len <= pipe_threshold:
        return ((0, bucket_len),)
    if outer_chunk < block_size or outer_chunk % block_size:
        raise ValueError(
            f"prefill outer chunk {outer_chunk} must be a positive multiple of block size {block_size}"
        )
    return tuple(
        (offset, min(outer_chunk, bucket_len - offset)) for offset in range(0, bucket_len, outer_chunk)
    )
