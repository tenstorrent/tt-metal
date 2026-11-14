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
from typing import Any, Iterable, Mapping


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
        if not (len(self.chunk_offsets) == n and len(self.position_ids) == n and len(self.chunk_start_idxs) == n):
            raise ValueError("prefill runtime chunk metadata and tensors must have equal lengths")
        if self.chunk_offsets[0] != 0:
            raise ValueError(f"prefill runtime first chunk must start at zero, got {self.chunk_offsets[0]}")
        expected = 0
        for i, (offset, length) in enumerate(zip(self.chunk_offsets, self.chunk_lengths)):
            if length <= 0:
                raise ValueError(f"prefill runtime chunk {i} has non-positive length {length}")
            if offset != expected:
                raise ValueError(f"prefill runtime chunk {i} starts at {offset}, expected contiguous offset {expected}")
            expected += length
        if expected != self.bucket_len:
            raise ValueError(
                f"prefill runtime chunks cover {expected} tokens, expected bucket length {self.bucket_len}"
            )
        for kind, outputs in self.rope_outputs.items():
            if len(outputs) != n:
                raise ValueError(f"prefill runtime RoPE kind {kind!r} has {len(outputs)} chunks, expected {n}")


@dataclass(frozen=True)
class PrefillStreamChunk:
    """One independently executed chunk in a ragged long-prefill plan.

    ``real_len`` rows come from the request and ``bucket_len - real_len`` rows
    are causally-safe right padding.  ``relative_start`` is relative to the
    scheduler-provided start position; callers retain the absolute position for
    RoPE, paged fill, and chunked SDPA.
    """

    relative_start: int
    real_len: int
    bucket_len: int

    def __post_init__(self):
        if self.relative_start < 0:
            raise ValueError(f"prefill stream chunk has negative start {self.relative_start}")
        if self.real_len <= 0:
            raise ValueError(f"prefill stream chunk has non-positive real length {self.real_len}")
        if self.bucket_len < self.real_len:
            raise ValueError(f"prefill stream bucket {self.bucket_len} is smaller than real length {self.real_len}")

    @property
    def relative_end(self) -> int:
        return self.relative_start + self.real_len


def prefill_stream_plan(
    real_len: int,
    *,
    bucket_lens: Iterable[int],
    outer_chunk: int,
    canonical_tail: bool = False,
) -> tuple[PrefillStreamChunk, ...]:
    """Plan chunk-major prefill with a selectable final-query geometry.

    Every complete ``outer_chunk`` is executed at exactly that shape.  The only
    padding is in the final partial chunk.  By default that tail rounds to the
    smallest warmed bucket.  ``canonical_tail=True`` instead pads it to
    ``outer_chunk`` so every query in a long stream uses the same SDPA reduction
    geometry.  Processing each returned chunk through the complete decoder
    stack is causally equivalent to processing one monolithic prompt: every
    layer's earlier-chunk K/V is resident before the next chunk reaches that
    layer.

    Keeping the finite bucket set at or below ``outer_chunk`` removes the
    power-of-two cliff for long prompts and means warmup need not compile or
    allocate hidden/selector geometry proportional to the model context.
    """

    real_len = int(real_len)
    outer_chunk = int(outer_chunk)
    if real_len <= 0:
        raise ValueError(f"prefill stream real length must be positive, got {real_len}")
    if outer_chunk <= 0:
        raise ValueError(f"prefill stream outer chunk must be positive, got {outer_chunk}")

    buckets = tuple(sorted({int(length) for length in bucket_lens}))
    if not buckets or buckets[0] <= 0:
        raise ValueError("prefill stream buckets must contain positive lengths")
    if buckets[-1] != outer_chunk:
        raise ValueError(f"prefill stream largest bucket {buckets[-1]} must equal outer chunk {outer_chunk}")

    chunks = []
    relative_start = 0
    remaining = real_len
    while remaining > outer_chunk:
        chunks.append(PrefillStreamChunk(relative_start, outer_chunk, outer_chunk))
        relative_start += outer_chunk
        remaining -= outer_chunk

    bucket = outer_chunk if canonical_tail else next((length for length in buckets if remaining <= length), None)
    if bucket is None:  # guarded by buckets[-1] == outer_chunk, retained for a clear failure mode
        raise ValueError(f"no warmed prefill bucket covers final {remaining}-token chunk")
    chunks.append(PrefillStreamChunk(relative_start, remaining, bucket))
    return tuple(chunks)


def streaming_prefill_capacity(max_model_len: int, *, outer_chunk: int) -> int:
    """Smallest RoPE/page-table horizon covering any padded streamed prompt.

    Only the final chunk can be padded, so the required horizon is the requested
    context rounded to one outer-chunk boundary—not ``context + context`` as in
    monolithic power-of-two bucketing.
    """

    max_model_len = int(max_model_len)
    outer_chunk = int(outer_chunk)
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {max_model_len}")
    if outer_chunk <= 0:
        raise ValueError(f"prefill stream outer chunk must be positive, got {outer_chunk}")
    return ((max_model_len + outer_chunk - 1) // outer_chunk) * outer_chunk


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
        raise ValueError(f"prefill outer chunk {outer_chunk} must be a positive multiple of block size {block_size}")
    return tuple((offset, min(outer_chunk, bucket_len - offset)) for offset in range(0, bucket_len, outer_chunk))
