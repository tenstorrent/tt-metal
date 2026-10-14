# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure prefill request planning values and transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

_SUPPORTED_PREFILL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
_MAX_BATCHED_PREFILL_TOKENS = 128 * 1024
_PAGE_TABLE_WIDTH_ALIGNMENT = 8

PrefillKind = Literal["single", "batched"]


@dataclass(frozen=True)
class PrefillChunk:
    """One planned invocation over a slice of a request's uncached tokens."""

    token_slice: slice
    chunk_start_idx: int
    chunk_size: int
    chunk_page_table: torch.Tensor | None
    contains_last_token: bool

    def __post_init__(self) -> None:
        if self.token_slice.step not in (None, 1):
            raise ValueError("prefill chunk slices must be contiguous")
        if self.token_slice.start is None or self.token_slice.stop is None:
            raise ValueError("prefill chunk slices must have explicit bounds")
        if self.token_slice.start < 0 or self.token_slice.stop <= self.token_slice.start:
            raise ValueError("prefill chunk slices must be non-empty and nonnegative")
        if self.token_slice.stop - self.token_slice.start != self.chunk_size:
            raise ValueError("prefill chunk slice and chunk_size disagree")
        if self.chunk_start_idx < 0 or self.chunk_size <= 0:
            raise ValueError("prefill chunk positions must be nonnegative and non-empty")
        if self.chunk_page_table is not None:
            if not isinstance(self.chunk_page_table, torch.Tensor) or self.chunk_page_table.ndim != 2:
                raise ValueError("chunk_page_table must be a rank-2 torch.Tensor")


@dataclass(frozen=True)
class PrefillRequest:
    """One immutable planned prefill unit with all chunk decisions retained."""

    kind: PrefillKind
    source_rows: tuple[int, ...]
    slots: tuple[int, ...]
    tokens: torch.Tensor
    page_table: torch.Tensor
    prompt_lengths: tuple[int, ...]
    cached_tokens: tuple[int, ...]
    last_token_indices: tuple[int, ...]
    padded_sequence_length: int
    padded_batch_size: int
    chunks: tuple[PrefillChunk, ...]
    uses_chunked_prefill: bool

    def __post_init__(self) -> None:
        row_count = len(self.source_rows)
        if self.kind not in ("single", "batched"):
            raise ValueError(f"unsupported prefill request kind {self.kind!r}")
        if row_count == 0 or not (
            len(self.slots)
            == len(self.prompt_lengths)
            == len(self.cached_tokens)
            == len(self.last_token_indices)
            == row_count
        ):
            raise ValueError("prefill request row metadata must be non-empty and aligned")
        if self.kind == "single" and row_count != 1:
            raise ValueError("single prefill requests must describe exactly one source row")
        if not isinstance(self.tokens, torch.Tensor) or self.tokens.ndim != 2:
            raise ValueError("planned prefill tokens must be a rank-2 torch.Tensor")
        if not isinstance(self.page_table, torch.Tensor) or self.page_table.ndim != 2:
            raise ValueError("planned prefill page_table must be a rank-2 torch.Tensor")
        if int(self.tokens.shape[0]) != self.padded_batch_size:
            raise ValueError("planned token batch does not match padded_batch_size")
        if int(self.tokens.shape[1]) != self.padded_sequence_length:
            raise ValueError("planned token width does not match padded_sequence_length")
        if int(self.page_table.shape[0]) != self.padded_batch_size:
            raise ValueError("planned page-table batch does not match padded_batch_size")
        if not self.chunks:
            raise ValueError("a prefill request must contain at least one planned chunk")
        if sum(chunk.contains_last_token for chunk in self.chunks) != 1:
            raise ValueError("exactly one planned chunk must contain the actual last token")
        if not self.chunks[-1].contains_last_token:
            raise ValueError("planning must stop at the chunk containing the actual last token")
        if self.uses_chunked_prefill != any(chunk.chunk_page_table is not None for chunk in self.chunks):
            raise ValueError("chunked-prefill classification disagrees with planned chunks")

    @property
    def page_table_width(self) -> int:
        return int(self.page_table.shape[-1])


def _plan_prefill_requests(
    *,
    tokens: torch.Tensor,
    page_table: torch.Tensor,
    prompt_lens: torch.Tensor | None,
    empty_slots: Sequence[int] | None,
    start_pos: torch.Tensor | None,
    block_size: int,
    max_batch_size: int,
    max_prefill_chunk_size: int,
    supports_batched_prefill: bool | None = None,
    disable_batched_prefill: bool = False,
    max_prefill_batch_size: int = 8,
    max_actual_page_table_width: int | None = None,
    canonical_page_table_width: int | None = None,
) -> tuple[PrefillRequest, ...]:
    """Plan prefix-caching and chunked-prefill semantics exactly once."""

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise ValueError("prefill tokens must be a rank-2 torch.Tensor")
    if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
        raise ValueError("prefill page_table must be a rank-2 torch.Tensor")
    batch_size, token_width = map(int, tokens.shape)
    if int(page_table.shape[0]) != batch_size:
        raise ValueError("prefill token and page-table batches must match")
    if prompt_lens is None:
        prompt_lens = torch.full((batch_size,), token_width, dtype=torch.long)
    if not isinstance(prompt_lens, torch.Tensor) or prompt_lens.ndim != 1:
        raise ValueError("prompt_lens must be a rank-1 torch.Tensor")
    if int(prompt_lens.shape[0]) != batch_size:
        raise ValueError("prompt_lens batch must match tokens")
    if start_pos is None:
        start_pos = torch.zeros(batch_size, dtype=torch.long)
    if not isinstance(start_pos, torch.Tensor) or start_pos.ndim != 1:
        raise ValueError("start_pos must be a rank-1 torch.Tensor")
    if int(start_pos.shape[0]) != batch_size:
        raise ValueError("start_pos batch must match tokens")

    slots = list(range(batch_size)) if empty_slots is None else [int(slot) for slot in empty_slots]
    if len(slots) != batch_size:
        raise ValueError("empty_slots length must match prefill batch")
    if len(set(slots)) != len(slots) or any(slot < 0 or slot >= max_batch_size for slot in slots):
        raise ValueError("empty_slots must contain unique lane-local slots")
    if (max_actual_page_table_width is None) != (canonical_page_table_width is None):
        raise ValueError("canonical page-table widths must be provided together")
    if max_actual_page_table_width is not None:
        if max_actual_page_table_width <= 0 or canonical_page_table_width < max_actual_page_table_width:
            raise ValueError("invalid canonical page-table widths")
        if canonical_page_table_width % _PAGE_TABLE_WIDTH_ALIGNMENT:
            raise ValueError("canonical page-table width must be 8-entry aligned")

    lengths = [int(value) for value in prompt_lens]
    cached = [int(value) for value in start_pos]
    for row, (length, num_cached_tokens) in enumerate(zip(lengths, cached)):
        if num_cached_tokens < 0 or length < 0 or num_cached_tokens > length or length > token_width:
            raise ValueError(f"invalid prompt/cached-token lengths for prefill row {row}")
        if num_cached_tokens % block_size:
            raise ValueError(f"cached prefill start for row {row} must be block aligned")
    uncached_lengths = [length - num_cached_tokens for length, num_cached_tokens in zip(lengths, cached)]
    padded_lengths = [_padded_prefill_length(length) if length > 0 else 0 for length in uncached_lengths]

    batched_requests = []
    sequential_rows = []
    buckets: dict[int, list[int]] = {}
    for source_row, uncached_length in enumerate(uncached_lengths):
        if uncached_length > 0:
            buckets.setdefault(padded_lengths[source_row], []).append(source_row)
    legacy_implicit_batching = supports_batched_prefill is None
    for sequence_length, source_rows in buckets.items():
        if legacy_implicit_batching:
            padded_batch = _legacy_batched_prefill_size(
                len(source_rows),
                sequence_length,
                [cached[source_row] for source_row in source_rows],
                disabled=(
                    disable_batched_prefill
                    or len(buckets) != 1
                    or slots != list(range(batch_size))
                    or any(length <= 0 for length in uncached_lengths)
                ),
                max_batch_size=max_batch_size,
                max_prefill_chunk_size=max_prefill_chunk_size,
            )
        else:
            padded_batch = _batched_prefill_size(
                len(source_rows),
                sequence_length,
                [cached[source_row] for source_row in source_rows],
                supports_batched_prefill=supports_batched_prefill,
                disable_batched_prefill=disable_batched_prefill,
                max_batch_size=max_batch_size,
                max_prefill_batch_size=max_prefill_batch_size,
                max_prefill_chunk_size=max_prefill_chunk_size,
            )
        if padded_batch is None:
            sequential_rows.extend(source_rows)
            continue
        batched_requests.append(
            _make_batched_request(
                tokens=tokens,
                page_table=page_table,
                lengths=lengths,
                cached=cached,
                slots=slots,
                source_rows=source_rows,
                padded_batch=padded_batch,
                sequence_length=sequence_length,
                block_size=block_size,
                max_actual_page_table_width=max_actual_page_table_width,
                canonical_page_table_width=canonical_page_table_width,
            )
        )

    requests = list(batched_requests)
    sequential_rows.sort()
    for source_row in sequential_rows:
        slot = slots[source_row]
        uncached_length = uncached_lengths[source_row]
        # Gate 1 remains intentionally behavior-preserving until its public
        # cache-hit output contract is decided.
        if uncached_length <= 0:
            continue
        sequence_length = padded_lengths[source_row]
        request_tokens = torch.zeros((1, sequence_length), dtype=torch.long, device=tokens.device)
        request_tokens[0, :uncached_length] = tokens[
            source_row,
            cached[source_row] : lengths[source_row],
        ]
        actual_width = _num_blocks(lengths[source_row], block_size)
        page_width = canonical_page_table_width or _num_blocks(sequence_length + cached[source_row], block_size)
        _validate_page_table_width(
            actual_width,
            page_table,
            max_actual_page_table_width,
            f"prefill row {source_row}",
        )
        if cached[source_row] + sequence_length > page_width * block_size:
            raise ValueError(f"padded prefill row {source_row} exceeds the canonical page-table capacity")
        uses_chunked_prefill = sequence_length > max_prefill_chunk_size or cached[source_row] > 0
        request_page_table = torch.full(
            (1, page_width),
            0 if uses_chunked_prefill else -1,
            dtype=torch.int32,
            device=page_table.device,
        )
        request_page_table[0, :actual_width] = page_table[source_row, :actual_width].to(torch.int32)
        chunks = _plan_chunks(
            padded_sequence_length=sequence_length,
            actual_uncached_length=uncached_length,
            num_cached_tokens=cached[source_row],
            prompt_length=lengths[source_row],
            page_table=request_page_table,
            block_size=block_size,
            max_prefill_chunk_size=max_prefill_chunk_size,
            uses_chunked_prefill=uses_chunked_prefill,
        )
        requests.append(
            PrefillRequest(
                kind="single",
                source_rows=(source_row,),
                slots=(slot,),
                tokens=request_tokens,
                page_table=request_page_table,
                prompt_lengths=(lengths[source_row],),
                cached_tokens=(cached[source_row],),
                last_token_indices=(lengths[source_row] - 1,),
                padded_sequence_length=sequence_length,
                padded_batch_size=1,
                chunks=chunks,
                uses_chunked_prefill=uses_chunked_prefill,
            )
        )
    return tuple(requests)


def _make_batched_request(
    *,
    tokens: torch.Tensor,
    page_table: torch.Tensor,
    lengths: list[int],
    cached: list[int],
    slots: list[int],
    source_rows: list[int],
    padded_batch: int,
    sequence_length: int,
    block_size: int,
    max_actual_page_table_width: int | None,
    canonical_page_table_width: int | None,
) -> PrefillRequest:
    request_tokens = torch.zeros((padded_batch, sequence_length), dtype=torch.long, device=tokens.device)
    page_width = canonical_page_table_width or _num_blocks(sequence_length, block_size)
    # -1 is the paged-fill skip sentinel. Leaving padding rows and unused active
    # tails at -1 prevents stale vLLM row tails from writing reassigned blocks;
    # only each prompt's actually allocated prefix is safe to copy.
    request_page_table = torch.full(
        (padded_batch, page_width),
        -1,
        dtype=torch.int32,
        device=page_table.device,
    )
    for local_row, source_row in enumerate(source_rows):
        length = lengths[source_row]
        actual_width = _num_blocks(length, block_size)
        _validate_page_table_width(
            actual_width,
            page_table,
            max_actual_page_table_width,
            f"batched prefill row {source_row}",
        )
        request_tokens[local_row, :length] = tokens[source_row, :length]
        request_page_table[local_row, :actual_width] = page_table[source_row, :actual_width].to(torch.int32)
    chunk = PrefillChunk(
        token_slice=slice(0, sequence_length),
        chunk_start_idx=0,
        chunk_size=sequence_length,
        chunk_page_table=None,
        contains_last_token=True,
    )
    return PrefillRequest(
        kind="batched",
        source_rows=tuple(source_rows),
        slots=tuple(slots[source_row] for source_row in source_rows),
        tokens=request_tokens,
        page_table=request_page_table,
        prompt_lengths=tuple(lengths[source_row] for source_row in source_rows),
        cached_tokens=tuple(cached[source_row] for source_row in source_rows),
        last_token_indices=tuple(lengths[source_row] - 1 for source_row in source_rows),
        padded_sequence_length=sequence_length,
        padded_batch_size=padded_batch,
        chunks=(chunk,),
        uses_chunked_prefill=False,
    )


def _plan_chunks(
    *,
    padded_sequence_length: int,
    actual_uncached_length: int,
    num_cached_tokens: int,
    prompt_length: int,
    page_table: torch.Tensor,
    block_size: int,
    max_prefill_chunk_size: int,
    uses_chunked_prefill: bool,
) -> tuple[PrefillChunk, ...]:
    if not uses_chunked_prefill:
        return (
            PrefillChunk(
                token_slice=slice(0, padded_sequence_length),
                chunk_start_idx=0,
                chunk_size=padded_sequence_length,
                chunk_page_table=None,
                contains_last_token=True,
            ),
        )

    chunk_size = (
        _max_prefill_chunk_size(padded_sequence_length, max_prefill_chunk_size)
        if padded_sequence_length > max_prefill_chunk_size
        else padded_sequence_length
    )
    relative_last = actual_uncached_length - 1
    chunks = []
    for relative_start in range(0, padded_sequence_length, chunk_size):
        absolute_start = num_cached_tokens + relative_start
        chunk_start_block = absolute_start // block_size
        chunk_width = _num_blocks(chunk_size, block_size)
        mapped_blocks = min(
            chunk_width,
            max(0, _num_blocks(prompt_length, block_size) - chunk_start_block),
        )
        # Chunked SDPA consumes the full request table, so that table keeps its
        # nonnegative zero filler. This fill-only view is skip-aware and may use
        # -1; copying just the mapped prefix also excludes stale scheduler tails.
        chunk_page_table = torch.full(
            (int(page_table.shape[0]), chunk_width),
            -1,
            dtype=torch.int32,
            device=page_table.device,
        )
        if mapped_blocks:
            chunk_page_table[:, :mapped_blocks] = page_table[:, chunk_start_block : chunk_start_block + mapped_blocks]
        contains_last_token = relative_start <= relative_last < relative_start + chunk_size
        chunks.append(
            PrefillChunk(
                token_slice=slice(relative_start, relative_start + chunk_size),
                chunk_start_idx=absolute_start,
                chunk_size=chunk_size,
                chunk_page_table=chunk_page_table,
                contains_last_token=contains_last_token,
            )
        )
        if contains_last_token:
            break
    return tuple(chunks)


def _validate_page_table_width(
    actual_width: int,
    page_table: torch.Tensor,
    max_actual_page_table_width: int | None,
    label: str,
) -> None:
    if max_actual_page_table_width is not None and actual_width > max_actual_page_table_width:
        raise ValueError(f"{label} exceeds the configured paged-KV capacity")
    if int(page_table.shape[-1]) < actual_width:
        raise ValueError(f"page table is too narrow for {label}")


def _padded_prefill_length(sequence_length: int) -> int:
    if sequence_length <= 128:
        return 128
    if sequence_length <= 1024:
        return 1024
    return 1 << (sequence_length - 1).bit_length()


def _batched_prefill_size(
    batch_size,
    sequence_length,
    cached_tokens,
    *,
    supports_batched_prefill,
    disable_batched_prefill,
    max_batch_size,
    max_prefill_batch_size,
    max_prefill_chunk_size,
):
    if batch_size <= 1 or not supports_batched_prefill or disable_batched_prefill:
        return None
    if any(value != 0 for value in cached_tokens):
        return None
    if sequence_length > max_prefill_chunk_size:
        return None
    padded_batch = _next_supported_prefill_batch_size(batch_size)
    if (
        padded_batch is None
        or padded_batch > max_prefill_batch_size
        or padded_batch > max_batch_size
        or padded_batch * sequence_length >= _MAX_BATCHED_PREFILL_TOKENS
    ):
        return None
    return padded_batch


def _legacy_batched_prefill_size(
    batch_size,
    sequence_length,
    cached_tokens,
    *,
    disabled,
    max_batch_size,
    max_prefill_chunk_size,
):
    """Preserve the pre-D9 planner for callers that omit the new policy."""

    if disabled or batch_size <= 1 or sequence_length != 128:
        return None
    if any(value != 0 for value in cached_tokens) or sequence_length > max_prefill_chunk_size:
        return None
    padded_batch = _next_supported_prefill_batch_size(batch_size)
    if padded_batch is not None and padded_batch > max_batch_size:
        padded_batch = None
    if padded_batch is None or padded_batch * sequence_length >= _MAX_BATCHED_PREFILL_TOKENS:
        return None
    return padded_batch


def _next_supported_prefill_batch_size(batch_size: int) -> int | None:
    """Return the supported physical size for one whole lane-local wave."""

    return next((candidate for candidate in _SUPPORTED_PREFILL_BATCH_SIZES if candidate >= batch_size), None)


def _max_prefill_chunk_size(sequence_length: int, maximum: int) -> int:
    minimum_chunk = 2048
    if sequence_length <= 0 or maximum <= 0:
        raise ValueError("prefill chunk lengths must be positive")
    if sequence_length % minimum_chunk or maximum % minimum_chunk:
        raise ValueError("prefill chunk lengths must be multiples of 2048")
    for chunk_size in range(min(sequence_length, maximum), 0, -minimum_chunk):
        if sequence_length % chunk_size == 0:
            return chunk_size
    raise ValueError("no valid prefill chunk size")


def _num_blocks(sequence_length: int, block_size: int) -> int:
    return math.ceil(int(sequence_length) / int(block_size))
