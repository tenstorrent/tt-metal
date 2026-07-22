# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3 B1 model (host interface).

Orchestrates prefill (token-by-token prompt processing) and autoregressive decode
via injectable write/read callables. Caller provides write_fn(token_tensor) and
read_fn(output_tensor); e.g. Pipeline.write_token / Pipeline.read_output.

The on-device metadata struct (DeepseekMetadata, 512 bytes) carries input fields,
output tokens, and sampling results. See metadata.hpp / metadata.py for the
authoritative layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.metadata.metadata import (
    MAX_MTP_LEVELS,
    METADATA_P_INDICES_CAPACITY,
    METADATA_P_SCORES_CAPACITY,
    METADATA_Q_INDICES_CAPACITY,
    METADATA_Q_SCORES_CAPACITY,
    NUM_OUTPUT_TOKENS,
    DeepseekMetadata,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32

TOKEN_ID_BYTES: int = 4
PCIE_PAGE_ALIGNMENT_BYTES: int = DeepseekMetadata.aligned_size_bytes()


# ---------------------------------------------------------------------------
# Field indices into the 128-word (512 B) DeepseekMetadata struct.
# Must stay in sync with metadata.hpp.
# ---------------------------------------------------------------------------


class Field:
    """uint32 word indices into the DeepseekMetadata struct."""

    LANE_ID = 0
    SLOT_ID = 1
    TOKEN_ID = 2
    POSITION_ID = 3
    OUTPUT_TOKENS = 4  # words 4..8  (5 slots: base + 4 spec)
    PREFILL_TOKENS = 9  # words 9..12 (4 slots, one per MTP level)
    TEMPERATURE = 13  # float stored as uint32 bits
    TOP_K = 14
    TOP_P = 15  # float stored as uint32 bits

    P_INDICES = 16  # words 16..47  (32 uint32)
    P_SCORES = 48  # words 48..63  (32 bf16 packed two-per-uint32)
    Q_INDICES = 64  # words 64..95  (32 uint32)
    Q_SCORES = 96  # words 96..111 (32 bf16 packed two-per-uint32)


# ---------------------------------------------------------------------------
# Decode result
# ---------------------------------------------------------------------------


@dataclass
class DecodeResult:
    """Parsed output page from the pipeline (512-byte DeepseekMetadata struct).

    ``output_tokens[0]`` is the base prediction; ``output_tokens[1..N]`` are
    speculative MTP predictions.  Positions are derived from ``position_id``:
    base is at ``position_id``, spec level *i* is at ``position_id + 1 + i``.
    """

    output_tokens: list[int]
    position_id: int
    lane_id: int = 0
    slot_id: int = 0
    p_indices: list[int] | None = None
    p_scores: list[float] | None = None
    q_indices: list[int] | None = None
    q_scores: list[float] | None = None

    @property
    def base_token(self) -> int:
        return self.output_tokens[0]

    def spec_token(self, level: int) -> int:
        """Return the speculative token at MTP *level* (0-indexed)."""
        return self.output_tokens[1 + level]

    def spec_position(self, level: int) -> int:
        """Return the position for the speculative token at MTP *level*."""
        return self.position_id + 1 + level


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _unpack_bf16_scores(raw: torch.Tensor, start_word: int, count: int) -> list[float]:
    """Unpack *count* bf16 values packed two-per-uint32 starting at *start_word*."""
    num_words = (count + 1) // 2
    packed = raw[start_word : start_word + num_words].contiguous().view(torch.bfloat16)
    return packed[:count].float().tolist()


def parse_output_page(output_buffer: ttnn.Tensor) -> DecodeResult:
    """Parse a 512-byte DeepseekMetadata output page into a :class:`DecodeResult`."""
    raw = ttnn.to_torch(output_buffer).to(torch.int32).flatten()

    output_tokens = [int(raw[Field.OUTPUT_TOKENS + i].item()) for i in range(NUM_OUTPUT_TOKENS)]

    p_indices = raw[Field.P_INDICES : Field.P_INDICES + METADATA_P_INDICES_CAPACITY].tolist()
    p_scores = _unpack_bf16_scores(raw, Field.P_SCORES, METADATA_P_SCORES_CAPACITY)
    q_indices = raw[Field.Q_INDICES : Field.Q_INDICES + METADATA_Q_INDICES_CAPACITY].tolist()
    q_scores = _unpack_bf16_scores(raw, Field.Q_SCORES, METADATA_Q_SCORES_CAPACITY)

    return DecodeResult(
        output_tokens=output_tokens,
        position_id=int(raw[Field.POSITION_ID].item()),
        lane_id=int(raw[Field.LANE_ID].item()),
        slot_id=int(raw[Field.SLOT_ID].item()),
        p_indices=p_indices,
        p_scores=p_scores,
        q_indices=q_indices,
        q_scores=q_scores,
    )


# ---------------------------------------------------------------------------
# Input building
# ---------------------------------------------------------------------------


def to_spec_input(
    token_id: int,
    *,
    slot_id: int = 0,
    position_id: int = 0,
    page_size_datums: int,
    lane_id: int = 0,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    prefill_token_ids: list[int] | None = None,
) -> ttnn.Tensor:
    """Build a PCIe-aligned input page matching the DeepseekMetadata struct.

    ``prefill_token_ids`` supplies ground-truth next tokens for each MTP level
    during prefill (up to :data:`MAX_MTP_LEVELS` entries).  Set to ``None`` or
    pass ``-1`` entries for decode mode; the kernel uses ``(uint32_t)-1`` as
    the "no ground truth" sentinel and falls back to the just-sampled argmax.
    """
    page = torch.zeros(1, page_size_datums, dtype=torch.int32)
    page[0, Field.LANE_ID] = lane_id
    page[0, Field.SLOT_ID] = slot_id
    page[0, Field.TOKEN_ID] = token_id
    page[0, Field.POSITION_ID] = position_id
    for i in range(MAX_MTP_LEVELS):
        page[0, Field.PREFILL_TOKENS + i] = -1
    if prefill_token_ids:
        for i, ptid in enumerate(prefill_token_ids[:MAX_MTP_LEVELS]):
            page[0, Field.PREFILL_TOKENS + i] = ptid
    page[0, Field.TEMPERATURE] = float_to_uint32(temperature)
    page[0, Field.TOP_K] = top_k
    page[0, Field.TOP_P] = float_to_uint32(top_p)
    return ttnn.from_torch(page, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def align_up(value: int, alignment: int) -> int:
    """Round value up to the next multiple of alignment."""
    return (value + alignment - 1) // alignment * alignment


def page_size_bytes(batch_size: int) -> int:
    """PCIe-aligned page (and FIFO) size in bytes for (B, 1) token IDs."""
    return align_up(batch_size * TOKEN_ID_BYTES, PCIE_PAGE_ALIGNMENT_BYTES)


def create_output_buffer(page_size_datums: int) -> ttnn.Tensor:
    """Allocate a host output tensor (1, page_size_datums) uint32 for socket read_tensor."""
    return ttnn.from_torch(
        torch.zeros(1, page_size_datums, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def to_padded_input(
    token: torch.Tensor | ttnn.Tensor,
    batch_size: int,
    page_size_datums: int,
) -> ttnn.Tensor:
    """Copy (B, 1) token into a PCIe-aligned padded buffer for write_tensor."""
    if isinstance(token, ttnn.Tensor):
        token = ttnn.to_torch(token)
    torch_padded = torch.zeros(1, page_size_datums, dtype=torch.int32)
    torch_padded[0, :batch_size] = token.flatten()[:batch_size]
    return ttnn.from_torch(torch_padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class DeepSeekV3:
    """Host-side model interface for prefill and decode via injectable write/read."""

    def __init__(
        self,
        write_fn: Callable[[ttnn.Tensor], None],
        read_fn: Callable[[ttnn.Tensor], None],
        batch_size: int = 1,
        pipeline_depth: int = 1,
    ) -> None:
        if batch_size != 1:
            raise ValueError(f"DeepSeekV3 currently supports only batch_size=1, got {batch_size}")
        if pipeline_depth <= 0:
            raise ValueError(f"pipeline_depth must be > 0, got {pipeline_depth}")
        self._write_fn = write_fn
        self._read_fn = read_fn
        self.batch_size = batch_size
        self._pipeline_depth = pipeline_depth
        payload_bytes: int = batch_size * TOKEN_ID_BYTES
        self._tensor_size_bytes: int = align_up(payload_bytes, PCIE_PAGE_ALIGNMENT_BYTES)
        self._page_size_datums: int = self._tensor_size_bytes // TOKEN_ID_BYTES
        self._position: int = 0
        self._output_buffer: ttnn.Tensor = create_output_buffer(self._page_size_datums)
        logger.debug(f"Creating DeepSeekV3 model with batch size {batch_size}")

    def prefill(self, prompt_tokens: list[ttnn.Tensor]) -> list[DecodeResult]:
        """Prefill-by-decode with overlapped I/O.

        Returns the DecodeResult(s) from the final prompt token.
        """
        if len(prompt_tokens) == 0:
            raise ValueError("Expected at least one prompt token")

        num_writes_before_readback = min(self._pipeline_depth, len(prompt_tokens))
        total_reads = len(prompt_tokens)

        write_idx = 0
        read_count = 0

        while write_idx < num_writes_before_readback:
            self._write_fn(prompt_tokens[write_idx])
            write_idx += 1

        while write_idx < len(prompt_tokens):
            self._read_fn(self._output_buffer)
            read_count += 1
            self._write_fn(prompt_tokens[write_idx])
            write_idx += 1

        last_results: list[DecodeResult] = []
        while read_count < total_reads:
            self._read_fn(self._output_buffer)
            read_count += 1
            if read_count > total_reads - 1:
                last_results.append(parse_output_page(self._output_buffer))

        self._position += len(prompt_tokens)
        return last_results

    def write_input(
        self,
        token_id: int,
        *,
        slot_id: int = 0,
        position_id: int = 0,
        lane_id: int = 0,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        prefill_token_ids: list[int] | None = None,
    ) -> None:
        """Write a single input page to the pipeline."""
        input_tensor = to_spec_input(
            token_id,
            slot_id=slot_id,
            position_id=position_id,
            page_size_datums=self._page_size_datums,
            lane_id=lane_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefill_token_ids=prefill_token_ids,
        )
        self._write_fn(input_tensor)

    def read_result(self) -> DecodeResult:
        """Read one output page from the pipeline and return the parsed DecodeResult."""
        self._read_fn(self._output_buffer)
        return parse_output_page(self._output_buffer)

    @property
    def position(self) -> int:
        """Current sequence position (number of tokens processed so far)."""
        return self._position
