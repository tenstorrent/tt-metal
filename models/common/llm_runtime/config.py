# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static configuration and resolved geometry for the LLM runtime toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import ttnn

TraceMode = Literal["none", "decode_only", "all"]

_PAGE_TABLE_WIDTH_ALIGNMENT = 8


@dataclass(frozen=True)
class TraceConfig:
    """Select which already-compiled graphs receive trace artifacts."""

    mode: TraceMode = "decode_only"

    def __post_init__(self) -> None:
        if self.mode not in ("none", "decode_only", "all"):
            raise ValueError(f"Unsupported trace mode: {self.mode!r}")

    @property
    def prefill_enabled(self) -> bool:
        return self.mode == "all"

    @property
    def decode_enabled(self) -> bool:
        return self.mode in ("decode_only", "all")


@dataclass(frozen=True)
class WarmupConfig:
    """Select graph coverage while retaining model-derived defaults.

    ``prefill_seq_lens=None`` asks the executor to use the model runtime's
    supported trace lengths, falling back to 128. Logits coverage is always
    required. Forced-top-k prefill and greedy decode coverage are added when
    dynamic warmup enables device sampling. Additional top-k decode coverage is
    explicitly opt in.
    """

    prefill_seq_lens: tuple[int, ...] | None = None
    prefill_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    include_decode_top_k: bool = False

    def __post_init__(self) -> None:
        if self.prefill_seq_lens is not None:
            self._validate_positive_tuple("prefill_seq_lens", self.prefill_seq_lens)
        self._validate_positive_tuple("prefill_batch_sizes", self.prefill_batch_sizes)

    @staticmethod
    def _validate_positive_tuple(name: str, values: tuple[int, ...]) -> None:
        if not isinstance(values, tuple):
            raise TypeError(f"{name} must be a tuple so WarmupConfig remains immutable")
        if not values:
            raise ValueError(f"{name} must not be empty")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError(f"{name} values must be positive integers")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} values must be unique")


@dataclass(frozen=True)
class PagedKVCacheConfig:
    """Static paged-KV policy plus an optional resolved physical capacity."""

    block_size: int
    max_num_blocks: int
    dtype: ttnn.DataType
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    num_blocks: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if (
            not isinstance(self.max_num_blocks, int)
            or isinstance(self.max_num_blocks, bool)
            or self.max_num_blocks <= 0
        ):
            raise ValueError("max_num_blocks must be a positive integer")
        if self.num_blocks is not None:
            if not isinstance(self.num_blocks, int) or isinstance(self.num_blocks, bool) or self.num_blocks <= 0:
                raise ValueError("num_blocks must be a positive integer when resolved")
            if self.num_blocks > self.max_num_blocks:
                raise ValueError(f"num_blocks ({self.num_blocks}) exceeds max_num_blocks ({self.max_num_blocks})")

    def is_resolved(self) -> bool:
        return self.num_blocks is not None

    @property
    def max_capacity_tokens(self) -> int:
        return self.block_size * self.max_num_blocks

    @property
    def capacity_tokens(self) -> int | None:
        if self.num_blocks is None:
            return None
        return self.block_size * self.num_blocks


@dataclass(frozen=True)
class PageTableLayout:
    """Resolved page-table geometry shared by prefill, decode, and warmup."""

    block_size: int
    raw_capacity_width: int
    prefill_width: int
    decode_width: int

    @classmethod
    def resolve(
        cls,
        *,
        block_size: int,
        model_max_sequence_length: int,
        physical_num_blocks: int,
        max_prefill_chunk_size: int,
    ) -> "PageTableLayout":
        values = {
            "block_size": block_size,
            "model_max_sequence_length": model_max_sequence_length,
            "physical_num_blocks": physical_num_blocks,
            "max_prefill_chunk_size": max_prefill_chunk_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        model_width = _ceil_div(model_max_sequence_length, block_size)
        raw_width = min(model_width, physical_num_blocks)
        decode_width = _round_up(raw_width, _PAGE_TABLE_WIDTH_ALIGNMENT)
        padding_blocks = _ceil_div(max_prefill_chunk_size - 1, block_size)
        prefill_width = _round_up(raw_width + padding_blocks, _PAGE_TABLE_WIDTH_ALIGNMENT)
        return cls(
            block_size=block_size,
            raw_capacity_width=raw_width,
            prefill_width=prefill_width,
            decode_width=decode_width,
        )

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.raw_capacity_width <= 0:
            raise ValueError("page-table block size and capacity width must be positive")
        if self.prefill_width < self.raw_capacity_width or self.decode_width < self.raw_capacity_width:
            raise ValueError("canonical page-table widths cannot be smaller than raw capacity")
        if self.prefill_width % _PAGE_TABLE_WIDTH_ALIGNMENT or self.decode_width % _PAGE_TABLE_WIDTH_ALIGNMENT:
            raise ValueError("canonical page-table widths must satisfy alignment")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment
