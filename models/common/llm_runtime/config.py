# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static configuration for the generic LLM execution runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import ttnn

TraceMode = Literal["none", "decode_only", "all"]


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
class LLMExecutorConfig:
    """The complete immutable static policy paired with one LLMExecutor."""

    trace: TraceConfig
    warmup: WarmupConfig
    paged_kv_cache: PagedKVCacheConfig
    device_sampling_enabled: bool

    def __post_init__(self) -> None:
        nested_configs = (
            ("trace", self.trace, TraceConfig),
            ("warmup", self.warmup, WarmupConfig),
            ("paged_kv_cache", self.paged_kv_cache, PagedKVCacheConfig),
        )
        for name, value, expected_type in nested_configs:
            if type(value) is not expected_type:
                raise TypeError(f"{name} must be exactly {expected_type.__name__}")
        if not isinstance(self.device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")
