# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Static memory helpers for DiffusionGemma bring-up budgets.

The estimates in this module are scoped to per-step canvas K/V scratch tensors.
They are not a total model working-set estimate: prompt-prefix K/V, weights,
paged cache allocation, activations, and SDPA concatenation buffers are measured
separately by the QB2 memory probes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TextConfig


@dataclass(frozen=True)
class CanvasKVScratchEstimate:
    """Per-chip K/V scratch bytes for one denoise canvas step."""

    sliding_bytes: int
    full_attention_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.sliding_bytes + self.full_attention_bytes


def _full_attention_layers(config: TextConfig) -> int:
    return config.num_hidden_layers // config.sliding_window_pattern


def _local_kv_heads(num_kv_heads: int, tp: int) -> int:
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    # Match Gemma4 split_qkv/cache allocation: if KV heads < TP, each device
    # gets one assigned KV head; otherwise KV heads are TP-sharded.
    return 1 if num_kv_heads < tp else num_kv_heads // tp


def estimate_canvas_kv_scratch_bytes(
    config: TextConfig | None = None,
    *,
    tp: int = 4,
    batch_size: int = 1,
    bytes_per_elem: int = 2,
) -> CanvasKVScratchEstimate:
    """Estimate per-chip scratch for per-step canvas K/V tensors.

    The current Gemma4 attention path materializes separate K and V tensors even
    for full-attention layers whose weights are K=V tied, so this counts both.
    This intentionally counts only the denoise canvas K/V side; prompt-prefix K/V
    concatenated by denoise SDPA is outside this helper's scope.
    """

    if config is None:
        config = TextConfig()
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if bytes_per_elem < 1:
        raise ValueError(f"bytes_per_elem must be >= 1, got {bytes_per_elem}")

    full_layers = _full_attention_layers(config)
    sliding_layers = config.num_hidden_layers - full_layers
    canvas = config.canvas_length
    kv_tensors = 2  # K and V are separate device tensors in the current TT path.

    sliding_bytes = (
        sliding_layers
        * batch_size
        * kv_tensors
        * _local_kv_heads(config.num_key_value_heads, tp)
        * canvas
        * config.head_dim
        * bytes_per_elem
    )
    full_attention_bytes = (
        full_layers
        * batch_size
        * kv_tensors
        * _local_kv_heads(config.num_global_key_value_heads, tp)
        * canvas
        * config.global_head_dim
        * bytes_per_elem
    )
    return CanvasKVScratchEstimate(sliding_bytes=sliding_bytes, full_attention_bytes=full_attention_bytes)
