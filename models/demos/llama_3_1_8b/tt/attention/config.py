# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention configuration — the interface commitment D2 fixes and D3 implements against."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
    """Per-layer attention shape. Everything here is read from the checkpoint config, never guessed.

    Llama-3.1-8B is plain GQA: no QK-norm, no attention sinks, no sliding window, no partial rotary
    and no biases. Those fields are absent rather than defaulted to False, so a model that does have
    them cannot silently run through this path.
    """

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    max_seq_len: int
    # Sequence-parallel prefill: Q/K/V arrive as per-device sequence shards and attention is the ring
    # SDPA across the SP axis. False is the single-device (sp=1) unit-test path.
    sequence_parallel: bool = True

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def scale(self) -> float:
        return self.head_dim**-0.5

    @classmethod
    def from_model_config(cls, cfg, *, max_seq_len: int, sequence_parallel: bool = True) -> "AttentionConfig":
        return cls(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            max_seq_len=max_seq_len,
            sequence_parallel=sequence_parallel,
        )
