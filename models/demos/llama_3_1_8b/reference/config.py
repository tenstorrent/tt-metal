# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B-Instruct dimension constants.

Everything here is readable from the checkpoint's ``config.json`` and is therefore NOT in the
prefill spec (the spec carries only mesh / parallelism / dataformats / PCC). The values are
duplicated as class constants so the package imports without a checkpoint, and
``tests/torch_ref/test_reference_llama.py`` asserts every one of them against the vendored
``configs/config.json`` (itself a byte copy of the checkpoint's) so the two cannot drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_JSON = Path(__file__).resolve().parents[1] / "configs" / "config.json"


@dataclass
class LlamaConfig:
    """Dense GQA decoder-only transformer. Field names match HF ``LlamaConfig``."""

    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    # llama3 rope rescaling; the three factors and the original context length.
    rope_scaling: dict = field(
        default_factory=lambda: {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    )

    @property
    def head_dim(self) -> int:
        """Llama-3.1 does not set head_dim independently; it is hidden / n_heads = 128."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path=CONFIG_JSON) -> "LlamaConfig":
        """Build from a checkpoint ``config.json`` (or the vendored copy)."""
        with open(path) as f:
            raw = json.load(f)
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            hidden_act=raw["hidden_act"],
            attention_bias=raw["attention_bias"],
            mlp_bias=raw["mlp_bias"],
            tie_word_embeddings=raw["tie_word_embeddings"],
            rope_scaling=raw["rope_scaling"],
        )

    def reduced(self, *, num_hidden_layers=None, hidden_size=None, vocab_size=None) -> "LlamaConfig":
        """A dimension-reduced copy for host-side diagnostics.

        Reduced runs are NEVER a result (recipe §4): every number produced with one of these must be
        labelled reduced wherever it is quoted. Head counts are kept so head_dim stays 128 unless
        hidden_size is overridden, in which case the head count scales with it.
        """
        cfg = LlamaConfig(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        if num_hidden_layers is not None:
            cfg.num_hidden_layers = num_hidden_layers
        if hidden_size is not None:
            assert hidden_size % self.head_dim == 0, "reduced hidden must stay a multiple of head_dim"
            cfg.num_attention_heads = hidden_size // self.head_dim
            cfg.num_key_value_heads = max(1, cfg.num_attention_heads // self.num_key_value_groups)
            cfg.hidden_size = hidden_size
            cfg.intermediate_size = self.intermediate_size * hidden_size // self.hidden_size
        if vocab_size is not None:
            cfg.vocab_size = vocab_size
        return cfg
