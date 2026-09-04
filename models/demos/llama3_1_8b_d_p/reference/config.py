# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B-Instruct config constants + loader.

The constants class is the single source of truth the TT modules read; ``tests/torch/
test_llama_reference.py`` asserts every field against the vendored ``configs/Llama-3.1-8B-Instruct/
config.json``, so a checkpoint bump that changes a dim fails loudly instead of silently mis-shaping
a weight.

Everything here is EVIDENCED by the prefill spec (``spec_llama31_8b_v0.json``) and the vendored HF
config, which agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "Llama-3.1-8B-Instruct"
CONFIG_JSON = CONFIG_DIR / "config.json"


class Llama31_8BConfig:
    """Frozen architecture constants for Llama 3.1 8B Instruct (spec ``architecture`` block)."""

    HIDDEN_SIZE = 4096
    NUM_LAYERS = 32
    VOCAB_SIZE = 128256
    RMS_NORM_EPS = 1e-5

    # Attention: GQA, 32 Q heads over 8 KV heads (group 4), full rotary.
    NUM_ATTENTION_HEADS = 32
    NUM_KEY_VALUE_HEADS = 8
    HEAD_DIM = 128
    ATTENTION_BIAS = False
    ROTARY_DIM = 128  # full rotary: rotary_dim == head_dim

    # FFN: dense gated SwiGLU (silu), three matrices, no bias.
    INTERMEDIATE_SIZE = 14336
    HIDDEN_ACT = "silu"
    MLP_BIAS = False
    MLP_MATS = 3  # gate / up / down

    # RoPE: llama3 smooth-ramp scaling (NOT YaRN).
    ROPE_THETA = 500000.0
    ROPE_TYPE = "llama3"
    ROPE_FACTOR = 8.0
    ROPE_LOW_FREQ_FACTOR = 1.0
    ROPE_HIGH_FREQ_FACTOR = 4.0
    ROPE_ORIGINAL_MAX_POSITION = 8192

    MAX_POSITION_EMBEDDINGS = 131072
    TIE_WORD_EMBEDDINGS = False
    TORCH_DTYPE = "bfloat16"

    @property
    def gqa_group_size(self) -> int:
        return self.NUM_ATTENTION_HEADS // self.NUM_KEY_VALUE_HEADS


@dataclass
class LlamaConfig:
    """Runtime config object (HF-attribute-compatible) the TT modules take.

    Named to match the HF ``LlamaConfig`` attributes the donor modules read (``hidden_size``,
    ``rms_norm_eps``, ...), so a real ``transformers`` config can be substituted anywhere this is
    accepted, and a reduced config can be built for fast unit tests.
    """

    hidden_size: int = Llama31_8BConfig.HIDDEN_SIZE
    num_hidden_layers: int = Llama31_8BConfig.NUM_LAYERS
    num_attention_heads: int = Llama31_8BConfig.NUM_ATTENTION_HEADS
    num_key_value_heads: int = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
    head_dim: int = Llama31_8BConfig.HEAD_DIM
    intermediate_size: int = Llama31_8BConfig.INTERMEDIATE_SIZE
    vocab_size: int = Llama31_8BConfig.VOCAB_SIZE
    rms_norm_eps: float = Llama31_8BConfig.RMS_NORM_EPS
    attention_bias: bool = Llama31_8BConfig.ATTENTION_BIAS
    mlp_bias: bool = Llama31_8BConfig.MLP_BIAS
    hidden_act: str = Llama31_8BConfig.HIDDEN_ACT
    rope_theta: float = Llama31_8BConfig.ROPE_THETA
    max_position_embeddings: int = Llama31_8BConfig.MAX_POSITION_EMBEDDINGS
    tie_word_embeddings: bool = Llama31_8BConfig.TIE_WORD_EMBEDDINGS
    rope_scaling: dict = field(
        default_factory=lambda: {
            "rope_type": Llama31_8BConfig.ROPE_TYPE,
            "factor": Llama31_8BConfig.ROPE_FACTOR,
            "low_freq_factor": Llama31_8BConfig.ROPE_LOW_FREQ_FACTOR,
            "high_freq_factor": Llama31_8BConfig.ROPE_HIGH_FREQ_FACTOR,
            "original_max_position_embeddings": Llama31_8BConfig.ROPE_ORIGINAL_MAX_POSITION,
        }
    )

    @property
    def rotary_dim(self) -> int:
        """Full rotary: Llama rotates the whole head."""
        return self.head_dim

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path = CONFIG_JSON) -> "LlamaConfig":
        """Build from an HF ``config.json`` (the vendored one by default)."""
        with open(path) as f:
            raw = json.load(f)
        raw = raw.get("text_config", raw)
        head_dim = raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"])
        return cls(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=head_dim,
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            attention_bias=raw.get("attention_bias", False),
            mlp_bias=raw.get("mlp_bias", False),
            hidden_act=raw.get("hidden_act", "silu"),
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            rope_scaling=raw.get("rope_scaling"),
        )

    def reduced(self, *, num_hidden_layers=2, intermediate_size=256, vocab_size=512) -> "LlamaConfig":
        """A small config with the SAME head geometry, for fast host-side reference tests."""
        return LlamaConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            rms_norm_eps=self.rms_norm_eps,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            hidden_act=self.hidden_act,
            rope_theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_scaling=dict(self.rope_scaling) if self.rope_scaling else None,
        )
