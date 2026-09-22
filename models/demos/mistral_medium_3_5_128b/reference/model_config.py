# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config constants for Mistral-Medium-3.5-128B, and the loader for the vendored ``config.json``.

The checkpoint is a ``Mistral3ForConditionalGeneration`` wrapper: the language model lives under
``text_config`` (``model_type: ministral3``) and the weights under the ``model.language_model.``
prefix. The vision tower is **out of scope** for prefill bring-up and is not read here.

Everything in :class:`MistralMediumConfig` is readable from ``config.json`` — per the recipe, such
values belong to the config and not to the prefill spec, and D1's first test asserts every constant
here against the vendored file. The only values the spec contributes are target hardware, TP/SP,
chunk size, dtypes and the two PCC thresholds; those live in :class:`PrefillSpec`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VENDORED_CONFIG = Path(__file__).parent / "config.json"


@dataclass(frozen=True)
class MistralMediumConfig:
    """The text-model constants. Frozen: a bring-up that mutates its own config is not reproducible."""

    hidden_size: int = 12288
    num_hidden_layers: int = 88
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 28672
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    vocab_size: int = 131072
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    sliding_window: Any = None  # null for this model: every layer is full attention
    attention_dropout: float = 0.0

    # --- YaRN rope_parameters ---
    rope_type: str = "yarn"
    rope_theta: float = 1000000.0
    rope_factor: float = 64.0
    rope_beta_fast: float = 4.0
    rope_beta_slow: float = 1.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 0.0
    rope_original_max_position_embeddings: int = 4096
    # Ministral3Attention multiplies Q by `1 + beta*log(1 + floor(pos/original_max_pos))`. This
    # checkpoint sets beta = 0, which makes the whole term identically 1.0, so neither the
    # reference nor the device implements it. It is carried here so the assumption is checked
    # against the config rather than left implicit — a nonzero beta is a position-dependent Q
    # scale that no amount of staring at attention code would reveal.
    rope_llama_4_scaling_beta: float = 0.0

    # --- checkpoint quantization ---
    quant_method: str = "fp8"
    quant_activation_scheme: str = "static"
    quant_weight_block_size: Any = None  # null => per-tensor scalar weight_scale_inv
    weight_prefix: str = "model.language_model."

    @property
    def num_key_value_groups(self) -> int:
        """Q heads per KV head — 96 / 8 = 12."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def attention_scaling(self) -> float:
        """YaRN's attention (mscale) factor, applied to cos/sin.

        transformers' ``yarn`` rope init computes ``0.1 * ln(factor) + 1.0`` when the config does not
        select the DeepSeek two-mscale form. The golden trace records the reference's actual value
        (``metadata.json -> rope.attention_scaling``) as ``1.4158883083359672``, and
        ``0.1*ln(64)+1 = 1.41588830833596...`` matches it exactly — so this is the right branch, and
        the config's explicit ``mscale: 1.0`` / ``mscale_all_dim: 0.0`` do not select a different one.
        ``tests/unit/test_reference_config.py`` pins this against the trace.
        """
        return 0.1 * math.log(self.rope_factor) + 1.0

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    @classmethod
    def from_json(cls, path: Path | str = VENDORED_CONFIG) -> "MistralMediumConfig":
        """Build from a HF ``config.json`` (reads ``text_config``), so the constants above can be
        asserted against the checkpoint rather than trusted."""
        raw = json.loads(Path(path).read_text())
        t = raw["text_config"]
        rope = t["rope_parameters"]
        quant = raw.get("quantization_config", {})
        return cls(
            hidden_size=t["hidden_size"],
            num_hidden_layers=t["num_hidden_layers"],
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            intermediate_size=t["intermediate_size"],
            hidden_act=t["hidden_act"],
            rms_norm_eps=t["rms_norm_eps"],
            vocab_size=t["vocab_size"],
            max_position_embeddings=t["max_position_embeddings"],
            tie_word_embeddings=t.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)),
            sliding_window=t["sliding_window"],
            attention_dropout=t.get("attention_dropout", 0.0),
            rope_type=rope["rope_type"],
            rope_theta=rope["rope_theta"],
            rope_factor=rope["factor"],
            rope_beta_fast=rope["beta_fast"],
            rope_beta_slow=rope["beta_slow"],
            rope_mscale=rope["mscale"],
            rope_mscale_all_dim=rope["mscale_all_dim"],
            rope_original_max_position_embeddings=rope["original_max_position_embeddings"],
            rope_llama_4_scaling_beta=rope.get("llama_4_scaling_beta", 0.0),
            quant_method=quant.get("quant_method", "fp8"),
            quant_activation_scheme=quant.get("activation_scheme", "static"),
            quant_weight_block_size=quant.get("weight_block_size"),
        )

    def reduced(self, **overrides) -> "MistralMediumConfig":
        """A smaller config for host-side diagnostics.

        Reduced runs are **diagnostics, never the result** (recipe §4): anything measured on one of
        these must be labelled reduced wherever it is quoted. Acceptance always runs at full depth
        and width.
        """
        from dataclasses import replace

        return replace(self, **overrides)


def host_reduced_config() -> MistralMediumConfig:
    """The one small config every **host-only** test shares.

    Shrunk in every dimension that costs CPU time but keeping the structure that can go wrong: GQA
    with a group size > 1, an even hidden/intermediate split, and the real YaRN parameters. Device
    tests do **not** use this — they run at full width so TP divisibility stays real, and vary only
    depth and sequence length. Anything measured on it is a diagnostic, never an acceptance result.
    """
    return MistralMediumConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=1024,
        vocab_size=2048,
    )


@dataclass(frozen=True)
class PrefillSpec:
    """The binding prefill spec's values — the parts that are NOT readable from ``config.json``.

    Defaults mirror the run's ``spec.json``; ``from_json`` reads the snapshot named by
    ``PREFILL_SPEC`` so the spec stays the single source of truth.
    """

    target_hw: str = "bh_galaxy"
    tp: int = 4
    sp: int = 8
    max_seq_len: int = 262144
    chunk_size: int = 5120
    pcc_target: float = 0.99
    pcc_lower_bound: float = 0.85
    activations_dtype: str = "bfloat16"
    kv_cache_dtype: str = "bfloat8_b"
    weights_dtype: str = "bfloat8_b"

    @property
    def mesh_shape(self) -> tuple[int, int]:
        """(rows, cols) = (SP, TP)."""
        return (self.sp, self.tp)

    @classmethod
    def from_json(cls, path: Path | str) -> "PrefillSpec":
        raw = json.loads(Path(path).read_text())
        par, shapes, acc = raw["parallelism"], raw["shapes"], raw["acceptance"]
        df = raw["dataformats"]

        def _group(name, key="default"):
            g = df.get(name, {})
            return g.get(key) or g.get("default")

        return cls(
            target_hw=raw["target_hw"],
            tp=par["tp"],
            sp=par["sp"],
            max_seq_len=shapes["max_seq_len"],
            chunk_size=shapes["chunk_size"],
            pcc_target=acc["pcc_target"],
            pcc_lower_bound=acc["pcc_lower_bound"],
            activations_dtype=_group("activations"),
            kv_cache_dtype=_group("kv_cache"),
            weights_dtype=_group("weights"),
        )

    @classmethod
    def from_env(cls) -> "PrefillSpec":
        """Read ``PREFILL_SPEC`` if set, else the built-in defaults."""
        import os

        path = os.getenv("PREFILL_SPEC")
        return cls.from_json(path) if path else cls()
