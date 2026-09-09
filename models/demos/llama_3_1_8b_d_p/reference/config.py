# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B-Instruct config constants — the single source of truth for dims.

Everything here is READ FROM `configs/Llama-3.1-8B-Instruct/config.json`, which is the vendored
copy of the checkpoint's own config. Nothing in this file is a bring-up choice: the prefill spec
(`llama_3_1_8b.spec.json`) carries the choices (target hardware, TP/SP, chunk size, dtypes, PCC),
and the recipe is explicit that anything readable from config.json belongs here instead.

`tests/torch_ref/test_llama_reference.py` asserts every constant below against the vendored JSON,
so a checkpoint swap that changes a dim fails a host test rather than a PCC number.

Torch only. No ttnn, no device code — this module is imported by the reference and by host tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

CONFIG_JSON = Path(__file__).parent.parent / "configs" / "Llama-3.1-8B-Instruct" / "config.json"


def load_config_json(path: Path | str = CONFIG_JSON) -> dict[str, Any]:
    """The vendored config.json as a plain dict (no transformers import)."""
    with open(path) as f:
        return json.load(f)


@dataclass(frozen=True)
class LlamaConfigConstants:
    """Llama-3.1-8B-Instruct dims. Built from the vendored config.json via :meth:`from_json`.

    Frozen so a test that mutates it for a reduced run has to build a new one (``replace()``),
    which keeps a reduced config from leaking into a full-depth measurement.
    """

    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    vocab_size: int = 128256
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    # llama3 rope scaling. Kept as a dict because that is the shape both HF's ROPE_INIT_FUNCTIONS
    # and tt_transformers' rope_scaling_model_factory expect.
    rope_scaling: dict[str, Any] = field(
        default_factory=lambda: {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    )

    @classmethod
    def from_json(cls, path: Path | str = CONFIG_JSON) -> LlamaConfigConstants:
        cfg = load_config_json(path)
        return cls(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            max_position_embeddings=cfg["max_position_embeddings"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            vocab_size=cfg["vocab_size"],
            hidden_act=cfg["hidden_act"],
            attention_bias=cfg["attention_bias"],
            mlp_bias=cfg["mlp_bias"],
            tie_word_embeddings=cfg["tie_word_embeddings"],
            torch_dtype=cfg["torch_dtype"],
            rope_scaling=dict(cfg["rope_scaling"]),
        )

    # --- derived ---

    @cached_property
    def head_dim(self) -> int:
        """Not in config.json for this checkpoint; HF derives it the same way."""
        return self.hidden_size // self.num_attention_heads  # 128

    @cached_property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 4

    @cached_property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5

    def kv_heads_per_chip(self, tp: int) -> int:
        """KV heads a single chip holds under tensor parallelism.

        **This is 2 at the spec's TP=4**, not 1. Every prefill package on the common/prefill engine
        (minimax_m3 4 kv heads / TP=4, gpt_oss_d_p 8 kv heads / TP=8) has exactly one, and their KV
        cache code says so in comments even where the shape is parametrized. Nothing raises if this
        is got wrong — it silently corrupts the per-chip cache row width, the ring-gather buffer and
        every host-side read-back. Call this rather than hardcoding 1.
        """
        assert self.num_key_value_heads % tp == 0, (
            f"num_key_value_heads ({self.num_key_value_heads}) must be divisible by tp ({tp}): "
            "a KV head cannot straddle two chips"
        )
        return self.num_key_value_heads // tp

    def q_heads_per_chip(self, tp: int) -> int:
        assert self.num_attention_heads % tp == 0
        return self.num_attention_heads // tp

    def to_hf_config(self):
        """A live `transformers.LlamaConfig` carrying exactly these constants.

        Import-local so this module stays torch-only for callers that do not need HF. Used by the
        D1 upstream-parity test and by the golden trace generator, never by the vendored reference.

        transformers >= 5 moved rope settings from ``config.rope_scaling`` to
        ``config.rope_parameters``, with ``rope_theta`` folded INTO that dict. Both spellings are
        set so this works across the 4.x/5.x boundary — the vendored checkpoint config says 4.42.3
        while the installed transformers is 5.x.

        **Assignment order is load-bearing on 5.x.** There, ``rope_scaling`` and ``rope_theta`` are
        properties backed by ``rope_parameters``, and assigning ``rope_scaling`` *rewrites*
        ``rope_parameters`` from the given dict — which drops ``rope_theta`` and leaves it None.
        ``_compute_llama3_parameters`` then dies with ``pow(): 'NoneType' and 'Tensor'``. So
        ``rope_parameters`` is written LAST, with the theta folded in, and
        ``standardize_rope_params`` is called here rather than left to the first consumer.
        """
        from transformers import LlamaConfig

        cfg = LlamaConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            rope_scaling=dict(self.rope_scaling),
        )
        # 4.x spelling first, then the 5.x one — see the order note above.
        cfg.rope_scaling = dict(self.rope_scaling)
        if hasattr(cfg, "rope_parameters"):
            cfg.rope_parameters = {"rope_theta": self.rope_theta, **self.rope_scaling}
            if hasattr(cfg, "standardize_rope_params"):
                cfg.standardize_rope_params()
            assert cfg.rope_parameters.get("rope_theta") == self.rope_theta, (
                f"rope_theta was lost in the rope_parameters round-trip (got "
                f"{cfg.rope_parameters.get('rope_theta')}): check the assignment order"
            )
        cfg.head_dim = self.head_dim
        cfg._attn_implementation = "eager"
        return cfg
