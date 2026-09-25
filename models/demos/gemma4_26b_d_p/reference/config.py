# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text-model config for Gemma-4 MoE checkpoints (26B-A4B), decoupled from HF transformers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CKPT_DIR = os.environ.get("GEMMA4_D_P_CKPT", "/localdev/mstaletovic/hf_models/gemma-4-26B-A4B-it")
REPO_CONFIG = Path(__file__).resolve().parents[2] / "gemma4" / "configs" / "gemma-4-26B-A4B-it" / "config.json"

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass
class RopeSpec:
    theta: float
    head_dim: int
    # Number of rotated frequency pairs; the rest of the head sees cos=1, sin=0 ("proportional" rope).
    rotated_pairs: int


@dataclass
class Gemma4TextConfig:
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    head_dim: int = 256
    global_head_dim: int = 512
    attention_k_eq_v: bool = True
    sliding_window: int = 1024
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float | None = 30.0
    vocab_size: int = 262144
    tie_word_embeddings: bool = True
    layer_types: list[str] = field(default_factory=list)
    rope_parameters: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | os.PathLike | None = None) -> "Gemma4TextConfig":
        if path is None:
            cand = Path(DEFAULT_CKPT_DIR) / "config.json"
            path = cand if cand.exists() else REPO_CONFIG
        raw = json.load(open(path))
        tc = raw.get("text_config", raw)
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in tc.items() if k in names})

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == SLIDING

    def layer_head_dim(self, layer_idx: int) -> int:
        return self.head_dim if self.is_sliding(layer_idx) else self.global_head_dim

    def layer_kv_heads(self, layer_idx: int) -> int:
        return self.num_key_value_heads if self.is_sliding(layer_idx) else self.num_global_key_value_heads

    def layer_k_eq_v(self, layer_idx: int) -> bool:
        return self.attention_k_eq_v and not self.is_sliding(layer_idx)

    def rope_spec(self, layer_type: str) -> RopeSpec:
        p = self.rope_parameters[layer_type]
        hd = self.head_dim if layer_type == SLIDING else self.global_head_dim
        if p.get("rope_type", "default") == "proportional":
            pairs = int(p.get("partial_rotary_factor", 1.0) * hd // 2)
        else:
            pairs = hd // 2
        return RopeSpec(theta=float(p["rope_theta"]), head_dim=hd, rotated_pairs=pairs)

    def with_layers(self, n: int) -> "Gemma4TextConfig":
        """Truncated copy (first ``n`` layers) for fast tests."""
        import copy

        c = copy.deepcopy(self)
        c.num_hidden_layers = n
        c.layer_types = c.layer_types[:n]
        return c
