# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config for the vendored reference, projected from the pinned config.json snapshot.

config.json is vendored at the pinned revision and is the source of truth, so every value
here is read from it rather than restated. The dataclass adds only the quantities
modeling_nomic_moe.py derives from those values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

VENDORED_CONFIG_PATH = Path(__file__).parent / "config.json"

# Upstream carries GPT-2 names for the four dimensions. Every other field matches.
HF_FIELD_NAMES = {
    "hidden_size": "n_embd",
    "num_hidden_layers": "n_layer",
    "num_attention_heads": "n_head",
    "intermediate_size": "n_inner",
}


@dataclass(frozen=True)
class NomicMoEConfig:
    """The subset of the upstream config that the vendored reference reads."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    type_vocab_size: int
    pad_token_id: int
    layer_norm_epsilon: float
    rotary_emb_base: float
    num_experts: int
    moe_top_k: int
    moe_every_n_layers: int
    max_trained_positions: int
    pad_vocab_size_multiple: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        # rotary_emb_fraction is 1.0 in config.json, so rotary covers the whole head.
        return self.head_dim

    @property
    def attention_scale(self) -> float:
        return self.head_dim**-0.5

    @property
    def qkv_dim(self) -> int:
        return 3 * self.hidden_size

    @property
    def num_ladder_points(self) -> int:
        """Capture points for a per-layer parity ladder: emb_ln plus every block."""
        return self.num_hidden_layers + 1

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Upstream uses `moe=i % every_n == 1`, so layer 0 is dense and layer 1 is the first
        MoE layer. The more natural `== 0` gives the opposite assignment."""
        return layer_idx % self.moe_every_n_layers == 1

    @property
    def moe_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.is_moe_layer(i))

    @property
    def dense_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if not self.is_moe_layer(i))


def load_vendored_hf_config() -> dict:
    """The pinned config.json snapshot, so the no-network tests read the real config."""
    with open(VENDORED_CONFIG_PATH) as f:
        return json.load(f)


def load_vendored_config() -> NomicMoEConfig:
    hf_config = load_vendored_hf_config()
    return NomicMoEConfig(
        **{field.name: hf_config[HF_FIELD_NAMES.get(field.name, field.name)] for field in fields(NomicMoEConfig)}
    )
