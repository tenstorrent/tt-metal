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
        """Width of one attention head.

        Returns:
            int: hidden_size // num_attention_heads, 64 for this checkpoint.
        """
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        """Number of head lanes rotary is applied to.

        Returns:
            int: The full head_dim, because rotary_emb_fraction is 1.0 in config.json.
        """
        return self.head_dim

    @property
    def qkv_dim(self) -> int:
        """Output width of the fused Wqkv projection.

        Returns:
            int: 3 * hidden_size, 2304 for this checkpoint, laid out three-major as [q | k | v].
        """
        return 3 * self.hidden_size

    @property
    def num_ladder_points(self) -> int:
        """Number of capture points in the per-layer parity ladder.

        Returns:
            int: num_hidden_layers + 1, being emb_ln plus every block, so 13 here.
        """
        return self.num_hidden_layers + 1

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Report whether a given block uses the MoE FFN rather than the dense one.

        Upstream uses i % every_n == 1, so layer 0 is dense and layer 1 is the first MoE layer.
        The more natural == 0 gives the opposite assignment, which loads cleanly and computes
        the wrong thing.

        Args:
            layer_idx: Block index, in [0, num_hidden_layers).

        Returns:
            bool: True for MoE layers, which are the odd indices here.
        """
        return layer_idx % self.moe_every_n_layers == 1

    @property
    def moe_layers(self) -> tuple[int, ...]:
        """Indices of the blocks that route through experts.

        Returns:
            tuple[int, ...]: (1, 3, 5, 7, 9, 11) for this checkpoint.
        """
        return tuple(i for i in range(self.num_hidden_layers) if self.is_moe_layer(i))

    @property
    def dense_layers(self) -> tuple[int, ...]:
        """Indices of the blocks that use the dense FFN.

        Returns:
            tuple[int, ...]: (0, 2, 4, 6, 8, 10) for this checkpoint.
        """
        return tuple(i for i in range(self.num_hidden_layers) if not self.is_moe_layer(i))


def load_vendored_hf_config() -> dict:
    """Read the vendored config.json snapshot.

    A byte-exact copy of upstream's file at the pinned revision, so tests can read the real
    config with no network. test_vendored_config_matches_the_pinned_revision asserts it still
    equals the live one.

    Returns:
        dict: The raw config, all 62 upstream keys including ones the reference never reads.
    """
    with open(VENDORED_CONFIG_PATH) as f:
        return json.load(f)


def load_vendored_config() -> NomicMoEConfig:
    """Project the vendored config.json onto the fields the reference reads.

    Returns:
        NomicMoEConfig: The 14 fields the model uses, read from the pinned snapshot rather
        than restated in Python. HF_FIELD_NAMES maps the four GPT-2-style names.
    """
    hf_config = load_vendored_hf_config()
    return NomicMoEConfig(
        **{field.name: hf_config[HF_FIELD_NAMES.get(field.name, field.name)] for field in fields(NomicMoEConfig)}
    )
