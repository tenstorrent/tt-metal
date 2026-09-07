# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config for the vendored reference, plus validation of everything the reference hard-codes.

modeling_nomic_moe.py implements one path through upstream's code. Every config field upstream
branches on, but the reference does not, is listed in REQUIRED_FIELDS and checked by
from_hf_config. A config that violates one fails at construction instead of producing a
plausible but wrong model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VENDORED_CONFIG_PATH = Path(__file__).parent / "config.json"


class ConfigAssumptionError(ValueError):
    """A config field contradicts an assumption baked into the vendored reference."""


@dataclass(frozen=True)
class NomicMoEConfig:
    """The subset of the upstream config that the vendored reference reads."""

    vocab_size: int = 250048
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    type_vocab_size: int = 1
    pad_token_id: int = 1
    layer_norm_epsilon: float = 1e-5
    rotary_emb_base: float = 10000.0
    num_experts: int = 8
    moe_top_k: int = 2
    moe_every_n_layers: int = 2
    max_trained_positions: int = 2048
    pad_vocab_size_multiple: int = 64

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        # rotary_emb_fraction is asserted to be 1.0, so rotary covers the whole head.
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


# Config fields the reference hard-codes rather than branches on, and the value each must have.
REQUIRED_FIELDS: dict[str, object] = {
    # Block structure.
    "prenorm": False,
    "parallel_block": False,
    "causal": False,
    "add_pooling_layer": False,
    # Rotary.
    "rotary_emb_fraction": 1.0,
    "rotary_emb_interleaved": False,
    "rotary_emb_scale_base": None,
    "rotary_scaling_factor": None,
    # MLP and MoE.
    "activation_function": "gelu",
    "moe_normalize_expert_weights": False,
    "num_shared_experts": 0,
    "expert_choice_router": False,
    "moe_top_k": 2,
    "moe_every_n_layers": 2,
    "num_experts": 8,
    "ffn_div": 1,
    # Biases.
    "qkv_proj_bias": True,
    "mlp_fc1_bias": True,
    "mlp_fc2_bias": True,
    # Single token-type row, so the term folds to a constant.
    "type_vocab_size": 1,
    # Inference: all dropout must be inert.
    "attn_pdrop": 0.0,
    "resid_pdrop": 0.0,
    "moe_resid_pdrop": 0.0,
}


def from_hf_config(hf_config: dict) -> NomicMoEConfig:
    """Validate an upstream config.json dict and project it onto NomicMoEConfig."""
    for field, expected in REQUIRED_FIELDS.items():
        if field not in hf_config:
            raise ConfigAssumptionError(
                f"config is missing {field!r}; the vendored reference assumes {field}={expected!r}. "
                "A missing GPT-2-style key usually means the config was parsed by the native "
                "transformers nomic_bert class instead of the remote code. See hf_reference.py."
            )
        actual = hf_config[field]
        if actual != expected:
            raise ConfigAssumptionError(
                f"config has {field}={actual!r}, but the vendored reference implements only "
                f"{field}={expected!r}. Upstream branches on this field; we do not."
            )

    # embd_pdrop is 0.1 in this checkpoint but upstream never reads it (embedding dropout uses
    # resid_pdrop), so asserting it would be a false positive.

    hidden_size = hf_config["n_embd"]
    num_heads = hf_config["n_head"]
    if hidden_size % num_heads != 0:
        raise ConfigAssumptionError(f"n_embd={hidden_size} is not divisible by n_head={num_heads}")
    if (hidden_size // num_heads) % 2 != 0:
        raise ConfigAssumptionError(f"head_dim={hidden_size // num_heads} must be even for rotary halves")

    # Upstream rounds vocab_size up to this multiple inside NomicBertModel.__init__. It is a
    # no-op for this checkpoint; if it were not, the embedding table would be larger than the
    # config says and the generated key/shape contract would be wrong.
    multiple = hf_config.get("pad_vocab_size_multiple", 1)
    vocab_size = hf_config["vocab_size"]
    if multiple and vocab_size % multiple != 0:
        raise ConfigAssumptionError(
            f"vocab_size={vocab_size} is not a multiple of pad_vocab_size_multiple={multiple}; "
            "upstream would silently grow the embedding table."
        )

    return NomicMoEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=hf_config["n_layer"],
        num_attention_heads=num_heads,
        intermediate_size=hf_config["n_inner"],
        type_vocab_size=hf_config["type_vocab_size"],
        pad_token_id=hf_config["pad_token_id"],
        layer_norm_epsilon=hf_config["layer_norm_epsilon"],
        rotary_emb_base=float(hf_config["rotary_emb_base"]),
        num_experts=hf_config["num_experts"],
        moe_top_k=hf_config["moe_top_k"],
        moe_every_n_layers=hf_config["moe_every_n_layers"],
        max_trained_positions=hf_config["max_trained_positions"],
        pad_vocab_size_multiple=multiple,
    )


def load_vendored_hf_config() -> dict:
    """The pinned config.json snapshot, so the no-network tests validate the real config."""
    with open(VENDORED_CONFIG_PATH) as f:
        return json.load(f)


def load_vendored_config() -> NomicMoEConfig:
    return from_hf_config(load_vendored_hf_config())
