# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the vendored nomic-embed-text-v2-moe reference.

The reference in `modeling_nomic_moe.py` covers the *inference* path of one specific
checkpoint. Every branch upstream takes at runtime that we do not implement is instead
baked in here as an assertion, so a checkpoint or config that violates one fails loudly
at construction rather than silently producing a plausible-but-wrong model.

That failure mode is not hypothetical. `transformers` ships a native `nomic_bert` model
class targeting nomic-embed-text-v1.5 (separate q/k/v, no bias, SwiGLU, eps 1e-12, MoE
absent). Its config is `@strict`, yet it accepts this checkpoint's `config.json` while
silently dropping the GPT-2-style keys it does not recognise. See `hf_reference.py`.
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
    """The subset of the upstream config that the vendored reference actually reads."""

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

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        # rotary_emb_fraction is asserted to be 1.0, so rotary covers the whole head.
        return self.head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Upstream: `NomicBertBlock(config, moe=i % every_n == 1)`.

        Note the `== 1`, not `== 0`: layer 0 is dense and layer 1 is the first MoE layer.
        """
        return layer_idx % self.moe_every_n_layers == 1

    @property
    def moe_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if self.is_moe_layer(i))

    @property
    def dense_layers(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_hidden_layers) if not self.is_moe_layer(i))


# Every upstream config field whose value the vendored reference hard-codes rather than
# branches on, mapped to the value it must have. Grouped by what a violation would mean.
_REQUIRED_FIELDS: dict[str, object] = {
    # --- block structure -------------------------------------------------------------
    "prenorm": False,  # we implement post-norm only
    "parallel_block": False,  # we implement sequential attn -> mlp only
    "causal": False,  # encoder: bidirectional attention
    "add_pooling_layer": False,  # no `pooler.*` in the checkpoint
    # --- rotary ----------------------------------------------------------------------
    "rotary_emb_fraction": 1.0,  # rotary covers the full head dim
    "rotary_emb_interleaved": False,  # GPT-NeoX halves, not GPT-J pairs
    "rotary_emb_scale_base": None,  # no xPos
    "rotary_scaling_factor": None,  # no DynamicNTK
    # --- mlp / moe -------------------------------------------------------------------
    "activation_function": "gelu",  # exact-erf GELU, not tanh and not a gated variant
    "moe_normalize_expert_weights": False,  # top-k weights are NOT renormalised
    "num_shared_experts": 0,  # no always-on expert
    "expert_choice_router": False,  # token-choice routing
    "moe_top_k": 2,
    "moe_every_n_layers": 2,
    "num_experts": 8,
    "ffn_div": 1,
    # --- biases ----------------------------------------------------------------------
    "qkv_proj_bias": True,  # Wqkv and out_proj both carry bias
    "mlp_fc1_bias": True,
    "mlp_fc2_bias": True,
    # --- embeddings ------------------------------------------------------------------
    "type_vocab_size": 1,  # single token-type row => foldable to a constant
    # --- dropout (inference: all must be inert) ---------------------------------------
    "attn_pdrop": 0.0,
    "resid_pdrop": 0.0,
    "moe_resid_pdrop": 0.0,
}


def from_hf_config(hf_config: dict) -> NomicMoEConfig:
    """Validate an upstream `config.json` dict and project it onto `NomicMoEConfig`.

    Raises `ConfigAssumptionError` on the first field that contradicts the reference.
    """
    for field, expected in _REQUIRED_FIELDS.items():
        if field not in hf_config:
            raise ConfigAssumptionError(
                f"config is missing {field!r}; the vendored reference assumes {field}={expected!r}. "
                "A missing GPT-2-style key usually means the config was parsed by the native "
                "transformers `nomic_bert` class instead of the remote code -- see hf_reference.py."
            )
        actual = hf_config[field]
        if actual != expected:
            raise ConfigAssumptionError(
                f"config has {field}={actual!r}, but the vendored reference implements only "
                f"{field}={expected!r}. Upstream branches on this field; we do not."
            )

    # `embd_pdrop` is present in the checkpoint's config as 0.1, but upstream never reads it
    # (the embedding dropout is `resid_pdrop`). Asserting it would be a false positive.

    hidden_size = hf_config["n_embd"]
    num_heads = hf_config["n_head"]
    if hidden_size % num_heads != 0:
        raise ConfigAssumptionError(f"n_embd={hidden_size} is not divisible by n_head={num_heads}")
    head_dim = hidden_size // num_heads
    if head_dim % 2 != 0:
        raise ConfigAssumptionError(f"head_dim={head_dim} must be even for rotary halves")

    # Upstream rounds vocab_size up to a multiple of `pad_vocab_size_multiple` inside
    # NomicBertModel.__init__. For this checkpoint 250048 % 64 == 0, so it is a no-op; if it
    # were not, the embedding table would be larger than the config says and our generated
    # key/shape contract would be wrong.
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
    )


def load_vendored_hf_config() -> dict:
    """The pinned `config.json` snapshot, so the no-network tests still validate the real thing."""
    with open(VENDORED_CONFIG_PATH) as f:
        return json.load(f)


def load_vendored_config() -> NomicMoEConfig:
    return from_hf_config(load_vendored_hf_config())
