# SPDX-License-Identifier: MIT
"""Configuration-driven ESM-2 TT backend config.

The pinned checkpoint config (/weights/config.json) is authoritative; nothing
in this package hardcodes dimensions beyond documented defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Esm2TTConfig:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    layer_norm_eps: float
    pad_token_id: int
    mask_token_id: int
    max_position_embeddings: int
    position_embedding_type: str = "rotary"
    token_dropout: bool = True
    hidden_act: str = "gelu"
    rotary_base: float = 10000.0
    # ESM training-time mask ratio hardcoded in the reference implementation.
    mask_ratio_train: float = 0.15 * 0.8

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads

    def __post_init__(self):
        if self.position_embedding_type != "rotary":
            raise NotImplementedError(
                f"only rotary position embeddings are supported, got {self.position_embedding_type!r}"
            )
        if self.hidden_act != "gelu":
            raise NotImplementedError(f"only gelu is supported, got {self.hidden_act!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "Esm2TTConfig":
        fields = cls.__dataclass_fields__
        missing = [
            k
            for k in (
                "num_hidden_layers",
                "hidden_size",
                "num_attention_heads",
                "intermediate_size",
                "vocab_size",
                "layer_norm_eps",
                "pad_token_id",
                "mask_token_id",
                "max_position_embeddings",
            )
            if k not in d
        ]
        if missing:
            raise ValueError(f"config missing required keys: {missing}")
        kwargs = {k: d[k] for k in fields if k in d}
        # mask_ratio_train is derived, never read from the checkpoint config.
        kwargs.pop("mask_ratio_train", None)
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str) -> "Esm2TTConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
