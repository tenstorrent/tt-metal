# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mistral hyper-parameters: a Llama config with an explicit ``head_dim``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from ..llama import LlamaConfig, _read_hf_config

__all__ = ["MistralConfig"]


@dataclass(frozen=True)
class MistralConfig(LlamaConfig):
    """``LlamaConfig`` read from a ``model_type: mistral`` checkpoint.

    Mistral-Small-24B sets ``head_dim`` (128) so that ``num_attention_heads * head_dim`` (4096) differs
    from ``hidden_size`` (5120); everything else is the Llama architecture. Sliding-window attention is
    not supported: the config must have ``sliding_window: null``.
    """

    @classmethod
    def from_hf(cls, hf_config: str | os.PathLike | Mapping, max_position_embeddings: int, **overrides):
        if not isinstance(hf_config, Mapping):
            hf_config = _read_hf_config(hf_config)
        model_type = hf_config.get("model_type", "mistral")
        if model_type != "mistral":
            raise ValueError(f"model_type {model_type!r} is not a Mistral checkpoint")
        if hf_config.get("sliding_window") is not None:
            raise ValueError(f"sliding_window={hf_config['sliding_window']} is not supported, only null (full causal)")
        # The Llama reader does the rest (biases, activation, RoPE, head_dim).
        return super().from_hf({**hf_config, "model_type": "llama"}, max_position_embeddings, **overrides)
