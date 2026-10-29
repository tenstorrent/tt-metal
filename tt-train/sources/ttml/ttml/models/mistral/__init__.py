# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mistral (``MistralForCausalLM``) for tt-train: the Llama stack with an explicit ``head_dim``.

Model: :class:`Mistral` (a :class:`ttml.models.llama.Llama` built from a :class:`MistralConfig`, with a
no-untilize lookup when the embedding table is frozen). Weights: :func:`load_from_safetensors` (the Llama
loader; Mistral checkpoints use the same HF tensor names). FLOPs: :mod:`.flops` for the LoRA MFU
numerator. Bring-up plan and evidence: ``doc/``.
"""

from __future__ import annotations

import os

import ttml
from ..llama import Llama
from ..llama import load_from_safetensors as _load_llama
from .config import MistralConfig
from .frozen_embedding import embed_tokens
from .flops import useful_flops_per_step, useful_flops_per_token


class Mistral(Llama):
    """Llama decoder with Mistral's decoupled head width; see :class:`MistralConfig`."""

    def __init__(self, config: MistralConfig) -> None:
        if not isinstance(config, MistralConfig):
            raise TypeError(f"Mistral needs a MistralConfig, got {type(config).__name__}")
        super().__init__(config)

    def _embed(self, input_ids: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        # Frozen table (LoRA): untilize once, no backward. Trainable table: the ttml op.
        return embed_tokens(self.tok_emb, input_ids)


def load_from_safetensors(model: Mistral, safetensors_path: str | os.PathLike):
    """Load a HuggingFace Mistral checkpoint directory (``model-*.safetensors`` + ``config.json``).

    Mistral repos also ship a ``consolidated.safetensors`` in Mistral's own key format; the loader reads only
    the shards named by ``model.safetensors.index.json``, so it is ignored.
    """
    return _load_llama(model, safetensors_path)


__all__ = [
    "Mistral",
    "MistralConfig",
    "load_from_safetensors",
    "useful_flops_per_step",
    "useful_flops_per_token",
]
