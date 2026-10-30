# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mistral (``MistralForCausalLM``) for tt-train: the Llama stack with an explicit ``head_dim``.

Model: :class:`Mistral` (a :class:`ttml.models.llama.Llama` built from a :class:`MistralConfig`, with a
no-untilize lookup when the embedding table is frozen and, opted in by
:func:`.fused_residual_norm.use_fused_residual_norm`, the residual adds fused into the norms). Weights: :func:`load_from_safetensors` (the Llama
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
from .fused_residual_norm import fused_forward
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

    def forward(self, input, mask, kv_cache=None, new_tokens=None):
        # `use_fused_residual_norm`: the training forward with every residual add fused into the next norm.
        if (
            getattr(self, "fused_residual_norm", False)
            and kv_cache is None
            and self.config.runner_type == ttml.models.RunnerType.Default
        ):
            return fused_forward(self, input, mask)
        return super().forward(input, mask, kv_cache, new_tokens)


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
