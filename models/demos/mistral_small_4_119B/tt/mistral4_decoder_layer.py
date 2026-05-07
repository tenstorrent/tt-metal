# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single Mistral-4 decoder layer — HF-compatible wiring built from ``decoder_block`` dense/MoE stacks.

Analogous to composing DeepSeek ``DecoderBlock2D`` / ``MoEDecoderBlock2D`` behind a layer API
(`decoder_block folder <https://github.com/tenstorrent/tt-metal/tree/main/models/demos/deepseek_v3/tt/decoder_block>`_).

:wfunc:`build_mistral4_decoder_block` selects dense ``Mistral4MLP`` vs HF ``Mistral4MoE``
using ``first_k_dense_replace``, matching :class:`transformers.models.mistral4.modeling_mistral4.Mistral4DecoderLayer`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_base import build_mistral4_decoder_block


class TtMistral4DecoderLayer(nn.Module):
    """Pre-norm GPT block: Attn → residual → MLP → residual (same contract as HF ``Mistral4DecoderLayer``)."""

    def __init__(self, config: Mistral4Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        inner = build_mistral4_decoder_block(config, layer_idx)
        self.self_attn = inner.self_attn
        self.mlp = inner.mlp
        self.input_layernorm = inner.input_layernorm
        self.post_attention_layernorm = inner.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


__all__ = ["TtMistral4DecoderLayer"]
