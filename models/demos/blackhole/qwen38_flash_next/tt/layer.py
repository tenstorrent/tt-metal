# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first Qwen4Exp decoder-layer composition.

This combines the exact checkpoint-backed component oracles.  It is deliberately
CPU-only: the QSA value path is dense for short contexts, and the partition-A
fabric blocker currently prevents validating TTNN collectives.  The class is an
integration oracle and must not be reported as a device layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDN, Qwen38GDNState, Qwen38GDNWeights
from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidual, Qwen38GatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38MoE, Qwen38MoERouting, Qwen38MoEWeights
from models.demos.blackhole.qwen38_flash_next.tt.ple import (
    Qwen38HostPLEEmbedding,
    Qwen38PLE,
    Qwen38PLEState,
    Qwen38PLEWeights,
)
from models.demos.blackhole.qwen38_flash_next.tt.qsa import Qwen38QSA, Qwen38QSAState, Qwen38QSAWeights


@dataclass(frozen=True)
class Qwen38DecoderLayerState:
    attention: Qwen38GDNState | Qwen38QSAState | None
    ple: Qwen38PLEState | None


@dataclass(frozen=True)
class Qwen38DecoderLayerAux:
    routing: Qwen38MoERouting
    selected_tokens: torch.Tensor | None


class Qwen38DecoderLayer:
    def __init__(
        self,
        *,
        layer_index: int,
        attention: Qwen38GDN | Qwen38QSA,
        attention_gr: Qwen38GatedResidual,
        mlp: Qwen38MoE,
        mlp_gr: Qwen38GatedResidual,
        ple: Qwen38PLE | None,
    ):
        self.layer_index = layer_index
        self.attention = attention
        self.attention_gr = attention_gr
        self.mlp = mlp
        self.mlp_gr = mlp_gr
        self.ple = ple

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Qwen38Checkpoint, placement: Qwen38Placement, *, layer_index: int
    ) -> "Qwen38DecoderLayer":
        config = checkpoint.config
        if not 0 <= layer_index < config.num_hidden_layers:
            raise ValueError(f"layer index is outside the 48-layer backbone: {layer_index}")
        layer_type = config.layer_types[layer_index]
        if layer_type == "linear_attention":
            attention = Qwen38GDN(Qwen38GDNWeights.from_checkpoint(checkpoint, layer_index))
        elif layer_type == "full_attention":
            attention = Qwen38QSA(Qwen38QSAWeights.from_checkpoint(checkpoint, layer_index))
        else:
            raise ValueError(f"unsupported pinned layer type: {layer_type!r}")
        ple = None
        if layer_index == config.ple_checkpoint_layer:
            ple = Qwen38PLE(Qwen38HostPLEEmbedding(checkpoint), Qwen38PLEWeights.from_checkpoint(checkpoint))
        return cls(
            layer_index=layer_index,
            attention=attention,
            attention_gr=Qwen38GatedResidual(
                Qwen38GatedResidualWeights.from_checkpoint(checkpoint, placement, layer_index=layer_index, block="attn")
            ),
            mlp=Qwen38MoE(Qwen38MoEWeights(checkpoint, placement, layer_index=layer_index)),
            mlp_gr=Qwen38GatedResidual(
                Qwen38GatedResidualWeights.from_checkpoint(checkpoint, placement, layer_index=layer_index, block="mlp")
            ),
            ple=ple,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        state: Qwen38DecoderLayerState | None = None,
    ) -> tuple[torch.Tensor, Qwen38DecoderLayerState, Qwen38DecoderLayerAux]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.attention_gr.config.residual_width:
            raise ValueError(f"decoder residual must be [batch, sequence, {self.attention_gr.config.residual_width}]")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"decoder oracle requires BF16 activations, got {hidden_states.dtype}")

        previous_attention = None if state is None else state.attention
        previous_ple = None if state is None else state.ple
        next_ple = previous_ple
        if self.ple is not None:
            if input_ids is None:
                raise ValueError(f"PLE layer {self.layer_index} requires exact token IDs")
            ple_output, next_ple = self.ple.forward(hidden_states, input_ids, previous_ple)
            hidden_states = hidden_states + ple_output
        elif previous_ple is not None:
            raise ValueError("non-PLE layer received PLE state")

        block_input, attention_gr_state = self.attention_gr.read(hidden_states)
        selected_tokens = None
        if isinstance(self.attention, Qwen38GDN):
            attention_output, next_attention = self.attention.forward(block_input, previous_attention)
        else:
            if previous_attention is not None and not isinstance(previous_attention, Qwen38QSAState):
                raise ValueError("QSA layer received a non-QSA attention state")
            if position_embeddings is None or attention_mask is None:
                raise ValueError("QSA layer requires position embeddings and a causal attention mask")
            attention_output, next_attention, selected_tokens = self.attention.forward(
                block_input, position_embeddings, attention_mask, state=previous_attention
            )
        hidden_states = self.attention_gr.write(attention_output, attention_gr_state)

        block_input, mlp_gr_state = self.mlp_gr.read(hidden_states)
        mlp_output, routing = self.mlp(block_input)
        hidden_states = self.mlp_gr.write(mlp_output, mlp_gr_state)
        next_state = Qwen38DecoderLayerState(attention=next_attention, ple=next_ple)
        return hidden_states, next_state, Qwen38DecoderLayerAux(routing=routing, selected_tokens=selected_tokens)
