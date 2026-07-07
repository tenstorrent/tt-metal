# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 DiT layer stack (TTTv2-pattern composition module).

The generative core of the AceStepDiTModel: a stack of `num_hidden_layers` AceStepDiTLayer,
each with alternating full/sliding self-attention, cross-attention on the shared encoder
context, and AdaLN modulation from a shared timestep embedding.

This module validates that stacking the DiT block (the real 24-layer depth) accumulates
correctly. It reuses AceStepDiTLayer directly; per-layer full/sliding is bound at construction
(no forward branching) and each sliding layer receives the windowed mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.common.lightweightmodule import LightweightModule
from models.experimental.acestep.tt.dit_layer import AceStepDiTLayer


@dataclass
class AceStepDiTStackConfig:
    # Per-layer AceStepDiTLayerConfig in order.
    layer_configs: list = field(default_factory=list)
    # attention_type per layer ("full_attention" | "sliding_attention"), same order.
    layer_types: list = field(default_factory=list)


class AceStepDiTStack(LightweightModule):
    """forward(hidden [1,1,seq,dim], cos, sin, temb [1,6,B,dim], encoder, sliding_mask=None)
    -> [1,1,seq,dim]. Full layers use mask=None; sliding layers use sliding_mask."""

    def __init__(self, config: AceStepDiTStackConfig):
        self.config = config
        self.layers = [AceStepDiTLayer(lc) for lc in config.layer_configs]
        self.layer_types = config.layer_types

    @classmethod
    def from_config(cls, config: AceStepDiTStackConfig):
        return cls(config)

    def compute_cross_kv(self, encoder_hidden_states):
        """Precompute every layer's cross-attention K/V from the encoder context (denoise-invariant),
        for reuse across all denoise steps. Returns a list (per layer) of (k, v) or None."""
        return [layer.compute_cross_kv(encoder_hidden_states) for layer in self.layers]

    def forward(self, hidden_states, cos, sin, temb, encoder_hidden_states, sliding_mask=None, cross_kv=None):
        for idx, (layer, attn_type) in enumerate(zip(self.layers, self.layer_types)):
            mask = sliding_mask if attn_type == "sliding_attention" else None
            hidden_states = layer.forward(
                hidden_states,
                cos,
                sin,
                temb,
                encoder_hidden_states=encoder_hidden_states,
                attn_mask=mask,
                cross_kv=cross_kv[idx] if cross_kv is not None else None,
            )
        return hidden_states
