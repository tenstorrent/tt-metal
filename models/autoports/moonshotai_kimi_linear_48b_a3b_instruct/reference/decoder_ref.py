# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-norm decoder layer oracle for all four layer kinds, with explicit carried state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.kda_ref import (
    KDAReferenceState,
    kda_layer_reference,
)
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.mla_ref import mla_forward_reference, rms_norm
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.moe_ref import (
    dense_mlp_reference,
    moe_reference,
)


@dataclass
class LayerState:
    kda: KDAReferenceState | None = None  # KDA layers
    latent: torch.Tensor | None = None  # MLA layers: [B, S, 576]


def decoder_layer_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cfg: KimiLinearConfig,
    layer_idx: int,
    state: LayerState | None = None,
) -> tuple[torch.Tensor, LayerState, dict[str, torch.Tensor]]:
    """hidden [B,T,H] -> (hidden [B,T,H] fp32, new state, intermediates {attn, mlp})."""
    state = state or LayerState()
    x = hidden.float()
    h = rms_norm(x, weights["input_layernorm.weight"], cfg.rms_norm_eps)
    if cfg.is_kda_layer(layer_idx):
        attn, kda_state = kda_layer_reference(h, weights, cfg.kda_config(), state.kda)
        new_state = LayerState(kda=kda_state)
    else:
        attn, latent = mla_forward_reference(h, weights, cfg, past_latent=state.latent)
        new_state = LayerState(latent=latent)
    x = x + attn
    h2 = rms_norm(x, weights["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    mlp = moe_reference(h2, weights, cfg) if cfg.is_moe_layer(layer_idx) else dense_mlp_reference(h2, weights)
    return x + mlp, new_state, {"attn": attn, "mlp": mlp}
