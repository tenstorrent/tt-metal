# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention torch oracle: re-exports the DeepSeek-prefill demo's stateless reference."""

from __future__ import annotations

from typing import Mapping

import torch

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict

__all__ = [
    "KDAReferenceState",
    "kda_forward_reference",
    "kda_layer_reference",
    "normalize_kda_state_dict",
    "zero_state",
]


def zero_state(config: KDAConfig, batch: int = 1) -> KDAReferenceState:
    h = config.conv_kernel_size - 1
    return KDAReferenceState(
        recurrent=torch.zeros(batch, config.num_heads, config.head_k_dim, config.head_v_dim),
        q_convolution=torch.zeros(batch, h, config.q_dim),
        k_convolution=torch.zeros(batch, h, config.k_dim),
        v_convolution=torch.zeros(batch, h, config.v_dim),
    )


def kda_layer_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState | None = None,
) -> tuple[torch.Tensor, KDAReferenceState]:
    """``hidden`` [B,T,H] (any float dtype) -> (output [B,T,H] fp32, new state). Weights use checkpoint names."""
    return kda_forward_reference(hidden.float(), normalize_kda_state_dict(weights, config), config, state)
