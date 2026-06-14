# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm block — TP=4 on QB 4-chip Blackhole."""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _rep_keyed

NORM_EPS = 1e-5


def layer_norm_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [B, S, hidden_size] bf16 already on device
    weight: torch.Tensor,  # [hidden_size] bf16 CPU
    eps: float = NORM_EPS,
) -> ttnn.Tensor:
    """Returns [B, S, hidden_size] bfloat16 on device (replicated)."""
    w_tt = _rep_keyed(id(weight), weight.bfloat16().unsqueeze(0), mesh_device)
    return ttnn.rms_norm(hidden_states, epsilon=eps, weight=w_tt)
