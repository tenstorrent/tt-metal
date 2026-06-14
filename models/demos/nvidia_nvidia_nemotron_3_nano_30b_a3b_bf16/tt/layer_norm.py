# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm block — TP=4 on QB 4-chip Blackhole."""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _host_rep, _rep

NORM_EPS = 1e-5


def layer_norm_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    eps: float = NORM_EPS,
) -> torch.Tensor:
    B = hidden_states.shape[0]
    h_tt = _rep(hidden_states, mesh_device)
    w_tt = _rep(weight.unsqueeze(0), mesh_device)
    out_tt = ttnn.rms_norm(h_tt, epsilon=eps, weight=w_tt)
    return _host_rep(out_tt, mesh_device, B)
