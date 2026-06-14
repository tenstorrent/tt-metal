# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SharedExpert — TP=4 column/row-parallel on QB 4-chip Blackhole.

relu2 MLP, intermediate=3712 (moe_shared_expert_intermediate_size).
Input is pre-normed (no pre-norm or residual here).

TP strategy:
  w_up   [3712, 2688]: column-parallel → [928, 2688]/device
  w_down [2688, 3712]: row-parallel   → [2688, 928]/device
  all_reduce after down projection
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _col, _row, all_reduce


def shared_expert_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [B, S, 2688] bf16 on device (pre-normed)
    w_up: torch.Tensor,  # [3712, 2688] bf16 CPU
    w_down: torch.Tensor,  # [2688, 3712] bf16 CPU
) -> ttnn.Tensor:
    """Returns [B, S, 2688] bfloat16 on device (replicated)."""
    wu_tt = _col(w_up, mesh_device)
    up_tt = ttnn.linear(hidden_states, wu_tt, transpose_b=True)  # [B, S, 928]/device

    act_tt = ttnn.pow(ttnn.relu(up_tt), 2)  # relu2

    wd_tt = _row(w_down, mesh_device)
    out_tt = ttnn.linear(act_tt, wd_tt, transpose_b=True)  # [B, S, 2688] partial/device

    return all_reduce(out_tt)  # full [B, S, 2688] on all devices
