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

from .tp import _col, _host_rep, _rep, _row, all_reduce


def shared_expert_forward(
    mesh_device: MeshDevice,
    hidden_states: torch.Tensor,  # [B, S, 2688] bf16 CPU (pre-normed)
    w_up: torch.Tensor,  # [3712, 2688] bf16 CPU
    w_down: torch.Tensor,  # [2688, 3712] bf16 CPU
) -> torch.Tensor:
    """Returns [B, S, 2688] bfloat16 (CPU)."""
    B = hidden_states.shape[0]

    # 1. Replicate hidden_states to all devices
    h_tt = _rep(hidden_states, mesh_device)

    # 2. Up projection: column-parallel → [B, S, 928]/device
    wu_tt = _col(w_up, mesh_device)  # [928, 2688]/device
    up_tt = ttnn.linear(h_tt, wu_tt, transpose_b=True)  # [B, S, 928]/device

    # 3. relu2 activation on device
    act_tt = ttnn.pow(ttnn.relu(up_tt), 2)  # [B, S, 928]/device

    # 4. Down projection: row-parallel → partial [B, S, 2688]/device
    wd_tt = _row(w_down, mesh_device)  # [2688, 928]/device
    out_tt = ttnn.linear(act_tt, wd_tt, transpose_b=True)  # [B, S, 2688] partial/device

    # 5. All-reduce to sum partials → full [B, S, 2688] on all devices
    result_tt = all_reduce(out_tt)

    # 6. Bring to host
    return _host_rep(result_tt, mesh_device, B)
