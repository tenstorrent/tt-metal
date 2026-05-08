# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dense SwiGLU MLP matching Hugging Face ``Mistral4MLP`` (shared expert / dense layers)."""

from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.tt.mistral_small_4.linear import (
    linear_bf16_no_bias_device,
    to_tt_x_bsh_flat,
    tt_flat_to_torch_bsh,
)


def dense_mlp_bf16(
    mesh_device,
    x_bsh: torch.Tensor,
    gate_weight_out_in: torch.Tensor,
    up_weight_out_in: torch.Tensor,
    down_weight_out_in: torch.Tensor,
) -> torch.Tensor:
    """
    ``down( silu(gate(x)) * up(x) )`` on device; returns host bf16 ``[B, S, hidden]``.

    Weights are HF ``nn.Linear.weight`` layouts: ``gate/up`` are ``[I, H]``, ``down`` is ``[H, I]``.
    ``H`` and ``I`` should be multiples of 32.
    """
    b, s, h = int(x_bsh.shape[0]), int(x_bsh.shape[1]), int(x_bsh.shape[2])
    i_gate = int(gate_weight_out_in.shape[1])
    i_up = int(up_weight_out_in.shape[1])
    h_down_in = int(down_weight_out_in.shape[1])
    h_down_out = int(down_weight_out_in.shape[0])
    if i_gate != h or i_up != h:
        raise ValueError("gate/up in_features must match x hidden size")
    if h_down_in != gate_weight_out_in.shape[0]:
        raise ValueError("down in_features must match intermediate (gate out)")
    if h_down_out != h:
        raise ValueError("down out_features must match hidden size")
    inter = h_down_in

    tt_x = to_tt_x_bsh_flat(mesh_device, x_bsh)

    tt_gate = linear_bf16_no_bias_device(mesh_device, tt_x, gate_weight_out_in)
    tt_up = linear_bf16_no_bias_device(mesh_device, tt_x, up_weight_out_in)
    ttnn.deallocate(tt_x)

    tt_gate_act = ttnn.silu(tt_gate)
    ttnn.deallocate(tt_gate)

    tt_hid = ttnn.mul(tt_gate_act, tt_up)
    ttnn.deallocate(tt_gate_act)
    ttnn.deallocate(tt_up)

    tt_out = linear_bf16_no_bias_device(mesh_device, tt_hid, down_weight_out_in)
    ttnn.deallocate(tt_hid)

    y = tt_flat_to_torch_bsh(mesh_device, tt_out, b=b, s=s, out_f=h)
    ttnn.deallocate(tt_out)
    return y
