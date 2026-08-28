# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextMLP`, tensor-parallel over TP chips.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextMLP`.

    out = down_proj( silu(gate_proj(x)) * up_proj(x) )

Tensor-parallel scheme -- the textbook gated-MLP split:

  * gate_proj and up_proj are COLUMN-parallel. Their outputs meet only in `silu(gate) * up`, which is
    per-element, so a chip that owns intermediate features [c*I/TP, (c+1)*I/TP) of BOTH can finish
    that product without ever seeing another chip's features.
  * down_proj is the projection that reduces back to model dim, so it is ROW-parallel on the same
    intermediate slice; each chip emits a partial sum over the full model dim and one all_reduce
    reassembles the golden output.
"""
from __future__ import annotations

import ttnn

from models.demos.qwen3_coder_next._stubs.gated_delta_net import (
    matmul_weight,
    num_devices,
    shard_mapper,
    to_device,
)


class TtQwen3NextMLP:
    """Native ttnn Qwen3-Next gated MLP, column/row-parallel over the TP mesh."""

    def __init__(self, device, *, w_gate, w_up, w_down, hidden_size) -> None:
        self.device = device
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.hidden_size = hidden_size
        self.num_devices = num_devices(device)
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("m_l_p stub needs the torch reference module for its weights")
        sd = torch_module.state_dict()
        hidden = int(torch_module.hidden_size)
        inter = int(torch_module.intermediate_size)

        n = num_devices(device)
        tp = n if inter % n == 0 else 1
        shard_out = shard_mapper(device, tp, -1)  # column-parallel: split intermediate features
        shard_in = shard_mapper(device, tp, -2)  # row-parallel: split the same slice back down

        return cls(
            device,
            w_gate=to_device(matmul_weight(sd["gate_proj.weight"].float()), device, mesh_mapper=shard_out),
            w_up=to_device(matmul_weight(sd["up_proj.weight"].float()), device, mesh_mapper=shard_out),
            w_down=to_device(matmul_weight(sd["down_proj.weight"].float()), device, mesh_mapper=shard_in),
            hidden_size=hidden,
        )

    def __call__(self, hidden_states, *args, **kwargs):
        seq = int(hidden_states.shape[-2])
        x = ttnn.reshape(hidden_states, (1, 1, seq, self.hidden_size))
        h = ttnn.multiply(
            ttnn.silu(ttnn.linear(x, self.w_gate, compute_kernel_config=self.compute_config)),
            ttnn.linear(x, self.w_up, compute_kernel_config=self.compute_config),
        )
        partial = ttnn.linear(h, self.w_down, compute_kernel_config=self.compute_config)
        if self.num_devices > 1:
            partial = ttnn.all_reduce(partial)
        return ttnn.reshape(partial, (1, seq, self.hidden_size))


def build(device, torch_module=None):
    return TtQwen3NextMLP.build(device, torch_module)


def m_l_p(device, torch_module=None):
    return TtQwen3NextMLP.build(device, torch_module)
