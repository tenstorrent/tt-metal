# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated shared expert for Qwen3.5-MoE.

HF: out = sigmoid(shared_expert_gate(x)) * shared_expert_mlp(x), added to the routed
experts. The SwiGLU body is the same shape as a dense MLP, so it reuses Qwen36MLP —
which on the (1,4) mesh already reduce-scatters (dim=3) to fractured hidden, matching
the routed-experts output for the final add. The sigmoid gate value [1,1,S,1] is
replicated (from the replicated input x) and broadcasts across the fractured hidden.
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.utils.substate import substate


class Qwen36SharedExpert:
    def __init__(self, mesh_device, mlp_state, tensor_cache_path=None, args=None, tt_ccl=None):
        shared_state = substate(mlp_state, "shared_expert")  # gate_proj/up_proj/down_proj .weight
        shared_cache = (tensor_cache_path / "shared_expert") if tensor_cache_path else None
        self.mlp = Qwen36MLP(mesh_device, shared_state, shared_cache, args=args, tt_ccl=tt_ccl)

        # shared_expert_gate.weight is [1, H] -> [1,1,H,1] for ttnn.linear, replicated.
        is_mesh = hasattr(mesh_device, "shape")
        gate_w = mlp_state["shared_expert_gate.weight"].to(torch.bfloat16).transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        self.gate_weight = ttnn.as_tensor(
            gate_w,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
            cache_file_name=(str(tensor_cache_path / "moe.shared_expert_gate.weight") if tensor_cache_path else None),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x):
        gate = ttnn.sigmoid(ttnn.linear(x, self.gate_weight))  # [1,1,S,1] replicated
        shared_out = self.mlp.forward(x)  # fractured hidden on TP, full on single device
        return ttnn.mul(shared_out, gate)
