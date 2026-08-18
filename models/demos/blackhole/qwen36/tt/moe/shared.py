# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated shared expert.

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
        # The shared expert reuses Qwen36MLP, whose TP matmul program configs / weight memcfgs are
        # sized from ModelArgs.hidden_dim — which on a MoE config is the injected moe_intermediate_size
        # stand-in, not shared_expert_intermediate_size. That is only correct while the two sizes are
        # equal (they are on the shipped 35B-A3B: both 512). Fail fast if a checkpoint diverges them,
        # rather than emitting an opaque program-config/weight-width mismatch at decode.
        if args is not None and getattr(args, "moe_shared_intermediate_size", None):
            assert args.moe_shared_intermediate_size == args.moe_intermediate_size, (
                f"shared_expert_intermediate_size ({args.moe_shared_intermediate_size}) != "
                f"moe_intermediate_size ({args.moe_intermediate_size}); the shared expert reuses the "
                f"routed-expert MLP program configs and needs them equal (see tt/moe/shared.py)."
            )
        # The shared expert receives already-gathered (full/replicated) hidden — the MoE layer's
        # ff_norm does its own all-gather (layer._fuse_ff_agmm is off for MoE). So it must NOT run
        # the fused gate/up all-gather-matmul (that would re-gather full input → K mismatch).
        self.mlp = Qwen36MLP(mesh_device, shared_state, shared_cache, args=args, tt_ccl=tt_ccl, use_gateup_agmm=False)

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
