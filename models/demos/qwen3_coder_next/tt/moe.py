# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Mixture of Experts layer for Qwen3-Coder-Next (LightweightModule / ttnn).

Combines the routing gate, 64 local expert MLPs (per device), and 1 shared expert.
Each token is routed to top-10 experts; the shared expert processes all tokens.

Expert-parallel strategy (8 devices):
    - 512 experts / 8 devices = 64 experts per device (contiguous assignment)
    - Device 0: experts 0-63, Device 1: experts 64-127, ..., Device 7: experts 448-511
    - Shared expert replicated on all devices
    - Router weights replicated on all devices

State dict keys:
    Router:        model.layers.{layer}.mlp.gate.weight
    Experts:       model.layers.{layer}.mlp.experts.{id}.{gate_proj|up_proj|down_proj}.weight
    Shared expert: model.layers.{layer}.mlp.shared_expert.{gate_proj|up_proj|down_proj}.weight
"""


import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_coder_next.tt.expert_mlp import ExpertMLPBank
from models.demos.qwen3_coder_next.tt.moe_gate import MoEGate


class SharedExpert(LightweightModule):
    """Shared expert MLP (always active for every token).

    Single SwiGLU: hidden_size (2048) -> shared_expert_intermediate_size (512) -> hidden_size.

    State dict keys:
        model.layers.{layer}.mlp.shared_expert.gate_proj.weight  (512, 2048)
        model.layers.{layer}.mlp.shared_expert.up_proj.weight    (512, 2048)
        model.layers.{layer}.mlp.shared_expert.down_proj.weight  (2048, 512)
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype):
        super().__init__()
        self.mesh_device = mesh_device

        # Support both with and without "model." prefix
        state_dict_prefix = f"model.layers.{layer_num}.mlp.shared_expert"
        if f"{state_dict_prefix}.gate_proj.weight" not in state_dict:
            state_dict_prefix = f"layers.{layer_num}.mlp.shared_expert"

        # gate_proj: (intermediate, hidden) -> transposed to (hidden, intermediate)
        gate_w = state_dict[f"{state_dict_prefix}.gate_proj.weight"].transpose(-2, -1).contiguous()
        self.gate_proj = ttnn.as_tensor(
            gate_w.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # up_proj: (intermediate, hidden) -> transposed to (hidden, intermediate)
        up_w = state_dict[f"{state_dict_prefix}.up_proj.weight"].transpose(-2, -1).contiguous()
        self.up_proj = ttnn.as_tensor(
            up_w.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # down_proj: (hidden, intermediate) -> transposed to (intermediate, hidden)
        down_w = state_dict[f"{state_dict_prefix}.down_proj.weight"].transpose(-2, -1).contiguous()
        self.down_proj = ttnn.as_tensor(
            down_w.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, x):
        """Shared expert forward: SwiGLU(gate(x), up(x)) -> down.

        Args:
            x: ttnn tensor (1, 1, batch*seq, hidden_size) on device.

        Returns:
            ttnn tensor (1, 1, batch*seq, hidden_size).
        """
        gate_out = ttnn.linear(x, self.gate_proj)
        gate_out = ttnn.silu(gate_out)

        up_out = ttnn.linear(x, self.up_proj)

        hidden = ttnn.multiply(gate_out, up_out)
        ttnn.deallocate(gate_out)
        ttnn.deallocate(up_out)

        out = ttnn.linear(hidden, self.down_proj)
        ttnn.deallocate(hidden)
        return out


class MoELayer(LightweightModule):
    """Full Mixture of Experts layer.

    Contains: MoEGate (router) + ExpertMLPBank (64 local experts) + SharedExpert.

    Forward flow:
        1. Gate routing: compute top-10 expert assignments on host
        2. Expert computation: run local experts on routed tokens
        3. Shared expert: run on all tokens
        4. Combine: expert_output + shared_expert_output
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype, device_id=0):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args

        # Routing gate (replicated on all devices)
        self.gate = MoEGate(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
        )

        # Local expert bank (64 experts for this device)
        self.experts = ExpertMLPBank(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            device_id=device_id,
            dtype=dtype,
        )

        # Shared expert (replicated, always active)
        self.shared_expert = SharedExpert(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
        )

        # Shared expert output gate: scales the shared expert's contribution
        # HF key: model.layers.{layer}.mlp.shared_expert_gate.weight
        shared_gate_key = f"model.layers.{layer_num}.mlp.shared_expert_gate.weight"
        if shared_gate_key not in state_dict:
            shared_gate_key = f"layers.{layer_num}.mlp.shared_expert_gate.weight"
        if shared_gate_key in state_dict:
            shared_gate_w = state_dict[shared_gate_key]
            # Shape: (hidden_size,) or (1, hidden_size) — used as element-wise scaling
            if shared_gate_w.dim() == 1:
                shared_gate_w = shared_gate_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.shared_expert_gate = ttnn.as_tensor(
                shared_gate_w,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        else:
            self.shared_expert_gate = None

    def forward(self, x):
        """MoE forward pass.

        Args:
            x: ttnn tensor (1, 1, batch*seq, hidden_size) on device.

        Returns:
            ttnn tensor (1, 1, batch*seq, hidden_size).
        """
        # 1. Gate routing (returns torch tensors on host)
        topk_weights, topk_indices = self.gate(x)

        # 2. Expert computation (expert-parallel: only local experts computed)
        expert_output = self.experts(x, topk_weights, topk_indices)

        # 3. Shared expert (always active on all tokens)
        shared_output = self.shared_expert(x)

        # 4. Apply shared expert gate if present (scales shared expert contribution)
        if self.shared_expert_gate is not None:
            shared_output = ttnn.multiply(shared_output, self.shared_expert_gate)

        # 5. Combine expert output + gated shared expert output
        combined = ttnn.add(expert_output, shared_output)
        ttnn.deallocate(expert_output)
        ttnn.deallocate(shared_output)

        return combined
