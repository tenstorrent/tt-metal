# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Expert MLP bank for Qwen3-Coder-Next MoE (LightweightModule / ttnn).

Expert-parallel strategy: 512 experts / 8 devices = 64 experts per device.
Each device holds a contiguous shard: device_id * 64 .. (device_id + 1) * 64 - 1.

Each expert is a SwiGLU FFN: hidden_size (2048) -> moe_intermediate_size (512) -> hidden_size.

State dict keys:
    model.layers.{layer}.mlp.experts.{expert_id}.gate_proj.weight  (512, 2048)
    model.layers.{layer}.mlp.experts.{expert_id}.up_proj.weight    (512, 2048)
    model.layers.{layer}.mlp.experts.{expert_id}.down_proj.weight  (2048, 512)
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class ExpertMLPBank(LightweightModule):
    """Bank of local experts for one device in expert-parallel MoE.

    Holds 64 experts (contiguous shard) and routes tokens to local experts only.
    Cross-device routing is handled by the MoELayer.
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, device_id, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.device_id = device_id
        self.num_local_experts = args.num_experts // 8  # 64
        self.expert_offset = device_id * self.num_local_experts  # global index of first local expert
        self.hidden_size = args.hidden_size  # 2048
        self.intermediate_size = args.moe_intermediate_size  # 512

        # Support both with and without "model." prefix
        state_dict_prefix = f"model.layers.{layer_num}.mlp.experts"
        test_key = f"{state_dict_prefix}.0.gate_proj.weight"
        if test_key not in state_dict:
            state_dict_prefix = f"layers.{layer_num}.mlp.experts"

        # Load weights for each local expert
        # Store as lists of ttnn tensors on device
        self.gate_proj_weights = []
        self.up_proj_weights = []
        self.down_proj_weights = []

        for local_idx in range(self.num_local_experts):
            global_idx = self.expert_offset + local_idx
            expert_prefix = f"{state_dict_prefix}.{global_idx}"

            # gate_proj: (intermediate_size, hidden_size) -> transpose to (hidden_size, intermediate_size)
            gate_w = state_dict[f"{expert_prefix}.gate_proj.weight"].transpose(-2, -1).contiguous()
            gate_w = gate_w.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size, intermediate_size)

            # up_proj: (intermediate_size, hidden_size) -> transpose to (hidden_size, intermediate_size)
            up_w = state_dict[f"{expert_prefix}.up_proj.weight"].transpose(-2, -1).contiguous()
            up_w = up_w.unsqueeze(0).unsqueeze(0)

            # down_proj: (hidden_size, intermediate_size) -> transpose to (intermediate_size, hidden_size)
            down_w = state_dict[f"{expert_prefix}.down_proj.weight"].transpose(-2, -1).contiguous()
            down_w = down_w.unsqueeze(0).unsqueeze(0)  # (1, 1, intermediate_size, hidden_size)

            self.gate_proj_weights.append(
                ttnn.as_tensor(
                    gate_w,
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            )
            self.up_proj_weights.append(
                ttnn.as_tensor(
                    up_w,
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            )
            self.down_proj_weights.append(
                ttnn.as_tensor(
                    down_w,
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            )

    def forward(self, x, routing_weights, selected_experts):
        """Run local experts on routed tokens and accumulate weighted outputs.

        Args:
            x: ttnn tensor (1, 1, batch*seq, hidden_size) on device.
            routing_weights: torch tensor (batch*seq, num_experts_per_tok) — normalized probs.
            selected_experts: torch tensor (batch*seq, num_experts_per_tok) — global expert indices.

        Returns:
            ttnn tensor (1, 1, batch*seq, hidden_size) — accumulated expert outputs.
        """
        num_tokens = routing_weights.shape[0]

        # Transfer input to host for token selection (needed for scatter/gather routing)
        if hasattr(self, "mesh_device") and self.mesh_device.get_num_devices() > 1:
            x_torch = (
                ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0:1]
                .squeeze(0)
                .squeeze(0)
            )
        else:
            x_torch = ttnn.to_torch(x).squeeze(0).squeeze(0)  # (B*S, hidden_size)

        # Accumulate output on host then transfer back
        output = torch.zeros(num_tokens, self.hidden_size, dtype=x_torch.dtype)

        for local_idx in range(self.num_local_experts):
            global_idx = self.expert_offset + local_idx

            # Find tokens routed to this expert
            token_mask = selected_experts == global_idx  # (B*S, num_experts_per_tok)
            if not token_mask.any():
                continue

            token_indices, position_indices = token_mask.nonzero(as_tuple=True)
            weights = routing_weights[token_indices, position_indices]  # (num_routed,)

            # Get routed tokens and send to device
            expert_input = x_torch[token_indices]  # (num_routed, hidden_size)
            expert_input_4d = expert_input.unsqueeze(0).unsqueeze(0)  # (1, 1, num_routed, hidden_size)

            expert_input_tt = ttnn.from_torch(
                expert_input_4d,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            # SwiGLU: silu(gate_proj(x)) * up_proj(x) -> down_proj
            gate_out = ttnn.linear(expert_input_tt, self.gate_proj_weights[local_idx])
            gate_out = ttnn.silu(gate_out)

            up_out = ttnn.linear(expert_input_tt, self.up_proj_weights[local_idx])
            ttnn.deallocate(expert_input_tt)

            hidden = ttnn.multiply(gate_out, up_out)
            ttnn.deallocate(gate_out)
            ttnn.deallocate(up_out)

            expert_out_tt = ttnn.linear(hidden, self.down_proj_weights[local_idx])
            ttnn.deallocate(hidden)

            # Transfer back to host and accumulate with routing weights
            if hasattr(self, "mesh_device") and self.mesh_device.get_num_devices() > 1:
                expert_out = (
                    ttnn.to_torch(expert_out_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0:1]
                    .squeeze(0)
                    .squeeze(0)
                )
            else:
                expert_out = ttnn.to_torch(expert_out_tt).squeeze(0).squeeze(0)  # (num_routed, hidden_size)
            ttnn.deallocate(expert_out_tt)

            weighted_out = expert_out * weights.unsqueeze(-1)
            output.index_add_(0, token_indices, weighted_out.to(output.dtype))

        # Convert accumulated output back to ttnn tensor on device
        output_4d = output.unsqueeze(0).unsqueeze(0)  # (1, 1, B*S, hidden_size)
        output_tt = ttnn.from_torch(
            output_4d,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return output_tt
