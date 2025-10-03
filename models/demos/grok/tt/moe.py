# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from ttnn import ReplicateTensorToMesh, ShardTensorToMesh


class TtGrokMoeLayer(LightweightModule):
    def __init__(self, mesh_device, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()

        # Gate weight name for Grok
        gate_name = f"model.layers.{layer_num}.block_sparse_moe.gate.weight"

        # Simplified cache naming for minimal implementation
        cache_name = None

        self.num_devices = 8
        # Prepare gate tensor - pad to 64 for top-k operation compatibility
        gates_tensor = (
            torch.nn.functional.pad(state_dict[gate_name].permute(1, 0), (0, 56), "constant", 0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Create device-specific gate tensors (same pattern as Mixtral)
        gates_tensor_list = []
        for dev in range(self.num_devices):
            i, j = 0, dev
            gates_tensor_dev = gates_tensor.clone()
            gates_tensor_dev[:, :, :, [i, j]] = gates_tensor_dev[:, :, :, [j, i]]
            gates_tensor_list.append(gates_tensor_dev)

        self.gates_H8 = ttnn.as_tensor(
            torch.cat(gates_tensor_list, dim=1),
            dtype=ttnn.bfloat16,
            layout=self.model_config["GATE_W_LAYOUT_TILE_EXPERTS"],
            memory_config=self.model_config["GATE_WEIGHTS_MEMCFG_EXPERTS"],
            # cache_file_name=cache_name,  # Simplified for minimal implementation
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )

        self.tile_size = 32
        self.compute_kernel = args.compute_kernel_config_hifi2
        self.compute_kernel_reduce = args.compute_kernel_config_hifi2

        # Top-8 mask for expert selection (mask out experts beyond 8)
        top8_mask = torch.full((1, 1, 1, 64), fill_value=torch.finfo(torch.float).min)
        top8_mask[:, :, :, :8] = 0.0
        self.top8_mask_11B_64 = ttnn.from_torch(
            top8_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.top8_mask_11B_64 = ttnn.sum(self.top8_mask_11B_64, dim=2, keepdim=True)

        # Top-2 mask for final expert selection (Grok uses top-2 experts per token)
        top2_mask = torch.full((1, 1, 1, 32), fill_value=torch.finfo(torch.float).min)
        top2_mask[:, :, :, :2] = 0.0
        self.top2_mask_11BB = ttnn.from_torch(
            top2_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.top2_mask_11BB = ttnn.sum(self.top2_mask_11BB, dim=2, keepdim=True)

        # Reduction mask for final output aggregation
        reduce_mask_torch = torch.zeros(1, 1, self.tile_size, self.tile_size * 8)
        for i in range(self.tile_size):
            reduce_mask_torch[:, :, i, range(i, self.tile_size * 8, self.tile_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, inputs):
        """
        Grok MoE forward pass (decode mode only)
        Simplified version without prefill mode
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts

        # Get gate logits for expert selection
        gate_logits_1SB8 = ttnn.matmul(
            input_i_1SBH,
            self.gates_H8,
            memory_config=self.model_config["GATE_MM_OUTPUT_MEMCFG_EXPERTS"],
            compute_kernel_config=self.model_config["GATE_MM_OUTPUT_KERNEL_CONFIG_EXPERTS"],
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
        )

        # Mask out experts beyond the 8 available experts
        gate_logits_1SB8 = ttnn.add(gate_logits_1SB8, self.top8_mask_11B_64)

        # Get top-2 expert weights (decode mode only)
        weights_1SB1 = ttnn.moe(gate_logits_1SB8, self.top8_mask_11B_64, self.top2_mask_11BB, 32)

        gate_logits_1SB8.deallocate()

        # Apply expert MLP
        expert_output = expert_i_HH(input_i_1SBH)

        # Apply expert weights
        results_11BH = ttnn.mul(expert_output, weights_1SB1)

        expert_output.deallocate(True)
        weights_1SB1.deallocate(True)

        # Simplified all-gather and reduction for decode mode
        # Note: This is a simplified version - in practice you might need more sophisticated
        # communication patterns depending on your mesh configuration
        output_11BH_gathered = results_11BH  # Simplified - no actual all-gather for minimal implementation

        # Final reduction
        output_11BH_reduced = ttnn.matmul(
            self.reduce_mask, output_11BH_gathered, compute_kernel_config=self.compute_kernel_reduce
        )

        return output_11BH_reduced
