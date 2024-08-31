# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh
from models.common.lightweightmodule import LightweightModule


class TtMoeLayer(LightweightModule):
    def __init__(self, mesh_device, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()

        gate_name = f"layers.{layer_num}.feed_forward.gate.weight"
        if args.dummy_weights:
            cache_name = None
        else:
            cache_name = args.weight_cache_path(dtype) / (gate_name + "_multidevice_repadded")

        self.num_devices = 8
        # make the index of the expert on each devices equal to zero
        gates_tensor = (
            torch.nn.functional.pad(state_dict[gate_name].permute(1, 0), (0, 56), "constant", 0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gates_tensor_list = []
        for dev in range(self.num_devices):
            i, j = 0, dev
            gates_tensor_dev = gates_tensor.clone()
            gates_tensor_dev[:, :, :, [i, j]] = gates_tensor_dev[:, :, :, [j, i]]
            gates_tensor_list.append(gates_tensor_dev)

        self.gates_H8 = ttnn.as_tensor(
            torch.cat(gates_tensor_list, dim=1),
            dtype=ttnn.bfloat16,
            layout=self.model_config["GATE_W_LAYOUT_TILE"],
            memory_config=self.model_config["GATE_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )

        self.tile_size = 32
        self.compute_kernel = args.get_compute_kernel_attn_config()

        top8_mask = torch.full((1, 1, 1, 64), fill_value=torch.finfo(torch.float).min)
        top8_mask[:, :, :, :8] = 0.0
        self.top8_mask_11B_64 = ttnn.from_torch(
            top8_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.top8_mask_11B_64 = ttnn.sum(self.top8_mask_11B_64, dim=2)

        top2_mask = torch.full((1, 1, 1, 32), fill_value=torch.finfo(torch.float).min)
        top2_mask[:, :, :, :2] = 0.0
        self.top2_mask_11BB = ttnn.from_torch(
            top2_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.top2_mask_11BB = ttnn.sum(self.top2_mask_11BB, dim=2)

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

    def forward(self, inputs, mode="decode"):
        """
        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        S : seq len
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts
        # get logits for the experts
        gate_logits_1SB8 = ttnn.matmul(
            input_i_1SBH,
            self.gates_H8,
            memory_config=self.model_config["GATE_MM_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_config["GATE_MM_OUTPUT_KERNEL_CONFIG"],
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
        )

        # get weights for top-2 experts -- masking out everything except the 8 experts (needed because top-k works with a min input of size 64)
        gate_logits_1SB8 = ttnn.add(gate_logits_1SB8, self.top8_mask_11B_64)

        if mode == "decode":
            weights_1SB1 = ttnn.moe(gate_logits_1SB8, self.top8_mask_11B_64, self.top2_mask_11BB, 32)
        else:
            topk_values, topk_indices = ttnn.topk(gate_logits_1SB8, 32)
            topk_values = ttnn.add(topk_values, self.top2_mask_11BB)
            mask_B2 = ttnn.eqz(topk_indices)
            weights_1SB1 = ttnn.sum(ttnn.softmax(topk_values, dim=-1) * mask_B2, dim=3)

            topk_values.deallocate(True)
            topk_indices.deallocate(True)
            mask_B2.deallocate(True)

        gate_logits_1SB8.deallocate()

        # MLP and masking
        weights = expert_i_HH(input_i_1SBH, mode=mode)

        results_11BH = ttnn.mul(weights, weights_1SB1)

        weights.deallocate(True)
        weights_1SB1.deallocate(True)

        # All gather
        if mode == "prefill":
            output_11BH_gathered = ttnn.all_gather(results_11BH, dim=1, num_links=1)
            results_11BH.deallocate(True)
            # Sum reduction
            output_11BH_reduced = ttnn.experimental.fast_reduce_nc(
                output_11BH_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            output_11BH_gathered.deallocate(True)
        else:  # Decode mode
            output_11BH_gathered = ttnn.all_gather(results_11BH, dim=2, num_links=1)
            results_11BH.deallocate(True)
            # Reduction
            output_11BH_reduced = ttnn.matmul(self.reduce_mask, output_11BH_gathered)

        return output_11BH_reduced
