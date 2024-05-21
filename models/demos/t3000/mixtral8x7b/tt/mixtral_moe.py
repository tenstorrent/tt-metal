# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh


def top_2(gate_logits_1SB8, top_2_mask, expert_mask, ones_1118, ones_11B1, compute_kernel):
    # get the highest value and position
    weights_ex0_1SB1 = ttnn.max(gate_logits_1SB8, dim=3)
    exp_0_repeated = ttnn.matmul(weights_ex0_1SB1, ones_1118, dtype=ttnn.bfloat16, compute_kernel_config=compute_kernel)
    cond0 = ttnn.eq(gate_logits_1SB8, exp_0_repeated)

    # mask out the maximum value
    gate_logits_1SB8_masked = ttnn.where(cond0, top_2_mask, gate_logits_1SB8)

    # get the second highest value and position
    weights_ex1_1SB1 = ttnn.max(gate_logits_1SB8_masked, dim=3)
    exp_1_repeated = ttnn.matmul(weights_ex1_1SB1, ones_1118, compute_kernel_config=compute_kernel, dtype=ttnn.bfloat16)
    cond1 = ttnn.eq(gate_logits_1SB8, exp_1_repeated)

    # calculate the softmax
    weights_exp = ttnn.exp(weights_ex1_1SB1 - weights_ex0_1SB1)
    weights_1SB1_pre_softmax = ttnn.reciprocal(ones_11B1 + weights_exp)

    # select whether a batch for was selected first or second for the i-th head
    cond0 = ttnn.matmul(cond0, expert_mask, compute_kernel_config=compute_kernel, dtype=ttnn.bfloat16)
    cond1 = ttnn.matmul(cond1, expert_mask, compute_kernel_config=compute_kernel, dtype=ttnn.bfloat16)

    # calculate the weight
    weights_1SB1 = cond0 * weights_1SB1_pre_softmax - cond1 * (weights_1SB1_pre_softmax - ones_11B1)

    return weights_1SB1


class TtMoeLayer(torch.nn.Module):
    def __init__(self, device_mesh, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        self.device_mesh = device_mesh
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()

        gate_name = f"layers.{layer_num}.feed_forward.gate.weight"
        self.gates_H8 = ttnn.as_tensor(
            state_dict[gate_name].permute(1, 0),
            dtype=ttnn.bfloat16,
            layout=self.model_config["GATE_W_LAYOUT_TILE"],
            memory_config=self.model_config["GATE_WEIGHTS_MEMCFG"],
            cache_file_name=args.weight_cache_path(dtype) / (gate_name + "_multidevice"),
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        self.num_devices = 8
        self.compute_kernel = args.get_compute_kernel_attn_config()

        self.top_2_mask = ttnn.from_torch(
            torch.full(
                (1, 1, self.args.max_batch_size, self.args.num_experts), fill_value=torch.finfo(torch.float).min
            ),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        self.top_2_mask = ttnn.to_device(self.top_2_mask, device_mesh)

        expert_mask_torch = []
        for i in range(8):
            torch_tensor = torch.zeros(1, 1, self.args.num_experts, 1)
            torch_tensor[:, :, i, :] = 1
            expert_mask_torch.append(torch_tensor)
        expert_mask_torch = torch.cat(expert_mask_torch, dim=3)
        self.expert_mask = ttnn.from_torch(
            expert_mask_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
        )
        self.expert_mask = ttnn.to_device(self.expert_mask, device_mesh)

        self.ones_1118 = ttnn.from_torch(
            torch.ones(1, 1, 1, self.args.num_experts),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        self.ones_1118 = ttnn.to_device(self.ones_1118, device_mesh)

        self.ones_11B1 = ttnn.from_torch(
            torch.ones(1, 1, self.args.max_batch_size, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        self.ones_11B1 = ttnn.to_device(self.ones_11B1, device_mesh)

        reduce_mask_torch = torch.zeros(1, 1, self.args.max_batch_size, self.args.max_batch_size * 8)
        for i in range(self.args.max_batch_size):
            reduce_mask_torch[:, :, i, range(i, self.args.max_batch_size * 8, self.args.max_batch_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        self.reduce_mask = ttnn.to_device(self.reduce_mask, device_mesh)

    def forward(self, inputs):
        """
        inputs: (seq_len, 1, batch, hidden_dim)

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        S : seq len (1)
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts
        # get logits for the experts
        gate_logits_1SB8 = ttnn.linear(
            input_i_1SBH,
            self.gates_H8,
            memory_config=self.model_config["GATE_MM_OUTPUT_MEMCFG"],
            compute_kernel_config=self.compute_kernel,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            dtype=ttnn.bfloat16,
        )
        gate_logits_1SB8 = ttnn.to_device(gate_logits_1SB8, self.device_mesh)

        # get weights for top-2 experts
        weights_1SB1 = top_2(
            gate_logits_1SB8,
            self.top_2_mask,
            self.expert_mask,
            self.ones_1118,
            self.ones_11B1,
            self.compute_kernel,
        )

        # MLP and masking
        weights = expert_i_HH(input_i_1SBH)

        results_11BH = ttnn.mul(weights, weights_1SB1)

        # convert to bf8 for all-gather perf
        results_11BH = ttnn.clone(results_11BH, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)

        # all gather
        output_11BH_gathered = ttnn.all_gather(results_11BH, dim=2, num_links=1)
        # sum on each device
        output_11BH_gathered = ttnn.matmul(self.reduce_mask, output_11BH_gathered)
        return output_11BH_gathered
