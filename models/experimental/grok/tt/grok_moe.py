# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh
from models.experimental.grok.tt.grok_common import LightweightModule
from models.experimental.grok.scripts.tlog import tlog, tlog_mesh_device


class TtMoeLayer(LightweightModule):
    def __init__(self, mesh_device, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        tlog_mesh_device = mesh_device
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()

        gate_name = f"model.layers.{layer_num}.moe_block.gate.weight"
        if args.dummy_weights:
            cache_name = None
        else:
            cache_name = args.weight_cache_path(dtype) / (gate_name)

        self.gates_H_64 = ttnn.as_tensor(
            torch.nn.functional.pad(state_dict[gate_name].permute(1, 0), (1, 55), "constant", 0)
            .unsqueeze(0)
            .unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=self.model_config["GATE_W_LAYOUT_TILE"],
            memory_config=self.model_config["GATE_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        self.num_devices = 8
        self.compute_kernel = args.get_compute_kernel_attn_config()

        reduce_mask_torch = torch.zeros(1, 1, self.args.max_batch_size, self.args.max_batch_size * 8)
        for i in range(self.args.max_batch_size):
            reduce_mask_torch[:, :, i, range(i, self.args.max_batch_size * 8, self.args.max_batch_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.expert_mask_11BB = ttnn.from_torch(
            torch.cat([torch.full((1, 1, 32, 32), fill_value=i + 1) for i in range(8)], dim=3),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
        )
        top8_mask = torch.full((1, 1, 32, 64), fill_value=torch.finfo(torch.float).min)
        top8_mask[:, :, :, 1:9] = 0.0
        self.top8_mask_11B_64 = ttnn.from_torch(
            top8_mask,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        top2_mask = torch.full((1, 1, 32, 32), fill_value=0.0)
        top2_mask[:, :, :, :2] = 1.0
        self.top2_mask_11BB = ttnn.from_torch(
            top2_mask,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.softmax_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.softmax_program_config = ttnn.SoftmaxDefaultProgramConfig()

    def forward(self, inputs):
        """
        inputs: (seq_len, 1, batch, hidden_dim)

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (6144)
        S : seq len (1)
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts
        # get logits for the experts
        gate_logits_1SB_64 = ttnn.matmul(
            input_i_1SBH,
            self.gates_H_64,
            program_config=self.model_config["GATE_MM_OUTPUT_PROGCFG"],
            memory_config=self.model_config["GATE_MM_OUTPUT_MEMCFG"],
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat16,
        )

        # get weights for top-2 experts
        gate_logits_1SB_64 = ttnn.add(gate_logits_1SB_64, self.top8_mask_11B_64)
        # tlog('our_gate_logits', gate_logits_1SB_64)

        # Grok does softmax before top-k, seems wrong but ¯\_(ツ)_/¯
        # gate_probs_1SB_64 = ttnn.softmax(gate_logits_1SB_64, dim=-1)
        # del gate_logits_1SB_64

        gate_probs_1SB_64 = ttnn.scale_mask_softmax_in_place(
            gate_logits_1SB_64,
            program_config=self.softmax_program_config,
            compute_kernel_config=self.softmax_compute_config,
        )
        # tlog('our_gate_probs', gate_probs_1SB_64)

        ttl_topk_values, ttl_topk_indices = ttnn.topk(gate_probs_1SB_64, 32)  # selects 6, 5 as 8.1, 1.8
        # tlog('our_topk_indices', ttl_topk_indices)
        ttl_topk_values = ttl_topk_values * self.top2_mask_11BB  # masked unwanted ones to 0
        mask_B2 = ttnn.eq(self.expert_mask_11BB, ttl_topk_indices)  # Each device masks for its own expert index 1-8
        weights_1SB1 = ttnn.sum(ttl_topk_values * mask_B2, dim=3)

        # MLP and masking
        weights = expert_i_HH(input_i_1SBH)
        # tlog('our_expert_output', weights)
        results_11BH = ttnn.mul(weights, weights_1SB1)
        # tlog('our_weighted_expert_output', weights)

        # all gather
        output_11BH_gathered = ttnn.all_gather(results_11BH, dim=2, num_links=1)
        # sum on each device
        output_11BH_gathered = ttnn.matmul(self.reduce_mask, output_11BH_gathered)
        return output_11BH_gathered
