# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.grok.tt.ccl import tt_all_reduce
from ttnn import ReplicateTensorToMesh


def topk_router(g, mask, experts_per_token):
    compute_config = ttnn.init_device_compute_kernel_config(
        g.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    # Gate logit softcapping
    g = ttnn.div(g, 30.0)
    g = ttnn.tanh(g)
    g = ttnn.mul(g, 30.0)
    g = ttnn.add(g, mask)

    expert_weights = ttnn.softmax(g, dim=-1, numeric_stable=True, compute_kernel_config=compute_config)
    expert_weights, expert_indices = ttnn.topk(expert_weights, k=experts_per_token, dim=-1, sorted=True)
    router_scores = ttnn.scatter(ttnn.zeros_like(g), dim=-1, index=expert_indices, src=expert_weights)
    return router_scores, expert_weights, expert_indices


class TtMoE(LightweightModule):
    def __init__(self, mesh_device, tt_ccl, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()

        # Gate weight name for Grok
        gate_name = f"model.layers.{layer_num}.block_sparse_moe.gate.weight"

        # Simplified cache naming for minimal implementation
        cache_name = None

        # Prepare gate tensor - pad to 64 for top-k operation compatibility
        gates_tensor = (
            torch.nn.functional.pad(state_dict[gate_name].permute(1, 0), (0, 56), "constant", 0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.gates_H8 = ttnn.as_tensor(
            gates_tensor,
            dtype=ttnn.bfloat16,
            layout=self.model_config["GATE_W_LAYOUT_TILE_EXPERTS"],
            memory_config=self.model_config["GATE_WEIGHTS_MEMCFG_EXPERTS"],
            # cache_file_name=cache_name,  # Simplified for minimal implementation
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=(8, 4)),
        )

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

        self.tile_size = 32
        self.compute_kernel = args.compute_kernel_config_hifi2
        self.compute_kernel_reduce = args.compute_kernel_config_hifi2

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
        gate_logits_1SB8 = tt_all_reduce(
            gate_logits_1SB8,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            dim=2,
            use_composite=True,
            skip_reshape=True,
        )

        router_scores, expert_weights, expert_indices = topk_router(gate_logits_1SB8, self.top8_mask_11B_64, 2)
        router_scores = router_scores[:, :, :, :8]
        router_scores = ttnn.permute(router_scores, (0, 3, 2, 1))

        gate_logits_1SB8.deallocate()

        # Apply expert MLP
        expert_output = expert_i_HH(input_i_1SBH)

        # Apply expert weights
        results_11BH = ttnn.mul(expert_output, router_scores)
        results_11BH = ttnn.sum(results_11BH, dim=1, keepdim=True)
        ttnn.deallocate(expert_output)

        return results_11BH
