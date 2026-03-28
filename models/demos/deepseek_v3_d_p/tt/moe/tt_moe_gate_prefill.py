# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, extract_mesh_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup


@dataclass
class TtMoEGateConfig:
    # gate_params

    ccl_config = {}
    mm_configs = {}

    dim: int = DeepSeekV3Config.EMB_SIZE
    max_seq_len = 4096 * 32
    sp_dim = 4096
    n_routed_experts: int = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    n_shared_experts: int = DeepSeekV3Config.NUM_SHARED_EXPERTS  # PREVIOUS VALUE: 2 @ddjekic to check
    n_activated_experts: int = DeepSeekV3Config.NUM_EXPERTS_PER_TOKEN
    n_expert_groups: int = DeepSeekV3Config.NUM_EXPERT_GROUPS
    n_limited_groups: int = DeepSeekV3Config.NUM_LIMITED_GROUPS  # = topk_groups
    route_scale: float = DeepSeekV3Config.ROUTE_SCALE  # PREVIOUS VALUE: 1.0 @ddjekic to check
    score_func: str = "sigmoid"

    # grid_config
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
    num_cores = 110

    mm_configs["DEFAULT_PROGRAM_CONFIG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        in0_block_w=56,
        out_subblock_h=2,
        out_subblock_w=4,
        out_block_h=2,
        out_block_w=4,
        per_core_M=2,
        per_core_N=8,
        fuse_batch=True,
        mcast_in0=False,
    )
    mm_configs["DEFAULT_COMPUTE_CONFIG"] = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    ccl_config["DISPATCH_AXIS"] = 0
    ccl_config["TP_AXIS"] = 1
    ccl_config["NUM_LINKS"] = 2


class TtMoEGatePrefill(LightweightModule):
    """MoE gate module from DeepSeek-R1."""

    def __init__(self, config, mesh_device):
        self.config = config
        self.mesh_device = mesh_device

        self.weight = ttnn.from_torch(
            torch.zeros([config.dim, config.n_routed_experts]),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0),
                mesh_shape=mesh_device.shape,
            ),
        )

        self.bias = ttnn.from_torch(
            # ttnn.experimental.deepseek_grouped_gate() requires bias to be broadcasted already
            torch.zeros([config.n_routed_experts]).repeat(config.sp_dim).view(config.sp_dim, -1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        mesh_config = extract_mesh_config(mesh_device)
        dispatch_table = ExpertMapping.create_dispatch_table(
            num_routed_experts=config.n_routed_experts,
            dispatch_group_size=mesh_config.dispatch_group_size,
            num_dispatch_groups=mesh_config.num_dispatch_groups,
        )
        self.routing_setup = TtMoERoutingSetup(mesh_device, dispatch_table, num_links=config.ccl_config["NUM_LINKS"])

    def forward(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        signpost(header="moe_gate_linear_allreduce")
        logits = ttnn.matmul(
            x,
            self.weight,
            compute_kernel_config=self.config.mm_configs["DEFAULT_COMPUTE_CONFIG"],
            program_config=self.config.mm_configs["DEFAULT_PROGRAM_CONFIG"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        logits = ttnn.experimental.all_reduce_async(
            logits,
            cluster_axis=self.config.ccl_config["TP_AXIS"],
            mesh_device=self.mesh_device,
            num_links=self.config.ccl_config["NUM_LINKS"],
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        signpost(header="moe_gate_linear_allreduce")

        signpost(header="moe_gate_deepseek_grouped_gate")
        ttnn_scores, ttnn_top_k_experts_indices = ttnn.experimental.deepseek_grouped_gate(
            logits,
            self.bias,
            n_groups=self.config.n_expert_groups,
            summed_experts_per_group=self.config.n_expert_groups // self.config.n_limited_groups,
            topk_groups=self.config.n_limited_groups,
            n_activated_experts=self.config.n_activated_experts,
            route_scale=self.config.route_scale,
            epsilon=1e-20,
        )
        signpost(header="moe_gate_deepseek_grouped_gate")

        signpost(header="moe_gate_calculate_dispatch_offsets")
        ttnn_top_k_experts_indices = ttnn.to_layout(ttnn_top_k_experts_indices, ttnn.ROW_MAJOR_LAYOUT)

        dispatch_offsets, total_counts_per_expert, _ = self.routing_setup(
            ttnn_top_k_experts_indices=ttnn_top_k_experts_indices,
            num_routed_experts=self.config.n_routed_experts,
            seq_len_per_chip=self.config.sp_dim,
            num_experts_per_tok=self.config.n_activated_experts,
        )
        signpost(header="moe_gate_calculate_dispatch_offsets")

        return (ttnn_scores, ttnn_top_k_experts_indices, logits, dispatch_offsets, total_counts_per_expert)
