# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule


@dataclass
class MoEGateConfig:
    # gate_params

    ccl_config = {}
    mm_configs = {}

    dim: int = 7168
    max_seq_len = 4096 * 32
    sp_dim = 4096
    n_routed_experts: int = 256
    n_shared_experts: int = 2
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 1.0
    score_func: str = "sigmoid"
    summed_experts_per_group: int = 2
    topk_groups: int = 4

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


class MoEGatePrefill(LightweightModule):
    """MoE gate module from DeepSeek-R1."""

    def __init__(self, config, mesh_device):
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.mesh_device = mesh_device
        self.seq_len_per_chip = config.sp_dim

        self.core_grid = config.core_grid

        self.n_routed_experts = config.n_routed_experts
        self.experts_per_chip = 8
        self.mm_compute_config = config.mm_configs["DEFAULT_COMPUTE_CONFIG"]
        self.mm_program_config = config.mm_configs["DEFAULT_PROGRAM_CONFIG"]
        self.ccl_config = config.ccl_config

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

        self.experts_in_dispatch_row = ttnn.from_torch(
            torch.ones(config.n_routed_experts, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.expert_index_sharded_mem_config = ttnn.create_sharded_memory_config(
            shape=(config.sp_dim // 64, self.topk),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

    def forward(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        signpost(header="moe_gate_linear_allreduce")
        logits = ttnn.matmul(
            x,
            self.weight,
            compute_kernel_config=self.mm_compute_config,
            program_config=self.mm_program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        logits = ttnn.experimental.all_reduce_async(
            logits,
            cluster_axis=self.ccl_config["TP_AXIS"],
            mesh_device=self.mesh_device,
            num_links=self.ccl_config["NUM_LINKS"],
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        signpost(header="moe_gate_linear_allreduce")

        signpost(header="moe_gate_deepseek_grouped_gate")
        ttnn_scores, ttnn_top_k_experts_indices = ttnn.experimental.deepseek_grouped_gate(
            logits,
            self.bias,
            n_groups=self.n_groups,
            summed_experts_per_group=2,
            topk_groups=self.topk_groups,
            n_activated_experts=self.topk,
            route_scale=self.route_scale,
            epsilon=1e-20,
        )
        signpost(header="moe_gate_deepseek_grouped_gate")

        signpost(header="moe_gate_calculate_dispatch_offsets")
        ttnn_top_k_experts_indices = ttnn.to_layout(ttnn_top_k_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        ttnn_top_k_experts_indices = ttnn.to_memory_config(
            ttnn_top_k_experts_indices, self.expert_index_sharded_mem_config
        )

        expert_histograms = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn_top_k_experts_indices, self.experts_in_dispatch_row, self.n_routed_experts
        )

        dispatch_offsets, total_counts_per_expert = ttnn.experimental.deepseek_prefill.offset_cumsum(
            expert_histograms,
            cluster_axis=self.ccl_config["DISPATCH_AXIS"],
            num_links=self.ccl_config["NUM_LINKS"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        signpost(header="moe_gate_calculate_dispatch_offsets")

        return (ttnn_scores, ttnn_top_k_experts_indices, logits, dispatch_offsets, total_counts_per_expert)


class MoEGate_prep_dispatch_combine(LightweightModule):
    """MoE gate module from DeepSeek-R1."""

    def __init__(self, mesh_device, expert_dispatch_table):
        self.mesh_device = mesh_device

        # expert_dispatch_table (Raw table shape: (1, 64))
        # int value => chip location; -1 not in this group
        self.experts_in_dispatch_group = ttnn.from_torch(
            (expert_dispatch_table >= 0).to(torch.uint32),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def forward(
        self,
        ttnn_top_k_experts_indices: ttnn.Tensor | torch.Tensor,
        dispatch_group_size: int,
        num_routed_experts: int,
        experts_per_chip: int,
        seq_len_per_chip: int,
        num_experts_per_tok: int,
    ):
        signpost(header="MoEGate_prep_dispatch_combine")
        # indices: Expert indices tensor (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)

        if isinstance(ttnn_top_k_experts_indices, torch.Tensor):
            # ttnn_top_k_experts_indices.shape
            # Shape([4096, 8])
            # ttnn_top_k_experts_indices.tensor_topology()
            mesh_mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device,
                mesh_shape=self.mesh_device.shape,
                dims=(0, None),  # shard cols; replicate rows
            )
            # TensorTopology(distribution_shape=MeshShape([4, 2]), placements=SmallVector([PlacementShard(0), PlacementReplicate()]), mesh_coords=MeshCoordinate([0, 0]), MeshCoordinate([0, 1]), MeshCoordinate([1, 0]), MeshCoordinate([1, 1]), MeshCoordinate([2, 0]), MeshCoordinate([2, 1]), MeshCoordinate([3, 0]), MeshCoordinate([3, 1]))
            ttnn_top_k_experts_indices = ttnn.from_torch(
                ttnn_top_k_experts_indices,
                device=self.mesh_device,
                dtype=ttnn.uint16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            )
        else:
            ttnn_top_k_experts_indices = ttnn.to_layout(ttnn_top_k_experts_indices, ttnn.ROW_MAJOR_LAYOUT)

        ttnn_top_k_experts_indices = ttnn.pad(
            ttnn_top_k_experts_indices,
            padding=((0, 0), (0, num_routed_experts - num_experts_per_tok)),
            value=257,
        )

        self.expert_index_sharded_mem_config = ttnn.create_sharded_memory_config(
            shape=(seq_len_per_chip // 64, num_routed_experts),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        ttnn_top_k_experts_indices = ttnn.to_memory_config(
            ttnn_top_k_experts_indices, self.expert_index_sharded_mem_config
        )

        # reference
        # expert_offsets: Base offset for each expert from each chip
        #     Shape: (dispatch_group_size, num_routed_experts)
        # shard (0,None) -> per chip (1,num_routed_experts)
        # expert_token_counts: Total tokens per expert per chip
        #     Shape: (dispatch_group_size, experts_per_chip)
        # cum_sum: Cumulative sum of token counts across chips
        #     Shape: (dispatch_group_size, num_routed_experts)

        # Torch Input shapes: x.shape=torch.Size([8, 3200, 7168])
        # Sharded to seq parallel => [1, 3200, 7168] per chip
        # unsqueeze here for simplicty
        if len(ttnn_top_k_experts_indices.shape) == 3:
            ttnn_top_k_experts_indices = ttnn.squeeze(ttnn_top_k_experts_indices, 0)
        logger.warning(f"{ttnn_top_k_experts_indices.shape=}")

        # expert table (dispatch_groups, routed_experts)
        # sharded to each dispatch group => (1, routed_experts)
        # [create_expert_dispatch_table] OUTPUT: table.shape=torch.Size([1, 64])
        # need to squeeze, as op doesn't support tensor rank 2.
        logger.error(f"{len(self.experts_in_dispatch_group.shape)}==1?")
        if len(self.experts_in_dispatch_group.shape) != 1:
            self.experts_in_dispatch_group = ttnn.squeeze(self.experts_in_dispatch_group, 0)
        logger.warning(f"{self.experts_in_dispatch_group.shape=}")

        expert_histograms = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn_top_k_experts_indices, self.experts_in_dispatch_group, num_routed_experts
        )

        dispatch_offsets, total_counts_per_expert = ttnn.experimental.deepseek_prefill.offset_cumsum(
            expert_histograms,
            cluster_axis=0,
            num_links=1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        signpost(header="moe_gate_calculate_dispatch_offsets")

        # return expert_offsets, expert_token_counts, cum_sum
        # return (ttnn_scores, ttnn_top_k_experts_indices, logits, dispatch_offsets, total_counts_per_expert)
        return dispatch_offsets, total_counts_per_expert, expert_histograms
