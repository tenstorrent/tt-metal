# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMoERoutingSetup(LightweightModule):
    """Prepares routing metadata (offsets, token counts) for MoE dispatch."""

    def __init__(self, mesh_device, expert_dispatch_table, num_links=1):
        self.mesh_device = mesh_device
        self.num_links = num_links

        self.experts_in_dispatch_group = ttnn.from_torch(
            expert_dispatch_table,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(None, 0)),
        )

    def forward(
        self,
        ttnn_top_k_experts_indices: ttnn.Tensor | torch.Tensor,
        num_routed_experts: int,
        seq_len_per_chip: int,
        num_experts_per_tok: int,
    ):
        signpost(header="MoERoutingSetup")

        if isinstance(ttnn_top_k_experts_indices, torch.Tensor):
            mesh_mapper = ttnn.ShardTensor2dMesh(
                self.mesh_device,
                mesh_shape=self.mesh_device.shape,
                dims=(0, None),  # shard cols; replicate rows
            )
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

        L1_ALIGNMENT_BYTES = 16
        UINT16_BYTES = 2
        assert ttnn_top_k_experts_indices.dtype == ttnn.uint16, "Expected uint16 dtype for expert indices"

        min_shard_width = L1_ALIGNMENT_BYTES // UINT16_BYTES  # 8

        shard_width = num_experts_per_tok
        if num_experts_per_tok < min_shard_width:
            shard_width = min_shard_width
            ttnn_top_k_experts_indices = ttnn.pad(
                ttnn_top_k_experts_indices,
                padding=((0, 0), (0, shard_width - num_experts_per_tok)),
                value=257,
            )

        bincount_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        num_cores = bincount_core_grid.num_cores()
        assert (
            seq_len_per_chip % num_cores == 0
        ), f"seq_len_per_chip ({seq_len_per_chip}) must be divisible by num_cores ({num_cores}) for sharding to work correctly"

        self.expert_index_sharded_mem_config = ttnn.create_sharded_memory_config(
            shape=(seq_len_per_chip // num_cores, shard_width),
            core_grid=bincount_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        ttnn_top_k_experts_indices = ttnn.to_memory_config(
            ttnn_top_k_experts_indices, self.expert_index_sharded_mem_config
        )

        if len(ttnn_top_k_experts_indices.shape) == 3:
            ttnn_top_k_experts_indices = ttnn.squeeze(ttnn_top_k_experts_indices, 0)
        logger.debug(f"{ttnn_top_k_experts_indices.shape=}")

        # Squeeze to rank 1 — masked_bincount doesn't support rank 2
        if len(self.experts_in_dispatch_group.shape) != 1:
            assert (
                self.experts_in_dispatch_group.shape[0] == 1
            ), "Expected first dimension to be 1 after sharding expert dispatch table"
            self.experts_in_dispatch_group = ttnn.squeeze(self.experts_in_dispatch_group, 0)
        logger.debug(f"{self.experts_in_dispatch_group.shape=}")

        expert_histograms = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn_top_k_experts_indices, self.experts_in_dispatch_group, num_routed_experts, num_experts_per_tok
        )

        dispatch_offsets, total_counts_per_expert = ttnn.experimental.deepseek_prefill.offset_cumsum(
            expert_histograms,
            cluster_axis=0,
            num_links=self.num_links,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        signpost(header="moe_gate_calculate_dispatch_offsets")

        return dispatch_offsets, total_counts_per_expert, expert_histograms
