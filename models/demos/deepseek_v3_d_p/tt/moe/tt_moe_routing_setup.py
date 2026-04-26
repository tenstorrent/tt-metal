# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MoE Routing Setup Module (TTNN Implementation)

This module prepares the routing metadata (per-expert dispatch offsets and token counts)
required by the MoE dispatch stage. It sits between the gate (which produces top-k expert
indices) and the dispatch module (TtDispatchModule) which physically moves tokens across chips.

This is a TTNN wrapper around two device operations executed in sequence:

  1. masked_bincount — Builds a per-expert histogram counting how many tokens each expert
     receives from the local chip. Uses a height-sharded parallel algorithm where BRISC and
     NCRISC cooperatively count on each core, followed by a binary-tree reduction across
     cores. The expert_dispatch_table acts as a mask: experts with mask[e] < 0 are not
     present on this device and are skipped.

     Input:  ttnn_top_k_experts_indices
               Shape per device: (seq_len_per_chip, num_experts_per_tok), uint16, height-sharded
               across an 8x8 core grid (64 cores). Padded to min width of 8 if
               num_experts_per_tok < 8 (L1 alignment = 16 bytes / 2 bytes per uint16).
             experts_in_dispatch_group
               Shape per device: (num_routed_experts,), int32, interleaved DRAM
               Mask where values >= 0 indicate experts present on this device, -1 means absent.
     Output: expert_histograms
               Shape per device: (num_routed_experts,), uint32, interleaved DRAM
               Token count per expert for this chip.

  2. offset_cumsum — First reshapes each device's histogram to (1, num_routed_experts) and
     all-gathers along cluster_axis=0 (SP axis) to produce a gathered tensor of shape
     (dispatch_group_size, num_routed_experts) on every device. Then runs a device kernel
     that computes, for each source device, the starting token index in the destination
     device's local dispatch buffer where that source device must write its own tokens for
     each expert.

     Each device in the dispatch group has its own local dispatch buffer with shape
     (max_dispatch_buffer_token_size, emb_dim). Source devices write
     tokens into the expert's designated region of the destination device's dispatch buffer:
     either locally via NOC (if the expert is hosted on the same device) or remotely via
     fabric (if the expert is on a different device in the dispatch group). Multiple source
     devices can send tokens for the same expert, so each source device needs a unique
     non-overlapping write position. The offset combines two components:

     Component 1 — Local offset (unique per source device):
       For source device at position k in the dispatch group:
       local_offset_k[e] = sum of histogram[d, e] for d in 0..k-1.
       Device 0 starts at token index 0. Device 1 starts after device 0's tokens. Device k
       starts after all tokens from devices 0..k-1. This partitions the expert's token
       indices among all source devices without overlap. Each source device holds a
       distinct value.

     Component 2 — Expert region offset (shared across all source devices):
       Within the destination device's flat dispatch buffer, each of the experts_per_chip
       experts it hosts occupies a contiguous region of token indices. The
       expert_region_offset[e] is the starting token index of expert e's region within
       that flat buffer:
       expert_region_offset[e] = sum of ceil(total[j] / TILE_HEIGHT) * TILE_HEIGHT for all
       experts j that are hosted on the same destination device and precede e.
       total[j] = total tokens across all source devices for expert j (= global sum).
       TILE_HEIGHT (32) alignment ensures each expert's region starts at a tile boundary,
       enabling efficient hardware access patterns. All source devices compute the same
       expert_region_offset since it depends only on the global totals, which are
       identical after the all-gather.

     global_offset_k[e] = local_offset_k[e] + expert_region_offset[e].
     Source device k writes its tokens for expert e starting at token index
     global_offset_k[e] in the destination device's local dispatch buffer.

     Input:  expert_histograms (from step 1)
               Shape per device: (num_routed_experts,), uint32
               Internally reshaped to (1, num_routed_experts) and all-gathered to
               (dispatch_group_size, num_routed_experts).
     Output: global_dispatch_offsets
               Shape per device: (1, num_routed_experts), uint32, interleaved DRAM
               Each source device k holds a unique offset per expert: global_offset_k[e]
               = local_offset_k[e] + expert_region_offset[e]. This is the starting token
               index into the destination device's local dispatch buffer
               (max_dispatch_buffer_token_size, emb_dim) where source
               device k writes its tokens for expert e. Different source devices obtain
               distinct starting token indices for the same expert (via local_offset), and
               different experts hosted on the same destination device start at
               TILE_HEIGHT-aligned token indices (via expert_region_offset).
               Used as tt_expert_offsets in TtDispatchModule.forward() and
               TtCombineModule.forward().
             total_counts_per_expert
               Shape per device: (1, num_routed_experts), uint32, interleaved DRAM
               Total token count per expert summed across all dispatch_group_size devices.
               Identical across all devices within a dispatch group.
             expert_region_offsets
               Shape per device: (1, num_routed_experts), uint32, interleaved DRAM
               Only the expert region component of global_dispatch_offsets (shared across
               all source devices in a dispatch group). Useful when callers need the
               destination-side expert region layout without the per-source-device local
               offset mixed in.
"""

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMoERoutingSetup(LightweightModule):
    """Prepares routing metadata (offsets, token counts) for MoE dispatch.

    Given the top-k expert indices produced by the gate, this module:
      1. Builds per-expert histograms (token counts) via masked_bincount
      2. Computes cross-chip cumulative offsets via offset_cumsum

    The outputs feed directly into TtDispatchModule.forward() and
    TtCombineModule.forward().
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        expert_dispatch_table: torch.Tensor,
        num_links: int = 1,
        experts_per_chip: int = 32,
    ):
        """
        Initialize routing setup with the expert-to-chip mapping.

        Args:
            mesh_device: The mesh device to place tensors on
            expert_dispatch_table: Mapping of expert IDs to chip IDs within each dispatch group.
                Used as a mask in masked_bincount so each chip only counts experts it is responsible for.
                Values >= 0 indicate the destination chip ID for that expert; -1 means the expert is
                absent from this dispatch group.
                Shape: (num_dispatch_groups, num_routed_experts), int32
                Sharded across mesh with dims=(None, 0): replicated across rows (dispatch axis),
                sharded on dim 0 across columns (dispatch groups).
                Per-device shape after sharding: (1, num_routed_experts)
            num_links: Number of fabric links to use for cross-chip communication in offset_cumsum
            experts_per_chip: Number of experts per chip (for expert region offset grouping in offset_cumsum)
        """
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.experts_per_chip = experts_per_chip

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
        """
        Compute per-expert dispatch offsets and token counts from top-k expert indices.

        Args:
            ttnn_top_k_experts_indices: Top-k expert indices from the gate.
                Shape per device: (seq_len_per_chip, num_experts_per_tok), uint16
                Can be a torch.Tensor (will be converted and sharded automatically) or
                a pre-sharded ttnn.Tensor.
            num_routed_experts: Total number of routed experts across all chips (e.g. 64 or 256)
            seq_len_per_chip: Number of tokens per chip. Must be divisible by 64 (the 8x8 core grid
                used for height sharding in masked_bincount).
            num_experts_per_tok: Number of experts each token is routed to (e.g. 2 for top-2 routing)

        Returns:
            global_dispatch_offsets: Starting token index per expert for this source device.
                Shape per device: (1, num_routed_experts), uint32
                For expert e: global_offset_k[e] = local_offset_k[e] + expert_region_offset[e].
                Indexes into the destination device's flat dispatch buffer
                (max_dispatch_buffer_token_size, emb_dim).
                See module docstring for full offset semantics.
            total_counts_per_expert: Total token count per expert across all devices in the dispatch group.
                Shape per device: (1, num_routed_experts), uint32
                Identical across all devices within a dispatch group.
            expert_region_offsets: Expert region component of global_dispatch_offsets
                (shared across all source devices in a dispatch group).
                Shape per device: (1, num_routed_experts), uint32
            expert_histograms: Per-device token count per expert (before cross-chip aggregation).
                Shape per device: (num_routed_experts,), uint32
        """
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
                value=(num_routed_experts + 1),
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

        # Constraint imposed by masked_bincount
        if len(self.experts_in_dispatch_group.shape) != 1:
            assert (
                self.experts_in_dispatch_group.shape[0] == 1
            ), "Expected first dimension to be 1 after sharding expert dispatch table"
        logger.debug(f"{self.experts_in_dispatch_group.shape=}")

        expert_histograms = ttnn.experimental.deepseek_prefill.masked_bincount(
            ttnn_top_k_experts_indices, self.experts_in_dispatch_group, num_routed_experts, num_experts_per_tok
        )

        (
            global_dispatch_offsets,
            total_counts_per_expert,
            expert_region_offsets,
        ) = ttnn.experimental.deepseek_prefill.offset_cumsum(
            expert_histograms,
            cluster_axis=0,
            num_links=self.num_links,
            experts_per_chip=self.experts_per_chip,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        signpost(header="moe_gate_calculate_global_dispatch_offsets")

        return global_dispatch_offsets, total_counts_per_expert, expert_region_offsets, expert_histograms
