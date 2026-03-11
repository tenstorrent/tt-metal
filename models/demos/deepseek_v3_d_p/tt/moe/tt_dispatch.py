# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Expert-centric MoE Dispatch Module (TTNN Implementation)

This module implements token dispatching for Mixture-of-Experts (MoE) layers using TTNN.

This is a TTNN wrapper around the prefill_dispatch operation, which performs the same
logic as the PyTorch reference implementation but on Tenstorrent hardware.
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtDispatchModule(LightweightModule):
    """Expert-centric MoE dispatch module."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dispatch_group_size: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        seq_len_per_chip: int,
        hidden_dim: int = 7 * 1024,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        """
        Initialize dispatch module with configuration parameters.

        Args:
            dispatch_group_size: Number of chips in each dispatch group
            experts_per_chip: Number of experts per chip
            num_routed_experts: Total number of routed experts across all chips
            metadata_len: Length of metadata per token (stores: chip, token, topk_idx, routed_expert, weight)
            max_dispatched_tokens_per_expert: Maximum number of tokens that can be dispatched to each expert
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.dispatch_group_size = dispatch_group_size
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.metadata_len = metadata_len
        self.max_dispatched_tokens_per_expert = max_dispatched_tokens_per_expert
        self.seq_len_per_chip = seq_len_per_chip
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology

    @staticmethod
    def shard_expert_offsets(
        mesh_device: ttnn.MeshDevice,
        expert_offsets: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Convert and shard the offset tensor for dispatch operation.

        Args:
            mesh_device: The mesh device to place the tensor on
            expert_offsets: Base offset for each expert from each chip
                Shape: (dispatch_group_size, num_routed_experts) - from get_gate_outputs()

        Returns:
            TTNN tensor sharded across mesh devices
        """
        logger.debug(f"[shard_expert_offsets] INPUT: expert_offsets.shape={expert_offsets.shape}")
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(0, None),
        )
        result = ttnn.from_torch(
            expert_offsets,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )
        logger.debug(f"[shard_expert_offsets] OUTPUT: result.shape={result.shape}")
        return result

    @staticmethod
    def shard_expert_dispatch_table(
        mesh_device: ttnn.MeshDevice,
        expert_dispatch_table: torch.Tensor,
        dispatch_axis: int,
    ) -> ttnn.Tensor:
        """
        Shard expert dispatch table: shard across dispatch groups, replicate across dispatch axis.

        The expert_dispatch_table maps expert IDs to destination chip IDs within the
        dispatch axis. It has shape (num_dispatch_groups, num_routed_experts) where:
        - Dim 0 is sharded across dispatch groups
        - Dim 1 is replicated across dispatch axis

        Args:
            mesh_device: Mesh device
            expert_dispatch_table: Shape (num_dispatch_groups, num_routed_experts)
            dispatch_axis: Dispatch axis (0 or 1)

        Returns:
            TTNN tensor sharded appropriately
        """
        # For dispatch_axis=0: mesh axis 0 = dispatch axis, mesh axis 1 = dispatch groups
        #   dims = (None, 0): replicate on mesh rows (dispatch), shard tensor dim 0 on mesh cols (groups)
        # For dispatch_axis=1: mesh axis 0 = dispatch groups, mesh axis 1 = dispatch axis
        #   dims = (0, None): shard tensor dim 0 on mesh rows (groups), replicate on mesh cols (dispatch)
        logger.debug(
            f"[shard_expert_dispatch_table] INPUT: expert_dispatch_table.shape={expert_dispatch_table.shape}, dispatch_axis={dispatch_axis}"
        )
        if dispatch_axis == 0:
            dims = (None, 0)
        else:
            dims = (0, None)
        logger.debug(f"[shard_expert_dispatch_table] Using dims={dims}")

        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=dims,
        )
        result = ttnn.from_torch(
            expert_dispatch_table,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )
        logger.debug(f"[shard_expert_dispatch_table] OUTPUT: result.shape={result.shape}")
        return result

    def forward(
        self,
        x: ttnn.Tensor,
        weights: ttnn.Tensor,
        indices: ttnn.Tensor,
        tt_expert_offsets: ttnn.Tensor,
        tt_expert_dispatch_table: ttnn.Tensor,
    ):
        """
        Route tokens from their original positions to expert-specific buffers distributed across chips.

        Simulates MoE dispatch: each token is routed to multiple experts based on router indices.
        Tokens are gathered into per-expert buffers with metadata tracking their origin for later recombination.

        Args:
            x: Input tensor of shape (dispatch_group_size, seq_len, hidden_dim)
            weights: Router weights of shape (dispatch_group_size, seq_len, num_experts_per_tok)
            indices: Expert indices of shape (dispatch_group_size, seq_len, num_experts_per_tok)
            tt_expert_offsets: Base offset for each expert from each chip (TTNN tensor)
                Shape: (dispatch_group_size, num_routed_experts) - use shard_expert_offsets() to create
            tt_expert_dispatch_table: Expert dispatch table mapping expert ID to chip ID (TTNN tensor)
                Shape: (num_dispatch_groups, num_routed_experts) - use shard_expert_dispatch_table() to create

        Returns:
            dispatched_buffer: Dispatched tokens of shape (dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
            metadata: Metadata tensor of shape (dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
        """
        logger.debug(f"[TtDispatchModule.forward] INPUT SHAPES:")
        logger.debug(f"  x.shape={x.shape}")
        logger.debug(f"  weights.shape={weights.shape}")
        logger.debug(f"  indices.shape={indices.shape}")
        logger.debug(f"  tt_expert_offsets.shape={tt_expert_offsets.shape}")
        logger.debug(f"  tt_expert_dispatch_table.shape={tt_expert_dispatch_table.shape}")
        logger.debug(f"[TtDispatchModule.forward] CONFIG:")
        logger.debug(f"  dispatch_group_size={self.dispatch_group_size}, experts_per_chip={self.experts_per_chip}")
        logger.debug(f"  num_routed_experts={self.num_routed_experts}, num_experts_per_tok={self.num_experts_per_tok}")
        logger.debug(
            f"  metadata_len={self.metadata_len}, max_dispatched_tokens_per_expert={self.max_dispatched_tokens_per_expert}"
        )
        logger.debug(f"  cluster_axis={self.cluster_axis}, num_links={self.num_links}, topology={self.topology}")

        (
            tt_dispatched_buffer,
            tt_dispatch_metadata,
        ) = ttnn.experimental.deepseek_prefill.dispatch(
            input_tensor=x,
            weights_tensor=weights,
            indices_tensor=indices,
            expert_offsets_tensor=tt_expert_offsets,
            expert_dispatch_table_tensor=tt_expert_dispatch_table,
            dispatch_group_size=self.dispatch_group_size,
            experts_per_chip=self.experts_per_chip,
            num_routed_experts=self.num_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            metadata_len=self.metadata_len,
            max_dispatched_tokens_per_expert=self.max_dispatched_tokens_per_expert,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
        )

        tt_dispatched_buffer_shape = tt_dispatched_buffer.shape
        tt_dispatched_metadata_shape = tt_dispatch_metadata.shape
        logger.debug(f"[TtDispatchModule.forward] OUTPUT SHAPES:")
        logger.debug(f"  tt_dispatched_buffer.shape={tt_dispatched_buffer_shape}")
        logger.debug(f"  tt_dispatch_metadata.shape={tt_dispatched_metadata_shape}")

        return (tt_dispatched_buffer, tt_dispatch_metadata)
