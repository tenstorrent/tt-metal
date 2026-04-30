# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MoE Dispatch Module (TTNN Implementation)

This module routes input tokens to the experts that are supposed to process them.
It sits between TtMoERoutingSetup (which computes where each source device should write)
and TtRoutedExpert (which processes the tokens after they arrive).

For each token and each of its top-k experts, the dispatch kernel:
  1. Looks up which destination device hosts that expert via the expert_dispatch_table.
  2. Reads the write position (global_dispatch_offset) for this source device and expert
     from tt_expert_offsets (produced by TtMoERoutingSetup).
  3. Writes the token embedding into the destination device's local dispatch buffer at that
     position: locally via NOC if the expert is on the same device, or remotely via fabric
     if it is on a different device in the dispatch group.
  4. Writes a metadata entry alongside each token recording:
       [0] linearized_mesh_coord  — source device coordinate
       [1] token_idx              — original token index within the source device's sequence
       [2] topk_idx               — which top-k slot this routing corresponds to
       [3] routed_expert          — global expert ID
       [4] weight                 — router weight for this (token, expert) pair

Each destination device accumulates an expert-centric dispatch buffer from all source
devices. The buffer is flat: all experts_per_chip experts share a single token
dimension of total capacity max_dispatch_buffer_token_size, packed dynamically with
each expert's region starting at a TILE_HEIGHT-aligned offset. The per-device shape is:
  dispatched_buffer: (1, 1, max_dispatch_buffer_token_size, emb_dim)
  metadata:          (1, 1, max_dispatch_buffer_token_size, metadata_len=5)

TtCombineModule reads from these buffers using the same offsets to reconstruct the
original token ordering after expert processing.
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtDispatchModule(LightweightModule):
    """TTNN wrapper around the prefill_dispatch device operation.

    Routes input tokens to destination devices based on top-k expert indices and writes
    them into flat dispatch buffers. Produces dispatched_buffer and metadata tensors
    consumed by TtRoutedExpert and TtCombineModule respectively.
    See module docstring for full dispatch buffer layout details.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dispatch_group_size: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatch_buffer_token_size: int,
        seq_len_per_chip: int,
        emb_dim: int = 7 * 1024,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        """
        Initialize dispatch module with configuration parameters.

        Args:
            mesh_device: TTNN mesh device.
            dispatch_group_size: Number of devices in each dispatch group (mesh rows for cluster_axis=0).
            experts_per_chip: Number of experts hosted on each destination device.
            num_routed_experts: Total number of routed experts across all devices.
            num_experts_per_tok: Number of experts each token is routed to (top-k).
            metadata_len: Number of fields in per-token metadata (5: chip, token, topk_idx,
                routed_expert, weight).
            max_dispatch_buffer_token_size: Total token capacity of the flat dispatch
                buffer per chip. Tokens that would push the total past this cap are
                silently dropped by the kernel (prevents out-of-bounds DRAM writes).
            seq_len_per_chip: Number of tokens on each source device.
            emb_dim: Embedding dimension of each token.
            cluster_axis: Mesh axis along which dispatch communicates (0 = SP/dispatch axis).
            num_links: Number of fabric links for remote token writes.
            topology: Fabric topology for remote token writes.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.dispatch_group_size = dispatch_group_size
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.metadata_len = metadata_len
        self.max_dispatch_buffer_token_size = max_dispatch_buffer_token_size
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
            expert_offsets: Base offset for each expert from each chip (sparse per group)
                Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts) - from get_gate_outputs()

        Returns:
            TTNN tensor sharded across mesh devices
        """
        logger.debug(f"[shard_expert_offsets] INPUT: expert_offsets.shape={expert_offsets.shape}")
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(1, 0),
        )
        result = ttnn.from_torch(
            expert_offsets,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )
        # To be consistent with the moe_routing_setup output, we squeeze expert_offsets to have 2D per device shape: (1, num_routed_experts)
        result = ttnn.squeeze(result, 0)
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
        Route input tokens to destination device dispatch buffers based on top-k expert indices.

        For each token on each source device, the kernel looks up the destination device for
        each of its top-k experts via tt_expert_dispatch_table, then writes the token embedding
        at the position given by tt_expert_offsets in the destination device's flat dispatch
        buffer. Writes to the local device use NOC; writes to remote devices use fabric.
        A metadata entry is written alongside each token for later recombination by TtCombineModule.

        Args:
            x: Input token embeddings.
                Shape per device: (1, seq_len_per_chip, emb_dim)
            weights: Router weights for each token's top-k experts.
                Shape per device: (1, seq_len_per_chip, num_experts_per_tok)
            indices: Top-k expert indices for each token.
                Shape per device: (1, seq_len_per_chip, num_experts_per_tok)
            tt_expert_offsets: Starting token index per source device per expert in the
                destination device's flat dispatch buffer. Produced by TtMoERoutingSetup.forward().
                Shape per device: (1, num_routed_experts)
            tt_expert_dispatch_table: Maps each expert ID to the destination chip ID within the
                dispatch group. Produced by shard_expert_dispatch_table().
                Shape per device: (1, num_routed_experts)
                Values >= 0 are destination chip IDs; -1 means the expert is not present in
                this dispatch group.

        Returns:
            dispatched_buffer: Flat expert-centric token buffer on each destination device.
                Shape per device: (1, 1, max_dispatch_buffer_token_size, emb_dim)
                Token at index i belongs to the expert whose region covers index i; regions are
                TILE_HEIGHT-aligned and laid out by the expert region offsets from offset_cumsum.
            metadata: Per-token metadata written alongside dispatched_buffer.
                Shape per device: (1, 1, max_dispatch_buffer_token_size, metadata_len=5),
                int32, ROW_MAJOR.
                Fields per token: [linearized_mesh_coord, token_idx, topk_idx, routed_expert, weight].
                Used by TtCombineModule to route processed tokens back to their origin.
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
            f"  metadata_len={self.metadata_len}, max_dispatch_buffer_token_size={self.max_dispatch_buffer_token_size}"
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
            max_dispatch_buffer_token_size=self.max_dispatch_buffer_token_size,
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
