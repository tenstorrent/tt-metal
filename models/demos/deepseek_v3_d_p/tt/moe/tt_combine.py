# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MoE Combine Module (TTNN Implementation)

This module routes expert-processed tokens back to their origin devices and accumulates
weighted contributions at each token's original position. It is the inverse of TtDispatchModule
and sits between TtRoutedExpert (which processes the dispatched tokens) and the MoE aggregation
step (which reduces the num_experts_per_tok contributions per token).

For each expert slot in dispatched_buffer and its corresponding metadata entry, the combine kernel:
  1. Reads metadata fields written by dispatch:
       [0] linearized_mesh_coord  — source device coordinate
       [1] token_idx              — original token index within the source device's sequence
       [2] topk_idx               — which top-k slot this expert contribution corresponds to
       [3] routed_expert          — global expert ID
       [4] weight                 — router weight for this (token, expert) pair
  2. Multiplies the expert output embedding by the router weight.
  3. Writes the weighted embedding to the origin device's output buffer at position
     [token_idx, topk_idx]: locally via NOC if the origin is the same device, or remotely
     via fabric if it is a different device in the dispatch group.

Each destination device accumulates a token-centric output buffer: for each token, up to
num_experts_per_tok expert contributions are written at their respective top-k indices.
Only slots corresponding to experts in this dispatch group are populated; slots for experts
from other dispatch groups contain uninitialized values. The per-device output shape is:
  output: (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim)

TtDispatchModule produces the dispatched_buffer and metadata consumed here.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtCombineModule(LightweightModule):
    """TTNN wrapper around the prefill_combine device operation.

    Reads expert-processed token embeddings from dispatched_buffer and routes them back
    to their origin devices using dispatch metadata, accumulating weighted contributions
    at each token's original top-k slot. Produces the combined output consumed by the
    MoE aggregation step.
    See module docstring for full output buffer layout details.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        dispatch_group_size: int,
        num_dispatch_groups: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        init_zeros: bool = True,
    ):
        """
        Initialize combine module with configuration parameters.

        Args:
            mesh_device: TTNN mesh device.
            dispatch_group_size: Number of devices in each dispatch group (mesh rows for cluster_axis=0).
            num_dispatch_groups: Number of independent dispatch groups (mesh columns for cluster_axis=0).
            experts_per_chip: Number of experts hosted on each device.
            num_experts_per_tok: Number of experts each token is routed to (top-k).
            seq_len_per_chip: Number of tokens on each source device (output token dimension size).
            cluster_axis: Mesh axis along which combine communicates (0 = SP/dispatch axis).
            num_links: Number of fabric links for remote token writes.
            topology: Fabric topology for remote token writes.
            memory_config: Output memory configuration. Must be interleaved (L1 or DRAM).
            init_zeros: Whether to zero-initialize the output buffer before writing.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.dispatch_group_size = dispatch_group_size
        self.num_dispatch_groups = num_dispatch_groups
        self.experts_per_chip = experts_per_chip
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology
        self.memory_config = memory_config
        self.init_zeros = init_zeros

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        dispatched_metadata: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
        expert_region_offsets: ttnn.Tensor,
    ):
        """
        Route expert-processed tokens back to origin devices and accumulate weighted contributions.

        For each expert slot in dispatched_buffer, the kernel reads the corresponding metadata
        entry to determine the origin device, original token index, top-k slot, and router weight.
        It multiplies the expert output by the weight and writes it to the origin device's output
        buffer: locally via NOC if the origin is the same device, or remotely via fabric if the
        origin is a different device in the dispatch group.

        Args:
            dispatched_buffer: Expert-processed token embeddings produced by TtRoutedExpert.
                Shape per device: (1, 1, max_dispatch_buffer_token_size, emb_dim).
                BFLOAT16 ROW_MAJOR.
            dispatched_metadata: Per-token routing metadata produced by TtDispatchModule.forward().
                Shape per device: (1, 1, max_dispatch_buffer_token_size, metadata_len=5).
                INT32 ROW_MAJOR. Fields per token: [linearized_mesh_coord, token_idx, topk_idx, routed_expert, weight].
            expert_token_counts: Number of tokens dispatched to each expert, used to bound the
                valid range of token slots read per expert in dispatched_buffer.
                Shape per device: (1, 1, num_routed_experts). INT32 ROW_MAJOR.
            expert_region_offsets: Expert region offsets (shared across source devices in a
                dispatch group) giving each expert's region start position in dispatched_buffer.
                Same shape/layout as expert_token_counts. Produced by offset_cumsum.
                Shape per device: (1, 1, num_routed_experts). INT32 or UINT32 ROW_MAJOR.

        Returns:
            output: Combined token embeddings with weighted expert contributions at each token's
                original top-k slot. Produced by ttnn.experimental.deepseek_prefill.combine.
                Shape per device: (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim).
                BFLOAT16 ROW_MAJOR. Token slots for experts outside this dispatch group contain
                uninitialized values.
        """
        output = ttnn.experimental.deepseek_prefill.combine(
            dispatched_buffer,
            dispatched_metadata,
            expert_token_counts,
            expert_region_offsets,
            dispatch_group_size=self.dispatch_group_size,
            experts_per_chip=self.experts_per_chip,
            num_experts_per_tok=self.num_experts_per_tok,
            seq_len_per_chip=self.seq_len_per_chip,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=self.memory_config,
            init_zeros=self.init_zeros,
        )
        return output
