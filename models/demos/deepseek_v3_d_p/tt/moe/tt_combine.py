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
        fp8_output: bool = False,
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
            fp8_output: Emit the combined output in fp8_e4m3. Requires Blackhole hardware.
        """
        if fp8_output and mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            raise ValueError("fp8_output requires Blackhole hardware")
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
        self.fp8_output = fp8_output

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
                Shape per device: (1, 1, max_dispatch_buffer_token_size, metadata_len=3).
                INT32 ROW_MAJOR. Fields per token: [linearized_mesh_coord, token_idx, topk_idx].
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
        # FP8 output only works when the dispatched buffer is TILE: the BF16 -> FP8 conversion
        # happens in the packer at the untilize stage, which only exists on the TILE path.
        # The C++ validator enforces this too; checking here gives a clearer Python-side error.
        if self.fp8_output and dispatched_buffer.layout != ttnn.TILE_LAYOUT:
            raise ValueError(
                f"fp8_output=True requires dispatched_buffer in TILE_LAYOUT (got {dispatched_buffer.layout})"
            )

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
            use_fp8_combine=self.fp8_output,
        )

        return output


class TtCombine2dModule(LightweightModule):
    """TTNN wrapper around the combine_fabric2d device operation.

    Same job as TtCombineModule — expert-processed tokens back to their origin devices, weighted
    and placed at each token's original top-k slot — over a different transport. combine reaches
    every chip in the dispatch group directly; combine_fabric2d forwards hop by hop along the ring
    axis, chip-local DRAM -> eth -> the next chip's DRAM, managing the multi-hop routing itself
    rather than leaving it to fabric.

    That transport needs one input the direct op does not: `expert_offsets`, which tells every chip
    where each ORIGIN chip's run starts inside each expert's region. It is a separate class rather
    than a flag on TtCombineModule because the two ops take different inputs and have different
    capabilities — notably there is no fp8 output path here.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    ):
        """
        Initialize the fabric2d combine module.

        Args:
            mesh_device: TTNN mesh device.
            experts_per_chip: Number of experts hosted on each device.
            num_experts_per_tok: Number of experts each token is routed to (top-k).
            seq_len_per_chip: Number of tokens on each source device (output token dimension size).
            cluster_axis: Mesh axis the ring runs along (0 = SP/dispatch axis).
            num_links: Number of fabric links per chip-to-chip connection. Each (link, direction)
                pair is one stream, so the op runs 2 * num_links streams per chip.
            topology: Fabric topology for the ring.
            memory_config: Output memory configuration. Must be interleaved (L1 or DRAM).

        Note there is no dispatch_group_size: combine_fabric2d derives the ring from cluster_axis
        and the mesh shape, and no fp8_output, because the op has no fp8 stage on the way out.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.experts_per_chip = experts_per_chip
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology
        self.memory_config = memory_config

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        dispatched_metadata: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
        expert_region_offsets: ttnn.Tensor,
        expert_offsets: ttnn.Tensor,
    ):
        """
        Route expert-processed tokens back to origin devices over the explicitly-forwarded fabric2d.

        Args:
            dispatched_buffer: Expert-processed token embeddings produced by TtRoutedExpert.
                A chip's page range for one expert holds that expert's tokens grouped by the chip
                they ORIGINATED on. BFLOAT16.
            dispatched_metadata: Per-token routing metadata produced by TtDispatchModule.forward().
                3 int32 per token: (linearized_mesh_coord, token_idx, topk_idx). INT32 ROW_MAJOR.
            expert_token_counts: Number of tokens dispatched to each expert; also closes the last
                run. INT32 ROW_MAJOR.
            expert_region_offsets: Where each expert's region starts in dispatched_buffer.
                Same shape/layout as expert_token_counts.
            expert_offsets: Where each ORIGIN chip's run starts inside each expert's region. Must be
                REPLICATED along the dispatch-group axis — every chip needs every origin chip's
                boundaries for the experts it hosts, which is what lets each kernel compute the token
                sequence locally instead of exchanging addresses.

        Returns:
            output: Combined token embeddings, shape per device
                (1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim), BFLOAT16 ROW_MAJOR.
        """
        return ttnn.experimental.deepseek_prefill.combine_fabric2d(
            dispatched_buffer,
            dispatched_metadata,
            expert_token_counts,
            expert_region_offsets,
            expert_offsets,
            experts_per_chip=self.experts_per_chip,
            num_experts_per_tok=self.num_experts_per_tok,
            seq_len_per_chip=self.seq_len_per_chip,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=self.memory_config,
        )
