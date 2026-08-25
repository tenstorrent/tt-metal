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
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_dram_alignment

# Bytes of routing tail the store-and-forward path appends to a staged token. Must match
# sf::tail_bytes() in combine/device/combine_sf.hpp.
COMBINE_SF_TAIL_BYTES = 16


def combine_sf_levels(mesh_device, topology, cluster_axis=0):
    """Number of relay levels the store-and-forward path needs on this mesh.

    Zero means no token is ever more than one hop from its destination, so the path is inert and
    no staging buffer is required. Mirrors sf::num_levels() in combine_sf.hpp.
    """
    extent = mesh_device.shape[cluster_axis]
    max_distance = extent // 2 if topology == ttnn.Topology.Ring else extent - 1
    return max_distance - 1 if max_distance >= 2 else 0


def combine_sf_page_bytes(emb_dim, output_dtype=ttnn.bfloat16):
    """Stride of one staging page: a token payload plus its routing tail, DRAM-aligned.

    Mirrors sf::page_bytes(). The op rejects a staging buffer whose page size disagrees, because a
    relay reads a page straight into an L1 ring slot and an unaligned stride corrupts silently.
    """
    element_bytes = 1 if output_dtype == ttnn.fp8_e4m3 else 2
    alignment = get_dram_alignment()

    def align_up(value):
        return ((value + alignment - 1) // alignment) * alignment

    return align_up(align_up(emb_dim * element_bytes) + COMBINE_SF_TAIL_BYTES)


def make_combine_staging_buffer(
    mesh_device,
    emb_dim,
    output_dtype=ttnn.bfloat16,
    num_links=1,
    topology=ttnn.Topology.Linear,
    cluster_axis=0,
    slots_per_stream=16,
):
    """Allocate the DRAM scratch the store-and-forward combine path relays through.

    Returns None when the mesh is too shallow for any relay to exist.

    The buffer holds no state between invocations, so allocate it once for the whole model and hand
    the same tensor to every layer -- allocating per layer would multiply it by the layer count for
    no benefit. `slots_per_stream` sets the ring depth per (direction, level, sender core) and must
    be a power of two of at least 2; it is sized from the credit round-trip, so ~16 already covers
    a link's bandwidth-delay product several times over.
    """
    levels = combine_sf_levels(mesh_device, topology, cluster_axis)
    if levels == 0:
        return None

    page_bytes = combine_sf_page_bytes(emb_dim, output_dtype)
    num_pages = 2 * levels * min(num_links, 4) * slots_per_stream
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, num_pages, page_bytes // 4]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


def make_combine_staging_buffer_for_moe(
    mesh_device,
    emb_dim,
    num_links,
    topology,
    output_dtype=ttnn.bfloat16,
    slots_per_stream=16,
):
    """Allocate the shared staging buffer from TtMoe's per-axis num_links / topology arguments.

    Combine runs on the row axis (cluster_axis=0), so only the row half of a per-axis tuple applies.
    Callers allocating the buffer sit above TtMoe and would otherwise have to re-derive that
    convention; keeping it here means the buffer and the op that reads it cannot disagree about
    which axis they were sized for.
    """
    row_num_links = num_links[0] if isinstance(num_links, tuple) else num_links
    row_topology = topology[0] if isinstance(topology, tuple) else topology
    return make_combine_staging_buffer(
        mesh_device,
        emb_dim,
        output_dtype=output_dtype,
        num_links=row_num_links,
        topology=row_topology,
        cluster_axis=0,
        slots_per_stream=slots_per_stream,
    )


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
        use_store_and_forward: bool = False,
        staging_buffer=None,
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
            use_store_and_forward: Relay a token more than one hop from its destination through the
                first neighbour's staging_buffer instead of sending a multi-hop fabric unicast, so
                every ethernet packet terminates at its receiver and no router eRISC re-injects
                forwarded traffic.
            staging_buffer: DRAM scratch for that path, from make_combine_staging_buffer(). Shared
                across layers, so it is injected rather than allocated here. Required when
                use_store_and_forward is set and combine_sf_levels() is non-zero.
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
        self.use_store_and_forward = use_store_and_forward
        self.staging_buffer = staging_buffer
        if use_store_and_forward and staging_buffer is None and combine_sf_levels(mesh_device, topology, cluster_axis):
            raise ValueError(
                "use_store_and_forward needs a staging_buffer on this mesh; build one with "
                "make_combine_staging_buffer() and share it across layers"
            )

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
            staging_buffer=self.staging_buffer,
            use_store_and_forward=self.use_store_and_forward,
        )

        return output
