# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Initialization and configuration helpers for MoE testing.

This module provides shared helper functions used across MoE tests including:
- Mesh configuration extraction (MeshConfig, extract_mesh_config)
- Configuration computation (compute_constants)
- Test input generation (initialize_test_inputs, initialize_predictable_test_inputs)
- Fabric configuration helpers (create_fabric_router_config)
"""

from dataclasses import dataclass

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.utility_functions import is_blackhole

# Fabric packet payload limits (conservative round values below hardware maximums).
MAX_PAYLOAD_SIZE_BH = 14 * 1024  # Blackhole hardware max ~15232 B
MAX_PAYLOAD_SIZE_WH = 7 * 1024  # Wormhole hardware max ~7616 B


def get_max_payload_size() -> int:
    """Return the arch-appropriate fabric payload size. Deferred to avoid probing hardware at import time."""
    return MAX_PAYLOAD_SIZE_BH if is_blackhole() else MAX_PAYLOAD_SIZE_WH


@dataclass
class MeshConfig:
    """Mesh configuration extracted from mesh_device."""

    sp_axis: int
    dispatch_group_size: int
    num_dispatch_groups: int


def extract_mesh_config(mesh_device) -> MeshConfig:
    """
    Extract dispatch configuration from mesh device shape.

    For 2D meshes (both dimensions > 1):
      - sp_axis = 0 (sequence parallel along rows)
      - dispatch_group_size = number of rows
      - num_dispatch_groups = number of columns (EP ranks)

    For 1D meshes:
      - dispatch_group_size = total number of devices
      - num_dispatch_groups = 1
      - sp_axis = whichever dimension has size > 1
    """
    sp_axis = 0
    dispatch_group_size = mesh_device.shape[sp_axis]
    num_dispatch_groups = mesh_device.shape[1]

    return MeshConfig(
        sp_axis=sp_axis,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )


class ExpertMapping:
    """
    Encapsulates expert-to-device mapping logic for MoE weight distribution.

    All methods in this class must stay in sync - they implement the same underlying
    mapping between global expert indices and mesh device positions:

    - `get_global_expert_idx()`: Computes global expert index from (group, chip, local_expert)
    - `gather_weights_for_mesh_distribution()`: Gathers weights in mesh order using the above
    - `create_dispatch_table()`: Creates runtime dispatch table for token routing
    - `create_global_expert_idx_table()`: Builds a (group, chip, local_expert) -> global expert id lookup tensor
    - `get_weights_mesh_mapper()`: Returns mesh mapper with dims=(0, 1) for weight distribution

    The default layout is column-major: experts 0..N are placed in group 0, then N+1..2N in group 1, etc.
    Within each group, experts are distributed across chips sequentially.
    """

    @staticmethod
    def get_global_expert_idx(
        group: int,
        chip: int,
        local_expert: int,
        experts_per_chip: int,
        dispatch_group_size: int,
        num_dispatch_groups: int,
        is_col_major: bool = True,
    ) -> int:
        """
        Get global expert index for a given (group, chip, local_expert) position.

        Args:
            group: Dispatch group index (corresponds to mesh column)
            chip: Chip index within dispatch group (corresponds to mesh row)
            local_expert: Local expert index on the chip
            experts_per_chip: Number of experts per chip
            dispatch_group_size: Number of chips per dispatch group (mesh rows)
            num_dispatch_groups: Number of dispatch groups (mesh columns)
            is_col_major: If True, column-major (group-first): experts 0-N in group 0
                         If False, row-major (chip-first): experts interleaved across groups

        Returns:
            Global expert index
        """
        if is_col_major:
            device_idx = group * dispatch_group_size + chip
        else:
            device_idx = chip * num_dispatch_groups + group
        return device_idx * experts_per_chip + local_expert

    @staticmethod
    def compute_linearized_mesh_coord(
        logical_chip_id: int,
        dispatch_group_idx: int,
        num_dispatch_groups: int,
    ) -> int:
        """
        Convert logical chip ID to linearized mesh coordinate.

        The linearized coord encodes both the chip and dispatch group:
        linearized = logical_chip * num_dispatch_groups + dispatch_group_idx

        TorchCombineModule extracts:
        - dispatch_group = linearized % num_dispatch_groups
        - logical_chip = linearized // num_dispatch_groups

        Args:
            logical_chip_id: Chip index within dispatch group (corresponds to mesh row)
            dispatch_group_idx: Dispatch group index (corresponds to mesh column)
            num_dispatch_groups: Number of dispatch groups (mesh columns)

        Returns:
            Linearized mesh coordinate
        """
        return logical_chip_id * num_dispatch_groups + dispatch_group_idx

    @staticmethod
    def gather_weights_for_mesh_distribution(
        torch_weights: list[dict],
        local_expert_idx: int,
        mesh_rows: int,
        mesh_cols: int,
        experts_per_chip: int,
        is_col_major: bool = True,
    ) -> tuple[list, list, list]:
        """
        Gather gate/up/down weights for a local expert, ordered for mesh distribution.

        Uses row-major iteration (for row: for col:) to match `get_weights_mesh_mapper()`.
        The returned lists are ordered so that after stacking and reshaping to
        (mesh_rows, mesh_cols, ...), position [row, col] contains weights for the device
        at mesh position (row, col).

        Args:
            torch_weights: List of weight dicts indexed by global expert ID
            local_expert_idx: Local expert index on each chip (0 to experts_per_chip-1)
            mesh_rows: Number of mesh rows (= dispatch_group_size = chips per group)
            mesh_cols: Number of mesh cols (= num_dispatch_groups)
            experts_per_chip: Number of experts per chip
            is_col_major: Expert ordering (True=column-major, False=row-major)

        Returns:
            (gate_weights, up_weights, down_weights) - each a list of tensors in mesh order
        """
        gate_weights = []
        up_weights = []
        down_weights = []

        # Row-major iteration matches mesh mapper dims=(0, 1)
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                global_idx = ExpertMapping.get_global_expert_idx(
                    group=col,
                    chip=row,
                    local_expert=local_expert_idx,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=mesh_rows,
                    num_dispatch_groups=mesh_cols,
                    is_col_major=is_col_major,
                )
                gate_weights.append(torch_weights[global_idx]["gate_proj"])
                up_weights.append(torch_weights[global_idx]["up_proj"])
                down_weights.append(torch_weights[global_idx]["down_proj"])

        return gate_weights, up_weights, down_weights

    @staticmethod
    def create_dispatch_table(
        num_routed_experts: int,
        dispatch_group_size: int,
        num_dispatch_groups: int = 1,
    ) -> torch.Tensor:
        """
        Create expert dispatch table mapping experts to destination chips in dispatch axis.

        This table translates expert ID to logical location of the expert in the dispatch axis.
        -1 means the expert is not present in that dispatch group and should be ignored.

        The chip mapping uses the same formula as the kernel: chip_id = expert_id // experts_per_chip
        where experts_per_chip = num_routed_experts // dispatch_group_size.

        Args:
            num_routed_experts: Total number of routed experts
            dispatch_group_size: Number of chips in each dispatch group
            num_dispatch_groups: Number of parallel dispatch groups

        Returns:
            expert_dispatch_table: Shape (num_dispatch_groups, num_routed_experts + 1)
                Values are logical chip IDs (0 to dispatch_group_size-1) or -1 if not present.
                The trailing sentinel column (index num_routed_experts) is always -1: padding-aware
                routing sentinel-marks padded tokens with expert id == num_routed_experts, so the
                dispatch reader's unguarded table[idx] lookup maps them to -1 (skip). masked_bincount
                only reads indices < num_routed_experts, so the extra column is harmless there.

        Example:
            # num_chips=8, dispatch_group_size=4, num_dispatch_groups=2, num_routed_experts=16
            # experts_per_group = 16/2 = 8, experts_per_chip = 8/4 = 2
            # Group 0 handles experts 0-7, Group 1 handles experts 8-15
            # chip_id = local_expert_id // 2 (local within each group)
            expert_dispatch_table = [
                [ 0, 0, 1, 1,  2, 2, 3, 3, -1,-1,-1,-1, -1,-1,-1,-1], # group 0: experts 0-7 -> chips 0-3
                [-1,-1,-1,-1, -1,-1,-1,-1,  0, 0, 1, 1,  2, 2, 3, 3], # group 1: experts 8-15 -> chips 0-3
            ]
        """
        # Each dispatch group handles a subset of experts, distributed across chips
        experts_per_group = num_routed_experts // num_dispatch_groups
        experts_per_chip = experts_per_group // dispatch_group_size  # Experts per chip within each group

        # Width is num_routed_experts + 1: the extra trailing column is the padding sentinel
        # (always -1). Padded tokens carry expert id == num_routed_experts and the dispatch reader
        # looks them up unguarded, so they map to -1 and are skipped.
        table = torch.full((num_dispatch_groups, num_routed_experts + 1), -1, dtype=torch.int32)
        for group in range(num_dispatch_groups):
            group_start = group * experts_per_group
            group_end = group_start + experts_per_group
            for expert_id in range(group_start, group_end):
                # Use local expert ID within the group to compute chip mapping
                local_expert_id = expert_id - group_start
                chip_id = local_expert_id // experts_per_chip
                table[group, expert_id] = chip_id

        logger.debug(f"[ExpertMapping.create_dispatch_table] OUTPUT: table.shape={table.shape}")
        return table

    @staticmethod
    def create_global_expert_idx_table(
        experts_per_chip: int,
        dispatch_group_size: int,
        num_dispatch_groups: int,
        is_col_major: bool = True,
    ) -> torch.Tensor:
        """
        Build a lookup tensor mapping mesh position (group, chip, local_expert) to
        global expert index, using `get_global_expert_idx` as the source of truth.

        Args:
            experts_per_chip: Number of experts per chip
            dispatch_group_size: Number of chips per dispatch group (mesh rows)
            num_dispatch_groups: Number of dispatch groups (mesh columns)
            is_col_major: Expert ordering (True=column-major, False=row-major)

        Returns:
            Tensor of shape (num_dispatch_groups, dispatch_group_size, experts_per_chip)
            where table[g, c, e] is the global expert id assigned to local_expert e
            on chip c of dispatch group g.
        """
        table = torch.zeros(
            (num_dispatch_groups, dispatch_group_size, experts_per_chip),
            dtype=torch.int32,
        )
        for group in range(num_dispatch_groups):
            for chip in range(dispatch_group_size):
                for local_expert in range(experts_per_chip):
                    table[group, chip, local_expert] = ExpertMapping.get_global_expert_idx(
                        group=group,
                        chip=chip,
                        local_expert=local_expert,
                        experts_per_chip=experts_per_chip,
                        dispatch_group_size=dispatch_group_size,
                        num_dispatch_groups=num_dispatch_groups,
                        is_col_major=is_col_major,
                    )

        logger.debug(f"[ExpertMapping.create_global_expert_idx_table] OUTPUT: table.shape={table.shape}")
        return table

    @staticmethod
    def get_weights_mesh_mapper(mesh_device):
        """
        Create mesh mapper for routed expert weight tensors.

        Each device gets its own unique expert weights, sharded across both mesh dimensions.
        The input tensor should have shape (mesh_rows, mesh_cols, in_features, out_features).

        This mapper uses dims=(0, 1) which means:
        - Tensor dim 0 is sharded across mesh rows
        - Tensor dim 1 is sharded across mesh cols
        - Device at mesh[row, col] receives tensor[row, col, :, :]

        Args:
            mesh_device: TTNN mesh device

        Returns:
            ShardTensor2dMesh mapper configured for routed expert weights
        """
        return ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(0, 1),  # Shard dim 0 across mesh rows, dim 1 across mesh cols
        )


def get_gate_outputs(
    indices: torch.Tensor,
    dispatch_group_size: int,
    num_routed_experts: int,
    experts_per_chip: int,
    seq_len_per_chip: int,
    num_experts_per_tok: int,
    expert_dispatch_table: torch.Tensor = None,
) -> tuple:
    """
    Compute dispatch offsets and token counts from router indices.

    This processes the gate/router output indices to determine:
    1. Where each token should be written in the dispatch buffer (offsets)
    2. How many tokens each expert receives (counter)

    All outputs are in sparse format per dispatch group: experts not belonging
    to a group are zeroed out.

    Args:
        indices: Expert indices tensor (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
        dispatch_group_size: Number of chips in each dispatch group
        num_routed_experts: Total number of routed experts across all chips
        experts_per_chip: Number of experts per chip
        seq_len_per_chip: Sequence length per chip
        num_experts_per_tok: Number of experts each token routes to
        expert_dispatch_table: Expert to chip mapping table
            Shape: (num_dispatch_groups, num_routed_experts) or, with padding awareness,
            (num_dispatch_groups, num_routed_experts + 1) where the trailing sentinel column
            (index num_routed_experts, always -1) is ignored here. If None, computed internally.

    Returns:
        expert_offsets: Base offset for each expert from each chip (sparse per group)
            Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts)
        expert_token_counts: Total tokens per expert (sparse per group, replicated across dispatch_group_size)
            Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts)
        expert_region_offsets: Expert region component of expert_offsets — identical across
            the dispatch_group_size dimension. Equals expert_offsets minus the
            per-source-device local offset.
            Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts)
        expert_counter: Per-chip token counts for each expert (sparse per group).
            Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts)
    """
    num_dispatch_groups = num_routed_experts // experts_per_chip // dispatch_group_size

    # Build dispatch table if not provided
    if expert_dispatch_table is None:
        expert_dispatch_table = ExpertMapping.create_dispatch_table(
            num_routed_experts=num_routed_experts,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
        )

    # Drop the padding sentinel column (index num_routed_experts, always -1) if present, so the
    # rest of the body operates at width num_routed_experts and matches expert_counter_dense.
    expert_dispatch_table = expert_dispatch_table[:, :num_routed_experts]

    # Count tokens per expert per chip (dense)
    expert_counter_dense = torch.zeros((dispatch_group_size, num_routed_experts), dtype=torch.int32)
    for chip in range(dispatch_group_size):
        for token in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                routed_expert = indices[chip, token, topk_idx]
                expert_counter_dense[chip, routed_expert] += 1

    # Create group masks from dispatch table: (num_dispatch_groups, 1, num_routed_experts)
    group_masks = (expert_dispatch_table >= 0).unsqueeze(1).to(torch.int32)

    # Sparse expert_counter: (num_dispatch_groups, dispatch_group_size, num_routed_experts)
    expert_counter = expert_counter_dense.unsqueeze(0).expand(num_dispatch_groups, -1, -1) * group_masks

    # Cumsum along dispatch_group_size (dim=1) per group
    cum_sum = torch.cumsum(expert_counter, dim=1)
    local_expert_offsets = torch.cat(
        [torch.zeros((num_dispatch_groups, 1, num_routed_experts), dtype=torch.int32), cum_sum[:, :-1, :]], dim=1
    )

    # Token counts: last row of cumsum, replicated across dispatch_group_size
    expert_token_counts = cum_sum[:, -1:, :].expand(-1, dispatch_group_size, -1).contiguous()

    # Partial cumsum along expert dimension (dim=-1)
    # Split num_routed_experts into (num_chips, experts_per_chip), cumsum within each chip
    # Pad each expert's count to TILE_SIZE so each expert starts at a tile boundary in the dispatch buffer
    aligned_token_counts = ((expert_token_counts + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    global_expert_offsets = torch.reshape(
        aligned_token_counts,
        (num_dispatch_groups, dispatch_group_size, num_routed_experts // experts_per_chip, experts_per_chip),
    )
    global_expert_offsets = torch.cumsum(global_expert_offsets, dim=-1)
    global_expert_offsets = torch.cat(
        [
            torch.zeros(
                (num_dispatch_groups, dispatch_group_size, num_routed_experts // experts_per_chip, 1), dtype=torch.int32
            ),
            global_expert_offsets[:, :, :, :-1],
        ],
        dim=-1,
    )
    global_expert_offsets = torch.reshape(
        global_expert_offsets,
        (num_dispatch_groups, dispatch_group_size, num_routed_experts),
    )
    # Snapshot expert region component (shared across source devices) before adding local offsets
    expert_region_offsets = global_expert_offsets.clone()
    global_expert_offsets += local_expert_offsets

    logger.debug(f"[get_gate_outputs] OUTPUT SHAPES:")
    logger.debug(f"  expert_counter.shape={expert_counter.shape}")
    logger.debug(f"  local_expert_offsets.shape={local_expert_offsets.shape}")
    logger.debug(f"  global_expert_offsets.shape={global_expert_offsets.shape}")
    logger.debug(f"  expert_region_offsets.shape={expert_region_offsets.shape}")
    logger.debug(f"  expert_token_counts.shape={expert_token_counts.shape}")
    return global_expert_offsets, expert_token_counts, expert_region_offsets, expert_counter


def compute_constants(
    seq_len_per_chip,
    num_routed_experts,
    num_experts_per_tok,
    num_devices,
    dispatch_group_size,
    dispatch_buffer_capacity_factor,
    experts_per_chip_override: int | None = None,
    emb_dim: int | None = None,
    fp8_scaled_input: bool = False,
):
    """
    Compute derived constants for MoE configuration.

    Args:
        seq_len_per_chip: Sequence length per chip
        num_routed_experts: Total number of routed experts across all chips
        num_experts_per_tok: Number of experts each token is routed to
        num_devices: Number of devices across which experts are distributed
        dispatch_group_size: Number of devices in each dispatch group
        dispatch_buffer_capacity_factor: Multiplier for the flat dispatch
            buffer; callers must pick the smallest integer such that
            dgs*seq*factor is not smaller than the theoretical worst-case
            required buffer size.
        experts_per_chip_override: If not None, bypass the
            num_routed_experts // num_devices derivation and use this value.
            Required when simulating one Galaxy column on a single-column LB
            mesh: the table indexes 256 global expert IDs but only 8 of them
            physically live on each chip (not 256/8=32).
        emb_dim: Embedding (hidden) dimension. Only required when fp8_scaled_input
            is True, to size the per-token fp32 scale tail in the metadata.
        fp8_scaled_input: If True, each token appends its emb_dim/128 fp32 scales
            (bit-cast int32) after the 3 routing fields, growing metadata_len.

    Returns:
        experts_per_chip: Number of experts per chip
        metadata_len: Length of metadata per token
        max_dispatch_buffer_token_size: Total token capacity of the dispatch buffer
        max_dispatched_tokens_per_expert: Maximum tokens per expert
    """
    assert (
        seq_len_per_chip % ttnn.TILE_SIZE == 0
    ), f"seq_len_per_chip ({seq_len_per_chip}) must be a multiple of TILE_SIZE ({ttnn.TILE_SIZE})"

    if experts_per_chip_override is not None:
        assert (
            experts_per_chip_override > 0
        ), f"experts_per_chip_override must be positive, got {experts_per_chip_override}"
        experts_per_chip = experts_per_chip_override
    else:
        experts_per_chip = num_routed_experts // num_devices
    metadata_len = 3  # chip, token, topk_idx
    if fp8_scaled_input:
        # Each token appends its emb_dim/128 fp32 scales (bit-cast int32) after the 3 routing fields.
        assert emb_dim is not None, "emb_dim is required when fp8_scaled_input is True"
        metadata_len += emb_dim // 128

    # TODO: For now, we are ignoring the num_experts_per_tok, but it will be needed once
    # we support replicated experts (See Issue #41293)
    max_dispatched_tokens_per_expert = dispatch_group_size * seq_len_per_chip
    max_dispatch_buffer_token_size = max_dispatched_tokens_per_expert * dispatch_buffer_capacity_factor

    return experts_per_chip, metadata_len, max_dispatch_buffer_token_size, max_dispatched_tokens_per_expert


def initialize_test_inputs(
    dispatch_group_size: int,
    seq_len_per_chip: int,
    emb_dim: int,
    num_routed_experts: int,
    num_experts_per_tok: int,
    max_dispatched_tokens_per_expert: int,
    validate: bool = True,
    num_dispatch_groups: int = 1,
    skip_x_initialization: bool = False,
    seed: int = None,
):
    """
    Initialize test inputs (x, weights, indices) with random data.

    Args:
        dispatch_group_size: Number of chips in each dispatch group
        seq_len_per_chip: Sequence length per chip
        emb_dim: Embedding dimension
        num_routed_experts: Total number of routed experts across all chips
        num_experts_per_tok: Number of experts each token is routed to
        max_dispatched_tokens_per_expert: Maximum number of tokens per expert
        validate: Whether to validate expert activations
        num_dispatch_groups: Number of parallel dispatch groups

    Returns:
        x: Input tensor (dispatch_group_size, seq_len_per_chip, emb_dim)
        weights: Router weights (num_dispatch_groups, dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
        indices: Expert indices (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    """

    if seed is not None:
        torch.manual_seed(seed)

    input_shape = (dispatch_group_size, seq_len_per_chip, emb_dim)
    x = torch.randn(input_shape, dtype=torch.bfloat16) if not skip_x_initialization else None

    weights_shape = (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    indices_shape = (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)

    weights = torch.randn(weights_shape, dtype=torch.bfloat16)
    weights = torch.sigmoid(weights.float()).to(torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    indices = torch.randint(0, num_routed_experts, indices_shape, dtype=torch.int32)

    # Validate expert activations
    if validate:
        expert_activations = torch.zeros((num_routed_experts,), dtype=torch.int32)
        for c in range(indices.shape[0]):
            for t in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    expert_activations[indices[c, t, k]] += 1
        checksum = expert_activations.sum().item()
        logger.debug(f"{expert_activations.shape=}")
        assert (
            checksum == dispatch_group_size * seq_len_per_chip * num_experts_per_tok
        ), f"Expected checksum {dispatch_group_size * seq_len_per_chip * num_experts_per_tok}, got {checksum}"
        assert (
            expert_activations.max().item() <= max_dispatched_tokens_per_expert
        ), f"Expected max activations per expert to be <= {max_dispatched_tokens_per_expert}, got {expert_activations.max().item()}"

    logger.debug(f"[initialize_test_inputs] OUTPUT SHAPES:")
    logger.debug(f"  x.shape={x.shape if x is not None else None}")
    logger.debug(f"  weights.shape={weights.shape}")
    logger.debug(f"  indices.shape={indices.shape}")
    return x, weights, indices


def initialize_predictable_test_inputs(
    dispatch_group_size: int,
    seq_len_per_chip: int,
    emb_dim: int,
    num_routed_experts: int,
    num_experts_per_tok: int,
    max_dispatched_tokens_per_expert: int,
    num_dispatch_groups: int = 1,
):
    """
    Initialize test inputs with predictable patterns for debugging.

    Pattern:
    - x: Simple sequential values starting from 0.0
    - weights: Sequential values per dispatch group
      - Group 0: [1, 2, 3, 4], Group 1: [5, 6, 7, 8], etc. (for num_experts_per_tok=4)
    - indices: Round-robin pattern cycling through experts

    This makes it easy to verify writes:
    - Token 0 -> experts [0, 1, 2, 3]
    - Token 1 -> experts [4, 5, 6, 7]
    - Token 2 -> experts [8, 9, 10, 11]
    - etc.

    Args:
        dispatch_group_size: Number of chips in each dispatch group
        seq_len_per_chip: Sequence length per chip
        emb_dim: Embedding dimension
        num_routed_experts: Total number of routed experts across all chips
        num_experts_per_tok: Number of experts each token is routed to
        max_dispatched_tokens_per_expert: Maximum number of tokens per expert (unused but kept for signature consistency)
        num_dispatch_groups: Number of parallel dispatch groups

    Returns:
        x: Input tensor (dispatch_group_size, seq_len_per_chip, emb_dim)
        weights: Router weights (num_dispatch_groups, dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
        indices: Expert indices (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    """
    input_shape = (dispatch_group_size, seq_len_per_chip, emb_dim)
    # Fill with sequential values: 0.0, 1.0, 2.0, ...
    x = torch.arange(dispatch_group_size * seq_len_per_chip * emb_dim, dtype=torch.float32).reshape(input_shape)
    x = x.to(torch.bfloat16)

    weights_shape = (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    indices_shape = (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)

    # Predictable weights: group 0 = [1,2,3,4], group 1 = [5,6,7,8], etc.
    weights = torch.zeros(weights_shape, dtype=torch.bfloat16)

    for k in range(num_experts_per_tok):
        weights[:, :, k] = float(num_experts_per_tok + k + 1)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize so topk sums to 1

    # Round-robin indices pattern
    indices = torch.zeros(indices_shape, dtype=torch.int32)
    expert_idx = 0
    for chip in range(dispatch_group_size):
        for token in range(seq_len_per_chip):
            for k in range(num_experts_per_tok):
                if chip % 2 == 0:
                    indices[chip, token, k] = max(
                        0, expert_idx % (num_routed_experts) - 1
                    )  # max (0, x -1) to create a of unequal distribution
                else:
                    indices[chip, token, k] = (
                        num_routed_experts - 1 - (expert_idx % num_routed_experts)
                    )  # reverse order
                expert_idx += 1

    logger.debug(f"[initialize_predictable_test_inputs] OUTPUT SHAPES:")
    logger.debug(f"  x.shape={x.shape}")
    logger.debug(f"  weights.shape={weights.shape}")
    logger.debug(f"  indices.shape={indices.shape}")
    return x, weights, indices


def load_captured_routing(
    dispatch_group_size: int,
    seq_len_per_chip: int,
    num_routed_experts: int,
    num_experts_per_tok: int,
    layer: int,
    col: int,
    captured_indices_path: str = None,
):
    """Load real captured Galaxy gate indices, remapped to run one Galaxy column on LB 8x1.

    What the capture contains
    -------------------------
    `expert_routing.safetensors[expert_ids_layer_<L>]` holds one tensor per MoE layer
    of shape `(total_tokens=25600, top_k=8)` int32, with values in `[0, num_routed_experts=256)`.
    Those are **Galaxy-global expert IDs**. We `view` it into the worker's expected
    `(dispatch_group_size=8, seq_len_per_chip=3200, top_k=8)` layout.

    Galaxy 8x4 owns 256 experts split across 4 dispatch columns × 8 chips:

        col 0:  expert IDs [  0,  64), 8 experts per chip (chips 0..7 = ids 0..7, 8..15, ...)
        col 1:  expert IDs [ 64, 128)
        col 2:  expert IDs [128, 192)
        col 3:  expert IDs [192, 256)

    LB 8x1 has only one column, 8 chips, 8 experts/chip = 64 physical experts.
    The LB combine kernel hard-codes `first_expert_id=0`, so every expert ID it
    sees in metadata must fit in `[0, num_routed_experts_per_col=64)` or be a
    skip-sentinel — anything in `[64, 256)` would index past the per-chip
    `expert_token_counts` array and silently corrupt outputs.

    The remap
    ---------
    For a chosen Galaxy column `k`, we transform every captured value `v`:

        in-col routes (v in [k*64, (k+1)*64))   →  v - k*64      (shifts into [0, 64))
        out-of-col routes (everything else)     →  255           (sentinel)

    We then build the dispatch table.  `ExpertMapping.create_dispatch_table(256, 8, 4)`
    returns the full Galaxy 4-row table of shape `(4, 256)`; we slice `[0:1]` to get a
    `(1, 256)` tensor — one row, to match LB's single-col mesh.  The slice's contents:

        table[0,  0..63]   = [0,0,0,0,0,0,0,0, 1,1,...,1, ..., 7,7,7,7,7,7,7,7]   (chip ids, 8 per chip)
        table[0, 64..255]  = -1                                                    (kernel reads -1 → skip)
        table[0, 255]      = -1                                                    (== sentinel target)

    The chip-assignment function `chip_id = local_id // 8` is identical across every
    Galaxy column's row, so using row 0 against remapped (in-col-shifted) indices
    routes each expert to the same chip Galaxy would have:

        Galaxy expert 135 (col 2 local 7)  →  Galaxy table[2, 135] = chip 0
        After remap:        value becomes 7  →  LB table[0,   7]   = chip 0   (match)

    Out-of-col routes (sentinel 255) hit `table[0, 255] = -1` and the kernel skips
    them — preserving Galaxy col k's true per-col routing share 1:1 with no spurious
    work on the other 192 globals.

    Worked example: longbook L27 token 0, captured-col=2
    ----------------------------------------------------
    The 8 picks for chip 0 token 0 in longbook_qa_eng_25600 L27 are::

        raw   :  [138, 147,  79,  30, 150, 120,  72, 154]
        in-col:  [ ✓,   ✓,   ✗,   ✗,   ✓,   ✗,   ✗,   ✓ ]    (col 2 range = [128, 192))
        remap :  [ 10,  19, 255, 255,  22, 255, 255,  26]
        route :  [ch1, ch2, skip, skip, ch2, skip, skip, ch3]   (chip_id = remap_value // 8)

    Verification — Galaxy would have routed the same picks via its col-2 row::

        table[2, 138] = (138 - 128)//8 = chip 1   (same as our remapped 10 // 8)
        table[2, 147] = (147 - 128)//8 = chip 2   (same as our remapped 19 // 8)
        table[2,  79] = -1                         (col 1, Galaxy col 2 also skips)
        ...

    Args (beyond the existing shape/config args)
    ---------------------------------------------
        layer:                  int, MoE layer index (e.g. 27)
        col:                    int, Galaxy column [0, 4) to simulate
        captured_indices_path:  optional path override for the safetensors;
                                defaults to LONGBOOK_QA_ENG_25600/expert_routing.safetensors

    Returns
    -------
    (indices, expert_dispatch_table) where:
        indices                 (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
                                int32, values in [0, 64) ∪ {255}.
        expert_dispatch_table   (1, 256) int32 — col 0's row of the Galaxy 4-col table,
                                with chip IDs [0, 8) for [0, 64) and -1 elsewhere.
    """
    from pathlib import Path

    GALAXY_NUM_DISPATCH_GROUPS = 4

    if not 0 <= col < GALAXY_NUM_DISPATCH_GROUPS:
        raise ValueError(f"col must be in [0, {GALAXY_NUM_DISPATCH_GROUPS}), got {col}")
    if num_routed_experts != 256:
        raise ValueError(
            f"Captured indices require num_routed_experts=256, got {num_routed_experts}. "
            "Use a parametrize entry with the matching kernel config (perf_real_indices)."
        )

    if captured_indices_path:
        path = Path(captured_indices_path)
    else:
        # Lazy import: transformer_helpers itself imports from this module in places.
        from models.demos.deepseek_v3_d_p.utils.transformer_helpers import LONGBOOK_QA_ENG_25600

        path = LONGBOOK_QA_ENG_25600 / "expert_routing.safetensors"
    if not path.exists():
        raise FileNotFoundError(
            f"Captured indices file not found at {path}. "
            "Pass captured_indices_path to override, or configure DEEPSEEK_V3_TRACE_DIR."
        )

    from safetensors import safe_open

    key = f"expert_ids_layer_{layer}"
    with safe_open(str(path), framework="pt") as f:
        available = list(f.keys())
        if key not in available:
            raise KeyError(f"Layer key {key!r} not in {path}. Available keys (first 5): {available[:5]}")
        flat = f.get_tensor(key)

    expected_numel = dispatch_group_size * seq_len_per_chip * num_experts_per_tok
    if flat.numel() != expected_numel:
        raise ValueError(
            f"Captured indices for layer {layer} have shape={tuple(flat.shape)} numel={flat.numel()}; "
            f"worker expects {expected_numel} "
            f"(dispatch_group_size={dispatch_group_size}, seq_len_per_chip={seq_len_per_chip}, "
            f"num_experts_per_tok={num_experts_per_tok})"
        )
    indices = flat.view(dispatch_group_size, seq_len_per_chip, num_experts_per_tok).to(torch.int32).contiguous()
    max_idx = int(indices.max().item())
    if max_idx >= num_routed_experts:
        raise ValueError(f"Captured indices contain expert ID {max_idx} >= num_routed_experts={num_routed_experts}")

    # Remap captured Galaxy-global expert IDs [0, 256) → LB-local [0, 64) ∪ {sentinel}.
    # LB's combine kernel uses first_expert_id=0 on a single-col mesh, so metadata expert
    # IDs must fit in [0, num_routed_experts_per_col=64). In-col routings get shifted to
    # [0, 64); out-of-col routings get sentinel 255 which maps to -1 in col 0's dispatch
    # table → kernel skips (preserving the per-col routing share 1:1 with Galaxy col k).
    SENTINEL = 255
    experts_per_col = num_routed_experts // GALAXY_NUM_DISPATCH_GROUPS
    in_col_mask = (indices >= col * experts_per_col) & (indices < (col + 1) * experts_per_col)
    in_col_share = in_col_mask.float().mean().item() * 100.0
    indices = torch.where(
        in_col_mask,
        indices - col * experts_per_col,
        torch.tensor(SENTINEL, dtype=indices.dtype),
    ).contiguous()

    # Always use col 0's row of the (4, 256) Galaxy dispatch table — chip IDs [0, 8) for
    # experts [0, 64), and -1 for [64, 256). Combined with the remap above, this routes
    # in-col indices correctly and skips out-of-col (sentinel) ones.
    galaxy_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=GALAXY_NUM_DISPATCH_GROUPS,
    )
    expert_dispatch_table = galaxy_table[0:1].contiguous()

    logger.info(
        f"[captured_routing] layer={layer} col={col} src={path}: "
        f"indices.shape={tuple(indices.shape)} in-col share={in_col_share:.1f}% "
        f"(remapped to [0, {experts_per_col}) ∪ {{{SENTINEL}}})  "
        f"expert_dispatch_table.shape={tuple(expert_dispatch_table.shape)}"
    )
    return indices, expert_dispatch_table


def create_fabric_router_config(max_payload_size):
    """
    Helper to create FabricRouterConfig with custom max payload size.

    Args:
        max_payload_size: Maximum packet payload size in bytes

    Returns:
        FabricRouterConfig configured with the specified payload size
    """
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def get_dispatch_input_mesh_mapper(mesh_device, sp_axis: int):
    """
    Create mesh mapper for dispatch input tensors (x, weights, indices).

    These tensors are sharded across the SP axis and replicated across EP ranks.

    Args:
        mesh_device: TTNN mesh device
        sp_axis: Sequence parallel axis (0 or 1)

    Returns:
        ShardTensor2dMesh mapper configured for dispatch inputs
    """
    return ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(sp_axis, None),
    )


def get_ep_mesh_mapper(mesh_device):
    """
    Create mesh mapper for expert-parallel tensors.

    Shards tensor dim 1 across mesh rows (dispatch_group_size),
    tensor dim 0 across mesh cols (num_dispatch_groups).
    """
    return ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(1, 0),
    )


def get_expert_token_counts_mesh_mapper(mesh_device):
    """
    Create mesh mapper for expert_token_counts tensor.

    Shape: [num_dispatch_groups, dispatch_group_size, num_routed_experts]

    Shards dimension 1 (dispatch_group_size) across mesh rows and
    dimension 0 (num_dispatch_groups) across mesh columns.
    Each device receives [1, 1, num_routed_experts].
    """
    return ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(1, 0),  # Shard dim 1 across rows, shard dim 0 across cols
    )


def get_tp_mesh_composer(mesh_device):
    """
    Create mesh composer for TP sharded tensors; TP embedding dim over columns
    """
    return ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[0, -1],
        ),
    )


def get_sp_mesh_composer(mesh_device):
    """
    Create mesh composer for tensors replicated across TP columns.

    Composes along SP axis (column) only, taking column 0 from each row.
    Use for gate outputs that are replicated across TP after all_reduce_async
    (e.g., logits, topk_indices, topk_weights).
    """
    return ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            [0, 1],
            ttnn.MeshShape(mesh_device.shape[0], 1),
        ),
    )


def get_ep_mesh_composer(mesh_device):
    """
    Create mesh composer for expert-parallel tensors.

    Composes tensor dim 1 from mesh rows (dispatch_group_size),
    tensor dim 0 from mesh cols (num_dispatch_groups).
    """
    return ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(dims=[1, 0]),
    )


def create_gate_weights(
    num_routed_experts: int,
    emb_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    seed: int | None = None,
) -> dict:
    """
    Create random gate weights with proper scaling for stable sigmoid routing.

    Args:
        seed: When provided, weights are drawn from a local ``torch.Generator``
            seeded with this value, making the output a pure function of
            ``(num_routed_experts, emb_dim, dtype, seed)`` and independent of the
            global RNG state / call order. This is required when the result is
            persisted to a shape-keyed weight cache so that the cached tensor
            always matches the in-memory reference (see TtMoe cache builders).

    Returns dict matching MoEGate format:
        "weight": (n_routed_experts, dim)
        "e_score_correction_bias": (n_routed_experts,)
    """

    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    weight = torch.randn(num_routed_experts, emb_dim, dtype=dtype, generator=gen)
    scale = 1.0 / (emb_dim**0.5)  # kaiming-like scale
    weight = weight * scale
    return {
        "weight": weight,
        "e_score_correction_bias": torch.randn(num_routed_experts, dtype=dtype, generator=gen) * 0.01,
    }


def load_gate_weights_from_hf(
    model_id: str,
    layer_idx: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Load MoE gate (router) weights from a HuggingFace checkpoint.

    Args:
        model_id: HuggingFace model ID or local checkpoint path
        layer_idx: Transformer layer index (must be an MoE layer, i.e. >= 3 for DeepSeek-V3)
        dtype: Target dtype for the returned tensors

    Returns dict matching MoEGate / ``create_gate_weights`` format:
        "weight": (n_routed_experts, dim) — HF convention
        "e_score_correction_bias": (n_routed_experts,)

    Raises:
        FileNotFoundError: If checkpoint files cannot be found
        KeyError: If the expected gate keys are missing (e.g. non-MoE layer)
    """
    from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

    prefix = f"model.layers.{layer_idx}.mlp.gate."
    state_dict = load_hf_state_dict_filtered(model_id, [prefix])

    weight_key = f"model.layers.{layer_idx}.mlp.gate.weight"
    bias_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"

    if weight_key not in state_dict:
        raise KeyError(f"Gate weight not found at {weight_key}. Layer {layer_idx} may not be an MoE layer.")
    if bias_key not in state_dict:
        raise KeyError(f"Gate bias not found at {bias_key}. Layer {layer_idx} may not be an MoE layer.")

    gate_weight = state_dict[weight_key].to(dtype)
    gate_bias = state_dict[bias_key].to(dtype)

    logger.info(
        f"Loaded gate weights from {model_id} layer {layer_idx}: weight={gate_weight.shape}, bias={gate_bias.shape}"
    )
    return {
        "weight": gate_weight,
        "e_score_correction_bias": gate_bias,
    }


def create_torch_expert_weights(
    num_experts: int,
    emb_dim: int,
    hidden_dim: int,
    seed: int | None = None,
) -> list[dict]:
    """
    Create random weights for torch experts.

    Args:
        num_experts: Number of experts to create weights for
        emb_dim: Embedding dimension
        hidden_dim: Hidden/intermediate dimension
        seed: When provided, weights are drawn from a local ``torch.Generator``
            seeded with this value, making the output independent of the global
            RNG state / call order (required for stable shape-keyed weight caches).

    Returns:
        List of dicts with gate_proj, up_proj, down_proj per expert
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    weights_list = []
    for _ in tqdm(range(num_experts), desc="Creating expert weights"):
        weights = {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32, generator=gen) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32, generator=gen) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32, generator=gen) * 0.02,
        }
        weights_list.append(weights)
    return weights_list


def create_shared_expert_weights(
    emb_dim: int,
    hidden_dim: int,
    seed: int | None = None,
) -> dict:
    """
    Create random weights for shared expert in HF format.

    Args:
        emb_dim: Embedding dimension
        hidden_dim: Hidden/intermediate dimension
        seed: When provided, weights are drawn from a local ``torch.Generator``
            seeded with this value, making the output independent of the global
            RNG state / call order (required for stable shape-keyed weight caches).

    Returns:
        Dict with gate_proj, up_proj, down_proj in HF format (out_features, in_features)
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32, generator=gen) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32, generator=gen) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32, generator=gen) * 0.02,
    }


def create_sparse_combine_output(
    num_chips: int,
    seq_len: int,
    topk: int,
    emb_dim: int,
    sparsity: float = 0.75,
) -> torch.Tensor:
    """
    Create synthetic sparse combine output for testing.

    In real MoE, combine output is sparse because each chip only has valid data
    for tokens routed to its local experts. This function simulates that sparsity.

    Args:
        num_chips: Number of chips in the reduction group
        seq_len: Sequence length per chip
        topk: Number of expert slots per token
        emb_dim: Embedding dimension
        sparsity: Fraction of positions that are zero (default 0.75)

    Returns:
        Sparse tensor of shape [num_chips, seq_len, topk, emb_dim]
    """

    # Create random data
    data = torch.randn(num_chips, seq_len, topk, emb_dim, dtype=torch.bfloat16)

    # Apply sparsity mask (zero out random positions in topk dimension)
    mask = torch.rand(num_chips, seq_len, topk, 1) > sparsity
    data = data * mask.to(torch.bfloat16)

    return data
