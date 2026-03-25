# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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

import ttnn


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
    if mesh_device.shape[0] > 1 and mesh_device.shape[1] > 1:
        sp_axis = 0
        dispatch_group_size = mesh_device.shape[sp_axis]
        num_dispatch_groups = mesh_device.shape[1]
    else:
        dispatch_group_size = mesh_device.get_num_devices()
        num_dispatch_groups = 1
        sp_axis = 0 if mesh_device.shape[0] > 1 else 1

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
            expert_dispatch_table: Shape (num_dispatch_groups, num_routed_experts)
                Values are logical chip IDs (0 to dispatch_group_size-1) or -1 if not present

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

        table = torch.full((num_dispatch_groups, num_routed_experts), -1, dtype=torch.int32)
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
) -> tuple:
    """
    Compute dispatch offsets and token counts from router indices.

    This processes the gate/router output indices to determine:
    1. Where each token should be written in the dispatch buffer (offsets)
    2. How many tokens each expert receives (counter)

    Args:
        indices: Expert indices tensor (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
        dispatch_group_size: Number of chips in each dispatch group
        num_routed_experts: Total number of routed experts across all chips
        experts_per_chip: Number of experts per chip
        seq_len_per_chip: Sequence length per chip
        num_experts_per_tok: Number of experts each token routes to

    Returns:
        expert_offsets: Base offset for each expert from each chip
            Shape: (dispatch_group_size, num_routed_experts)
        expert_token_counts: Total tokens per expert per chip
            Shape: (dispatch_group_size, experts_per_chip)
        cum_sum: Cumulative sum of token counts across chips
            Shape: (dispatch_group_size, num_routed_experts)
    """
    # Count tokens per expert per chip
    expert_counter = torch.zeros((dispatch_group_size, num_routed_experts), dtype=torch.int32)
    for chip in range(dispatch_group_size):
        for token in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                routed_expert = indices[chip, token, topk_idx]
                expert_counter[chip, routed_expert] += 1

    # Compute cumulative offsets
    cum_sum = torch.cumsum(expert_counter, dim=0)
    expert_offsets = torch.vstack([torch.zeros([1, num_routed_experts], dtype=torch.int32), cum_sum[:-1]])
    expert_token_counts = (
        cum_sum[-1]
        .view(num_routed_experts // experts_per_chip // dispatch_group_size, dispatch_group_size, experts_per_chip)
        .to(torch.int32)
    )
    # expert_token_counts = expert_token_counts.permute(1, 0, 2)
    logger.debug(f"[get_gate_outputs] OUTPUT SHAPES:")
    logger.debug(f"  expert_counter.shape={expert_counter.shape}")
    logger.debug(f"  expert_offsets.shape={expert_offsets.shape}")
    logger.debug(f"  expert_token_counts.shape={expert_token_counts.shape}")
    logger.debug(f"  cum_sum.shape={cum_sum.shape}")
    return expert_offsets, expert_token_counts, cum_sum, expert_counter


def compute_constants(
    seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
):
    """
    Compute derived constants for MoE configuration.

    Args:
        seq_len_per_chip: Sequence length per chip
        num_routed_experts: Total number of routed experts across all chips
        num_experts_per_tok: Number of experts each token is routed to
        num_devices: Number of devices across which experts are distributed
        dispatch_group_size: Number of devices in each dispatch group
        capacity_factor: Capacity factor for load balancing

    Returns:
        experts_per_chip: Number of experts per chip
        metadata_len: Length of metadata per token
        max_dispatched_tokens_per_expert: Maximum tokens per expert
    """
    experts_per_chip = num_routed_experts // num_devices
    metadata_len = 5  # chip, token, topk_idx, routed_expert, weight
    # total number of tokens in group times x distribution ratio (8/256 == 2/64)
    balanced_load = (dispatch_group_size * seq_len_per_chip) * num_experts_per_tok // num_routed_experts
    max_dispatched_tokens_per_expert = int(balanced_load * capacity_factor)
    return experts_per_chip, metadata_len, max_dispatched_tokens_per_expert


def initialize_test_inputs(
    dispatch_group_size: int,
    seq_len_per_chip: int,
    emb_dim: int,
    num_routed_experts: int,
    num_experts_per_tok: int,
    max_dispatched_tokens_per_expert: int,
    seed: int = 42,
    validate: bool = True,
    num_dispatch_groups: int = 1,
    skip_x_initialization: bool = False,
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
        seed: Random seed for reproducibility
        validate: Whether to validate expert activations
        num_dispatch_groups: Number of parallel dispatch groups

    Returns:
        x: Input tensor (dispatch_group_size, seq_len_per_chip, emb_dim)
        weights: Router weights (num_dispatch_groups, dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
        indices: Expert indices (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    """
    torch.manual_seed(seed)

    input_shape = (dispatch_group_size, seq_len_per_chip, emb_dim)
    x = torch.randn(input_shape, dtype=torch.bfloat16) if not skip_x_initialization else None

    weights_shape = (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
    indices_shape = (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)

    weights = torch.randn(weights_shape, dtype=torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize so topk sums to 1
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


def create_sparse_combine_output(
    num_chips: int,
    seq_len: int,
    topk: int,
    emb_dim: int,
    sparsity: float = 0.75,
    seed: int = 42,
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
        seed: Random seed for reproducibility

    Returns:
        Sparse tensor of shape [num_chips, seq_len, topk, emb_dim]
    """
    torch.manual_seed(seed)

    # Create random data
    data = torch.randn(num_chips, seq_len, topk, emb_dim, dtype=torch.bfloat16)

    # Apply sparsity mask (zero out random positions in topk dimension)
    mask = torch.rand(num_chips, seq_len, topk, 1) > sparsity
    data = data * mask.to(torch.bfloat16)

    return data
