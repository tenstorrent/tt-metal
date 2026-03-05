"""
Common utilities for MoE testing and configuration.

This module provides shared helper functions used across MoE tests including:
- Configuration computation (compute_constants)
- Test input generation (initialize_test_inputs, initialize_predictable_test_inputs)
- Fabric configuration helpers (create_fabric_router_config)
"""

import torch
from loguru import logger

import ttnn


def get_gate_outputs(
    indices: torch.Tensor,
    num_chips: int,
    n_routed_experts: int,
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
        indices: Expert indices tensor (num_chips, seq_len_per_chip, num_experts_per_tok)
        num_chips: Number of chips in the system
        n_routed_experts: Total number of routed experts across all chips
        experts_per_chip: Number of experts per chip
        seq_len_per_chip: Sequence length per chip
        num_experts_per_tok: Number of experts each token routes to

    Returns:
        chip_to_n_routed_expert_offset: Base offset for each expert from each chip
            Shape: (num_chips, n_routed_experts)
        chip_to_routed_expert_tokens: Total tokens per expert per chip
            Shape: (num_chips, experts_per_chip)
        cum_sum: Cumulative sum of token counts across chips
            Shape: (num_chips, n_routed_experts)
    """
    # Count tokens per expert per chip
    chip_to_n_routed_expert_counter = torch.zeros((num_chips, n_routed_experts), dtype=torch.int32)
    for chip in range(num_chips):
        for token in range(seq_len_per_chip):
            for topk_indice in range(num_experts_per_tok):
                routed_expert = indices[chip, token, topk_indice]
                chip_to_n_routed_expert_counter[chip, routed_expert] += 1

    # Compute cumulative offsets
    cum_sum = torch.cumsum(chip_to_n_routed_expert_counter, dim=0)
    chip_to_n_routed_expert_offset = torch.vstack([torch.zeros([1, n_routed_experts], dtype=torch.int32), cum_sum[:-1]])
    chip_to_routed_expert_tokens = cum_sum[-1].view(num_chips, experts_per_chip).to(torch.int32)

    return chip_to_n_routed_expert_offset, chip_to_routed_expert_tokens, cum_sum


def compute_constants(seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor):
    """
    Compute derived constants for MoE configuration.

    Args:
        seq_len_per_chip: Sequence length per chip
        n_routed_experts: Total number of routed experts across all chips
        num_experts_per_tok: Number of experts each token is routed to
        num_chips: Number of chips in the system
        capacity_factor: Capacity factor for load balancing

    Returns:
        experts_per_chip: Number of experts per chip
        metadata_len: Length of metadata per token
        max_dispatched_tokens_per_expert: Maximum tokens per expert
    """
    experts_per_chip = n_routed_experts // num_chips
    metadata_len = 5  # chip, token, topk_indice, routed_expert, weight
    balanced_load = num_chips * seq_len_per_chip * num_experts_per_tok // n_routed_experts
    max_dispatched_tokens_per_expert = int(balanced_load * capacity_factor)
    return experts_per_chip, metadata_len, max_dispatched_tokens_per_expert


def initialize_test_inputs(
    num_chips: int,
    seq_len_per_chip: int,
    hidden_dim: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    max_dispatched_tokens_per_expert: int,
    seed: int = 42,
    validate: bool = True,
    num_ep_ranks: int = 1,
):
    """
    Initialize test inputs (x, weights, indices) with random data.

    Args:
        num_chips: Number of chips in the system
        seq_len_per_chip: Sequence length per chip
        hidden_dim: Hidden dimension
        n_routed_experts: Total number of routed experts across all chips
        num_experts_per_tok: Number of experts each token is routed to
        max_dispatched_tokens_per_expert: Maximum number of tokens per expert
        seed: Random seed for reproducibility
        validate: Whether to validate expert activations
        num_ep_ranks: Number of Expert Parallelism ranks (for different weights per rank)

    Returns:
        x: Input tensor (num_chips, seq_len_per_chip, hidden_dim)
        weights: Router weights (num_ep_ranks, num_chips, seq_len_per_chip, num_experts_per_tok)
        indices: Expert indices (num_chips, seq_len_per_chip, num_experts_per_tok)
    """
    torch.manual_seed(seed)

    input_shape = (num_chips, seq_len_per_chip, hidden_dim)
    x = torch.randn(input_shape, dtype=torch.bfloat16)

    weights_shape = (num_ep_ranks, num_chips, seq_len_per_chip, num_experts_per_tok)
    indices_shape = (num_chips, seq_len_per_chip, num_experts_per_tok)

    weights = torch.randn(weights_shape, dtype=torch.bfloat16)
    indices = torch.randint(0, n_routed_experts, indices_shape, dtype=torch.int32)

    # Validate expert activations
    if validate:
        expert_activations = torch.zeros((n_routed_experts,), dtype=torch.int32)
        for c in range(indices.shape[0]):
            for t in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    expert_activations[indices[c, t, k]] += 1
        checksum = expert_activations.sum().item()
        logger.info(f"{expert_activations.shape=}")
        assert (
            checksum == num_chips * seq_len_per_chip * num_experts_per_tok
        ), f"Expected checksum {num_chips * seq_len_per_chip * num_experts_per_tok}, got {checksum}"
        assert (
            expert_activations.max().item() <= max_dispatched_tokens_per_expert
        ), f"Expected max activations per expert to be <= {max_dispatched_tokens_per_expert}, got {expert_activations.max().item()}"

    return x, weights, indices


def initialize_predictable_test_inputs(
    num_chips: int,
    seq_len_per_chip: int,
    hidden_dim: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    max_dispatched_tokens_per_expert: int,
    num_ep_ranks: int = 1,
):
    """
    Initialize test inputs with predictable patterns for debugging.

    Pattern:
    - x: Simple sequential values starting from 0.0
    - weights: Sequential values per EP rank
      - Rank 0: [1, 2, 3, 4], Rank 1: [5, 6, 7, 8], etc. (for num_experts_per_tok=4)
    - indices: Round-robin pattern cycling through experts

    This makes it easy to verify writes:
    - Token 0 -> experts [0, 1, 2, 3]
    - Token 1 -> experts [4, 5, 6, 7]
    - Token 2 -> experts [8, 9, 10, 11]
    - etc.

    Args:
        num_chips: Number of chips in the system
        seq_len_per_chip: Sequence length per chip
        hidden_dim: Hidden dimension
        n_routed_experts: Total number of routed experts across all chips
        num_experts_per_tok: Number of experts each token is routed to
        max_dispatched_tokens_per_expert: Maximum number of tokens per expert (unused but kept for signature consistency)
        num_ep_ranks: Number of Expert Parallelism ranks (for different weights per rank)

    Returns:
        x: Input tensor (num_chips, seq_len_per_chip, hidden_dim)
        weights: Router weights (num_ep_ranks, num_chips, seq_len_per_chip, num_experts_per_tok)
        indices: Expert indices (num_chips, seq_len_per_chip, num_experts_per_tok)
    """
    input_shape = (num_chips, seq_len_per_chip, hidden_dim)
    # Fill with sequential values: 0.0, 1.0, 2.0, ...
    x = torch.arange(num_chips * seq_len_per_chip * hidden_dim, dtype=torch.float32).reshape(input_shape)
    x = x.to(torch.bfloat16)

    weights_shape = (num_ep_ranks, num_chips, seq_len_per_chip, num_experts_per_tok)
    indices_shape = (num_chips, seq_len_per_chip, num_experts_per_tok)

    # Predictable weights: rank 0 = [1,2,3,4], rank 1 = [5,6,7,8], etc.
    weights = torch.zeros(weights_shape, dtype=torch.bfloat16)
    for r in range(num_ep_ranks):
        for k in range(num_experts_per_tok):
            weights[r, :, :, k] = float(r * num_experts_per_tok + k + 1)

    # Round-robin indices pattern
    indices = torch.zeros(indices_shape, dtype=torch.int32)
    expert_idx = 0
    for chip in range(num_chips):
        for token in range(seq_len_per_chip):
            for k in range(num_experts_per_tok):
                if chip % 2 == 0:
                    indices[chip, token, k] = max(
                        0, expert_idx % (n_routed_experts) - 1
                    )  # max (0, x -1) to create a of unequal distribution
                else:
                    indices[chip, token, k] = n_routed_experts - 1 - (expert_idx % n_routed_experts)  # reverse order
                expert_idx += 1

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
