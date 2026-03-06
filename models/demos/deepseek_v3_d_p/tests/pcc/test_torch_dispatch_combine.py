"""
Test for end-to-end MoE dispatch→combine round-trip using PyTorch reference implementation.

This test verifies that tokens dispatched to experts and then combined back produce the
original input, validating the full round-trip through the dispatch and combine operations.
"""

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.common import (
    compute_constants,
    create_expert_dispatch_table,
    get_gate_outputs,
    initialize_test_inputs,
)


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor",
    [
        (32, 64, 16, 4, 2, 2),
        (512, 32, 256, 8, 4, 2),
    ],
    ids=["xs", "small"],
)
def test_torch_dispatch_combine(
    seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
):
    """Test dispatch→combine round-trip using PyTorch reference implementation."""
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )
    print("\n")

    # Initialize inputs using helper function
    x, weights, indices = initialize_test_inputs(
        num_chips=num_chips,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=42,
    )
    # Squeeze the ep_rank dimension since this is a single-rank pure torch test
    weights = weights.squeeze(0)

    # Create expert dispatch table (single EP rank for this test)
    expert_dispatch_table = create_expert_dispatch_table(
        n_routed_experts=n_routed_experts,
        num_chips_sp=num_chips,
        num_chips_rep=1,
    )
    logger.info(f"{expert_dispatch_table.shape=}")
    logger.info(f"{expert_dispatch_table=}")

    # Initialize dispatch and combine modules
    dispatch_module = TorchDispatchModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        expert_dispatch_table=expert_dispatch_table,
    )
    combine_module = TorchCombineModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
    )

    # Compute gate outputs before dispatch
    chip_to_n_routed_expert_offset, experts_counter, cum_sum = get_gate_outputs(
        indices,
        num_chips,
        n_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Forward pass through dispatch module
    logger.info(f"{x.shape=}")
    logger.info(f"{weights.shape=}")
    logger.info(f"{indices.shape=}")
    dispatched, metadata = dispatch_module(x, weights, indices, chip_to_n_routed_expert_offset)

    torch.set_printoptions(profile="full")
    logger.info(f"{experts_counter.shape=}")
    logger.info(f"{metadata.shape=}")
    logger.info(f"{dispatched.shape=}")
    torch.set_printoptions(profile="default")

    # Forward pass through combine module
    y = combine_module(
        dispatched,
        metadata,
        experts_counter,
    )
    logger.info(f"{y.shape=}")
    y /= num_experts_per_tok  # since we are summing contributions from multiple experts, we need to average them
    y = y.sum(dim=2)  # sum contributions from multiple experts per token
    logger.info(f"{y.shape=}")
    assert torch.allclose(
        x, y, atol=1e-6
    ), f"Expected output to match input, but got max diff {torch.max(torch.abs(x-y)).item()}"
