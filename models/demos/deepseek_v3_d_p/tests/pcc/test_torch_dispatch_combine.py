# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import ValidationResult
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, num_routed_experts, num_experts_per_tok, dispatch_group_size, capacity_factor",
    [
        (32, 64, 16, 4, 2, 2),
        (512, 32, 256, 8, 4, 2),
    ],
    ids=["xs", "small"],
)
def test_torch_dispatch_combine(
    seq_len_per_chip, hidden_dim, num_routed_experts, num_experts_per_tok, dispatch_group_size, capacity_factor
):
    """Test dispatch→combine round-trip using PyTorch reference implementation."""
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices=dispatch_group_size,
        dispatch_group_size=dispatch_group_size,
        capacity_factor=capacity_factor,
    )
    print("\n")

    # Initialize inputs using helper function
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=42,
    )
    # Squeeze the dispatch_group dimension since this is a single-rank pure torch test
    weights = weights.squeeze(0)

    # Create expert dispatch table (single EP rank for this test)
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )
    logger.debug(f"{expert_dispatch_table.shape=}")
    logger.debug(f"{expert_dispatch_table=}")

    # Initialize dispatch and combine modules
    dispatch_module = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        expert_dispatch_table=expert_dispatch_table,
    )
    combine_module = TorchCombineModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
    )

    # Compute gate outputs before dispatch
    expert_offsets, expert_token_counts, cum_sum = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Forward pass through dispatch module
    logger.debug(f"{x.shape=}")
    logger.debug(f"{weights.shape=}")
    logger.debug(f"{indices.shape=}")
    dispatched, metadata = dispatch_module(x, weights, indices, expert_offsets)

    torch.set_printoptions(profile="full")
    logger.debug(f"{expert_token_counts.shape=}")
    logger.debug(f"{metadata.shape=}")
    logger.debug(f"{dispatched.shape=}")
    torch.set_printoptions(profile="default")

    # Forward pass through combine module
    y = combine_module(
        dispatched,
        metadata,
        expert_token_counts,
    )
    logger.debug(f"{y.shape=}")
    y /= num_experts_per_tok  # since we are summing contributions from multiple experts, we need to average them
    y = y.sum(dim=2)  # sum contributions from multiple experts per token
    logger.debug(f"{y.shape=}")
    assert torch.allclose(
        x, y, atol=1e-6
    ), f"Expected output to match input, but got max diff {torch.max(torch.abs(x-y)).item()}"


@pytest.mark.parametrize("dispatch_group_size,num_dispatch_groups", [(4, 2), (2, 4), (8, 4)], ids=["4x2", "2x4", "8x4"])
def test_visualize_expert_dispatch_table(dispatch_group_size, num_dispatch_groups):
    """Visualize expert dispatch table for different mesh configurations."""
    num_routed_experts = 256
    num_experts_per_tok = 8

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    log_expert_dispatch_table(
        expert_dispatch_table=expert_dispatch_table,
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=num_routed_experts,
        title=f"Expert Dispatch Table ({dispatch_group_size}x{num_dispatch_groups} mesh, {num_routed_experts} experts, topk={num_experts_per_tok})",
    )


def test_visualize_validation_results_synthetic():
    """Test validation visualization with synthetic pass/fail data."""
    num_dispatch_groups = 4
    dispatch_group_size = 2

    # Create synthetic buffer result: DG0 and DG2 pass, DG1 chip0 fails, DG3 chip1 fails
    buffer_mismatches = [
        (1, 0, 0, "synthetic buffer failure"),  # DG1, chip0
        (3, 1, 1, "synthetic buffer failure"),  # DG3, chip1
    ]
    buffer_validated = {(dg, chip) for dg in range(num_dispatch_groups) for chip in range(dispatch_group_size)}
    buffer_result = ValidationResult(
        passed=False,
        matches=6,
        total=8,
        mismatches=buffer_mismatches,
        name="buffer",
        validated_cells=buffer_validated,
    )

    # Create synthetic metadata result: DG0 passes, DG1 all fail, DG2 chip1 fails, DG3 passes
    metadata_mismatches = [
        (1, 0, 0, "synthetic metadata failure"),  # DG1, chip0
        (1, 1, 0, "synthetic metadata failure"),  # DG1, chip1
        (2, 1, 0, "synthetic metadata failure"),  # DG2, chip1
    ]
    metadata_validated = {(dg, chip) for dg in range(num_dispatch_groups) for chip in range(dispatch_group_size)}
    metadata_result = ValidationResult(
        passed=False,
        matches=5,
        total=8,
        mismatches=metadata_mismatches,
        name="metadata",
        validated_cells=metadata_validated,
    )

    logger.debug("\n=== Synthetic Validation Results Test ===")
    logger.debug("Expected pattern:")
    logger.debug("  DG0: ✅✅ (both pass)")
    logger.debug("  DG1: ❌❌ chip0, ✅❌ chip1 (buffer pass chip1, metadata fail both)")
    logger.debug("  DG2: ✅✅ chip0, ✅❌ chip1 (metadata fail chip1)")
    logger.debug("  DG3: ✅✅ chip0, ❌✅ chip1 (buffer fail chip1)")

    log_validation_results(
        results=[buffer_result, metadata_result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Synthetic Validation Results",
    )
