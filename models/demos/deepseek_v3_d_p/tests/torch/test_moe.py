# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for DeepSeek V3-like MoE architecture (PyTorch reference implementation).

This test validates the full MoE dispatch -> expert -> combine -> weighted sum flow:
1. Tokens are dispatched to expert buffers based on router indices
2. Routed experts (FFN networks) process their assigned tokens
3. Expert outputs are combined back to original token positions
4. Gate weights are applied to each expert contribution (split connection)
5. Shared expert output is added to the final result

Configuration:
- 24 routed experts (each is an FFN with gate_proj, up_proj, down_proj)
- num_experts_per_tok = 4 (each token routes to 4 experts)
- 1 shared expert (same FFN structure as routed experts)
- Dispatch group size = 4
- All experts initialized with identity matrices for flow verification
"""

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    get_gate_outputs,
    initialize_test_inputs,
)


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, num_routed_experts, num_experts_per_tok, dispatch_group_size, capacity_factor, model_id, layer_idx",
    [
        # Identity weights (fast, for flow testing)
        pytest.param(32, 64, 24, 4, 4, 2, None, None, id="identity-weights"),
        # Real HF weights (slow, downloads ~18GB) - DeepSeek V3 layer 3
        pytest.param(
            32,
            DeepSeekV3Config.HIDDEN_SIZE,
            DeepSeekV3Config.NUM_ROUTED_EXPERTS,
            DeepSeekV3Config.NUM_EXPERTS_PER_TOKEN,
            4,
            4,
            "deepseek-ai/DeepSeek-V3",
            3,
            id="hf-weights",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_moe(
    seq_len_per_chip,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_group_size,
    capacity_factor,
    model_id,
    layer_idx,
):
    """
    Test TorchMoe module with return_intermediates flag.

    Validates that the module produces correct output and intermediates.
    Can run with identity weights (fast) or real HF weights (slow).
    """
    use_real_weights = model_id is not None

    logger.debug(f"\n{'='*60}")
    logger.debug(f"TorchMoe Test {'(Real Weights)' if use_real_weights else '(Identity Weights)'}")
    if use_real_weights:
        logger.debug(f"Model: {model_id}, Layer: {layer_idx}")
    logger.debug(f"{'='*60}\n")

    # Compute derived constants
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices=dispatch_group_size,
        dispatch_group_size=dispatch_group_size,
        capacity_factor=capacity_factor,
    )

    # Initialize test inputs
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=42,
    )

    # Create expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )

    # Compute gate outputs
    expert_offsets, expert_token_counts, cum_sum = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Create MinimalMoE module
    logger.debug(f"Creating MoE{' with real weights from ' + model_id if use_real_weights else ''}...")
    moe = TorchMoe(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        expert_dispatch_table=expert_dispatch_table,
        model_id=model_id,
        layer_idx=layer_idx,
    )

    if use_real_weights:
        # Log weight shapes from first routed expert
        logger.debug("Weight shapes (first routed expert):")
        logger.debug(f"  gate_proj: {moe.routed_experts[0].gate_proj.shape}")
        logger.debug(f"  up_proj: {moe.routed_experts[0].up_proj.shape}")
        logger.debug(f"  down_proj: {moe.routed_experts[0].down_proj.shape}")

    # Test without intermediates (only for identity weights - faster)
    if not use_real_weights:
        logger.debug("Testing forward pass without intermediates...")
        final_output, intermediates = moe(
            x, weights, indices, expert_offsets, expert_token_counts, return_intermediates=False
        )
        assert intermediates is None, "Expected no intermediates when return_intermediates=False"
        assert final_output.shape == x.shape, f"Expected output shape {x.shape}, got {final_output.shape}"
        logger.debug(f"Output shape: {final_output.shape}")
        logger.debug(f"Output sum (abs): {final_output.abs().sum().item():.4f}")

    # Test with intermediates
    logger.debug("\nTesting forward pass with intermediates...")
    final_output_2, intermediates = moe(
        x, weights, indices, expert_offsets, expert_token_counts, return_intermediates=True
    )
    assert intermediates is not None, "Expected intermediates when return_intermediates=True"

    # Verify intermediates shapes
    logger.debug("Intermediate shapes:")
    logger.debug(f"  dispatched_buffer: {intermediates.dispatched_buffer.shape}")
    logger.debug(f"  metadata: {intermediates.metadata.shape}")
    logger.debug(f"  expert_outputs: {intermediates.expert_outputs.shape}")
    logger.debug(f"  shared_output: {intermediates.shared_output.shape}")
    logger.debug(f"  combined_output: {intermediates.combined_output.shape}")
    logger.debug(f"  routed_output: {intermediates.routed_output.shape}")

    # Verify shapes
    assert intermediates.dispatched_buffer.shape == (
        1,
        dispatch_group_size,
        experts_per_chip,
        max_dispatched_tokens_per_expert,
        hidden_dim,
    )
    assert intermediates.shared_output.shape == (dispatch_group_size, seq_len_per_chip, hidden_dim)
    assert intermediates.combined_output.shape == (
        dispatch_group_size,
        seq_len_per_chip,
        num_experts_per_tok,
        hidden_dim,
    )
    assert intermediates.routed_output.shape == (dispatch_group_size, seq_len_per_chip, hidden_dim)

    # Verify both runs produce same output (only for identity weights)
    if not use_real_weights:
        assert torch.allclose(
            final_output, final_output_2
        ), "Outputs should be identical regardless of return_intermediates"

    # Verify no NaN/Inf
    assert not torch.isnan(final_output_2).any(), "Final output contains NaN values"
    assert not torch.isinf(final_output_2).any(), "Final output contains Inf values"
    assert not torch.isnan(intermediates.shared_output).any(), "Shared expert output contains NaN"
    assert not torch.isnan(intermediates.routed_output).any(), "Routed expert output contains NaN"

    logger.debug(
        f"\nOutput stats - min: {final_output_2.min().item():.4f}, max: {final_output_2.max().item():.4f}, mean: {final_output_2.mean().item():.4f}"
    )

    logger.debug("\n" + "=" * 60)
    logger.debug("TorchMoe Test PASSED!")
    logger.debug("=" * 60)
