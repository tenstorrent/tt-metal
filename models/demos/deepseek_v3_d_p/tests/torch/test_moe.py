# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
- Experts initialized with random weights for flow verification
"""

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_gate_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    get_gate_outputs,
    initialize_test_inputs,
)


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, dispatch_group_size, capacity_factor, use_gate, model_id, layer_idx",
    [
        # fmt: off
        pytest.param(32, 64, 128, 24, 4, 4, 2, False, None, None, id="random-weights"),
        pytest.param(32, 224, 64, 256, 8, 4, 4, True, None, None, id="random-weights-gate"),
        pytest.param(32,DeepSeekV3Config.EMB_SIZE,DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,DeepSeekV3Config.NUM_ROUTED_EXPERTS,DeepSeekV3Config.NUM_EXPERTS_PER_TOKEN,4,4,False,"deepseek-ai/DeepSeek-V3",3,id="hf-weights",marks=pytest.mark.slow,
        ),
        # fmt: on
    ],
)
def test_moe(
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_group_size,
    capacity_factor,
    use_gate,
    model_id,
    layer_idx,
):
    """
    Test TorchMoe module with and without integrated gate.

    Without gate: pre-computed weights/indices are passed to forward.
    With gate: gate_weights are passed to TorchMoe, forward only takes x.
    Can run with random weights (fast) or real HF weights (slow).
    """
    torch.manual_seed(42)
    use_hf_weights = model_id is not None

    logger.debug(f"\n{'='*60}")
    label = "HF Weights" if use_hf_weights else ("Gate" if use_gate else "Random Weights")
    logger.debug(f"TorchMoe Test ({label})")
    if use_hf_weights:
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

    # Create expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )

    # Create weights
    if use_hf_weights:
        routed_expert_weights = None
        shared_expert_weights = None
        gate_weights_dict = None
    else:
        logger.debug("Creating random weights for experts...")
        routed_expert_weights = create_torch_expert_weights(num_routed_experts, emb_dim, hidden_dim)
        shared_expert_weights = create_shared_expert_weights(emb_dim, hidden_dim)
        if use_gate:
            gate_weights_dict = create_gate_weights(num_routed_experts, emb_dim, dtype=torch.float32)
        else:
            gate_weights_dict = None

    # Prepare gate inputs (pre-computed weights/indices) when not using gate
    if not use_gate:
        x, weights, indices = initialize_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        )
        expert_offsets, expert_token_counts, _ = get_gate_outputs(
            indices,
            dispatch_group_size,
            num_routed_experts,
            experts_per_chip,
            seq_len_per_chip,
            num_experts_per_tok,
            expert_dispatch_table=expert_dispatch_table,
        )
    else:
        x = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.float32)

    # Create TorchMoe module
    logger.debug(f"Creating MoE{' with HF weights from ' + model_id if use_hf_weights else ' with random weights'}...")
    moe = TorchMoe(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        expert_dispatch_table=expert_dispatch_table,
        model_id=model_id,
        layer_idx=layer_idx,
        routed_expert_weights=routed_expert_weights,
        shared_expert_weights=shared_expert_weights,
        gate_weights=gate_weights_dict,
    )

    if use_hf_weights:
        logger.debug("Weight shapes (first routed expert):")
        logger.debug(f"  gate_proj: {moe.routed_experts[0].gate_proj.shape}")
        logger.debug(f"  up_proj: {moe.routed_experts[0].up_proj.shape}")
        logger.debug(f"  down_proj: {moe.routed_experts[0].down_proj.shape}")

    # Test without intermediates (only for random weights without gate - faster)
    if not use_hf_weights and not use_gate:
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
    if use_gate:
        final_output_2, intermediates = moe(x, return_intermediates=True)
    else:
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

    assert intermediates.dispatched_buffer.shape == (
        1,
        dispatch_group_size,
        experts_per_chip,
        max_dispatched_tokens_per_expert,
        emb_dim,
    )
    assert intermediates.shared_output.shape == (dispatch_group_size, seq_len_per_chip, emb_dim)
    assert intermediates.combined_output.shape == (
        dispatch_group_size,
        seq_len_per_chip,
        num_experts_per_tok,
        emb_dim,
    )
    assert intermediates.routed_output.shape == (dispatch_group_size, seq_len_per_chip, emb_dim)

    # Gate-specific checks
    if use_gate:
        assert intermediates.gate_scores is not None, "Expected gate_scores in intermediates"
        assert intermediates.gate_indices is not None, "Expected gate_indices in intermediates"
        logger.debug(f"Gate scores shape: {intermediates.gate_scores.shape}")
        logger.debug(f"Gate indices shape: {intermediates.gate_indices.shape}")

    # Verify both runs produce same output (only for random weights without gate)
    if not use_hf_weights and not use_gate:
        assert torch.allclose(
            final_output, final_output_2
        ), "Outputs should be identical regardless of return_intermediates"

    assert final_output_2.shape == x.shape, f"Expected output shape {x.shape}, got {final_output_2.shape}"

    # Verify no NaN/Inf
    assert not torch.isnan(final_output_2).any(), "Final output contains NaN values"
    assert not torch.isinf(final_output_2).any(), "Final output contains Inf values"
    assert not torch.isnan(intermediates.shared_output).any(), "Shared expert output contains NaN"
    assert not torch.isnan(intermediates.routed_output).any(), "Routed expert output contains NaN"

    logger.debug(
        f"\nOutput stats - min: {final_output_2.min().item():.4f}, max: {final_output_2.max().item():.4f}, mean: {final_output_2.mean().item():.4f}"
    )

    logger.debug("\n" + "=" * 60)
    logger.debug(f"TorchMoe Test ({label}) PASSED!")
    logger.debug("=" * 60)
