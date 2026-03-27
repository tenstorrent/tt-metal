# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test deepseek_moe_post_combine_reduce with known, predictable values.

This test uses simple patterns to make it easy to track data flow:
- Each expert has a unique constant value
- Weights are predictable
- Expected output is easy to manually verify
"""

import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import comp_pcc


def create_incremental_activations(seq_len, num_experts, emb_dim, start_val=0.0, increment=0.1, dtype=torch.bfloat16):
    """
    Create combine_output with incremental values across all experts.

    Values increment by 0.1 across the entire 8×7168 tensor:
    [0.0, 0.1, 0.2, 0.3, ..., 5734.3]

    Shape: [1, 1, seq_len, num_experts, emb_dim]
    """
    total_elements = num_experts * emb_dim

    # Create incremental values
    values = torch.arange(0, total_elements, dtype=torch.float32) * increment + start_val
    values = values.to(dtype)

    # Reshape to [num_experts, emb_dim]
    values = values.reshape(num_experts, emb_dim)

    # Expand to full shape [1, 1, seq_len, num_experts, emb_dim]
    data = values.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, -1, -1).contiguous()

    return data


def create_simple_weights(seq_len, num_experts, dtype=torch.bfloat16):
    """
    Create weights: [1, 2, 3, 4, 5, 6, 7, 8] for 8 experts.

    Shape: [1, 1, seq_len, num_experts]
    """
    weights = torch.zeros(1, 1, seq_len, num_experts, dtype=dtype)

    for expert_idx in range(num_experts):
        # Weight for expert i = i + 1 (so expert 0 gets weight 1, expert 1 gets weight 2, etc.)
        weights[:, :, :, expert_idx] = expert_idx + 1

    return weights


def compute_expected_output(combine_output, weights):
    """
    Reference implementation: weighted sum across experts.

    output[seq, emb] = sum over experts of (combine_output[seq, expert, emb] * weights[seq, expert])
    """
    # Expand weights: [1, 1, seq_len, num_experts] -> [1, 1, seq_len, num_experts, 1]
    weights_expanded = weights.unsqueeze(-1)

    # Element-wise multiply
    weighted = combine_output * weights_expanded

    # Sum across expert dimension (dim=3)
    result = weighted.sum(dim=3)

    return result


def test_known_values_simple(device):
    """
    Test with incremental activation values and simple weights.

    Activations: 8 experts × 7168 elements = 57344 total values
                 Values: [0.0, 0.1, 0.2, 0.3, ..., 5734.3]

    Weights: [1, 2, 3, 4, 5, 6, 7, 8] for experts 0-7

    Expected behavior:
    Expert 0 (weight=1): elements [0.0, 0.1, 0.2, ..., 716.7]
    Expert 1 (weight=2): elements [716.8, 716.9, 717.0, ..., 1433.5]
    ...
    Expert 7 (weight=8): elements [5017.6, 5017.7, ..., 5734.3]

    Each output element at position i is the sum of:
      expert_0[i] * 1 + expert_1[i] * 2 + ... + expert_7[i] * 8
    """
    seq_len = 1
    num_experts = 8
    emb_dim = 7168  # DeepSeek actual size

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing with INCREMENTAL VALUES")
    logger.info(f"  seq_len={seq_len}, num_experts={num_experts}, emb_dim={emb_dim}")
    logger.info(f"{'='*80}")

    # Create inputs with incremental pattern
    combine_output = create_incremental_activations(seq_len, num_experts, emb_dim, start_val=0.0, increment=0.1)
    weights = create_simple_weights(seq_len, num_experts)

    # Print what we're testing
    logger.info("\nInput patterns:")
    for expert_idx in range(num_experts):
        first_val = combine_output[0, 0, 0, expert_idx, 0].item()
        last_val = combine_output[0, 0, 0, expert_idx, -1].item()
        weight_val = weights[0, 0, 0, expert_idx].item()
        logger.info(f"  Expert {expert_idx}: values [{first_val:.1f} ... {last_val:.1f}], weight={weight_val:.1f}")

    # Compute expected output
    expected = compute_expected_output(combine_output, weights)

    logger.info(f"\nExpected output shape: {expected.shape}")
    logger.info(f"Expected output first 10 values: {expected[0, 0, 0, 0:10]}")
    logger.info(f"Expected output last 10 values: {expected[0, 0, 0, -10:]}")

    # Convert to TTNN tensors (ROW_MAJOR)
    tt_combine = ttnn.from_torch(
        combine_output,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_weights = ttnn.from_torch(
        weights,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info("\nInput tensor shapes:")
    logger.info(f"  combine_output: {tt_combine.shape}, layout={tt_combine.layout}")
    logger.info(f"  weights: {tt_weights.shape}, layout={tt_weights.layout}")

    # Run the operation
    logger.info("\nRunning ttnn.experimental.deepseek_moe_post_combine_reduce...")
    tt_output = ttnn.experimental.deepseek_moe_post_combine_reduce(
        tt_combine,
        tt_weights,
    )

    # Convert back to torch
    output = ttnn.to_torch(tt_output)

    logger.info(f"\nOutput tensor shape: {output.shape}")
    logger.info(f"Output layout: {tt_output.layout}")

    # Check first few values
    logger.info("\nActual output first 10 values:")
    logger.info(f"  output[0,0,0,0:10] = {output[0,0,0,0:10]}")

    logger.info("\nActual output last 10 values:")
    logger.info(f"  output[0,0,0,-10:] = {output[0,0,0,-10:]}")

    # Manual calculation for first element to verify
    logger.info("\nManual verification for output[0]:")
    manual_sum = 0.0
    for expert_idx in range(num_experts):
        activation_val = combine_output[0, 0, 0, expert_idx, 0].item()
        weight_val = weights[0, 0, 0, expert_idx].item()
        contribution = activation_val * weight_val
        manual_sum += contribution
        logger.info(f"  Expert {expert_idx}: {activation_val:.1f} × {weight_val:.1f} = {contribution:.1f}")
    logger.info(f"  Manual sum: {manual_sum:.1f}")
    logger.info(f"  Expected[0]: {expected[0, 0, 0, 0].item():.1f}")
    logger.info(f"  Actual[0]:   {output[0, 0, 0, 0].item():.1f}")

    # Output statistics
    output_min = output.min().item()
    output_max = output.max().item()
    output_mean = output.mean().item()
    expected_min = expected.min().item()
    expected_max = expected.max().item()

    logger.info(f"\nOutput statistics:")
    logger.info(f"  Actual:   min={output_min:.3f}, max={output_max:.3f}, mean={output_mean:.3f}")
    logger.info(f"  Expected: min={expected_min:.3f}, max={expected_max:.3f}")

    # Compare with expected
    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    logger.info(f"\nPCC: {pcc:.6f}")

    if passing:
        logger.info("✓ Test PASSED!")
    else:
        logger.error("✗ Test FAILED!")
        logger.error(f"Expected vs Actual difference: {abs(expected_val - output_mean):.6f}")

    assert passing, f"PCC {pcc} is below threshold 0.99"


@pytest.mark.parametrize("seq_len", [1, 8])
@pytest.mark.parametrize("num_experts", [2, 4, 8])
def test_known_values_parameterized(device, seq_len, num_experts):
    """
    Parameterized test with different sequence lengths and expert counts.
    """
    emb_dim = 1024  # Use smaller size for faster testing

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing seq_len={seq_len}, num_experts={num_experts}, emb_dim={emb_dim}")
    logger.info(f"{'='*80}")

    # Create inputs with incremental pattern
    combine_output = create_incremental_activations(seq_len, num_experts, emb_dim, start_val=0.0, increment=0.1)
    weights = create_simple_weights(seq_len, num_experts)

    # Expected output
    expected = compute_expected_output(combine_output, weights)

    # Convert to TTNN
    tt_combine = ttnn.from_torch(
        combine_output,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_weights = ttnn.from_torch(
        weights,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run operation
    tt_output = ttnn.experimental.deepseek_moe_post_combine_reduce(tt_combine, tt_weights)
    output = ttnn.to_torch(tt_output)

    # Verify
    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    logger.info(f"PCC: {pcc:.6f} - {'PASS' if passing else 'FAIL'}")

    assert passing, f"PCC {pcc} is below threshold 0.99"


def test_known_values_per_expert_tracking(device):
    """
    Test that tracks contribution of each expert individually.

    This helps debug if specific experts are processed incorrectly.
    """
    seq_len = 1
    num_experts = 8
    emb_dim = 7168

    logger.info(f"\n{'='*80}")
    logger.info(f"PER-EXPERT TRACKING TEST")
    logger.info(f"{'='*80}")

    # Create data where each expert has a unique value
    combine_output = torch.zeros(1, 1, seq_len, num_experts, emb_dim, dtype=torch.bfloat16)

    # Use powers of 2 for easy identification
    for expert_idx in range(num_experts):
        combine_output[:, :, :, expert_idx, :] = 2.0**expert_idx

    # Test with one expert at a time
    for active_expert in range(num_experts):
        logger.info(f"\n--- Testing with ONLY Expert {active_expert} active ---")

        # Create weights where only one expert is active
        weights = torch.zeros(1, 1, seq_len, num_experts, dtype=torch.bfloat16)
        weights[:, :, :, active_expert] = 1.0

        # Expected: only the active expert's value
        expected_val = 2.0**active_expert

        # Convert to TTNN
        tt_combine = ttnn.from_torch(
            combine_output,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_weights = ttnn.from_torch(
            weights,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run operation
        tt_output = ttnn.experimental.deepseek_moe_post_combine_reduce(tt_combine, tt_weights)
        output = ttnn.to_torch(tt_output)

        # Check output
        actual_val = output[0, 0, 0, 0].item()
        logger.info(f"  Expected: {expected_val:.1f}, Got: {actual_val:.6f}")

        # Allow some tolerance due to bfloat16
        diff = abs(expected_val - actual_val)
        if diff < 0.1:
            logger.info(f"  ✓ Expert {active_expert} correct!")
        else:
            logger.error(f"  ✗ Expert {active_expert} WRONG! Difference: {diff:.6f}")
            pytest.fail(f"Expert {active_expert} produced wrong output")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        test_known_values_simple(device)
        test_known_values_per_expert_tracking(device)
    finally:
        ttnn.close_device(device)
