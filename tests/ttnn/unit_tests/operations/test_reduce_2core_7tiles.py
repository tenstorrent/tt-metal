# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test deepseek_moe_post_combine_reduce with hardware tilization.

Configuration:
- Requires exactly 32 tokens per core
- Output is TILE layout (hardware tilized)
- Shape: [1, 3200, 8, 7168] - 3200 tokens, 8 experts, 7 tiles
- 100 cores: each processes 32 tokens
- Simplified tile values: tile N has all elements = N
- Weights: sequential [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
"""

import pytest
import torch
import ttnn
from loguru import logger


def create_simple_tile_data(num_tokens, num_experts, emb_dim, tile_start=1.0, tile_increment=1.0):
    """
    Create data where each tile has a constant value.

    Pattern:
    - Token 0, Expert 0, Tile 0: all values = tile_start
    - Token 0, Expert 0, Tile 1: all values = tile_start + tile_increment
    - ...
    - Token 0, Expert 0, Tile 6: all values = tile_start + 6*tile_increment
    - Token 0, Expert 1, Tile 0: all values = tile_start + 7*tile_increment
    - ...

    Returns tensor of shape [1, num_tokens, num_experts, emb_dim]
    """
    assert emb_dim % 1024 == 0, "emb_dim must be divisible by 1024"
    num_tiles = emb_dim // 1024

    data = torch.zeros(1, num_tokens, num_experts, emb_dim, dtype=torch.bfloat16)

    tile_value = tile_start
    for token_idx in range(num_tokens):
        for expert_idx in range(num_experts):
            for tile_idx in range(num_tiles):
                start = tile_idx * 1024
                end = start + 1024
                data[0, token_idx, expert_idx, start:end] = tile_value
                tile_value += tile_increment

    return data


def create_simple_weights(num_tokens, num_experts):
    """
    Create weights: [1.0, 2.0, 3.0, ..., 8.0] for each token.

    Returns tensor of shape [1, num_tokens, num_experts, 1]
    """
    weights = torch.zeros(1, num_tokens, num_experts, 1, dtype=torch.bfloat16)

    for token_idx in range(num_tokens):
        for expert_idx in range(num_experts):
            weights[0, token_idx, expert_idx, 0] = expert_idx + 1.0

    return weights


def compute_expected_output(combine_output, weights):
    """
    Compute expected output: weighted sum across experts.

    output[token, emb] = sum over experts of (combine_output[token, expert, emb] * weights[token, expert])
    """
    # Expand weights: [1, num_tokens, num_experts, 1] -> [1, num_tokens, num_experts, emb_dim]
    weights_expanded = weights.expand(-1, -1, -1, combine_output.shape[-1])

    # Element-wise multiply
    weighted = combine_output * weights_expanded

    # Sum across expert dimension (dim=2)
    result = weighted.sum(dim=2)

    return result


def verify_with_pcc(result, expected, pcc_threshold=0.9999, max_rel_error_threshold=1.0):
    """
    Verify result using PCC (Pearson Correlation Coefficient) metric.

    Returns True if verification passes, False otherwise.
    Also prints diagnostic information.
    """
    import numpy as np

    # Flatten and convert to numpy
    expected_flat = expected.flatten().float().numpy()
    result_flat = result.flatten().float().numpy()

    # Check for NaN/Inf
    has_nan = np.isnan(result_flat).any()
    has_inf = np.isinf(result_flat).any()

    if has_nan or has_inf:
        print(f"❌ Result contains NaN: {has_nan}, Inf: {has_inf}")
        return False

    # Compute PCC
    mean_exp = np.mean(expected_flat)
    mean_res = np.mean(result_flat)
    numerator = np.sum((expected_flat - mean_exp) * (result_flat - mean_res))
    denominator = np.sqrt(np.sum((expected_flat - mean_exp) ** 2) * np.sum((result_flat - mean_res) ** 2))
    pcc = numerator / denominator if denominator > 0 else 0.0

    # Compute relative error
    relative_errors = np.abs(expected_flat - result_flat) / (np.abs(expected_flat) + 1e-5)
    mean_rel_error = np.mean(relative_errors) * 100
    max_rel_error = np.max(relative_errors) * 100

    print(f"PCC: {pcc:.6f} (threshold: {pcc_threshold})")
    print(f"Mean relative error: {mean_rel_error:.3f}% (threshold: {max_rel_error_threshold}%)")
    print(f"Max relative error: {max_rel_error:.3f}%")

    # Check thresholds
    pcc_pass = pcc > pcc_threshold
    error_pass = mean_rel_error < max_rel_error_threshold

    if pcc_pass and error_pass:
        print("✅ Verification PASSED")
        return True
    else:
        print(f"❌ Verification FAILED: PCC={pcc_pass}, Error={error_pass}")
        return False


def test_reduce_full_scale_3200tokens(device):
    """
    Test with full-scale realistic dimensions using hardware tilization:
    - 3200 tokens
    - 8 experts
    - 7 tiles (7168 embedding)
    - 100 cores: each processes exactly 32 tokens
    - Output is TILE layout (hardware tilized)

    Uses PCC metric for verification.
    Verifies that output layout is TILE.
    """
    num_tokens = 3200
    num_experts = 8
    emb_dim = 7168
    num_tiles = emb_dim // 1024

    logger.info("=" * 80)
    logger.info(f"Test: Full Scale with Hardware Tilization")
    logger.info(f"  {num_tokens} tokens, {num_experts} experts, {num_tiles} tiles")
    logger.info(f"  Shape: [1, {num_tokens}, {num_experts}, {emb_dim}]")
    logger.info(f"  100 cores: each processes exactly 32 tokens")
    logger.info(f"  Using smaller values to avoid overflow: tile_start=0.1, tile_increment=0.1")
    logger.info("=" * 80)

    # Create inputs with simple tile pattern using smaller values to avoid overflow
    combine_output = create_simple_tile_data(num_tokens, num_experts, emb_dim, tile_start=0.1, tile_increment=0.1)
    weights = create_simple_weights(num_tokens, num_experts)

    # Compute expected output
    expected = compute_expected_output(combine_output, weights)

    logger.info("\nConverting to TTNN tensors...")
    combine_output_tt = ttnn.from_torch(
        combine_output,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    weights_tt = ttnn.from_torch(
        weights,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"  combine_output: {combine_output_tt.shape}")
    logger.info(f"  weights: {weights_tt.shape}")

    # Run the operation
    logger.info("\nRunning operation with hardware tilization...")
    result_tt = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_output_tt, weights_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"OUTPUT LAYOUT VERIFICATION:")
    logger.info(f"  Output shape: {result_tt.shape}")
    logger.info(f"  Output layout: {result_tt.layout}")

    # Verify output is TILE layout
    assert result_tt.layout == ttnn.TILE_LAYOUT, f"❌ Expected TILE_LAYOUT but got {result_tt.layout}"
    logger.info(f"  ✅ Output layout is TILE_LAYOUT (hardware tilized)")
    logger.info(f"{'='*80}")

    result = ttnn.to_torch(result_tt)

    # Check for NaN/Inf
    nan_mask = torch.isnan(result[0, :, 0])
    inf_mask = torch.isinf(result[0, :, 0])
    nan_tokens = torch.where(nan_mask)[0]
    inf_tokens = torch.where(inf_mask)[0]

    # Calculate core distribution
    tokens_per_core = 32  # Hardware tilization requires exactly 32 tokens per core
    num_cores_used = num_tokens // tokens_per_core

    print(f"\n{'='*80}")
    print(f"CORE DISTRIBUTION:")
    print(f"  Total cores used: {num_cores_used}")
    print(f"  Tokens per core: {tokens_per_core} (fixed for hardware tilization)")
    print(f"  Total tokens: {num_tokens}")
    print(f"{'='*80}")

    if len(nan_tokens) > 0 or len(inf_tokens) > 0:
        print(f"\n❌ NaN/Inf detected:")
        print(f"  Tokens with NaN (total {len(nan_tokens)}): {nan_tokens.tolist()}")
        print(f"  Tokens with Inf (total {len(inf_tokens)}): {inf_tokens.tolist()}")

        for token_idx in nan_tokens[:10].tolist():
            # Find which core handles this token
            core_idx = token_idx // tokens_per_core
            token_in_core = token_idx % tokens_per_core
            print(f"  Token {token_idx} -> Core {core_idx} (token {token_in_core} within core)")

        # Check PCC for non-NaN tokens
        print(f"\n✓ Checking PCC for non-NaN tokens:")
        good_mask = ~nan_mask & ~inf_mask
        if good_mask.any():
            good_result = result[0, good_mask, :]
            good_expected = expected[0, good_mask, :]

            import numpy as np

            result_flat = good_result.flatten().float().numpy()
            expected_flat = good_expected.flatten().float().numpy()

            mean_exp = np.mean(expected_flat)
            mean_res = np.mean(result_flat)
            numerator = np.sum((expected_flat - mean_exp) * (result_flat - mean_res))
            denominator = np.sqrt(np.sum((expected_flat - mean_exp) ** 2) * np.sum((result_flat - mean_res) ** 2))
            pcc = numerator / denominator if denominator > 0 else 0.0

            relative_errors = np.abs(expected_flat - result_flat) / (np.abs(expected_flat) + 1e-5)
            mean_rel_error = np.mean(relative_errors) * 100

            print(f"  Non-NaN tokens: {good_mask.sum().item()} / {num_tokens}")
            print(f"  PCC (non-NaN only): {pcc:.6f}")
            print(f"  Mean relative error (non-NaN only): {mean_rel_error:.3f}%")
    else:
        print(f"\n✓ No NaN/Inf detected")

    logger.info("\nVerification:")
    passed = verify_with_pcc(result, expected, pcc_threshold=0.9999, max_rel_error_threshold=2.0)

    if not passed:
        pytest.fail("Verification failed - see output above")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        test_reduce_full_scale_3200tokens(device)
    finally:
        ttnn.close_device(device)
