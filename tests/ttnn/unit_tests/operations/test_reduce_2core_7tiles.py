# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test deepseek_moe_post_combine_reduce with 2 cores and 7 tiles.

Configuration:
- Shape: [1, 2, 8, 7168] - 2 tokens, 8 experts, 7 tiles
- 2 cores: each processes 1 token
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


def test_reduce_2core_7tiles_simple(device):
    """
    Test with 2 tokens, 8 experts, 7 tiles (7168 embedding).

    Uses 2 cores, each processing 1 token.
    """
    num_tokens = 2
    num_experts = 8
    emb_dim = 7168
    num_tiles = emb_dim // 1024

    logger.info("=" * 80)
    logger.info(f"Test: 2 Cores, 7 Tiles")
    logger.info(f"Shape: [1, {num_tokens}, {num_experts}, {emb_dim}]")
    logger.info(f"Cores: 2 (each processes 1 token)")
    logger.info("=" * 80)

    # Create inputs
    combine_output = create_simple_tile_data(num_tokens, num_experts, emb_dim)
    weights = create_simple_weights(num_tokens, num_experts)

    logger.info("\nInput pattern (Token 0):")
    for expert_idx in range(num_experts):
        tile_values = []
        for tile_idx in range(num_tiles):
            start = tile_idx * 1024
            val = combine_output[0, 0, expert_idx, start].item()
            tile_values.append(f"{val:.0f}")
        logger.info(f"  Expert {expert_idx}: tiles = [{', '.join(tile_values)}]")

    logger.info("\nWeights (Token 0):")
    weight_vals = [weights[0, 0, e, 0].item() for e in range(num_experts)]
    logger.info(f"  {weight_vals}")

    # Compute expected output
    expected = compute_expected_output(combine_output, weights)

    # logger.info("\nExpected output (Token 0):")
    for tile_idx in range(num_tiles):
        start = tile_idx * 1024
        val = expected[0, 0, start].item()
        logger.info(f"  Tile {tile_idx}: {val:.0f}")

    # Manual verification for Token 0, Tile 0:
    # = 1×1 + 8×2 + 15×3 + 22×4 + 29×5 + 36×6 + 43×7 + 50×8
    # = 1 + 16 + 45 + 88 + 145 + 216 + 301 + 400 = 1212
    manual_tile0 = sum((expert_idx * 7 + 1) * (expert_idx + 1) for expert_idx in range(num_experts))
    # logger.info(f"\nManual calculation for Token 0, Tile 0: {manual_tile0}")
    # logger.info(f"Expected from formula: {expected[0, 0, 0].item():.0f}")

    # Convert to TTNN
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

    logger.info(f"  combine_output: {combine_output_tt.shape}, {combine_output_tt.layout}")
    logger.info(f"  weights: {weights_tt.shape}, {weights_tt.layout}")

    # Run the operation
    logger.info("\nRunning ttnn.experimental.deepseek_moe_post_combine_reduce...")

    try:
        result_tt = ttnn.experimental.deepseek_moe_post_combine_reduce(
            combine_output_tt,
            weights_tt,
            expert_dim=2,  # Expert dimension is at index 2
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        logger.info(f"✅ Operation completed!")
        logger.info(f"  Output shape: {result_tt.shape}")
        logger.info(f"  Output layout: {result_tt.layout}")

        # Convert back to PyTorch
        result = ttnn.to_torch(result_tt)

        logger.info("\n" + "=" * 80)
        logger.info("VERIFICATION:")
        logger.info("=" * 80)

        # Verify using PCC
        passed = verify_with_pcc(result, expected, pcc_threshold=0.9999, max_rel_error_threshold=2.0)

        if not passed:
            pytest.fail("Verification failed - see output above")

    except Exception as e:
        logger.error(f"\n❌ Operation failed with error:")
        logger.error(f"   {type(e).__name__}: {e}")
        raise


def test_reduce_single_token_single_expert(device):
    """
    Simplest test: 1 token, 1 expert, 7 tiles.

    This validates basic mechanics without accumulation.
    """
    num_tokens = 1
    num_experts = 1
    emb_dim = 7168

    logger.info("\n" + "=" * 80)
    logger.info(f"SIMPLE TEST: 1 Token, 1 Expert, 7 Tiles")
    logger.info("=" * 80)

    # Create simple data
    combine_output = create_simple_tile_data(num_tokens, num_experts, emb_dim)
    weights = torch.ones(1, num_tokens, num_experts, 1, dtype=torch.bfloat16) * 2.0  # Weight = 2.0

    logger.info(f"\nInput tiles: [1, 2, 3, 4, 5, 6, 7]")
    logger.info(f"Weight: 2.0")
    logger.info(f"Expected output: [2, 4, 6, 8, 10, 12, 14]")

    # Expected: each tile multiplied by 2 (after reducing expert dim)
    # Output shape will be [1, 1, 7168] after reducing expert dimension
    expected = (combine_output * 2.0).squeeze(2)  # Remove expert dim

    # Convert to TTNN
    combine_output_tt = ttnn.from_torch(
        combine_output, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weights_tt = ttnn.from_torch(
        weights, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Run operation
    result_tt = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_output_tt, weights_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    result = ttnn.to_torch(result_tt)

    # Verify using PCC
    logger.info("\nVerification:")
    passed = verify_with_pcc(result, expected, pcc_threshold=0.9999, max_rel_error_threshold=2.0)

    if not passed:
        pytest.fail("Verification failed - see output above")


def test_reduce_2experts_2tiles(device):
    """
    Minimal multi-expert test: 2 tokens, 2 experts, 2 tiles (2048 emb_dim).

    This validates packer L1 accumulation with 2 cores (one per token).

    Input pattern (Token 0):
    - Expert 0: tiles = [1, 2]
    - Expert 1: tiles = [3, 4]
    - Weights: [1.0, 2.0]

    Input pattern (Token 1):
    - Expert 0: tiles = [5, 6]
    - Expert 1: tiles = [7, 8]
    - Weights: [3.0, 4.0]

    Expected output Token 0:
    - Tile 0: 1*1 + 3*2 = 1 + 6 = 7
    - Tile 1: 2*1 + 4*2 = 2 + 8 = 10

    Expected output Token 1:
    - Tile 0: 5*3 + 7*4 = 15 + 28 = 43
    - Tile 1: 6*3 + 8*4 = 18 + 32 = 50
    """
    num_tokens = 2
    num_experts = 2
    emb_dim = 2048  # 2 tiles
    num_tiles = emb_dim // 1024

    logger.info("\n" + "=" * 80)
    logger.info(f"MINIMAL MULTI-EXPERT TEST: 2 Tokens, 2 Experts, 2 Tiles")
    logger.info("=" * 80)

    # Create simple data
    combine_output = create_simple_tile_data(num_tokens, num_experts, emb_dim)

    # Weights: Token 0: [1.0, 2.0], Token 1: [3.0, 4.0]
    weights = torch.zeros(1, num_tokens, num_experts, 1, dtype=torch.bfloat16)
    weights[0, 0, 0, 0] = 1.0
    weights[0, 0, 1, 0] = 2.0
    weights[0, 1, 0, 0] = 3.0
    weights[0, 1, 1, 0] = 4.0

    logger.info("\nInput pattern (Token 0):")
    logger.info("  Expert 0: tiles = [1, 2]")
    logger.info("  Expert 1: tiles = [3, 4]")
    logger.info("  Weights: [1.0, 2.0]")
    logger.info("\nInput pattern (Token 1):")
    logger.info("  Expert 0: tiles = [5, 6]")
    logger.info("  Expert 1: tiles = [7, 8]")
    logger.info("  Weights: [3.0, 4.0]")
    logger.info("\nExpected output Token 0: [7, 10]")
    logger.info("Expected output Token 1: [43, 50]")

    # Expected: weighted sum across experts
    expected = compute_expected_output(combine_output, weights)

    logger.info(f"\nComputed expected output:")
    for token_idx in range(num_tokens):
        logger.info(f"  Token {token_idx}:")
        for tile_idx in range(num_tiles):
            start = tile_idx * 1024
            val = expected[0, token_idx, start].item()
            logger.info(f"    Tile {tile_idx}: {val:.0f}")

    # Convert to TTNN
    combine_output_tt = ttnn.from_torch(
        combine_output, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weights_tt = ttnn.from_torch(
        weights, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Run operation
    logger.info("\nRunning operation...")
    result_tt = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_output_tt, weights_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    result = ttnn.to_torch(result_tt)

    # Verify using PCC
    logger.info("\nVerification:")
    passed = verify_with_pcc(result, expected, pcc_threshold=0.9999, max_rel_error_threshold=2.0)

    if not passed:
        pytest.fail("Verification failed - see output above")


def test_reduce_3experts_3tiles_3tokens(device):
    """
    Test: 3 tokens, 3 experts, 3 tiles (3072 emb_dim).
    Uses 3 cores (one per token).

    Input pattern (Token 0):
    - Expert 0: tiles = [1, 2, 3]
    - Expert 1: tiles = [4, 5, 6]
    - Expert 2: tiles = [7, 8, 9]
    - Weights: [1.0, 2.0, 3.0]

    Input pattern (Token 1):
    - Expert 0: tiles = [10, 11, 12]
    - Expert 1: tiles = [13, 14, 15]
    - Expert 2: tiles = [16, 17, 18]
    - Weights: [4.0, 5.0, 6.0]

    Input pattern (Token 2):
    - Expert 0: tiles = [19, 20, 21]
    - Expert 1: tiles = [22, 23, 24]
    - Expert 2: tiles = [25, 26, 27]
    - Weights: [7.0, 8.0, 9.0]

    Expected output Token 0:
    - Tile 0: 1*1 + 4*2 + 7*3 = 30
    - Tile 1: 2*1 + 5*2 + 8*3 = 36
    - Tile 2: 3*1 + 6*2 + 9*3 = 42

    Expected output Token 1:
    - Tile 0: 10*4 + 13*5 + 16*6 = 201
    - Tile 1: 11*4 + 14*5 + 17*6 = 216
    - Tile 2: 12*4 + 15*5 + 18*6 = 231

    Expected output Token 2:
    - Tile 0: 19*7 + 22*8 + 25*9 = 534
    - Tile 1: 20*7 + 23*8 + 26*9 = 558
    - Tile 2: 21*7 + 24*8 + 27*9 = 582
    """
    num_tokens = 3
    num_experts = 3
    emb_dim = 3072  # 3 tiles
    num_tiles = emb_dim // 1024

    # logger.info("\n" + "=" * 80)
    # logger.info(f"TEST: 3 Tokens, 3 Experts, 3 Tiles")
    # logger.info("=" * 80)

    # Create simple data
    combine_output = create_simple_tile_data(num_tokens, num_experts, emb_dim)

    # Weights: Token 0: [1, 2, 3], Token 1: [4, 5, 6], Token 2: [7, 8, 9]
    weights = torch.zeros(1, num_tokens, num_experts, 1, dtype=torch.bfloat16)
    weights[0, 0, 0, 0] = 1.0
    weights[0, 0, 1, 0] = 2.0
    weights[0, 0, 2, 0] = 3.0
    weights[0, 1, 0, 0] = 4.0
    weights[0, 1, 1, 0] = 5.0
    weights[0, 1, 2, 0] = 6.0
    weights[0, 2, 0, 0] = 7.0
    weights[0, 2, 1, 0] = 8.0
    weights[0, 2, 2, 0] = 9.0

    # logger.info("\nInput pattern (Token 0):")
    # logger.info("  Expert 0: tiles = [1, 2, 3], Weight: 1.0")
    # logger.info("  Expert 1: tiles = [4, 5, 6], Weight: 2.0")
    # logger.info("  Expert 2: tiles = [7, 8, 9], Weight: 3.0")
    # logger.info("  Expected: [30, 36, 42]")

    # logger.info("\nInput pattern (Token 1):")
    # logger.info("  Expert 0: tiles = [10, 11, 12], Weight: 4.0")
    # logger.info("  Expert 1: tiles = [13, 14, 15], Weight: 5.0")
    # logger.info("  Expert 2: tiles = [16, 17, 18], Weight: 6.0")
    # logger.info("  Expected: [201, 216, 231]")

    # logger.info("\nInput pattern (Token 2):")
    # logger.info("  Expert 0: tiles = [19, 20, 21], Weight: 7.0")
    # logger.info("  Expert 1: tiles = [22, 23, 24], Weight: 8.0")
    # logger.info("  Expert 2: tiles = [25, 26, 27], Weight: 9.0")
    # logger.info("  Expected: [534, 558, 582]")

    # Expected: weighted sum across experts
    expected = compute_expected_output(combine_output, weights)

    # logger.info(f"\nComputed expected output:")
    for token_idx in range(num_tokens):
        # logger.info(f"  Token {token_idx}:")
        for tile_idx in range(num_tiles):
            start = tile_idx * 1024
            val = expected[0, token_idx, start].item()
            # logger.info(f"    Tile {tile_idx}: {val:.0f}")

    # Convert to TTNN
    combine_output_tt = ttnn.from_torch(
        combine_output, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weights_tt = ttnn.from_torch(
        weights, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Run operation
    # logger.info("\nRunning operation...")
    result_tt = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_output_tt, weights_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    result = ttnn.to_torch(result_tt)

    # Verify using PCC
    logger.info("\nVerification:")
    passed = verify_with_pcc(result, expected, pcc_threshold=0.9999, max_rel_error_threshold=2.0)

    if not passed:
        pytest.fail("Verification failed - see output above")


def test_reduce_4experts_4tiles_4tokens(device):
    """
    Test: 4 tokens, 4 experts, 4 tiles (4096 emb_dim).
    Uses 4 cores (one per token).
    """
    num_tokens = 4
    num_experts = 4
    emb_dim = 4096  # 4 tiles
    num_tiles = emb_dim // 1024

    # Create simple data
    combine_output = create_simple_tile_data(num_tokens, num_experts, emb_dim)

    # Weights: Token i has weights [4*i+1, 4*i+2, 4*i+3, 4*i+4]
    weights = torch.zeros(1, num_tokens, num_experts, 1, dtype=torch.bfloat16)
    for token_idx in range(num_tokens):
        for expert_idx in range(num_experts):
            weights[0, token_idx, expert_idx, 0] = float(token_idx * num_experts + expert_idx + 1)

    # Expected: weighted sum across experts
    expected = compute_expected_output(combine_output, weights)

    # Convert to TTNN
    combine_output_tt = ttnn.from_torch(
        combine_output, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weights_tt = ttnn.from_torch(
        weights, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Verify data after conversion to TTNN
    combine_output_readback = ttnn.to_torch(combine_output_tt)
    weights_readback = ttnn.to_torch(weights_tt)

    # Run operation
    result_tt = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_output_tt, weights_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    result = ttnn.to_torch(result_tt)

    # Verify using PCC
    logger.info("\nVerification:")
    passed = verify_with_pcc(result, expected, pcc_threshold=0.9999, max_rel_error_threshold=2.0)

    if not passed:
        pytest.fail("Verification failed - see output above")


def test_reduce_full_scale_3200tokens(device):
    """
    Test with full-scale realistic dimensions:
    - 3200 tokens
    - 8 experts
    - 7 tiles (7168 embedding)

    Uses PCC metric instead of exact matching.
    Verifies token distribution across cores.
    """
    num_tokens = 3200
    num_experts = 8
    emb_dim = 7168
    num_tiles = emb_dim // 1024

    logger.info("=" * 80)
    logger.info(f"Test: Full Scale - {num_tokens} tokens, {num_experts} experts, {num_tiles} tiles")
    logger.info(f"Shape: [1, {num_tokens}, {num_experts}, {emb_dim}]")
    logger.info("Using smaller values to avoid overflow: tile_start=0.1, tile_increment=0.1")
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
    logger.info("\nRunning operation...")
    result_tt = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_output_tt, weights_tt, expert_dim=2, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    result = ttnn.to_torch(result_tt)

    # Check for NaN/Inf and identify which tokens are affected
    nan_mask = torch.isnan(result[0, :, 0])
    inf_mask = torch.isinf(result[0, :, 0])
    nan_tokens = torch.where(nan_mask)[0]
    inf_tokens = torch.where(inf_mask)[0]

    # Calculate core distribution
    num_cores = 88  # Blackhole has 88 cores
    tokens_per_core = num_tokens // num_cores
    extra_tokens = num_tokens % num_cores

    print(f"\n{'='*80}")
    print(f"CORE DISTRIBUTION:")
    print(f"  Total cores available: 88 (11x8 grid)")
    print(f"  Total tokens: {num_tokens}")
    print(f"  Tokens per core: {tokens_per_core}")
    print(f"  Extra tokens (first N cores): {extra_tokens}")
    print(f"  First {extra_tokens} cores get {tokens_per_core + 1} tokens")
    print(f"  Remaining {88 - extra_tokens} cores get {tokens_per_core} tokens")
    print(f"{'='*80}")

    if len(nan_tokens) > 0 or len(inf_tokens) > 0:
        print(f"\n❌ NaN/Inf detected:")
        print(f"  Tokens with NaN (total {len(nan_tokens)}): {nan_tokens.tolist()}")
        print(f"  Tokens with Inf (total {len(inf_tokens)}): {inf_tokens.tolist()}")

        for token_idx in nan_tokens[:10].tolist():
            # Find which core handles this token
            tokens_before = 0
            for core_idx in range(num_cores):
                tokens_for_core = tokens_per_core + (1 if core_idx < extra_tokens else 0)
                if tokens_before <= token_idx < tokens_before + tokens_for_core:
                    print(
                        f"  Token {token_idx} -> Core {core_idx} (handles tokens {tokens_before}-{tokens_before + tokens_for_core - 1})"
                    )
                    break
                tokens_before += tokens_for_core

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

            # Sample some values from non-NaN tokens
            print(f"\n  Sample values from non-NaN tokens:")
            sample_tokens = [0, 10, 100, 200, 300, 400, 495, 504, 1000, 2000, 3000, 3199]
            for token_idx in sample_tokens:
                if token_idx >= num_tokens:
                    continue
                exp_val = expected[0, token_idx, 0].item()
                act_val = result[0, token_idx, 0].item()
                is_nan = torch.isnan(result[0, token_idx, 0]).item()
                status = "NaN" if is_nan else f"expected={exp_val:.4f}, actual={act_val:.4f}"
                print(f"    Token {token_idx}: {status}")
    else:
        print(f"\n✓ No NaN/Inf detected")

    logger.info("\nVerification:")
    passed = verify_with_pcc(result, expected, pcc_threshold=0.9999, max_rel_error_threshold=2.0)

    if not passed:
        pytest.fail("Verification failed - see output above")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        # Run simple test first
        test_reduce_single_token_single_expert(device)

        # Then run 2-core test
        test_reduce_2core_7tiles_simple(device)

    finally:
        ttnn.close_device(device)
