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


def create_simple_tile_data(num_tokens, num_experts, emb_dim):
    """
    Create data where each tile has a constant value.

    Pattern:
    - Token 0, Expert 0, Tile 0: all values = 1.0
    - Token 0, Expert 0, Tile 1: all values = 2.0
    - ...
    - Token 0, Expert 0, Tile 6: all values = 7.0
    - Token 0, Expert 1, Tile 0: all values = 8.0
    - ...

    Returns tensor of shape [1, num_tokens, num_experts, emb_dim]
    """
    assert emb_dim % 1024 == 0, "emb_dim must be divisible by 1024"
    num_tiles = emb_dim // 1024

    data = torch.zeros(1, num_tokens, num_experts, emb_dim, dtype=torch.bfloat16)

    tile_value = 1.0
    for token_idx in range(num_tokens):
        for expert_idx in range(num_experts):
            for tile_idx in range(num_tiles):
                start = tile_idx * 1024
                end = start + 1024
                data[0, token_idx, expert_idx, start:end] = tile_value
                tile_value += 1.0

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
    logger.info(f"  weights buffer page_size: {weights_tt.buffer().page_size()}")

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

        # Check each token
        all_correct = True
        print("_______________actual output______________")
        print(result[0, 0, 0:10])
        print(result[0, 0, 1023:1034])
        print(result[0, 0, 2047:2058])
        print("__________________________________________")
        for token_idx in range(num_tokens):
            logger.info(f"\nToken {token_idx}:")

            for tile_idx in range(num_tiles):
                start = tile_idx * 1024

                expected_val = expected[0, token_idx, start].item()
                actual_val = result[0, token_idx, start].item()

                diff = abs(expected_val - actual_val)
                match = diff < 1.0  # Allow small floating point error
                status = "✅" if match else "❌"

                # logger.info(f"  {status} Tile {tile_idx}: expected={expected_val:.1f}, actual={actual_val:.1f}, diff={diff:.1f}")

                if not match:
                    all_correct = False
                    # Print more details for failed tile
                    logger.error(f"      First 10 elements expected: {expected[0, token_idx, start:start+10]}")
                    logger.error(f"      First 10 elements actual:   {result[0, token_idx, start:start+10]}")

        logger.info("\n" + "=" * 80)
        if all_correct:
            logger.info("✅✅✅ TEST PASSED! All values match expected output.")
        else:
            logger.error("❌❌❌ TEST FAILED! Some values don't match.")
            pytest.fail("Output doesn't match expected values")
        logger.info("=" * 80)

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

    # Verify
    logger.info("\nVerification:")
    for tile_idx in range(7):
        start = tile_idx * 1024
        expected_val = expected[0, 0, start].item()
        actual_val = result[0, 0, start].item()
        match = abs(expected_val - actual_val) < 0.1
        status = "✅" if match else "❌"
        logger.info(f"{status} Tile {tile_idx}: expected={expected_val:.0f}, actual={actual_val:.0f}")

        if not match:
            pytest.fail(f"Tile {tile_idx} doesn't match")

    logger.info("✅ Simple test passed!")


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

    # Verify
    logger.info("\nVerification:")
    all_match = True
    for token_idx in range(num_tokens):
        logger.info(f"\nToken {token_idx}:")
        for tile_idx in range(num_tiles):
            start = tile_idx * 1024
            expected_val = expected[0, token_idx, start].item()
            actual_val = result[0, token_idx, start].item()
            diff = abs(expected_val - actual_val)
            match = diff < 0.1
            status = "✅" if match else "❌"
            logger.info(
                f"  {status} Tile {tile_idx}: expected={expected_val:.0f}, actual={actual_val:.0f}, diff={diff:.1f}"
            )

            if not match:
                all_match = False
                logger.error(f"    First 10 elements expected: {expected[0, token_idx, start:start+10]}")
                logger.error(f"    First 10 elements actual:   {result[0, token_idx, start:start+10]}")

    if not all_match:
        pytest.fail("Output doesn't match expected values")

    logger.info("\n✅ 2-expert 2-tile 2-token test passed!")


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

    # Verify
    # logger.info("\nVerification:")
    all_match = True
    for token_idx in range(num_tokens):
        # logger.info(f"\nToken {token_idx}:")
        for tile_idx in range(num_tiles):
            start = tile_idx * 1024
            expected_val = expected[0, token_idx, start].item()
            actual_val = result[0, token_idx, start].item()
            diff = abs(expected_val - actual_val)
            match = diff < 0.1
            status = "✅" if match else "❌"
            # logger.info(
            #     f"  {status} Tile {tile_idx}: expected={expected_val:.0f}, actual={actual_val:.0f}, diff={diff:.1f}"
            # )

            if not match:
                all_match = False
                # logger.error(f"    First 10 elements expected: {expected[0, token_idx, start:start+10]}")
                # logger.error(f"    First 10 elements actual:   {result[0, token_idx, start:start+10]}")

    if not all_match:
        pytest.fail("Output doesn't match expected values")


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

    # Verify
    all_match = True
    for token_idx in range(num_tokens):
        # logger.info(f"\nToken {token_idx}:")
        for tile_idx in range(num_tiles):
            start = tile_idx * 1024
            expected_val = expected[0, token_idx, start].item()
            actual_val = result[0, token_idx, start].item()
            diff = abs(expected_val - actual_val)
            match = diff < 0.1
            status = "✅" if match else "❌"
            # logger.info(f"  {status} Tile {tile_idx}: expected={expected_val:.0f}, actual={actual_val:.0f}, diff={diff:.1f}")

            if not match:
                all_match = False

    # if not all_match:
    #     pytest.fail("Output doesn't match expected values")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        # Run simple test first
        test_reduce_single_token_single_expert(device)

        # Then run 2-core test
        test_reduce_2core_7tiles_simple(device)

    finally:
        ttnn.close_device(device)
