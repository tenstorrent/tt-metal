# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal test to validate the "fake tiles" concept.

Tests if we can:
1. Create ROW_MAJOR tensor [1, 8, 7168]
2. Read it as 7 "tiles" per expert (7168 / 1024 = 7)
3. Get back the correct data

This validates our core assumption before testing the full operation.
"""

import pytest
import torch
import ttnn
from loguru import logger


@pytest.mark.parametrize(
    "num_experts,emb_dim",
    [
        (8, 7168),  # DeepSeek actual size
        (2, 1024),  # Minimal test (1 tile per expert)
        (4, 2048),  # Medium test (2 tiles per expert)
    ],
)
def test_fake_tiles_data_layout(num_experts, emb_dim, device):
    """
    Test that ROW_MAJOR data can be read as 'fake tiles'.

    Creates: [1, num_experts, emb_dim] ROW_MAJOR tensor
    Validates: Can read as (num_experts × emb_dim/1024) tiles
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Fake Tiles: num_experts={num_experts}, emb_dim={emb_dim}")
    logger.info(f"{'='*80}")

    # Create test data
    torch.manual_seed(42)
    data_torch = torch.randn(1, num_experts, emb_dim, dtype=torch.bfloat16)

    logger.info(f"Created PyTorch tensor: {data_torch.shape}")

    # Convert to TTNN ROW_MAJOR
    data_tt = ttnn.from_torch(
        data_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    logger.info(f"Converted to TTNN ROW_MAJOR")
    logger.info(f"  Shape: {data_tt.shape}")
    logger.info(f"  Layout: {data_tt.layout}")
    logger.info(f"  Memory: {data_tt.memory_config()}")

    # Check buffer properties
    buffer = data_tt.buffer()
    logger.info(f"\nBuffer properties:")
    logger.info(f"  Page size: {buffer.page_size()} bytes")
    logger.info(f"  Num pages: {buffer.num_pages()}")
    logger.info(f"  Size: {buffer.size()} bytes")
    logger.info(f"  Buffer type: {buffer.buffer_type()}")

    # Calculate expected values
    tile_size = 1024  # elements per "fake tile"
    tiles_per_expert = emb_dim // tile_size
    total_tiles = num_experts * tiles_per_expert
    bytes_per_page = emb_dim * 2  # bfloat16 = 2 bytes

    logger.info(f"\nExpected:")
    logger.info(f"  Tiles per expert: {tiles_per_expert} (emb_dim={emb_dim} / tile_size={tile_size})")
    logger.info(f"  Total tiles: {total_tiles}")
    logger.info(f"  Bytes per page: {bytes_per_page}")

    # Validate buffer layout matches our assumptions
    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDATION:")
    logger.info(f"{'='*80}")

    # Check 1: Page size should be emb_dim * 2 bytes (one row)
    expected_page_size = emb_dim * 2
    if buffer.page_size() == expected_page_size:
        logger.info(f"✅ Page size CORRECT: {buffer.page_size()} == {expected_page_size}")
    else:
        logger.error(f"❌ Page size WRONG: {buffer.page_size()} != {expected_page_size}")
        logger.error(f"   This means ROW_MAJOR isn't laid out as we expect!")
        pytest.fail(f"Page size mismatch: {buffer.page_size()} != {expected_page_size}")

    # Check 2: Number of pages should be num_experts (one page per expert)
    if buffer.num_pages() == num_experts:
        logger.info(f"✅ Num pages CORRECT: {buffer.num_pages()} == {num_experts}")
    else:
        logger.error(f"❌ Num pages WRONG: {buffer.num_pages()} != {num_experts}")
        pytest.fail(f"Num pages mismatch: {buffer.num_pages()} != {num_experts}")

    # Check 3: Total size should be num_experts * emb_dim * 2
    expected_total_size = num_experts * emb_dim * 2
    if buffer.size() == expected_total_size:
        logger.info(f"✅ Total size CORRECT: {buffer.size()} == {expected_total_size}")
    else:
        logger.error(f"❌ Total size WRONG: {buffer.size()} != {expected_total_size}")
        pytest.fail(f"Total size mismatch: {buffer.size()} != {expected_total_size}")

    # Check 4: Convert back and verify data integrity
    logger.info(f"\n{'='*80}")
    logger.info(f"DATA INTEGRITY CHECK:")
    logger.info(f"{'='*80}")

    data_back = ttnn.to_torch(data_tt)

    max_diff = (data_torch - data_back).abs().max().item()
    logger.info(f"Max difference: {max_diff}")

    if torch.allclose(data_torch, data_back, rtol=1e-2, atol=1e-2):
        logger.info(f"✅ Data integrity VERIFIED: Input == Output")
    else:
        logger.error(f"❌ Data integrity FAILED: Input != Output")
        logger.error(f"   Max diff: {max_diff}")
        pytest.fail(f"Data integrity check failed")

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ ALL CHECKS PASSED!")
    logger.info(f"{'='*80}")
    logger.info(f"\nConclusion: ROW_MAJOR [1, {num_experts}, {emb_dim}] can be read as {total_tiles} 'fake tiles'")
    logger.info(f"  - Each expert = {tiles_per_expert} tiles of {tile_size} elements")
    logger.info(f"  - Buffer page = one expert's data ({emb_dim} elements = {bytes_per_page} bytes)")
    logger.info(f"  - This validates our assumption for the main operation! 🎉")


@pytest.mark.parametrize("emb_dim", [7168, 1024, 2048, 4096])
def test_tile_size_calculation(emb_dim, device):
    """
    Simple test: verify emb_dim divides evenly into 1024-element tiles.
    """
    tile_size = 1024

    if emb_dim % tile_size == 0:
        tiles = emb_dim // tile_size
        logger.info(f"✅ {emb_dim} elements = {tiles} tiles (tile_size={tile_size})")
    else:
        pytest.fail(f"❌ {emb_dim} elements doesn't divide evenly into {tile_size}-element tiles!")


if __name__ == "__main__":
    # Quick manual test
    device = ttnn.open_device(device_id=0)

    try:
        logger.info("=" * 80)
        logger.info("FAKE TILES VALIDATION TEST")
        logger.info("=" * 80)

        # Test the actual DeepSeek dimensions
        test_fake_tiles_data_layout(8, 7168, device)

        # Test smaller dimensions
        logger.info("\n" + "=" * 80)
        test_fake_tiles_data_layout(2, 1024, device)

    finally:
        ttnn.close_device(device)
