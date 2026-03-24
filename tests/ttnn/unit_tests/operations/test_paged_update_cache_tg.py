# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test paged_update_cache correctness on WH Galaxy TG mesh (8x4 = 32 devices).

Validates that paged_update_cache actually writes data to DRAM on multi-device
mesh configurations. This test was created because paged_update_cache was
proven to be a silent no-op on BH Galaxy mesh — we need to confirm it works
on WH Galaxy TG.

Tests:
  1. Write to multiple cache positions, read back, compare
  2. BF16 and BF8_B cache dtypes
  3. Multiple batch sizes
  4. Shuffled page tables (virtual -> physical block mapping)
  5. Position-level diagnostics to pinpoint exactly which writes fail
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal


def run_paged_update_cache_tg_test(
    mesh_device,
    num_users: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    cache_dtype,
    update_positions: list[int],
):
    """
    Core test: write to specific cache positions via paged_update_cache,
    read back the entire cache, verify each written position matches.

    Args:
        mesh_device: TG mesh device
        num_users: Batch size
        num_heads: Number of KV heads (typically 1 for GQA)
        head_dim: Head dimension (128 for GLM-4.7)
        max_seq_len: Maximum sequence length for the cache
        block_size: Page/block size (64 or 128)
        cache_dtype: Cache tensor dtype (bfloat16 or bfloat8_b)
        update_positions: List of sequence positions to write to
    """
    torch.manual_seed(42)

    max_num_blocks_per_seq = max_seq_len // block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    # --- Create torch reference tensors ---
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    cache = torch.randn(cache_shape).bfloat16().float()

    # Reshape cache into paged format: [max_num_blocks, num_heads, block_size, head_dim]
    paged_cache = (
        cache.reshape(num_users, num_heads, max_num_blocks_per_seq, block_size, head_dim)
        .transpose(1, 2)
        .reshape(max_num_blocks, num_heads, block_size, head_dim)
    )

    # Shuffle pages with a random permutation (simulates real vLLM page allocation)
    permutation = torch.randperm(max_num_blocks)
    shuffled_page_cache = paged_cache[permutation]
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq)

    # Verify the shuffle/unshuffle round-trips correctly
    unshuffled = shuffled_page_cache[reverse_permutation]
    paged_back = (
        unshuffled.reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
        .transpose(1, 2)
        .reshape(num_users, num_heads, max_seq_len, head_dim)
    )
    assert torch.allclose(paged_back, cache), "Page table round-trip failed"

    # --- Create device tensors ---
    # Cache tensor on device (paged, shuffled, DRAM interleaved)
    cachett = ttnn.Tensor(shuffled_page_cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(mesh_device)

    # Page table on device
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(mesh_device)

    # --- Write to each position and track what we expect ---
    for pos_idx, update_pos in enumerate(update_positions):
        # Create unique input data for this position (easily identifiable)
        input_shape = [1, num_users, num_heads, head_dim]
        # Use position-specific seed for unique, identifiable data
        x = torch.randn(input_shape).bfloat16().float() * (pos_idx + 1)
        x_pad = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_heads), "constant", 0)

        xt = ttnn.Tensor(x_pad, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
        # Reshape to set logical batch
        xt = ttnn.reshape(xt, ttnn.Shape(input_shape))

        # Shard input across cores (HEIGHT_SHARDED, one shard per batch element)
        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        num_cores = min(num_users, compute_grid_size.x * compute_grid_size.y)
        shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
        input_shard_spec = ttnn.ShardSpec(
            shard_grid,
            [
                xt.volume() // xt.padded_shape[-1] // num_cores,
                xt.padded_shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
        )
        xt = xt.to(mesh_device, input_mem_config)

        # Create update indices: all users write to the same position
        cache_idxs = [update_pos] * num_users
        cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs, dtype=torch.int32), ttnn.int32).to(mesh_device)

        # Execute the operation under test
        ttnn.experimental.paged_update_cache(
            cachett, xt, update_idxs_tensor=cache_idxs_tt, page_table=page_table_tt
        )

        # Update torch reference
        for i in range(num_users):
            x_view = x.permute(1, 2, 0, 3)[i, ...]  # [num_heads, 1, head_dim]
            cache[i, 0:num_heads, update_pos : update_pos + 1, 0 : x.shape[-1]] = x_view

    # --- Read back and compare ---
    tt_got_back_shuffled = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_got_back_unshuffled = tt_got_back_shuffled[reverse_permutation]
    tt_got_back = (
        tt_got_back_unshuffled.reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
        .transpose(1, 2)
        .reshape(num_users, num_heads, max_seq_len, head_dim)
    )

    # --- Position-level diagnostics ---
    all_pass = True
    expected_pcc = 0.98 if cache_dtype == ttnn.bfloat8_b else 0.999

    # Check each updated position specifically
    for update_pos in update_positions:
        expected_slice = cache[:, :, update_pos : update_pos + 1, :]
        actual_slice = tt_got_back[:, :, update_pos : update_pos + 1, :]

        if cache_dtype == ttnn.bfloat16:
            eq, output = comp_equal(expected_slice, actual_slice)
        else:
            eq, output = comp_pcc(expected_slice, actual_slice, pcc=expected_pcc)

        if not eq:
            logger.error(f"FAIL at position {update_pos}: {output}")
            # Check if the position is all zeros (silent no-op)
            if torch.all(actual_slice == 0):
                logger.error(f"  -> Position {update_pos} is ALL ZEROS — silent no-op detected!")
            else:
                # Check if it matches the original (pre-update) cache value
                logger.error(f"  -> actual max={actual_slice.abs().max().item():.6f}, "
                           f"expected max={expected_slice.abs().max().item():.6f}")
            all_pass = False
        else:
            logger.info(f"PASS at position {update_pos}: {output}")

    # Also check the entire cache for unexpected corruption
    if cache_dtype == ttnn.bfloat16:
        eq_full, output_full = comp_equal(cache, tt_got_back)
    else:
        eq_full, output_full = comp_pcc(cache, tt_got_back, pcc=expected_pcc)

    logger.info(f"Full cache comparison: {output_full}")
    if not eq_full:
        logger.error("Full cache comparison FAILED — possible corruption at non-updated positions")
        all_pass = False

    assert all_pass, f"paged_update_cache validation failed on mesh {list(mesh_device.shape)}"


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

# GLM-4.7 Full realistic config: 1 KV head, head_dim=128, block_size=64
GLM47_FULL_POSITIONS = [0, 1, 31, 32, 63, 64, 127, 500, 1023, 2000]


@pytest.mark.parametrize(
    "cache_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["bf16", "bf8"],
)
@pytest.mark.parametrize(
    "num_users",
    [1, 4, 8],
    ids=["bs1", "bs4", "bs8"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_tg")], indirect=True)
def test_paged_update_cache_tg_glm47(mesh_device, num_users, cache_dtype):
    """Test with GLM-4.7 Full realistic parameters on WH TG (8x4) mesh."""
    run_paged_update_cache_tg_test(
        mesh_device=mesh_device,
        num_users=num_users,
        num_heads=1,
        head_dim=128,
        max_seq_len=2048,
        block_size=64,
        cache_dtype=cache_dtype,
        update_positions=GLM47_FULL_POSITIONS,
    )


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_tg")], indirect=True)
def test_paged_update_cache_tg_multi_head(mesh_device):
    """Test with 8 KV heads (non-GQA config) to stress multi-head path."""
    run_paged_update_cache_tg_test(
        mesh_device=mesh_device,
        num_users=4,
        num_heads=8,
        head_dim=128,
        max_seq_len=2048,
        block_size=64,
        cache_dtype=ttnn.bfloat16,
        update_positions=[0, 31, 64, 500, 1023],
    )


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_tg")], indirect=True)
def test_paged_update_cache_tg_block128(mesh_device):
    """Test with block_size=128 (another common vLLM configuration)."""
    run_paged_update_cache_tg_test(
        mesh_device=mesh_device,
        num_users=4,
        num_heads=1,
        head_dim=128,
        max_seq_len=2048,
        block_size=128,
        cache_dtype=ttnn.bfloat16,
        update_positions=[0, 1, 63, 127, 128, 500, 1023],
    )


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_tg")], indirect=True)
def test_paged_update_cache_tg_sequential_writes(mesh_device):
    """
    Stress test: write to 50 sequential positions (simulates 50 decode steps).
    This tests whether repeated paged_update_cache calls accumulate correctly
    without corrupting each other.
    """
    positions = list(range(50))
    run_paged_update_cache_tg_test(
        mesh_device=mesh_device,
        num_users=1,
        num_heads=1,
        head_dim=128,
        max_seq_len=2048,
        block_size=64,
        cache_dtype=ttnn.bfloat16,
        update_positions=positions,
    )


# Fallback: smaller mesh for environments without full TG
@pytest.mark.parametrize(
    "cache_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=["bf16", "bf8"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_mesh")], indirect=True)
def test_paged_update_cache_2x4_mesh(mesh_device, cache_dtype):
    """Smaller mesh test for environments without full 8x4 TG."""
    run_paged_update_cache_tg_test(
        mesh_device=mesh_device,
        num_users=4,
        num_heads=1,
        head_dim=128,
        max_seq_len=2048,
        block_size=64,
        cache_dtype=cache_dtype,
        update_positions=[0, 1, 31, 64, 500, 1023],
    )
