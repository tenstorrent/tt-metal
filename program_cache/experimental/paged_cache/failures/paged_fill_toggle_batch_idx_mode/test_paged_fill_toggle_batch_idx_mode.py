# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.test_paged_update_cache import (
    get_expected_paged_fill_cache_output,
)


@pytest.mark.timeout(60)
def test_paged_fill_toggle_batch_idx_mode_program_cache(device):
    torch.manual_seed(0)

    # Use the same conventions as the existing unit tests for paged_fill_cache
    TILE_HEIGHT = 32
    TILE_WIDTH = 32

    # Small, fast configuration that respects tile multiples
    num_input_heads = 1
    input_seq_len = 128  # multiple of TILE_HEIGHT
    head_dim = 128  # multiple of TILE_WIDTH
    cache_block_size = 64  # multiple of TILE_HEIGHT
    page_table_batch_size = 8  # number of users
    pt_max_blocks_per_seq = 4  # virtual blocks per sequence

    # Total number of physical cache blocks
    cache_max_blocks = page_table_batch_size * pt_max_blocks_per_seq

    # Prepare inputs similar to test_paged_fill_cache_variants
    initial_cache_torch = torch.randn(cache_max_blocks, 1, cache_block_size, head_dim).bfloat16() * 100
    input_torch = (
        torch.arange(1 * num_input_heads * input_seq_len * head_dim, dtype=torch.float32)
        .reshape(1, num_input_heads, input_seq_len, head_dim)
        .bfloat16()
        / 1000.0
    )

    # Build a simple page table mapping that sequentially maps virtual blocks to physical blocks
    page_table_torch_data = []
    next_block_idx = 0
    for _ in range(page_table_batch_size):
        row = []
        for _ in range(pt_max_blocks_per_seq):
            row.append(next_block_idx % cache_max_blocks)
            next_block_idx += 1
        page_table_torch_data.append(row)
    page_table_torch = torch.tensor(page_table_torch_data, dtype=torch.int32)

    # Device tensors
    cache_tt = ttnn.from_torch(initial_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table_tt = ttnn.from_torch(page_table_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    # First run: batch_idx provided as tensor. Fallback scalar intentionally different.
    effective_batch_idx_to_test = 0
    batch_idx_tensor_dev = ttnn.from_torch(
        torch.tensor([effective_batch_idx_to_test], dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    fallback_batch_idx = (effective_batch_idx_to_test + 1) % page_table_batch_size

    logger.debug("First run: batch_idx as tensor, scalar fallback differs")
    num_cache_start = device.num_program_cache_entries()
    cache_tt = ttnn.experimental.paged_fill_cache(
        cache_tt,
        input_tt,
        page_table_tt,
        batch_idx_tensor=batch_idx_tensor_dev,
        batch_idx=fallback_batch_idx,
    )
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1, "Expected one new program cache entry on first run"

    # Validate correctness for the tensor batch_idx path
    expected_after_first = get_expected_paged_fill_cache_output(
        initial_cache_torch, input_torch, page_table_torch, effective_batch_idx_to_test
    )
    got_after_first = ttnn.to_torch(cache_tt)
    ok_first, msg_first = comp_equal(got_after_first, expected_after_first)
    logger.debug(msg_first)
    assert ok_first, "First run should be correct with tensor batch_idx"

    # Second run: toggle to scalar-only batch_idx (compile-time path differs)
    logger.debug("Second run: batch_idx as scalar only (toggle mode)")
    num_cache_start2 = device.num_program_cache_entries()
    cache_tt = ttnn.experimental.paged_fill_cache(
        cache_tt,
        input_tt,
        page_table_tt,
        batch_idx=effective_batch_idx_to_test,
    )
    num_cache_end2 = device.num_program_cache_entries()

    # The program cache key must include whether batch_idx comes from a tensor vs scalar.
    # If it doesn't, this assertion will fail, exposing the under-keyed hash issue.
    assert (
        num_cache_end2 == num_cache_start2 + 1
    ), "Toggling batch_idx tensor->scalar should compile a distinct program; cache entries did not increase"

    # Optional: verify correctness again after the second run
    expected_after_second = get_expected_paged_fill_cache_output(
        expected_after_first, input_torch, page_table_torch, effective_batch_idx_to_test
    )
    got_after_second = ttnn.to_torch(cache_tt)
    ok_second, msg_second = comp_equal(got_after_second, expected_after_second)
    logger.debug(msg_second)
    assert ok_second, "Second run output mismatch"
