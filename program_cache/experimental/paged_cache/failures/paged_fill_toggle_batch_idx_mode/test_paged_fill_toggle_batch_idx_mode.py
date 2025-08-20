# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


@pytest.mark.timeout(30)
def test_paged_fill_toggle_batch_idx_mode_program_cache(device):
    torch.manual_seed(0)

    # Shapes (small to keep test fast)
    num_users = 8
    num_heads = 1
    head_dim = 128
    block_size = 64
    max_seq_len = 256
    max_num_blocks_per_seq = max_seq_len // block_size
    max_num_blocks = num_users * max_num_blocks_per_seq

    # Prepare a shuffle page table and the shuffled cache on host
    cache = torch.randn([num_users, num_heads, max_seq_len, head_dim]).bfloat16().float()
    paged_cache = (
        cache.reshape(num_users, num_heads, max_num_blocks_per_seq, block_size, head_dim)
        .transpose(1, 2)
        .reshape(max_num_blocks, num_heads, block_size, head_dim)
    )
    permutation = torch.randperm(max_num_blocks)
    shuffled_page_cache = paged_cache[permutation]
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq)

    # Create device tensors
    cache_tt = ttnn.Tensor(shuffled_page_cache, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # One input per user
    input_shape = [1, num_heads, block_size, head_dim]
    x = torch.randn(input_shape).bfloat16().float()
    x_tt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    # First run: use batch_idx as a tensor (compile-time path A)
    logger.debug("Executing first run (batch_idx as tensor)")
    batch_idx_tensor = ttnn.Tensor(torch.tensor([0], dtype=torch.int32), ttnn.uint32).to(device)
    num_cache_start = device.num_program_cache_entries()
    cache_tt = ttnn.experimental.paged_fill_cache(
        cache_tt, x_tt, page_table_tt, batch_idx_tensor=batch_idx_tensor, batch_idx=0
    )
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1, "Expected one new program cache entry on first run"

    # Validate first run correctness
    got = cache_tt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()[reverse_permutation]
    expected = (
        shuffled_page_cache[reverse_permutation]
        .clone()
        .reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
        .transpose(1, 2)
        .reshape(num_users, num_heads, max_seq_len, head_dim)
    )
    ok1, _ = comp_pcc(got, expected)
    assert ok1, "First run should be correct"

    # Second run: hit cache, switch to scalar fallback (compile-time path B) but hash is unchanged → expect failure
    logger.debug("Executing second run (cache-hit expected, batch_idx scalar fallback)")
    cache_tt = ttnn.experimental.paged_fill_cache(cache_tt, x_tt, page_table_tt, batch_idx=0)

    got2 = cache_tt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()[reverse_permutation]
    ok2, pcc2 = comp_pcc(got2, expected)
    logger.debug(f"Second run PCC: ok={ok2}, pcc={pcc2}")
    # Let this assertion fail on cache-hit if override/compile-time args are incorrect
    assert (
        ok2
    ), "PCC mismatch on cache-hit when toggling batch_idx tensor vs scalar indicates under-keyed hash and/or missing override"
