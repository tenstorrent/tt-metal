# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row-major decode input for ``paged_update_cache`` (single cache tensor).

TILE inputs are untilized before the token-row splice. ROW_MAJOR inputs are
already contiguous per head, so that untilize is skipped.
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def _height_sharded_rm(device, x, input_dtype):
    num_users = x.shape[1]
    xt = ttnn.Tensor(x, input_dtype).to(ttnn.ROW_MAJOR_LAYOUT)
    shard_grid = ttnn.num_cores_to_corerangeset(num_users, device.compute_with_storage_grid_size(), True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [xt.volume() // xt.padded_shape[-1] // num_users, xt.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return xt.to(device, mem_config)


@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("paged", [False, True])
def test_paged_update_cache_row_major_input(device, cache_dtype, paged):
    torch.manual_seed(0)
    num_users, num_heads, head_dim, max_seq_len, cache_idx = 4, 8, 128, 256, 17
    block_size = 32
    input_shape = [1, num_users, num_heads, head_dim]
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    cache = torch.randn(cache_shape).bfloat16().float()
    x = torch.randn(input_shape).bfloat16().float()
    cache_idxs = [cache_idx + i * 17 for i in range(num_users)]

    if paged:
        max_num_blocks_per_seq = max_seq_len // block_size
        max_num_blocks = num_users * max_num_blocks_per_seq
        paged_cache = (
            cache.reshape(num_users, num_heads, max_num_blocks_per_seq, block_size, head_dim)
            .transpose(1, 2)
            .reshape(max_num_blocks, num_heads, block_size, head_dim)
        )
        permutation = torch.randperm(max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq)
        cachett = ttnn.Tensor(paged_cache[permutation], cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)
    else:
        reverse_permutation = None
        page_table_tt = None
        cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)

    xt = _height_sharded_rm(device, x, ttnn.bfloat16)
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)
    cachett = ttnn.experimental.paged_update_cache(
        cachett, xt, update_idxs_tensor=cache_idxs_tt, page_table=page_table_tt
    )

    for i in range(num_users):
        update_idx = cache_idxs[i]
        x_view = x.permute(1, 2, 0, 3)[i, ...]
        cache[i, 0:num_heads, update_idx : update_idx + 1, 0:head_dim] = x_view

    tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    if paged:
        max_num_blocks_per_seq = max_seq_len // block_size
        tt_got_back = (
            tt_got_back[reverse_permutation]
            .reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
            .transpose(1, 2)
            .reshape(num_users, num_heads, max_seq_len, head_dim)
        )

    if cache_dtype == ttnn.bfloat16:
        eq, _ = comp_equal(cache, tt_got_back)
        assert eq
    else:
        from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

        eq, _ = comp_pcc(cache, tt_got_back, pcc=0.99)
        assert eq
