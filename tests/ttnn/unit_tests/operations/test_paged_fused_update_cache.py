# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from models.utility_functions import nearest_32, pad_by_zero, skip_for_grayskull
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal


def run_test_paged_fused_update_cache_decode(
    paged_update,
    cache_idx,
    block_size,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
    pcc,
    sub_core_grids=None,
    row_major=False,
):
    max_num_blocks_per_seq = max_seq_len // block_size
    assert max_num_blocks_per_seq * block_size == max_seq_len
    max_num_blocks = num_users * max_seq_len // block_size
    assert max_num_blocks * block_size == num_users * max_seq_len

    input_shape = [1, num_users, num_heads, head_dim]
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]

    # Initialize two caches
    cache1 = torch.randn(cache_shape).bfloat16().float()
    cache2 = torch.randn(cache_shape).bfloat16().float()

    def prepare_paged_cache(cache, permutation):
        paged_cache = (
            cache.reshape(num_users, num_heads, max_num_blocks_per_seq, block_size, head_dim)
            .transpose(1, 2)
            .reshape(max_num_blocks, num_heads, block_size, head_dim)
        )
        shuffled_page_cache = paged_cache[permutation]
        return shuffled_page_cache

    if paged_update:
        # Generate a common permutation for both caches
        permutation = torch.randperm(max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq)
        page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

        # Prepare paged caches for both cache1 and cache2
        shuffled_cache1 = prepare_paged_cache(cache1, permutation)
        shuffled_cache2 = prepare_paged_cache(cache2, permutation)

        # Convert to tile layout
        cachett1 = ttnn.Tensor(shuffled_cache1, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        cachett2 = ttnn.Tensor(shuffled_cache2, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
    else:
        page_table_tt = None
        cachett1 = ttnn.Tensor(cache1, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        cachett2 = ttnn.Tensor(cache2, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)

    # Prepare inputs
    x1 = torch.randn(input_shape).bfloat16().float()
    x2 = torch.randn(input_shape).bfloat16().float()
    if row_major:
        max_num_heads = 8
        assert num_heads <= max_num_heads, f"num_heads must be less than or equal to {max_num_heads}"
        x1_pad = torch.nn.functional.pad(x1, (0, 0, 0, max_num_heads - num_heads), "constant", 0)
        x2_pad = torch.nn.functional.pad(x2, (0, 0, 0, max_num_heads - num_heads), "constant", 0)

        xt1 = ttnn.Tensor(x1_pad, input_dtype).to(ttnn.ROW_MAJOR_LAYOUT)
        xt2 = ttnn.Tensor(x2_pad, input_dtype).to(ttnn.ROW_MAJOR_LAYOUT)
    else:
        x1_pad = torch.nn.functional.pad(x1, (0, 0, 0, 32 - num_heads), "constant", 0)
        x2_pad = torch.nn.functional.pad(x2, (0, 0, 0, 32 - num_heads), "constant", 0)

        xt1 = ttnn.Tensor(x1_pad, input_dtype).to(ttnn.TILE_LAYOUT)
        xt2 = ttnn.Tensor(x2_pad, input_dtype).to(ttnn.TILE_LAYOUT)

    # Sharding setup
    num_cores_per_cache = num_users
    assert (
        num_users % 8 == 0 or num_users == 1
    ), "num_users must be a multiple of 8 or less than 8 for fused_qk rotary embedding"

    if sub_core_grids is None:
        compute_grid_size = device.compute_with_storage_grid_size()
        grid_start_coord = ttnn.CoreCoord(0, 0)
        grid_end_coord = ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
        available_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(grid_start_coord, grid_end_coord)})
    else:
        grid_start_coord = sub_core_grids.ranges()[0].start
        available_core_grid = sub_core_grids

    shard_grid1 = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        grid_start_coord, num_cores_per_cache, available_core_grid, True
    )
    available_core_grid = available_core_grid.subtract(shard_grid1)
    grid_start_coord = available_core_grid.ranges()[0].start

    shard_grid2 = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        grid_start_coord, num_cores_per_cache, available_core_grid, True
    )
    input_shard_spec1 = ttnn.ShardSpec(
        shard_grid1,
        [
            xt1.volume() // xt1.padded_shape[-1] // num_cores_per_cache,
            xt1.padded_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_shard_spec2 = ttnn.ShardSpec(
        shard_grid2,
        [
            xt1.volume() // xt1.padded_shape[-1] // num_cores_per_cache,
            xt1.padded_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec1)
    input_mem_config2 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec2)
    xt1 = xt1.to(device, input_mem_config1)
    xt2 = xt2.to(device, input_mem_config2)

    # Update indices
    cache_idxs = [cache_idx + i * 17 for i in range(num_users)]
    if num_heads == 1:
        cache_idxs[num_users // 2] = -1
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    # Perform fused update cache operation
    cachett1, cachett2 = ttnn.experimental.paged_fused_update_cache(
        cachett1, xt1, cachett2, xt2, update_idxs_tensor=cache_idxs_tt, page_table=page_table_tt
    )

    # Verification for cache1 and cache2
    for cache_idx, cache, cachett, x in [(1, cache1, cachett1, x1), (2, cache2, cachett2, x2)]:
        for i in range(num_users):
            update_idx = cache_idxs[i]
            if update_idx == -1:
                continue
            x_view = x.permute(1, 2, 0, 3)[i, ...]
            cache[i, 0:num_heads, update_idx : update_idx + 1, 0 : x.shape[-1]] = x_view

        # Unshuffle paged cache and restore as unpaged cache
        tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if paged_update:
            tt_got_back = tt_got_back[reverse_permutation]
            tt_got_back = (
                tt_got_back.reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
                .transpose(1, 2)
                .reshape(num_users, num_heads, max_seq_len, head_dim)
            )

        # Verify updated slices
        tt_updated_slice = []
        for i in range(num_users):
            update_idx = cache_idxs[i]
            if update_idx == -1:
                # Skipped users should compare to the original cache
                update_idx = 0
                x[:, i : i + 1, :, :] = cache[
                    i : i + 1, 0:num_heads, update_idx : update_idx + x.shape[-2], 0 : x.shape[-1]
                ]
            tt_slice = tt_got_back[i, 0:num_heads, update_idx : update_idx + 1, 0 : x.shape[-1]]  # n_heads, 1, head_dim
            tt_updated_slice.append(tt_slice)
        tt_updated_slice = torch.stack(tt_updated_slice, dim=0).permute(2, 0, 1, 3)

        # Final validation
        if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
            eq_cache, output_cache = comp_equal(cache, tt_got_back)  # checks the entire kv cache
            eq_update, output_update = comp_equal(x, tt_updated_slice)  # checks the updated parts
        else:
            eq_cache, output_cache = comp_pcc(cache, tt_got_back, pcc)  # checks the entire kv cache
            eq_update, output_update = comp_pcc(x, tt_updated_slice, pcc)  # checks the updated parts

        logger.debug(output_cache)
        logger.debug(output_update)
        assert eq_cache and eq_update, f"Cache{cache_idx} and update slice mismatch"


@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("paged_update", [True, False])
@pytest.mark.parametrize("block_size", [64, 128], ids=["block64", "block128"])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [8])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx", [0, 1, 127, 1057])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("pcc", [0.9995])
def test_paged_fused_update_cache_decode(
    paged_update,
    cache_idx,
    block_size,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
    pcc,
):
    run_test_paged_fused_update_cache_decode(
        paged_update,
        cache_idx,
        block_size,
        head_dim,
        max_seq_len,
        num_users,
        num_heads,
        input_dtype,
        cache_dtype,
        device,
        pcc,
    )


@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("paged_update", [True, False])
@pytest.mark.parametrize("block_size", [64, 128], ids=["block64", "block128"])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [8])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx", [127, 1057])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("pcc", [0.9995])
def test_paged_fused_update_cache_decode_program_caching(
    paged_update,
    cache_idx,
    block_size,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
    pcc,
):
    dummy_tensors = []
    for i in range(2):
        # Create dram tensors to check for overflow
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        dram_low = ttnn.Tensor(torch.zeros(cache_shape), cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        dummy_tensors.append(dram_low)

        # Create sharded tensors to check for overflow
        input_shape = [1, num_users, num_heads, head_dim]
        x = torch.zeros(input_shape)
        x_pad = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_heads), "constant", 0)

        xt = ttnn.Tensor(x_pad, input_dtype).to(ttnn.TILE_LAYOUT)
        # Input is sharded
        compute_grid_size = device.compute_with_storage_grid_size()
        num_cores = num_users
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
        sharded_low = xt.to(device, input_mem_config)
        dummy_tensors.append(sharded_low)

        run_test_paged_fused_update_cache_decode(
            paged_update,
            cache_idx,
            block_size,
            head_dim,
            max_seq_len,
            num_users,
            num_heads,
            input_dtype,
            cache_dtype,
            device,
            pcc,
        )

        run_test_paged_fused_update_cache_decode(
            paged_update,
            cache_idx + 10,
            block_size,
            head_dim,
            max_seq_len,
            num_users,
            num_heads,
            input_dtype,
            cache_dtype,
            device,
            pcc,
        )

    assert device.num_program_cache_entries() == 1
