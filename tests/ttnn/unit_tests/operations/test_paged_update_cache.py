# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from models.utility_functions import nearest_32, pad_by_zero, skip_for_grayskull
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal


def run_test_update_cache_decode(
    cache_idx,
    cache_idx_tensor,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
    share_cache=False,
):
    input_shape = [1, num_users, num_heads, head_dim]
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    if share_cache:
        cache = torch.randn([1, num_heads, max_seq_len, head_dim]).bfloat16().float()
    else:
        cache = torch.randn(cache_shape).bfloat16().float()
    cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
    x = torch.randn(input_shape).bfloat16().float()
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_heads), "constant", 0)

    xt = ttnn.Tensor(x_pad, input_dtype).to(ttnn.TILE_LAYOUT)
    xt = ttnn.reshape(xt, ttnn.Shape(input_shape))
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
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    xt = xt.to(device, input_mem_config)

    # Create arbitrary update indices
    if share_cache:
        cache_idxs = [cache_idx + i for i in range(num_users)]
    else:
        cache_idxs = [cache_idx + i * 17 for i in range(num_users)]
    if cache_idx_tensor and not share_cache:
        cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)
        cachett = ttnn.experimental.paged_update_cache(cachett, xt, update_idxs_tensor=cache_idxs_tt, share_cache=False)
    else:
        cachett = ttnn.experimental.paged_update_cache(cachett, xt, update_idxs=cache_idxs, share_cache=share_cache)

    for i in range(num_users):
        update_idx = cache_idxs[i]
        x_view = x.permute(1, 2, 0, 3)[i, ...]
        i = 0 if share_cache else i
        cache[i, 0:num_heads, update_idx : update_idx + 1, 0 : x.shape[-1]] = x_view

    tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    tt_updated_slice = []
    for i in range(num_users):
        update_idx = cache_idxs[i]
        i = 0 if share_cache else i
        tt_slice = tt_got_back[i, 0:num_heads, update_idx : update_idx + 1, 0 : x.shape[-1]]  # n_heads, 1, head_dim
        tt_updated_slice.append(tt_slice)
    tt_updated_slice = torch.stack(tt_updated_slice, dim=0).permute(2, 0, 1, 3)

    expected_pcc = 0.98 if cache_dtype == ttnn.bfloat4_b else 0.99
    if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
        eq_cache, output_cache = comp_equal(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_equal(x, tt_updated_slice)  # checks the updated parts
    else:
        eq_cache, output_cache = comp_pcc(cache, tt_got_back, pcc=expected_pcc)  # checks the entire kv cache
        eq_update, output_update = comp_pcc(x, tt_updated_slice, pcc=expected_pcc)  # checks the updated parts
    logger.debug(output_cache)
    logger.debug(output_update)

    if (not eq_cache or not eq_update) and logger.level("DEBUG"):
        # find deltas between cache and tt_got_back
        cache_delta = cache - tt_got_back
        for i in range(max_seq_len):
            if cache_dtype == ttnn.bfloat16:
                if not torch.sum(cache_delta[:, :, i, :]) == 0:
                    logger.error(f"cache_delta at {i}: {cache_delta[:, :, i, :]}")
                    logger.info(f"cache at {i}: {cache[:, :, i, :]}")
                    logger.info(f"tt_got_back at {i}: {tt_got_back[:, :, i, :]}")
            else:
                eq, pcc = comp_pcc(cache[:, :, i, :], tt_got_back[:, :, i, :], pcc=expected_pcc)
                if not eq:
                    logger.error(f"cache_delta {pcc} pcc at {i}: {cache_delta[:, :, i, :]}")
                    logger.info(f"cache at {i}: {cache[:, :, i, :]}")
                    logger.info(f"tt_got_back at {i}: {tt_got_back[:, :, i, :]}")

    assert eq_cache and eq_update


@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.parametrize("check_memory", [False])
@pytest.mark.parametrize("share_cache", [True, False])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048, 32 * 1024])
@pytest.mark.parametrize("num_users", [4])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx_tensor", [True, False])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat16])
def test_update_cache_decode(
    check_memory,
    share_cache,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_idx_tensor,
    cache_dtype,
    device,
):
    if cache_dtype == ttnn.bfloat4_b and (share_cache or max_seq_len == 2048 or not cache_idx_tensor):
        pytest.skip("just need to sanity-check a select test case for bfp4")

    torch.manual_seed(0)

    if check_memory:
        # Create dram tensors to check for overflow
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        dram_low = ttnn.Tensor(torch.zeros(cache_shape), cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        reserved_space = ttnn.Tensor(torch.zeros(cache_shape), cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        dram_high = ttnn.Tensor(torch.zeros(cache_shape), cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
        reserved_space.deallocate(True)

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
        sharded_reserved = ttnn.Tensor(x_pad, input_dtype).to(ttnn.TILE_LAYOUT).to(device, input_mem_config)
        sharded_high = ttnn.Tensor(x_pad, input_dtype).to(ttnn.TILE_LAYOUT).to(device, input_mem_config)
        sharded_reserved.deallocate(True)

    for idx in [0, max_seq_len // 2]:
        run_test_update_cache_decode(
            max_seq_len // 2,
            cache_idx_tensor,
            head_dim,
            max_seq_len,
            num_users,
            num_heads,
            input_dtype,
            cache_dtype,
            device,
            share_cache,
        )

    if check_memory:
        # Check for overflow
        def check_zero(tensor):
            assert (tensor == 0).all()

        dram_low = dram_low.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        dram_high = dram_high.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        sharded_low = sharded_low.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        sharded_high = sharded_high.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        check_zero(dram_low)
        check_zero(dram_high)
        check_zero(sharded_low)
        check_zero(sharded_high)


@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx", [127, 1057])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_update_cache_decode_program_cache(
    cache_idx,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
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

        run_test_update_cache_decode(
            cache_idx, False, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
        )
        # Test that cache_idx is correctly updated between cached runs
        run_test_update_cache_decode(
            cache_idx + 1, False, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
        )

    assert device.num_program_cache_entries() == 1


def run_test_tensor_index_update_cache_decode(
    cache_idx, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
):
    input_shape = [1, num_users, num_heads, head_dim]
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    cache = torch.randn(cache_shape).bfloat16().float()

    cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
    x = torch.randn(input_shape).bfloat16().float()
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
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    xt = xt.to(device, input_mem_config)

    # Create arbitrary update indices
    cache_idxs = [cache_idx + i * 17 for i in range(num_users)]
    logger.info(f"cache_idxs: {cache_idxs}")
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)

    cachett = ttnn.experimental.paged_update_cache(cachett, xt, update_idxs_tensor=cache_idxs_tt)

    for i in range(num_users):
        update_idx = cache_idxs[i]
        x_view = x.permute(1, 0, 2, 3)[i, ...]
        cache[i, 0:num_heads, update_idx : update_idx + x.shape[-2], 0 : x.shape[-1]] = x_view

    tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    tt_updated_slice = []
    for i in range(num_users):
        update_idx = cache_idxs[i]
        tt_slice = tt_got_back[i, 0:num_heads, update_idx : update_idx + x.shape[-2], 0 : x.shape[-1]]
        tt_updated_slice.append(tt_slice)
    tt_updated_slice = torch.stack(tt_updated_slice, dim=0).permute(1, 0, 2, 3)

    if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
        eq_cache, output_cache = comp_equal(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_equal(x, tt_updated_slice)  # checks the updated parts
    else:
        eq_cache, output_cache = comp_pcc(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_pcc(x, tt_updated_slice)  # checks the updated parts
    logger.debug(output_cache)
    logger.debug(output_update)
    assert eq_cache and eq_update


@pytest.mark.skip("Test case covered by others")
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx", [0, 1, 127, 1057])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_tensor_index_update_cache_decode(
    cache_idx,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
):
    run_test_tensor_index_update_cache_decode(
        cache_idx, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
    )


@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx", [127, 1057])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_tensor_index_update_cache_decode_program_cache(
    cache_idx,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
):
    for _ in range(2):
        run_test_tensor_index_update_cache_decode(
            cache_idx, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
        )

    assert device.num_program_cache_entries() == 1


def run_test_paged_update_cache_decode(
    cache_idx, block_size, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
):
    max_num_blocks_per_seq = max_seq_len // block_size
    assert max_num_blocks_per_seq * block_size == max_seq_len
    max_num_blocks = num_users * max_seq_len // block_size
    assert max_num_blocks * block_size == num_users * max_seq_len

    input_shape = [1, num_users, num_heads, head_dim]
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    cache = torch.randn(cache_shape).bfloat16().float()
    paged_cache = (
        cache.reshape(num_users, num_heads, max_num_blocks_per_seq, block_size, head_dim)
        .transpose(1, 2)
        .reshape(max_num_blocks, num_heads, block_size, head_dim)
    )  # [num_users * max_num_blocks_per_seq, num_heads, block_size, head_dim]

    # Shuffle paged KV cache according to some random page_table
    permutation = torch.randperm(max_num_blocks)
    shuffled_page_cache = paged_cache[permutation]
    reverse_permutation = torch.argsort(permutation)
    # page_table is the reverse permutation from shuffled -> unshuffled, and is used to map
    # a virtual block to the physical block id.
    page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq)
    # logger.info(f"page_table: {page_table}")
    unshuffled_page_cache = shuffled_page_cache[reverse_permutation]

    paged_cache_back = (
        unshuffled_page_cache.reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
        .transpose(1, 2)
        .reshape(num_users, num_heads, max_seq_len, head_dim)
    )
    assert torch.allclose(paged_cache_back, cache)

    cachett = ttnn.Tensor(shuffled_page_cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
    x = torch.randn(input_shape).bfloat16().float()
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
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    xt = xt.to(device, input_mem_config)

    # Create arbitrary update indices
    cache_idxs = [cache_idx + i * 17 for i in range(num_users)]
    # Arbitrary user is "dropped", to test skipping in kernel
    if num_heads == 1:
        cache_idxs[num_users // 2] = -1
    # logger.info(f"cache_idxs: {cache_idxs}")
    cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    cachett = ttnn.experimental.paged_update_cache(
        cachett, xt, update_idxs_tensor=cache_idxs_tt, page_table=page_table_tt
    )

    for i in range(num_users):
        update_idx = cache_idxs[i]
        if update_idx == -1:
            continue
        x_view = x.permute(1, 2, 0, 3)[i, ...]
        cache[i, 0:num_heads, update_idx : update_idx + 1, 0 : x.shape[-1]] = x_view

    # Unshuffle paged cache and review it as unpaged cache
    tt_got_back_shuffled = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_got_back_unshuffled = tt_got_back_shuffled[reverse_permutation]
    tt_got_back = (
        tt_got_back_unshuffled.reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
        .transpose(1, 2)
        .reshape(num_users, num_heads, max_seq_len, head_dim)
    )

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

    if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
        eq_cache, output_cache = comp_equal(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_equal(x, tt_updated_slice)  # checks the updated parts
    else:
        eq_cache, output_cache = comp_pcc(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_pcc(x, tt_updated_slice)  # checks the updated parts
    logger.debug(output_cache)
    logger.debug(output_update)
    assert eq_cache and eq_update


@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.parametrize("block_size", [64, 128], ids=["block64", "block128"])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx", [0, 1, 127, 1057])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_paged_update_cache_decode(
    cache_idx,
    block_size,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
):
    run_test_paged_update_cache_decode(
        cache_idx, block_size, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
    )


@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.parametrize("block_size", [64, 128], ids=["block64", "block128"])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_idx", [127, 1057])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_paged_update_cache_decode_program_caching(
    cache_idx,
    block_size,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
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

        run_test_paged_update_cache_decode(
            cache_idx, block_size, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
        )

        run_test_paged_update_cache_decode(
            cache_idx + 10, block_size, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
        )

    assert device.num_program_cache_entries() == 1


def run_test_paged_fill_cache(
    block_size, head_dim, user_seq_len, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
):
    max_num_blocks_per_seq = max_seq_len // block_size
    assert max_num_blocks_per_seq * block_size == max_seq_len
    max_num_blocks = num_users * max_seq_len // block_size
    assert max_num_blocks * block_size == num_users * max_seq_len

    input_shape = [1, num_heads, user_seq_len, head_dim]
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    cache = torch.randn(cache_shape).bfloat16().float()

    # Turn cache into paged cache
    paged_cache = (
        cache.reshape(num_users, num_heads, max_num_blocks_per_seq, block_size, head_dim)
        .transpose(1, 2)
        .reshape(max_num_blocks, num_heads, block_size, head_dim)
    )  # [num_users * max_num_blocks_per_seq, num_heads, block_size, head_dim]

    # Shuffle paged KV cache according to some random page_table
    permutation = torch.randperm(max_num_blocks)
    shuffled_page_cache = paged_cache[permutation]
    reverse_permutation = torch.argsort(permutation)
    # page_table is the reverse permutation from shuffled -> unshuffled, and is used to map
    # a virtual block to the physical block id.
    page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq)

    # logger.info(f"page_table: {page_table}")
    unshuffled_page_cache = shuffled_page_cache[reverse_permutation]

    paged_cache_back = (
        unshuffled_page_cache.reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
        .transpose(1, 2)
        .reshape(num_users, num_heads, max_seq_len, head_dim)
    )
    # Check that we can convert from normal to paged to normal
    assert torch.allclose(paged_cache_back, cache)

    cachett = ttnn.Tensor(shuffled_page_cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # Update cache for every user
    for i in range(num_users):
        x = torch.randn(input_shape).bfloat16().float()
        xt = ttnn.Tensor(x, input_dtype).to(ttnn.TILE_LAYOUT).to(device)

        cachett = ttnn.experimental.paged_fill_cache(cachett, xt, page_table_tt, batch_idx=i)
        cache[i : i + 1, :, : x.shape[-2], :] = x

    # Unshuffle paged cache and review it as unpaged cache
    tt_got_back_shuffled = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_got_back_unshuffled = tt_got_back_shuffled[reverse_permutation]
    tt_got_back = (
        tt_got_back_unshuffled.reshape(num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim)
        .transpose(1, 2)
        .reshape(num_users, num_heads, max_seq_len, head_dim)
    )

    if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
        eq, output = comp_equal(cache, tt_got_back)
    else:
        eq, output = comp_pcc(cache, tt_got_back)
    logger.info(output)
    assert eq


@pytest.mark.skip("Test case covered by others")
@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.parametrize("block_size", [64, 128], ids=["block64", "block128"])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("user_seq_len", [128, 160, 1984, 2048])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_paged_fill_cache(
    block_size,
    head_dim,
    user_seq_len,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    device,  # use_program_cache
):
    logger.warning("Forcing cache_dtype to be same as input_dtype. Change test case when this is not required")
    cache_dtype = input_dtype
    run_test_paged_fill_cache(
        block_size, head_dim, user_seq_len, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
    )


@skip_for_grayskull("Grayskull does not support paged cache")
@pytest.mark.parametrize("block_size", [64, 128], ids=["block64", "block128"])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("user_seq_len", [128, 160, 1984, 2048])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_paged_fill_cache_program_cache(
    block_size, head_dim, user_seq_len, max_seq_len, num_users, num_heads, input_dtype, device
):
    cache_dtype = input_dtype
    run_test_paged_fill_cache(
        block_size, head_dim, user_seq_len, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
    )
    run_test_paged_fill_cache(
        block_size, head_dim, user_seq_len, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
    )

    assert device.num_program_cache_entries() == 1


# Helper function to compute expected output for paged_fill_cache
def get_expected_paged_fill_cache_output(
    initial_cache_torch,
    input_torch,
    page_table_torch,
    effective_batch_idx,
    skip_if_page_table_val_is_neg_one=True,  # To match our earlier kernel mod
):
    expected_cache_torch = initial_cache_torch.clone()
    TILE_HEIGHT = 32
    TILE_WIDTH = 32

    # Assuming input_torch shape: [1, num_heads, input_seq_len, head_dim]
    # Assuming cache_torch shape: [max_num_blocks, 1, block_size, head_dim]
    # Assuming page_table_torch shape: [batch_size, max_num_blocks_per_seq]

    num_input_heads = input_torch.shape[1]
    input_seq_len = input_torch.shape[2]
    head_dim = input_torch.shape[3]

    max_cache_blocks = expected_cache_torch.shape[0]
    cache_block_size = expected_cache_torch.shape[2]

    input_seq_len_t = input_seq_len // TILE_HEIGHT
    Wt = head_dim // TILE_WIDTH
    cache_block_size_t = cache_block_size // TILE_HEIGHT  # Number of tile-rows in a cache block

    # Parameters for virtual_seq_tile_id_to_physical_tile_id logic
    # const uint32_t num_heads = num_input_heads;
    # const uint32_t block_size_t = cache_block_size_t;
    # const uint32_t Wt = Wt;

    # block_stride in virtual_seq_tile_id_to_physical_tile_id calculation
    # This is the stride in tiles if all heads were packed into each physical block layer
    conceptual_block_stride_in_tiles = num_input_heads * cache_block_size_t * Wt

    for h_idx in range(num_input_heads):  # cur_head
        for s_tr_idx in range(input_seq_len_t):  # seq_tile_id or seq_tile_idx
            virtual_block_in_seq = s_tr_idx // cache_block_size_t

            if virtual_block_in_seq >= page_table_torch.shape[1]:
                continue

            physical_block_mapped_idx = page_table_torch[effective_batch_idx, virtual_block_in_seq].item()

            if skip_if_page_table_val_is_neg_one and physical_block_mapped_idx == -1:
                continue

            if physical_block_mapped_idx >= max_cache_blocks:
                continue

            tile_row_within_virtual_block = s_tr_idx % cache_block_size_t  # this is block_row_offset in kernel

            # Calculate physical_tile_id_start using logic from virtual_seq_tile_id_to_physical_tile_id
            head_offset_in_tiles = h_idx * cache_block_size_t * Wt
            block_offset_in_tiles = tile_row_within_virtual_block * Wt

            physical_tile_id_start = (
                physical_block_mapped_idx * conceptual_block_stride_in_tiles
                + head_offset_in_tiles
                + block_offset_in_tiles
            )

            # Get the source data slice from input_torch
            # This is a (TILE_HEIGHT, head_dim) slice
            input_data_slice = input_torch[0, h_idx, s_tr_idx * TILE_HEIGHT : (s_tr_idx + 1) * TILE_HEIGHT, :]

            # Write Wt tiles from input_data_slice to expected_cache_torch
            for w_col_tile_idx in range(Wt):
                current_global_tile_id = physical_tile_id_start + w_col_tile_idx

                # Map global tile ID to 4D cache coordinates (K, 1, R, C)
                # Actual cache block stride: 1 (dummy_head_dim) * cache_block_size_t * Wt
                actual_tiles_per_physical_block_layer = cache_block_size_t * Wt

                target_k = current_global_tile_id // actual_tiles_per_physical_block_layer
                remaining_tiles_in_target_block = current_global_tile_id % actual_tiles_per_physical_block_layer

                target_r_tile = remaining_tiles_in_target_block // Wt
                target_c_tile = remaining_tiles_in_target_block % Wt

                if target_k >= max_cache_blocks:
                    # This check is important if conceptual_block_stride makes g_id large
                    continue

                # Extract the source tile (TILE_HEIGHT, TILE_WIDTH)
                source_tile = input_data_slice[:, w_col_tile_idx * TILE_WIDTH : (w_col_tile_idx + 1) * TILE_WIDTH]

                # Place it in the expected cache
                expected_cache_torch[
                    target_k,
                    0,
                    target_r_tile * TILE_HEIGHT : (target_r_tile + 1) * TILE_HEIGHT,
                    target_c_tile * TILE_WIDTH : (target_c_tile + 1) * TILE_WIDTH,
                ] = source_tile

    return expected_cache_torch


@pytest.mark.parametrize(
    "num_input_heads, input_seq_len, head_dim, cache_max_blocks, cache_block_size, page_table_batch_size, pt_max_blocks_per_seq, effective_batch_idx_to_test, use_tensor_for_batch_idx",
    [
        (1, 128, 128, 1024, 64, 32, 32, 0, True),  # Llama, batch_idx_tensor, batch 0
        (1, 2048, 128, 1024, 64, 32, 32, 15, True),  # Llama, batch_idx_tensor, batch 15
    ],
)
def test_paged_fill_cache_variants(
    device,
    num_input_heads,
    input_seq_len,
    head_dim,
    cache_max_blocks,
    cache_block_size,
    page_table_batch_size,
    pt_max_blocks_per_seq,
    effective_batch_idx_to_test,
    use_tensor_for_batch_idx,
):
    torch.manual_seed(0)
    TILE_HEIGHT = 32
    TILE_WIDTH = 32

    assert head_dim % TILE_WIDTH == 0, "head_dim must be multiple of TILE_WIDTH"
    assert input_seq_len % TILE_HEIGHT == 0, "input_seq_len must be multiple of TILE_HEIGHT"
    assert cache_block_size % TILE_HEIGHT == 0, "cache_block_size must be multiple of TILE_HEIGHT"
    assert effective_batch_idx_to_test < page_table_batch_size, "effective_batch_idx_to_test out of bounds"

    # Prepare Tensors
    initial_cache_torch = torch.randn(cache_max_blocks, 1, cache_block_size, head_dim).bfloat16() * 100
    input_torch = (
        torch.arange(1 * num_input_heads * input_seq_len * head_dim, dtype=torch.float32)
        .reshape(1, num_input_heads, input_seq_len, head_dim)
        .bfloat16()
        / 1000.0
    )

    page_table_torch_data = []
    next_block_idx = 0
    for b in range(page_table_batch_size):
        row = []
        for m in range(pt_max_blocks_per_seq):
            row.append(next_block_idx % cache_max_blocks)
            next_block_idx += 1
        page_table_torch_data.append(row)
    page_table_torch = torch.tensor(page_table_torch_data, dtype=torch.int32)

    if page_table_batch_size > 0 and pt_max_blocks_per_seq > 1 and use_tensor_for_batch_idx:
        page_table_torch[effective_batch_idx_to_test, pt_max_blocks_per_seq - 1] = -1

    # Assuming cache_dtype for the ttnn.Tensor will be bfloat16 based on input torch tensor.
    # The op should handle this dtype for the cache.
    cache_tt = ttnn.from_torch(initial_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table_tt = ttnn.from_torch(page_table_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    batch_idx_tensor_dev = None
    batch_idx_fallback_val = 0

    if use_tensor_for_batch_idx:
        batch_idx_torch_scalar = torch.tensor([effective_batch_idx_to_test], dtype=torch.int32)
        batch_idx_tensor_dev = ttnn.from_torch(
            batch_idx_torch_scalar.to(torch.int32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
        )
        batch_idx_fallback_val = (effective_batch_idx_to_test + 1) % page_table_batch_size
    else:
        batch_idx_fallback_val = effective_batch_idx_to_test

    output_cache_tt = ttnn.experimental.paged_fill_cache(
        cache_tt,
        input_tt,
        page_table_tt,
        batch_idx_tensor=batch_idx_tensor_dev,
        batch_idx=batch_idx_fallback_val,
    )

    expected_cache_torch = get_expected_paged_fill_cache_output(
        initial_cache_torch, input_torch, page_table_torch, effective_batch_idx_to_test
    )

    output_cache_torch = ttnn.to_torch(output_cache_tt)

    # Use comp_equal for bfloat16 comparisons, similar to other tests in the file
    Passing, Message = comp_equal(output_cache_torch, expected_cache_torch)
    logger.info(Message)  # Log the comparison message regardless
    assert Passing, Message
