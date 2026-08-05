# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
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
        cache_idxs_tt = ttnn.Tensor(torch.tensor(cache_idxs), ttnn.int32).to(device)
        cachett = ttnn.experimental.paged_update_cache(
            cachett, xt, update_idxs_tensor=cache_idxs_tt, share_cache=share_cache
        )

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


def run_update_cache_decode_attr_idxs(
    cache_idx, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
):
    """Drives the NON-index-tensor path: update positions are passed via the `update_idxs` list
    (operation_attributes.update_idxs), which the C++ factory bakes into runtime args. This is the
    path exercised by the DynamicRuntimeArg re-patching — the index-tensor path is already covered by
    run_test_update_cache_decode and is correct on cache hits for a different reason (buffer re-patch)."""
    input_shape = [1, num_users, num_heads, head_dim]
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    cache = torch.randn(cache_shape).bfloat16().float()
    cachett = ttnn.Tensor(cache, cache_dtype).to(ttnn.TILE_LAYOUT).to(device)
    x = torch.randn(input_shape).bfloat16().float()
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_heads), "constant", 0)

    xt = ttnn.Tensor(x_pad, input_dtype).to(ttnn.TILE_LAYOUT)
    xt = ttnn.reshape(xt, ttnn.Shape(input_shape))
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = num_users
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        [xt.volume() // xt.padded_shape[-1] // num_cores, xt.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    xt = xt.to(device, input_mem_config)

    # Distinct per-user positions; varying cache_idx between cached calls must move where the cache is
    # written. update_idxs is passed as a plain list (NO update_idxs_tensor) -> attribute path.
    cache_idxs = [cache_idx + i * 17 for i in range(num_users)]
    cachett = ttnn.experimental.paged_update_cache(cachett, xt, update_idxs=cache_idxs, share_cache=False)

    for i in range(num_users):
        update_idx = cache_idxs[i]
        x_view = x.permute(1, 2, 0, 3)[i, ...]
        cache[i, 0:num_heads, update_idx : update_idx + 1, 0 : x.shape[-1]] = x_view

    tt_got_back = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    tt_updated_slice = []
    for i in range(num_users):
        update_idx = cache_idxs[i]
        tt_slice = tt_got_back[i, 0:num_heads, update_idx : update_idx + 1, 0 : x.shape[-1]]
        tt_updated_slice.append(tt_slice)
    tt_updated_slice = torch.stack(tt_updated_slice, dim=0).permute(2, 0, 1, 3)

    if input_dtype == ttnn.bfloat16 and cache_dtype == input_dtype:
        eq_cache, _ = comp_equal(cache, tt_got_back)
        eq_update, _ = comp_equal(x, tt_updated_slice)
    else:
        eq_cache, _ = comp_pcc(cache, tt_got_back, pcc=0.99)
        eq_update, _ = comp_pcc(x, tt_updated_slice, pcc=0.99)
    assert eq_cache and eq_update


@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16])
def test_update_cache_decode_attr_idxs_program_cache(
    head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
):
    """Regression for the DynamicRuntimeArg fix on the NON-index-tensor path. update_idxs is excluded
    from the program hash, so the two calls below must share ONE cache entry; and because the cache
    write offset (cache_start_id / tile_update_offset_B) is re-patched on the cache hit, the second
    call must write at its OWN positions (not the first call's frozen ones). Pins both halves:
      - num_program_cache_entries() == 1  guards the hash exclusion (no re-hashing on differing idxs)
      - the in-process correctness check on the 2nd call guards the frozen-arg bug."""
    # First call: cache miss, builds + caches the program at one set of positions.
    run_update_cache_decode_attr_idxs(
        127, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
    )
    # Second call: DIFFERENT positions, same shapes -> program-cache HIT. Output must reflect the new
    # positions (would fail with frozen cache_start_id before the get_dynamic_runtime_args fix).
    run_update_cache_decode_attr_idxs(
        1057, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device
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


@pytest.mark.parametrize(
    "num_input_heads, input_seq_len, head_dim, cache_max_blocks, cache_block_size, page_table_batch_size, pt_max_blocks_per_seq, effective_batch_idx_to_test",
    [
        (1, 128, 128, 1024, 64, 32, 32, 0),
        (1, 128, 128, 1024, 64, 32, 32, 15),
    ],
)
def test_paged_fill_cache_mesh_coords(
    device,
    num_input_heads,
    input_seq_len,
    head_dim,
    cache_max_blocks,
    cache_block_size,
    page_table_batch_size,
    pt_max_blocks_per_seq,
    effective_batch_idx_to_test,
):
    torch.manual_seed(0)
    TILE_HEIGHT = 32
    TILE_WIDTH = 32

    assert head_dim % TILE_WIDTH == 0, "head_dim must be multiple of TILE_WIDTH"
    assert input_seq_len % TILE_HEIGHT == 0, "input_seq_len must be multiple of TILE_HEIGHT"
    assert cache_block_size % TILE_HEIGHT == 0, "cache_block_size must be multiple of TILE_HEIGHT"
    assert effective_batch_idx_to_test < page_table_batch_size, "effective_batch_idx_to_test out of bounds"

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

    cache_tt = ttnn.from_torch(initial_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table_tt = ttnn.from_torch(page_table_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    # For a single device we can only use coordinate (0, 0)
    mesh_coords = {ttnn.MeshCoordinate(0, 0)}
    output_cache_tt = ttnn.experimental.paged_fill_cache(
        cache_tt,
        input_tt,
        page_table_tt,
        batch_idx=effective_batch_idx_to_test,
        mesh_coords=mesh_coords,
    )

    expected_cache_torch = get_expected_paged_fill_cache_output(
        initial_cache_torch, input_torch, page_table_torch, effective_batch_idx_to_test
    )

    output_cache_torch = ttnn.to_torch(output_cache_tt)

    passing, message = comp_equal(output_cache_torch, expected_cache_torch)
    logger.info(message)  # Log the comparison message regardless
    assert passing, message


# ----------------------------------------------------------------------------
# Batched paged_fill_cache: batch_idx_tensor with shape == [input_batch]
# ----------------------------------------------------------------------------
#
# These tests exercise the new path where a single op call fills `input_batch`
# rows of the cache in one shot, rather than the Python-loop pattern of one
# scalar-batch_idx call per row. They validate against the per-row reference
# (`get_expected_paged_fill_cache_output` applied iteratively).


def _build_batched_inputs(
    num_input_batch,
    num_input_heads,
    input_seq_len,
    head_dim,
    cache_max_blocks,
    cache_block_size,
    page_table_batch_size,
    pt_max_blocks_per_seq,
    batch_idxs,
    inject_skip_for_batch_idx=None,
    seed=0,
):
    """Construct a randomized cache + batched input + page_table for tests.

    Cache is allocated as `[cache_max_blocks, 1, cache_block_size, head_dim]`,
    matching how `test_paged_fill_cache_variants` and real vLLM/JAX callers
    set up paged KV caches (cache_num_heads = 1 because GQA-style models share
    a single KV-head dimension). The device-op now requires
    `input_num_heads == cache_num_heads`, so callers of this helper must pass
    `num_input_heads = 1`. The `num_input_heads` parameter is kept in the
    signature for future use if a multi-head variant is added (which would
    also need an updated cache shape and reference function).

    cache_max_blocks is auto-bumped to a safe lower bound so all
    `page_table_batch_size * pt_max_blocks_per_seq` slots fit even after the
    kernel's per-head spill arithmetic; harmless for num_input_heads=1 (factor
    of 1).
    """
    torch.manual_seed(seed)

    required_blocks = page_table_batch_size * pt_max_blocks_per_seq * num_input_heads
    if cache_max_blocks < required_blocks:
        cache_max_blocks = required_blocks

    initial_cache_torch = torch.randn(cache_max_blocks, 1, cache_block_size, head_dim).bfloat16() * 100
    input_torch = (
        torch.arange(num_input_batch * num_input_heads * input_seq_len * head_dim, dtype=torch.float32)
        .reshape(num_input_batch, num_input_heads, input_seq_len, head_dim)
        .bfloat16()
        / 1000.0
    )

    page_table_torch_data = []
    next_block_idx = 0
    for _b in range(page_table_batch_size):
        row = []
        for _m in range(pt_max_blocks_per_seq):
            row.append(next_block_idx % cache_max_blocks)
            next_block_idx += 1
        page_table_torch_data.append(row)
    page_table_torch = torch.tensor(page_table_torch_data, dtype=torch.int32)

    if inject_skip_for_batch_idx is not None and pt_max_blocks_per_seq > 1:
        page_table_torch[inject_skip_for_batch_idx, pt_max_blocks_per_seq - 1] = -1

    return initial_cache_torch, input_torch, page_table_torch


def _reference_batched_paged_fill_cache(initial_cache_torch, input_torch, page_table_torch, batch_idxs):
    """Apply paged_fill_cache once per input batch row, accumulating into a clone."""
    num_input_batch = input_torch.shape[0]
    assert num_input_batch == len(batch_idxs)
    expected = initial_cache_torch.clone()
    for b in range(num_input_batch):
        # The single-batch reference expects input shape [1, num_heads, seq_len, head_dim].
        single_batch_input = input_torch[b : b + 1]
        expected = get_expected_paged_fill_cache_output(
            expected,
            single_batch_input,
            page_table_torch,
            int(batch_idxs[b]),
        )
    return expected


def _run_batched_paged_fill_cache(
    device,
    num_input_batch,
    num_input_heads,
    input_seq_len,
    head_dim,
    cache_max_blocks,
    cache_block_size,
    page_table_batch_size,
    pt_max_blocks_per_seq,
    batch_idxs,
    inject_skip_for_batch_idx=None,
):
    TILE_HEIGHT = 32
    TILE_WIDTH = 32
    assert head_dim % TILE_WIDTH == 0
    assert input_seq_len % TILE_HEIGHT == 0
    assert cache_block_size % TILE_HEIGHT == 0
    for bi in batch_idxs:
        assert int(bi) < page_table_batch_size

    initial_cache_torch, input_torch, page_table_torch = _build_batched_inputs(
        num_input_batch=num_input_batch,
        num_input_heads=num_input_heads,
        input_seq_len=input_seq_len,
        head_dim=head_dim,
        cache_max_blocks=cache_max_blocks,
        cache_block_size=cache_block_size,
        page_table_batch_size=page_table_batch_size,
        pt_max_blocks_per_seq=pt_max_blocks_per_seq,
        batch_idxs=batch_idxs,
        inject_skip_for_batch_idx=inject_skip_for_batch_idx,
    )

    cache_tt = ttnn.from_torch(initial_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table_tt = ttnn.from_torch(page_table_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    batch_idxs_torch = torch.tensor(batch_idxs, dtype=torch.int32)
    batch_idx_tensor_dev = ttnn.from_torch(
        batch_idxs_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )

    output_cache_tt = ttnn.experimental.paged_fill_cache(
        cache_tt,
        input_tt,
        page_table_tt,
        batch_idx_tensor=batch_idx_tensor_dev,
        batch_idx=0,
    )

    expected_cache_torch = _reference_batched_paged_fill_cache(
        initial_cache_torch, input_torch, page_table_torch, batch_idxs
    )

    output_cache_torch = ttnn.to_torch(output_cache_tt)
    passing, message = comp_equal(output_cache_torch, expected_cache_torch)
    logger.info(message)
    assert passing, message


# Only num_input_heads=1 is exercised here. The device-op validation
# (paged_fill_cache_device_operation.cpp:62) requires
# `input_num_heads == cache_num_heads`, and the test helper allocates the cache
# with `cache_num_heads = 1` (matching how real vLLM/JAX callers configure paged
# KV caches with GQA-style head sharing). Multi-head batched coverage would need
# a separate helper that allocates cache as `[max_blocks, num_input_heads, ...]`
# and an adapted reference function; not added here since no in-tree caller uses
# that shape and the legacy tests (test_paged_fill_cache_variants etc.) only
# exercise num_input_heads=1 as well.
@pytest.mark.parametrize("num_input_heads", [1])
@pytest.mark.parametrize("input_seq_len", [128, 2048])
@pytest.mark.parametrize(
    "num_input_batch, batch_idxs",
    [
        (2, [0, 1]),
        (8, list(range(8))),
        (32, list(range(32))),
        # Non-contiguous batch_idxs: each input row writes to a non-sequential cache row.
        (8, [5, 12, 0, 31, 17, 3, 9, 24]),
    ],
    ids=["b2_contig", "b8_contig", "b32_contig", "b8_permuted"],
)
def test_paged_fill_cache_batched(device, num_input_heads, input_seq_len, num_input_batch, batch_idxs):
    """Single batched call fills `num_input_batch` cache rows in one op."""
    _run_batched_paged_fill_cache(
        device,
        num_input_batch=num_input_batch,
        num_input_heads=num_input_heads,
        input_seq_len=input_seq_len,
        head_dim=128,
        cache_max_blocks=1024,
        cache_block_size=64,
        page_table_batch_size=32,
        pt_max_blocks_per_seq=32,
        batch_idxs=batch_idxs,
    )


@pytest.mark.parametrize(
    "num_input_batch, batch_idxs, skip_for_batch_idx",
    [
        (4, [0, 1, 2, 3], 2),  # core may span the skipped batch
        (8, [3, 1, 7, 0, 4, 2, 6, 5], 7),
    ],
    ids=["b4_skip_b2", "b8_perm_skip_b7"],
)
def test_paged_fill_cache_batched_skip_entries(device, num_input_batch, batch_idxs, skip_for_batch_idx):
    """Per-block -1 (SKIP) sentinel entries in the page_table under the batched path."""
    _run_batched_paged_fill_cache(
        device,
        num_input_batch=num_input_batch,
        num_input_heads=1,
        input_seq_len=2048,
        head_dim=128,
        cache_max_blocks=1024,
        cache_block_size=64,
        page_table_batch_size=32,
        pt_max_blocks_per_seq=32,
        batch_idxs=batch_idxs,
        inject_skip_for_batch_idx=skip_for_batch_idx,
    )


def test_paged_fill_cache_batched_program_cache(device):
    """Identical batched shapes share one program cache entry; new shape allocates a new one."""
    common = dict(
        num_input_heads=1,
        input_seq_len=128,
        head_dim=128,
        cache_max_blocks=1024,
        cache_block_size=64,
        page_table_batch_size=32,
        pt_max_blocks_per_seq=32,
    )

    initial_entries = device.num_program_cache_entries()

    _run_batched_paged_fill_cache(device, num_input_batch=4, batch_idxs=[0, 1, 2, 3], **common)
    after_first = device.num_program_cache_entries()
    assert after_first - initial_entries == 1, "first call should create one program cache entry"

    _run_batched_paged_fill_cache(device, num_input_batch=4, batch_idxs=[4, 5, 6, 7], **common)
    after_second = device.num_program_cache_entries()
    assert after_second == after_first, "same batched shape should reuse the program cache entry"

    _run_batched_paged_fill_cache(device, num_input_batch=8, batch_idxs=list(range(8)), **common)
    after_third = device.num_program_cache_entries()
    assert after_third - after_second == 1, "different input_batch should allocate a new entry"


@pytest.mark.parametrize("bad_tensor_size", [1, 3, 5, 7])
def test_paged_fill_cache_batched_rejects_mismatched_batch_idx_tensor(device, bad_tensor_size):
    """batch_idx_tensor must have exactly input_batch elements; anything else FATALs.

    Covers the silent-wrong-result hole where input_batch > 1 paired with a
    single-element batch_idx_tensor previously passed validation but produced
    out-of-range writes in the kernel.
    """
    initial_cache_torch, input_torch, page_table_torch = _build_batched_inputs(
        num_input_batch=4,
        num_input_heads=1,
        input_seq_len=128,
        head_dim=128,
        cache_max_blocks=1024,
        cache_block_size=64,
        page_table_batch_size=32,
        pt_max_blocks_per_seq=32,
        batch_idxs=[0, 1, 2, 3],
    )
    cache_tt = ttnn.from_torch(initial_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table_tt = ttnn.from_torch(page_table_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    bad = ttnn.from_torch(
        torch.arange(bad_tensor_size, dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    with pytest.raises(RuntimeError):
        ttnn.experimental.paged_fill_cache(cache_tt, input_tt, page_table_tt, batch_idx_tensor=bad, batch_idx=0)


def test_paged_fill_cache_rejects_multi_batch_input_without_tensor(device):
    """input_batch > 1 with no batch_idx_tensor was previously a silent wrong-result; now FATAL."""
    initial_cache_torch, input_torch, page_table_torch = _build_batched_inputs(
        num_input_batch=4,
        num_input_heads=1,
        input_seq_len=128,
        head_dim=128,
        cache_max_blocks=1024,
        cache_block_size=64,
        page_table_batch_size=32,
        pt_max_blocks_per_seq=32,
        batch_idxs=[0, 1, 2, 3],
    )
    cache_tt = ttnn.from_torch(initial_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table_tt = ttnn.from_torch(page_table_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    with pytest.raises(RuntimeError):
        ttnn.experimental.paged_fill_cache(cache_tt, input_tt, page_table_tt, batch_idx=0)


def test_paged_fill_cache_batched_mesh_coords(device):
    """Mesh workload variant of the batched path on a single device.

    The mesh workload factory dispatches to PagedFillCacheProgramFactory per
    coordinate, so it inherits the batched behavior. This test pins it down
    against the per-row reference.
    """
    num_input_batch = 4
    batch_idxs = [0, 1, 2, 3]
    initial_cache_torch, input_torch, page_table_torch = _build_batched_inputs(
        num_input_batch=num_input_batch,
        num_input_heads=1,
        input_seq_len=128,
        head_dim=128,
        cache_max_blocks=1024,
        cache_block_size=64,
        page_table_batch_size=32,
        pt_max_blocks_per_seq=32,
        batch_idxs=batch_idxs,
    )
    cache_tt = ttnn.from_torch(initial_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table_tt = ttnn.from_torch(page_table_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    batch_idx_tensor_dev = ttnn.from_torch(
        torch.tensor(batch_idxs, dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    mesh_coords = {ttnn.MeshCoordinate(0, 0)}
    output_cache_tt = ttnn.experimental.paged_fill_cache(
        cache_tt,
        input_tt,
        page_table_tt,
        batch_idx_tensor=batch_idx_tensor_dev,
        batch_idx=0,
        mesh_coords=mesh_coords,
    )

    expected_cache_torch = _reference_batched_paged_fill_cache(
        initial_cache_torch, input_torch, page_table_torch, batch_idxs
    )
    output_cache_torch = ttnn.to_torch(output_cache_tt)
    passing, message = comp_equal(output_cache_torch, expected_cache_torch)
    logger.info(message)
    assert passing, message


def test_paged_fill_cache_batched_rejects_non_row_major_batch_idx_tensor(device):
    """The writer kernel reads batch_idx_tensor as a single contiguous noc page;
    require ROW_MAJOR layout so that read covers the whole 1D buffer."""
    initial_cache_torch, input_torch, page_table_torch = _build_batched_inputs(
        num_input_batch=4,
        num_input_heads=1,
        input_seq_len=128,
        head_dim=128,
        cache_max_blocks=1024,
        cache_block_size=64,
        page_table_batch_size=32,
        pt_max_blocks_per_seq=32,
        batch_idxs=[0, 1, 2, 3],
    )
    cache_tt = ttnn.from_torch(initial_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    input_tt = ttnn.from_torch(input_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    page_table_tt = ttnn.from_torch(page_table_torch, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    # batch_idx_tensor must be ROW_MAJOR; TILE_LAYOUT should be rejected.
    # int dtypes can't be tilized directly, so build a ROW_MAJOR tensor and
    # tilize it; if that path is not supported, the test will skip itself.
    try:
        rm_tensor = ttnn.from_torch(
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        bad = ttnn.to_layout(rm_tensor, ttnn.TILE_LAYOUT)
    except (RuntimeError, ValueError):
        pytest.skip("Cannot construct a TILE_LAYOUT uint32 tensor on this build; layout assertion still in place.")

    with pytest.raises(RuntimeError):
        ttnn.experimental.paged_fill_cache(cache_tt, input_tt, page_table_tt, batch_idx_tensor=bad, batch_idx=0)
