# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_py_test_utils_common.paged_fused_update_cache import run_test_paged_fused_update_cache_decode

import torch
import pytest
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal


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


@pytest.mark.timeout(1200)
# Non-paged only: the paged fused-update path validates that update positions arrive via an index
# TENSOR (which is Buffer-bound and re-patched anyway), so the frozen-attr bug only applies here.
@pytest.mark.parametrize("paged_update", [False])
@pytest.mark.parametrize("block_size", [128], ids=["block128"])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [8])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("pcc", [0.9995])
def test_paged_fused_update_cache_decode_attr_idxs_program_caching(
    paged_update, block_size, head_dim, max_seq_len, num_users, num_heads, input_dtype, cache_dtype, device, pcc
):
    """Regression for override_runtime_arguments on the NON-index-tensor (update_idxs list) path. The two
    cached calls below differ only in their positions, which are excluded from the program hash — so they
    must share ONE cache entry, and the second call must write at its OWN positions (cache_start_id /
    tile_update_offset_B re-derived on the cache hit, not frozen at the first call's values)."""
    # Cache miss, then a HIT at different positions, same shapes.
    run_test_paged_fused_update_cache_decode(
        paged_update,
        127,
        block_size,
        head_dim,
        max_seq_len,
        num_users,
        num_heads,
        input_dtype,
        cache_dtype,
        device,
        pcc,
        use_attr_idxs=True,
    )
    run_test_paged_fused_update_cache_decode(
        paged_update,
        1057,
        block_size,
        head_dim,
        max_seq_len,
        num_users,
        num_heads,
        input_dtype,
        cache_dtype,
        device,
        pcc,
        use_attr_idxs=True,
    )
    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("use_index_tensor", [False, True])
def test_paged_fused_update_cache_preserves_fp32_neighbors(input_dtype, use_index_tensor, device):
    """Tiled fused update must retain untouched FLOAT32 cache rows exactly."""
    seq_len, head_dim, update_idx = 64, 64, 5
    exponent = torch.arange(seq_len * head_dim) % 23 + 1
    cache = (1.0 + torch.pow(2.0, -exponent.double())).float().reshape(1, 1, seq_len, head_dim)
    cache1_tt = ttnn.from_torch(cache, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    cache2_tt = ttnn.from_torch(cache, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    padded_heads = 32
    inputs = []
    for core, value in enumerate((3.0, 5.0)):
        update = torch.full((1, 1, padded_heads, head_dim), value, dtype=torch.float32)
        update_tt = ttnn.Tensor(update, input_dtype).to(ttnn.TILE_LAYOUT)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(core, 0), ttnn.CoreCoord(core, 0))})
        shard_spec = ttnn.ShardSpec(grid, [padded_heads, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        inputs.append(update_tt.to(device, memory_config))

    expected1, expected2 = cache.clone(), cache.clone()
    entries_before = device.num_program_cache_entries()
    for index in (update_idx, update_idx + 32):
        kwargs = (
            {"update_idxs_tensor": ttnn.Tensor(torch.tensor([index]), ttnn.int32).to(device)}
            if use_index_tensor
            else {"update_idxs": [index]}
        )
        ttnn.experimental.paged_fused_update_cache(cache1_tt, inputs[0], cache2_tt, inputs[1], **kwargs)
        expected1[0, 0, index, :] = 3.0
        expected2[0, 0, index, :] = 5.0
        assert torch.equal(ttnn.to_torch(cache1_tt), expected1)
        assert torch.equal(ttnn.to_torch(cache2_tt), expected2)
    assert device.num_program_cache_entries() - entries_before == 1


@pytest.mark.parametrize(
    "cache_dtypes,input_dtypes,error",
    [
        ((ttnn.bfloat16, ttnn.float32), (ttnn.bfloat16, ttnn.bfloat16), "matching cache tensor dtypes"),
        ((ttnn.bfloat16, ttnn.bfloat16), (ttnn.bfloat16, ttnn.float32), "matching input tensor dtypes"),
    ],
)
def test_paged_fused_update_cache_rejects_mixed_dtypes(cache_dtypes, input_dtypes, error, device, expect_error):
    """The fused program shares buffer formats between its two tensor pairs."""
    cache = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    update = torch.ones((1, 1, 32, 64), dtype=torch.float32)
    caches = [ttnn.from_torch(cache, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) for dtype in cache_dtypes]
    inputs = []
    for core, dtype in enumerate(input_dtypes):
        update_tt = ttnn.Tensor(update, dtype).to(ttnn.TILE_LAYOUT)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(core, 0), ttnn.CoreCoord(core, 0))})
        shard_spec = ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR)
        memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        inputs.append(update_tt.to(device, memory_config))

    with expect_error(RuntimeError, error):
        ttnn.experimental.paged_fused_update_cache(caches[0], inputs[0], caches[1], inputs[1], update_idxs=[5])
