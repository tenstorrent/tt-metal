# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import tt_lib as ttl
from loguru import logger
from models.utility_functions import nearest_32
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal


@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32, 64])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("in_sharded", [True, False])
@pytest.mark.parametrize("input_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
class TestUpdateCache:
    @pytest.mark.parametrize("seq_len", [32, 512, 2048])
    def test_fill_cache(
        self, seq_len, head_dim, max_seq_len, num_users, num_heads, in_sharded, input_dtype, device, use_program_cache
    ):
        if not in_sharded and num_heads > 1 and seq_len == 2048:
            pytest.skip(
                "For interleaved, each core can only have 1 tile along seq_len if num_heads > 1, so there is a restriction on max seq_len!"
            )

        cache_dtype = input_dtype

        input_shape = [1, num_heads, seq_len, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttl.tensor.Tensor(cache, cache_dtype).to(ttl.tensor.Layout.TILE).to(device)
        for i in range(num_users):
            x = torch.randn(input_shape).bfloat16().float()
            xt = ttl.tensor.Tensor(x, input_dtype).to(ttl.tensor.Layout.TILE)
            if in_sharded:
                compute_grid_size = device.compute_with_storage_grid_size()
                num_cores = min(seq_len // 32 * num_heads, 32)  # Always use max 32 cores for testing
                shard_grid = ttl.tensor.CoreRangeSet(
                    ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
                )
                input_shard_spec = ttl.tensor.ShardSpec(
                    shard_grid,
                    [
                        xt.volume() // xt.shape()[-1] // num_cores,
                        xt.shape()[-1],
                    ],
                    ttl.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                )
                input_mem_config = ttl.tensor.MemoryConfig(
                    ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
                )
                xt = xt.to(device, input_mem_config)
            else:
                xt = xt.to(device)

            cachett = ttl.tensor.fill_cache(cachett, xt, i)
            cache[i : i + 1, :, : x.shape[-2], :] = x

        tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if input_dtype == ttl.tensor.DataType.BFLOAT16 and cache_dtype == input_dtype:
            eq, output = comp_equal(cache, tt_got_back)
        else:
            eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq

    @pytest.mark.parametrize("cache_idx", [0, 1, 127, 1057])
    @pytest.mark.parametrize("cache_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
    def test_update_cache_decode(
        self,
        cache_idx,
        head_dim,
        max_seq_len,
        num_users,
        num_heads,
        in_sharded,
        input_dtype,
        cache_dtype,
        device,
        use_program_cache,
    ):
        input_shape = [num_users, num_heads, 1, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttl.tensor.Tensor(cache, cache_dtype).to(ttl.tensor.Layout.TILE).to(device)
        x = torch.randn(input_shape).bfloat16().float()
        xt = ttl.tensor.Tensor(x.permute(2, 1, 0, 3), input_dtype).to(ttl.tensor.Layout.TILE)
        if in_sharded:
            compute_grid_size = device.compute_with_storage_grid_size()
            num_cores = min(num_users // 32 * num_heads, compute_grid_size.x * compute_grid_size.y)
            shard_grid = ttl.tensor.CoreRangeSet(
                ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
            )
            input_shard_spec = ttl.tensor.ShardSpec(
                shard_grid,
                [
                    xt.volume() // xt.shape()[-1] // num_cores,
                    xt.shape()[-1],
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            )
            input_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
            )
            xt = xt.to(device, input_mem_config)
        else:
            xt = xt.to(device)

        cachett = ttl.tensor.update_cache(cachett, xt, cache_idx)
        cache[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if input_dtype == ttl.tensor.DataType.BFLOAT16 and cache_dtype == input_dtype:
            eq, output = comp_equal(cache, tt_got_back)
        else:
            eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq
