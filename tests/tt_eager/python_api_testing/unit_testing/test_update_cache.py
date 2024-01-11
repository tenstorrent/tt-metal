# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import tt_lib as ttl
from models.utility_functions import nearest_32


@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32, 64])
class TestUpdateCache:
    @pytest.mark.parametrize("seq_len", [128, 512, 1024])
    def test_fill_cache(self, seq_len, head_dim, max_seq_len, num_users, device):
        input_shape = [1, 1, seq_len, head_dim]
        cache_shape = [num_users, 1, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttl.tensor.Tensor(cache, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
        for i in range(num_users):
            x = torch.randn(input_shape).bfloat16().float()
            xt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
            cachett = ttl.tensor.fill_cache(cachett, xt, i)
            cache[i : i + 1, 0:1, 0 : x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        eq = torch.equal(tt_got_back, cache)
        assert eq

    @pytest.mark.parametrize("num_heads", [1, 2, 8])
    @pytest.mark.parametrize("cache_idx", [0, 1, 127, 128, 1024, 1057])
    @pytest.mark.parametrize("in_sharded", [True, False])
    def test_update_cache_decode(
        self, head_dim, max_seq_len, num_users, num_heads, cache_idx, device, in_sharded, use_program_cache
    ):
        input_shape = [num_users, num_heads, 1, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttl.tensor.Tensor(cache, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
        x = torch.randn(input_shape).bfloat16().float()
        xt = ttl.tensor.Tensor(x.permute(2, 1, 0, 3), ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE)
        if in_sharded:
            compute_grid_size = device.compute_with_storage_grid_size()
            num_cores = min(num_users // 32 * num_heads, compute_grid_size.x * compute_grid_size.y)
            if num_cores == 1:
                pytest.skip("Issue #4706: Can't write 1 core sharded tensors directly to device")
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
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1
            )
            xt = xt.to(device, input_mem_config, input_shard_spec)
        else:
            xt = xt.to(device)

        cachett = ttl.tensor.update_cache(cachett, xt, cache_idx)
        cache[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        eq = torch.equal(tt_got_back, cache)
        assert eq

    @pytest.mark.parametrize("cache_idx", [0, 1, 127, 128, 1024, 1057])
    def test_update_cache_and_slice_decode(self, head_dim, max_seq_len, num_users, cache_idx, device):
        input_shape = [num_users, 1, 1, head_dim]
        cache_shape = [num_users, 1, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = ttl.tensor.Tensor(cache, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
        x = torch.randn(input_shape).bfloat16().float()
        xt = (
            ttl.tensor.Tensor(x.permute(2, 1, 0, 3), ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
        )
        cachett = ttl.tensor.update_cache(cachett, xt, cache_idx)
        cache[0:num_users, 0:1, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]] = x

        cachett = ttl.tensor.unpad(
            cachett,
            [0, 0, 0, 0],
            [num_users - 1, 0, nearest_32(cache_idx + 1) - 1, head_dim - 1],
        )
        cachett_t = ttl.tensor.transpose(cachett, -2, -1)
        cache = cache[:, :, : nearest_32(cache_idx + 1), :]
        cache_t = torch.transpose(cache, -2, -1)

        tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        tt_t_got_back = cachett_t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        eq = torch.equal(tt_got_back, cache)
        assert eq
        eq = torch.equal(tt_t_got_back, cache_t)
        assert eq
