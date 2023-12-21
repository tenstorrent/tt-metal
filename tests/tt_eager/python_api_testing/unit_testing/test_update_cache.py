# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import tt_lib as ttl
from models.utility_functions import nearest_32
from models.utility_functions import skip_for_wormhole_b0


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

    @skip_for_wormhole_b0()
    @pytest.mark.parametrize("cache_idx", [0, 1, 127, 128, 1024, 1057])
    def test_update_cache_decode(self, head_dim, max_seq_len, num_users, cache_idx, device):
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

        tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        eq = torch.equal(tt_got_back, cache)
        assert eq

    @skip_for_wormhole_b0()
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
