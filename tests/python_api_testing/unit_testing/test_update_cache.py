import torch
import pytest

import tt_lib as ttl


@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])

class TestUpdateCache:
    @pytest.mark.parametrize("seq_len", [128, 512, 1024])
    def test_update_cache_prefill(self, seq_len, head_dim, max_seq_len, num_users, device):

        input_shape = [1, 1, seq_len, head_dim]
        cache_shape = [num_users, 1, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = (
            ttl.tensor.Tensor(cache, ttl.tensor.DataType.BFLOAT16)
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )
        for i in range(num_users):
            x = torch.randn(input_shape).bfloat16().float()
            xt = (
                ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16)
                .to(ttl.tensor.Layout.TILE)
                .to(device)
            )
            cachett = ttl.tensor.update_cache(cachett, xt, i, 0)
            cache[i:i+1, 0:1, 0 : x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        assert torch.equal(tt_got_back, cache)

    @pytest.mark.parametrize("cache_idx", [128, 512, 1024])
    def test_update_cache_decode(self, head_dim, max_seq_len, num_users, cache_idx, device):

        input_shape = [num_users, 1, 1, head_dim]
        cache_shape = [num_users, 1, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()
        cachett = (
            ttl.tensor.Tensor(cache.permute(2, 1, 0, 3), ttl.tensor.DataType.BFLOAT16)
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )
        x = torch.randn(input_shape).bfloat16().float()
        xt = (
            ttl.tensor.Tensor(x.permute(2, 1, 0, 3), ttl.tensor.DataType.BFLOAT16)
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )
        cachett = ttl.tensor.update_cache(cachett, xt, cache_idx, 0)
        cache[0:num_users, 0:1, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().permute(2, 1, 0, 3)

        assert torch.equal(tt_got_back, cache)
