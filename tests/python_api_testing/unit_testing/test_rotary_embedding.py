import torch
import pytest
from loguru import logger

import tt_lib as ttl
from tt_models.utility_functions import comp_pcc


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos_cached, sin_cached):
    seq_len = x.shape[-2]
    cos = cos_cached[:, :, :seq_len, ...]
    sin = sin_cached[:, :, :seq_len, ...]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


@pytest.mark.parametrize(
    "W, Z, Y, X",
    (
        [1, 1, 128, 64],
        [1, 71, 128, 64],
        [32, 1, 1, 64],
        [32, 71, 1, 64],
        [64, 1, 1, 64],
        [64, 71, 1, 64],
    ),
)
@pytest.mark.parametrize("cache_size", [2048])
def test_rotary_embedding(W, Z, Y, X, cache_size, device):
    torch.manual_seed(0)

    input_shape = [W, Z, Y, X]
    sin_cos_shape = [W, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()
    seq_len = input_shape[-2]

    xt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16)
    if xt.shape()[-2] % 32 == 0 and xt.shape()[-1] % 32 == 0:
        xt = xt.to(ttl.tensor.Layout.TILE)
    xt = xt.to(device)

    cost = (
        ttl.tensor.Tensor(cos_cached, ttl.tensor.DataType.BFLOAT16)
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    sint = (
        ttl.tensor.Tensor(sin_cached, ttl.tensor.DataType.BFLOAT16)
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    xtt = ttl.tensor.rotary_embedding(xt, cost, sint)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached)

    p, o = comp_pcc(pt_out, tt_got_back)
    logger.info(o)
    assert p
