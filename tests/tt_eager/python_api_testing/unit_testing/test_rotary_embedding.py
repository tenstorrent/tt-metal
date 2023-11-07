# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc
from models.utility_functions import skip_for_wormhole_b0


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx=None):
    seq_len = x.shape[-2]
    if token_idx is None:
        cos = cos_cached[:, :, :seq_len, ...]
        sin = sin_cached[:, :, :seq_len, ...]
    else:
        cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
        sin = sin_cached[:, :, token_idx : token_idx + 1, ...]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "W, Z, Y, X",
    ([1, 1, 128, 64], [1, 71, 128, 64], [32, 1, 32, 64], [32, 71, 32, 64]),
)
@pytest.mark.parametrize("cache_size", [2048])
def test_rotary_embedding_prefill(W, Z, Y, X, cache_size, device):
    torch.manual_seed(0)

    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()
    seq_len = input_shape[-2]

    xt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16)
    if xt.shape()[-2] % 32 == 0 and xt.shape()[-1] % 32 == 0:
        xt = xt.to(ttl.tensor.Layout.TILE)
    xt = xt.to(device)

    cost = ttl.tensor.Tensor(cos_cached, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    sint = ttl.tensor.Tensor(sin_cached, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.rotary_embedding(xt, cost, sint)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached)

    p, o = comp_pcc(pt_out, tt_got_back)
    logger.info(o)
    assert p


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "W, Z, Y, X",
    ([1, 1, 32, 64], [1, 71, 32, 64], [1, 1, 64, 64], [1, 71, 64, 64]),
)
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("token_idx", [0, 128, 129, 1024, 1025])
def test_rotary_embedding_decode(W, Z, Y, X, cache_size, token_idx, device):
    torch.manual_seed(0)

    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.randn(input_shape).bfloat16().float()
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()
    seq_len = input_shape[-2]

    xt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16)
    if xt.shape()[-2] % 32 == 0 and xt.shape()[-1] % 32 == 0:
        xt = xt.to(ttl.tensor.Layout.TILE)
    xt = xt.to(device)

    cost = ttl.tensor.Tensor(cos_cached, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    sint = ttl.tensor.Tensor(sin_cached, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.rotary_embedding(xt, cost, sint, token_idx)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx)

    p, o = comp_pcc(pt_out, tt_got_back)
    logger.info(o)
    assert p
