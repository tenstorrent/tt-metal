import torch
import pytest
from loguru import logger

import tt_lib as ttl

from tt_models.utility_functions import comp_pcc, pad_by_zero


@pytest.mark.parametrize("shape", [[1, 1, 32, 128]])
def test_softmax(shape, device):
    x = torch.randn(shape).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16)
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    xtt = ttl.operations.primary.softmax_in_place(xt)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = torch.nn.functional.softmax(x, dim=-1)

    passing, output = comp_pcc(pt_out, tt_got_back)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("shape", [[1, 1, 32, 128]])
def test_layernorm(shape, device):
    x = torch.randn(shape).bfloat16().float()
    gamma = torch.randn([shape[-1]]).bfloat16().float()
    beta = torch.randn([shape[-1]]).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16)
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    gammat = pad_by_zero(gamma, device)[0]
    betat = pad_by_zero(beta, device)[0]

    xtt = ttl.tensor.layernorm(xt, 1e-5, gammat, betat)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, 1e-5)

    passing, output = comp_pcc(pt_out, tt_got_back)
    logger.info(output)
    assert passing
