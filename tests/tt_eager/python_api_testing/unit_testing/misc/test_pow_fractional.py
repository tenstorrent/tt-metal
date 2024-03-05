# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch

import tt_lib as ttl
from models.utility_functions import comp_pcc
from loguru import logger


def test_pow_fractional_composite(device):
    torch.manual_seed(577215)
    N = 1
    C = 2
    H = 32
    W = 32
    x = torch.randn((N, C, H, W)).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    y = 3 + torch.randn(1, 1, 1, 1).bfloat16().float()
    yt = y
    yt_floor = math.floor(yt)
    yt_trunc = yt - yt_floor
    pow_trunc_log = ttl.tensor.mul_unary(ttl.tensor.log(xt), yt_trunc)
    pow_frac = ttl.tensor.exp(pow_trunc_log)
    xtt = ttl.tensor.mul(ttl.tensor.pow(xt, yt_floor), pow_frac)
    assert list(xtt.get_legacy_shape()) == [N, C, H, W]
    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    xtt_fp = ttl.tensor.pow(xt, yt)
    fp_tt_got_back = xtt_fp.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_ref = torch.pow(x, y)

    passing, out = comp_pcc(pt_ref, tt_got_back)
    logger.info(out)
    assert passing

    passing, out = comp_pcc(pt_ref, fp_tt_got_back)
    logger.info(out)
    assert passing
