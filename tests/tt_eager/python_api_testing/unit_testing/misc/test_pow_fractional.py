# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch

import ttnn
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
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device)
    )

    y = 3 + torch.randn(1, 1, 1, 1).bfloat16().float()
    yt = y
    yt_floor = math.floor(yt)
    yt_trunc = yt - yt_floor
    pow_trunc_log = ttnn.multiply(ttnn.log(xt), yt_trunc)
    pow_frac = ttnn.exp(pow_trunc_log)
    xtt = ttnn.mul(ttnn.pow(xt, yt_floor), pow_frac)
    assert list(xtt.get_legacy_shape()) == [N, C, H, W]
    tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    xtt_fp = ttnn.pow(xt, yt.item())
    fp_tt_got_back = xtt_fp.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    pt_ref = torch.pow(x, y)

    passing, out = comp_pcc(pt_ref, tt_got_back)
    logger.info(out)
    assert passing

    passing, out = comp_pcc(pt_ref, fp_tt_got_back)
    logger.info(out)
    assert passing
