# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch

import ttnn
from models.utility_functions import comp_pcc
from loguru import logger


def test_pow_fractional_composite(device):
    torch.manual_seed(577215)
    range = 10
    N = 1
    C = 1
    H = 1024
    W = 1024

    x = torch.rand((N, C, H, W)).bfloat16().float() * range
    xt = (
        ttnn.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device)
    )

    y = torch.rand((N, C, H, W)).bfloat16().float() * range
    yt = (
        ttnn.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device)
    )

    print(xt)
    print(yt)

    # yt_floor = math.floor(yt)
    # yt_trunc = yt - yt_floor
    # pow_trunc_log = ttnn.multiply(ttnn.log(xt), yt_trunc)
    # pow_frac = ttnn.exp(pow_trunc_log)
    # xtt = ttnn.mul(ttnn.pow(xt, yt_floor), pow_frac)
    # assert list(xtt.shape.with_tile_padding()) == [N, C, H, W]
    # tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    # xtt_fp = ttnn.pow(xt, yt.item())
    # fp_tt_got_back = xtt_fp.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    # pt_ref = torch.pow(x, y)

    # passing, out = comp_pcc(pt_ref, tt_got_back)
    # logger.info(out)
    # assert passing

    # passing, out = comp_pcc(pt_ref, fp_tt_got_back)
    # logger.info(out)
    # assert passing

    # res = ttnn.exp(ttnn.multiply(ttnn.log(xt), yt))
    # fp_tt_got_back2 = res.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    # passing, out = comp_pcc(pt_ref, fp_tt_got_back2)
    # logger.info(out)
    # assert passing

    # pt_ref = torch.pow(x, y)

    # res = ttnn.exp(ttnn.multiply(ttnn.log(xt), yt))
    # fp_tt_got_back2 = res.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    # passing, out = comp_pcc(pt_ref, fp_tt_got_back2)
    # logger.info(out)
    # assert passing

    res = ttnn.add(xt, yt)
    res = res.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    pt_ref2 = torch.exp(torch.multiply(y, torch.log(x)))
    pt_ref3 = torch.pow(x, y)

    passing, out = comp_pcc(pt_ref2, res)
    logger.info(out)
    passing, out = comp_pcc(pt_ref3, res)
    logger.info(out)
    print(res)
    print(pt_ref2)
    assert passing
