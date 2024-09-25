# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn
from models.utility_functions import comp_pcc
from loguru import logger
from models.utility_functions import is_wormhole_b0


def test_eltwise_unary_chain(device):
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

    xtt = ttnn.unary_chain(
        xt,
        [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2),
        ],
    )
    assert list(xtt.shape.with_tile_padding()) == [N, C, H, W]

    tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    pt_ref = torch.pow(torch.exp(torch.nn.functional.relu(x)), 2)

    passing, out = comp_pcc(pt_ref, tt_got_back)
    logger.info(out)
    assert passing


def test_eltwise_binary_fused(device):
    N = 1
    C = 2
    H = 32
    W = 32
    x = torch.randn((N, C, H, W)).bfloat16().float()
    y = torch.randn((N, C, H, W)).bfloat16().float()

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
    yt = (
        ttnn.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        .to(ttnn.TILE_LAYOUT)
        .to(device)
    )

    xtt = ttnn.add(
        xt,
        yt,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2)],
    )
    assert list(xtt.shape.with_tile_padding()) == [N, C, H, W]

    tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    pt_ref = torch.pow(torch.nn.functional.relu(x + y), 2)

    passing, out = comp_pcc(pt_ref, tt_got_back)
    logger.info(out)
    assert passing
