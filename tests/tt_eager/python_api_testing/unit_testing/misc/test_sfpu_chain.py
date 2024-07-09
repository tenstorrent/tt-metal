# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

import tt_lib as ttl
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
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    xtt = ttl.tensor.unary_chain(
        xt,
        [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2),
        ],
    )
    assert list(xtt.get_legacy_shape()) == [N, C, H, W]

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

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
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    yt = (
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )

    xtt = ttnn.add(
        xt,
        yt,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2)],
    )
    assert list(xtt.get_legacy_shape()) == [N, C, H, W]

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_ref = torch.pow(torch.nn.functional.relu(x + y), 2)

    passing, out = comp_pcc(pt_ref, tt_got_back)
    logger.info(out)
    assert passing
