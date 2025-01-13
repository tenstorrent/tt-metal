# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.utility_functions import print_diff_argmax
from models.utility_functions import is_wormhole_b0


def test_fill_rm(device):
    N = 2
    C = 3
    H = 64
    W = 96

    fillH = 33
    fillW = 31

    if is_wormhole_b0():
        N, C, H, W = [1, 1, 32, 32]
        fillH = 31
        fillW = 31

    x = torch.zeros((N, C, H, W))
    xp = torch.clone(x)
    xp[:, :, :fillH, :fillW] = 1.0

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
    xtt = ttnn.fill_ones_rm(N, C, H, W, fillH, fillW, xt)
    assert list(xtt.shape.with_tile_padding()) == [N, C, H, W]

    tt_got_back = xtt.cpu().to_torch()

    # x[1,1,2,2] = 2.0
    print("reshape() max absdiff=")
    print_diff_argmax(tt_got_back, xp)
    eq = torch.equal(tt_got_back, xp)
    assert eq
