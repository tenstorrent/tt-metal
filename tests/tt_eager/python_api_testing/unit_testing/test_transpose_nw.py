# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch
from random import randint

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0, skip_for_wormhole_b0

@skip_for_wormhole_b0
def test_transpose_nw(device):
    for i in range(0,200):
        N = 32 * randint(1,3)
        C = 32 * randint(1,2)
        H = 32 * randint(1,4)
        W = 32 * randint(1,3)
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
        xtt = ttl.tensor.transpose_nw(xt)
        assert xtt.shape() == [W, C, H, N]

        tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        print("reshape() max absdiff=")
        transposed_ref = torch.transpose(x, dim0=0, dim1=-1)
        print_diff_argmax(tt_got_back, transposed_ref)

        assert torch.equal(tt_got_back, transposed_ref)
