# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import tt_lib as ttl


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@pytest.mark.parametrize("shape", [[1, 1, 128, 64], [1, 71, 128, 64]])
def test_rotate_half(shape, device):
    x = torch.randn(shape).bfloat16().float()

    xt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttl.tensor.rotate_half(xt)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = rotate_half(x)

    eq = torch.equal(tt_got_back, pt_out)
    assert eq
