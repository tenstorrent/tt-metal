# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch

from .. import tensor, device
from ..utils import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax


def softmax(x: tensor.Tensor, stable=False):
    """
    Performs Softmax on a ``tt_lib.tensor.Tensor``.
    """

    RMAX = tensor.ReduceOpMath.MAX
    RSUM = tensor.ReduceOpMath.SUM
    RW = tensor.ReduceOpDim.W
    BCW = tensor.BcastOpDim.W
    BCMUL = tensor.BcastOpMath.MUL
    BCSUB = tensor.BcastOpMath.SUB

    if stable:
        sumsW = tensor.reduce(x, RMAX, RW, 1.0)
        z = tensor.bcast(x, sumsW, BCSUB, BCW)  # x-max(x)
    else:
        z = x
    numerator = tensor.exp(z)  # exp(z)
    denom1 = tensor.reduce(numerator, RSUM, RW, 1.0)  # torch.sum(x, 3)
    denom = tensor.recip(denom1)
    output = tensor.bcast(numerator, denom, BCMUL, BCW)

    return output


def ref_stable_softmax(x):
    """
    z = x - torch.max(x, dim=3, keepdim=True)[0]
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, 3)
    denom1 = torch.reciprocal(denominator)
    softmax = numerator*denom1
    """
    softmax = torch.nn.Softmax(3)(x)

    return softmax


if __name__ == "__main__":
    device = device.CreateDevice(0)

    H, W = 64, 96
    torch.manual_seed(123)

    x = torch.randn((1, 1, H, W))
    ref_sm = ref_stable_softmax(x)

    x_t = tilize_to_list(x)
    t0 = tensor.Tensor(x_t, [1, 1, H, W], tensor.DataType.BFLOAT16, tensor.Layout.TILE, device)
    func = softmax
    t1 = func(t0)

    tt_got_back = t1.cpu().to_torch()
    tt_got_back = untilize(tt_got_back)

    print("Max diff=")
    print_diff_argmax(tt_got_back, ref_sm)

    device.CloseDevice(device)
