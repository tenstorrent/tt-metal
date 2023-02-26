import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from gpai import gpai
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax


def softmax(x, stable=False):
    RMAX = gpai.tensor.ReduceOpMath.MAX
    RSUM = gpai.tensor.ReduceOpMath.SUM
    RW = gpai.tensor.ReduceOpDim.W
    BCW = gpai.tensor.BcastOpDim.W
    BCMUL = gpai.tensor.BcastOpMath.MUL
    BCSUB = gpai.tensor.BcastOpMath.MUL

    if stable:
        sumsW = gpai.tensor.reduce(x, RMAX, RW, 1.0)
        z = gpai.tensor.bcast(x, sumsW, BCSUB, BCW) # x-max(x)
    else:
        z = x
    numerator = gpai.tensor.exp(z) # exp(z)
    denom1 = gpai.tensor.reduce(numerator, RSUM, RW, 1.0) # torch.sum(x, 3)
    denom = gpai.tensor.recip(denom1)
    output = gpai.tensor.bcast(numerator, denom, BCMUL, BCW)

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
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()
    H, W = 64, 96
    torch.manual_seed(123)

    x = torch.randn((1,1,H,W))
    ref_sm = ref_stable_softmax(x)

    x_t = tilize_to_list(x)
    t0 = gpai.tensor.Tensor(x_t, [1, 1, H, W], gpai.tensor.DataFormat.FLOAT32, gpai.tensor.Layout.TILE, device)
    func = softmax
    t1 = func(t0)
    t2_data = t1.to(host).data()

    tt_got_back = torch.Tensor(t2_data).reshape((1,1,H,W))
    tt_got_back = untilize(tt_got_back)

    print("Max diff=")
    print_diff_argmax(tt_got_back, ref_sm)

    print(t2_data)
    gpai.device.CloseDevice(device)
