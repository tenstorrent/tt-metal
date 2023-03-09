import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from ttm import gttm
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax
from python_api_testing.models.utility_functions import tt2torch, tt2torch_rm


def softmax(x, stable=False):
    RMAX = ttm.tensor.ReduceOpMath.MAX
    RSUM = ttm.tensor.ReduceOpMath.SUM
    RW = ttm.tensor.ReduceOpDim.W
    BCW = ttm.tensor.BcastOpDim.W
    BCMUL = ttm.tensor.BcastOpMath.MUL
    BCSUB = ttm.tensor.BcastOpMath.MUL

    if stable:
        sumsW = ttm.tensor.reduce(x, RMAX, RW, 1.0)
        z = ttm.tensor.bcast(x, sumsW, BCSUB, BCW) # x-max(x)
    else:
        z = x
    numerator = ttm.tensor.exp(z) # exp(z)
    denom1 = ttm.tensor.reduce(numerator, RSUM, RW, 1.0) # torch.sum(x, 3)
    denom = ttm.tensor.recip(denom1)
    output = ttm.tensor.bcast(numerator, denom, BCMUL, BCW)

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
    device = ttm.device.CreateDevice(gttmdevice.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    H, W = 64, 96
    torch.manual_seed(123)

    x = torch.randn((1,1,H,W))
    ref_sm = ref_stable_softmax(x)

    x_t = tilize_to_list(x)
    t0 = ttm.tensor.Tensor(x_t, [1, 1, H, W], gttmtensor.DataType.BFLOAT16, gpttmensor.Layout.TILE, device)
    func = softmax
    t1 = func(t0)
    t2_data = t1.to(host).data()

    tt_got_back = torch.Tensor(t2_data).reshape((1,1,H,W))
    tt_got_back = untilize(tt_got_back)

    print("Max diff=")
    print_diff_argmax(tt_got_back, ref_sm)

    print(t2_data)
    ttm.device.CloseDevice(device)
