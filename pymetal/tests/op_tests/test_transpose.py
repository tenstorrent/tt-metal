import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

import ttmetal
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = ttmetal.device.CreateDevice(ttmetal.device.Arch.GRAYSKULL, 0)
ttmetal.device.InitializeDevice(device)
host = ttmetal.device.GetHost()
ttmetal.device.StartDebugPrintServer(device)

RSUM = ttmetal.tensor.ReduceOpMath.SUM
RW = ttmetal.tensor.ReduceOpDim.W
RH = ttmetal.tensor.ReduceOpDim.H
RHW = ttmetal.tensor.ReduceOpDim.HW

if __name__ == "__main__":
    N = 7
    C = 5
    H = 32*2
    W = 32*3
    torch.manual_seed(123)
    x = (torch.randn((N,C,H,W))+0.05).to(torch.bfloat16).to(torch.float32)

    xt = ttmetal.tensor.Tensor(tilize_to_list(x), [N, C, H, W], ttmetal.tensor.DataType.BFLOAT16, ttmetal.tensor.Layout.TILE, device)
    tt_res = ttmetal.tensor.transpose(xt)
    assert(tt_res.shape() == [N,C,W,H])
    tt_host_rm = tt_res.to(host).data()

    pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(tt_res.shape())
    pyt_got_back_rm = untilize(pyt_got_back_rm)

    ref_transposed = x.permute(0, 1, 3, 2)
    allok = torch.all(torch.isclose(pyt_got_back_rm, ref_transposed, rtol=0.01, atol=0.01))
    if not allok:
        print_diff_argmax(pyt_got_back_rm, ref_transposed)

    assert(allok)

ttmetal.device.CloseDevice(device)
