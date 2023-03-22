import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from pymetal import ttlib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = ttlib.device.CreateDevice(ttlib.device.Arch.GRAYSKULL, 0)
ttlib.device.InitializeDevice(device)
host = ttlib.device.GetHost()
ttlib.device.StartDebugPrintServer(device)

RSUM = ttlib.tensor.ReduceOpMath.SUM
RW = ttlib.tensor.ReduceOpDim.W
RH = ttlib.tensor.ReduceOpDim.H
RHW = ttlib.tensor.ReduceOpDim.HW

if __name__ == "__main__":
    N = 7
    C = 5
    H = 32*2
    W = 32*3
    torch.manual_seed(123)
    x = (torch.randn((N,C,H,W))+0.05).to(torch.bfloat16).to(torch.float32)

    xt = ttlib.tensor.Tensor(tilize_to_list(x), [N, C, H, W], ttlib.tensor.DataType.BFLOAT16, ttlib.tensor.Layout.TILE, device)
    tt_res = ttlib.tensor.transpose(xt)
    assert(tt_res.shape() == [N,C,W,H])
    tt_host_rm = tt_res.to(host).data()

    pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(tt_res.shape())
    pyt_got_back_rm = untilize(pyt_got_back_rm)

    ref_transposed = x.permute(0, 1, 3, 2)
    allok = torch.all(torch.isclose(pyt_got_back_rm, ref_transposed, rtol=0.01, atol=0.01))
    if not allok:
        print_diff_argmax(pyt_got_back_rm, ref_transposed)

    assert(allok)

ttlib.device.CloseDevice(device)
