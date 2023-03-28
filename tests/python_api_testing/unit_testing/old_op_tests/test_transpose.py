import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from libs import tt_lib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
tt_lib.device.InitializeDevice(device)
host = tt_lib.device.GetHost()
tt_lib.device.StartDebugPrintServer(device)

RSUM = tt_lib.tensor.ReduceOpMath.SUM
RW = tt_lib.tensor.ReduceOpDim.W
RH = tt_lib.tensor.ReduceOpDim.H
RHW = tt_lib.tensor.ReduceOpDim.HW

if __name__ == "__main__":
    N = 7
    C = 5
    H = 32*2
    W = 32*3
    torch.manual_seed(123)
    x = (torch.randn((N,C,H,W))+0.05).to(torch.bfloat16).to(torch.float32)

    xt = tt_lib.tensor.Tensor(tilize_to_list(x), [N, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    tt_res = tt_lib.tensor.transpose(xt)
    assert(tt_res.shape() == [N,C,W,H])
    tt_host_rm = tt_res.to(host).data()

    pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(tt_res.shape())
    pyt_got_back_rm = untilize(pyt_got_back_rm)

    ref_transposed = x.permute(0, 1, 3, 2)
    allok = torch.all(torch.isclose(pyt_got_back_rm, ref_transposed, rtol=0.01, atol=0.01))
    if not allok:
        print_diff_argmax(pyt_got_back_rm, ref_transposed)

    assert(allok)

tt_lib.device.CloseDevice(device)
