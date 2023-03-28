import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from libs import tt_lib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close

RSUM = tt_lib.tensor.ReduceOpMath.SUM
RW = tt_lib.tensor.ReduceOpDim.W
RH = tt_lib.tensor.ReduceOpDim.H
RHW = tt_lib.tensor.ReduceOpDim.HW

# Initialize the device
device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
tt_lib.device.InitializeDevice(device)
host = tt_lib.device.GetHost()
tt_lib.device.StartDebugPrintServer(device)

if __name__ == "__main__":
    N = 1
    C = 1
    H = 32
    W = 32
    shape = [N,C,H,W]
    torch.manual_seed(123)
    for op in ["add", "sub", "mul"]:
        x = (torch.randn((N,C,H,W))+0.01).to(torch.bfloat16).to(torch.float32)
        y = (torch.randn((N,C,H,W))+0.01).to(torch.bfloat16).to(torch.float32)
        xt = tt_lib.tensor.Tensor(tilize_to_list(x), [N, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
        yt = tt_lib.tensor.Tensor(tilize_to_list(y), [N, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
        if op == "add":
            ref_torch = x+y
            tt_res = tt_lib.tensor.add(xt, yt)
        elif op == "sub":
            ref_torch = x-y
            tt_res = tt_lib.tensor.sub(xt, yt)
        else:
            ref_torch = x*y
            tt_res = tt_lib.tensor.mul(xt, yt)
        tt_host_rm = tt_res.to(host).data()

        pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(shape)
        pyt_got_back_rm = untilize(pyt_got_back_rm)
        print_diff_argmax(pyt_got_back_rm, ref_torch)

tt_lib.device.CloseDevice(device)
