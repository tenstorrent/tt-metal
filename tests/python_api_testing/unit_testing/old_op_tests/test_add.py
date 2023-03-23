import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from pymetal import ttlib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close

RSUM = ttlib.tensor.ReduceOpMath.SUM
RW = ttlib.tensor.ReduceOpDim.W
RH = ttlib.tensor.ReduceOpDim.H
RHW = ttlib.tensor.ReduceOpDim.HW

# Initialize the device
device = ttlib.device.CreateDevice(ttlib.device.Arch.GRAYSKULL, 0)
ttlib.device.InitializeDevice(device)
host = ttlib.device.GetHost()
ttlib.device.StartDebugPrintServer(device)

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
        xt = ttlib.tensor.Tensor(tilize_to_list(x), [N, C, H, W], ttlib.tensor.DataType.BFLOAT16, ttlib.tensor.Layout.TILE, device)
        yt = ttlib.tensor.Tensor(tilize_to_list(y), [N, C, H, W], ttlib.tensor.DataType.BFLOAT16, ttlib.tensor.Layout.TILE, device)
        if op == "add":
            ref_torch = x+y
            tt_res = ttlib.tensor.add(xt, yt)
        elif op == "sub":
            ref_torch = x-y
            tt_res = ttlib.tensor.sub(xt, yt)
        else:
            ref_torch = x*y
            tt_res = ttlib.tensor.mul(xt, yt)
        tt_host_rm = tt_res.to(host).data()

        pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(shape)
        pyt_got_back_rm = untilize(pyt_got_back_rm)
        print_diff_argmax(pyt_got_back_rm, ref_torch)

ttlib.device.CloseDevice(device)
