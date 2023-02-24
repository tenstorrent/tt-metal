import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

import ll_buda_bindings.ll_buda_bindings._C as _C
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
_C.device.InitializeDevice(device)
host = _C.device.GetHost()
_C.device.StartDebugPrintServer(device)

RSUM = _C.tensor.ReduceOpMath.SUM
RW = _C.tensor.ReduceOpDim.W
RH = _C.tensor.ReduceOpDim.H
RHW = _C.tensor.ReduceOpDim.HW

if __name__ == "__main__":
    N = 7
    C = 5
    H = 32*2
    W = 32*3
    torch.manual_seed(123)
    x = (torch.randn((N,C,H,W))+0.05).to(torch.bfloat16).to(torch.float32)

    xt = _C.tensor.Tensor(tilize_to_list(x), [N, C, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    tt_res = _C.tensor.transpose(xt)
    assert(tt_res.shape() == [N,C,W,H])
    tt_host_rm = tt_res.to(host).data()

    pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(tt_res.shape())
    pyt_got_back_rm = untilize(pyt_got_back_rm)

    ref_transposed = x.permute(0, 1, 3, 2)
    allok = torch.all(torch.isclose(pyt_got_back_rm, ref_transposed, rtol=0.01, atol=0.01))
    if not allok:
        print_diff_argmax(pyt_got_back_rm, ref_transposed)

    assert(allok)

_C.device.CloseDevice(device)
