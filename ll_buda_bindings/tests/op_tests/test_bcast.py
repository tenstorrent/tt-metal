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
BCW = _C.tensor.BcastOpDim.W
BCH = _C.tensor.BcastOpDim.H
BCHW = _C.tensor.BcastOpDim.HW
BCMUL = _C.tensor.BcastOpMath.MUL
BCSUB = _C.tensor.BcastOpMath.SUB
BCADD = _C.tensor.BcastOpMath.ADD

if __name__ == "__main__":
    torch.manual_seed(123)
    N = 3
    C = 5
    H = 32*2
    W = 32*5
    x = torch.randn((N,C,H,W))
    bw = torch.randn((N,C,H,1))
    bh = torch.randn((N,C,1,W))
    bhw = torch.randn((N,C,1,1))

    for btorch, btype in zip([bw, bh, bhw], [BCW, BCH, BCHW]):
        xt = _C.tensor.Tensor(tilize_to_list(x), [N, C, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
        btshape = list(btorch.shape)
        bpadded = pad_weight(btorch) # need to zero-pad the bcast tensors to 32 in H,W
        bt = _C.tensor.Tensor(tilize_to_list(bpadded), list(bpadded.shape), _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)

        tt_res = _C.tensor.bcast(xt, bt, BCADD, btype)
        # test that reading back from row major is about the same (+/- BF16 conversion)
        tt_host_rm = tt_res.to(host).data()
        pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape((N,C,H,W))
        pyt_got_back_rm = untilize(pyt_got_back_rm)

        print("row_major read back max absdiff=")

        ref = x + btorch
        absdiff = print_diff_argmax(pyt_got_back_rm, ref)
        assert(absdiff < 0.05)

_C.device.CloseDevice(device)
