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

BCW = tt_lib.tensor.BcastOpDim.W
BCH = tt_lib.tensor.BcastOpDim.H
BCHW = tt_lib.tensor.BcastOpDim.HW
BCMUL = tt_lib.tensor.BcastOpMath.MUL
BCSUB = tt_lib.tensor.BcastOpMath.SUB
BCADD = tt_lib.tensor.BcastOpMath.ADD

if __name__ == "__main__":
    torch.manual_seed(123)
    N = 3
    C = 5
    H = 32*2
    W = 32*5
    x = torch.randn((N,C,H,W))
    bw = torch.randn((1,1,H,1))
    bh = torch.randn((1,1,1,W))
    bhw = torch.randn((1,1,1,1))

    for btorch, btype in zip([bw, bh, bhw], [BCW, BCH, BCHW]):
        xt = tt_lib.tensor.Tensor(tilize_to_list(x), [N, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
        btshape = list(btorch.shape)
        bpadded = pad_weight(btorch) # need to zero-pad the bcast tensors to 32 in H,W
        bt = tt_lib.tensor.Tensor(tilize_to_list(bpadded), list(bpadded.shape), tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)

        tt_res = tt_lib.tensor.bcast(xt, bt, BCADD, btype)
        # test that reading back from row major is about the same (+/- BF16 conversion)
        tt_host_rm = tt_res.to(host).data()
        pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape((N,C,H,W))
        pyt_got_back_rm = untilize(pyt_got_back_rm)

        print("row_major read back max absdiff=")

        ref = x + btorch
        absdiff = print_diff_argmax(pyt_got_back_rm, ref)
        assert(absdiff < 0.05)

tt_lib.device.CloseDevice(device)
