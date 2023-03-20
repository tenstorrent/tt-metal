import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from pymetal import ttmetal
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = ttmetal.device.CreateDevice(ttmetal.device.Arch.GRAYSKULL, 0)
ttmetal.device.InitializeDevice(device)
host = ttmetal.device.GetHost()
ttmetal.device.StartDebugPrintServer(device)

BCW = ttmetal.tensor.BcastOpDim.W
BCH = ttmetal.tensor.BcastOpDim.H
BCHW = ttmetal.tensor.BcastOpDim.HW
BCMUL = ttmetal.tensor.BcastOpMath.MUL
BCSUB = ttmetal.tensor.BcastOpMath.SUB
BCADD = ttmetal.tensor.BcastOpMath.ADD

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
        xt = ttmetal.tensor.Tensor(tilize_to_list(x), [N, C, H, W], ttmetal.tensor.DataType.BFLOAT16, ttmetal.tensor.Layout.TILE, device)
        btshape = list(btorch.shape)
        bpadded = pad_weight(btorch) # need to zero-pad the bcast tensors to 32 in H,W
        bt = ttmetal.tensor.Tensor(tilize_to_list(bpadded), list(bpadded.shape), ttmetal.tensor.DataType.BFLOAT16, ttmetal.tensor.Layout.TILE, device)

        tt_res = ttmetal.tensor.bcast(xt, bt, BCADD, btype)
        # test that reading back from row major is about the same (+/- BF16 conversion)
        tt_host_rm = tt_res.to(host).data()
        pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape((N,C,H,W))
        pyt_got_back_rm = untilize(pyt_got_back_rm)

        print("row_major read back max absdiff=")

        ref = x + btorch
        absdiff = print_diff_argmax(pyt_got_back_rm, ref)
        assert(absdiff < 0.05)

ttmetal.device.CloseDevice(device)
