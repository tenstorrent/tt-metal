import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from libs import tt_lib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight
from python_api_testing.models.utility_functions import tt2torch

# Initialize the device
device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
tt_lib.device.InitializeDevice(device)
host = tt_lib.device.GetHost()

if __name__ == "__main__":
    N = 2
    C = 3
    H = 64
    W = 96
    fillH = 33
    fillW = 31

    x = torch.zeros((N,C,H,W))
    x[:, :, :fillH, :fillW] = 1.0
    xp = pad_weight(x)

    xt = tt_lib.tensor.Tensor(tilize_to_list(xp), [N, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    xtt = tt_lib.tensor.fill_ones_rm(N, C, H, W, fillH, fillW, xt)
    assert(xtt.shape() == [N,C,H,W])

    xtt_data = xtt.to(host).data()
    tt_got_back = torch.Tensor(xtt_data).reshape((N,C,H,W))

    #x[1,1,2,2] = 2.0
    print("reshape() max absdiff=")
    print_diff_argmax(tt_got_back, xp)

tt_lib.device.CloseDevice(device)
