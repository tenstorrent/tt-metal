import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from pymetal import ttmetal as ttm
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
ttm.device.InitializeDevice(device)
host = ttm.device.GetHost()

if __name__ == "__main__":
    N = 3
    C = 32*2
    H = 32*4
    W = 32*3
    x = torch.randn((N,C,H,W))
    xp = pad_weight(x)

    xt = ttm.tensor.Tensor(tilize_to_list(xp), [N, C, H, W], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    xtt = ttm.tensor.transpose_hc(xt)
    assert(xtt.shape() == [N,H,C,W])

    xtt_data = xtt.to(host).data()
    tt_got_back = torch.Tensor(xtt_data).reshape((N,H,C,W))
    tt_got_back = untilize(tt_got_back)

    print("reshape() max absdiff=")
    transposed_ref = x.permute(0, 2, 1, 3)
    print_diff_argmax(tt_got_back, transposed_ref)

ttm.device.CloseDevice(device)
