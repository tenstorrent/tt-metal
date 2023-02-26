import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from gpai import gpai
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
gpai.device.InitializeDevice(device)
host = gpai.device.GetHost()

if __name__ == "__main__":
    N = 3
    C = 32*2
    H = 32*4
    W = 32*3
    x = torch.randn((N,C,H,W))
    xp = pad_weight(x)

    xt = gpai.tensor.Tensor(tilize_to_list(xp), [N, C, H, W], gpai.tensor.DataFormat.FLOAT32, gpai.tensor.Layout.TILE, device)
    xtt = gpai.tensor.transpose_hc(xt)
    assert(xtt.shape() == [N,H,C,W])

    xtt_data = xtt.to(host).data()
    tt_got_back = torch.Tensor(xtt_data).reshape((N,H,C,W))
    tt_got_back = untilize(tt_got_back)

    print("reshape() max absdiff=")
    transposed_ref = x.permute(0, 2, 1, 3)
    print_diff_argmax(tt_got_back, transposed_ref)

gpai.device.CloseDevice(device)
