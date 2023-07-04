from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib
from python_api_testing.models.utility_functions import print_diff_argmax


def test_fill_rm():
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()

    N = 2
    C = 3
    H = 64
    W = 96
    fillH = 33
    fillW = 31

    x = torch.zeros((N, C, H, W))
    xp = torch.clone(x)
    xp[:, :, :fillH, :fillW] = 1.0

    xt = (
        tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        .to(tt_lib.tensor.Layout.TILE)
        .to(device)
    )
    xtt = tt_lib.tensor.fill_ones_rm(N, C, H, W, fillH, fillW, xt)
    assert xtt.shape() == [N, C, H, W]

    xtt_data = xtt.to(host).data()
    tt_got_back = torch.Tensor(xtt_data).reshape((N, C, H, W))

    # x[1,1,2,2] = 2.0
    print("reshape() max absdiff=")
    print_diff_argmax(tt_got_back, xp)
    assert torch.equal(tt_got_back, xp)

    del xtt

    tt_lib.device.CloseDevice(device)
