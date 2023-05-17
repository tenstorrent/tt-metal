import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    pad_activation,
    pad_weight,
    tilize,
    untilize,
    tilize_to_list,
    print_diff_argmax,
    pad_weight,
)


def test_transpose_hc():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    x = torch.randn((N, C, H, W))
    xp = pad_weight(x)

    xt = ttl.tensor.Tensor(
        tilize_to_list(xp),
        [N, C, H, W],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )
    xtt = ttl.tensor.transpose_hc(xt)
    assert xtt.shape() == [N, H, C, W]

    xtt_data = xtt.to(host).data()
    tt_got_back = torch.Tensor(xtt_data).reshape((N, H, C, W))
    tt_got_back = untilize(tt_got_back)

    print("reshape() max absdiff=")
    transposed_ref = x.permute(0, 2, 1, 3)
    print_diff_argmax(tt_got_back, transposed_ref)

    del xtt

    ttl.device.CloseDevice(device)
