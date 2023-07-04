from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib as ttl
from python_api_testing.models.utility_functions import print_diff_argmax


def test_transpose_hc():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    N = 3
    C = 32 * 2
    H = 32 * 4
    W = 32 * 3
    x = torch.randn((N, C, H, W)).bfloat16().float()

    xt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    xtt = ttl.tensor.transpose_hc(xt)
    assert xtt.shape() == [N, H, C, W]

    xtt_data = xtt.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()
    tt_got_back = torch.Tensor(xtt_data).reshape(xtt.shape())

    print("reshape() max absdiff=")
    transposed_ref = x.permute(0, 2, 1, 3)
    print_diff_argmax(tt_got_back, transposed_ref)

    assert torch.equal(tt_got_back, transposed_ref)

    del xtt

    ttl.device.CloseDevice(device)
