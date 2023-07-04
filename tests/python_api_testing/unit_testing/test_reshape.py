import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib as ttl
from python_api_testing.models.utility_functions import print_diff_argmax


def test_tile_major_reshape():
    torch.manual_seed(0)
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    N = 3
    C = 5
    H = 64
    W = 96
    x = torch.randn((N, C, H, W)).to(torch.bfloat16).to(torch.float32)

    xtt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device)
    )
    xtt = ttl.tensor.reshape(xtt, 5, 3, 96, 64)
    assert xtt.shape() == [5, 3, 96, 64]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([5, 3, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, 5, 64, 96)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, -1, 5, 96, 64)
    assert xtt.shape() == [3, 5, 96, 64]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, -1, 64, 96)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, 5, -1, 64)
    assert xtt.shape() == [3, 5, 96, 64]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, 5, 64, -1)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, 5, 32, -1)
    assert xtt.shape() == [3, 5, 32, 96 * 2]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 32, 96 * 2])
    assert torch.equal(x, tt_got_back)

    print("reshape() max absdiff=")
    print_diff_argmax(tt_got_back, x)

    del xtt

    ttl.device.CloseDevice(device)


def test_row_major_reshape():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    # Power of 2 reshape
    N = 1
    C = 1
    H = 128
    W = 128
    x = torch.rand(N * C * H * W).reshape(N, C, H, W).bfloat16().float()
    xtt = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device,
    )

    reshaped = ttl.tensor.reshape(xtt, 1, 128, 2, 64)
    reshaped = torch.Tensor(reshaped.to(host).data()).reshape(reshaped.shape())
    torch_reshaped = torch.Tensor(x).reshape(1, 128, 2, 64)
    assert torch.equal(torch_reshaped, reshaped)
    ttl.device.CloseDevice(device)
