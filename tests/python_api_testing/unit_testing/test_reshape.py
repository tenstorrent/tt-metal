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
    xp = pad_weight(x)

    xtt = ttl.tensor.Tensor(
        tilize_to_list(xp),
        [N, C, H, W],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )
    xtt = ttl.tensor.reshape(xtt, 5, 3, 96, 64)
    assert xtt.shape() == [5, 3, 96, 64]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.data()).reshape(xtt_host.shape())
    tt_got_back = untilize(tt_got_back)
    x = x.reshape([5, 3, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, 5, 64, 96)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.data()).reshape(xtt_host.shape())
    tt_got_back = untilize(tt_got_back)
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, -1, 5, 96, 64)
    assert xtt.shape() == [3, 5, 96, 64]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.data()).reshape(xtt_host.shape())
    tt_got_back = untilize(tt_got_back)
    x = x.reshape([3, 5, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, -1, 64, 96)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.data()).reshape(xtt_host.shape())
    tt_got_back = untilize(tt_got_back)
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, 5, -1, 64)
    assert xtt.shape() == [3, 5, 96, 64]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.data()).reshape(xtt_host.shape())
    tt_got_back = untilize(tt_got_back)
    x = x.reshape([3, 5, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, 5, 64, -1)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.data()).reshape(xtt_host.shape())
    tt_got_back = untilize(tt_got_back)
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = ttl.tensor.reshape(xtt, 3, 5, 32, -1)
    assert xtt.shape() == [3, 5, 32, 96 * 2]
    xtt_host = xtt.to(host)
    tt_got_back = torch.Tensor(xtt_host.data()).reshape(xtt_host.shape())
    tt_got_back = untilize(tt_got_back)
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
    x = torch.rand(N * C * H * W).reshape(N, C, H, W).float()
    xp = pad_activation(x).view(-1).tolist()
    xtt = ttl.tensor.Tensor(
        xp,
        [N, C, H, W],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device,
    )

    reshaped = ttl.tensor.reshape(xtt, 1, 128, 2, 64)
    reshaped = torch.Tensor(reshaped.to(host).data()).reshape(reshaped.shape())
    torch_reshaped = torch.Tensor(x).reshape(1, 128, 2, 64)
    assert (abs(torch_reshaped - reshaped) < 0.02).all().item(), "Failure"
    ttl.device.CloseDevice(device)
