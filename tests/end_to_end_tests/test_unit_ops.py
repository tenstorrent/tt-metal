import pytest
import torch

import tt_lib


@pytest.mark.eager_package_silicon
def test_tile_major_reshape_sweep(reset_seeds, first_grayskull_device):
    device = first_grayskull_device
    tt_lib.device.InitializeDevice(device)

    N = 3
    C = 5
    H = 64
    W = 96
    x = torch.randn((N, C, H, W)).to(torch.bfloat16).to(torch.float32)

    xtt = (
         tt_lib.tensor.Tensor(x.contiguous().to(torch.bfloat16))
        .to(tt_lib.tensor.Layout.TILE)
        .to(device)
    )
    xtt = tt_lib.tensor.reshape(xtt, 5, 3, 96, 64)
    assert xtt.shape() == [5, 3, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = torch.Tensor(xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([5, 3, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = tt_lib.tensor.reshape(xtt, 3, 5, 64, 96)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = torch.Tensor(xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = tt_lib.tensor.reshape(xtt, -1, 5, 96, 64)
    assert xtt.shape() == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = torch.Tensor(xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = tt_lib.tensor.reshape(xtt, 3, -1, 64, 96)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = torch.Tensor(xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = tt_lib.tensor.reshape(xtt, 3, 5, -1, 64)
    assert xtt.shape() == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = torch.Tensor(xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 96, 64])
    assert torch.equal(x, tt_got_back)

    xtt = tt_lib.tensor.reshape(xtt, 3, 5, 64, -1)
    assert xtt.shape() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = torch.Tensor(xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 64, 96])
    assert torch.equal(x, tt_got_back)

    xtt = tt_lib.tensor.reshape(xtt, 3, 5, 32, -1)
    assert xtt.shape() == [3, 5, 32, 96 * 2]
    xtt_host = xtt.cpu()
    tt_got_back = torch.Tensor(xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).data()).reshape(
        xtt_host.shape()
    )
    x = x.reshape([3, 5, 32, 96 * 2])
    assert torch.equal(x, tt_got_back)

    del xtt

    tt_lib.device.CloseDevice(device)
