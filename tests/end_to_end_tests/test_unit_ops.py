# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.eager_package_silicon
def test_tile_major_reshape_sweep(reset_seeds, first_grayskull_device):
    device = first_grayskull_device

    N = 3
    C = 5
    H = 64
    W = 96
    x = torch.randn((N, C, H, W)).to(torch.bfloat16).to(torch.float32)

    xtt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    xtt = ttnn.reshape_on_device(xtt, 5, 3, 96, 64)
    assert xtt.shape.with_tile_padding() == [5, 3, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([5, 3, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, 5, 64, 96)
    assert xtt.shape.with_tile_padding() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, -1, 5, 96, 64)
    assert xtt.shape.with_tile_padding() == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, -1, 64, 96)
    assert xtt.shape.with_tile_padding() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, 5, -1, 64)
    assert xtt.shape.with_tile_padding() == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, 5, 64, -1)
    assert xtt.shape.with_tile_padding() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, 5, 32, -1)
    assert xtt.shape.with_tile_padding() == [3, 5, 32, 96 * 2]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 32, 96 * 2])
    eq = torch.equal(x, tt_got_back)
    assert eq

    del xtt

    ttnn.device.CloseDevice(device)
