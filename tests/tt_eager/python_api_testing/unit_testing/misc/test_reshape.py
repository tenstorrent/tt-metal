# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import ttnn


def test_tile_major_reshape(device):
    torch.manual_seed(0)

    N = 3
    C = 5
    H = 64
    W = 96
    x = torch.randn((N, C, H, W), dtype=torch.float32).bfloat16().float()

    xtt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    xtt = ttnn.reshape_on_device(xtt, 5, 3, 96, 64)
    assert list(xtt.padded_shape) == [5, 3, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([5, 3, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, 5, 64, 96)
    assert list(xtt.padded_shape) == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, -1, 5, 96, 64)
    assert list(xtt.padded_shape) == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, -1, 64, 96)
    assert list(xtt.padded_shape) == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, 5, -1, 64)
    assert list(xtt.padded_shape) == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, 5, 64, -1)
    assert list(xtt.padded_shape) == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = ttnn.reshape_on_device(xtt, 3, 5, 32, -1)
    assert list(xtt.padded_shape) == [3, 5, 32, 96 * 2]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x = x.reshape([3, 5, 32, 96 * 2])
    eq = torch.equal(x, tt_got_back)
    assert eq


def test_row_major_reshape(device):
    # Power of 2 reshape
    N = 1
    C = 1
    H = 128
    W = 128
    x = torch.rand(N * C * H * W).reshape(N, C, H, W).bfloat16().float()
    xtt = ttnn.Tensor(x, ttnn.bfloat16).to(device)

    reshaped = ttnn.reshape_on_device(xtt, 1, 128, 2, 64)
    reshaped = reshaped.cpu().to_torch()
    torch_reshaped = torch.Tensor(x).reshape(1, 128, 2, 64)
    eq = torch.equal(torch_reshaped, reshaped)
    assert eq


def test_tile_major_reshape_var(device):
    torch.manual_seed(0)

    N = 1
    C = 1
    H = 32
    W = 96
    final_shape = [C, N, W, H]
    x = torch.randn((N, C, H, W), dtype=torch.bfloat16)

    xtt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    xtt = ttnn.reshape_on_device(xtt, C, N, W, H)
    assert list(xtt.padded_shape) == final_shape

    tt_got_back = ttnn.to_torch(xtt)
    x = x.reshape(final_shape)
    eq = torch.equal(x, tt_got_back)

    assert eq
