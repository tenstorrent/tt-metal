# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import tt_lib


@pytest.mark.eager_package_silicon
def test_tile_major_reshape_sweep(reset_seeds, first_grayskull_device):
    device = first_grayskull_device

    N = 3
    C = 5
    H = 64
    W = 96
    x = torch.randn((N, C, H, W)).to(torch.bfloat16).to(torch.float32)

    xtt = tt_lib.tensor.Tensor(x, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    xtt = tt_lib.tensor.reshape(xtt, 5, 3, 96, 64)
    assert xtt.get_legacy_shape() == [5, 3, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([5, 3, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = tt_lib.tensor.reshape(xtt, 3, 5, 64, 96)
    assert xtt.get_legacy_shape() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = tt_lib.tensor.reshape(xtt, -1, 5, 96, 64)
    assert xtt.get_legacy_shape() == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = tt_lib.tensor.reshape(xtt, 3, -1, 64, 96)
    assert xtt.get_legacy_shape() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = tt_lib.tensor.reshape(xtt, 3, 5, -1, 64)
    assert xtt.get_legacy_shape() == [3, 5, 96, 64]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 96, 64])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = tt_lib.tensor.reshape(xtt, 3, 5, 64, -1)
    assert xtt.get_legacy_shape() == [3, 5, 64, 96]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 64, 96])
    eq = torch.equal(x, tt_got_back)
    assert eq

    xtt = tt_lib.tensor.reshape(xtt, 3, 5, 32, -1)
    assert xtt.get_legacy_shape() == [3, 5, 32, 96 * 2]
    xtt_host = xtt.cpu()
    tt_got_back = xtt_host.to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    x = x.reshape([3, 5, 32, 96 * 2])
    eq = torch.equal(x, tt_got_back)
    assert eq

    del xtt

    tt_lib.device.CloseDevice(device)
