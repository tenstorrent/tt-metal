# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@pytest.mark.parametrize("shape", [[1, 1, 128, 64], [1, 71, 128, 64]])
def test_rotate_half(shape, device):
    x = torch.randn(shape).bfloat16().float()

    xt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.Layout.TILE).to(device)
    xtt = ttnn.experimental.rotate_half(xt)

    tt_got_back = xtt.cpu().to(ttnn.Layout.ROW_MAJOR).to_torch()

    pt_out = rotate_half(x)

    eq = torch.equal(tt_got_back, pt_out)
    assert eq


@pytest.mark.parametrize("shape", [[1, 1, 128, 64], [1, 71, 128, 64]])
def test_rotate_half_program_cache(shape, device):
    spacers, input_addresses = [], set()
    num_entries_before = device.num_program_cache_entries()
    for i in range(3):
        # A growing live allocation moves every tensor below to a new address on each cache hit, and fresh
        # data per iteration makes a stale address show up as a mismatch.
        spacers.append(ttnn.Tensor(torch.zeros(1, 1, 32, 32 * (i + 1)), ttnn.bfloat16).to(ttnn.Layout.TILE).to(device))
        x = torch.randn(shape).bfloat16().float()
        xt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.Layout.TILE).to(device)
        input_addresses.add(xt.buffer_address())
        tt_got_back = ttnn.experimental.rotate_half(xt).cpu().to(ttnn.Layout.ROW_MAJOR).to_torch()

        assert torch.equal(tt_got_back, rotate_half(x))

    assert len(input_addresses) > 1
    assert device.num_program_cache_entries() - num_entries_before == 1


@pytest.mark.parametrize("shape", [[1, 1, 64, 64]])
def test_rotate_half_row_major(shape, device):
    """Test rotate_half with row-major layout inputs."""
    x = torch.randn(shape).bfloat16().float()

    xt = ttnn.Tensor(x, ttnn.bfloat16)
    assert xt.get_layout() == ttnn.ROW_MAJOR_LAYOUT, "Test expects input to be in ROW_MAJOR layout"

    xt = xt.to(device)
    xtt = ttnn.experimental.rotate_half(xt)

    tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    pt_out = rotate_half(x)

    eq = torch.equal(tt_got_back, pt_out)
    assert eq
