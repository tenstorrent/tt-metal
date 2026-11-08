# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

LAYOUTS = [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT]


@pytest.mark.parametrize(
    "shape, perm",
    [
        ((0, 29), (1, 0)),
        ((0, 32), (1, 0)),
        ((4, 0), (1, 0)),
        ((2, 3, 0, 32), (0, 1, 3, 2)),
        ((2, 0, 32, 29), (0, 1, 3, 2)),
    ],
)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_permute_and_transpose_empty(device, shape, perm, layout):
    # An empty tensor has nothing to move; the row-major factories used to divide by its zero extents.
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, layout=layout, device=device)
    assert tuple(ttnn.permute(ttnn_input, perm).shape) == tuple(torch_input.permute(perm).shape)
    assert tuple(ttnn.transpose(ttnn_input, -2, -1).shape) == tuple(torch_input.transpose(-2, -1).shape)


@pytest.mark.parametrize("widths", [(29, 3), (30, 2), (31, 1), (8, 24), (16, 16)])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_concat_empty_rows(device, widths, layout):
    # Unequal widths take the transpose-based path, which used to divide by the empty tensor's zero extents.
    torch_inputs = [torch.zeros((0, w), dtype=torch.bfloat16) for w in widths]
    ttnn_inputs = [ttnn.from_torch(t, layout=layout, device=device) for t in torch_inputs]
    output = ttnn.concat(ttnn_inputs, dim=1)
    assert tuple(output.shape) == tuple(torch.cat(torch_inputs, dim=1).shape)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_concat_empty_rank4_last_dim(device, layout):
    # A zero extent in the leading dims also reaches the last-dim concat path, which transposes and divided by it.
    torch_inputs = [torch.zeros((2, 0, 32, w), dtype=torch.bfloat16) for w in (29, 3)]
    ttnn_inputs = [ttnn.from_torch(t, layout=layout, device=device) for t in torch_inputs]
    output = ttnn.concat(ttnn_inputs, dim=3)
    assert tuple(output.shape) == tuple(torch.cat(torch_inputs, dim=3).shape)


@pytest.mark.parametrize("shape", [(4, 0), (0, 32), (2, 0, 32, 32)])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_clone_empty(device, shape, layout):
    # The row-major clone factory divides by the last logical dim, which is zero here.
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, layout=layout, device=device)
    output = ttnn.clone(ttnn_input)
    assert tuple(output.shape) == tuple(torch_input.clone().shape)


@pytest.mark.parametrize("shape, repeats", [((4, 0), (1, 2)), ((4, 0), (2, 1)), ((0, 32), (2, 1))])
def test_repeat_empty(device, shape, repeats):
    # Dispatching a repeat program for an empty tensor used to wait on the device forever.
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.repeat(ttnn_input, repeats)
    assert tuple(output.shape) == tuple(torch_input.repeat(*repeats).shape)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_empty_guards_nonempty_control(device, layout):
    # Non-empty inputs must keep going through the device programs and match torch exactly.
    x = torch.arange(32 * 64, dtype=torch.bfloat16).reshape(32, 64)
    xt = ttnn.from_torch(x, layout=layout, device=device)
    assert_equal(ttnn.to_torch(ttnn.permute(xt, (1, 0))), x.permute(1, 0))
    assert_equal(ttnn.to_torch(ttnn.transpose(xt, 0, 1)), x.transpose(0, 1))
    assert_equal(ttnn.to_torch(ttnn.clone(xt)), x.clone())
    assert_equal(ttnn.to_torch(ttnn.repeat(xt, (2, 1))), x.repeat(2, 1))

    y = torch.arange(32 * 32, dtype=torch.bfloat16).reshape(32, 32)
    yt = ttnn.from_torch(y, layout=layout, device=device)
    # The second operand is built on the host, not as `yt + 1`: bf16 add rounds ties differently from torch above 256.
    y2 = y.flip(0)
    y2t = ttnn.from_torch(y2, layout=layout, device=device)
    assert_equal(ttnn.to_torch(ttnn.concat([yt, y2t], dim=1)), torch.cat([y, y2], dim=1))
