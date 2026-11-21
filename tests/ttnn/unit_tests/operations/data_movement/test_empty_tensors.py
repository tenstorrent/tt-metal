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
def test_permute_empty(device, shape, perm, layout):
    # An empty tensor has nothing to move; the row-major factories used to divide by its zero extents.
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, layout=layout, device=device)
    assert tuple(ttnn.permute(ttnn_input, perm).shape) == tuple(torch_input.permute(perm).shape)


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
    assert_equal(ttnn.to_torch(ttnn.clone(xt)), x.clone())
    assert_equal(ttnn.to_torch(ttnn.repeat(xt, (2, 1))), x.repeat(2, 1))
