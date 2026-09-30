# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# Shapes the ternary kernel does not have a broadcast pattern for. A composite
# of gtz, lez, two multiplies and an add used to be run for these, and the
# multiply inside it rejected them in turn, so the caller saw a complaint about
# an op they never wrote.
UNSUPPORTED = [
    ([1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 64, 32]),
    ([1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 64]),
    ([1, 1, 32, 32], [1, 1, 1, 32], [1, 1, 32, 64]),
    ([1, 1, 64, 32], [1, 1, 32, 64], [1, 1, 32, 32]),
]

SUPPORTED = [
    ([1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]),
    ([1, 1, 32, 32], [1, 1, 1, 32], [1, 1, 32, 32]),
    ([1, 1, 32, 32], [1, 1, 32, 1], [1, 1, 32, 32]),
    ([1, 1, 1, 1], [1, 1, 32, 32], [1, 1, 32, 32]),
]


def dev(device, shape):
    return ttnn.from_torch(
        torch.rand(shape, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )


@pytest.mark.parametrize("p_shape, t_shape, f_shape", UNSUPPORTED)
def test_where_names_the_shapes_it_cannot_broadcast(device, expect_error, p_shape, t_shape, f_shape):
    with expect_error(RuntimeError, "do not broadcast in a pattern"):
        ttnn.where(dev(device, p_shape), dev(device, t_shape), dev(device, f_shape))


@pytest.mark.parametrize("p_shape, t_shape, f_shape", SUPPORTED)
def test_where_still_broadcasts_what_it_supported(device, p_shape, t_shape, f_shape):
    p, t, f = dev(device, p_shape), dev(device, t_shape), dev(device, f_shape)
    got = ttnn.to_torch(ttnn.where(p, t, f))
    want = torch.where(ttnn.to_torch(p) > 0, ttnn.to_torch(t), ttnn.to_torch(f))
    assert torch.equal(got, want)
