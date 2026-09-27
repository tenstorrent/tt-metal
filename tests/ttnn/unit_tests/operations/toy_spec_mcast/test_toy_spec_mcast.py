# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn.operations.toy_spec_mcast import toy_spec_mcast, toy_spec_mcast_2d

TILE = 32


@pytest.fixture(scope="module")
def device():
    ttnn.CONFIG.validate_program_args = True
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _input(device, rows, seed):
    torch.manual_seed(seed)
    t = torch.randn(1, 1, TILE * rows, TILE, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _check(t, out, rows, cols):
    got = ttnn.to_torch(out).float()
    assert tuple(got.shape) == (1, 1, TILE * rows, TILE * cols)
    for x in range(cols):
        column = got[..., x * TILE : (x + 1) * TILE]
        assert torch.equal(column, t.float()), f"column {x} did not receive the broadcast tile"


@pytest.mark.parametrize("rows,cols", [(1, 4), (2, 4), (4, 2), (3, 3)])
def test_broadcast_across_row(device, rows, cols):
    t, inp = _input(device, rows, seed=rows * 10 + cols)
    _check(t, toy_spec_mcast(inp, rows, cols), rows, cols)


def test_single_column_is_degenerate(device):
    """cols == 1: no remote receivers; the sender still publishes its local tile."""
    t, inp = _input(device, 2, seed=1)
    _check(t, toy_spec_mcast(inp, 2, 1), 2, 1)


def test_cache_hit_refreshes_tensor_addresses(device):
    rows, cols = 2, 4
    t1, inp1 = _input(device, rows, seed=2)
    first = toy_spec_mcast(inp1, rows, cols)
    _check(t1, first, rows, cols)

    entries = device.num_program_cache_entries()
    t2, inp2 = _input(device, rows, seed=3)
    second = toy_spec_mcast(inp2, rows, cols)

    assert device.num_program_cache_entries() == entries, "expected a cache hit, got a new entry"
    _check(t2, second, rows, cols)
    _check(t1, first, rows, cols)


# ---------------------------------------------------------------------------------------------
# 2D topology: one mcast over a rectangle, from one sender core.
# ---------------------------------------------------------------------------------------------


def _input_tile(device, seed):
    torch.manual_seed(seed)
    t = torch.randn(1, 1, TILE, TILE, dtype=torch.bfloat16)
    return t, ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _check_2d(t, out, num_cores):
    got = ttnn.to_torch(out).float()
    assert tuple(got.shape) == (1, 1, TILE, TILE * num_cores)
    for i in range(num_cores):
        tile = got[..., i * TILE : (i + 1) * TILE]
        assert torch.equal(tile, t.float()), f"participating core {i} did not receive the broadcast tile"


@pytest.mark.parametrize("rows,cols", [(1, 4), (2, 4), (4, 2), (3, 3), (2, 2)])
def test_broadcast_over_rectangle(device, rows, cols):
    """The sender sits at the rect origin, so the data mcast is a loopback over the whole rect."""
    t, inp = _input_tile(device, seed=rows * 100 + cols)
    _check_2d(t, toy_spec_mcast_2d(inp, rows, cols), rows * cols)


def test_2d_interior_sender(device):
    """A sender strictly inside the rect: fan-out is area - 1 in every direction at once."""
    t, inp = _input_tile(device, seed=7)
    _check_2d(t, toy_spec_mcast_2d(inp, 3, 3, sender=(1, 1)), 9)


def test_2d_sender_outside_the_rect(device):
    """The sender is not a receiver: the participating set is rect + {sender}, and the semaphores
    have to reach that extra core or its consumer_ready ack never exists."""
    t, inp = _input_tile(device, seed=8)
    _check_2d(t, toy_spec_mcast_2d(inp, 2, 3, sender=(3, 0)), 2 * 3 + 1)


def test_2d_single_core_is_degenerate(device):
    """A 1x1 rect with the sender inside it has no remote receivers and preserves the local tile."""
    t, inp = _input_tile(device, seed=9)
    _check_2d(t, toy_spec_mcast_2d(inp, 1, 1), 1)


def test_2d_cache_hit_refreshes_tensor_addresses(device):
    t1, inp1 = _input_tile(device, seed=10)
    first = toy_spec_mcast_2d(inp1, 2, 3)
    _check_2d(t1, first, 6)

    entries = device.num_program_cache_entries()
    t2, inp2 = _input_tile(device, seed=11)
    second = toy_spec_mcast_2d(inp2, 2, 3)

    assert device.num_program_cache_entries() == entries, "expected a cache hit, got a new entry"
    _check_2d(t2, second, 6)
    _check_2d(t1, first, 6)
