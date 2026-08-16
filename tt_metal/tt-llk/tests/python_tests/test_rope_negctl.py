# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Throwaway negative control for test_rope.py. Every test here MUST fail."""

import torch
from conftest import blackhole_only
from helpers.golden_generators import RopeGolden, get_golden_generator
from test_rope import TILE_SLOT_STRIDE, _dest_tiles, _geometry, _run, _stimuli

pytestmark = blackhole_only


def test_negctl_device_actually_rotates():
    """Fails unless the device leaves the x rows unchanged, i.e. must fail."""
    geometry = _geometry(ht=2, wt=1, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=999)
    device = _run(geometry, tiles, dest)
    assert torch.equal(device, dest), "device output differs from input (expected)"


def test_negctl_sign_convention():
    """Golden with the rotation sign flipped, i.e. must fail."""
    geometry = _geometry(ht=2, wt=1, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=999)
    device = _run(geometry, tiles, dest)

    golden = get_golden_generator(RopeGolden)(dest, **geometry)
    rotated = RopeGolden.rotated_rows(**geometry)
    even = torch.arange(0, 16, 2)
    flipped = golden.clone()
    flipped[rotated, :] = golden[rotated, :]
    flipped[torch.tensor(rotated)[:, None], even] = -golden[
        torch.tensor(rotated)[:, None], even
    ]
    assert torch.equal(
        device[rotated].to(torch.float32), flipped[rotated]
    ), "device does not match the sign-flipped golden (expected)"


def test_negctl_stimuli_reach_the_device():
    """Two different stimuli must give two different results."""
    geometry = _geometry(ht=2, wt=1, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)
    first = _run(geometry, tiles, _stimuli(geometry, tiles, seed=1))
    second = _run(geometry, tiles, _stimuli(geometry, tiles, seed=2))
    assert torch.equal(first, second), "different stimuli gave different results"
