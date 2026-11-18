# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""MeshDevice.l1_allocator_base and MeshDevice.lowest_occupied_compute_l1_address: where a program's circular buffers
start, and the lowest L1 buffer they must end below."""

import torch

import ttnn


def _l1_tensor(device, width):
    return ttnn.from_torch(
        torch.zeros(1, 1, 32, width),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def test_l1_allocator_base(device):
    base = device.l1_allocator_base()
    assert base > 0
    t = _l1_tensor(device, 32 * 64)
    assert t.buffer_address() >= base  # every L1 buffer is above the allocator base
    ttnn.deallocate(t)


def test_lowest_occupied_compute_l1_address(device):
    base = device.l1_allocator_base()
    before = device.lowest_occupied_compute_l1_address()

    a = _l1_tensor(device, 32 * 64)
    low_a = device.lowest_occupied_compute_l1_address()
    assert low_a is not None and base <= low_a <= a.buffer_address()
    assert before is None or low_a <= before

    b = _l1_tensor(device, 32 * 64)
    low_ab = device.lowest_occupied_compute_l1_address()
    assert low_ab <= min(a.buffer_address(), b.buffer_address())

    ttnn.deallocate(b)
    assert device.lowest_occupied_compute_l1_address() == low_a
    ttnn.deallocate(a)
    assert device.lowest_occupied_compute_l1_address() == before
