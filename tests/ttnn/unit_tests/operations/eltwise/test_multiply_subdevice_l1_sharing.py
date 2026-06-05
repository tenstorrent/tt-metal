# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Demonstrate that two ops dispatched simultaneously to disjoint sub-devices
share L1 budget: the same shape that fits when only one sub-device is active
fails when both run back-to-back."""

import pytest
import torch
import ttnn
from models.common.utility_functions import skip_for_slow_dispatch


# Wormhole 8x8 worker grid split in half along Y.
UPPER_HALF = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
LOWER_HALF = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(7, 7))})

# Tune upward until the both-active test fails while the one-active test still passes.
SHAPE = [1, 1, 8192, 8192]


def setup_two_subdevices(device):
    sub_device_0 = ttnn.SubDevice([UPPER_HALF])
    sub_device_1 = ttnn.SubDevice([LOWER_HALF])
    sub_device_manager = device.create_sub_device_manager([sub_device_0, sub_device_1], 3200)
    device.load_sub_device_manager(sub_device_manager)
    return sub_device_manager


def teardown_sub_device(device, sub_device_manager):
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)


def make_pair(device):
    torch_a = torch.randn(SHAPE, dtype=torch.bfloat16)
    torch_b = torch.randn(SHAPE, dtype=torch.bfloat16)
    tt_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return torch_a, torch_b, tt_a, tt_b


@skip_for_slow_dispatch()
def test_multiply_both_subdevices_active(device):
    torch.manual_seed(0)

    sub_device_manager = setup_two_subdevices(device)
    try:
        torch_a0, torch_b0, tt_a0, tt_b0 = make_pair(device)
        torch_a1, torch_b1, tt_a1, tt_b1 = make_pair(device)

        ttnn.multiply_(tt_a0, tt_b0, sub_device_id=ttnn.SubDeviceId(0))
        ttnn.multiply_(tt_a1, tt_b1, sub_device_id=ttnn.SubDeviceId(1))
        ttnn.synchronize_device(device)

        r0 = ttnn.to_torch(tt_a0)
        r1 = ttnn.to_torch(tt_a1)

        assert torch.allclose(r0, torch_a0 * torch_b0, atol=0.1, rtol=0.01)
        assert torch.allclose(r1, torch_a1 * torch_b1, atol=0.1, rtol=0.01)
    finally:
        teardown_sub_device(device, sub_device_manager)


@skip_for_slow_dispatch()
def test_multiply_only_first_subdevice_active(device):
    torch.manual_seed(0)

    sub_device_manager = setup_two_subdevices(device)
    try:
        torch_a0, torch_b0, tt_a0, tt_b0 = make_pair(device)

        ttnn.multiply_(tt_a0, tt_b0, sub_device_id=ttnn.SubDeviceId(0))
        ttnn.synchronize_device(device)

        r0 = ttnn.to_torch(tt_a0)
        assert torch.allclose(r0, torch_a0 * torch_b0, atol=0.1, rtol=0.01)
    finally:
        teardown_sub_device(device, sub_device_manager)
