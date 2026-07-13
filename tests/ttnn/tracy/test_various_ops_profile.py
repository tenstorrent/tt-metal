# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.utility_functions import skip_for_slow_dispatch


def _setup_sub_devices(device):
    sub_device_0_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    sub_device_1_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 4))})
    sub_device_manager = device.create_sub_device_manager(
        [ttnn.SubDevice([sub_device_0_cores]), ttnn.SubDevice([sub_device_1_cores])],
        3200,
    )
    device.load_sub_device_manager(sub_device_manager)
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
    return sub_device_manager


def _teardown_sub_devices(device, sub_device_manager):
    device.reset_sub_device_stall_group()
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(sub_device_manager)


@skip_for_slow_dispatch()
def test_various_ops_profile(device):
    """Run a mix of TTNN ops across two sub-devices for Tracy ops-perf-report integration tests."""

    torch.manual_seed(0)
    shape = [1, 1, 64, 64]
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    sub_device_manager = _setup_sub_devices(device)
    sub_device_0 = ttnn.SubDeviceId(0)
    sub_device_1 = ttnn.SubDeviceId(1)

    try:
        a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Only use ops that accept sub_device_id. Softmax/to_layout do not and fail once a
        # sub-device manager is loaded (kernel cores must match the target sub-device).
        ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=2, x=2), sub_device_id=sub_device_0)
        ttnn.add(a, b, sub_device_id=sub_device_1)
        ttnn.multiply(a, b, sub_device_id=sub_device_1)
        ttnn.subtract(a, b, sub_device_id=sub_device_0)
        ttnn.add(b, a, sub_device_id=sub_device_0)

        ttnn.synchronize_device(device)
    finally:
        _teardown_sub_devices(device, sub_device_manager)
