# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def is_n300_with_eth_dispatch_cores(mesh_device) -> bool:
    all_devices_using_full_grid = all(
        [(8 == device.core_grid.x and 8 == device.core_grid.y) for device in mesh_device.get_devices()]
    )
    return all_devices_using_full_grid and (len(mesh_device.get_devices()) == 2)


def is_t3k_with_eth_dispatch_cores(mesh_device) -> bool:
    all_devices_using_full_grid = all(
        [(8 == device.core_grid.x and 8 == device.core_grid.y) for device in mesh_device.get_devices()]
    )
    return all_devices_using_full_grid and (len(mesh_device.get_devices()) == 8)


def check_pcc_conv(torch_tensor, ttnn_tensor, pcc=0.999, mesh_composer=None):
    B, C, H, W = torch_tensor.shape
    ttnn_tensor = ttnn.to_torch(ttnn_tensor, mesh_composer=mesh_composer).reshape(B, H, W, C).permute(0, 3, 1, 2)
    assert_with_pcc(torch_tensor, ttnn_tensor, pcc)


def check_pcc_pool(torch_tensor, ttnn_tensor, pcc=0.999, mesh_composer=None):
    B, C, H, W = torch_tensor.shape
    ttnn_tensor = (
        ttnn.to_torch(ttnn_tensor, mesh_composer=mesh_composer).reshape(B, H, W, -1).permute(0, 3, 1, 2)[:, :C, :, :]
    )
    assert_with_pcc(torch_tensor, ttnn_tensor, pcc)
