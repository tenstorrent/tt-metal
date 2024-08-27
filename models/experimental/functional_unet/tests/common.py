# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def is_n300_with_eth_dispatch_cores(device_mesh) -> bool:
    all_devices_using_full_grid = all(
        [(8 == device.core_grid.x and 8 == device.core_grid.y) for device in device_mesh.get_devices()]
    )
    return all_devices_using_full_grid and (len(device_mesh.get_devices()) == 2)


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
