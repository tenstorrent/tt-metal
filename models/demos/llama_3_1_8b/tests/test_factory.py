# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


def _available_device_count():
    try:
        return ttnn.get_num_devices()
    except Exception:
        return 0


def parametrize_mesh_with_fabric(mesh_shapes=None, linear_fabric=False):
    if mesh_shapes is None:
        mesh_shapes = [(8, 4)]
    available = _available_device_count()
    params = []
    for shape in mesh_shapes:
        n = shape[0] * shape[1]
        if n > available:
            continue
        if n == 1:
            fc = None
        elif linear_fabric:
            fc = ttnn.FabricConfig.FABRIC_1D
        else:
            fc = ttnn.FabricConfig.FABRIC_1D_RING
        dp = {"fabric_config": fc}
        params.append(pytest.param(shape, dp, id=f"{shape[0]}x{shape[1]}"))
    if not params:
        params.append(pytest.param((8, 4), {}, id="8x4", marks=pytest.mark.skip("not enough devices")))
    return pytest.mark.parametrize("mesh_device,device_params", params, indirect=True)
