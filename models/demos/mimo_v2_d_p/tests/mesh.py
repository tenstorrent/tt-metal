# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh parametrization for mimo_v2_d_p device tests: SP on rows, TP on cols.

QuietBox bringup runs 2x2 (SP2 x TP2); the same code targets a BH Galaxy as 8x4 (SP8 x TP4, EP32).
"""

import pytest

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

QB_MESHES = [pytest.param((2, 2), fabric2d_device_params(), id="2x2")]

MESH_PARAMS = pytest.mark.parametrize("mesh_device, device_params", QB_MESHES, indirect=["mesh_device", "device_params"])


def sp_tp(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return rows, cols, 0, 1


def mesh_id(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return f"{rows}x{cols}"
