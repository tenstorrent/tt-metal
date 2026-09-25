# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QuietBox mesh parametrization shared by gemma4_26b_d_p device tests: SP on rows, TP on cols."""

import os

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_y_device_params,
)

def _4x1_params():
    # GEMMA4_4X1_FABRIC=1d_ring: 1D ring fabric on 4x1 (enables MoE dispatch sparse multicast; 2x faster with
    # GEMMA4_NUM_LINKS=2). Default 1d_ring; GEMMA4_4X1_FABRIC=2d_torus_y for the 2D torus.
    p = torus_y_device_params()
    if os.environ.get("GEMMA4_4X1_FABRIC", "1d_ring") == "1d_ring":
        p["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    return p


QB_MESHES = [
    pytest.param((2, 2), fabric2d_device_params(), id="2x2"),
    pytest.param((1, 4), torus_x_device_params(), id="1x4"),
    pytest.param((4, 1), _4x1_params(), id="4x1"),
]

MESH_PARAMS = pytest.mark.parametrize("mesh_device, device_params", QB_MESHES, indirect=["mesh_device", "device_params"])


def sp_tp(mesh_device):
    """(sp, tp, sp_axis, tp_axis) for the SP-rows / TP-cols convention."""
    rows, cols = tuple(mesh_device.shape)
    return rows, cols, 0, 1


def mesh_id(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    return f"{rows}x{cols}"
