# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Conftest for single-pod BH pipeline tests.

Overrides the parent exabox conftest's mesh_device fixture with one that
handles per-rank (4, 2) BH meshes plus the FABRIC_2D_TORUS_Y + fabric_router_config
+ worker_l1_size device_params required by the 16-stage single-pod pipeline.

Execution model differs from the unified 16x4 / 32x4 meshes used by the
sibling exabox tests: here each BH slice is its own MPI rank with an 8-device
(4x2) mesh, and the 16 ranks coordinate via distributed sockets.
"""

from __future__ import annotations

import pytest

import ttnn
from conftest import bh_2d_mesh_device_context


@pytest.fixture(scope="function")
def mesh_device(request, device_params):
    """Per-rank (4, 2) BH mesh via bh_2d_mesh_device_context.

    The parent exabox fixture drops fabric_router_config and worker_l1_size
    when opening the mesh; bh_2d_mesh_device_context threads them through
    set_fabric so FABRIC_2D_TORUS_Y + max_packet_payload_size_bytes=15232
    (required by the single-pod fabric) are honored.
    """
    try:
        param = request.param
    except (ValueError, AttributeError):
        param = (4, 2)
    assert param == (4, 2), f"single-pod tests expect mesh_device=(4, 2) per rank, got {param}"

    with bh_2d_mesh_device_context(device_params) as md:
        yield md
