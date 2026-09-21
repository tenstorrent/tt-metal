# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared by the device test suites."""

from __future__ import annotations

import pytest

import ttnn


@pytest.fixture
def mesh(mesh_device, submesh_shape):
    """The sub-mesh of the requested shape, carved out of the pod's full allocation.

    Direct ``open_mesh_device(MeshShape(2, 2))`` on a galaxy fails fabric router sync: the
    mesh-graph descriptor describes the whole 8x4 fabric and a smaller allocation cannot complete
    the remote ethernet handshake. ``create_submesh`` carves a real sub-mesh out of the full
    allocation instead, which maps — a correctness proxy for a smaller pod, not a performance one.
    """
    shape = tuple(submesh_shape)
    if tuple(mesh_device.shape) == shape:
        yield mesh_device
        return
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*shape), ttnn.MeshCoordinate(0, 0))
    try:
        yield submesh
    finally:
        ttnn.synchronize_device(submesh)
