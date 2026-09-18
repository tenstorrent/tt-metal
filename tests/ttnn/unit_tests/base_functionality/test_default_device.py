# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Closing a device must drop ttnn's default device when it points at that device.

The default device (``ttnn.SetDefaultDevice`` / ``ttnn.GetDefaultDevice``) is a raw, non-owning
pointer. If it still points at a device, or at one of its submeshes, after that device is closed,
every later ``ttnn.GetDefaultDevice()`` call dereferences freed memory. ttnn forgets the default
device in its close paths so a stale default cannot outlive the device it points at.
"""

import ttnn


def test_close_device_forgets_matching_default_device():
    device = ttnn.open_device(device_id=0)
    try:
        ttnn.SetDefaultDevice(device)
        assert ttnn.GetDefaultDevice() is not None
    finally:
        ttnn.close_device(device)
    assert ttnn.GetDefaultDevice() is None


def test_close_mesh_device_forgets_default_set_to_submesh():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        submesh = mesh.create_submesh(ttnn.MeshShape(1, 1))
        ttnn.SetDefaultDevice(submesh)
        assert ttnn.GetDefaultDevice() is not None
    finally:
        ttnn.close_mesh_device(mesh)
    assert ttnn.GetDefaultDevice() is None


def test_closing_a_submesh_keeps_a_default_set_to_its_parent():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        submesh = mesh.create_submesh(ttnn.MeshShape(1, 1))
        ttnn.SetDefaultDevice(mesh)
        ttnn.close_mesh_device(submesh)
        assert ttnn.GetDefaultDevice() is not None
    finally:
        ttnn.close_mesh_device(mesh)
    assert ttnn.GetDefaultDevice() is None
