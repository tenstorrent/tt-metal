# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest that enables HYBRID allocator mode for all tests in this directory.
The env var is set before device creation and removed on teardown.

Range lockstep only differs from ordinary lockstep in HYBRID mode: it narrows which banks the
allocator scans for per-core occupied ranges, and outside HYBRID there are no per-core
allocators to scan.
"""

import os

import pytest
import ttnn


@pytest.fixture(scope="function")
def hybrid_mesh_device():
    """Single-device 1x1 mesh with HYBRID allocator mode.

    A mesh_mapper is what selects the on-device construction branch in
    create_tt_tensor_from_host_data; the single-device path sends any sharded config to host via
    is_data_transformation_required. Tests that allocate therefore need a mesh, but not more
    than one device.
    """
    os.environ["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    yield mesh
    ttnn.close_mesh_device(mesh)
    os.environ.pop("TT_METAL_ALLOCATOR_MODE_HYBRID", None)
