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


def _sharded_l1_config(grid_start, grid_end, shard_shape, layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED):
    """Sharded L1 memory config over the inclusive core grid grid_start..grid_end."""
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*grid_start), ttnn.CoreCoord(*grid_end))])
    return ttnn.MemoryConfig(
        layout,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_ranges, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )


# Exposed as two factories rather than one taking a boolean, so a call site says which kind it
# is building -- same convention as the per_core_allocation suite next door.


@pytest.fixture(scope="module")
def lockstep_sharded_config():
    """Factory (grid_start, grid_end, shard_shape) -> ordinary lockstep-allocated config."""
    return _sharded_l1_config


@pytest.fixture(scope="module")
def range_lockstep_sharded_config():
    """Factory (grid_start, grid_end, shard_shape) -> config requesting range lockstep."""

    def build(grid_start, grid_end, shard_shape):
        mem_config = _sharded_l1_config(grid_start, grid_end, shard_shape)
        mem_config.experimental_set_range_lockstep_allocation(True)
        return mem_config

    return build


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
