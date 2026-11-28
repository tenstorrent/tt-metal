# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the block_max_bf16 pre-migration suite."""

import pytest
import ttnn


@pytest.fixture(scope="module")
def golden_mesh():
    """The golden suite's topology: every device of the machine as one mesh, FABRIC_2D, l1_small_size 24576."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh = ttnn.open_mesh_device(l1_small_size=24576)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture
def fresh_cache(device):
    """Start from an empty program cache (caching is always on: the counter must move on a miss)."""
    device.clear_program_cache()
    assert device.num_program_cache_entries() == 0
    return device
