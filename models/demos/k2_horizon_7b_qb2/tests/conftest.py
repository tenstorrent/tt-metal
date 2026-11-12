# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest

import ttnn


@pytest.fixture
def qb2_mesh():
    if (
        ttnn.get_num_devices() != 4
        or ttnn.get_arch_name() != "blackhole"
        or ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.P300_X2
    ):
        pytest.skip("Requires a four-device Blackhole P300_X2 QB2")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    # 32 MB holds one traced decode step of a single layer.
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32_000_000)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
