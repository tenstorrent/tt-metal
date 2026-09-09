# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.utils.trace_region_sizes import build_trace_device_params


@pytest.fixture
def qb2_mesh():
    if ttnn.get_num_devices() != 4 or ttnn.get_arch_name() != "blackhole":
        pytest.skip("Requires a four-device Blackhole QB2")
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router)
    params = build_trace_device_params("llama3.1-8b-qb2-decoder")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=16384, **params)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
