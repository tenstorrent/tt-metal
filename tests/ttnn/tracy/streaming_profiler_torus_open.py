# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One tt-run rank of test_streaming_profiler_torus_routers_fit: open this rank's 4x2 TORUS_Y mesh with tt-blaze's GLM-5
fabric payload, which builds and lays out the fabric routers, then close it."""

import ttnn

router_config = ttnn.FabricRouterConfig()
router_config.max_packet_payload_size_bytes = 8192
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    router_config,
)
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2))
print("TORUS_OPEN_OK", flush=True)
ttnn.close_mesh_device(mesh_device)
