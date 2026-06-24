# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Mesh-device open with fabric config — model-agnostic."""

import ttnn


def open_mesh_device(mesh_shape: tuple, fabric_payload_size: int, l1_small_size: int = 0) -> ttnn.MeshDevice:
    """Configure fabric (1D for sp<=8, else 2D) and open the mesh device. `l1_small_size` > 0 carves an
    L1_SMALL region (needed when an op routes its semaphores there, e.g. the Kimi MoE routing all-gather
    with use_l1_small_for_semaphores)."""
    sp = mesh_shape[0]
    fabric_config = ttnn.FabricConfig.FABRIC_1D if sp <= 8 else ttnn.FabricConfig.FABRIC_2D

    # FabricRouterConfig with the model's max packet payload (was create_fabric_router_config()).
    fabric_router_config = ttnn._ttnn.fabric.FabricRouterConfig()
    fabric_router_config.max_packet_payload_size_bytes = fabric_payload_size

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=l1_small_size)
