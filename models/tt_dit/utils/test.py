# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def create_fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 8192
    return config


line_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
ring_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
line_params_8k = {**line_params, "fabric_router_config": create_fabric_router_config()}
ring_params_8k = {**ring_params, "fabric_router_config": create_fabric_router_config()}
