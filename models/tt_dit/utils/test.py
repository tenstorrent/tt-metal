# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def create_fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 8192
    return config


def mesh_device_config_to_string(val) -> str:
    """Readable pytest id for a combined (mesh_device, device_params) param.

    Called once per value by pytest: a (rows, cols) mesh shape -> '2x4', or a
    device_params dict -> a fabric tag like 'fabric_1d' / 'no_fabric'.
    """
    if isinstance(val, dict):
        fabric_config = val.get("fabric_config")
        return "no_fabric" if fabric_config is None else fabric_config.name.lower()
    rows, cols = val
    return f"{rows}x{cols}"


# TODO: consider using these fixtures instead of copy-pasting the full dict.
line_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
ring_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
line_params_8k = {**line_params, "fabric_router_config": create_fabric_router_config()}
ring_params_8k = {**ring_params, "fabric_router_config": create_fabric_router_config()}
