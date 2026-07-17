# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

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
line_params_req_exact_devices = {**line_params, "require_exact_physical_num_devices": True}
ring_params_req_exact_devices = {**ring_params, "require_exact_physical_num_devices": True}


def skip_if_unsupported_num_links(mesh_device, num_links):
    """Skip the test if the mesh device does not support the requested number of links."""
    from models.common.modules.tt_ccl import get_num_links

    available_links = get_num_links(mesh_device)

    # WARNING: get_num_links() returns 0 for 1x1 device meshes.
    # Some tests requested a 1x1 device mesh with nl=1. They will
    # be erroneously skipped. TODO Fix all parameterizations which request
    # 1x1 mesh_device with num_links = 1.
    if available_links < num_links:
        pytest.skip(
            f"Mesh device supports {available_links} link(s) but test requires {num_links}. "
            f"Mesh shape: {mesh_device.shape}"
        )
