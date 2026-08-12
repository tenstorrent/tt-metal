# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared device-fixture profiles for scoped prefill tests.

Return fresh dictionaries because the repository's ``device_params`` fixture consumes entries with
``pop`` while opening a mesh.
"""

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size


def fabric2d_device_params(**overrides) -> dict:
    """Unwrapped local Fabric2D profile for 2x2/2x4/4x2 Blackhole tests."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def torus_xy_device_params(**overrides) -> dict:
    """Production 8x4 Ring/Ring profile; requires a cabling-certified explicit descriptor."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
    }
    params.update(overrides)
    return params
