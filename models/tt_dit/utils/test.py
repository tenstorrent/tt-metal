# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def create_fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 8192
    return config


line_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}
ring_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "l1_small_size": 32768}
line_params_8k = {**line_params, "fabric_router_config": create_fabric_router_config()}
ring_params_8k = {**ring_params, "fabric_router_config": create_fabric_router_config()}

# WH Loud Box 2x4: SP on mesh axis 0, TP on axis 1 (gather+SDPA self-attn path).
wh_lb_params = {**line_params, "l1_small_size": 16384}
wh_lb_2x4_id = "wh_lb_2x4sp0tp1"
wh_lb_2x4_params = (
    (2, 4),
    (2, 4),
    0,
    1,
    2,
    False,
    wh_lb_params,
    ttnn.Topology.Linear,
    False,
)

# BH Loud Box 2x4: matches Wan2.2 "bh_2x4sp1tp0" (SP axis 1, TP axis 0).
# dynamic_load=True: DiT and VAE must not stay resident together on LB (see OOM on
# ~12GB VAE decode buffer). Self-attn uses ring_joint_sdpa vs WH gather fallback.
bh_lb_2x4_id = "bh_lb_2x4sp1tp0"
bh_lb_2x4_params = (
    (2, 4),
    (2, 4),
    1,
    0,
    2,
    True,
    line_params,
    ttnn.Topology.Linear,
    False,
)


def skip_ltx_mesh_config_unless_matching_arch(config_id: str) -> None:
    """Skip WH mesh configs on Blackhole and vice versa."""
    import pytest

    from models.common.utility_functions import is_blackhole, is_wormhole_b0

    if config_id.startswith("bh_") and not is_blackhole():
        pytest.skip(f"{config_id} requires Blackhole")
    if config_id.startswith("wh_") and not is_wormhole_b0():
        pytest.skip(f"{config_id} requires Wormhole B0")
