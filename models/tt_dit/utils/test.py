# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

router_config_wan2_2 = ttnn._ttnn.fabric.FabricRouterConfig()
router_config_wan2_2.max_packet_payload_size_bytes = 8192

# num_command_queues=2 required for halo-buffer mode (NP on CQ1, conv3d on CQ0)
line_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "num_command_queues": 2}
ring_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "num_command_queues": 2}
ring_params_bh_wan2_2 = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "fabric_router_config": router_config_wan2_2,
    "num_command_queues": 2,
}
