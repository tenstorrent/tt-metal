# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

router_config_wan2_2 = ttnn._ttnn.fabric.FabricRouterConfig()
router_config_wan2_2.max_packet_payload_size_bytes = 8192

line_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
ring_params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
ring_params_wan2_2 = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "fabric_router_config": router_config_wan2_2}
