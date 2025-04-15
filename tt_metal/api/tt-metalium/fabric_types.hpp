// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_fabric {
// routing mode macro for (mainly) kernel code
#define ROUTING_MODE_UNDEFINED 0x0000
#define ROUTING_MODE_1D 0x0001
#define ROUTING_MODE_2D 0x0002
#define ROUTING_MODE_3D 0x0004
#define ROUTING_MODE_LINE 0x0008
#define ROUTING_MODE_RING 0x0010
#define ROUTING_MODE_MESH 0x0020
#define ROUTING_MODE_TORUS 0x0040
#define ROUTING_MODE_LOW_LATENCY 0x0080
#define ROUTING_MODE_DYNAMIC 0x0100

// routing mode type for host code
// other type can be represented by (RoutingMode)(ROUTING_MODE_1D | ROUTING_MODE_RING) etc.
enum class RoutingMode : uint16_t {
    RoutingModeUndefined = ROUTING_MODE_UNDEFINED,
};

}  // namespace tt::tt_fabric

namespace tt::tt_metal {
enum class FabricConfig : uint32_t {
    DISABLED = 0,
    FABRIC_1D = 1,       // Instatiates fabric with 1D routing and no deadlock avoidance
    FABRIC_1D_RING = 2,  // Instatiates fabric with 1D routing and with deadlock avoidance using datelines
    FABRIC_2D = 3,       // Instatiates fabric with 2D routing in pull mode
    FABRIC_2D_PUSH = 4,  // Instatiates fabric with 2D routing in push mode
    CUSTOM = 5
};

}  // namespace tt::tt_metal
