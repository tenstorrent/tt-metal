// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_fabric {
// fabric mode macro for (mainly) kernel code
#define FABRIC_MODE_UNDEFINED 0x0000
#define FABRIC_MODE_1D 0x0001
#define FABRIC_MODE_2D 0x0002
#define FABRIC_MODE_3D 0x0004
#define FABRIC_MODE_LINE 0x0008
#define FABRIC_MODE_RING 0x0010
#define FABRIC_MODE_MESH 0x0020
#define FABRIC_MODE_TORUS 0x0040
#define FABRIC_MODE_LOW_LATENCY 0x0080
#define FABRIC_MODE_DYNAMIC 0x0100

#define FABRIC_MODE_1D_LINE (FABRIC_MODE_1D | FABRIC_MODE_LINE)
#define FABRIC_MODE_1D_RING (FABRIC_MODE_1D | FABRIC_MODE_RING)
#define FABRIC_MODE_1D_LINE_LOW_LATENCY (FABRIC_MODE_1D_LINE | FABRIC_MODE_LOW_LATENCY)
#define FABRIC_MODE_1D_RING_LOW_LATENCY (FABRIC_MODE_1D_RING | FABRIC_MODE_LOW_LATENCY)
#define FABRIC_MODE_2D_MESH (FABRIC_MODE_2D | FABRIC_MODE_MESH)
#define FABRIC_MODE_2D_TORUS (FABRIC_MODE_2D | FABRIC_MODE_TORUS)
#define FABRIC_MODE_2D_MESH_LOW_LATENCY (FABRIC_MODE_2D_MESH | FABRIC_MODE_LOW_LATENCY)
#define FABRIC_MODE_2D_TORUS_LOW_LATENCY (FABRIC_MODE_2D_TORUS | FABRIC_MODE_LOW_LATENCY)
#define FABRIC_MODE_2D_MESH_DYNAMIC (FABRIC_MODE_2D_MESH | FABRIC_MODE_DYNAMIC)
#define FABRIC_MODE_2D_TORUS_DYNAMIC (FABRIC_MODE_2D_TORUS | FABRIC_MODE_DYNAMIC)

// fabric mode type for host code
enum class FabricMode : uint16_t {
    FabricModeUndefined = FABRIC_MODE_UNDEFINED,
    FabricMode1DLine = FABRIC_MODE_1D_LINE,
    FabricMode1DRing = FABRIC_MODE_1D_RING,
    FabricMode1DLineLowLatency = FABRIC_MODE_1D_LINE_LOW_LATENCY,
    FabricMode1DRingLowLatency = FABRIC_MODE_1D_RING_LOW_LATENCY,
    FabricMode2DMesh = FABRIC_MODE_2D_MESH,
    FabricMode2DTorus = FABRIC_MODE_2D_TORUS,
    FabricMode2DMeshLowLatency = FABRIC_MODE_2D_MESH_LOW_LATENCY,
    FabricMode2DTorusLowLatency = FABRIC_MODE_2D_TORUS_LOW_LATENCY,
    FabricMode2DMeshDynamic = FABRIC_MODE_2D_MESH_DYNAMIC,
    FabricMode2DTorusDynamic = FABRIC_MODE_2D_TORUS_DYNAMIC
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
