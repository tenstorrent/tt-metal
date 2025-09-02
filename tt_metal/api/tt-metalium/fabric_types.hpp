// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_fabric {

enum class FabricConfig : uint32_t {
    DISABLED = 0,
    FABRIC_1D = 1,                    // 1D routing and no deadlock avoidance
    FABRIC_1D_RING = 2,               // 1D routing and deadlock avoidance using datelines
    FABRIC_2D = 3,                    // 2D routing
    FABRIC_2D_TORUS_X = 4,            // 2D routing and deadlock avoidance along X axis
    FABRIC_2D_TORUS_Y = 5,            // 2D routing and deadlock avoidance along Y axis
    FABRIC_2D_TORUS_XY = 6,           // 2D routing and deadlock avoidance along XY axes
    FABRIC_2D_DYNAMIC = 7,            // 2D routing with dynamic routing
    FABRIC_2D_DYNAMIC_TORUS_X = 8,    // 2D routing with dynamic routing and deadlock avoidance along X axis
    FABRIC_2D_DYNAMIC_TORUS_Y = 9,    // 2D routing with dynamic routing and deadlock avoidance along Y axis
    FABRIC_2D_DYNAMIC_TORUS_XY = 10,  // 2D routing with dynamic routing and deadlock avoidance along XY axes
    CUSTOM = 11
};

// tensix extension for fabric routers, used to build connections between worker - fabric router, upstream fabric router
// - downstream fabric router.
enum class FabricTensixConfig : uint32_t {
    DISABLED = 0,  // not using tensix extension
    MUX = 1,       // using mux kernel as tensix extension
};

enum class FabricReliabilityMode : uint32_t {

    // When fabric is initialized, user expects live links/devices to exactly match the mesh graph descriptor.
    // Any downed devices/links will result in some sort of error condition being reported.
    STRICT_SYSTEM_HEALTH_SETUP_MODE = 0,

    // When fabric is initialized, user is flexible towards downed links/devices. This mode specifies that fabric
    // can be initialized with fewer routing planes than are in the mesh graph descriptor, according to the number
    // of live links in the system
    RELAXED_SYSTEM_HEALTH_SETUP_MODE = 1,

    // Unsupported - fabric can be setup at runtime. Placeholder
    DYNAMIC_RECONFIGURATION_SETUP_MODE = 2,
};

}  // namespace tt::tt_metal

namespace tt::tt_fabric {

using MeshId = tt::stl::StrongType<uint32_t, struct MeshIdTag>;
using MeshHostRankId = tt::stl::StrongType<uint32_t, struct HostRankTag>;

}  // namespace tt::tt_fabric
