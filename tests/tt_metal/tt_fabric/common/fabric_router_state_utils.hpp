// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>
#include <cstdint>

#include "tt_metal/fabric/control_plane.hpp"
#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric::test_utils {

// Convert RouterStateCommon enum to human-readable string
// Returns a const char* representation of the router state
// For undefined values, returns "UNKNOWN"
const char* router_state_to_string(RouterStateCommon state);

// Log current state of all routers in the fabric
// Provides observability for debugging test failures (NFR-5)
// Iterates through all specified meshes and logs router state for each device/channel
void log_all_router_states(
    ControlPlane& control_plane,
    const std::vector<MeshId>& mesh_ids);

// Count routers in each state across the fabric
// Returns map: {RUNNING: 4, PAUSED: 0, etc.}
// Aggregates router state counts across all specified meshes and devices
std::map<RouterStateCommon, uint32_t> count_routers_by_state(
    ControlPlane& control_plane,
    const std::vector<MeshId>& mesh_ids);

} // namespace tt::tt_fabric::test_utils
