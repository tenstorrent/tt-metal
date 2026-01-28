// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <vector>
#include <utility>

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Forward declaration of ControlPlane
class ControlPlane;

// Default timeout and poll interval constants
constexpr std::chrono::milliseconds DEFAULT_PAUSE_TIMEOUT{5000};
constexpr std::chrono::milliseconds DEFAULT_POLL_INTERVAL{100};

/**
 * FabricCommandInterface provides high-level test APIs for controlling fabric routers.
 *
 * This class implements a basic programming interface for fabric router state:
 * - Pausing and resuming routers
 * - Waiting for state transitions with timeout
 * - Querying router state across all active channels
 *
 */
class FabricCommandInterface {
public:
    explicit FabricCommandInterface(const ControlPlane& control_plane);

    // Issue pause command to all active routers
    void pause_routers() const;

    // Issue resume (RUN) command to all routers
    void resume_routers() const;

    // Check if all routers are in specified state
    bool all_routers_in_state(RouterState expected_state) const;

    // Wait for all routers to enter pause state with timeout
    bool wait_for_pause(
        std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT) const;

    // Generic wait for any state
    bool wait_for_state(
        RouterState target_state,
        std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT,
        std::chrono::milliseconds poll_interval = DEFAULT_POLL_INTERVAL) const;

    // Get state of specific router
    RouterState read_router_state(const FabricNodeId& fabric_node_id, chan_id_t channel_id) const;

private:
    // Helper for sending commands to all routers
    void issue_command_to_routers(RouterCommand router_command) const;

    // Helper: Get all active router cores from control plane
    std::vector<std::pair<FabricNodeId, chan_id_t>> get_all_router_cores() const;

    const ControlPlane& control_plane;

};

}  // namespace tt::tt_fabric
