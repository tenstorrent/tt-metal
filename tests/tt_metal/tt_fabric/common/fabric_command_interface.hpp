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
 * This class wraps FabricRouterStateManager with test-friendly methods for:
 * - Pausing and resuming routers
 * - Waiting for state transitions with timeout
 * - Querying router state across all active channels
 *
 * Requirements from FR-4 through FR-6 of the specification.
 */
class FabricCommandInterface {
public:
    /**
     * Constructor: Accept ControlPlane reference
     * FR-1: Must accept ControlPlane& reference and store it for later use
     */
    explicit FabricCommandInterface(ControlPlane& control_plane);

    // FR-4: Issue pause command to all active routers
    void pause_routers(const ControlPlane &control_plane) const;

    // Issue resume (RUN) command to all routers
    void resume_routers(const ControlPlane &control_plane) const;

    // FR-5: Check if all routers are in specified state
    bool all_routers_in_state(const ControlPlane &control_plane, RouterStateCommon expected_state) const;

    // FR-6: Wait for all routers to enter pause state with timeout
    bool wait_for_pause(const ControlPlane &control_plane,
        std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT) const;

    // FR-6: Generic wait for any state
    bool wait_for_state(const ControlPlane &control_plane,
        RouterStateCommon target_state,
        std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT,
        std::chrono::milliseconds poll_interval = DEFAULT_POLL_INTERVAL) const;

    // FR-5: Get state of specific router
    RouterStateCommon get_router_state(const ControlPlane &control_plane, const FabricNodeId& fabric_node_id, chan_id_t channel_id) const;

private:
    // Helper: Get all active router cores from control plane
    // FR-2: Must query control plane for all active routers
    std::vector<std::pair<FabricNodeId, chan_id_t>> get_all_router_cores(const ControlPlane &control_plane) const;

};

}  // namespace tt::tt_fabric
