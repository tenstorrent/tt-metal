// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <vector>
#include <utility>

// Forward declarations for mock testing
namespace tt::tt_fabric {

// Mock types for testing - these would come from real implementation
enum class RouterStateCommon {
    RUNNING = 0,
    PAUSED = 1,
};

enum class RouterCommand {
    PAUSE = 0,
    RUN = 1,
};

struct FabricNodeId {
    uint32_t mesh_id = 0;
    uint32_t logical_x = 0;
    uint32_t logical_y = 0;

    bool operator==(const FabricNodeId& other) const {
        return mesh_id == other.mesh_id &&
               logical_x == other.logical_x &&
               logical_y == other.logical_y;
    }
};

using chan_id_t = uint32_t;
using MeshId = uint32_t;
using ChipId = uint32_t;

}  // namespace tt::tt_fabric

namespace tt::tt_fabric::test_utils {

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
    void pause_routers();

    // Issue resume (RUN) command to all routers
    void resume_routers();

    // FR-5: Check if all routers are in specified state
    bool all_routers_in_state(RouterStateCommon expected_state);

    // FR-6: Wait for all routers to enter pause state with timeout
    bool wait_for_pause(
        std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT);

    // FR-6: Generic wait for any state
    bool wait_for_state(
        RouterStateCommon target_state,
        std::chrono::milliseconds timeout = DEFAULT_PAUSE_TIMEOUT,
        std::chrono::milliseconds poll_interval = DEFAULT_POLL_INTERVAL);

    // FR-5: Get state of specific router
    RouterStateCommon get_router_state(
        const FabricNodeId& fabric_node_id,
        chan_id_t channel_id);

protected:
    // Helper: Get all active router cores from control plane
    // FR-2: Must query control plane for all active routers
    std::vector<std::pair<FabricNodeId, chan_id_t>> get_all_router_cores() const;

private:
    ControlPlane& control_plane_;
};

}  // namespace tt::tt_fabric::test_utils
