// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fabric_command_interface.hpp"
#include <thread>
#include <chrono>

namespace tt::tt_fabric::test_utils {

FabricCommandInterface::FabricCommandInterface(ControlPlane& control_plane)
    : control_plane_(control_plane) {
}

void FabricCommandInterface::pause_routers() {
    // FR-4: Issue PAUSE command to all active routers
    const auto& router_cores = get_all_router_cores();
    auto& state_manager = control_plane_.get_state_manager();

    for (const auto& [fabric_node_id, channel_id] : router_cores) {
        state_manager.queue_state_transition(RouterCommand::PAUSE);
    }
}

void FabricCommandInterface::resume_routers() {
    // FR-4: Issue RUN command to all active routers
    const auto& router_cores = get_all_router_cores();
    auto& state_manager = control_plane_.get_state_manager();

    for (const auto& [fabric_node_id, channel_id] : router_cores) {
        state_manager.queue_state_transition(RouterCommand::RUN);
    }
}

bool FabricCommandInterface::all_routers_in_state(RouterStateCommon expected_state) {
    // FR-5: Check if all routers are in specified state
    const auto& router_cores = get_all_router_cores();

    // Empty topology returns false (no routers to verify)
    if (router_cores.empty()) {
        return false;
    }

    auto& state_manager = control_plane_.get_state_manager();

    for (const auto& [fabric_node_id, channel_id] : router_cores) {
        RouterStateCommon state = state_manager.get_core_state(fabric_node_id, channel_id);
        if (state != expected_state) {
            return false;
        }
    }

    return true;
}

bool FabricCommandInterface::wait_for_pause(std::chrono::milliseconds timeout) {
    // FR-6: Wrapper for wait_for_state with PAUSED state
    return wait_for_state(RouterStateCommon::PAUSED, timeout);
}

bool FabricCommandInterface::wait_for_state(
    RouterStateCommon target_state,
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval) {
    // FR-6: Wait for all routers to reach target state with timeout
    // Uses polling with std::this_thread::sleep_for between polls (NO BUSY-WAIT)

    auto& state_manager = control_plane_.get_state_manager();
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        // Refresh all core states
        state_manager.refresh_all_core_states();

        // Check if target state is reached
        if (all_routers_in_state(target_state)) {
            return true;
        }

        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed >= timeout) {
            return false;
        }

        // Sleep before next poll (NO BUSY-WAIT)
        std::this_thread::sleep_for(poll_interval);
    }
}

RouterStateCommon FabricCommandInterface::get_router_state(
    const FabricNodeId& fabric_node_id,
    chan_id_t channel_id) {
    // FR-5: Query state of specific router
    auto& state_manager = control_plane_.get_state_manager();
    return state_manager.get_core_state(fabric_node_id, channel_id);
}

std::vector<std::pair<FabricNodeId, chan_id_t>> FabricCommandInterface::get_all_router_cores() const {
    // FR-2: Get all active router cores from control plane
    std::vector<std::pair<FabricNodeId, chan_id_t>> result;

    // Get all mesh IDs
    const auto& mesh_ids = control_plane_.get_user_physical_mesh_ids();

    for (MeshId mesh_id : mesh_ids) {
        // Get all chip IDs for this mesh
        const auto& chip_ids = control_plane_.get_physical_chip_ids(mesh_id);

        for (ChipId chip_id : chip_ids) {
            // Get fabric node ID for this chip
            FabricNodeId fabric_node_id = control_plane_.get_fabric_node_id_from_physical_chip_id(chip_id);

            // Get all active channels for this device
            const auto& channels = control_plane_.get_active_fabric_eth_channels_for_device(chip_id);

            for (chan_id_t channel_id : channels) {
                result.emplace_back(fabric_node_id, channel_id);
            }
        }
    }

    return result;
}

}  // namespace tt::tt_fabric::test_utils
