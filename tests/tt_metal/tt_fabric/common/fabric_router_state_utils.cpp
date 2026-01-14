// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_state_utils.hpp"
#include "tt_metal/common/logger.hpp"

namespace tt::tt_fabric::test_utils {

const char* router_state_to_string(RouterStateCommon state) {
    switch (state) {
        case RouterStateCommon::INITIALIZING:
            return "INITIALIZING";
        case RouterStateCommon::RUNNING:
            return "RUNNING";
        case RouterStateCommon::PAUSED:
            return "PAUSED";
        case RouterStateCommon::DRAINING:
            return "DRAINING";
        case RouterStateCommon::RETRAINING:
            return "RETRAINING";
        default:
            return "UNKNOWN";
    }
}

void log_all_router_states(
    ControlPlane& control_plane,
    const std::vector<MeshId>& mesh_ids) {

    log_info(LogTest, "=== Router State Summary ===");

    for (const auto& mesh_id : mesh_ids) {
        log_info(LogTest, "Mesh ID: {}", mesh_id);

        // TODO: Implement actual querying of devices and channels
        // Placeholder implementation - will be filled in by developer
        // Pattern from spec:
        // - Get all devices in mesh via control_plane.get_devices_in_mesh(mesh_id)
        // - For each device, get all channels via control_plane.get_channels_for_device(node_id)
        // - For each channel, get state via control_plane.get_router_state(node_id, channel_id)
        // - Log using router_state_to_string(state)
    }

    // Log summary counts
    auto counts = count_routers_by_state(control_plane, mesh_ids);
    log_info(LogTest, "State Counts:");
    for (const auto& [state, count] : counts) {
        log_info(LogTest, "  {}: {}", router_state_to_string(state), count);
    }

    log_info(LogTest, "============================");
}

std::map<RouterStateCommon, uint32_t> count_routers_by_state(
    ControlPlane& control_plane,
    const std::vector<MeshId>& mesh_ids) {

    std::map<RouterStateCommon, uint32_t> counts;

    for (const auto& mesh_id : mesh_ids) {
        // TODO: Implement actual counting
        // Placeholder implementation - will be filled in by developer
        // Pattern from spec:
        // - Get all devices in mesh via control_plane.get_devices_in_mesh(mesh_id)
        // - For each device, get all channels via control_plane.get_channels_for_device(node_id)
        // - For each channel, get state via control_plane.get_router_state(node_id, channel_id)
        // - Increment the corresponding count in the map
    }

    return counts;
}

} // namespace tt::tt_fabric::test_utils
