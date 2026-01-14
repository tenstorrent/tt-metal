// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fabric_command_interface.hpp"

namespace tt::tt_fabric::test_utils {

FabricCommandInterface::FabricCommandInterface(ControlPlane& control_plane)
    : control_plane_(control_plane) {
}

void FabricCommandInterface::pause_routers() {
    // TODO: Implement FR-4
}

void FabricCommandInterface::resume_routers() {
    // TODO: Implement FR-4 (RUN command)
}

bool FabricCommandInterface::all_routers_in_state(RouterStateCommon expected_state) {
    // TODO: Implement FR-5
    return false;
}

bool FabricCommandInterface::wait_for_pause(std::chrono::milliseconds timeout) {
    // TODO: Implement FR-6
    return wait_for_state(RouterStateCommon::PAUSED, timeout);
}

bool FabricCommandInterface::wait_for_state(
    RouterStateCommon target_state,
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval) {
    // TODO: Implement FR-6
    // Must use std::this_thread::sleep_for between polls
    return false;
}

RouterStateCommon FabricCommandInterface::get_router_state(
    const FabricNodeId& fabric_node_id,
    chan_id_t channel_id) {
    // TODO: Implement FR-5
    return RouterStateCommon::RUNNING;
}

std::vector<std::pair<FabricNodeId, chan_id_t>> FabricCommandInterface::get_all_router_cores() const {
    // TODO: Implement FR-2
    return {};
}

}  // namespace tt::tt_fabric::test_utils
