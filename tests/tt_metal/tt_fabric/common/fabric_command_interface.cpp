// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fabric_command_interface.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <thread>
#include <chrono>

namespace tt::tt_fabric {

FabricCommandInterface::FabricCommandInterface(const ControlPlane& control_plane) :
    control_plane(control_plane) {
    // Constructor - control_plane passed to methods as needed
}

void FabricCommandInterface::issue_command_to_routers(RouterCommand router_command) const {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto router_cmd_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ROUTER_COMMAND);

        const auto& router_cores = get_all_router_cores();
        for (const auto& [fabric_node_id, channel_id] : router_cores) {
            ChipId physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            CoreCoord eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, channel_id);

            cluster.write_core(
            &router_command,
            sizeof(RouterCommand),
            tt_cxy_pair(physical_chip_id, eth_core),
            router_cmd_addr);
    }
}

void FabricCommandInterface::pause_routers() const {
    issue_command_to_routers(RouterCommand::PAUSE);
}

void FabricCommandInterface::resume_routers() const {
    issue_command_to_routers(RouterCommand::RUN);
}

bool FabricCommandInterface::all_routers_in_state(
    RouterState expected_state) const {
    const auto& router_cores = get_all_router_cores();

    // Empty topology returns false (no routers to verify)
    if (router_cores.empty()) {
        return false;
    }

    for (const auto& [fabric_node_id, channel_id] : router_cores) {
        RouterState state = read_router_state(fabric_node_id, channel_id);

        if (state != expected_state) {
            return false;
        }
    }

    return true;
}

bool FabricCommandInterface::wait_for_pause(
    std::chrono::milliseconds timeout) const {
    return wait_for_state(RouterState::PAUSED, timeout);
}

bool FabricCommandInterface::wait_for_state(
    RouterState target_state,
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval) const {
    // Wait for all routers to reach target state with timeout
    // Uses polling with std::this_thread::sleep_for between polls (NO BUSY-WAIT)

    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        if (all_routers_in_state(target_state)) {
            return true;
        }

        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed >= timeout) {
            const auto& router_cores = get_all_router_cores();

            // Empty topology returns false (no routers to verify)
            if (router_cores.empty()) {
                return false;
            }

            for (const auto& [fabric_node_id, channel_id] : router_cores) {
                RouterState state = read_router_state(fabric_node_id, channel_id);

                log_debug(LogTest, "Router state: {} (fabric_node_id: (m={}, c={}), channel_id: {})", state, fabric_node_id.mesh_id, fabric_node_id.chip_id, channel_id);
            }
            return false;
        }

        std::this_thread::sleep_for(poll_interval);
    }
}

RouterState FabricCommandInterface::read_router_state(
    const FabricNodeId& fabric_node_id,
    chan_id_t channel_id) const {
    // Query state of specific router
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto router_state_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ROUTER_STATE);

    ChipId physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
    CoreCoord eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, channel_id);

    RouterState state;
    cluster.read_core(
        &state,
        sizeof(RouterState),
        tt_cxy_pair(physical_chip_id, eth_core),
        router_state_addr);

    return state;
}

std::vector<std::pair<FabricNodeId, chan_id_t>> FabricCommandInterface::get_all_router_cores() const {
    // Get all active router cores from control plane
    std::vector<std::pair<FabricNodeId, chan_id_t>> result;

    const auto& mesh_ids = control_plane.get_user_physical_mesh_ids();

    for (MeshId mesh_id : mesh_ids) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh_id);
        size_t num_chips = mesh_shape.mesh_size();

        for (size_t chip_idx = 0; chip_idx < num_chips; ++chip_idx) {
            FabricNodeId fabric_node_id(mesh_id, static_cast<uint32_t>(chip_idx));

            auto channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);

            for (const auto& [channel_id, direction] : channels) {
                result.emplace_back(fabric_node_id, channel_id);
            }
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
