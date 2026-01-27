// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fabric_command_interface.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <thread>
#include <chrono>

namespace tt::tt_fabric {

FabricCommandInterface::FabricCommandInterface([[maybe_unused]] ControlPlane& control_plane) {
    // Constructor - control_plane passed to methods as needed
}

void FabricCommandInterface::pause_routers(const ControlPlane& control_plane) const {
    // FR-4: Issue PAUSE command to all active routers
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto router_cmd_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ROUTER_COMMAND);

    RouterCommand cmd = RouterCommand::PAUSE;

    const auto& router_cores = get_all_router_cores(control_plane);
    for (const auto& [fabric_node_id, channel_id] : router_cores) {
        ChipId physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
        CoreCoord eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, channel_id);

        cluster.write_core(
            &cmd,
            sizeof(RouterCommand),
            tt_cxy_pair(physical_chip_id, eth_core),
            router_cmd_addr);
    }
}

void FabricCommandInterface::resume_routers(const ControlPlane& control_plane) const {
    // FR-4: Issue RUN command to all active routers
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto router_cmd_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ROUTER_COMMAND);

    RouterCommand cmd = RouterCommand::RUN;

    const auto& router_cores = get_all_router_cores(control_plane);
    for (const auto& [fabric_node_id, channel_id] : router_cores) {
        ChipId physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
        CoreCoord eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, channel_id);

        cluster.write_core(
            &cmd,
            sizeof(RouterCommand),
            tt_cxy_pair(physical_chip_id, eth_core),
            router_cmd_addr);
    }
}

bool FabricCommandInterface::all_routers_in_state(
    const ControlPlane& control_plane,
    RouterStateCommon expected_state) const {
    // FR-5: Check if all routers are in specified state
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto router_state_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ROUTER_STATE);

    const auto& router_cores = get_all_router_cores(control_plane);

    // Empty topology returns false (no routers to verify)
    if (router_cores.empty()) {
        return false;
    }

    for (const auto& [fabric_node_id, channel_id] : router_cores) {
        ChipId physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
        CoreCoord eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, channel_id);

        RouterStateCommon state;
        cluster.read_core(
            &state,
            sizeof(RouterStateCommon),
            tt_cxy_pair(physical_chip_id, eth_core),
            router_state_addr);

        if (state != expected_state) {
            return false;
        }
    }

    return true;
}

bool FabricCommandInterface::wait_for_pause(
    const ControlPlane& control_plane,
    std::chrono::milliseconds timeout) const {
    // FR-6: Wrapper for wait_for_state with PAUSED state
    return wait_for_state(control_plane, RouterStateCommon::PAUSED, timeout);
}

bool FabricCommandInterface::wait_for_state(
    const ControlPlane& control_plane,
    RouterStateCommon target_state,
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval) const {
    // FR-6: Wait for all routers to reach target state with timeout
    // Uses polling with std::this_thread::sleep_for between polls (NO BUSY-WAIT)

    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        // Check if target state is reached
        if (all_routers_in_state(control_plane, target_state)) {
            return true;
        }

        // Check timeout
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed >= timeout) {
            // report the states of all routers
            auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            const auto& hal = tt::tt_metal::MetalContext::instance().hal();

            const auto router_state_addr = hal.get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ROUTER_STATE);

            const auto& router_cores = get_all_router_cores(control_plane);

            // Empty topology returns false (no routers to verify)
            if (router_cores.empty()) {
                return false;
            }

            for (const auto& [fabric_node_id, channel_id] : router_cores) {
                ChipId physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
                CoreCoord eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, channel_id);

                RouterStateCommon state;
                cluster.read_core(
                    &state,
                    sizeof(RouterStateCommon),
                    tt_cxy_pair(physical_chip_id, eth_core),
                    router_state_addr);

                log_info(LogTest, "Router state: {} (fabric_node_id: {}, channel_id: {})", state, fabric_node_id.mesh_id, fabric_node_id.chip_id, channel_id);
            }
            return false;
        }

        // Sleep before next poll (NO BUSY-WAIT)
        std::this_thread::sleep_for(poll_interval);
    }
}

RouterStateCommon FabricCommandInterface::get_router_state(
    const ControlPlane& control_plane,
    const FabricNodeId& fabric_node_id,
    chan_id_t channel_id) const {
    // FR-5: Query state of specific router
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto router_state_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::ROUTER_STATE);

    ChipId physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
    CoreCoord eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, channel_id);

    RouterStateCommon state;
    cluster.read_core(
        &state,
        sizeof(RouterStateCommon),
        tt_cxy_pair(physical_chip_id, eth_core),
        router_state_addr);

    return state;
}

std::vector<std::pair<FabricNodeId, chan_id_t>> FabricCommandInterface::get_all_router_cores(
    const ControlPlane& control_plane) const {
    // FR-2: Get all active router cores from control plane
    std::vector<std::pair<FabricNodeId, chan_id_t>> result;

    // Get all mesh IDs
    const auto& mesh_ids = control_plane.get_user_physical_mesh_ids();

    for (MeshId mesh_id : mesh_ids) {
        // Get mesh shape to iterate over all chips
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh_id);
        size_t num_chips = mesh_shape.mesh_size();

        for (size_t chip_idx = 0; chip_idx < num_chips; ++chip_idx) {
            // Construct fabric node id for this chip position
            FabricNodeId fabric_node_id(mesh_id, static_cast<uint32_t>(chip_idx));

            // Get all active channels for this device
            auto channels = control_plane.get_active_fabric_eth_channels(fabric_node_id);

            for (const auto& [channel_id, direction] : channels) {
                result.emplace_back(fabric_node_id, channel_id);
            }
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
