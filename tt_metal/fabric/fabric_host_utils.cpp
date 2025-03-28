// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/fabric_host_utils.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_graph.hpp>

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <variant>

#include "assert.hpp"
#include "control_plane.hpp"
#include "fabric_edm_packet_header.hpp"

#include "metal_soc_descriptor.h"
#include "impl/context/metal_context.hpp"
#include <magic_enum/magic_enum.hpp>
#include <umd/device/types/xy_pair.h>

namespace tt {
namespace tt_metal {
class Program;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_fabric {

bool is_1d_fabric_config(const tt::tt_metal::FabricConfig& fabric_config) {
    return fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_1D_RING;
}

bool is_2d_fabric_config(const tt::tt_metal::FabricConfig& fabric_config) {
    return fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D ||
           fabric_config == tt::tt_metal::FabricConfig::FABRIC_2D_PUSH;
}

Topology get_1d_topology(const tt::tt_metal::FabricConfig& fabric_config) {
    switch (fabric_config) {
        case tt::tt_metal::FabricConfig::FABRIC_1D: return tt::tt_fabric::Topology::Linear;
        case tt::tt_metal::FabricConfig::FABRIC_1D_RING: return tt::tt_fabric::Topology::Ring;
        case tt::tt_metal::FabricConfig::DISABLED:
        case tt::tt_metal::FabricConfig::FABRIC_2D:
        case tt::tt_metal::FabricConfig::FABRIC_2D_PUSH:
        case tt::tt_metal::FabricConfig::CUSTOM:
            TT_THROW("Unsupported fabric config for 1D: {}", magic_enum::enum_name(fabric_config));
    }
    return tt::tt_fabric::Topology::Linear;
}

// TODO: We should store this somewhere instead of constantly regenerating
tt::tt_fabric::FabricEriscDatamoverConfig get_1d_fabric_config() {
    constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);
    auto control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    tt::tt_metal::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_config();
    Topology topology = get_1d_topology(fabric_config);
    return tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, topology);
}

void append_fabric_connection_rt_args(
    chip_id_t src_chip_id,
    chip_id_t dst_chip_id,
    uint32_t link_idx,
    tt::tt_metal::Program& worker_program,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args) {
    TT_FATAL(
        src_chip_id != dst_chip_id,
        "Expected different src and dst chip ids but got same, src: {}, dst: {}",
        src_chip_id,
        dst_chip_id);

    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    // for now, both the src and dest chips should be on the same mesh
    auto [src_mesh_id, src_logical_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(src_chip_id);
    auto [dst_mesh_id, dst_logical_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(dst_chip_id);
    TT_FATAL(
        src_mesh_id == dst_mesh_id,
        "Currently only the chips on the same mesh are supported. Src mesh id: {}, Dst mesh id: {}",
        src_mesh_id,
        dst_mesh_id);

    auto routing_directions = {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W};
    std::optional<std::set<chan_id_t>> candidate_ethernet_cores;
    // mimic the 1d fabric connection setup steps to correctly find the candidate links
    for (const auto& direction : routing_directions) {
        auto neighbors = control_plane->get_intra_chip_neighbors(src_mesh_id, src_logical_chip_id, direction);
        if (neighbors.empty() || neighbors[0] != dst_logical_chip_id) {
            continue;
        }

        candidate_ethernet_cores =
            control_plane->get_active_fabric_eth_channels_in_direction(src_mesh_id, src_logical_chip_id, direction);
    }

    TT_FATAL(
        candidate_ethernet_cores.has_value(), "Could not find any fabric ethernet cores between src and dst chips");

    TT_FATAL((link_idx + 1) <= candidate_ethernet_cores.value().size(), "link idx out of bounds");

    auto it = candidate_ethernet_cores.value().begin();
    std::advance(it, link_idx);
    auto fabric_router_channel = *it;

    const auto& edm_config = get_1d_fabric_config();
    CoreCoord fabric_router_virtual_core =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
            src_chip_id, fabric_router_channel);

    tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
        .edm_noc_x = fabric_router_virtual_core.x,
        .edm_noc_y = fabric_router_virtual_core.y,
        .edm_buffer_base_addr = edm_config.sender_channels_base_address[0],
        .num_buffers_per_channel = edm_config.sender_channels_num_buffers[0],
        .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[0],
        .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[0],
        .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[0],
        .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
        .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[0],
        .persistent_fabric = true};

    auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(worker_program, {worker_core}, 0);
    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(worker_program, {worker_core}, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(worker_program, {worker_core}, 0);

    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_args);
}

}  // namespace tt::tt_fabric
