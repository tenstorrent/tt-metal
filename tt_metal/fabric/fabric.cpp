// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <optional>
#include <set>
#include <vector>

#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.h>

#include "fabric_host_utils.hpp"
#include "fabric_context.hpp"

namespace tt {
namespace tt_metal {
class Program;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_fabric {

size_t get_tt_fabric_channel_buffer_size_bytes() {
    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    return control_plane->get_fabric_context().get_fabric_channel_buffer_size_bytes();
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

    auto [src_mesh_id, src_logical_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(src_chip_id);
    auto [dst_mesh_id, dst_logical_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(dst_chip_id);

    const auto& fabric_context = control_plane->get_fabric_context();
    auto topology = fabric_context.get_fabric_topology();
    bool is_2d_fabric = topology == Topology::Mesh;
    if (!is_2d_fabric) {
        TT_FATAL(
            src_mesh_id == dst_mesh_id,
            "Currently only the chips on the same mesh are supported for 1D fabric. Src mesh id: {}, Dst mesh id: {}",
            src_mesh_id,
            dst_mesh_id);
    }

    auto routing_directions = {RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W};
    std::optional<std::set<chan_id_t>> candidate_ethernet_cores;
    // mimic the 1d fabric connection setup steps to correctly find the candidate links
    for (const auto& direction : routing_directions) {
        // This assumes all neighbor chips to the dst mesh are the same
        auto neighbors = control_plane->get_chip_neighbors(src_mesh_id, src_logical_chip_id, direction);
        auto neighbor_mesh_chips = neighbors.find(dst_mesh_id);
        if (neighbor_mesh_chips == neighbors.end() || neighbor_mesh_chips->second[0] != dst_logical_chip_id) {
            continue;
        }

        candidate_ethernet_cores =
            control_plane->get_active_fabric_eth_channels_in_direction(src_mesh_id, src_logical_chip_id, direction);
        break;
    }

    TT_FATAL(
        candidate_ethernet_cores.has_value(),
        "Could not find any fabric ethernet cores between src {} and dst {} chips",
        src_chip_id,
        dst_chip_id);

    TT_FATAL(link_idx < candidate_ethernet_cores.value().size(), "link idx out of bounds");

    auto fabric_router_channel = get_ordered_fabric_eth_chans(src_chip_id, candidate_ethernet_cores.value())[link_idx];
    auto router_direction =
        control_plane->get_eth_chan_direction(src_mesh_id, src_logical_chip_id, fabric_router_channel);

    CoreCoord fabric_router_virtual_core =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
            src_chip_id, fabric_router_channel);

    const auto& edm_config = fabric_context.get_fabric_router_config();
    const auto sender_channel = is_2d_fabric ? router_direction : 0;
    tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
        .edm_noc_x = fabric_router_virtual_core.x,
        .edm_noc_y = fabric_router_virtual_core.y,
        .edm_buffer_base_addr = edm_config.sender_channels_base_address[sender_channel],
        .num_buffers_per_channel = edm_config.sender_channels_num_buffers[sender_channel],
        .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[sender_channel],
        .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[sender_channel],
        .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[sender_channel],
        .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
        .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[sender_channel],
        .persistent_fabric = true,
        .edm_direction = router_direction};

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
