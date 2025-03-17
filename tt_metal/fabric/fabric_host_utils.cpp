// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/mesh_graph.hpp>

#include "fabric_host_utils.hpp"

namespace tt::tt_fabric {

tt::tt_fabric::FabricEriscDatamoverConfig get_default_fabric_config() {
    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);
    return tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, 1, 2);
}

// src and neighbor chips should be adjacent
void append_fabric_connection_rt_args(
    chip_id_t src_phys_chip_id,
    chip_id_t dst_phys_chip_id,
    routing_plane_id_t routing_plane,
    tt::tt_metal::Program& worker_program,
    CoreRangeSet worker_cores,
    std::vector<uint32_t>& worker_args) {
    TT_ASSERT(src_phys_chip_id != dst_phys_chip_id);

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // for now, both the src and dest chips should be on the same mesh
    auto [src_mesh_id, src_logical_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(src_phys_chip_id);
    auto [dst_mesh_id, dst_logical_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(dst_phys_chip_id);
    TT_ASSERT(src_mesh_id == dst_mesh_id);

    // available fabric eth channels
    const auto fabric_ethernet_channels = tt::Cluster::instance().get_fabric_ethernet_channels(src_phys_chip_id);

    // list of available eth cores
    chan_id_t fabric_router_channel;
    bool found_fabric_router = false;

    // currently the src and dest should be adjacent, until the control plane has enough logic to check on the same line
    auto src_chip_neighbors = tt::Cluster::instance().get_ethernet_cores_grouped_by_connected_chips(src_phys_chip_id);
    for (const auto& [neighbor_chip, eth_cores] : src_chip_neighbors) {
        if (neighbor_chip != dst_phys_chip_id) {
            continue;
        }

        for (const auto& eth_core : eth_cores) {
            auto eth_chan =
                tt::Cluster::instance().get_soc_desc(neighbor_chip).logical_eth_core_to_chan_map.at(eth_core);

            // check the routing plane
            if (routing_plane != control_plane->get_routing_plane_id(eth_chan)) {
                continue;
            }

            // check if is in the list of active fabric channel
            if (std::find(fabric_ethernet_channels.begin(), fabric_ethernet_channels.end(), eth_chan) !=
                fabric_ethernet_channels.end()) {
                fabric_router_channel = eth_chan;
                found_fabric_router = true;
                break;
            }
        }

        if (found_fabric_router) {
            break;
        }
    }

    TT_ASSERT(found_fabric_router);

    // get config
    const auto edm_config = get_default_fabric_config();
    CoreCoord fabric_router_virtual_core =
        tt::Cluster::instance().get_virtual_eth_core_from_channel(src_phys_chip_id, fabric_router_channel);

    // generate connection
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

    // create semaphores
    auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(worker_program, worker_cores, 0);
    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(worker_program, worker_cores, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(worker_program, worker_cores, 0);

    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_args);
}

}  // namespace tt::tt_fabric
