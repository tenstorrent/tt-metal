// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/mesh_graph.hpp>

#include <tt-metalium/fabric_host_utils.hpp>

namespace tt::tt_fabric {

tt::tt_fabric::FabricEriscDatamoverConfig get_default_fabric_config() {
    constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);
    return tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size, 1, 2);
}

void append_fabric_connection_rt_args(
    chip_id_t src_chip_id,
    chip_id_t dst_chip_id,
    routing_plane_id_t routing_plane,
    tt::tt_metal::Program& worker_program,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args) {
    TT_FATAL(
        src_chip_id != dst_chip_id,
        "Expected different src and dst chip ids but got same, src: {}, dst: {}",
        src_chip_id,
        dst_chip_id);

    auto* control_plane = tt::Cluster::instance().get_control_plane();

    // for now, both the src and dest chips should be on the same mesh
    auto [src_mesh_id, src_logical_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(src_chip_id);
    auto [dst_mesh_id, dst_logical_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(dst_chip_id);
    TT_FATAL(
        src_mesh_id == dst_mesh_id,
        "Currently only the chips on the same mesh are supported. Src mesh id: {}, Dst mesh id: {}",
        src_mesh_id,
        dst_mesh_id);

    // currently the src and dest should be adjacent, until the control plane has enough logic to check on the same line
    const auto& neighbor_chips_and_cores =
        tt::Cluster::instance().get_ethernet_cores_grouped_by_connected_chips(src_chip_id);
    const auto& dst_chip_and_cores = neighbor_chips_and_cores.find(dst_chip_id);
    TT_FATAL(dst_chip_and_cores != neighbor_chips_and_cores.end(), "Src and Dst chips are not physically adjacent");

    const auto& fabric_ethernet_channels = tt::Cluster::instance().get_fabric_ethernet_channels(src_chip_id);
    const auto& candidate_ethernet_cores = dst_chip_and_cores->second;
    const auto& logical_eth_core_to_chan_map =
        tt::Cluster::instance().get_soc_desc(src_chip_id).logical_eth_core_to_chan_map;

    std::optional<chan_id_t> fabric_router_channel;

    for (const auto& eth_core : candidate_ethernet_cores) {
        auto eth_chan = logical_eth_core_to_chan_map.at(eth_core);

        // selected channel should match the requested routing plane and should be one of the active fabric channels
        if (routing_plane != control_plane->get_routing_plane_id(eth_chan)) {
            continue;
        }
        if (fabric_ethernet_channels.find(eth_chan) != fabric_ethernet_channels.end()) {
            fabric_router_channel = eth_chan;
            break;
        }
    }

    TT_FATAL(
        fabric_router_channel.has_value(),
        "Could not find any fabric router for requested routing plane: {}",
        routing_plane);

    const auto& edm_config = get_default_fabric_config();
    CoreCoord fabric_router_virtual_core =
        tt::Cluster::instance().get_virtual_eth_core_from_channel(src_chip_id, fabric_router_channel.value());

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
