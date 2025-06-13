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

namespace {

// checks if the connection b/w src and dst is a connection b/w TG gateway and a remote chip
bool is_TG_gateway_connection(const chip_id_t src_chip_id, const chip_id_t dst_chip_id) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::ClusterType::TG) {
        return false;
    }

    const auto mmio_chip_id1 =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(src_chip_id);
    const auto mmio_chip_id2 =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dst_chip_id);

    // both of the chips should have the same associated mmio device and
    // one of the chips should be the mmio device itself
    if (mmio_chip_id1 == mmio_chip_id2 && (mmio_chip_id1 == src_chip_id || mmio_chip_id2 == dst_chip_id)) {
        return true;
    }

    return false;
}

}  // namespace

namespace tt::tt_fabric {

size_t get_tt_fabric_channel_buffer_size_bytes() {
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_fabric_context().get_fabric_channel_buffer_size_bytes();
}

size_t get_tt_fabric_packet_header_size_bytes() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_fabric_context().get_fabric_packet_header_size_bytes();
}

void append_fabric_connection_rt_args(
    const chip_id_t src_chip_id,
    const chip_id_t dst_chip_id,
    const uint32_t link_idx,
    tt::tt_metal::Program& worker_program,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    CoreType core_type) {
    TT_FATAL(
        src_chip_id != dst_chip_id,
        "Expected different src and dst chip ids but got same, src: {}, dst: {}",
        src_chip_id,
        dst_chip_id);

    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();

    const auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_chip_id);
    const auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_chip_id);

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    const bool is_2d_fabric = topology == Topology::Mesh;

    // Make an exception for TG gateway connections. TG gateways are on a different mesh compared to remote chips
    // but the routing is simple and doesnt need any special inter-mesh handling
    if (!is_2d_fabric && !is_TG_gateway_connection(src_chip_id, dst_chip_id)) {
        TT_FATAL(
            src_fabric_node_id.mesh_id == dst_fabric_node_id.mesh_id,
            "Currently only the chips on the same mesh are supported for 1D fabric. Src mesh id: {}, Dst mesh id: {}",
            src_fabric_node_id.mesh_id,
            dst_fabric_node_id.mesh_id);
    }

    // get the direction in which the data will be forwarded from the src_chip_id
    std::optional<RoutingDirection> forwarding_direction;
    if (is_2d_fabric) {
        forwarding_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    } else {
        // TODO: Workaround for #22524 routing tables not having wraparound links
        // for 1D fabric, we loop to match the dst chip since we need to ensure src and dst are on the same line
        // remove this once control plane has row/col info/view
        for (const auto& direction : FabricContext::routing_directions) {
            // This assumes all neighbor chips to the dst mesh are the same
            auto neighbors = control_plane.get_chip_neighbors(src_fabric_node_id, direction);
            auto neighbor_mesh_chips = neighbors.find(dst_fabric_node_id.mesh_id);
            if (neighbor_mesh_chips == neighbors.end() ||
                (std::find(
                     neighbor_mesh_chips->second.begin(),
                     neighbor_mesh_chips->second.end(),
                     dst_fabric_node_id.chip_id) == neighbor_mesh_chips->second.end())) {
                continue;
            }

            forwarding_direction = direction;
            break;
        }
    }
    TT_FATAL(
        forwarding_direction.has_value(),
        "Could not find any forwarding direction from src {} to dst {}",
        src_chip_id,
        dst_chip_id);

    const auto candidate_eth_chans =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, forwarding_direction.value());
    TT_FATAL(
        link_idx < candidate_eth_chans.size(),
        "requested link idx {}, out of bounds, max available {}",
        link_idx,
        candidate_eth_chans.size());

    const auto forwarding_links =
        get_forwarding_link_indices_in_direction(src_chip_id, dst_chip_id, forwarding_direction.value());
    TT_FATAL(
        std::find(forwarding_links.begin(), forwarding_links.end(), link_idx) != forwarding_links.end(),
        "requested link idx {}, cannot be used for forwarding b/w src {} and dst {}",
        link_idx,
        src_chip_id,
        dst_chip_id);

    const auto fabric_router_channel = candidate_eth_chans[link_idx];
    const auto router_direction = control_plane.routing_direction_to_eth_direction(forwarding_direction.value());

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
        .edm_direction = router_direction};

    auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(worker_program, {worker_core}, 0, core_type);
    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(worker_program, {worker_core}, 0, core_type);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(worker_program, {worker_core}, 0, core_type);
    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_args);
}

}  // namespace tt::tt_fabric
