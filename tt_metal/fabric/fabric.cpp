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
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/device.hpp>
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
bool is_TG_gateway_connection(
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id, const tt::tt_fabric::FabricNodeId& dst_fabric_node_id) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::ClusterType::TG) {
        return false;
    }
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    chip_id_t src_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    chip_id_t dst_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);
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

size_t get_tt_fabric_max_payload_size_bytes() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_fabric_context().get_fabric_max_payload_size_bytes();
}

FabricNodeId get_fabric_node_id_from_physical_chip_id(chip_id_t physical_chip_id) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);
}

std::vector<chan_id_t> get_active_fabric_eth_routing_planes_in_direction(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_active_fabric_eth_routing_planes_in_direction(fabric_node_id, routing_direction);
}

std::unordered_map<MeshId, MeshShape> get_physical_mesh_shapes() {
    std::unordered_map<MeshId, MeshShape> mesh_shapes;
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    for (auto mesh_id : control_plane.get_user_physical_mesh_ids()) {
        mesh_shapes[mesh_id] = control_plane.get_physical_mesh_shape(mesh_id);
    }
    return mesh_shapes;
}

void append_fabric_connection_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    const uint32_t link_idx,
    tt::tt_metal::Program& worker_program,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    CoreType core_type) {
    TT_FATAL(
        src_fabric_node_id != dst_fabric_node_id,
        "Expected different src and dst chip ids but got same, Src: {}, Dst: {}",
        src_fabric_node_id,
        dst_fabric_node_id);

    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    const bool is_2d_fabric = topology == Topology::Mesh;

    // Make an exception for TG gateway connections. TG gateways are on a different mesh compared to remote chips
    // but the routing is simple and doesnt need any special inter-mesh handling
    if (!is_2d_fabric && !is_TG_gateway_connection(src_fabric_node_id, dst_fabric_node_id)) {
        TT_FATAL(
            src_fabric_node_id.mesh_id == dst_fabric_node_id.mesh_id,
            "Currently only the chips on the same mesh are supported for 1D fabric. Src: {}, Dst: {}",
            src_fabric_node_id,
            dst_fabric_node_id);
    }

    // get the direction in which the data will be forwarded from the src_fabric_node_id
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
        src_fabric_node_id,
        dst_fabric_node_id);

    const auto candidate_eth_chans =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, forwarding_direction.value());
    TT_FATAL(
        link_idx < candidate_eth_chans.size(),
        "Requested link index {} is out of bounds. {} ethernet channels available to forward b/w src {} and dst {}",
        link_idx,
        candidate_eth_chans.size(),
        src_fabric_node_id,
        dst_fabric_node_id);

    const auto forwarding_links =
        get_forwarding_link_indices_in_direction(src_fabric_node_id, dst_fabric_node_id, forwarding_direction.value());
    TT_FATAL(
        std::find(forwarding_links.begin(), forwarding_links.end(), link_idx) != forwarding_links.end(),
        "Requested link index {} cannot be used for forwarding b/w src {} and dst {}. Valid forwarding links are {}",
        link_idx,
        src_fabric_node_id,
        dst_fabric_node_id,
        forwarding_links);

    const auto fabric_router_channel = candidate_eth_chans[link_idx];
    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(worker_program, {worker_core}, 0, core_type);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(worker_program, {worker_core}, 0, core_type);
    if (core_type == CoreType::WORKER) {
        append_worker_to_fabric_edm_sender_rt_args(
            fabric_router_channel, worker_teardown_semaphore_id, worker_buffer_index_semaphore_id, worker_args);
    } else {
        // TODO: will be deprecated. currently for ethernet dispatch case
        //       ethernet core need to have same memory mapping as worker
        const auto router_direction = control_plane.routing_direction_to_eth_direction(forwarding_direction.value());

        // src_chip_id is still required to get the fabric_router_virtual_core from tt_cluster
        chip_id_t src_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);

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
        append_worker_to_fabric_edm_sender_rt_args(
            edm_connection,
            worker_flow_control_semaphore_id,
            worker_teardown_semaphore_id,
            worker_buffer_index_semaphore_id,
            worker_args);
    }
}

std::vector<uint32_t> get_forwarding_link_indices(
    const FabricNodeId& src_fabric_node_id, const FabricNodeId& dst_fabric_node_id) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // find the forwarding direction b/w src and dest chip
    const auto& forwarding_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    if (!forwarding_direction.has_value()) {
        return {};
    }

    return get_forwarding_link_indices_in_direction(
        src_fabric_node_id, dst_fabric_node_id, forwarding_direction.value());
}

tt::tt_fabric::Topology get_fabric_topology() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_fabric_context().get_fabric_topology();
}

FabricConfig GetFabricConfig() { return tt::tt_metal::MetalContext::instance().get_fabric_config(); }

void SetFabricConfig(
    FabricConfig fabric_config, FabricReliabilityMode reliability_mode, std::optional<uint8_t> num_routing_planes) {
    tt::tt_metal::MetalContext::instance().set_fabric_config(fabric_config, reliability_mode, num_routing_planes);
}

namespace experimental {

size_t get_number_of_available_routing_planes(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, size_t cluster_axis, size_t row_or_col) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Get any device from the cluster to determine fabric node
    // For now, use chip 0 as a representative device

    size_t row_idx = cluster_axis == 0 ? 0 : row_or_col;
    size_t col_idx = cluster_axis == 0 ? row_or_col : 0;
    auto* first_chip = mesh_device.get_device(row_idx, col_idx);
    chip_id_t first_chip_id = first_chip->id();
    auto fabric_node_in_row_or_col = control_plane.get_fabric_node_id_from_physical_chip_id(first_chip_id);

    // Map cluster axis to routing directions
    constexpr std::array<std::array<RoutingDirection, 2>, 2> cluster_axis_directions_to_check = {
        std::array<RoutingDirection, 2>{RoutingDirection::N, RoutingDirection::S},
        std::array<RoutingDirection, 2>{RoutingDirection::E, RoutingDirection::W}};

    TT_FATAL(
        cluster_axis < cluster_axis_directions_to_check.size(),
        "Invalid cluster axis {}. Must be less than {}",
        cluster_axis,
        cluster_axis_directions_to_check.size());
    const auto& directions_to_check = cluster_axis_directions_to_check[cluster_axis];

    size_t planes_dir0 =
        control_plane.get_num_available_routing_planes_in_direction(fabric_node_in_row_or_col, directions_to_check[0]);
    size_t planes_dir1 =
        control_plane.get_num_available_routing_planes_in_direction(fabric_node_in_row_or_col, directions_to_check[1]);
    TT_FATAL(planes_dir0 == planes_dir1, "Routing planes are not equal");
    return planes_dir0;
}

}  // namespace experimental

}  // namespace tt::tt_fabric
