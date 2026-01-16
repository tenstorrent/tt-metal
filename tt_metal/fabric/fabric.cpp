// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include "erisc_datamover_builder.hpp"
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include <optional>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include <umd/device/types/xy_pair.hpp>

#include "fabric_host_utils.hpp"
#include "fabric_context.hpp"
#include "fabric_builder_context.hpp"

namespace tt::tt_metal {
class Program;
}  // namespace tt::tt_metal

namespace {

// checks if the connection b/w src and dst is a connection b/w TG gateway and a remote chip
bool is_TG_gateway_connection(
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id, const tt::tt_fabric::FabricNodeId& dst_fabric_node_id) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::TG) {
        return false;
    }
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    tt::ChipId src_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    tt::ChipId dst_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);
    const auto mmio_chip_id1 =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(src_chip_id);
    const auto mmio_chip_id2 =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dst_chip_id);

    // both of the chips should have the same associated mmio device and
    // one of the chips should be the mmio device itself
    return mmio_chip_id1 == mmio_chip_id2 && (mmio_chip_id1 == src_chip_id || mmio_chip_id2 == dst_chip_id);
}

}  // namespace

namespace tt::tt_fabric {

size_t get_tt_fabric_channel_buffer_size_bytes() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
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

FabricNodeId get_fabric_node_id_from_physical_chip_id(ChipId physical_chip_id) {
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

template <typename ProgramOrDescriptor>
void append_fabric_connection_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    const uint32_t link_idx,
    ProgramOrDescriptor& worker_program_or_desc,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    CoreType core_type) {
    TT_FATAL(
        src_fabric_node_id != dst_fabric_node_id,
        "Expected different src and dst chip ids but got same, Src: {}, Dst: {}",
        src_fabric_node_id,
        dst_fabric_node_id);

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();

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

    uint32_t worker_teardown_semaphore_id;
    uint32_t worker_buffer_index_semaphore_id;
    uint32_t worker_flow_control_semaphore_id;

    if constexpr (std::is_same_v<ProgramOrDescriptor, tt::tt_metal::ProgramDescriptor>) {
        auto teardown_sem_id_opt = worker_program_or_desc.find_available_semaphore_id(worker_core, core_type);
        TT_FATAL(teardown_sem_id_opt.has_value(), "No available semaphore ID for teardown semaphore");
        worker_teardown_semaphore_id = teardown_sem_id_opt.value();
        worker_program_or_desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = worker_teardown_semaphore_id,
            .core_type = core_type,
            .core_ranges = CoreRangeSet(CoreRange(worker_core, worker_core)),
            .initial_value = 0});

        auto buffer_index_sem_id_opt = worker_program_or_desc.find_available_semaphore_id(worker_core, core_type);
        TT_FATAL(buffer_index_sem_id_opt.has_value(), "No available semaphore ID for buffer index semaphore");
        worker_buffer_index_semaphore_id = buffer_index_sem_id_opt.value();
        worker_program_or_desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = worker_buffer_index_semaphore_id,
            .core_type = core_type,
            .core_ranges = CoreRangeSet(CoreRange(worker_core, worker_core)),
            .initial_value = 0});
    } else {
        worker_teardown_semaphore_id = tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0, core_type);
        worker_buffer_index_semaphore_id =
            tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0, core_type);
    }

    if (core_type == CoreType::WORKER) {
        append_worker_to_fabric_edm_sender_rt_args(
            fabric_router_channel, worker_teardown_semaphore_id, worker_buffer_index_semaphore_id, worker_args);
    } else {
        // TODO: will be deprecated. currently for ethernet dispatch case
        //       ethernet core need to have same memory mapping as worker
        const auto router_direction = control_plane.routing_direction_to_eth_direction(forwarding_direction.value());

        // src_chip_id is still required to get the fabric_router_virtual_core from tt_cluster
        ChipId src_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);

        CoreCoord fabric_router_virtual_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                src_chip_id, fabric_router_channel);

        const auto& edm_config = fabric_context.get_builder_context().get_fabric_router_config();
        auto* channel_allocator = edm_config.channel_allocator.get();
        auto* const static_channel_allocator =
            dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
        TT_FATAL(
            static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
        // Sender channel 0 is always for local worker in the new design
        const auto sender_channel = 0;
        tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
            .edm_noc_x = fabric_router_virtual_core.x,
            .edm_noc_y = fabric_router_virtual_core.y,
            .edm_buffer_base_addr = static_channel_allocator->get_sender_channel_base_address(sender_channel),
            .num_buffers_per_channel = static_channel_allocator->get_sender_channel_number_of_slots(sender_channel),
            .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[sender_channel],
            .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[sender_channel],
            .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[sender_channel],
            .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
            .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[sender_channel],
            .edm_direction = router_direction};

        if constexpr (std::is_same_v<ProgramOrDescriptor, tt::tt_metal::ProgramDescriptor>) {
            auto flow_control_sem_id_opt = worker_program_or_desc.find_available_semaphore_id(worker_core, core_type);
            TT_FATAL(flow_control_sem_id_opt.has_value(), "No available semaphore ID for flow control semaphore");
            worker_flow_control_semaphore_id = flow_control_sem_id_opt.value();

            worker_program_or_desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = worker_flow_control_semaphore_id,
                .core_type = core_type,
                .core_ranges = CoreRangeSet(CoreRange(worker_core, worker_core)),
                .initial_value = 0});
        } else {
            worker_flow_control_semaphore_id =
                tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0, core_type);
        }

        append_worker_to_fabric_edm_sender_rt_args(
            edm_connection,
            worker_flow_control_semaphore_id,
            worker_teardown_semaphore_id,
            worker_buffer_index_semaphore_id,
            worker_args);
    }
}

// append runtime parameter for RoutingPlaneConnectionManager
template <typename ProgramOrDescriptor>
void append_routing_plane_connection_manager_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const std::vector<FabricNodeId>& dst_nodes,
    const std::vector<uint32_t>& connection_link_indices,
    ProgramOrDescriptor& worker_program_or_desc,
    tt::tt_metal::KernelHandle& kernel_id,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    FabricApiType api_type,
    CoreType core_type) {
    // 1) append tag (like direction) and fabric connection info for each route
    TT_FATAL(
        connection_link_indices.empty() ||
            (connection_link_indices.size() == 1 || connection_link_indices.size() == dst_nodes.size()),
        "connection_link_indices must be empty or have size 1 or the same size as dst_nodes");

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();

    // TODO: Remove this restriction once multiple ethernet cores per direction are supported
    // https://github.com/tenstorrent/tt-metal/issues/27221
    // Check for duplicate directions in dst_nodes to prevent using multiple ethernet cores in same
    // direction
    std::unordered_set<eth_chan_directions> used_directions;
    for (const auto& dst_node : dst_nodes) {
        auto dir_opt = tt::tt_fabric::get_eth_forwarding_direction(src_fabric_node_id, dst_node);
        if (dir_opt.has_value()) {
            TT_FATAL(
                !used_directions.contains(dir_opt.value()),
                "Multiple ethernet cores in the same direction ({}) are not currently supported. "
                "This restriction will be removed in a future update when proper multi-core routing is implemented.",
                dir_opt.value());
            used_directions.insert(dir_opt.value());
        }
    }

    for (size_t i = 0; i < dst_nodes.size(); i++) {
        const auto& dst_node = dst_nodes[i];
        auto dir_opt = tt::tt_fabric::get_eth_forwarding_direction(src_fabric_node_id, dst_node);
        TT_FATAL(
            dir_opt.has_value(),
            "Could not determine forwarding direction from src {} to first hop {}",
            src_fabric_node_id,
            dst_node);
        // Use direction as tag for ConnectionSlot
        worker_args.push_back(static_cast<uint32_t>(dir_opt.value()));

        uint32_t link_idx = 0;
        if (!connection_link_indices.empty()) {
            if (connection_link_indices.size() == 1) {
                link_idx = connection_link_indices[0];
            } else {
                link_idx = connection_link_indices[i];
            }
        } else {
            const auto links = get_forwarding_link_indices(src_fabric_node_id, dst_node);
            TT_FATAL(!links.empty(), "No forwarding links available from {} to {}", src_fabric_node_id, dst_node);
            link_idx = links[0];
        }

        append_fabric_connection_rt_args<ProgramOrDescriptor>(
            src_fabric_node_id, dst_node, link_idx, worker_program_or_desc, worker_core, worker_args, core_type);
    }

    auto add_kernel_defines = [&, kernel_ref = [&]() {
        if constexpr (std::is_same_v<std::decay_t<ProgramOrDescriptor>, tt::tt_metal::ProgramDescriptor>) {
            return &worker_program_or_desc.kernels[kernel_id];
        } else {
            return worker_program_or_desc.impl().get_kernel(kernel_id);
        }
    }()](std::initializer_list<std::pair<std::string, std::string>> defines) {
        if constexpr (std::is_same_v<std::decay_t<ProgramOrDescriptor>, tt::tt_metal::ProgramDescriptor>) {
            for (const auto& define : defines) {
                kernel_ref->defines.push_back(define);
            }
        } else {
            kernel_ref->add_defines(std::map<std::string, std::string>(defines.begin(), defines.end()));
        }
    };

    switch (api_type) {
        case FabricApiType::Linear: add_kernel_defines({{"API_TYPE_Linear", "1"}}); break;
        case FabricApiType::Mesh: add_kernel_defines({{"API_TYPE_Mesh", "1"}}); break;
        default: TT_FATAL(false, "Unsupported FabricApiType: {}", static_cast<int>(api_type));
    }
    // 2) Append additional info for 2D Mesh
    if (fabric_context.is_2D_routing_enabled()) {
        add_kernel_defines({{"FABRIC_2D", "1"}});
        auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
        worker_args.push_back(mesh_shape[1]);                     // ew_dim
        worker_args.push_back(src_fabric_node_id.chip_id);        // my_chip_id
        worker_args.push_back(src_fabric_node_id.mesh_id.get());  // my_mesh_id

        // For each target, append dst_dev_id and dst_mesh_id (per-header)
        for (const auto& dst_node : dst_nodes) {
            // dst_dev_id
            worker_args.push_back(static_cast<uint16_t>(dst_node.chip_id));
            // dst_mesh_id
            worker_args.push_back(static_cast<uint16_t>(*dst_node.mesh_id));
        }
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
    FabricConfig fabric_config,
    FabricReliabilityMode reliability_mode,
    std::optional<uint8_t> num_routing_planes,
    FabricTensixConfig fabric_tensix_config,
    FabricUDMMode fabric_udm_mode,
    FabricManagerMode fabric_manager,
    FabricRouterConfig router_config) {
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        fabric_config,
        reliability_mode,
        num_routing_planes,
        fabric_tensix_config,
        fabric_udm_mode,
        fabric_manager,
        router_config);
}

std::optional<eth_chan_directions> get_eth_forwarding_direction(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto routing_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    if (!routing_direction.has_value()) {
        return std::nullopt;
    }
    return control_plane.routing_direction_to_eth_direction(routing_direction.value());
}

bool is_1d_fabric_config(tt::tt_fabric::FabricConfig fabric_config) {
    return fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_RING ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE;
}

bool is_2d_fabric_config(tt::tt_fabric::FabricConfig fabric_config) {
    return fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y ||
           fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
}

// TODO: this should subtract out links used by runtime for dispatching to non-mmio capable devices, tracked by #27196
size_t get_num_available_routing_planes_in_direction(FabricNodeId fabric_node_id, RoutingDirection routing_direction) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_num_available_routing_planes_in_direction(fabric_node_id, routing_direction);
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
    ChipId first_chip_id = first_chip->id();
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

template void append_fabric_connection_rt_args<tt::tt_metal::Program>(
    const FabricNodeId&,
    const FabricNodeId&,
    const uint32_t,
    tt::tt_metal::Program&,
    const CoreCoord&,
    std::vector<uint32_t>&,
    CoreType);

template void append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
    const FabricNodeId&,
    const FabricNodeId&,
    const uint32_t,
    tt::tt_metal::ProgramDescriptor&,
    const CoreCoord&,
    std::vector<uint32_t>&,
    CoreType);

template void append_routing_plane_connection_manager_rt_args<tt::tt_metal::ProgramDescriptor>(
    const FabricNodeId&,
    const std::vector<FabricNodeId>&,
    const std::vector<uint32_t>&,
    tt::tt_metal::ProgramDescriptor&,
    tt::tt_metal::KernelHandle&,
    const CoreCoord&,
    std::vector<uint32_t>&,
    FabricApiType,
    CoreType);

template void append_routing_plane_connection_manager_rt_args<tt::tt_metal::Program>(
    const FabricNodeId&,
    const std::vector<FabricNodeId>&,
    const std::vector<uint32_t>&,
    tt::tt_metal::Program&,
    tt::tt_metal::KernelHandle&,
    const CoreCoord&,
    std::vector<uint32_t>&,
    FabricApiType,
    CoreType);

}  // namespace tt::tt_fabric
