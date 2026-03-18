// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_vc2_connection.hpp"

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <optional>
#include <vector>

#include "erisc_datamover_builder.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "fabric_host_utils.hpp"
#include "fabric_context.hpp"
#include "fabric_builder_context.hpp"

namespace tt::tt_fabric {

template <typename ProgramOrDescriptor>
void append_fabric_vc2_connection_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    const uint32_t link_idx,
    ProgramOrDescriptor& worker_program_or_desc,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args) {
    TT_FATAL(
        src_fabric_node_id != dst_fabric_node_id,
        "Expected different src and dst chip ids but got same, Src: {}, Dst: {}",
        src_fabric_node_id,
        dst_fabric_node_id);

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    TT_FATAL(
        builder_context.requires_vc2(),
        "VC2 connection requested but VC2 is not enabled. "
        "VC2 requires: 2D fabric topology (Mesh/Torus), multi-mesh configuration, Blackhole architecture, "
        "no UDM/mux extension mode, and TT_METAL_ENABLE_FABRIC_VC2 runtime option. "
        "Current topology may not meet these conditions.");
    const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();

    // Resolve forwarding direction (same logic as public API)
    std::optional<RoutingDirection> forwarding_direction;
    if (is_2d_fabric) {
        forwarding_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    } else {
        for (const auto& direction : FabricContext::routing_directions) {
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

    const auto forwarding_links = get_forwarding_link_indices_in_direction(
        control_plane, src_fabric_node_id, dst_fabric_node_id, forwarding_direction.value());
    TT_FATAL(
        std::find(forwarding_links.begin(), forwarding_links.end(), link_idx) != forwarding_links.end(),
        "Requested link index {} cannot be used for forwarding b/w src {} and dst {}. Valid forwarding links are {}",
        link_idx,
        src_fabric_node_id,
        dst_fabric_node_id,
        forwarding_links);

    const auto fabric_router_channel = candidate_eth_chans[link_idx];

    // Create semaphores (same pattern as public API, WORKER core type only)
    uint32_t worker_teardown_semaphore_id;
    uint32_t worker_buffer_index_semaphore_id;
    uint32_t worker_flow_control_semaphore_id;

    if constexpr (std::is_same_v<ProgramOrDescriptor, tt::tt_metal::ProgramDescriptor>) {
        auto teardown_sem_id_opt = worker_program_or_desc.find_available_semaphore_id(worker_core, CoreType::WORKER);
        TT_FATAL(teardown_sem_id_opt.has_value(), "No available semaphore ID for teardown semaphore");
        worker_teardown_semaphore_id = teardown_sem_id_opt.value();
        worker_program_or_desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = worker_teardown_semaphore_id,
            .core_type = CoreType::WORKER,
            .core_ranges = CoreRangeSet(CoreRange(worker_core, worker_core)),
            .initial_value = 0});

        auto buffer_index_sem_id_opt =
            worker_program_or_desc.find_available_semaphore_id(worker_core, CoreType::WORKER);
        TT_FATAL(buffer_index_sem_id_opt.has_value(), "No available semaphore ID for buffer index semaphore");
        worker_buffer_index_semaphore_id = buffer_index_sem_id_opt.value();
        worker_program_or_desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = worker_buffer_index_semaphore_id,
            .core_type = CoreType::WORKER,
            .core_ranges = CoreRangeSet(CoreRange(worker_core, worker_core)),
            .initial_value = 0});

        auto flow_control_sem_id_opt =
            worker_program_or_desc.find_available_semaphore_id(worker_core, CoreType::WORKER);
        TT_FATAL(flow_control_sem_id_opt.has_value(), "No available semaphore ID for flow control semaphore");
        worker_flow_control_semaphore_id = flow_control_sem_id_opt.value();
        worker_program_or_desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = worker_flow_control_semaphore_id,
            .core_type = CoreType::WORKER,
            .core_ranges = CoreRangeSet(CoreRange(worker_core, worker_core)),
            .initial_value = 0});
    } else {
        worker_teardown_semaphore_id =
            tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0, CoreType::WORKER);
        worker_buffer_index_semaphore_id =
            tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0, CoreType::WORKER);
        worker_flow_control_semaphore_id =
            tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0, CoreType::WORKER);
    }

    // Resolve VC2 sender channel addresses via explicit SenderWorkerAdapterSpec
    const auto router_direction = control_plane.routing_direction_to_eth_direction(forwarding_direction.value());

    ChipId src_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    CoreCoord fabric_router_virtual_core =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
            src_chip_id, fabric_router_channel);

    const auto& edm_config = fabric_context.get_builder_context().get_fabric_router_config();
    auto* channel_allocator = edm_config.channel_allocator.get();
    auto* const static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be FabricStaticSizedChannelsAllocator");

    // VC2 sender is at the last flat index: after all VC0 and VC1 senders
    const auto vc2_sender_channel =
        static_channel_allocator->get_num_sender_channels(0) + static_channel_allocator->get_num_sender_channels(1);

    tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
        .edm_noc_x = fabric_router_virtual_core.x,
        .edm_noc_y = fabric_router_virtual_core.y,
        .edm_buffer_base_addr = static_channel_allocator->get_sender_channel_base_address(vc2_sender_channel),
        .num_buffers_per_channel = static_channel_allocator->get_sender_channel_number_of_slots(vc2_sender_channel),
        .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[vc2_sender_channel],
        .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[vc2_sender_channel],
        .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[vc2_sender_channel],
        .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
        .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[vc2_sender_channel],
        .edm_direction = router_direction};

    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_args);
}

// Explicit template instantiations
template void append_fabric_vc2_connection_rt_args<tt::tt_metal::Program>(
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    tt::tt_metal::Program&,
    const CoreCoord&,
    std::vector<uint32_t>&);

template void append_fabric_vc2_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    tt::tt_metal::ProgramDescriptor&,
    const CoreCoord&,
    std::vector<uint32_t>&);

}  // namespace tt::tt_fabric
