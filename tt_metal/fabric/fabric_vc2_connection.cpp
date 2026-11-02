// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_vc2_connection.hpp"

#include <tt_stl/fmt.hpp>
#include <array>
#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt_stl/assert.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId
#include <optional>
#include <vector>

#include "erisc_datamover_builder.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "fabric_host_utils.hpp"
#include "fabric_context.hpp"
#include "fabric_builder_context.hpp"

namespace tt::tt_fabric {

template <typename ProgramOrDescriptor>
void append_fabric_connection_rt_args_on_vc(
    const FabricContext& fabric_context,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    const uint32_t link_idx,
    const uint32_t vc,
    ProgramOrDescriptor& worker_program_or_desc,
    const tt::tt_metal::CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args) {
    TT_FATAL(
        src_fabric_node_id != dst_fabric_node_id,
        "Expected different src and dst chip ids but got same, Src: {}, Dst: {}",
        src_fabric_node_id,
        dst_fabric_node_id);

    const auto& control_plane = fabric_context.get_control_plane();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto& vc_config = builder_context.get_intermesh_vc_config();
    TT_FATAL(
        vc < builder_config::MAX_NUM_VCS && vc_config.vc_present(vc) &&
            builder_config::vc_routing_has_worker(vc_config.vc_routing[vc]),
        "A worker connection on VC{} needs a VC with a worker channel on this fabric: VC2 (TT_METAL_ENABLE_FABRIC_VC2, "
        "2D, Blackhole, no UDM/mux extension) or a 1D fabric's line VC (TT_METAL_FABRIC_LINE_VCS)",
        vc);
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
        worker_program_or_desc.semaphores.push_back(
            tt::tt_metal::SemaphoreDescriptor{
                .id = worker_teardown_semaphore_id,
                .core_type = CoreType::WORKER,
                .core_ranges = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(worker_core, worker_core)),
                .initial_value = 0});

        auto buffer_index_sem_id_opt =
            worker_program_or_desc.find_available_semaphore_id(worker_core, CoreType::WORKER);
        TT_FATAL(buffer_index_sem_id_opt.has_value(), "No available semaphore ID for buffer index semaphore");
        worker_buffer_index_semaphore_id = buffer_index_sem_id_opt.value();
        worker_program_or_desc.semaphores.push_back(
            tt::tt_metal::SemaphoreDescriptor{
                .id = worker_buffer_index_semaphore_id,
                .core_type = CoreType::WORKER,
                .core_ranges = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(worker_core, worker_core)),
                .initial_value = 0});

        auto flow_control_sem_id_opt =
            worker_program_or_desc.find_available_semaphore_id(worker_core, CoreType::WORKER);
        TT_FATAL(flow_control_sem_id_opt.has_value(), "No available semaphore ID for flow control semaphore");
        worker_flow_control_semaphore_id = flow_control_sem_id_opt.value();
        worker_program_or_desc.semaphores.push_back(
            tt::tt_metal::SemaphoreDescriptor{
                .id = worker_flow_control_semaphore_id,
                .core_type = CoreType::WORKER,
                .core_ranges = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(worker_core, worker_core)),
                .initial_value = 0});
    } else {
        worker_teardown_semaphore_id = tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0);
        worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0);
        worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(worker_program_or_desc, {worker_core}, 0);
    }

    // Resolve the worker channel addresses via explicit SenderWorkerAdapterSpec
    const auto router_direction = control_plane.routing_direction_to_eth_direction(forwarding_direction.value());

    ChipId src_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    tt::tt_metal::CoreCoord fabric_router_virtual_core =
        fabric_context.get_cluster().get_virtual_eth_core_from_channel(src_chip_id, fabric_router_channel);

    const auto& edm_config = fabric_context.get_builder_context().get_fabric_router_config();
    auto* channel_allocator = edm_config.channel_allocator.get();
    auto* const static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be FabricStaticSizedChannelsAllocator");

    // The VC's worker channel is its first sender channel; its flat index (edm_config arrays, indexed across all VCs)
    // comes after every lower VC's channels.
    const auto num_vc0_senders = static_channel_allocator->get_num_sender_channels(0);
    const auto num_vc1_senders = static_channel_allocator->get_num_sender_channels(1);
    const auto num_vc2_senders = static_channel_allocator->get_num_sender_channels(2);
    const std::array<size_t, builder_config::MAX_NUM_VCS> num_senders = {
        num_vc0_senders, num_vc1_senders, num_vc2_senders};
    size_t worker_channel_flat = 0;
    for (size_t lower = 0; lower < vc; ++lower) {
        worker_channel_flat += num_senders[lower];
    }
    const auto total_sender_channels = num_vc0_senders + num_vc1_senders + num_vc2_senders;

    log_trace(
        tt::LogFabric,
        "Worker channel flat index diagnostic: vc0_senders={}, vc1_senders={}, vc2_senders={}, "
        "vc2_flat_idx={}, total_senders={}, config.num_used_sender_channels={}",
        num_vc0_senders,
        num_vc1_senders,
        num_vc2_senders,
        worker_channel_flat,
        total_sender_channels,
        edm_config.num_used_sender_channels);
    TT_FATAL(
        worker_channel_flat < edm_config.num_used_sender_channels,
        "Worker channel flat index {} is out of bounds for edm_config arrays (num_used_sender_channels={}). "
        "Allocator counts: vc0={}, vc1={}, vc2={}",
        worker_channel_flat,
        edm_config.num_used_sender_channels,
        num_vc0_senders,
        num_vc1_senders,
        num_vc2_senders);
    TT_FATAL(num_senders[vc] > 0, "VC{} connection requested but allocator reports 0 VC{} sender channels", vc, vc);

    // Cross-check: verify the addresses at the worker channel flat index are non-zero
    TT_FATAL(
        edm_config.sender_channels_connection_semaphore_address[worker_channel_flat] != 0,
        "Worker channel connection_semaphore_address is 0 at flat index {}. "
        "This indicates the config did not allocate L1 for the worker channel.",
        worker_channel_flat);
    TT_FATAL(
        edm_config.sender_channels_worker_conn_info_base_address[worker_channel_flat] != 0,
        "Worker channel worker_conn_info_base_address is 0 at flat index {}.",
        worker_channel_flat);
    TT_FATAL(
        edm_config.sender_channels_local_flow_control_semaphore_address[worker_channel_flat] != 0,
        "Worker channel flow_control_semaphore_address is 0 at flat index {}.",
        worker_channel_flat);

    // Dump ALL sender channel addresses so we can see the full picture
    for (size_t i = 0; i < total_sender_channels; i++) {
        log_trace(
            tt::LogFabric,
            "  sender_ch[{}]: conn_sem=0x{:x}, conn_info=0x{:x}, flow_ctrl=0x{:x}, buf_idx=0x{:x}{}",
            i,
            edm_config.sender_channels_connection_semaphore_address[i],
            edm_config.sender_channels_worker_conn_info_base_address[i],
            edm_config.sender_channels_local_flow_control_semaphore_address[i],
            edm_config.sender_channels_buffer_index_semaphore_address[i],
            (i == worker_channel_flat) ? " <-- worker channel" : "");
    }

    // The worker channel is the VC's channel 0 (VC-relative), as the allocator indexes it.
    const size_t worker_vc = vc;
    static constexpr size_t worker_channel_in_vc = 0;

    tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
        .edm_noc_x = fabric_router_virtual_core.x,
        .edm_noc_y = fabric_router_virtual_core.y,
        .edm_buffer_base_addr =
            static_channel_allocator->get_sender_channel_base_address(worker_vc, worker_channel_in_vc),
        .num_buffers_per_channel =
            static_channel_allocator->get_sender_channel_number_of_slots(worker_vc, worker_channel_in_vc),
        .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[worker_channel_flat],
        .edm_connection_handshake_addr =
            edm_config.sender_channels_connection_semaphore_address[worker_channel_flat],
        .edm_worker_location_info_addr =
            edm_config.sender_channels_worker_conn_info_base_address[worker_channel_flat],
        .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
        .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[worker_channel_flat],
        .edm_direction = router_direction};

    log_trace(tt::LogFabric, "Worker connection. edm_buffer_base_addr: {}", edm_connection.edm_buffer_base_addr);
    log_trace(
        tt::LogFabric, "\tWorker connection. num_buffers_per_channel: {}", edm_connection.num_buffers_per_channel);
    log_trace(tt::LogFabric, "\tWorker connection. edm_l1_sem_addr: {}", edm_connection.edm_l1_sem_addr);
    log_trace(
        tt::LogFabric,
        "\tWorker connection. edm_connection_handshake_addr: {}",
        edm_connection.edm_connection_handshake_addr);
    log_trace(
        tt::LogFabric,
        "\tWorker connection. edm_worker_location_info_addr: {}",
        edm_connection.edm_worker_location_info_addr);
    log_trace(tt::LogFabric, "\tWorker connection. buffer_size_bytes: {}", edm_connection.buffer_size_bytes);
    log_trace(
        tt::LogFabric, "\tWorker connection. buffer_index_semaphore_id: {}", edm_connection.buffer_index_semaphore_id);
    log_trace(tt::LogFabric, "\tWorker connection. edm_direction: {}", static_cast<int>(edm_connection.edm_direction));
    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_args);
}

template <typename ProgramOrDescriptor>
void append_fabric_vc2_connection_rt_args(
    const FabricContext& fabric_context,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    const uint32_t link_idx,
    ProgramOrDescriptor& worker_program_or_desc,
    const tt::tt_metal::CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args) {
    append_fabric_connection_rt_args_on_vc(
        fabric_context, src_fabric_node_id, dst_fabric_node_id, link_idx, 2, worker_program_or_desc, worker_core,
        worker_args);
}

// Explicit template instantiations
template void append_fabric_connection_rt_args_on_vc<tt::tt_metal::Program>(
    const FabricContext&,
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    uint32_t,
    tt::tt_metal::Program&,
    const tt::tt_metal::CoreCoord&,
    std::vector<uint32_t>&);
template void append_fabric_connection_rt_args_on_vc<tt::tt_metal::ProgramDescriptor>(
    const FabricContext&,
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    uint32_t,
    tt::tt_metal::ProgramDescriptor&,
    const tt::tt_metal::CoreCoord&,
    std::vector<uint32_t>&);
template void append_fabric_vc2_connection_rt_args<tt::tt_metal::Program>(
    const FabricContext&,
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    tt::tt_metal::Program&,
    const tt::tt_metal::CoreCoord&,
    std::vector<uint32_t>&);
template void append_fabric_vc2_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
    const FabricContext&,
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    tt::tt_metal::ProgramDescriptor&,
    const tt::tt_metal::CoreCoord&,
    std::vector<uint32_t>&);
}  // namespace tt::tt_fabric
