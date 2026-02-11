
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_send_recv.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_forward.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_rate.hpp"

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <chrono>
#include <numeric>
#include <optional>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

// Fixture for single-galaxy pipeline tests (4 ranks; stage 4 loopback co-located with stage 0 on rank 0).
// Uses MeshDeviceExaboxFixture which auto-detects the system topology.
// Set TT_FABRIC_MESH_GRAPH_DESC_PATH to bh_galaxy_4x2_mesh_graph_descriptor.textproto when running.
class MeshDeviceSingleGalaxyPipelineFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture {};

// Fixture for Superpod 4 pipeline tests (16 ranks; 17 stages with loopback on rank 0).
// Requires 16 processes and mesh graph with 16 mesh IDs.
// Set TT_FABRIC_MESH_GRAPH_DESC_PATH to the superpod 4 mesh graph when running.
class MeshDeviceSuperpod4PipelineFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture {
public:
    void SetUp() override {
        if (not system_supported()) {
            GTEST_SKIP() << "Skipping: Superpod 4 pipeline requires 16 ranks and matching mesh graph.";
        }
        tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture::SetUp();
    }

    void TearDown() override {
        if (system_supported()) {
            tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture::TearDown();
        }
    }

    bool system_supported() {
        if (not tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture::system_supported()) {
            return false;
        }
        return *tt::tt_metal::MetalContext::instance().global_distributed_context().size() == 16u;
    }
};

// Pipeline config structs.

// User builds a pipeline in physical space (Host Ranks, Tray IDs, ASIC Locations)
struct PhysicalPipelineStageConfig {
    uint32_t entry_node_tray_id;
    uint32_t exit_node_tray_id;
    uint32_t entry_node_asic_location;
    uint32_t exit_node_asic_location;
};

// Logical Coords for start, intermed and end nodes in the pipeline are derived from the physical config.
struct LogicalPipelineStageConfig {
    std::size_t stage_index;
    distributed::MeshCoordinate entry_node_coord;
    distributed::MeshCoordinate exit_node_coord;
};

// Determine how the Multi Mesh Coordinate system is instantiated on the physical cluster.
std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> get_asic_id_to_mesh_coord_map(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> asic_id_to_mesh_coord_map;

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        tt_fabric::FabricNodeId fabric_node_id = mesh_device->get_fabric_node_id(coord);
        tt_metal::AsicID asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        asic_id_to_mesh_coord_map.emplace(asic_id, coord);
    }
    // Exchange this map across all hosts using distributed context
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    for (auto rank = 0; rank < *(distributed_context->size()); rank++) {
        if (rank == *(distributed_context->rank())) {
            // Loop over all entries of the map and send them to the other hosts
            std::size_t num_entries = asic_id_to_mesh_coord_map.size();
            distributed_context->broadcast(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&num_entries), sizeof(num_entries)),
                distributed::multihost::Rank{rank});
            for (auto& [asic_id, mesh_coord] : asic_id_to_mesh_coord_map) {
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(const_cast<tt_metal::AsicID*>(&asic_id)), sizeof(asic_id)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[0])), sizeof(mesh_coord[0])),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[1])), sizeof(mesh_coord[1])),
                    distributed::multihost::Rank{rank});
            }
        } else {
            // Receive the map from the other host
            std::size_t num_entries = 0;
            distributed_context->broadcast(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&num_entries), sizeof(num_entries)),
                distributed::multihost::Rank{rank});
            for (auto i = 0; i < num_entries; i++) {
                tt_metal::AsicID asic_id;
                distributed::MeshCoordinate mesh_coord = distributed::MeshCoordinate(0, 0);
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&asic_id), sizeof(asic_id)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[0])), sizeof(mesh_coord[0])),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&(mesh_coord[1])), sizeof(mesh_coord[1])),
                    distributed::multihost::Rank{rank});
                asic_id_to_mesh_coord_map.emplace(asic_id, mesh_coord);
            }
        }
    }
    return asic_id_to_mesh_coord_map;
}

// Pipeline type enum to toggle between different pipeline configurations.
enum class PipelineType {
    SINGLE_GALAXY,  // Single-galaxy pipeline (4 stages, 9 hops across 4 trays)
    DUAL_GALAXY,
    QUAD_GALAXY,
    SUPERPOD_4
};

// Get physical pipeline stage configs for the specified pipeline type.
// SINGLE_GALAXY: 5 stages. Stage 0 entry (T1D2) is the sender; stage 4 is the loop-back (T1D4 -> T1D2).
// Last stage exit and first stage entry are the same ASIC on the same tray (full loopback).
std::vector<PhysicalPipelineStageConfig> get_physical_pipeline_config(PipelineType type) {
    switch (type) {
        case PipelineType::SINGLE_GALAXY:
            return {
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 3,
                 .exit_node_tray_id = 3,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 2},
            };
        case PipelineType::SUPERPOD_4:
            return {
                // First Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Second Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Third Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Fourth Pod
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                // Wrap-around
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 1},
            };
        default: return {};
    }
}

// Overloaded build_pipeline that accepts an external physical pipeline config.
// Same pattern as ClosetBox: stage_index % num_procs for hostname. For loopback (last stage same tray
// as first), resolve last stage's ASICs using rank 0 host so they exist in asic_id_to_mesh_coord.
std::vector<LogicalPipelineStageConfig> build_pipeline(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate>& asic_id_to_mesh_coord,
    const std::vector<PhysicalPipelineStageConfig>& physical_pipeline_stage_configs) {
    const auto num_procs = *(tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->size());
    const std::size_t num_stages = physical_pipeline_stage_configs.size();
    const bool last_stage_loopback =
        (num_stages > 1u && physical_pipeline_stage_configs.back().exit_node_tray_id ==
                                physical_pipeline_stage_configs[0].entry_node_tray_id);
    std::vector<LogicalPipelineStageConfig> logical_pipeline_stage_configs;
    for (std::size_t stage_index = 0; stage_index < num_stages; stage_index++) {
        uint32_t rank_for_host = (last_stage_loopback && stage_index == num_stages - 1u)
                                     ? 0u
                                     : static_cast<uint32_t>(stage_index % num_procs);
        auto stage_hostname = physical_system_descriptor.get_hostname_for_rank(rank_for_host);
        const auto& phys = physical_pipeline_stage_configs[stage_index];
        auto entry_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(phys.entry_node_tray_id),
            tt::tt_metal::ASICLocation(phys.entry_node_asic_location));
        auto exit_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(phys.exit_node_tray_id),
            tt::tt_metal::ASICLocation(phys.exit_node_asic_location));
        logical_pipeline_stage_configs.emplace_back(LogicalPipelineStageConfig{
            .stage_index = stage_index,
            .entry_node_coord = asic_id_to_mesh_coord.at(entry_node_asic_id),
            .exit_node_coord = asic_id_to_mesh_coord.at(exit_node_asic_id)});
    }
    return logical_pipeline_stage_configs;
}

// Helper to get the device coords connecting the given pipeline stage and neighbor stage.
std::pair<distributed::MeshCoordinate, distributed::MeshCoordinate> get_connecting_coords(
    const std::vector<LogicalPipelineStageConfig>& pipeline_stages,
    uint32_t curr_stage_index,
    uint32_t neighbor_stage_index) {
    const auto& my_stage = pipeline_stages[curr_stage_index];
    const auto& neighbor_stage = pipeline_stages[neighbor_stage_index];

    if (curr_stage_index > neighbor_stage_index) {
        // Neighbor feeds into my stage
        return std::make_pair(my_stage.entry_node_coord, neighbor_stage.exit_node_coord);
    }
    // My stage feeds into neighbor
    return std::make_pair(my_stage.exit_node_coord, neighbor_stage.entry_node_coord);
}

// float convert_to_us(uint64_t cycles) {

// }

PhysicalSystemDescriptor create_physical_system_descriptor() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    constexpr bool run_discovery = true;
    const auto& driver = cluster.get_driver();

    return tt::tt_metal::PhysicalSystemDescriptor(driver, distributed_context, &hal, rtoptions, run_discovery);
}

// Single-galaxy pipeline test helper (multi-process). 5 stages, 4 ranks: stage 4 (loopback) on rank 0.
// Uses stage indices for coords and maps to ranks for global_bindings (downstream_rank 0 for loopback).
//
// Pipeline path: T1D2(send) -> T1D6 -> T3D6 -> T3D4 -> T4D4 -> T4D7 -> T2D7 -> T2D4 -> T1D4 -> T1D2(recv)
void run_single_galaxy_pipeline(
    std::shared_ptr<distributed::MeshDevice>& mesh_device, PipelineType pipeline_type, bool enable_correctness_check) {
    constexpr uint32_t XFER_SIZE = 14 * 1024;  // size of data being moved across pipeline stages for the workload
    constexpr uint32_t NUM_ITERATIONS = 500;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto my_rank = *distributed_context->rank();

    const auto logical_coord = CoreCoord(0, 0);
    const uint32_t socket_fifo_size = XFER_SIZE * 16;

    auto physical_system_descriptor = create_physical_system_descriptor();
    auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device);

    // Build pipeline from the given pipeline type (e.g. 4 stages, one per tray for SINGLE_GALAXY)
    auto physical_config = get_physical_pipeline_config(pipeline_type);
    auto pipeline_stages = build_pipeline(physical_system_descriptor, asic_id_to_mesh_coord, physical_config);

    const uint32_t num_stages = static_cast<uint32_t>(pipeline_stages.size());
    const uint32_t num_ranks = static_cast<uint32_t>(*(distributed_context->size()));
    // Stage indices 0..num_stages-1; ranks 0..num_ranks-1. Loopback stage (last) is on rank 0.
    const uint32_t downstream_stage = (my_rank + 1) % num_stages;
    const uint32_t upstream_stage = (my_rank + num_stages - 1) % num_stages;
    // Stage 4 (loopback) is on rank 0; stages 0..3 are on ranks 0..3.
    const uint32_t downstream_rank = (downstream_stage == num_stages - 1u) ? 0u : downstream_stage;
    const uint32_t upstream_rank = (upstream_stage == num_stages - 1u)
                                       ? (num_ranks - 1u)
                                       : upstream_stage;  // stage 4's upstream is stage 3 (rank 3)

    const auto& global_bindings =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_global_logical_bindings();
    const tt::tt_fabric::MeshId my_mesh_id = std::get<0>(global_bindings.at(distributed::multihost::Rank(my_rank)));
    const tt::tt_fabric::MeshId upstream_mesh_id =
        std::get<0>(global_bindings.at(distributed::multihost::Rank(upstream_rank)));
    const tt::tt_fabric::MeshId downstream_mesh_id =
        std::get<0>(global_bindings.at(distributed::multihost::Rank(downstream_rank)));

    const distributed::SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    // Metal-level buffer configuration
    const uint32_t num_elems = XFER_SIZE / sizeof(uint32_t);
    const DeviceAddr buffer_size = XFER_SIZE;
    const DeviceAddr page_size = XFER_SIZE;  // Single page buffer

    // Helper to create an intermediate socket pair for local forwarding
    auto create_intermed_socket_pair = [&](const distributed::MeshCoordinate& sender_coord,
                                           const distributed::MeshCoordinate& recv_coord) {
        auto connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(sender_coord, logical_coord),
            distributed::MeshCoreCoord(recv_coord, logical_coord));
        auto config = distributed::SocketConfig({connection}, socket_mem_config);
        return distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, config);
    };

    // Helper to run warmup iteration with barrier synchronization
    auto barrier = [&]() {
        Synchronize(mesh_device.get(), std::nullopt);
        distributed_context->barrier();
    };

    const bool is_pipeline_start = (my_rank == 0);

    // My stage coordinates (stage index == my_rank for 4 ranks; stage 4 loopback on rank 0)
    auto my_entry = pipeline_stages[my_rank].entry_node_coord;
    auto my_exit = pipeline_stages[my_rank].exit_node_coord;
    auto upstream_exit = pipeline_stages[upstream_stage].exit_node_coord;
    auto downstream_entry = pipeline_stages[downstream_stage].entry_node_coord;

    // Create Latency Measurement Buffer
    // Size: 8 bytes per iteration (uint64_t latency) + 32 bytes padding
    // First address is reused for credit/barrier synchronization
    constexpr auto latency_measurement_buffer_size = (8 * NUM_ITERATIONS) + 32;
    CoreRangeSet latency_core_range = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    auto shard_params = ShardSpecBuffer(latency_core_range, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    distributed::DeviceLocalBufferConfig latency_measurement_buffer_specs = {
        .page_size = latency_measurement_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    auto latency_measurement_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = latency_measurement_buffer_size},
        latency_measurement_buffer_specs,
        mesh_device.get());
    // Write 0 to latency measurement buffer (initializes credit/barrier to 0)
    std::vector<uint32_t> latency_init_data(latency_measurement_buffer_size / sizeof(uint32_t), 0);
    distributed::EnqueueWriteMeshBuffer(
        mesh_device->mesh_command_queue(), latency_measurement_buffer, latency_init_data, true);

    const uint32_t latency_measurement_address = latency_measurement_buffer->address();

    // Sender is stage 0 entry; last stage exit is the same ASIC (full loopback).
    const distributed::MeshCoordinate start_coord = pipeline_stages[0].entry_node_coord;
    TT_ASSERT(
        pipeline_stages.back().exit_node_coord == start_coord,
        "Loopback: last stage exit must equal first stage entry");

    if (is_pipeline_start) {
        // Send path: start_coord -> my_exit -> downstream (use stage indices)
        auto [my_sender, downstream_recv] = get_connecting_coords(pipeline_stages, my_rank, downstream_stage);
        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(start_coord, my_sender);

        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_sender, logical_coord),
            distributed::MeshCoreCoord(downstream_recv, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        // Recv path: from last stage into pipeline end entry, then local forward to start (ClosetBox pattern)
        auto [my_recv, upstream_send] = get_connecting_coords(pipeline_stages, num_stages - 1u, num_stages - 2u);
        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_send, logical_coord),
            distributed::MeshCoreCoord(my_recv, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        auto [intermed_send_2, intermed_recv_2] = create_intermed_socket_pair(my_recv, start_coord);

        // Create device buffer using metal-level API
        distributed::DeviceLocalBufferConfig buffer_config = {
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        auto input_mesh_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = buffer_size}, buffer_config, mesh_device.get());

        // Initialize buffer with data (arange equivalent: 0, 1, 2, ..., num_elems-1)
        std::vector<uint32_t> host_data(num_elems);
        std::iota(host_data.begin(), host_data.end(), 0u);

        // Write data to device buffer
        distributed::EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), input_mesh_buffer, host_data, true);

        // Extract buffer pointer for metal-level operations
        Buffer* input_buffer = input_mesh_buffer->get_reference_buffer();

        // Launch kernels:
        // - send_async on T1D2: sends data via intermed to T1D6, receives ack back via intermed_2 from T1D4
        // - socket_forward on T1D6: forwards from intermed to cross-mesh send socket (to T3D6)
        // - socket_forward on T1D4: forwards from cross-mesh recv socket (from T2D4) to intermed_2 (to T1D2)
        tt::tt_metal::send_async(
            mesh_device.get(),
            input_buffer,
            tt::DataFormat::UInt32,
            intermed_send,
            intermed_recv_2,
            latency_measurement_address,
            NUM_ITERATIONS,
            enable_correctness_check);
        tt::tt_metal::socket_forward(
            mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, latency_measurement_address, NUM_ITERATIONS);
        tt::tt_metal::socket_forward(
            mesh_device.get(), recv_socket, intermed_send_2, XFER_SIZE, latency_measurement_address, NUM_ITERATIONS);
    } else {
        // Non-start ranks: receive from upstream, forward locally, send to downstream

        // Cross-mesh recv from upstream: upstream_exit -> my_entry
        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_exit, logical_coord),
            distributed::MeshCoreCoord(my_entry, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        // Cross-mesh send to downstream: my_exit -> downstream_entry
        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_exit, logical_coord),
            distributed::MeshCoreCoord(downstream_entry, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        // Local intermed: my_entry -> my_exit
        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(my_entry, my_exit);

        // Create device buffer using metal-level API
        distributed::DeviceLocalBufferConfig buffer_config = {
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        auto output_mesh_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = buffer_size}, buffer_config, mesh_device.get());

        // Launch kernels: forward from upstream to downstream through local intermed
        tt::tt_metal::socket_forward(
            mesh_device.get(), recv_socket, intermed_send, XFER_SIZE, latency_measurement_address, NUM_ITERATIONS);
        tt::tt_metal::socket_forward(
            mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, latency_measurement_address, NUM_ITERATIONS);
    }
    barrier();
    if (is_pipeline_start) {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto start_device_id = mesh_device->get_device(start_coord)->id();
        auto start_core_coord = mesh_device->worker_core_from_logical_core(logical_coord);
        std::vector<uint64_t> latencies = std::vector<uint64_t>(NUM_ITERATIONS, 0);
        uint32_t base_addr = latency_measurement_address;
        cluster.read_core(
            latencies.data(),
            sizeof(uint64_t) * NUM_ITERATIONS,
            tt_cxy_pair(start_device_id, start_core_coord),
            base_addr);
        // Skip first iteration (often an outlier due to cold start)
        constexpr uint32_t LATENCY_ITERATIONS_FOR_AVG = NUM_ITERATIONS > 1 ? NUM_ITERATIONS - 1 : 1;
        double avg_latency_cycles = 0.0;
        for (uint32_t i = 1; i < NUM_ITERATIONS; i++) {
            avg_latency_cycles += static_cast<double>(latencies[i]);
        }
        avg_latency_cycles /= LATENCY_ITERATIONS_FOR_AVG;
        double freq_mhz = static_cast<double>(cluster.get_device_aiclk(start_device_id));
        double avg_latency_us = (avg_latency_cycles / (freq_mhz * 1e6)) * 1e6;
        double avd_latency_per_stage_us = avg_latency_us / static_cast<double>(num_stages);
        log_info(tt::LogTest, "Average latency in cycles: {:.2f}", avg_latency_cycles);
        log_info(tt::LogTest, "Average latency in microseconds: {:.2f}", avg_latency_us);
        log_info(tt::LogTest, "Average latency per stage in microseconds: {:.2f}", avd_latency_per_stage_us);
    }
}

TEST_F(MeshDeviceSingleGalaxyPipelineFixture, SendRecvPipelineSingleGalaxy) {
    run_single_galaxy_pipeline(mesh_device_, PipelineType::SINGLE_GALAXY, /*enable_correctness_check=*/false);
}

TEST_F(MeshDeviceSingleGalaxyPipelineFixture, SendRecvPipelineSingleGalaxyWithCorrectnessCheck) {
    run_single_galaxy_pipeline(mesh_device_, PipelineType::SINGLE_GALAXY, /*enable_correctness_check=*/true);
}

// SUPERPOD_4: 17 stages (4 pods × 4 stages + 1 wrap-around), 16 ranks; loopback stage on rank 0.
TEST_F(MeshDeviceSuperpod4PipelineFixture, SendRecvPipelineSuperpod4) {
    run_single_galaxy_pipeline(mesh_device_, PipelineType::SUPERPOD_4, /*enable_correctness_check=*/false);
}

TEST_F(MeshDeviceSuperpod4PipelineFixture, SendRecvPipelineSuperpod4WithCorrectnessCheck) {
    run_single_galaxy_pipeline(mesh_device_, PipelineType::SUPERPOD_4, /*enable_correctness_check=*/true);
// ─── Rate (throughput) pipeline test ─────────────────────────────────────────
// Linear pipeline (no loopback): data flows one-way through pipeline stages.
// Measures sustained pipeline throughput by pushing data for many iterations.

// Get physical pipeline config for rate testing (linear, no loopback).
// Uses 4 stages across 4 trays; last stage does NOT loop back to the first.
std::vector<PhysicalPipelineStageConfig> get_physical_pipeline_config_rate(PipelineType type) {
    switch (type) {
        case PipelineType::SINGLE_GALAXY:
            return {
                {.tray_id = 1, .entry_node_asic_location = 2, .exit_node_asic_location = 6},
                {.tray_id = 3, .entry_node_asic_location = 6, .exit_node_asic_location = 4},
                {.tray_id = 4, .entry_node_asic_location = 4, .exit_node_asic_location = 7},
                {.tray_id = 2, .entry_node_asic_location = 7, .exit_node_asic_location = 4},
            };
        default: return {};
    }
}

// Multi-host rate pipeline test helper.
// 4 stages, 4 ranks: sender on rank 0, fwd on ranks 1-2, receiver on rank 3.
// Pipeline path (one-way): T1D2 -> T1D6 -> T3D6 -> T3D4 -> T4D4 -> T4D7 -> T2D7 -> T2D4
// Timing is done on the host side using std::chrono, matching the original TTNN implementation.
void run_single_galaxy_rate_pipeline(
    std::shared_ptr<distributed::MeshDevice>& mesh_device,
    PipelineType pipeline_type,
    uint32_t num_iterations,
    bool enable_correctness_check) {
    constexpr uint32_t XFER_SIZE = 14 * 1024;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto my_rank = *distributed_context->rank();
    const auto num_ranks = static_cast<uint32_t>(*distributed_context->size());

    const auto logical_coord = CoreCoord(0, 0);
    const uint32_t socket_fifo_size = XFER_SIZE * 16;

    auto physical_system_descriptor = create_physical_system_descriptor();
    auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device);

    auto physical_config = get_physical_pipeline_config_rate(pipeline_type);
    auto pipeline_stages = build_pipeline(physical_system_descriptor, asic_id_to_mesh_coord, physical_config);

    // Linear pipeline: stage i is on rank i. No loopback.
    const uint32_t downstream_stage = my_rank + 1;
    const uint32_t upstream_stage = my_rank - 1;  // wraps for rank 0, but unused there
    const uint32_t downstream_rank = my_rank + 1;
    const uint32_t upstream_rank = my_rank - 1;

    const auto& global_bindings =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_global_logical_bindings();
    const tt::tt_fabric::MeshId my_mesh_id = std::get<0>(global_bindings.at(distributed::multihost::Rank(my_rank)));

    const distributed::SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    const uint32_t num_elems = XFER_SIZE / sizeof(uint32_t);
    const DeviceAddr buffer_size = XFER_SIZE;
    const DeviceAddr page_size = XFER_SIZE;

    auto create_intermed_socket_pair = [&](const distributed::MeshCoordinate& sender_coord,
                                           const distributed::MeshCoordinate& recv_coord) {
        auto connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(sender_coord, logical_coord),
            distributed::MeshCoreCoord(recv_coord, logical_coord));
        auto config = distributed::SocketConfig({connection}, socket_mem_config);
        return distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, config);
    };

    auto barrier = [&]() {
        Synchronize(mesh_device.get(), std::nullopt);
        distributed_context->barrier();
    };

    const bool is_pipeline_start = (my_rank == 0);
    const bool is_pipeline_end = (my_rank == num_ranks - 1);

    auto my_entry = pipeline_stages[my_rank].entry_node_coord;
    auto my_exit = pipeline_stages[my_rank].exit_node_coord;

    // Host-side timing variable — each rank records its own start/end time
    int64_t start_time = 0;
    int64_t end_time = 0;

    if (is_pipeline_start) {
        // Sender: start_coord -> my_exit (local), then my_exit -> downstream_entry (cross-mesh)
        auto start_coord = my_entry;
        auto [my_sender, downstream_recv] = get_connecting_coords(pipeline_stages, my_rank, downstream_stage);

        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(start_coord, my_sender);

        const tt::tt_fabric::MeshId downstream_mesh_id =
            std::get<0>(global_bindings.at(distributed::multihost::Rank(downstream_rank)));
        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_sender, logical_coord),
            distributed::MeshCoreCoord(downstream_recv, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        // Create device buffer
        distributed::DeviceLocalBufferConfig buffer_config = {
            .page_size = page_size,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        auto input_mesh_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = buffer_size}, buffer_config, mesh_device.get());
        std::vector<uint32_t> host_data(num_elems);
        std::iota(host_data.begin(), host_data.end(), 0u);
        distributed::EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), input_mesh_buffer, host_data, true);
        Buffer* input_buffer = input_mesh_buffer->get_reference_buffer();

        // Host-side timing: record start time just before launching kernels
        start_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now().time_since_epoch())
                         .count();

        // Launch rate-mode kernels:
        // - send_async_rate on start_coord: sends data to local intermed
        // - socket_forward_rate on my_exit: forwards from local intermed to cross-mesh
        tt::tt_metal::send_async_rate(
            mesh_device.get(), input_buffer, tt::DataFormat::UInt32, intermed_send, num_iterations);
        tt::tt_metal::socket_forward_rate(mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, num_iterations);
    } else if (is_pipeline_end) {
        // Receiver: upstream_exit -> my_entry (cross-mesh), then my_entry -> end_coord (local)
        auto upstream_exit = pipeline_stages[upstream_stage].exit_node_coord;
        auto end_coord = my_exit;

        const tt::tt_fabric::MeshId upstream_mesh_id =
            std::get<0>(global_bindings.at(distributed::multihost::Rank(upstream_rank)));
        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_exit, logical_coord),
            distributed::MeshCoreCoord(my_entry, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(my_entry, end_coord);

        // Host-side timing: record start time just before launching kernels
        start_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now().time_since_epoch())
                         .count();

        // Launch rate-mode kernels:
        // - socket_forward_rate on my_entry: forwards from cross-mesh recv to local intermed
        // - recv_async_rate on end_coord: drains data from local intermed
        tt::tt_metal::socket_forward_rate(mesh_device.get(), recv_socket, intermed_send, XFER_SIZE, num_iterations);
        tt::tt_metal::recv_async_rate(
            mesh_device.get(), intermed_recv, XFER_SIZE, num_iterations, enable_correctness_check);
    } else {
        // Intermediate: upstream_exit -> my_entry (cross-mesh), local forward, my_exit -> downstream_entry (cross-mesh)
        auto upstream_exit = pipeline_stages[upstream_stage].exit_node_coord;
        auto downstream_entry = pipeline_stages[downstream_stage].entry_node_coord;

        const tt::tt_fabric::MeshId upstream_mesh_id =
            std::get<0>(global_bindings.at(distributed::multihost::Rank(upstream_rank)));
        const tt::tt_fabric::MeshId downstream_mesh_id =
            std::get<0>(global_bindings.at(distributed::multihost::Rank(downstream_rank)));

        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_exit, logical_coord),
            distributed::MeshCoreCoord(my_entry, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_exit, logical_coord),
            distributed::MeshCoreCoord(downstream_entry, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(my_entry, my_exit);

        // Launch rate-mode kernels: forward from upstream to downstream through local intermed
        tt::tt_metal::socket_forward_rate(mesh_device.get(), recv_socket, intermed_send, XFER_SIZE, num_iterations);
        tt::tt_metal::socket_forward_rate(mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, num_iterations);
    }
    barrier();

    // Host-side timing: record end time after barrier (all ranks synchronized)
    end_time = std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
                   .count();

    // Report throughput from sender (rank 0)
    if (is_pipeline_start) {
        double elapsed_us = static_cast<double>(end_time - start_time);
        double total_bytes = static_cast<double>(num_iterations) * XFER_SIZE;
        double rate_gbps = (total_bytes * 8.0) / (elapsed_us * 1e3);

        log_info(tt::LogTest, "Rate pipeline: {} iterations, {} bytes/iter", num_iterations, XFER_SIZE);
        log_info(
            tt::LogTest,
            "Sender host-side elapsed: {:.2f} us, total bytes: {:.2f} MB, {:.4f} Gbps ({:.2f} Mbps)",
            elapsed_us,
            total_bytes / (1024.0 * 1024.0),
            rate_gbps,
            rate_gbps * 1e3);
    }

    // Report throughput from receiver (last rank)
    if (is_pipeline_end) {
        double elapsed_us = static_cast<double>(end_time - start_time);
        double total_bytes = static_cast<double>(num_iterations) * XFER_SIZE;
        double rate_gbps = (total_bytes * 8.0) / (elapsed_us * 1e3);

        log_info(tt::LogTest, "Rate pipeline: {} iterations, {} bytes/iter", num_iterations, XFER_SIZE);
        log_info(
            tt::LogTest,
            "Receiver host-side elapsed: {:.2f} us, total bytes: {:.2f} MB, {:.4f} Gbps ({:.2f} Mbps)",
            elapsed_us,
            total_bytes / (1024.0 * 1024.0),
            rate_gbps,
            rate_gbps * 1e3);
    }
}

TEST_F(MeshDeviceSingleGalaxyPipelineFixture, RatePipelineSingleGalaxy) {
    constexpr uint32_t NUM_ITERATIONS = 100000;
    run_single_galaxy_rate_pipeline(
        mesh_device_, PipelineType::SINGLE_GALAXY, NUM_ITERATIONS, /*enable_correctness_check=*/false);
}

TEST_F(MeshDeviceSingleGalaxyPipelineFixture, RatePipelineSingleGalaxyWithCorrectnessCheck) {
    constexpr uint32_t NUM_ITERATIONS = 100;
    run_single_galaxy_rate_pipeline(
        mesh_device_, PipelineType::SINGLE_GALAXY, NUM_ITERATIONS, /*enable_correctness_check=*/true);
}

}  // namespace tt::tt_metal
