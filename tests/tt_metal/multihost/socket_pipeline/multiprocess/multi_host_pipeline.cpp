
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_send_recv.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_forward.hpp"

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <optional>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

// Fixture for single-galaxy pipeline tests (4 ranks, one per tray).
// Uses MeshDeviceExaboxFixture which auto-detects the system topology.
// Set TT_FABRIC_MESH_GRAPH_DESC_PATH to bh_galaxy_4x2_mesh_graph_descriptor.textproto when running.
class MeshDevicePipelineFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture {};

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
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> asic_id_to_mesh_coord_map;

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        tt_fabric::FabricNodeId fabric_node_id = mesh_device->get_fabric_node_id(coord);
        tt_metal::AsicID asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        asic_id_to_mesh_coord_map.emplace(asic_id, coord);
    }
    // Exchange this map across all hosts using distributed context
    auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
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
    SINGLE_GALAXY_FWD,
    SINGLE_GALAXY_LOOPBACK,  // Single-galaxy pipeline (4 stages, 9 hops across 4 trays)
    DUAL_GALAXY,
    QUAD_GALAXY,
    SUPERPOD_4
};

// Get physical pipeline stage configs for the specified pipeline type.
std::vector<PhysicalPipelineStageConfig> get_physical_pipeline_config(PipelineType type) {
    switch (type) {
        case PipelineType::SINGLE_GALAXY_FWD:
            return {
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = 2,
                 .exit_node_tray_id = 2,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
            };
        case PipelineType::SINGLE_GALAXY_LOOPBACK:
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
                // Final pipeline stage routes data to the sender device
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 2},
            };

        default: return {};
    }
}

// Overloaded build_pipeline that accepts an external physical pipeline config.
std::vector<LogicalPipelineStageConfig> build_pipeline(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate>& asic_id_to_mesh_coord,
    const std::vector<PhysicalPipelineStageConfig>& physical_pipeline_stage_configs) {
    const auto num_procs = *(tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->size());
    std::vector<LogicalPipelineStageConfig> logical_pipeline_stage_configs;
    for (std::size_t stage_index = 0; stage_index < physical_pipeline_stage_configs.size(); stage_index++) {
        auto stage_hostname = physical_system_descriptor.get_hostname_for_rank(stage_index % num_procs);
        auto entry_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(physical_pipeline_stage_configs[stage_index].entry_node_tray_id),
            tt::tt_metal::ASICLocation(physical_pipeline_stage_configs[stage_index].entry_node_asic_location));
        auto exit_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(physical_pipeline_stage_configs[stage_index].exit_node_tray_id),
            tt::tt_metal::ASICLocation(physical_pipeline_stage_configs[stage_index].exit_node_asic_location));
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
    } else {
        // My stage feeds into neighbor
        return std::make_pair(my_stage.exit_node_coord, neighbor_stage.entry_node_coord);
    }
}

PhysicalSystemDescriptor create_physical_system_descriptor() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    constexpr bool run_discovery = true;
    const auto& driver = cluster.get_driver();

    return tt::tt_metal::PhysicalSystemDescriptor(driver, distributed_context, &hal, rtoptions, run_discovery);
}

// Single-galaxy pipeline test helper (multi-process, 4 ranks, one per tray).
// Uses the SINGLE_GALAXY pipeline config and follows the same setup logic
// as the ClosetBox SendRecvPipeline test, but with 4 stages across 4 trays
// and a separate sender device (T1D2).
//
// Pipeline path (9 hops):
//   T1D2(send) -> T1D6(fwd) -> T3D6(fwd) -> T3D4(fwd) -> T4D4(fwd) ->
//   T4D7(fwd) -> T2D7(fwd) -> T2D4(fwd) -> T1D4(fwd) -> T1D2(recv)
// This is to benchmark pipeline functionality for Deepseek Decode
void run_loopback_pipeline(
    std::shared_ptr<distributed::MeshDevice>& mesh_device, PipelineType pipeline_type, bool enable_correctness_check) {
    constexpr uint32_t XFER_SIZE = 14 * 1024;  // size of data being moved across pipeline stages for the workload
    constexpr uint32_t NUM_ITERATIONS = 100;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto my_rank = *distributed_context->rank();

    const auto logical_coord = CoreCoord(0, 0);
    const uint32_t socket_fifo_size = XFER_SIZE * 16;

    auto physical_system_descriptor = create_physical_system_descriptor();
    auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device);

    // Build pipeline from the given pipeline type (e.g. 4 stages, one per tray for SINGLE_GALAXY_LOOPBACK)
    auto physical_config = get_physical_pipeline_config(pipeline_type);
    auto pipeline_stages = build_pipeline(physical_system_descriptor, asic_id_to_mesh_coord, physical_config);

    // Loopback config - first and last stages are directly connected to each other
    const uint32_t upstream_rank = (my_rank == 0) ? (*distributed_context->size() - 1) : (my_rank - 1);
    const uint32_t downstream_rank = (my_rank == (*distributed_context->size() - 1)) ? 0 : (my_rank + 1);

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

    // My stage coordinates
    // Special handling for first rank entry node. Last stage feeds the first rank.
    auto my_entry = (my_rank == 0) ? pipeline_stages[pipeline_stages.size() - 1].entry_node_coord
                                   : pipeline_stages[my_rank].entry_node_coord;
    auto my_exit = pipeline_stages[my_rank].exit_node_coord;

    // Neighbor stage coordinates for cross-mesh sockets (with wrap-around)
    auto upstream_exit = pipeline_stages[upstream_rank].exit_node_coord;
    // Special handling for last rank exit node. Last stage feeds the first rank.
    auto downstream_entry =
        pipeline_stages[downstream_rank ? downstream_rank : (pipeline_stages.size() - 1)].entry_node_coord;

    // Create Latency Measurement Buffer
    // Size: 8 bytes per iteration (uint64_t latency) + 32 bytes padding
    // First address is reused for credit/barrier synchronization
    constexpr auto latency_measurement_buffer_size = 8 * NUM_ITERATIONS + 32;
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

    // Sender (injector) coord for pipeline start: from config when present, else stage 0 entry.
    distributed::MeshCoordinate start_coord = distributed::MeshCoordinate(0, 0);
    if (is_pipeline_start) {
        start_coord = pipeline_stages[my_rank].entry_node_coord;
        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(start_coord, my_exit);

        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_exit, logical_coord),
            distributed::MeshCoreCoord(downstream_entry, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection}, socket_mem_config, my_mesh_id, downstream_mesh_id, distributed_context);
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_exit, logical_coord),
            distributed::MeshCoreCoord(my_entry, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection}, socket_mem_config, upstream_mesh_id, my_mesh_id, distributed_context);
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);

        auto [intermed_send_2, intermed_recv_2] = create_intermed_socket_pair(my_entry, start_coord);

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
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Average latency in cycles: " << avg_latency_cycles << std::endl;
        std::cout << "Average latency in microseconds: " << avg_latency_us << std::endl;
    }
}

void run_unidirectional_pipeline(
    std::shared_ptr<distributed::MeshDevice>& mesh_device, PipelineType pipeline_type, bool enable_correctness_check) {
    constexpr uint32_t XFER_SIZE = 14 * 1024;
    constexpr uint32_t NUM_ITERS = 1000000;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    const auto logical_coord = CoreCoord(0, 0);
    const uint32_t socket_fifo_size = XFER_SIZE * 16;

    auto physical_system_descriptor = create_physical_system_descriptor();
    auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device);

    auto physical_config = get_physical_pipeline_config(pipeline_type);
    auto pipeline_stages = build_pipeline(physical_system_descriptor, asic_id_to_mesh_coord, physical_config);

    const auto my_rank = *distributed_context->rank();
    const auto upstream_rank = my_rank - 1;
    const auto downstream_rank = my_rank + 1;

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

    uint64_t start_time = 0;
    uint64_t end_time = 0;

    const bool is_pipeline_start = (*my_mesh_id == 0);
    const bool is_pipeline_end = (*my_mesh_id == (*distributed_context->size() - 1));
    const bool is_intermediate = !is_pipeline_start && !is_pipeline_end;

    if (is_pipeline_start) {
        // Pipeline start: Copy data from start coord to exit node using an intermediate socket
        auto [my_sender, downstream_recv] = get_connecting_coords(pipeline_stages, my_mesh_id, downstream_mesh_id);
        distributed::MeshCoordinate start_coord = pipeline_stages[*pipeline_start_rank].entry_node_coord;

        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(start_coord, my_sender);

        auto fwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(my_sender, logical_coord),
            distributed::MeshCoreCoord(downstream_recv, logical_coord));
        auto send_socket_config = distributed::SocketConfig(
            {fwd_connection},
            socket_mem_config,
            distributed_context->rank(),
            distributed::multihost::Rank(downstream_mesh_id));
        auto send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

        std::cout << "Start Coord: " << start_coord << "Sender: " << my_sender
                  << " Downstream Recv: " << downstream_recv << std::endl;

        distributed::DeviceLocalBufferConfig buffer_config = {
            .page_size - page_size,
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

        // Warmup iteration
        tt::tt_metal::send_async(
            mesh_device.get(),
            input_buffer,
            tt::DataFormat::UInt32,
            intermed_send,
            std::nullopt,  // intermed_recv
            std::nullopt,  // latency measurement address
            NUM_ITERS,
            enable_correctness_check);
        tt::tt_metal::socket_forward(
            nesh_device.get(),
            intermed_recv,
            send_socket,
            XFER_SIZE,
            std::nullopt,  // latency measurement address
            NUM_ITERS);
        barrier();

        // Timed iteration
        tt::tt_metal::send_async(
            mesh_device.get(),
            input_buffer,
            tt::DataFormat::UInt32,
            intermed_send,
            std::nullopt,  // intermed_recv
            std::nullopt,  // latency measurement address
            NUM_ITERS,
            enable_correctness_check);
        tt::tt_metal::socket_forward(
            mesh_device.get(),
            intermed_recv,
            send_socket,
            XFER_SIZE,
            std::nullopt,  // latency measurement address
            NUM_ITERS);
    } else {
        auto [my_recv, upstream_send] = get_connecting_coords(pipeline_stages, my_mesh_id, upstream_mesh_id);
        auto bwd_connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(upstream_send, logical_coord),
            distributed::MeshCoreCoord(my_recv, logical_coord));
        auto recv_socket_config = distributed::SocketConfig(
            {bwd_connection},
            socket_mem_config,
            distributed::multihost::Rank(upstream_mesh_id),
            distributed_context->rank());
        auto recv_socket = distributed::MeshSocket(mesh_device, recv_socket_config);
        distributed::MeshSocket send_socket;
        distributed::MeshSocket intermed_send;
        distributed::MeshSocket intermed_recv;

        if (is_intermediate) {
            auto [my_sender, downstream_recv] = get_connecting_coords(pipeline_stages, my_mesh_id, downstream_mesh_id);

            auto fwd_connection = distributed::SocketConnection(
                distributed::MeshCoreCoord(my_sender, logical_coord),
                distributed::MeshCoreCoord(downstream_recv, logical_coord));
            auto send_socket_config = distributed::SocketConfig(
                {fwd_connection},
                socket_mem_config,
                distributed_context->rank(),
                distributed::multihost::Rank(downstream_mesh_id));
            send_socket = distributed::MeshSocket(mesh_device, send_socket_config);

            std::tie(intermed_send, intermed_recv) = create_intermed_socket_pair(my_recv, my_sender);
            std::cout << "Recv Coord: " << my_recv << " Send Coord: " << my_sender
                      << " Downstream Recv: " << downstream_recv << std::endl;
        } else {
            // Pipeline end
            distributed::MeshCoordinate end_coord = pipeline_stages[*pipeline_end_rank].exit_node_coord;
            std::tie(intermed_send, intermed_recv) = create_intermed_socket_pair(my_recv, end_coord);
            std::cout << "Recv Coord: " << my_recv << " End Coord: " << end_coord << std::endl;
        }

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

        tt::tt_metal::socket_forward(
            mesh_device.get(),
            recv_socket,
            intermed_send,
            XFER_SIZE,
            std::nullopt,
            /* latency measurement address */,
            NUM_ITERS);
        if (is_intermediate) {
            tt::tt_metal::socket_forward(
                mesh_device.get(), intermed_recv, send_socket, XFER_SIZE, std::nullopt, NUM_ITERS);
        } else {
            // tt::tt_metal::recv_async()
        }
        barrier();

        // if (is_pipeline_end) {
        //     start_time = std::chrono::duration_cast<std::chrono::microseconds>(
        //                      std::chrono::high_resolution_clock::now().time_since_epoch())
        //                      .count();
        // }
        // ttnn::experimental::socket_forward(output_tensor, recv_socket, intermed_send, XFER_SIZE);
        // if (is_intermediate) {
        //     ttnn::experimental::socket_forward(output_tensor, intermed_recv, send_socket, XFER_SIZE);
        // } else {
        //     ttnn::experimental::recv_async(output_tensor, intermed_recv);
        // }
    }
    // barrier();

    // if (is_pipeline_end) {
    //     end_time = std::chrono::duration_cast<std::chrono::microseconds>(
    //                    std::chrono::high_resolution_clock::now().time_since_epoch())
    //                    .count();
    //     std::cout << "Time taken to forward: " << NUM_ITERS << " Packets: " << end_time - start_time << " us"
    //               << std::endl;
    //     std::cout << "Time per iteration: " << (end_time - start_time) / NUM_ITERS << " us" << std::endl;
    // }
}

TEST_F(MeshDevicePipelineFixture, SendRecvPipelineSingleGalaxy) {
    run_loopback_pipeline(mesh_device_, PipelineType::SINGLE_GALAXY_LOOPBACK, /*enable_correctness_check=*/false);
}

TEST_F(MeshDevicePipelineFixture, SendRecvPipelineSingleGalaxyWithCorrectnessCheck) {
    run_loopback_pipeline(mesh_device_, PipelineType::SINGLE_GALAXY_LOOPBACK, /*enable_correctness_check=*/true);
}

}  // namespace tt::tt_metal
