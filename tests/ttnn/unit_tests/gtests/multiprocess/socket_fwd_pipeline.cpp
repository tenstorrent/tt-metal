
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/socket_forward/socket_forward.hpp"

#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/send_recv_op_utils.hpp"
#include <chrono>
#include "tt_metal/fabric/physical_system_descriptor.hpp"

namespace tt::tt_metal {

// using MeshDeviceDualGalaxyPipelineSendRecvFixture =
//     tt::tt_fabric::fabric_router_tests::MeshDeviceDualGalaxyPipelineFixture;

using MeshDeviceClosetBoxSendRecvFixture = tt::tt_fabric::fabric_router_tests::MeshDeviceClosetBoxFabricFixture;

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

// For testing/benchmaring purposes only - build a pipeline on a Dual BH Galaxy.
// Each pipeline stage corresponds to a single tray. Pipeline stages are connected
// via intermediate sockets. We try to minimize turns as much as possible, to maximize throughput.
std::vector<LogicalPipelineStageConfig> build_pipeline(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate>& asic_id_to_mesh_coord) {
    // Setup pipeline stages in physical space (Host rank, Tray ID, ASIC Location)
    std::vector<PhysicalPipelineStageConfig> physical_pipeline_stage_configs = {
        {.entry_node_tray_id = 1, .exit_node_tray_id = 1, .entry_node_asic_location = 5, .exit_node_asic_location = 6},
        {.entry_node_tray_id = 1, .exit_node_tray_id = 1, .entry_node_asic_location = 7, .exit_node_asic_location = 8},
        {.entry_node_tray_id = 2, .exit_node_tray_id = 2, .entry_node_asic_location = 8, .exit_node_asic_location = 7},
        {.entry_node_tray_id = 2, .exit_node_tray_id = 2, .entry_node_asic_location = 6, .exit_node_asic_location = 5},

        {.entry_node_tray_id = 1, .exit_node_tray_id = 1, .entry_node_asic_location = 5, .exit_node_asic_location = 6},
        {.entry_node_tray_id = 1, .exit_node_tray_id = 1, .entry_node_asic_location = 7, .exit_node_asic_location = 8},
        {.entry_node_tray_id = 2, .exit_node_tray_id = 2, .entry_node_asic_location = 8, .exit_node_asic_location = 7},
        {.entry_node_tray_id = 2, .exit_node_tray_id = 2, .entry_node_asic_location = 6, .exit_node_asic_location = 5},

        {.entry_node_tray_id = 1, .exit_node_tray_id = 1, .entry_node_asic_location = 5, .exit_node_asic_location = 6},
        {.entry_node_tray_id = 1, .exit_node_tray_id = 1, .entry_node_asic_location = 7, .exit_node_asic_location = 8},
        {.entry_node_tray_id = 2, .exit_node_tray_id = 2, .entry_node_asic_location = 8, .exit_node_asic_location = 7},
        {.entry_node_tray_id = 2, .exit_node_tray_id = 2, .entry_node_asic_location = 6, .exit_node_asic_location = 5},

        {.entry_node_tray_id = 1, .exit_node_tray_id = 1, .entry_node_asic_location = 5, .exit_node_asic_location = 6},
        {.entry_node_tray_id = 1, .exit_node_tray_id = 1, .entry_node_asic_location = 7, .exit_node_asic_location = 8},
        {.entry_node_tray_id = 2, .exit_node_tray_id = 2, .entry_node_asic_location = 8, .exit_node_asic_location = 7},
        {.entry_node_tray_id = 2, .exit_node_tray_id = 2, .entry_node_asic_location = 6, .exit_node_asic_location = 5},
    };

    std::unordered_map<uint32_t, uint32_t> stage_to_rank_mapping = {
        {0, 0},
        {1, 1},
        {2, 2},
        {3, 3},
        {4, 4},
        {5, 5},
        {6, 6},
        {7, 7},
        {8, 8},
        {9, 9},
        {10, 10},
        {11, 11},
        {12, 12},
        {13, 13},
        {14, 14},
        {15, 15},
    };

    // const auto num_procs = *(tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->size());
    std::vector<LogicalPipelineStageConfig> logical_pipeline_stage_configs;
    for (auto stage_index = 0; stage_index < physical_pipeline_stage_configs.size(); stage_index++) {
        auto stage_hostname = physical_system_descriptor.get_hostname_for_rank(stage_to_rank_mapping[stage_index]);
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

// Main benchmark:
// - Setup the sender, forwarding and receiver sockets across all pipeline stages
// - Host writes data to the first pipeline stage (14K)
// - This data is streamed through the pipeline for 1000000 iterations to benchmark steady-state throughput.
// - Sender, Forwarding and Receiver kernels are launched once and loop on device
// - The final throughput is computed based on the amount of time spent on the last pipeline stage.
// - This value corresponds to amount of time the receiver spent waiting for data + time spent on the forwarding kernel.
// - During steady state, this represents the average pipeline stage throughput.
TEST_F(MeshDeviceClosetBoxSendRecvFixture, SendRecvPipeline) {
    constexpr uint32_t XFER_SIZE = 14 * 1024;
    constexpr uint32_t NUM_ITERS = 1000000;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const distributed::multihost::Rank pipeline_start_rank{0};
    const distributed::multihost::Rank pipeline_end_rank{*distributed_context->size() - 1};

    const auto logical_coord = CoreCoord(0, 0);
    const uint32_t socket_fifo_size = XFER_SIZE * 16;

    auto physical_system_descriptor = create_physical_system_descriptor();
    auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device_);
    auto pipeline_stages = build_pipeline(physical_system_descriptor, asic_id_to_mesh_coord);

    const auto my_mesh_id = *distributed_context->rank();
    const auto upstream_mesh_id = my_mesh_id - 1;
    const auto downstream_mesh_id = my_mesh_id + 1;

    const distributed::SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);

    const auto tensor_spec = TensorSpec(
        ttnn::Shape({1, 1, 1, XFER_SIZE / sizeof(uint32_t)}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM)));

    const uint32_t num_elems = tensor_spec.logical_shape().volume();

    // Helper to create an intermediate socket pair for local forwarding
    auto create_intermed_socket_pair = [&](const distributed::MeshCoordinate& sender_coord,
                                           const distributed::MeshCoordinate& recv_coord) {
        auto connection = distributed::SocketConnection(
            distributed::MeshCoreCoord(sender_coord, logical_coord),
            distributed::MeshCoreCoord(recv_coord, logical_coord));
        auto config = distributed::SocketConfig({connection}, socket_mem_config);
        return distributed::MeshSocket::create_socket_pair(mesh_device_, mesh_device_, config);
    };

    // Helper to run warmup iteration with barrier synchronization
    auto barrier = [&]() {
        Synchronize(mesh_device_.get(), std::nullopt);
        distributed_context->barrier();
    };

    uint64_t start_time = 0;
    uint64_t end_time = 0;

    const bool is_pipeline_start = (*distributed_context->rank() == *pipeline_start_rank);
    const bool is_pipeline_end = (*distributed_context->rank() == *pipeline_end_rank);
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
        auto send_socket = distributed::MeshSocket(mesh_device_, send_socket_config);

        std::cout << "Start Coord: " << start_coord << "Sender: " << my_sender
                  << " Downstream Recv: " << downstream_recv << std::endl;
        // return;

        auto input_tensor = ttnn::distributed::distribute_tensor(
                                ttnn::experimental::view(
                                    ttnn::arange(0, num_elems, 1, tensor_spec.data_type()), tensor_spec.logical_shape())
                                    .to_layout(tensor_spec.layout()),
                                *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
                                std::nullopt)
                                .to_device(mesh_device_.get(), tensor_spec.memory_config());

        // Warmup iteration
        ttnn::experimental::send_async(input_tensor, intermed_send);
        ttnn::experimental::socket_forward(input_tensor, intermed_recv, send_socket, XFER_SIZE);
        barrier();

        // Timed iteration
        ttnn::experimental::send_async(input_tensor, intermed_send);
        ttnn::experimental::socket_forward(input_tensor, intermed_recv, send_socket, XFER_SIZE);
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
        auto recv_socket = distributed::MeshSocket(mesh_device_, recv_socket_config);

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
            send_socket = distributed::MeshSocket(mesh_device_, send_socket_config);

            std::tie(intermed_send, intermed_recv) = create_intermed_socket_pair(my_recv, my_sender);
            std::cout << "Recv Coord: " << my_recv << " Send Coord: " << my_sender
                      << " Downstream Recv: " << downstream_recv << std::endl;
        } else {
            // Pipeline end
            distributed::MeshCoordinate end_coord = pipeline_stages[*pipeline_end_rank].exit_node_coord;
            std::tie(intermed_send, intermed_recv) = create_intermed_socket_pair(my_recv, end_coord);
            std::cout << "Recv Coord: " << my_recv << " End Coord: " << end_coord << std::endl;
        }
        // return;
        Tensor output_tensor = tt::tt_metal::create_device_tensor(tensor_spec, mesh_device_.get());

        // Warmup iteration
        ttnn::experimental::socket_forward(output_tensor, recv_socket, intermed_send, XFER_SIZE);
        if (is_intermediate) {
            ttnn::experimental::socket_forward(output_tensor, intermed_recv, send_socket, XFER_SIZE);
        } else {
            ttnn::experimental::recv_async(output_tensor, intermed_recv);
        }
        barrier();

        // Timed iteration
        if (is_pipeline_end) {
            start_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::high_resolution_clock::now().time_since_epoch())
                             .count();
        }
        ttnn::experimental::socket_forward(output_tensor, recv_socket, intermed_send, XFER_SIZE);
        if (is_intermediate) {
            ttnn::experimental::socket_forward(output_tensor, intermed_recv, send_socket, XFER_SIZE);
        } else {
            ttnn::experimental::recv_async(output_tensor, intermed_recv);
        }
    }
    barrier();

    if (is_pipeline_end) {
        end_time = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now().time_since_epoch())
                       .count();
        std::cout << "Time taken to forward: " << NUM_ITERS << " Packets: " << end_time - start_time << " us"
                  << std::endl;
        std::cout << "Time per iteration: " << (end_time - start_time) / NUM_ITERS << " us" << std::endl;
    }
}

}  // namespace tt::tt_metal
