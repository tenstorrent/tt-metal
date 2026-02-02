// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"

namespace tt::tt_metal {

class MeshDevice4StagePipelineSendRecvFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture {};

struct PhysicalPipelineStageConfig {
    uint32_t tray_id;
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
std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> generate_asic_id_to_mesh_coord_map(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate> asic_id_to_mesh_coord_map;

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        tt_fabric::FabricNodeId fabric_node_id = mesh_device->get_fabric_node_id(coord);
        tt_metal::AsicID asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        asic_id_to_mesh_coord_map.emplace(asic_id, coord);
    }
    // Exchange this map across all hosts using distributed context
    // Follow MPI broadcast semantics for this (sender + receivers all call the broadcast API)
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

std::vector<LogicalPipelineStageConfig> build_2x4_pipeline(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate>& asic_id_to_mesh_coord) {
    // Physical locations used here correspond to the BH Galaxy (Rev A and Rev B).
    // This setup will not work on the WH Galaxy or BH Galaxy Rev C.
    // Setup pipeline stages in physical space (Host rank, Tray ID, ASIC Location)
    std::vector<PhysicalPipelineStageConfig> physical_pipeline_stage_configs = {
        {.tray_id = 1, .entry_node_asic_location = 4, .exit_node_asic_location = 6},
        {.tray_id = 3, .entry_node_asic_location = 6, .exit_node_asic_location = 4},
        {.tray_id = 4, .entry_node_asic_location = 4, .exit_node_asic_location = 7},
        {.tray_id = 2, .entry_node_asic_location = 7, .exit_node_asic_location = 1}};

    std::vector<LogicalPipelineStageConfig> logical_pipeline_stage_configs;
    for (auto stage_index = 0; stage_index < physical_pipeline_stage_configs.size(); stage_index++) {
        auto stage_hostname = physical_system_descriptor.get_hostname_for_rank(stage_index);
        auto entry_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(physical_pipeline_stage_configs[stage_index].tray_id),
            tt::tt_metal::ASICLocation(physical_pipeline_stage_configs[stage_index].entry_node_asic_location));
        auto exit_node_asic_id = physical_system_descriptor.get_asic_id(
            stage_hostname,
            tt::tt_metal::TrayID(physical_pipeline_stage_configs[stage_index].tray_id),
            tt::tt_metal::ASICLocation(physical_pipeline_stage_configs[stage_index].exit_node_asic_location));
        logical_pipeline_stage_configs.emplace_back(LogicalPipelineStageConfig{
            .stage_index = stage_index,
            .entry_node_coord = asic_id_to_mesh_coord.at(entry_node_asic_id),
            .exit_node_coord = asic_id_to_mesh_coord.at(exit_node_asic_id)});
    }
    return logical_pipeline_stage_configs;
}

std::pair<distributed::MeshCoordinate, distributed::MeshCoordinate> get_connecting_coords(
    const std::vector<LogicalPipelineStageConfig>& pipeline_stages,
    uint32_t curr_stage_index,
    uint32_t neighbor_stage_index) {
    const auto& my_stage = pipeline_stages[curr_stage_index];
    const auto& neighbor_stage = pipeline_stages[neighbor_stage_index];

    if (curr_stage_index > neighbor_stage_index) {
        // Neighbor feeds into my stage
        return std::make_pair(my_stage.entry_node_coord, neighbor_stage.exit_node_coord);
    }  // My stage feeds into neighbor
    return std::make_pair(my_stage.exit_node_coord, neighbor_stage.entry_node_coord);
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

// This test does the following:
// - Split a single galaxy into 4 meshes (each mesh is on a single tray). Assign 1 process to each mesh.
// - Setup sockets (sender, intermediate send/recv and final receiver) to build a 4 stage pipelne
// - Write data to the to the first pipeline stage (14K) from the first process
// - This data is streamed through the pipeline for 10 iterations
// - Final pipeline stage validates data correctness
TEST_F(MeshDevice4StagePipelineSendRecvFixture, TestSendRecvPipeline) {
    auto arch = tt::tt_metal::MetalContext::instance().get_cluster().arch();
    if (arch != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "This test can only run on Blackhole systems";
    }
    constexpr uint32_t XFER_SIZE = 14 * 1024;
    constexpr uint32_t NUM_ITERATIONS = 10;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const distributed::multihost::Rank pipeline_start_rank{0};
    const distributed::multihost::Rank pipeline_end_rank{*distributed_context->size() - 1};

    const auto send_core_coord = CoreCoord(0, 0);
    const auto recv_core_coord = CoreCoord(0, 0);

    const uint32_t socket_fifo_size = XFER_SIZE * 8;

    auto physical_system_descriptor = create_physical_system_descriptor();
    auto asic_id_to_mesh_coord = generate_asic_id_to_mesh_coord_map(mesh_device_);
    auto pipeline_stages = build_2x4_pipeline(physical_system_descriptor, asic_id_to_mesh_coord);

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
        distributed::SocketConnection connection(
            distributed::MeshCoreCoord(sender_coord, send_core_coord),
            distributed::MeshCoreCoord(recv_coord, recv_core_coord));
        distributed::SocketConfig config({connection}, socket_mem_config);
        return distributed::MeshSocket::create_socket_pair(mesh_device_, mesh_device_, config);
    };

    // Helper to run warmup iteration with barrier synchronization
    auto barrier = [&]() {
        Synchronize(mesh_device_.get(), std::nullopt);
        distributed_context->barrier();
    };

    const bool is_pipeline_start = (*distributed_context->rank() == *pipeline_start_rank);
    const bool is_pipeline_end = (*distributed_context->rank() == *pipeline_end_rank);
    const bool is_intermediate = !is_pipeline_start && !is_pipeline_end;

    Tensor intermediate_tensor = tt::tt_metal::create_device_tensor(tensor_spec, mesh_device_.get());

    if (is_pipeline_start) {
        // Pipeline start: Copy data from start coord to exit node using an intermediate socket
        auto [my_sender, downstream_recv] = get_connecting_coords(pipeline_stages, my_mesh_id, downstream_mesh_id);
        distributed::MeshCoordinate start_coord = pipeline_stages[*pipeline_start_rank].entry_node_coord;

        auto [intermed_send, intermed_recv] = create_intermed_socket_pair(start_coord, my_sender);

        distributed::SocketConnection fwd_connection(
            distributed::MeshCoreCoord(my_sender, send_core_coord),
            distributed::MeshCoreCoord(downstream_recv, recv_core_coord));
        distributed::SocketConfig send_socket_config(
            {fwd_connection},
            socket_mem_config,
            distributed_context->rank(),
            distributed::multihost::Rank(downstream_mesh_id));
        auto send_socket = distributed::MeshSocket(mesh_device_, send_socket_config);

        auto run_sender_step = [&](uint32_t i) {
            auto input_tensor =
                ttnn::distributed::distribute_tensor(
                    ttnn::experimental::view(
                        ttnn::arange(i, num_elems + i, 1, tensor_spec.data_type()), tensor_spec.logical_shape())
                        .to_layout(tensor_spec.layout()),
                    *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
                    std::nullopt)
                    .to_device(mesh_device_.get(), tensor_spec.memory_config());

            ttnn::experimental::send_async(input_tensor, intermed_send);
            ttnn::experimental::recv_async(intermediate_tensor, intermed_recv);
            ttnn::experimental::send_async(intermediate_tensor, send_socket);
        };
        for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
            run_sender_step(i);
        }
    } else {
        auto [my_recv, upstream_send] = get_connecting_coords(pipeline_stages, my_mesh_id, upstream_mesh_id);

        distributed::SocketConnection bwd_connection(
            distributed::MeshCoreCoord(upstream_send, send_core_coord),
            distributed::MeshCoreCoord(my_recv, recv_core_coord));
        distributed::SocketConfig recv_socket_config(
            {bwd_connection},
            socket_mem_config,
            distributed::multihost::Rank(upstream_mesh_id),
            distributed_context->rank());
        auto recv_socket = distributed::MeshSocket(mesh_device_, recv_socket_config);

        if (is_intermediate) {
            auto [my_sender, downstream_recv] = get_connecting_coords(pipeline_stages, my_mesh_id, downstream_mesh_id);
            distributed::SocketConnection fwd_connection(
                distributed::MeshCoreCoord(my_sender, send_core_coord),
                distributed::MeshCoreCoord(downstream_recv, recv_core_coord));
            distributed::SocketConfig send_socket_config(
                {fwd_connection},
                socket_mem_config,
                distributed_context->rank(),
                distributed::multihost::Rank(downstream_mesh_id));
            auto send_socket = distributed::MeshSocket(mesh_device_, send_socket_config);

            auto [intermed_send, intermed_recv] = create_intermed_socket_pair(my_recv, my_sender);

            auto run_intermed_step = [&]() {
                ttnn::experimental::recv_async(intermediate_tensor, recv_socket);
                ttnn::experimental::send_async(intermediate_tensor, intermed_send);
                ttnn::experimental::recv_async(intermediate_tensor, intermed_recv);
                ttnn::experimental::send_async(intermediate_tensor, send_socket);
            };

            for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
                run_intermed_step();
            }
        } else {
            // Pipeline end
            distributed::MeshCoordinate end_coord = pipeline_stages[*pipeline_end_rank].exit_node_coord;
            auto [intermed_send, intermed_recv] = create_intermed_socket_pair(my_recv, end_coord);
            uint32_t output_linear_index = ((end_coord[0] * mesh_device_->shape()[1]) + end_coord[1]);

            auto run_receiver_step = [&](uint32_t i) {
                ttnn::experimental::recv_async(intermediate_tensor, recv_socket);
                ttnn::experimental::send_async(intermediate_tensor, intermed_send);
                Tensor output_tensor = tt::tt_metal::create_device_tensor(tensor_spec, mesh_device_.get());
                ttnn::experimental::recv_async(output_tensor, intermed_recv);
                auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device_, /*dim=*/0);
                auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<uint32_t>();
                auto expected_output_data = ttnn::arange(i, num_elems + i, 1, tt::tt_metal::DataType::UINT32);
                auto expected_output_data_vector = expected_output_data.to_vector<uint32_t>();
                auto chunked_output_vector = std::vector<uint32_t>(
                    output_data.begin() + output_linear_index * num_elems,
                    output_data.begin() + (output_linear_index + 1) * num_elems);
                EXPECT_EQ(chunked_output_vector, expected_output_data_vector);
            };

            for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
                run_receiver_step(i);
            }
        }
    }
    barrier();
}

}  // namespace tt::tt_metal
