// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/socket_copy/socket_copy.hpp"
#include <tt-metalium/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/send_recv_op_utils.hpp"
#include <chrono>
#include "tt_metal/fabric/physical_system_descriptor.hpp"

namespace tt::tt_metal {

class MeshDeviceDual2x4SendRecvFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceDual2x4Fixture,
                                         public testing::WithParamInterface<SocketTestArgs> {};

using MeshDeviceClosetBoxSendRecvFixture = tt::tt_fabric::fabric_router_tests::MeshDeviceClosetBoxFabricFixture;

INSTANTIATE_TEST_SUITE_P(
    MeshDeviceDual2x4SendRecvTests, MeshDeviceDual2x4SendRecvFixture, ::testing::ValuesIn(get_socket_test_args()));

class MeshDeviceSplit2x2SendRecvFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceSplit2x2Fixture,
                                          public testing::WithParamInterface<SocketTestArgs> {};

INSTANTIATE_TEST_SUITE_P(
    MeshDeviceSplit2x2SendRecvTests, MeshDeviceSplit2x2SendRecvFixture, ::testing::ValuesIn(get_socket_test_args()));

class MeshDeviceSplit1x2SendRecvFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceSplit1x2Fixture,
                                          public testing::WithParamInterface<SocketTestArgs> {};

INSTANTIATE_TEST_SUITE_P(
    MeshDeviceSplit1x2SendRecvTests, MeshDeviceSplit1x2SendRecvFixture, ::testing::ValuesIn(get_socket_test_args()));

class MeshDeviceNanoExabox2x4SendRecvFixture
    : public tt::tt_fabric::fabric_router_tests::MeshDeviceNanoExabox2x4Fixture,
      public testing::WithParamInterface<SocketTestArgs> {};
INSTANTIATE_TEST_SUITE_P(
    MeshDeviceNanoExabox2x4SendRecvTests,
    MeshDeviceNanoExabox2x4SendRecvFixture,
    ::testing::ValuesIn(get_socket_test_args()));

template <typename T>
void test_send_recv_async_(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const TensorSpec& tensor_spec,
    BufferType socket_buffer_type,
    distributed::multihost::Rank sender_rank,
    distributed::multihost::Rank receiver_rank,
    uint32_t seed) {
    // auto tag = tt::tt_metal::distributed::multihost::Tag{0};
    // auto sender_logical_coord = CoreCoord(0, 0);
    // auto recv_logical_coord = CoreCoord(0, 1);
    // uint32_t socket_fifo_size = 10 * 1024;
    // auto mesh_shape = mesh_device->shape();
    // std::vector<distributed::SocketConnection> forward_socket_connections;
    // forward_socket_connections.reserve(mesh_shape.mesh_size());
    // std::vector<distributed::SocketConnection> backward_socket_connections;
    // backward_socket_connections.reserve(mesh_shape.mesh_size());
    // for (const auto& coord : distributed::MeshCoordinateRange(mesh_shape)) {
    //     forward_socket_connections.push_back({
    //         .sender_core = {coord, sender_logical_coord},
    //         .receiver_core = {coord, recv_logical_coord},
    //     });
    //     backward_socket_connections.push_back({
    //         .sender_core = {coord, sender_logical_coord},
    //         .receiver_core = {coord, recv_logical_coord},
    //     });
    // }

    // distributed::SocketMemoryConfig socket_mem_config = {
    //     .socket_storage_type = socket_buffer_type,
    //     .fifo_size = socket_fifo_size,
    // };

    // distributed::SocketConfig forward_socket_config = {
    //     .socket_connection_config = forward_socket_connections,
    //     .socket_mem_config = socket_mem_config,
    //     .sender_rank = sender_rank,
    //     .receiver_rank = receiver_rank};
    // distributed::SocketConfig backward_socket_config = {
    //     .socket_connection_config = backward_socket_connections,
    //     .socket_mem_config = socket_mem_config,
    //     .sender_rank = receiver_rank,
    //     .receiver_rank = sender_rank};
    // auto forward_socket = distributed::MeshSocket(mesh_device, forward_socket_config);
    // auto backward_socket = distributed::MeshSocket(mesh_device, backward_socket_config);
    // const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    // const auto& input_shape = tensor_spec.logical_shape();
    // const auto& memory_config = tensor_spec.memory_config();
    // uint32_t num_elems = input_shape.volume();
    // auto layout = tensor_spec.layout();
    // auto dtype = tensor_spec.data_type();
    // if (*(distributed_context->rank()) == *sender_rank) {
    //     const Tensor input_tensor =
    //         ttnn::distributed::distribute_tensor(
    //             ttnn::experimental::view(ttnn::arange(seed, seed + num_elems, 1, dtype),
    //             input_shape).to_layout(layout), *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device),
    //             std::nullopt)
    //             .to_device(mesh_device.get(), memory_config);
    //     ttnn::experimental::send_async(input_tensor, mesh_device, forward_socket_config);
    //     distributed::Synchronize(mesh_device.get(), std::nullopt);
    //     auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device, /*dim=*/0);
    //     auto input_data = ttnn::distributed::aggregate_tensor(input_tensor, *composer).to_vector<T>();
    //     // Send test results to the receiver host
    //     distributed_context->send(
    //         tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(input_data.data()), input_data.size() * sizeof(T)),
    //         receiver_rank,  // send to receiver host
    //         tag             // exchange test results over tag 0
    //     );
    //     auto output_tensor = tt::tt_metal::allocate_tensor_on_device(
    //         TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout),
    //         memory_config)), mesh_device.get());
    //     ttnn::experimental::recv_async(output_tensor, mesh_device, backward_socket_config);
    //     distributed::Synchronize(mesh_device.get(), std::nullopt);
    //     auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<T>();
    //     std::vector<T> inc_output_data(output_data.size());
    //     distributed_context->recv(
    //         tt::stl::Span<std::byte>(
    //             reinterpret_cast<std::byte*>(inc_output_data.data()), inc_output_data.size() * sizeof(T)),
    //         receiver_rank,  // recv from receiver host
    //         tag             // exchange test results over tag 0
    //     );
    //     EXPECT_EQ(output_data, inc_output_data);
    // } else {
    //     auto output_tensor = tt::tt_metal::allocate_tensor_on_device(
    //         TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout),
    //         memory_config)), mesh_device.get());
    //     ttnn::experimental::recv_async(output_tensor, mesh_device, forward_socket_config);
    //     distributed::Synchronize(mesh_device.get(), std::nullopt);
    //     auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device, /*dim=*/0);
    //     auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<T>();
    //     std::vector<T> input_data(output_data.size());
    //     distributed_context->recv(
    //         tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(input_data.data()), input_data.size() * sizeof(T)),
    //         sender_rank,  // recv from sender host
    //         tag           // exchange test results over tag 0
    //     );
    //     EXPECT_EQ(input_data, output_data);
    //     auto inc_output_tensor = ttnn::add(output_tensor, 1);
    //     ttnn::experimental::send_async(inc_output_tensor, mesh_device, backward_socket_config);
    //     distributed::Synchronize(mesh_device.get(), std::nullopt);
    //     auto inc_output_data = ttnn::distributed::aggregate_tensor(inc_output_tensor, *composer).to_vector<T>();
    //     distributed_context->send(
    //         tt::stl::Span<std::byte>(
    //             reinterpret_cast<std::byte*>(inc_output_data.data()), inc_output_data.size() * sizeof(T)),
    //         sender_rank,  // send to sender host
    //         tag           // exchange test results over tag 0
    //     );
    // }
}

void test_send_recv_async(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const TensorSpec& tensor_spec,
    BufferType socket_buffer_type,
    distributed::multihost::Rank sender_rank,
    distributed::multihost::Rank receiver_rank,
    uint32_t seed) {
    switch (tensor_spec.data_type()) {
        case tt::tt_metal::DataType::BFLOAT16:
            test_send_recv_async_<bfloat16>(
                mesh_device, tensor_spec, socket_buffer_type, sender_rank, receiver_rank, seed);
            break;
        case tt::tt_metal::DataType::UINT32:
            test_send_recv_async_<uint32_t>(
                mesh_device, tensor_spec, socket_buffer_type, sender_rank, receiver_rank, seed);
            break;
        default: GTEST_SKIP() << "Unsupported data type: " << tensor_spec.data_type(); break;
    }
}

TEST_P(MeshDeviceDual2x4SendRecvFixture, SendRecvAsync) {
    auto [tensor_spec, socket_buffer_type] = GetParam();
    for (uint32_t i = 0; i < 10; i++) {
        test_send_recv_async(
            mesh_device_,
            tensor_spec,
            socket_buffer_type,
            distributed::multihost::Rank{0},
            distributed::multihost::Rank{1},
            i);
        test_send_recv_async(
            mesh_device_,
            tensor_spec,
            socket_buffer_type,
            distributed::multihost::Rank{1},
            distributed::multihost::Rank{0},
            i);
    }
}

TEST_P(MeshDeviceSplit2x2SendRecvFixture, SendRecvAsync) {
    auto [tensor_spec, socket_buffer_type] = GetParam();
    for (uint32_t i = 0; i < 10; i++) {
        test_send_recv_async(
            mesh_device_,
            tensor_spec,
            socket_buffer_type,
            distributed::multihost::Rank{0},
            distributed::multihost::Rank{1},
            i);
        test_send_recv_async(
            mesh_device_,
            tensor_spec,
            socket_buffer_type,
            distributed::multihost::Rank{1},
            distributed::multihost::Rank{0},
            i);
    }
}

TEST_P(MeshDeviceSplit1x2SendRecvFixture, MultiSendRecvAsync) {
    constexpr distributed::multihost::Rank receiver_rank = distributed::multihost::Rank{0};
    auto [tensor_spec, socket_buffer_type] = GetParam();
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    auto rank = *(distributed_context->rank());
    for (uint32_t i = 0; i < 10; i++) {
        if (rank == *receiver_rank || rank == 1) {
            test_send_recv_async(
                mesh_device_, tensor_spec, socket_buffer_type, distributed::multihost::Rank{1}, receiver_rank, i);
        }
        if (rank == *receiver_rank || rank == 2) {
            test_send_recv_async(
                mesh_device_, tensor_spec, socket_buffer_type, distributed::multihost::Rank{2}, receiver_rank, i);
        }
        if (rank == *receiver_rank || rank == 3) {
            test_send_recv_async(
                mesh_device_, tensor_spec, socket_buffer_type, distributed::multihost::Rank{3}, receiver_rank, i);
        }
    }
}

TEST_P(MeshDeviceNanoExabox2x4SendRecvFixture, MultiSendRecvAsync) {
    constexpr distributed::multihost::Rank receiver_rank = distributed::multihost::Rank{1};
    auto [tensor_spec, socket_buffer_type] = GetParam();
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    auto rank = *(distributed_context->rank());
    for (uint32_t i = 0; i < 10; i++) {
        for (uint32_t r = 0; r < *(distributed_context->size()); r++) {
            if (r == *receiver_rank) {
                continue;
            }
            if (rank == *receiver_rank || rank == r) {
                test_send_recv_async(
                    mesh_device_, tensor_spec, socket_buffer_type, distributed::multihost::Rank{r}, receiver_rank, i);
            }
        }
    }
}

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

std::pair<distributed::MeshCoordinate, distributed::MeshCoordinate> get_connecting_coords(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    uint32_t neighbor_mesh_id,
    const std::unordered_map<tt::tt_metal::AsicID, distributed::MeshCoordinate>& asic_id_to_mesh_coord,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    auto neighbor_rank = neighbor_mesh_id;
    auto my_host = physical_system_descriptor.my_host_name();
    std::string neighbor_host;
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    auto my_mesh_id = *distributed_context->rank();
    for (const auto& [host_name, rank] : physical_system_descriptor.get_host_to_rank_map()) {
        if (rank == neighbor_rank) {
            neighbor_host = host_name;
            break;
        }
    }
    auto exit_nodes = physical_system_descriptor.get_connecting_exit_nodes(my_host, neighbor_host);
    if (my_mesh_id == 0 && neighbor_mesh_id == 1) {
        return std::make_pair(distributed::MeshCoordinate(1, 1), distributed::MeshCoordinate(1, 2));
    }
    if (my_mesh_id == 1 && neighbor_mesh_id == 0) {
        return std::make_pair(distributed::MeshCoordinate(1, 2), distributed::MeshCoordinate(1, 1));
    }
    if (my_mesh_id == 1 && neighbor_mesh_id == 2) {
        return std::make_pair(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(0, 3));
    }
    if (my_mesh_id == 2 && neighbor_mesh_id == 1) {
        return std::make_pair(distributed::MeshCoordinate(0, 3), distributed::MeshCoordinate(0, 0));
    }
    if (my_mesh_id == 2 && neighbor_mesh_id == 3) {
        return std::make_pair(distributed::MeshCoordinate(1, 2), distributed::MeshCoordinate(0, 2));
    }
    if (my_mesh_id == 3 && neighbor_mesh_id == 2) {
        return std::make_pair(distributed::MeshCoordinate(0, 2), distributed::MeshCoordinate(1, 2));
    }
    TT_FATAL(false, "No connecting coords found for mesh {} and neighbor {}", my_mesh_id, neighbor_mesh_id);
    return std::make_pair(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(0, 0));
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

TEST_F(MeshDeviceClosetBoxSendRecvFixture, SendRecvPipeline) {
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    distributed::multihost::Rank pipeline_start_rank = distributed::multihost::Rank{0};
    distributed::multihost::Rank pipeline_end_rank = distributed::multihost::Rank{*distributed_context->size() - 1};
    std::cout << "starting test_send_recv_pipeline" << std::endl;

    // auto sender_logical_coord = CoreCoord(0, 0);
    // auto recv_logical_coord = CoreCoord(0, 1);

    uint32_t socket_fifo_size = 56 * 1024;
    auto mesh_shape = mesh_device_->shape();

    auto physical_system_descriptor = create_physical_system_descriptor();
    auto asic_id_to_mesh_coord = get_asic_id_to_mesh_coord_map(mesh_device_);

    auto my_mesh_id = *distributed_context->rank();
    auto send_mesh_id = my_mesh_id - 1;
    auto recv_mesh_id = my_mesh_id + 1;

    distributed::SocketConnection fwd_connection = {
        .sender_core = {distributed::MeshCoordinate(0, 0), CoreCoord(0, 0)},
        .receiver_core = {distributed::MeshCoordinate(0, 0), CoreCoord(0, 0)}};

    distributed::SocketConnection bwd_connection = {
        .sender_core = {distributed::MeshCoordinate(0, 0), CoreCoord(0, 0)},
        .receiver_core = {distributed::MeshCoordinate(0, 0), CoreCoord(0, 0)}};

    distributed::SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    auto tensor_spec = TensorSpec(
        ttnn::Shape({1, 1, 1, 3584}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM)));

    constexpr uint32_t num_iters = 100000;
    uint64_t start_time = 0;
    uint64_t end_time = 0;
    // for (const auto& coord : distributed::MeshCoordinateRange(mesh_device_->shape())) {
    //     std::cout << "Mesh: " << *distributed_context->rank() << " Coord: " << coord << " Fabric Node ID: " <<
    //     mesh_device_->get_fabric_node_id(coord) << std::endl;
    // }
    if (*distributed_context->rank() == *pipeline_start_rank) {
        const auto& input_shape = tensor_spec.logical_shape();
        const auto& memory_config = tensor_spec.memory_config();
        uint32_t num_elems = input_shape.volume();
        auto layout = tensor_spec.layout();
        auto dtype = tensor_spec.data_type();
        auto [my_sender, downstream_recv] =
            get_connecting_coords(physical_system_descriptor, recv_mesh_id, asic_id_to_mesh_coord, mesh_device_);
        distributed::MeshCoordinate start_coord = distributed::MeshCoordinate(0, 3);
        auto start_fabric_node_id = mesh_device_->get_fabric_node_id(start_coord);
        auto sender_fabric_node_id = mesh_device_->get_fabric_node_id(my_sender);
        std::cout << "Sender coords: " << my_sender[0] << ", " << my_sender[1] << std::endl;
        std::cout << "Sender Fabric Node ID: " << *sender_fabric_node_id.mesh_id << ", "
                  << sender_fabric_node_id.chip_id << std::endl;
        std::cout << "Start coords: " << start_coord[0] << ", " << start_coord[1] << std::endl;
        std::cout << "Start Fabric Node ID: " << *start_fabric_node_id.mesh_id << ", " << start_fabric_node_id.chip_id
                  << std::endl;

        distributed::SocketConnection intermed_connection = {
            .sender_core = {start_coord, CoreCoord(0, 0)}, .receiver_core = {my_sender, CoreCoord(0, 0)}};
        distributed::SocketConfig intermed_socket_config = {
            .socket_connection_config = {intermed_connection},
            .socket_mem_config = socket_mem_config,
        };
        auto [intermed_send, intermed_recv] =
            distributed::MeshSocket::create_socket_pair(mesh_device_, mesh_device_, intermed_socket_config);

        fwd_connection = {
            .sender_core = {my_sender, CoreCoord(0, 0)}, .receiver_core = {downstream_recv, CoreCoord(0, 0)}};
        distributed::SocketConfig send_socket_config = {
            .socket_connection_config = {fwd_connection},
            .socket_mem_config = socket_mem_config,
            .sender_rank = distributed_context->rank(),
            .receiver_rank = distributed::multihost::Rank(recv_mesh_id)};
        auto send_socket = distributed::MeshSocket(mesh_device_, send_socket_config);
        auto input_tensor =
            ttnn::distributed::distribute_tensor(
                ttnn::experimental::view(ttnn::arange(0, num_elems, 1, dtype), input_shape).to_layout(layout),
                *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
                std::nullopt)
                .to_device(mesh_device_.get(), memory_config);
        // Block after sending data. We will profile the time taken to forward data over sockets
        // after the write has completed.
        // Send data from start device to socked on exit node
        ttnn::experimental::send_async(input_tensor, intermed_send);
        ttnn::experimental::socket_copy(input_tensor, intermed_recv, send_socket, num_elems * sizeof(uint32_t));
        Synchronize(mesh_device_.get(), std::nullopt);
        distributed_context->barrier();
        ttnn::experimental::send_async(input_tensor, intermed_send);
        ttnn::experimental::socket_copy(input_tensor, intermed_recv, send_socket, num_elems * sizeof(uint32_t));
    } else {
        auto [my_recv, upstream_send] =
            get_connecting_coords(physical_system_descriptor, send_mesh_id, asic_id_to_mesh_coord, mesh_device_);
        bwd_connection = {.sender_core = {upstream_send, CoreCoord(0, 0)}, .receiver_core = {my_recv, CoreCoord(0, 0)}};

        distributed::SocketConfig recv_socket_config = {
            .socket_connection_config = {bwd_connection},
            .socket_mem_config = socket_mem_config,
            .sender_rank = distributed::multihost::Rank(send_mesh_id),
            .receiver_rank = distributed_context->rank()};
        auto recv_socket = distributed::MeshSocket(mesh_device_, recv_socket_config);
        distributed::MeshSocket send_socket;
        distributed::MeshSocket intermed_send;
        distributed::MeshSocket intermed_recv;
        if (*distributed_context->rank() < *pipeline_end_rank) {
            auto [my_sender, downstream_recv] =
                get_connecting_coords(physical_system_descriptor, recv_mesh_id, asic_id_to_mesh_coord, mesh_device_);
            fwd_connection = {
                .sender_core = {my_sender, CoreCoord(0, 0)}, .receiver_core = {downstream_recv, CoreCoord(0, 0)}};
            distributed::SocketConfig send_socket_config = {
                .socket_connection_config = {fwd_connection},
                .socket_mem_config = socket_mem_config,
                .sender_rank = distributed_context->rank(),
                .receiver_rank = distributed::multihost::Rank(recv_mesh_id)};
            send_socket = distributed::MeshSocket(mesh_device_, send_socket_config);
            // Create an intermediate socket to forward data from the recv socket to the send socket
            distributed::SocketConnection intermed_connection = {
                .sender_core = {my_recv, CoreCoord(0, 0)}, .receiver_core = {my_sender, CoreCoord(0, 0)}};

            distributed::SocketConfig intermed_socket_config = {
                .socket_connection_config = {intermed_connection},
                .socket_mem_config = socket_mem_config,
            };
            std::tie(intermed_send, intermed_recv) =
                distributed::MeshSocket::create_socket_pair(mesh_device_, mesh_device_, intermed_socket_config);
        } else {
            distributed::MeshCoordinate end_coord = distributed::MeshCoordinate(1, 0);
            distributed::SocketConnection end_connection = {
                .sender_core = {my_recv, CoreCoord(0, 0)}, .receiver_core = {end_coord, CoreCoord(0, 0)}};
            distributed::SocketConfig end_socket_config = {
                .socket_connection_config = {end_connection},
                .socket_mem_config = socket_mem_config,
            };
            std::tie(intermed_send, intermed_recv) =
                distributed::MeshSocket::create_socket_pair(mesh_device_, mesh_device_, end_socket_config);
        }
        // Allocate Output Tensor on the last device in the pipeline
        Tensor output_tensor = tt::tt_metal::allocate_tensor_on_device(tensor_spec, mesh_device_.get());

        if (*distributed_context->rank() < *pipeline_end_rank) {
            // Copy from previous pipeline stage to my exit node
            ttnn::experimental::socket_copy(output_tensor, recv_socket, intermed_send, 14336);
            // Copy from my exit node to the next pipeline stage
            ttnn::experimental::socket_copy(output_tensor, intermed_recv, send_socket, 14336);
        } else {
            // Receive into output tensor on last device in the pipeline
            ttnn::experimental::socket_copy(output_tensor, recv_socket, intermed_send, 14336);
            ttnn::experimental::recv_async(output_tensor, intermed_recv);
        }

        Synchronize(mesh_device_.get(), std::nullopt);
        distributed_context->barrier();

        if (*distributed_context->rank() < *pipeline_end_rank) {
            ttnn::experimental::socket_copy(output_tensor, recv_socket, intermed_send, 14336);
            ttnn::experimental::socket_copy(output_tensor, intermed_recv, send_socket, 14336);
        } else {
            start_time = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::high_resolution_clock::now().time_since_epoch())
                             .count();
            ttnn::experimental::socket_copy(output_tensor, recv_socket, intermed_send, 14336);
            ttnn::experimental::recv_async(output_tensor, intermed_recv);
        }
    }

    distributed::Synchronize(mesh_device_.get(), std::nullopt);
    if (distributed_context->rank() == pipeline_end_rank) {
        end_time = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now().time_since_epoch())
                       .count();
        std::cout << "Time taken to forward: " << num_iters << " Packets: " << end_time - start_time << " us"
                  << std::endl;
        std::cout << "Time per iteration: " << (end_time - start_time) / num_iters << " us" << std::endl;
    }
}

}  // namespace tt::tt_metal
