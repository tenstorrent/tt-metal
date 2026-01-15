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
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/send_recv_op_utils.hpp"

namespace tt::tt_metal {

class MeshDeviceDual2x4SendRecvFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceDual2x4Fixture,
                                         public testing::WithParamInterface<SocketTestArgs> {};

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
    auto tag = tt::tt_metal::distributed::multihost::Tag{0};
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    uint32_t socket_fifo_size = 10 * 1024;
    auto mesh_shape = mesh_device->shape();
    std::vector<distributed::SocketConnection> forward_socket_connections;
    forward_socket_connections.reserve(mesh_shape.mesh_size());
    std::vector<distributed::SocketConnection> backward_socket_connections;
    backward_socket_connections.reserve(mesh_shape.mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_shape)) {
        forward_socket_connections.push_back(distributed::SocketConnection(
            distributed::MeshCoreCoord(coord, sender_logical_coord),
            distributed::MeshCoreCoord(coord, recv_logical_coord)));
        backward_socket_connections.push_back(distributed::SocketConnection(
            distributed::MeshCoreCoord(coord, sender_logical_coord),
            distributed::MeshCoreCoord(coord, recv_logical_coord)));
    }

    distributed::SocketMemoryConfig socket_mem_config(socket_buffer_type, socket_fifo_size);

    distributed::SocketConfig forward_socket_config(
        forward_socket_connections, socket_mem_config, sender_rank, receiver_rank);
    distributed::SocketConfig backward_socket_config(
        backward_socket_connections, socket_mem_config, receiver_rank, sender_rank);
    auto forward_socket = distributed::MeshSocket(mesh_device, forward_socket_config);
    auto backward_socket = distributed::MeshSocket(mesh_device, backward_socket_config);
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    const auto& input_shape = tensor_spec.logical_shape();
    const auto& memory_config = tensor_spec.memory_config();
    uint32_t num_elems = input_shape.volume();
    auto layout = tensor_spec.layout();
    auto dtype = tensor_spec.data_type();
    if (*(distributed_context->rank()) == *sender_rank) {
        const Tensor input_tensor =
            ttnn::distributed::distribute_tensor(
                ttnn::experimental::view(ttnn::arange(seed, seed + num_elems, 1, dtype), input_shape).to_layout(layout),
                *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device),
                std::nullopt)
                .to_device(mesh_device.get(), memory_config);
        ttnn::experimental::send_async(input_tensor, forward_socket);
        distributed::Synchronize(mesh_device.get(), std::nullopt);
        auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device, /*dim=*/0);
        auto input_data = ttnn::distributed::aggregate_tensor(input_tensor, *composer).to_vector<T>();
        // Send test results to the receiver host
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(input_data.data()), input_data.size() * sizeof(T)),
            receiver_rank,  // send to receiver host
            tag             // exchange test results over tag 0
        );
        auto output_tensor = tt::tt_metal::create_device_tensor(
            TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
            mesh_device.get());
        ttnn::experimental::recv_async(output_tensor, backward_socket);
        distributed::Synchronize(mesh_device.get(), std::nullopt);
        auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<T>();
        std::vector<T> inc_output_data(output_data.size());
        distributed_context->recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(inc_output_data.data()), inc_output_data.size() * sizeof(T)),
            receiver_rank,  // recv from receiver host
            tag             // exchange test results over tag 0
        );
        EXPECT_EQ(output_data, inc_output_data);
    } else {
        auto output_tensor = tt::tt_metal::create_device_tensor(
            TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
            mesh_device.get());
        ttnn::experimental::recv_async(output_tensor, forward_socket);
        distributed::Synchronize(mesh_device.get(), std::nullopt);
        auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device, /*dim=*/0);
        auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<T>();
        std::vector<T> input_data(output_data.size());
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(input_data.data()), input_data.size() * sizeof(T)),
            sender_rank,  // recv from sender host
            tag           // exchange test results over tag 0
        );
        EXPECT_EQ(input_data, output_data);
        auto inc_output_tensor = ttnn::add(output_tensor, 1);
        ttnn::experimental::send_async(inc_output_tensor, backward_socket);
        distributed::Synchronize(mesh_device.get(), std::nullopt);
        auto inc_output_data = ttnn::distributed::aggregate_tensor(inc_output_tensor, *composer).to_vector<T>();
        distributed_context->send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(inc_output_data.data()), inc_output_data.size() * sizeof(T)),
            sender_rank,  // send to sender host
            tag           // exchange test results over tag 0
        );
    }
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

}  // namespace tt::tt_metal
