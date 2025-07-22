// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/mesh_socket.hpp>
#include "send_recv_op_utils.hpp"

namespace tt::tt_metal {

class T3K2DFabricSendRecvFixture : public T3000MeshDevice2DFabricFixture,
                                   public testing::WithParamInterface<SocketTestArgs> {};

template <typename T>
void test_send_recv_async_(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md0,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md1,
    const TensorSpec& tensor_spec,
    BufferType socket_buffer_type,
    uint32_t seed) {
    auto mesh_shape = md0->shape();
    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    uint32_t socket_fifo_size = 10 * 1024;
    std::vector<distributed::SocketConnection> socket_connections;
    socket_connections.reserve(mesh_shape.mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_shape)) {
        socket_connections.push_back({
            .sender_core = {coord, sender_logical_coord},
            .receiver_core = {coord, recv_logical_coord},
        });
    }

    distributed::SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = socket_buffer_type,
        .fifo_size = socket_fifo_size,
    };

    distributed::SocketConfig socket_config = {
        .socket_connection_config = socket_connections, .socket_mem_config = socket_mem_config};
    auto [forward_send_socket, forward_recv_socket] =
        distributed::MeshSocket::create_socket_pair(md0, md1, socket_config);
    auto [backward_send_socket, backward_recv_socket] =
        distributed::MeshSocket::create_socket_pair(md1, md0, socket_config);
    const auto& input_shape = tensor_spec.logical_shape();
    const auto& memory_config = tensor_spec.memory_config();
    uint32_t num_elems = input_shape.volume();
    auto layout = tensor_spec.layout();
    auto dtype = tensor_spec.data_type();
    // Replicate the tensor across (1, num_devices) submesh.
    const Tensor md0_input_tensor =
        ttnn::distributed::distribute_tensor(
            ttnn::experimental::view(ttnn::arange(seed, seed + num_elems, 1, dtype), input_shape).to_layout(layout),
            *ttnn::distributed::replicate_tensor_to_mesh_mapper(*md0),
            std::nullopt)
            .to_device(md0.get(), memory_config);
    auto md1_input_tensor = tt::tt_metal::allocate_tensor_on_device(
        TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
        md1.get());
    ttnn::experimental::send_async(md0_input_tensor, forward_send_socket);
    ttnn::experimental::recv_async(md1_input_tensor, forward_recv_socket);
    distributed::Synchronize(md0.get(), std::nullopt);
    distributed::Synchronize(md1.get(), std::nullopt);
    auto md0_composer = ttnn::distributed::concat_mesh_to_tensor_composer(*md0, /*dim=*/0);
    auto md1_composer = ttnn::distributed::concat_mesh_to_tensor_composer(*md1, /*dim=*/0);
    auto md0_input_data = ttnn::distributed::aggregate_tensor(md0_input_tensor, *md0_composer).to_vector<T>();
    auto md1_input_data = ttnn::distributed::aggregate_tensor(md1_input_tensor, *md1_composer).to_vector<T>();
    EXPECT_EQ(md0_input_data, md1_input_data);
    auto md1_inc_output_tensor = ttnn::add(md1_input_tensor, 1);
    auto md0_inc_output_tensor = tt::tt_metal::allocate_tensor_on_device(
        TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
        md0.get());
    ttnn::experimental::send_async(md1_inc_output_tensor, backward_send_socket);
    ttnn::experimental::recv_async(md0_inc_output_tensor, backward_recv_socket);
    distributed::Synchronize(md1.get(), std::nullopt);
    distributed::Synchronize(md0.get(), std::nullopt);
    auto md0_inc_output_data = ttnn::distributed::aggregate_tensor(md0_inc_output_tensor, *md0_composer).to_vector<T>();
    auto md1_inc_output_data = ttnn::distributed::aggregate_tensor(md1_inc_output_tensor, *md1_composer).to_vector<T>();
    EXPECT_EQ(md0_inc_output_data, md1_inc_output_data);
}

void test_send_recv_async(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md0,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& md1,
    const TensorSpec& tensor_spec,
    BufferType socket_buffer_type,
    uint32_t seed) {
    switch (tensor_spec.data_type()) {
        case tt::tt_metal::DataType::BFLOAT16:
            test_send_recv_async_<bfloat16>(md0, md1, tensor_spec, socket_buffer_type, seed);
            break;
        case tt::tt_metal::DataType::UINT32:
            test_send_recv_async_<uint32_t>(md0, md1, tensor_spec, socket_buffer_type, seed);
            break;
        default: GTEST_SKIP() << "Unsupported data type: " << tensor_spec.data_type(); break;
    }
}

TEST_P(T3K2DFabricSendRecvFixture, SendRecvAsync) {
    auto [tensor_spec, socket_buffer_type] = GetParam();
    auto mesh_device = get_mesh_device();
    auto mesh_shape = distributed::MeshShape(2, 2);
    auto md0 = mesh_device->create_submesh(mesh_shape, distributed::MeshCoordinate(0, 0));
    auto md1 = mesh_device->create_submesh(mesh_shape, distributed::MeshCoordinate(0, 2));
    for (uint32_t i = 0; i < 10; i++) {
        test_send_recv_async(md0, md1, tensor_spec, socket_buffer_type, i);
    }
}

INSTANTIATE_TEST_SUITE_P(
    T3K2DFabricSendRecvTests, T3K2DFabricSendRecvFixture, ::testing::ValuesIn(get_socket_test_args()));

}  // namespace tt::tt_metal
