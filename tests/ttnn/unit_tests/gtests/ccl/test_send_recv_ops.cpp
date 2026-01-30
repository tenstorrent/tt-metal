// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "ttnn/operations/experimental/ccl/send_recv_async/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/recv_async/recv_async.hpp"
#include "ttnn/operations/experimental/ccl/send_recv_async/socket_forward/socket_forward.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "send_recv_op_utils.hpp"

namespace tt::tt_metal {

class FabricSendRecv2x4Fixture : public MeshDevice4x8Fabric2DFixture,
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
        socket_connections.push_back(distributed::SocketConnection(
            distributed::MeshCoreCoord(coord, sender_logical_coord),
            distributed::MeshCoreCoord(coord, recv_logical_coord)));
    }

    distributed::SocketMemoryConfig socket_mem_config(socket_buffer_type, socket_fifo_size);

    distributed::SocketConfig socket_config(socket_connections, socket_mem_config);
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
    auto md1_input_tensor = tt::tt_metal::create_device_tensor(
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
    auto md0_inc_output_tensor = tt::tt_metal::create_device_tensor(
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

TEST_P(FabricSendRecv2x4Fixture, SendRecvAsync) {
    auto [tensor_spec, socket_buffer_type] = GetParam();
    auto mesh_device = get_mesh_device();
    auto mesh_shape = distributed::MeshShape(2, 2);
    auto md0 = mesh_device->create_submesh(mesh_shape, distributed::MeshCoordinate(0, 0));
    auto md1 = mesh_device->create_submesh(mesh_shape, distributed::MeshCoordinate(0, 2));
    for (uint32_t i = 0; i < 10; i++) {
        test_send_recv_async(md0, md1, tensor_spec, socket_buffer_type, i);
    }
}

INSTANTIATE_TEST_SUITE_P(FabricSendRecv2x4Tests, FabricSendRecv2x4Fixture, ::testing::ValuesIn(get_socket_test_args()));

TEST_F(FabricSendRecv2x4Fixture, SRTest) {
    using namespace tt::tt_metal::distributed;
    auto mesh_device = get_mesh_device();

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);
    auto copy_logical_coord = CoreCoord(0, 0);

    auto socket_fifo_size = 56 * 1024;

    auto start_device_coord = distributed::MeshCoordinate(0, 0);
    auto intermed_device_coord = distributed::MeshCoordinate(1, 0);
    // auto intermed_device_coord_2 = distributed::MeshCoordinate(2, 0);
    // auto intermed_device_coord_3 = distributed::MeshCoordinate(3, 0);
    // auto intermed_device_coord_4 = distributed::MeshCoordinate(4, 0);
    // auto intermed_device_coord_5 = distributed::MeshCoordinate(5, 0);
    // auto intermed_device_coord_6 = distributed::MeshCoordinate(6, 0);
    // auto intermed_device_coord_7 = distributed::MeshCoordinate(7, 0);
    // auto intermed_device_coord_8 = distributed::MeshCoordinate(7, 1);
    // auto intermed_device_coord_9 = distributed::MeshCoordinate(6, 1);
    // auto intermed_device_coord_10 = distributed::MeshCoordinate(5, 1);
    // auto intermed_device_coord_11 = distributed::MeshCoordinate(4, 1);
    // auto intermed_device_coord_12 = distributed::MeshCoordinate(3, 1);
    // auto intermed_device_coord_13 = distributed::MeshCoordinate(2, 1);
    // auto intermed_device_coord_14 = distributed::MeshCoordinate(1, 1);
    // auto intermed_device_coord_15 = distributed::MeshCoordinate(0, 1);
    auto end_device_coord = distributed::MeshCoordinate(0, 0);

    std::cout << "Start Device ID: " << mesh_device->get_device(start_device_coord)->id() << std::endl;
    std::cout << "Intermed Device ID: " << mesh_device->get_device(intermed_device_coord)->id() << std::endl;
    // std::cout << "Intermed Device 2 ID: " << mesh_device->get_device(intermed_device_coord_2)->id() << std::endl;
    // std::cout << "Intermed Device 3 ID: " << mesh_device->get_device(intermed_device_coord_3)->id() << std::endl;
    // std::cout << "Intermed Device 4 ID: " << mesh_device->get_device(intermed_device_coord_4)->id() << std::endl;
    // std::cout << "Intermed Device 5 ID: " << mesh_device->get_device(intermed_device_coord_5)->id() << std::endl;
    // std::cout << "Intermed Device 6 ID: " << mesh_device->get_device(intermed_device_coord_6)->id() << std::endl;
    // std::cout << "Intermed Device 7 ID: " << mesh_device->get_device(intermed_device_coord_7)->id() << std::endl;
    // std::cout << "Intermed Device 8 ID: " << mesh_device->get_device(intermed_device_coord_8)->id() << std::endl;
    // std::cout << "Intermed Device 9 ID: " << mesh_device->get_device(intermed_device_coord_9)->id() << std::endl;
    // std::cout << "Intermed Device 10 ID: " << mesh_device->get_device(intermed_device_coord_10)->id() << std::endl;
    // std::cout << "Intermed Device 11 ID: " << mesh_device->get_device(intermed_device_coord_11)->id() << std::endl;
    // std::cout << "Intermed Device 12 ID: " << mesh_device->get_device(intermed_device_coord_12)->id() << std::endl;
    // std::cout << "Intermed Device 13 ID: " << mesh_device->get_device(intermed_device_coord_13)->id() << std::endl;
    // std::cout << "Intermed Device 14 ID: " << mesh_device->get_device(intermed_device_coord_14)->id() << std::endl;
    // std::cout << "Intermed Device 15 ID: " << mesh_device->get_device(intermed_device_coord_15)->id() << std::endl;
    std::cout << "End Device ID: " << mesh_device->get_device(end_device_coord)->id() << std::endl;

    // Create connections for:
    // Stage 0 -> 1
    // Stage 1 -> 2
    distributed::SocketConnection socket_connection_01 = distributed::SocketConnection(
        distributed::MeshCoreCoord(start_device_coord, sender_logical_coord),
        distributed::MeshCoreCoord(intermed_device_coord, copy_logical_coord));
    // distributed::SocketConnection socket_connection_12 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord, copy_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_2, copy_logical_coord));
    // distributed::SocketConnection socket_connection_23 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_2, copy_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_3, copy_logical_coord));
    // distributed::SocketConnection socket_connection_34 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_3, copy_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_4, copy_logical_coord));
    // distributed::SocketConnection socket_connection_45 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_4, copy_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_5, copy_logical_coord));
    // distributed::SocketConnection socket_connection_56 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_5, copy_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_6, recv_logical_coord));
    // distributed::SocketConnection socket_connection_67 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_6, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_7, recv_logical_coord));
    // distributed::SocketConnection socket_connection_78 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_7, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_8, recv_logical_coord));
    // distributed::SocketConnection socket_connection_89 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_8, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_9, recv_logical_coord));
    // distributed::SocketConnection socket_connection_910 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_9, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_10, recv_logical_coord));
    // distributed::SocketConnection socket_connection_1011 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_10, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_11, recv_logical_coord));
    // distributed::SocketConnection socket_connection_1112 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_11, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_12, recv_logical_coord));
    // distributed::SocketConnection socket_connection_1213 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_12, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_13, recv_logical_coord));
    // distributed::SocketConnection socket_connection_1314 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_13, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_14, recv_logical_coord));
    // distributed::SocketConnection socket_connection_1415 = distributed::SocketConnection(
    // distributed::MeshCoreCoord(intermed_device_coord_14, recv_logical_coord),
    // distributed::MeshCoreCoord(intermed_device_coord_15, recv_logical_coord));
    distributed::SocketConnection socket_connection_12 = distributed::SocketConnection(
        distributed::MeshCoreCoord(intermed_device_coord, recv_logical_coord),
        distributed::MeshCoreCoord(end_device_coord, recv_logical_coord));

    distributed::SocketMemoryConfig socket_mem_config =
        distributed::SocketMemoryConfig(BufferType::L1, socket_fifo_size);

    distributed::SocketConfig socket_config_01 = distributed::SocketConfig({socket_connection_01}, socket_mem_config);

    distributed::SocketConfig socket_config_12 = distributed::SocketConfig({socket_connection_12}, socket_mem_config);

    // distributed::SocketConfig socket_config_23 = distributed::SocketConfig({socket_connection_23},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_34 = distributed::SocketConfig({socket_connection_34},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_45 = distributed::SocketConfig({socket_connection_45},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_56 = distributed::SocketConfig({socket_connection_56},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_67 = distributed::SocketConfig({socket_connection_67},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_78 = distributed::SocketConfig({socket_connection_78},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_89 = distributed::SocketConfig({socket_connection_89},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_910 = distributed::SocketConfig({socket_connection_910},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_1011 =
    // distributed::SocketConfig({socket_connection_1011}, socket_mem_config);

    // distributed::SocketConfig socket_config_1112 =
    // distributed::SocketConfig({socket_connection_1112}, socket_mem_config);

    // distributed::SocketConfig socket_config_1213 =
    // distributed::SocketConfig({socket_connection_1213}, socket_mem_config);

    // distributed::SocketConfig socket_config_1314 =
    // distributed::SocketConfig({socket_connection_1314}, socket_mem_config);

    // distributed::SocketConfig socket_config_1415 =
    // distributed::SocketConfig({socket_connection_1415}, socket_mem_config);

    // distributed::SocketConfig socket_config_15end =
    // distributed::SocketConfig({socket_connection_15end}, socket_mem_config);

    auto input_tensor_spec = TensorSpec(
        ttnn::Shape({1, 1, 1, 3584}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    const auto output_tensor_spec = TensorSpec(
        ttnn::Shape({1, 1, 1, 3584}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1)));

    const auto& input_shape = input_tensor_spec.logical_shape();
    const auto& memory_config = input_tensor_spec.memory_config();
    uint32_t num_elems = input_shape.volume();
    auto layout = input_tensor_spec.layout();
    auto dtype = input_tensor_spec.data_type();

    auto output_tensor = tt::tt_metal::create_device_tensor(output_tensor_spec, mesh_device.get());

    auto [send_socket_0, recv_socket_1] =
        distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_01);

    auto [send_socket_1, recv_socket_2] =
        distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_12);

    // auto [send_socket_2, recv_socket_3] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_23);

    // auto [send_socket_3, recv_socket_4] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_34);

    // auto [send_socket_4, recv_socket_5] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_45);

    // auto [send_socket_5, recv_socket_6] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_56);

    // auto [send_socket_6, recv_socket_7] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_67);

    // auto [send_socket_7, recv_socket_8] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_78);

    // auto [send_socket_8, recv_socket_9] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_89);

    // auto [send_socket_9, recv_socket_10] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_910);

    // auto [send_socket_10, recv_socket_11] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_1011);

    // auto [send_socket_11, recv_socket_12] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_1112);

    // auto [send_socket_12, recv_socket_13] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_1213);

    // auto [send_socket_13, recv_socket_14] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_1314);

    // auto [send_socket_14, recv_socket_15] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_1415);

    // auto [send_socket_15, recv_socket_end] =
    // distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_15end);

    // Create Barrier Buffer
    auto barrier_buffer_size = 832;
    CoreRangeSet barrier_core_range = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    auto shard_params = ShardSpecBuffer(barrier_core_range, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    DeviceLocalBufferConfig barrier_buffer_specs = {
        .page_size = barrier_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    auto barrier_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{.size = barrier_buffer_size}, barrier_buffer_specs, mesh_device.get());
    // Write 0 to barrier buffer
    std::vector<uint32_t> barrier_data(barrier_buffer_size / sizeof(uint32_t), 0);
    EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), barrier_buffer, barrier_data, true);
    std::cout << "Barrier buffer address: " << barrier_buffer->address() << std::endl;

    const uint32_t i = 0;
    const Tensor input_tensor =
        ttnn::distributed::distribute_tensor(
            ttnn::experimental::view(ttnn::arange(i, num_elems + i, 1, dtype), input_shape).to_layout(layout),
            *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device),
            std::nullopt)
            .to_device(mesh_device.get(), memory_config);

    // Sender forwards downstream and waits for ack from last receiver
    ttnn::experimental::send_async(input_tensor, send_socket_0, recv_socket_2);
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_1, send_socket_1, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_2, send_socket_2, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_3, send_socket_3, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_4, send_socket_4, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_5, send_socket_5, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_6, send_socket_6, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_7, send_socket_7, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_8, send_socket_8, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_9, send_socket_9, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_10, send_socket_10, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_11, send_socket_11, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_12, send_socket_12, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_13, send_socket_13, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_14, send_socket_14, num_elems * sizeof(uint32_t));
    ttnn::experimental::socket_forward(output_tensor, recv_socket_1, send_socket_1, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_6, send_socket_6, num_elems * sizeof(uint32_t));
    // ttnn::experimental::recv_async(output_tensor, recv_socket_7);
    // auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device, /*dim=*/0);
    // auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<uint32_t>();
    // auto expected_output_data = ttnn::arange(i, num_elems + i, 1, tt::tt_metal::DataType::UINT32);
    // auto expected_output_data_vector = expected_output_data.to_vector<uint32_t>();

    // auto chunked_output_vector =
    //     std::vector<uint32_t>(output_data.begin() + 31 * num_elems, output_data.begin() + 32 * num_elems);
    // EXPECT_EQ(chunked_output_vector, expected_output_data_vector);

    Synchronize(mesh_device.get(), std::nullopt);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto start_device_id = mesh_device_->get_device(start_device_coord)->id();
    auto start_core_coord = mesh_device_->worker_core_from_logical_core(sender_logical_coord);
    std::vector<uint64_t> latencies = std::vector<uint64_t>(100, 0);
    // uint32_t base_addr = 1572032;
    cluster.read_core(
        latencies.data(), sizeof(uint64_t) * 100, tt_cxy_pair(start_device_id, start_core_coord), barrier_buffer_size);
    for (auto latency : latencies) {
        std::cout << latency << std::endl;
    }
}

}  // namespace tt::tt_metal
