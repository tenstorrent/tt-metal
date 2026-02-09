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

    constexpr uint32_t XFER_SIZE = 14 * 1024;
    auto socket_fifo_size = XFER_SIZE * 16;

    auto start_device_coord = distributed::MeshCoordinate(0, 0);
    auto intermed_device_coord = distributed::MeshCoordinate(1, 0);
    auto intermed_device_coord_2 = distributed::MeshCoordinate(1, 1);
    auto intermed_device_coord_3 = distributed::MeshCoordinate(0, 1);
    auto end_device_coord = distributed::MeshCoordinate(0, 0);

    std::cout << "Start Device ID: " << mesh_device->get_device(start_device_coord)->id() << std::endl;
    std::cout << "Intermed Device ID: " << mesh_device->get_device(intermed_device_coord)->id() << std::endl;
    std::cout << "Intermed Device 2 ID: " << mesh_device->get_device(intermed_device_coord_2)->id() << std::endl;
    std::cout << "Intermed Device 3 ID: " << mesh_device->get_device(intermed_device_coord_3)->id() << std::endl;
    std::cout << "End Device ID: " << mesh_device->get_device(end_device_coord)->id() << std::endl;

    // Create connections for:
    // Stage 0 -> 1
    // Stage 1 -> 2
    distributed::SocketConnection socket_connection_01 = distributed::SocketConnection(
        distributed::MeshCoreCoord(start_device_coord, sender_logical_coord),
        distributed::MeshCoreCoord(intermed_device_coord, copy_logical_coord));
    distributed::SocketConnection socket_connection_10 = distributed::SocketConnection(
        distributed::MeshCoreCoord(intermed_device_coord, copy_logical_coord),
        distributed::MeshCoreCoord(end_device_coord, recv_logical_coord));

    // distributed::SocketConnection socket_connection_12 = distributed::SocketConnection(
    //     distributed::MeshCoreCoord(intermed_device_coord, copy_logical_coord),
    //     distributed::MeshCoreCoord(intermed_device_coord_2, copy_logical_coord));
    // distributed::SocketConnection socket_connection_23 = distributed::SocketConnection(
    //     distributed::MeshCoreCoord(intermed_device_coord_2, copy_logical_coord),
    //     distributed::MeshCoreCoord(intermed_device_coord_3, copy_logical_coord));
    // distributed::SocketConnection socket_connection_34 = distributed::SocketConnection(
    //     distributed::MeshCoreCoord(intermed_device_coord_3, copy_logical_coord),
    //     distributed::MeshCoreCoord(end_device_coord, recv_logical_coord));

    distributed::SocketMemoryConfig socket_mem_config =
        distributed::SocketMemoryConfig(BufferType::L1, socket_fifo_size);

    distributed::SocketConfig socket_config_01 = distributed::SocketConfig({socket_connection_01}, socket_mem_config);
    distributed::SocketConfig socket_config_10 = distributed::SocketConfig({socket_connection_10}, socket_mem_config);

    // distributed::SocketConfig socket_config_12 = distributed::SocketConfig({socket_connection_12},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_23 = distributed::SocketConfig({socket_connection_23},
    // socket_mem_config);

    // distributed::SocketConfig socket_config_34 = distributed::SocketConfig({socket_connection_34},
    // socket_mem_config);

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

    auto [send_socket_1, recv_socket_end] =
        distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_10);
    // auto [send_socket_1, recv_socket_2] =
    //     distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_12);

    // auto [send_socket_2, recv_socket_3] =
    //     distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_23);

    // auto [send_socket_3, recv_socket_end] =
    //     distributed::MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_34);

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
    // return;
    const uint32_t i = 0;
    const Tensor input_tensor =
        ttnn::distributed::distribute_tensor(
            ttnn::experimental::view(ttnn::arange(i, num_elems + i, 1, dtype), input_shape).to_layout(layout),
            *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device),
            std::nullopt)
            .to_device(mesh_device.get(), memory_config);

    // Sender forwards downstream and waits for ack from last receiver
    ttnn::experimental::send_async(input_tensor, send_socket_0, recv_socket_end);
    ttnn::experimental::socket_forward(output_tensor, recv_socket_1, send_socket_1, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_1, send_socket_1, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_2, send_socket_2, num_elems * sizeof(uint32_t));
    // ttnn::experimental::socket_forward(output_tensor, recv_socket_3, send_socket_3, num_elems * sizeof(uint32_t));
    Synchronize(mesh_device.get(), std::nullopt);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto start_device_id = mesh_device->get_device(start_device_coord)->id();
    auto start_core_coord = mesh_device->worker_core_from_logical_core(sender_logical_coord);
    std::vector<uint64_t> latencies = std::vector<uint64_t>(100, 0);
    uint32_t base_addr = barrier_buffer->address();
    cluster.read_core(
        latencies.data(), sizeof(uint64_t) * 100, tt_cxy_pair(start_device_id, start_core_coord), base_addr);

    int freq_mhz = cluster.get_device_aiclk(start_device_id);
    for (uint32_t i = 0; i < 100; i++) {
        double latency_ns = 1000 * ((float)latencies[i] / freq_mhz);
        std::cout << "Iteration " << i << " RTT latency (ns): " << latency_ns
                  << " Per hop latency (ns): " << latency_ns / 4.0f << std::endl;
    }
}

}  // namespace tt::tt_metal
