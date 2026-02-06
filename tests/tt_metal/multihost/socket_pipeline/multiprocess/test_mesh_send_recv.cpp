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
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_send_recv.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_forward.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <iomanip>

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

// Test using migrated code (metal-level APIs) from multiprocess utils instead of ttnn APIs
// This test mirrors the SRTest from test_send_recv_ops.cpp in the ccl directory
// but uses MeshBuffer and metal-level send_async/socket_forward instead of Tensor and ttnn APIs
// Uses GenericMeshDeviceFabric2DFixture to open all devices (fabric requires all devices active)
class FabricSendRecvMigratedFixture : public GenericMeshDeviceFabric2DFixture {};

void run_sr_test_migrated(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, bool enable_correctness_check) {
    using namespace tt::tt_metal::distributed;

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);
    auto copy_logical_coord = CoreCoord(0, 0);

    constexpr uint32_t XFER_SIZE = 14 * 1024;
    auto socket_fifo_size = XFER_SIZE * 16;

    auto start_device_coord = MeshCoordinate(0, 0);
    auto intermed_device_coord = MeshCoordinate(1, 0);
    auto intermed_device_coord_2 = MeshCoordinate(1, 1);
    auto intermed_device_coord_3 = MeshCoordinate(0, 1);
    auto end_device_coord = MeshCoordinate(0, 0);

    std::cout << "Start Device ID: " << mesh_device->get_device(start_device_coord)->id() << std::endl;
    std::cout << "Intermed Device ID: " << mesh_device->get_device(intermed_device_coord)->id() << std::endl;
    std::cout << "Intermed Device 2 ID: " << mesh_device->get_device(intermed_device_coord_2)->id() << std::endl;
    std::cout << "End Device ID: " << mesh_device->get_device(end_device_coord)->id() << std::endl;

    // Create connections for:
    // Stage 0 -> 1
    // Stage 1 -> 2
    SocketConnection socket_connection_01 = SocketConnection(
        MeshCoreCoord(start_device_coord, sender_logical_coord),
        MeshCoreCoord(intermed_device_coord, copy_logical_coord));
    SocketConnection socket_connection_12 = SocketConnection(
        MeshCoreCoord(intermed_device_coord, copy_logical_coord),
        MeshCoreCoord(intermed_device_coord_2, copy_logical_coord));
    SocketConnection socket_connection_23 = SocketConnection(
        MeshCoreCoord(intermed_device_coord_2, copy_logical_coord),
        MeshCoreCoord(intermed_device_coord_3, copy_logical_coord));
    SocketConnection socket_connection_34 = SocketConnection(
        MeshCoreCoord(intermed_device_coord_3, copy_logical_coord),
        MeshCoreCoord(end_device_coord, recv_logical_coord));

    SocketMemoryConfig socket_mem_config = SocketMemoryConfig(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config_01 = SocketConfig({socket_connection_01}, socket_mem_config);

    SocketConfig socket_config_12 = SocketConfig({socket_connection_12}, socket_mem_config);

    SocketConfig socket_config_23 = SocketConfig({socket_connection_23}, socket_mem_config);

    SocketConfig socket_config_34 = SocketConfig({socket_connection_34}, socket_mem_config);

    // Calculate buffer size from tensor spec equivalent
    constexpr uint32_t num_elems = 3584;
    constexpr uint32_t buffer_size = num_elems * sizeof(uint32_t);

    auto [send_socket_0, recv_socket_1] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_01);

    auto [send_socket_1, recv_socket_2] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_12);

    auto [send_socket_2, recv_socket_3] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_23);

    auto [send_socket_3, recv_socket_end] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_34);

    // Create Barrier Buffer
    constexpr uint32_t NUM_ITERATIONS = 100;
    // Size: 8 bytes per iteration (uint64_t latency) + 32 bytes padding
    // First address is reused for credit/barrier synchronization
    constexpr auto latency_measurement_buffer_size = 8 * NUM_ITERATIONS + 32;
    CoreRangeSet latency_core_range = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    auto shard_params = ShardSpecBuffer(latency_core_range, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    DeviceLocalBufferConfig latency_measurement_buffer_specs = {
        .page_size = latency_measurement_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    auto latency_measurement_buffer = MeshBuffer::create(
        ReplicatedBufferConfig{.size = latency_measurement_buffer_size},
        latency_measurement_buffer_specs,
        mesh_device.get());
    // Write 0 to latency measurement buffer (initializes credit/barrier to 0)
    std::vector<uint32_t> latency_init_data(latency_measurement_buffer_size / sizeof(uint32_t), 0);
    EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), latency_measurement_buffer, latency_init_data, true);
    const uint32_t latency_measurement_address = latency_measurement_buffer->address();
    std::cout << "Latency measurement buffer address: " << latency_measurement_address << std::endl;

    const uint32_t i = 0;
    // Create input buffer using metal-level API (no ttnn dependencies)
    DeviceLocalBufferConfig buffer_config = {
        .page_size = buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    auto input_mesh_buffer =
        MeshBuffer::create(ReplicatedBufferConfig{.size = buffer_size}, buffer_config, mesh_device.get());

    // Initialize buffer with data (arange equivalent: i, i+1, i+2, ..., num_elems+i-1)
    std::vector<uint32_t> host_data(num_elems);
    for (uint32_t j = 0; j < num_elems; j++) {
        host_data[j] = i + j;
    }

    // Write data to device buffer
    EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), input_mesh_buffer, host_data, true);

    // Extract buffer pointer for metal-level operations
    Buffer* input_buffer = input_mesh_buffer->get_reference_buffer();

    // Sender forwards downstream and waits for ack from last receiver
    tt::tt_metal::send_async(
        mesh_device.get(),
        input_buffer,
        tt::DataFormat::UInt32,
        send_socket_0,
        recv_socket_end,
        latency_measurement_address,
        NUM_ITERATIONS,
        enable_correctness_check);
    tt::tt_metal::socket_forward(
        mesh_device.get(),
        recv_socket_1,
        send_socket_1,
        num_elems * sizeof(uint32_t),
        latency_measurement_address,
        NUM_ITERATIONS);
    tt::tt_metal::socket_forward(
        mesh_device.get(),
        recv_socket_2,
        send_socket_2,
        num_elems * sizeof(uint32_t),
        latency_measurement_address,
        NUM_ITERATIONS);
    tt::tt_metal::socket_forward(
        mesh_device.get(),
        recv_socket_3,
        send_socket_3,
        num_elems * sizeof(uint32_t),
        latency_measurement_address,
        NUM_ITERATIONS);
    Synchronize(mesh_device.get(), std::nullopt);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto start_device_id = mesh_device->get_device(start_device_coord)->id();
    auto start_core_coord = mesh_device->worker_core_from_logical_core(sender_logical_coord);
    std::vector<uint64_t> latencies = std::vector<uint64_t>(NUM_ITERATIONS, 0);
    cluster.read_core(
        latencies.data(),
        sizeof(uint64_t) * NUM_ITERATIONS,
        tt_cxy_pair(start_device_id, start_core_coord),
        latency_measurement_address);
    double avg_latency_cycles = 0.0;
    for (uint32_t i = 0; i < NUM_ITERATIONS; i++) {
        avg_latency_cycles += static_cast<double>(latencies[i]);
    }
    avg_latency_cycles /= NUM_ITERATIONS;
    double avg_latency_us = (avg_latency_cycles / (1.35e9)) * 1e6;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average latency in cycles: " << avg_latency_cycles << std::endl;
    std::cout << "Average latency in microseconds: " << avg_latency_us << std::endl;
}

TEST_F(FabricSendRecvMigratedFixture, SRTestMigrated) {
    run_sr_test_migrated(get_mesh_device(), /*enable_correctness_check=*/false);
}

TEST_F(FabricSendRecvMigratedFixture, SRTestMigratedWithCorrectnessCheck) {
    run_sr_test_migrated(get_mesh_device(), /*enable_correctness_check=*/true);
}

}  // namespace tt::tt_metal
