// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_send_recv.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_forward.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <iomanip>

namespace tt::tt_metal {

// Test using migrated code (metal-level APIs) from multiprocess utils
// This test mirrors the SRTest from test_send_recv_ops.cpp in the ccl directory
// but uses MeshBuffer and metal-level send_async/socket_forward
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
    std::cout << "Intermed Device 3 ID: " << mesh_device->get_device(intermed_device_coord_3)->id() << std::endl;
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
    // Create input buffer using metal-level API
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
