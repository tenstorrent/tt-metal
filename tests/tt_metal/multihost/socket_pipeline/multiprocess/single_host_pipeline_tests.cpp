// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include <chrono>
#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_send_recv.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_forward.hpp"
#include "tt_metal/multihost/socket_pipeline/multiprocess/utils/mesh_socket_rate.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

// Single-host loopback pipeline test using metal-level APIs (send_async/socket_forward).
// Uses GenericMeshDeviceFabric2DFixture to open all devices (fabric requires all devices active).
class SingleHostLoopbackPipelineFixture : public GenericMeshDeviceFabric2DFixture {};

void run_single_host_loopback_pipeline(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, bool enable_correctness_check) {
    using namespace tt::tt_metal::distributed;

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);
    auto copy_logical_coord = CoreCoord(0, 0);

    constexpr uint32_t XFER_SIZE = 14 * 1024;
    // When using sender_reader kernel, pass num_loop_iterations as compile-time arg index 2 (e.g. 100000).
    auto socket_fifo_size = XFER_SIZE * 16;

    auto start_device_coord = MeshCoordinate(0, 0);
    auto intermed_device_coord = MeshCoordinate(1, 0);
    auto intermed_device_coord_2 = MeshCoordinate(1, 1);
    auto intermed_device_coord_3 = MeshCoordinate(0, 1);
    auto end_device_coord = MeshCoordinate(0, 0);

    log_info(tt::LogTest, "Start Device ID: {}", mesh_device->get_device(start_device_coord)->id());
    log_info(tt::LogTest, "Intermed Device ID: {}", mesh_device->get_device(intermed_device_coord)->id());
    log_info(tt::LogTest, "Intermed Device 2 ID: {}", mesh_device->get_device(intermed_device_coord_2)->id());
    log_info(tt::LogTest, "Intermed Device 3 ID: {}", mesh_device->get_device(intermed_device_coord_3)->id());
    log_info(tt::LogTest, "End Device ID: {}", mesh_device->get_device(end_device_coord)->id());

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

    constexpr uint32_t num_elems = XFER_SIZE / sizeof(uint32_t);
    constexpr uint32_t buffer_size = XFER_SIZE;

    auto [send_socket_0, recv_socket_1] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_01);

    auto [send_socket_1, recv_socket_2] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_12);

    auto [send_socket_2, recv_socket_3] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_23);

    auto [send_socket_3, recv_socket_end] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_34);

    // Create Barrier Buffer
    constexpr uint32_t NUM_ITERATIONS = 100;
    // Size: 8 bytes per iteration (uint64_t latency) + 32 bytes padding
    // First address is reused for credit/barrier synchronization
    constexpr auto latency_measurement_buffer_size = (8 * NUM_ITERATIONS) + 32;
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
    log_info(tt::LogTest, "Latency measurement buffer address: {}", latency_measurement_address);

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
        mesh_device.get(), recv_socket_2, send_socket_2, XFER_SIZE, latency_measurement_address, NUM_ITERATIONS);
    tt::tt_metal::socket_forward(
        mesh_device.get(), recv_socket_3, send_socket_3, XFER_SIZE, latency_measurement_address, NUM_ITERATIONS);
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
    // Skip first iteration (often an outlier due to cold start), match multi_host_pipeline
    constexpr uint32_t LATENCY_ITERATIONS_FOR_AVG = NUM_ITERATIONS > 1 ? NUM_ITERATIONS - 1 : 1;
    double avg_latency_cycles = 0.0;
    for (uint32_t i = 1; i < NUM_ITERATIONS; i++) {
        avg_latency_cycles += static_cast<double>(latencies[i]);
    }
    avg_latency_cycles /= LATENCY_ITERATIONS_FOR_AVG;
    double freq_mhz = static_cast<double>(cluster.get_device_aiclk(start_device_id));
    double avg_latency_us = (avg_latency_cycles / (freq_mhz * 1e6)) * 1e6;
    log_info(tt::LogTest, "Average latency in cycles: {:.2f}", avg_latency_cycles);
    log_info(tt::LogTest, "Average latency in microseconds: {:.2f}", avg_latency_us);
}

TEST_F(SingleHostLoopbackPipelineFixture, SingleHostLoopbackPipeline) {
    run_single_host_loopback_pipeline(get_mesh_device(), /*enable_correctness_check=*/false);
}

TEST_F(SingleHostLoopbackPipelineFixture, SingleHostLoopbackPipelineWithCorrectnessCheck) {
    run_single_host_loopback_pipeline(get_mesh_device(), /*enable_correctness_check=*/true);
}

// ─── Rate (throughput) pipeline test ─────────────────────────────────────────
// Linear pipeline (no loopback): sender -> fwd -> fwd -> receiver
// Measures sustained pipeline throughput by pushing data one-way for many iterations.

void run_single_host_rate_pipeline(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t num_iterations,
    bool enable_correctness_check) {
    using namespace tt::tt_metal::distributed;

    auto logical_coord = CoreCoord(0, 0);

    constexpr uint32_t XFER_SIZE = 14 * 1024;
    auto socket_fifo_size = XFER_SIZE * 16;

    // Linear pipeline: (0,0) -> (1,0) -> (1,1) -> (0,1)
    auto sender_device_coord = MeshCoordinate(0, 0);
    auto fwd_device_coord_1 = MeshCoordinate(1, 0);
    auto fwd_device_coord_2 = MeshCoordinate(1, 1);
    auto recv_device_coord = MeshCoordinate(0, 1);

    log_info(tt::LogTest, "Sender Device ID: {}", mesh_device->get_device(sender_device_coord)->id());
    log_info(tt::LogTest, "Fwd 1 Device ID: {}", mesh_device->get_device(fwd_device_coord_1)->id());
    log_info(tt::LogTest, "Fwd 2 Device ID: {}", mesh_device->get_device(fwd_device_coord_2)->id());
    log_info(tt::LogTest, "Recv Device ID: {}", mesh_device->get_device(recv_device_coord)->id());

    // Socket connections for linear pipeline
    SocketConnection socket_connection_01 = SocketConnection(
        MeshCoreCoord(sender_device_coord, logical_coord), MeshCoreCoord(fwd_device_coord_1, logical_coord));
    SocketConnection socket_connection_12 = SocketConnection(
        MeshCoreCoord(fwd_device_coord_1, logical_coord), MeshCoreCoord(fwd_device_coord_2, logical_coord));
    SocketConnection socket_connection_23 = SocketConnection(
        MeshCoreCoord(fwd_device_coord_2, logical_coord), MeshCoreCoord(recv_device_coord, logical_coord));

    SocketMemoryConfig socket_mem_config = SocketMemoryConfig(BufferType::L1, socket_fifo_size);

    SocketConfig socket_config_01 = SocketConfig({socket_connection_01}, socket_mem_config);
    SocketConfig socket_config_12 = SocketConfig({socket_connection_12}, socket_mem_config);
    SocketConfig socket_config_23 = SocketConfig({socket_connection_23}, socket_mem_config);

    constexpr uint32_t num_elems = XFER_SIZE / sizeof(uint32_t);
    constexpr uint32_t buffer_size = XFER_SIZE;

    auto [send_socket_0, recv_socket_1] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_01);
    auto [send_socket_1, recv_socket_2] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_12);
    auto [send_socket_2, recv_socket_end] = MeshSocket::create_socket_pair(mesh_device, mesh_device, socket_config_23);

    // Create input buffer
    DeviceLocalBufferConfig buffer_config = {
        .page_size = buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(std::nullopt, TensorMemoryLayout::INTERLEAVED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    auto input_mesh_buffer =
        MeshBuffer::create(ReplicatedBufferConfig{.size = buffer_size}, buffer_config, mesh_device.get());

    // Initialize buffer with data
    std::vector<uint32_t> host_data(num_elems);
    for (uint32_t j = 0; j < num_elems; j++) {
        host_data[j] = j;
    }
    EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), input_mesh_buffer, host_data, true);
    Buffer* input_buffer = input_mesh_buffer->get_reference_buffer();

    // Host-side timing: record start time just before launching kernels
    auto start_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::high_resolution_clock::now().time_since_epoch())
                          .count();

    // Launch rate-mode kernels: sender -> fwd -> fwd -> receiver
    tt::tt_metal::send_async_rate(
        mesh_device.get(), input_buffer, tt::DataFormat::UInt32, send_socket_0, num_iterations);
    tt::tt_metal::socket_forward_rate(
        mesh_device.get(), recv_socket_1, send_socket_1, num_elems * sizeof(uint32_t), num_iterations);
    tt::tt_metal::socket_forward_rate(mesh_device.get(), recv_socket_2, send_socket_2, XFER_SIZE, num_iterations);
    tt::tt_metal::recv_async_rate(
        mesh_device.get(), recv_socket_end, XFER_SIZE, num_iterations, enable_correctness_check);
    Synchronize(mesh_device.get(), std::nullopt);

    // Host-side timing: record end time after all kernels complete
    auto end_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch())
                        .count();

    // Compute throughput from host-side wall-clock time
    double elapsed_us = static_cast<double>(end_time - start_time);
    double total_bytes = static_cast<double>(num_iterations) * XFER_SIZE;
    double rate_gbps = (total_bytes * 8.0) / (elapsed_us * 1e3);

    log_info(tt::LogTest, "Pipeline rate test: {} iterations, {} bytes/iter", num_iterations, XFER_SIZE);
    log_info(
        tt::LogTest,
        "Host-side elapsed: {:.2f} us, total bytes: {:.2f} MB, {:.4f} Gbps ({:.2f} Mbps)",
        elapsed_us,
        total_bytes / (1024.0 * 1024.0),
        rate_gbps,
        rate_gbps * 1e3);
}

TEST_F(SingleHostLoopbackPipelineFixture, SingleHostRatePipeline) {
    constexpr uint32_t NUM_ITERATIONS = 100000;
    run_single_host_rate_pipeline(get_mesh_device(), NUM_ITERATIONS, /*enable_correctness_check=*/false);
}

TEST_F(SingleHostLoopbackPipelineFixture, SingleHostRatePipelineWithCorrectnessCheck) {
    constexpr uint32_t NUM_ITERATIONS = 100;
    run_single_host_rate_pipeline(get_mesh_device(), NUM_ITERATIONS, /*enable_correctness_check=*/true);
}

}  // namespace tt::tt_metal
