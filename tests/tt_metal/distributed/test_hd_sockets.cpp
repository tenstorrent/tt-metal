// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/work_split.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include "gmock/gmock.h"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/mesh_socket_serialization.hpp"
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <cstring>
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
namespace tt::tt_metal::distributed {

void test_h2d_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    H2DMode h2d_mode,
    uint32_t num_iterations = 100,
    const MeshCoreCoord& recv_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)}) {
    auto input_socket = H2DSocket(mesh_device, recv_core, BufferType::L1, socket_fifo_size, h2d_mode);
    input_socket.set_page_size(page_size);

    TT_FATAL(data_size % page_size == 0, "Data size must be a multiple of page size");

    // Create recv data buffer to drain data into
    const ReplicatedBufferConfig buffer_config{.size = data_size};
    auto recv_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(recv_core.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig recv_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, mesh_device.get());
    // Create Recv MeshWorkload
    auto recv_program = CreateProgram();
    CreateKernel(
        recv_program,
        h2d_mode == H2DMode::DEVICE_PULL ? "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_receiver.cpp"
                                         : "tests/tt_metal/tt_metal/test_kernels/misc/socket/receiver_worker.cpp",
        recv_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket.get_config_buffer_address()),
                static_cast<uint32_t>(recv_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
            }});

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(recv_core.device_coord);
    mesh_workload.add_program(devices, std::move(recv_program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

    uint32_t num_writes = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    // std::cout << "Page size: " << page_size << std::endl;
    // std::vector<std::chrono::nanoseconds> write_times = std::vector<std::chrono::nanoseconds>(num_writes);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_writes; j++) {
            input_socket.write(src_vec.data() + (j * page_size_words), 1);
        }
    }
    input_socket.barrier();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "Write time: " << duration.count() << "ns for: " << recv_core.device_coord << " "
              << num_writes * num_iterations << " writes" << std::endl;
    // std::cout << "Average write time: " << duration.count() << "ns for: "  << recv_core.device_coord << " " <<
    // num_writes << " writes" << std::endl; for (uint32_t j = 0; j < num_writes; j++) {
    //     auto start_time = std::chrono::high_resolution_clock::now();
    //     input_socket.write(src_vec.data() + (j * page_size_words), 1);
    //     input_socket.barrier();
    //     auto duration =
    //     std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_time);
    //     write_times[j] = duration;
    // }

    // std::cout << "Average write time: " << (std::accumulate(write_times.begin(), write_times.end(),
    // std::chrono::nanoseconds(0)) / num_writes).count() << "ns for: "  << recv_core.device_coord << " " <<  num_writes
    // << " writes" << std::endl;
}

void test_d2h_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations = 100,
    const MeshCoreCoord& sender_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)}) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    TT_FATAL(data_size % page_size == 0, "Data size must be a multiple of page size");

    const ReplicatedBufferConfig buffer_config{.size = data_size};
    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_core.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };

    uint32_t num_txns = data_size / page_size;
    uint32_t measurement_buffer_size = sizeof(uint64_t) * num_iterations;

    const ReplicatedBufferConfig measurement_buffer_config{.size = measurement_buffer_size};
    const DeviceLocalBufferConfig measurement_device_local_config{
        .page_size = measurement_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };

    auto measurement_buffer =
        MeshBuffer::create(measurement_buffer_config, measurement_device_local_config, mesh_device.get());
    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, mesh_device.get());

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_sender.cpp",
        sender_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), sender_data_buffer, src_vec, sender_core.device_coord, true);

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(sender_core.device_coord);
    mesh_workload.add_program(devices, std::move(send_program));

    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_txns; j++) {
            output_socket.read(dst_vec.data() + (j * page_size_words), 1);
        }
    }
    output_socket.barrier();
    EXPECT_EQ(src_vec, dst_vec);

    const auto& cluster = MetalContext::instance().get_cluster();
    std::vector<uint64_t> latency_data(1);
    cluster.read_core(
        latency_data.data(),
        sizeof(uint64_t),
        tt_cxy_pair(
            mesh_device->get_device(sender_core.device_coord)->id(),
            mesh_device->worker_core_from_logical_core(sender_core.core_coord)),
        measurement_buffer->address());

    auto physical_system_descriptor = PhysicalSystemDescriptor(
        MetalContext::instance().get_cluster().get_driver(),
        MetalContext::instance().get_distributed_context_ptr(),
        &MetalContext::instance().hal(),
        MetalContext::instance().rtoptions(),
        true);
    auto fabric_node_id = mesh_device->get_fabric_node_id(sender_core.device_coord);
    auto asic_id = MetalContext::instance().get_control_plane().get_asic_id_from_fabric_node_id(fabric_node_id);
    auto asic_desc = physical_system_descriptor.get_asic_descriptors()[asic_id];

    // Convert cycles to microseconds: Latency [us] = cycles / (1.35 * 10^3)
    uint64_t total_cycles = latency_data[0];
    uint64_t total_transactions = num_txns * num_iterations;
    double avg_cycles_per_transaction = static_cast<double>(total_cycles) / total_transactions;
    double avg_latency_us = avg_cycles_per_transaction / 1350.0;

    std::cout << "Average D2H Round-Trip Latency: " << avg_latency_us << " us"
              << " (cycles: " << avg_cycles_per_transaction << ")"
              << " for: " << sender_core.device_coord << " Tray ID: " << *(asic_desc.tray_id)
              << " ASIC Location: " << *(asic_desc.asic_location) << std::endl;
}

// Forward declaration for benchmark helper
std::pair<double, double> benchmark_d2h_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core);

void test_hd_socket_loopback(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    H2DMode h2d_mode,
    uint32_t num_iterations = 10,
    const MeshCoreCoord& socket_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)}) {
    auto input_socket = H2DSocket(mesh_device, socket_core, BufferType::L1, socket_fifo_size, h2d_mode);
    auto output_socket = D2HSocket(mesh_device, socket_core, socket_fifo_size);

    input_socket.set_page_size(page_size);
    output_socket.set_page_size(page_size);

    TT_FATAL(data_size % page_size == 0, "Data size must be a multiple of page size");

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
        socket_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(input_socket.get_config_buffer_address()),
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(num_iterations),
                h2d_mode == H2DMode::DEVICE_PULL,
            }});

    uint32_t num_txns = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(socket_core.device_coord);
    mesh_workload.add_program(devices, std::move(send_program));

    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_iterations; i++) {
        std::iota(src_vec.begin(), src_vec.end(), i);
        for (uint32_t j = 0; j < num_txns; j++) {
            input_socket.write(src_vec.data() + (j * page_size_words), 1);
            output_socket.read(dst_vec.data() + (j * page_size_words), 1);
        }
    }
    input_socket.barrier();
    output_socket.barrier();
    EXPECT_EQ(src_vec, dst_vec);
}

bool is_device_coord_mmio_mapped(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, const MeshCoordinate& device_coord) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto device_id = mesh_device->get_device(device_coord)->id();
    return cluster.get_associated_mmio_device(device_id) == device_id;
}

using HDSocketFixture = MeshDevice4x8Fixture;
TEST_F(HDSocketFixture, H2DSocket) {
    // Skip if mapping to NOC isn't supported on this system
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    for (auto h2d_mode : {H2DMode::HOST_PUSH}) {
        for (const auto& recv_coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (!is_device_coord_mmio_mapped(mesh_device_, recv_coord)) {
                continue;
            }
            // No wrap
            test_h2d_socket(mesh_device_, 1024, 64, 4096, h2d_mode, 500, MeshCoreCoord(recv_coord, CoreCoord(0, 0)));
            // // Even wrap
            // test_h2d_socket(mesh_device_, 1024, 64, 32768, h2d_mode, 50, MeshCoreCoord(recv_coord, CoreCoord(1, 1)));
            // // Uneven wrap
            // test_h2d_socket(mesh_device_, 4096, 1088, 78336, h2d_mode, 50, MeshCoreCoord(recv_coord, CoreCoord(0,
            // 1)));
            // // Uneven wrap with multiple pages on host allocated.
            // // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
            // test_h2d_socket(
            //     mesh_device_, 16512, 1088, 156672, h2d_mode, 50, MeshCoreCoord(recv_coord, CoreCoord(0, 1)));
        }
    }
}

TEST_F(HDSocketFixture, D2HSocket) {
    // Skip if mapping to NOC isn't supported on this system
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    for (const auto& sender_coord : MeshCoordinateRange(mesh_device_->shape())) {
        if (!is_device_coord_mmio_mapped(mesh_device_, sender_coord)) {
            continue;
        }
        // No wrap
        test_d2h_socket(mesh_device_, 1024, 64, 4096, 500, MeshCoreCoord(sender_coord, CoreCoord(0, 0)));
        // Even wrap
        // test_d2h_socket(mesh_device_, 1024, 64, 32768, MeshCoreCoord(sender_coord, CoreCoord(1, 1)));
        // // Uneven wrap
        // test_d2h_socket(mesh_device_, 4096, 1088, 78336, MeshCoreCoord(sender_coord, CoreCoord(0, 1)));
        // // Uneven wrap with multiple pages on host allocated.
        // // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
        // test_d2h_socket(mesh_device_, 16512, 1088, 156672, MeshCoreCoord(sender_coord, CoreCoord(0, 1)));
    }
}

TEST_F(HDSocketFixture, D2HSocketThroughputBenchmark) {
    // Skip if mapping to NOC isn't supported on this system
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    std::vector<std::size_t> total_data_sizes = {
        16 * 1024,            // 16KB
        32 * 1024,            // 32KB
        512 * 1024,           // 512KB
        1024 * 1024,          // 1MB
        16 * 1024 * 1024,     // 16MB
        512UL * 1024 * 1024,  // 512MB
        1024UL * 1024 * 1024  // 1GB
    };

    std::vector<std::size_t> page_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    std::vector<std::size_t> fifo_sizes = {
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024,
        128 * 1024 * 1024,
        256 * 1024 * 1024,
        512 * 1024 * 1024,
    };

    // Pages sent per kernel iteration. Capped so data_size fits in L1 (~1.46MB bank).
    constexpr std::size_t TARGET_PAGES_PER_ITER = 1;
    constexpr std::size_t L1_DATA_BUDGET = 1400000;

    auto compute_data_size = [&](std::size_t page_size) -> std::size_t {
        std::size_t pages = std::min<std::size_t>(TARGET_PAGES_PER_ITER, L1_DATA_BUDGET / page_size);
        return page_size * std::max<std::size_t>(pages, 1);
    };

    // Find first MMIO-mapped device
    MeshCoordinate sender_coord = mesh_device_->shape();  // invalid sentinel
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        if (is_device_coord_mmio_mapped(mesh_device_, coord)) {
            sender_coord = coord;
            break;
        }
    }
    MeshCoreCoord sender_core = {sender_coord, CoreCoord(0, 0)};

    std::cout << "page_size,socket_fifo_size,total_data,data_size,pages_per_iter,"
              << "num_iterations,total_pages,avg_per_page_us,avg_per_page_cycles,"
              << "throughput_gbps,device_coord" << std::endl;

    for (auto fifo_size : fifo_sizes) {
        for (auto page_size : page_sizes) {
            if (page_size > fifo_size) {
                continue;
            }
            std::size_t data_size = compute_data_size(page_size);
            std::size_t pages_per_iter = data_size / page_size;

            for (auto total_data : total_data_sizes) {
                uint32_t num_iterations = total_data / data_size;
                if (num_iterations == 0) {
                    continue;
                }
                uint64_t total_pages = static_cast<uint64_t>(pages_per_iter) * num_iterations;

                auto [us, cycles] =
                    benchmark_d2h_socket(mesh_device_, fifo_size, page_size, data_size, num_iterations, sender_core);

                double throughput_gbps = static_cast<double>(page_size) / (us * 1e3);

                std::cout << page_size << "," << fifo_size << "," << total_data << "," << data_size << ","
                          << pages_per_iter << "," << num_iterations << "," << total_pages << "," << us << "," << cycles
                          << "," << throughput_gbps << "," << sender_coord << std::endl;
                std::cout.flush();
            }
        }
    }
}

// Helper function for benchmark that returns results without printing extra info
std::pair<double, double> benchmark_d2h_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    uint32_t num_iterations,
    const MeshCoreCoord& sender_core) {
    auto output_socket = D2HSocket(mesh_device, sender_core, socket_fifo_size);
    output_socket.set_page_size(page_size);

    const ReplicatedBufferConfig buffer_config{.size = data_size};
    auto sender_data_shard_params =
        ShardSpecBuffer(CoreRangeSet(sender_core.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    const DeviceLocalBufferConfig sender_device_local_config{
        .page_size = data_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };

    uint32_t num_txns = data_size / page_size;
    uint32_t measurement_buffer_size = sizeof(uint64_t);

    const ReplicatedBufferConfig measurement_buffer_config{.size = measurement_buffer_size};
    const DeviceLocalBufferConfig measurement_device_local_config{
        .page_size = measurement_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };

    auto measurement_buffer =
        MeshBuffer::create(measurement_buffer_config, measurement_device_local_config, mesh_device.get());
    auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, mesh_device.get());

    auto send_program = CreateProgram();
    CreateKernel(
        send_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_sender.cpp",
        sender_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {
                static_cast<uint32_t>(output_socket.get_config_buffer_address()),
                static_cast<uint32_t>(sender_data_buffer->address()),
                static_cast<uint32_t>(page_size),
                static_cast<uint32_t>(data_size),
                static_cast<uint32_t>(measurement_buffer->address()),
                static_cast<uint32_t>(num_iterations),
            }});

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), sender_data_buffer, src_vec, sender_core.device_coord, true);

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(sender_core.device_coord);
    mesh_workload.add_program(devices, std::move(send_program));

    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    for (uint32_t i = 0; i < num_iterations; i++) {
        for (uint32_t j = 0; j < num_txns; j++) {
            output_socket.read(dst_vec.data() + (j * page_size_words), 1);
        }
    }
    output_socket.barrier();

    const auto& cluster = MetalContext::instance().get_cluster();
    std::vector<uint64_t> latency_data(1);
    cluster.read_core(
        latency_data.data(),
        sizeof(uint64_t),
        tt_cxy_pair(
            mesh_device->get_device(sender_core.device_coord)->id(),
            mesh_device->worker_core_from_logical_core(sender_core.core_coord)),
        measurement_buffer->address());

    uint64_t total_cycles = latency_data[0];
    uint64_t total_transactions = num_txns * num_iterations;
    double avg_cycles_per_transaction = static_cast<double>(total_cycles) / total_transactions;
    double avg_latency_us = avg_cycles_per_transaction / 1350.0;

    return {avg_latency_us, avg_cycles_per_transaction};
}

TEST_F(HDSocketFixture, H2DSocketLoopback) {
    // Skip if mapping to NOC isn't supported on this system
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    for (auto h2d_mode : {H2DMode::DEVICE_PULL, H2DMode::HOST_PUSH}) {
        for (const auto& socket_coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (!is_device_coord_mmio_mapped(mesh_device_, socket_coord)) {
                continue;
            }
            std::cout << "Testing H2DSocketLoopback on socket_coord: " << socket_coord << std::endl;
            // No wrap
            test_hd_socket_loopback(
                mesh_device_, 1024, 64, 1024, h2d_mode, 50, MeshCoreCoord(socket_coord, CoreCoord(0, 0)));
            // Even wrap
            test_hd_socket_loopback(
                mesh_device_, 1024, 64, 32768, h2d_mode, 50, MeshCoreCoord(socket_coord, CoreCoord(1, 1)));
            // Uneven wrap
            test_hd_socket_loopback(
                mesh_device_, 4096, 1088, 78336, h2d_mode, 50, MeshCoreCoord(socket_coord, CoreCoord(0, 1)));
            // Uneven wrap with multiple pages on host allocated.
            // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
            test_hd_socket_loopback(
                mesh_device_, 16512, 1088, 156672, h2d_mode, 50, MeshCoreCoord(socket_coord, CoreCoord(0, 1)));
        }
    }
}

}  // namespace tt::tt_metal::distributed
