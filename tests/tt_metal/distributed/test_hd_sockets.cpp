// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

namespace tt::tt_metal::distributed {

void test_h2d_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    H2DMode h2d_mode,
    uint32_t num_iterations = 10,
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

    auto recv_core_virtual = mesh_device->worker_core_from_logical_core(recv_core.core_coord);
    uint32_t page_size_words = page_size / sizeof(uint32_t);

    // Write a single page at a time
    const auto& cluster = MetalContext::instance().get_cluster();
    for (int i = 0; i < num_iterations; i++) {
        std::iota(src_vec.begin(), src_vec.end(), i);
        for (uint32_t j = 0; j < num_writes; j++) {
            input_socket.write(src_vec.data() + (j * page_size_words), 1);
        }
        input_socket.barrier();
        std::vector<uint32_t> recv_data_readback(data_size / sizeof(uint32_t));
        cluster.read_core(
            recv_data_readback.data(),
            data_size,
            tt_cxy_pair(mesh_device->get_device(recv_core.device_coord)->id(), recv_core_virtual),
            recv_data_buffer->address());
        EXPECT_EQ(src_vec, recv_data_readback);
    }
}

void test_d2h_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
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
            }});

    uint32_t num_reads = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), sender_data_buffer, src_vec, sender_core.device_coord);

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(sender_core.device_coord);
    mesh_workload.add_program(devices, std::move(send_program));

    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_reads; i++) {
        output_socket.read(dst_vec.data() + (i * page_size_words), 1);
    }
    output_socket.barrier();
    EXPECT_EQ(src_vec, dst_vec);
}

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

    // DEVICE_PULL landing slot (CT arg 6): the H2D FIFO lives in pinned host memory, so the
    // loopback kernel needs a page of local L1 to pull into before writing back to the D2H socket.
    const ReplicatedBufferConfig scratch_buffer_config{.size = page_size};
    auto scratch_shard_params =
        ShardSpecBuffer(CoreRangeSet(socket_core.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig scratch_device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(scratch_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto scratch_buffer = MeshBuffer::create(scratch_buffer_config, scratch_device_local_config, mesh_device.get());

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
                static_cast<uint32_t>(scratch_buffer->address()),
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
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, const MeshCoordinate& device_coord);

void test_hd_socket_multithreaded_loopback(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    H2DMode h2d_mode,
    uint32_t num_iterations = 10,
    const CoreCoord& socket_core_coord = CoreCoord(0, 0)) {
    TT_FATAL(data_size % page_size == 0, "Data size must be a multiple of page size");

    // Spawn an H2D and D2H socket targeting every device coordinate in the mesh.
    std::vector<MeshCoordinate> socket_device_coords;
    std::vector<std::unique_ptr<H2DSocket>> input_sockets;
    std::vector<std::unique_ptr<D2HSocket>> output_sockets;
    for (const auto& device_coord : MeshCoordinateRange(mesh_device->shape())) {
        if (!is_device_coord_mmio_mapped(mesh_device, device_coord)) {
            continue;
        }
        MeshCoreCoord socket_core(device_coord, socket_core_coord);
        socket_device_coords.push_back(device_coord);
        input_sockets.push_back(
            std::make_unique<H2DSocket>(mesh_device, socket_core, BufferType::L1, socket_fifo_size, h2d_mode));
        output_sockets.push_back(std::make_unique<D2HSocket>(mesh_device, socket_core, socket_fifo_size));
    }

    // DEVICE_PULL landing slot (CT arg 6): the H2D FIFO lives in pinned host memory, so the
    // loopback kernel needs a page of local L1 to pull into before writing back to the D2H socket.
    // A single replicated buffer (same L1 address on every device) serves every per-coord program.
    const ReplicatedBufferConfig scratch_buffer_config{.size = page_size};
    auto scratch_shard_params =
        ShardSpecBuffer(CoreRangeSet(socket_core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const DeviceLocalBufferConfig scratch_device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(scratch_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
    };
    auto scratch_buffer = MeshBuffer::create(scratch_buffer_config, scratch_device_local_config, mesh_device.get());

    // Build a single MeshWorkload that spans the entire device grid: one loopback program per
    // socketed device, each compiled with that device's H2D/D2H socket config addresses.
    auto mesh_workload = MeshWorkload();
    for (size_t i = 0; i < input_sockets.size(); i++) {
        auto send_program = CreateProgram();
        CreateKernel(
            send_program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_loopback.cpp",
            socket_core_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {
                    static_cast<uint32_t>(input_sockets[i]->get_config_buffer_address()),
                    static_cast<uint32_t>(output_sockets[i]->get_config_buffer_address()),
                    static_cast<uint32_t>(page_size),
                    static_cast<uint32_t>(data_size),
                    static_cast<uint32_t>(num_iterations),
                    h2d_mode == H2DMode::DEVICE_PULL,
                    static_cast<uint32_t>(scratch_buffer->address()),
                }});
        mesh_workload.add_program(MeshCoordinateRange(socket_device_coords[i]), std::move(send_program));
    }

    // Launch Loopback Kernels on all devices (each copies data from its H2D to its D2H socket).
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

    uint32_t num_txns = data_size / page_size;
    uint32_t page_size_words = page_size / sizeof(uint32_t);
    uint32_t data_size_words = data_size / sizeof(uint32_t);

    // Per-socket source and destination buffers so concurrent threads don't race on shared state.
    std::vector<std::vector<uint32_t>> src_vecs(input_sockets.size());
    std::vector<std::vector<uint32_t>> dst_vecs(input_sockets.size());
    for (size_t s = 0; s < input_sockets.size(); s++) {
        src_vecs[s].resize(data_size * num_iterations / sizeof(uint32_t));
        dst_vecs[s].resize(data_size * num_iterations / sizeof(uint32_t));
        std::iota(src_vecs[s].begin(), src_vecs[s].end(), static_cast<uint32_t>(s));
        // Set Required Page Size for Sockets.
        input_sockets[s]->set_page_size(page_size);
        output_sockets[s]->set_page_size(page_size);
    }

    // Spawn a writer and reader thread per socket and let them all run concurrently.
    std::vector<std::thread> write_threads;
    std::vector<std::thread> read_threads;
    write_threads.reserve(input_sockets.size());
    read_threads.reserve(output_sockets.size());
    for (size_t s = 0; s < input_sockets.size(); s++) {
        auto& input_socket = *input_sockets[s];
        auto& output_socket = *output_sockets[s];
        auto& src_vec = src_vecs[s];
        auto& dst_vec = dst_vecs[s];
        write_threads.emplace_back([&, num_iterations, num_txns, data_size_words, page_size_words]() {
            for (uint32_t i = 0; i < num_iterations; i++) {
                for (uint32_t j = 0; j < num_txns; j++) {
                    input_socket.write(src_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
                }
            }
        });
        read_threads.emplace_back([&, num_iterations, num_txns, data_size_words, page_size_words]() {
            for (uint32_t i = 0; i < num_iterations; i++) {
                for (uint32_t j = 0; j < num_txns; j++) {
                    output_socket.read(dst_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
                }
            }
        });
    }

    // Barrier with a timeout in the main thread ensures that the read/write threads are not hung.
    for (size_t s = 0; s < input_sockets.size(); s++) {
        input_sockets[s]->barrier(10000);
        output_sockets[s]->barrier(10000);
    }

    for (auto& t : write_threads) {
        t.join();
    }
    for (auto& t : read_threads) {
        t.join();
    }

    for (size_t s = 0; s < input_sockets.size(); s++) {
        EXPECT_EQ(src_vecs[s], dst_vecs[s]);
    }
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

    for (auto h2d_mode : {H2DMode::HOST_PUSH, H2DMode::DEVICE_PULL}) {
        for (const auto& recv_coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (!is_device_coord_mmio_mapped(mesh_device_, recv_coord)) {
                continue;
            }
            // No wrap
            test_h2d_socket(mesh_device_, 1024, 64, 1024, h2d_mode, 50, MeshCoreCoord(recv_coord, CoreCoord(0, 0)));
            // Even wrap
            test_h2d_socket(mesh_device_, 1024, 64, 32768, h2d_mode, 50, MeshCoreCoord(recv_coord, CoreCoord(1, 1)));
            // Uneven wrap
            test_h2d_socket(mesh_device_, 4096, 1088, 78336, h2d_mode, 50, MeshCoreCoord(recv_coord, CoreCoord(0, 1)));
            // Uneven wrap with multiple pages on host allocated.
            // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
            test_h2d_socket(
                mesh_device_, 16512, 1088, 156672, h2d_mode, 50, MeshCoreCoord(recv_coord, CoreCoord(0, 1)));
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
        test_d2h_socket(mesh_device_, 1024, 64, 1024, MeshCoreCoord(sender_coord, CoreCoord(0, 0)));
        // Even wrap
        test_d2h_socket(mesh_device_, 1024, 64, 32768, MeshCoreCoord(sender_coord, CoreCoord(1, 1)));
        // Uneven wrap
        test_d2h_socket(mesh_device_, 4096, 1088, 78336, MeshCoreCoord(sender_coord, CoreCoord(0, 1)));
        // Uneven wrap with multiple pages on host allocated.
        // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
        test_d2h_socket(mesh_device_, 16512, 1088, 156672, MeshCoreCoord(sender_coord, CoreCoord(0, 1)));
    }
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

TEST_F(HDSocketFixture, H2DSocketLoopbackMultiThreadedStress) {
    // Skip if mapping to NOC isn't supported on this system
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Mapping host memory to NOC is not supported on this system";
    }

    for (auto h2d_mode : {H2DMode::DEVICE_PULL, H2DMode::HOST_PUSH}) {
        // No wrap
        test_hd_socket_multithreaded_loopback(mesh_device_, 1024, 64, 1024, h2d_mode, 100, CoreCoord(0, 0));
        // Even wrap
        test_hd_socket_multithreaded_loopback(mesh_device_, 1024, 64, 32768, h2d_mode, 100, CoreCoord(1, 1));
        // Uneven wrap
        test_hd_socket_multithreaded_loopback(mesh_device_, 4096, 1088, 78336, h2d_mode, 100, CoreCoord(0, 1));
        // Uneven wrap with multiple pages on host allocated.
        // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
        test_hd_socket_multithreaded_loopback(mesh_device_, 16512, 1088, 156672, h2d_mode, 100, CoreCoord(0, 1));
    }
}

}  // namespace tt::tt_metal::distributed
