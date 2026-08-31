// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <internal/cluster_noc_helpers.hpp>
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
#include <umd/device/io_window/io_window.hpp>
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

// Read: consume the stream with read() and verify the data. Discard: drop pages with
// discard_pending_pages() without touching the data.
enum class D2HConsumeMode { Read, Discard };

void test_d2h_socket(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    const MeshCoreCoord& sender_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)},
    uint32_t pages_per_read = 1,
    D2HConsumeMode consume_mode = D2HConsumeMode::Read) {
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

    uint32_t num_pages = data_size / page_size;
    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), sender_data_buffer, src_vec, sender_core.device_coord);

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(sender_core.device_coord);
    mesh_workload.add_program(devices, std::move(send_program));

    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

    if (consume_mode == D2HConsumeMode::Discard) {
        // Drop every page without reading the data region. discard_pending_pages() must
        // advance bytes_acked exactly as read()/pop_bytes does, including the FIFO wrap
        // gap, otherwise host bytes_acked never reconciles with device bytes_sent and the
        // barrier deadlocks.
        uint32_t discarded = 0;
        while (discarded < num_pages) {
            discarded += output_socket.discard_pending_pages();
        }
        output_socket.barrier(/*timeout_ms=*/5000);
        return;
    }

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    for (uint32_t page = 0; page < num_pages; page += pages_per_read) {
        uint32_t pages = std::min<uint32_t>(pages_per_read, num_pages - page);
        output_socket.read(dst_vec.data() + (page * page_size_words), pages);
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

void test_hd_socket_multithreaded_loopback(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size,
    H2DMode h2d_mode,
    uint32_t num_iterations = 10,
    const MeshCoreCoord& socket_core = {MeshCoordinate(0, 0), CoreCoord(0, 0)}) {
    auto input_socket = H2DSocket(mesh_device, socket_core, BufferType::L1, socket_fifo_size, h2d_mode);
    auto output_socket = D2HSocket(mesh_device, socket_core, socket_fifo_size);

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
    std::vector<uint32_t> src_vec(data_size * num_iterations / sizeof(uint32_t));
    std::vector<uint32_t> dst_vec(data_size * num_iterations / sizeof(uint32_t));
    std::iota(src_vec.begin(), src_vec.end(), 0);

    auto mesh_workload = MeshWorkload();
    MeshCoordinateRange devices = MeshCoordinateRange(socket_core.device_coord);
    mesh_workload.add_program(devices, std::move(send_program));

    // Launch Loopback Kernel on device (copies data from H2D to D2H socket).
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

    // Set Required Page Size for Socket.
    input_socket.set_page_size(page_size);
    output_socket.set_page_size(page_size);

    uint32_t page_size_words = page_size / sizeof(uint32_t);
    uint32_t data_size_words = data_size / sizeof(uint32_t);

    // Socket Read/Write done over different threads.
    std::thread write_thread([&]() {
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                input_socket.write(src_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });

    std::thread read_thread([&]() {
        for (uint32_t i = 0; i < num_iterations; i++) {
            for (uint32_t j = 0; j < num_txns; j++) {
                output_socket.read(dst_vec.data() + (i * data_size_words) + (j * page_size_words), 1);
            }
        }
    });
    // Barrier with a timeout in the main thread ensure that the read/write threads are not hung.
    input_socket.barrier(10000);
    output_socket.barrier(10000);

    write_thread.join();
    read_thread.join();

    EXPECT_EQ(src_vec, dst_vec);
}

bool is_device_coord_mmio_mapped(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, const MeshCoordinate& device_coord) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto device_id = mesh_device->get_device(device_coord)->id();
    return cluster.get_associated_mmio_device(device_id) == device_id;
}

using HDSocketFixture = MeshDevice1x2Fixture;
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
        // Multi-page read whose span straddles the FIFO wrap boundary, exercising the head/tail split.
        test_d2h_socket(mesh_device_, 4096, 1088, 79424, MeshCoreCoord(sender_coord, CoreCoord(0, 1)), 2);
        // Drain a wrapping stream via discard_pending_pages() instead of reading. 4 pages
        // through a 3-page FIFO forces one wrap; the discard path must credit the wrap gap
        // or the closing barrier never reconciles with the device.
        test_d2h_socket(
            mesh_device_, 4096, 1088, 4352, MeshCoreCoord(sender_coord, CoreCoord(0, 1)), 1, D2HConsumeMode::Discard);
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
        for (const auto& socket_coord : MeshCoordinateRange(mesh_device_->shape())) {
            if (!is_device_coord_mmio_mapped(mesh_device_, socket_coord)) {
                continue;
            }
            // No wrap
            test_hd_socket_multithreaded_loopback(
                mesh_device_, 1024, 64, 1024, h2d_mode, 100, MeshCoreCoord(socket_coord, CoreCoord(0, 0)));
            // Even wrap
            test_hd_socket_multithreaded_loopback(
                mesh_device_, 1024, 64, 32768, h2d_mode, 100, MeshCoreCoord(socket_coord, CoreCoord(1, 1)));
            // Uneven wrap
            test_hd_socket_multithreaded_loopback(
                mesh_device_, 4096, 1088, 78336, h2d_mode, 100, MeshCoreCoord(socket_coord, CoreCoord(0, 1)));
            // Uneven wrap with multiple pages on host allocated.
            // On most hosts, page size is 4K, so this should lead to 5 pages being allocated on the host.
            test_hd_socket_multithreaded_loopback(
                mesh_device_, 16512, 1088, 156672, h2d_mode, 100, MeshCoreCoord(socket_coord, CoreCoord(0, 1)));
        }
    }
}

// L2CPU socket support: the DRAM bank table, the L2CPU static-TLB registration, and
// the L2CPU constructors' argument validation. These do not complete construction of
// an L2CPU socket, which would write to LIM; a partial-line write to LIM whose ECC has
// not been initialised can fault, and initialising it requires running code on the
// L2CPU itself. A single device is sufficient throughout.
using L2CpuSocketFixture = MeshDevice1x1Fixture;

namespace {

// TRANSLATED NOC coords of the unharvested L2CPU tiles on this device, or
// empty when the architecture has none.
std::vector<tt::umd::CoreCoord> get_l2cpu_cores(ChipId device_id) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
    return soc_desc.get_cores(tt::CoreType::L2CPU, tt::CoordSystem::TRANSLATED);
}

bool is_blackhole() { return MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE; }

}  // namespace

// Assert the bank table against the same allocator and soc-descriptor sources that
// RiscFirmwareInitializer programs BRISC from.
TEST_F(L2CpuSocketFixture, DramBankTableMatchesAllocatorAndSocDescriptor) {
    auto* device = mesh_device_->get_device(MeshCoordinate(0, 0));
    const ChipId device_id = device->id();

    const auto table = internal::get_dram_bank_table(device_id);

    const auto& allocator = *device->allocator();
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
    const uint32_t num_banks = allocator.get_num_banks(BufferType::DRAM);

    ASSERT_EQ(table.size(), num_banks) << "one entry per logical DRAM bank";

    for (uint32_t bank_id = 0; bank_id < num_banks; ++bank_id) {
        const auto& entry = table[bank_id];
        EXPECT_EQ(entry.bank_id, bank_id) << "table must be indexed by bank_id";
        EXPECT_EQ(
            entry.base_addr,
            static_cast<uint64_t>(static_cast<int64_t>(allocator.get_bank_offset(BufferType::DRAM, bank_id))))
            << "bank " << bank_id << " base_addr must equal the allocator's bank offset";
        EXPECT_EQ(entry.bank_size, soc_desc.dram_view_size) << "bank " << bank_id;

        // On virtualized-DRAM architectures the table reports the TRANSLATED coord
        // verbatim, so it must equal the soc descriptor's DRAM view exactly. Other
        // architectures route through hal.noc_coordinate().
        const CoreCoord preferred =
            soc_desc.get_preferred_worker_core_for_dram_view(static_cast<int>(bank_id), /*noc=*/0);
        if (is_blackhole()) {
            EXPECT_EQ(entry.noc_x, static_cast<uint32_t>(preferred.x))
                << "bank " << bank_id << " noc_x: table says " << entry.noc_x << ", DRAM view is at " << preferred.x;
            EXPECT_EQ(entry.noc_y, static_cast<uint32_t>(preferred.y))
                << "bank " << bank_id << " noc_y: table says " << entry.noc_y << ", DRAM view is at " << preferred.y;
        }
    }
}

// H2D/D2H socket setup maps an L2CPU tile with a window anchored at the LIM base. Every tile must
// be mappable that way, and the window must cover the LIM aperture the sockets address through it.
TEST_F(L2CpuSocketFixture, L2CpuIoWindowsAnchoredAtLimBase) {
    if (!is_blackhole()) {
        GTEST_SKIP() << "L2CPU tiles only exist on Blackhole";
    }
    auto* device = mesh_device_->get_device(MeshCoordinate(0, 0));
    const ChipId device_id = device->id();

    const auto l2cpu_cores = get_l2cpu_cores(device_id);
    if (l2cpu_cores.empty()) {
        GTEST_SKIP() << "No unharvested L2CPU tiles on this device";
    }

    // Spelled out here rather than taken from ll_api so the test checks the aperture the sockets
    // are built around, not just that it agrees with itself.
    constexpr uint64_t kL2CpuLimBase = 0x08000000ULL;
    constexpr uint64_t kL2CpuLimSize = 2ULL * 1024 * 1024;
    auto& cluster = MetalContext::instance().get_cluster();

    for (const auto& core : l2cpu_cores) {
        std::unique_ptr<tt::umd::IoWindow> window =
            cluster.get_driver()->create_io_window(device_id, core, kL2CpuLimBase, {.size = kL2CpuLimSize});
        ASSERT_NE(window, nullptr) << "L2CPU (" << core.x << ", " << core.y
                                   << ") cannot be mapped; H2D/D2H socket setup would throw";
        EXPECT_EQ(window->get_target_config().addr, kL2CpuLimBase)
            << "L2CPU (" << core.x << ", " << core.y << ") window must be anchored at the LIM base";
        // The config buffer and (in HOST_PUSH) the data FIFO are addressed through this window, so
        // the whole LIM aperture has to be reachable from the anchor.
        EXPECT_GE(window->get_size(), kL2CpuLimSize);
    }
}

// The L2CPU constructors take caller-reserved LIM addresses, so argument validation is
// the only guard against a mis-placed socket. All of these throw before any pinned
// memory is allocated or any LIM byte is written.
TEST_F(L2CpuSocketFixture, L2CpuSocketRejectsInvalidLimAddresses) {
    if (!is_blackhole()) {
        GTEST_SKIP() << "L2CPU sockets are Blackhole-only";
    }
    const auto l2cpu_cores = get_l2cpu_cores(mesh_device_->get_device(MeshCoordinate(0, 0))->id());
    if (l2cpu_cores.empty()) {
        GTEST_SKIP() << "No unharvested L2CPU tiles on this device";
    }

    const MeshCoreCoord l2cpu(MeshCoordinate(0, 0), CoreCoord(l2cpu_cores.front().x, l2cpu_cores.front().y));
    const uint32_t pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    constexpr uint32_t kLimBase = 0x08000000;
    constexpr uint32_t kFifoSize = 4096;
    const uint32_t config_addr = kLimBase;
    const uint32_t data_addr = kLimBase + 0x10000;

    // Zero addresses are never valid -- 0 is not in LIM at all.
    EXPECT_ANY_THROW(
        H2DSocket(*mesh_device_, l2cpu, kFifoSize, /*config_buffer_address=*/0, data_addr, H2DMode::HOST_PUSH));
    EXPECT_ANY_THROW(
        H2DSocket(*mesh_device_, l2cpu, kFifoSize, config_addr, /*data_fifo_address=*/0, H2DMode::HOST_PUSH));
    EXPECT_ANY_THROW(D2HSocket(*mesh_device_, l2cpu, kFifoSize, /*config_buffer_address=*/0));

    // Misaligned addresses would corrupt the wire structs.
    EXPECT_ANY_THROW(H2DSocket(*mesh_device_, l2cpu, kFifoSize, config_addr + 1, data_addr, H2DMode::HOST_PUSH));
    EXPECT_ANY_THROW(H2DSocket(*mesh_device_, l2cpu, kFifoSize, config_addr, data_addr + 1, H2DMode::HOST_PUSH));

    // A non-PCIe-aligned or zero FIFO breaks the ring arithmetic.
    EXPECT_ANY_THROW(H2DSocket(*mesh_device_, l2cpu, /*fifo_size=*/0, config_addr, data_addr, H2DMode::HOST_PUSH));
    EXPECT_ANY_THROW(H2DSocket(*mesh_device_, l2cpu, pcie_alignment + 1, config_addr, data_addr, H2DMode::HOST_PUSH));

    // In HOST_PUSH the ring lives in LIM and is reached through the LIM window,
    // so a ring running past the window end must be rejected at construction.
    EXPECT_ANY_THROW(H2DSocket(
        *mesh_device_,
        l2cpu,
        /*fifo_size=*/0x200000,
        /*config_buffer_address=*/kLimBase + 0x100000,
        kLimBase + 0x180000,
        H2DMode::HOST_PUSH));
    // A ring below the LIM window entirely is also invalid in HOST_PUSH.
    EXPECT_ANY_THROW(
        H2DSocket(*mesh_device_, l2cpu, kFifoSize, config_addr, /*data_fifo_address=*/0x1000, H2DMode::HOST_PUSH));
}

}  // namespace tt::tt_metal::distributed
