// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase 1 test: single-core PCIe H2D → DRAM streaming
//
// Validates the pcie_to_dram_puller.cpp kernel: a single Tensix core pulls data
// from host pinned memory through the PCIe link and writes it to DRAM using the
// double-buffered L1 pattern described in pcie_noc_dma_report.txt §17 Phase 1.
//
// This establishes the single-core baseline before the full 140-core grid
// (Phase 3) is deployed.  Expected single-core throughput: ~30–100 MB/s
// (RTT-limited by PCIe round-trip, ~500 ns per 16 KB chunk).
//
// Build & run:
//   ninja -C build distributed_unit_tests
//   ./build/test/tt_metal/distributed/distributed_unit_tests \
//       --gtest_filter="PCIeToDRAM*"
//
// DPRINT (to see kernel checkpoints on core (0,0)):
//   TT_METAL_DPRINT_CORES=0,0 ./distributed_unit_tests --gtest_filter="PCIeToDRAM*"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/sub_device.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_align.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "gmock/gmock.h"

namespace tt::tt_metal::distributed {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// 16 KB = NOC_MAX_BURST_SIZE.  Maximises single-transaction PCIe efficiency.
// Sweet-spot per the bandwidth analysis in the report (§14.3).
static constexpr size_t CHUNK_SIZE = 16 * 1024;

// Socket FIFO: 4 chunks gives enough BDP slack (31 KB needed at 500 ns RTT).
// Allows up to 4 host writes to be in-flight before the device ACKs.
static constexpr size_t FIFO_SIZE = 4 * CHUNK_SIZE;

// 64 MB — large enough to reach steady-state PCIe throughput and amortise
// kernel launch / DRAM allocation overhead across many chunks.
static constexpr size_t TOTAL_BYTES = 64 * 1024 * 1024;

static_assert(TOTAL_BYTES % CHUNK_SIZE == 0, "TOTAL_BYTES must be a multiple of CHUNK_SIZE");

// Fixed L1 addresses for the kernel ping-pong buffers.
// Placed at 512 KB, safely above the region used by socket config and the
// socket L1 FIFO buffer (both allocated by H2DSocket near the bottom of L1).
// An assertion in launch_puller_kernel() guards against overlap at runtime.
static constexpr uint32_t L1_BUF_A_ADDR = 512 * 1024;
static constexpr uint32_t L1_BUF_B_ADDR = 512 * 1024 + CHUNK_SIZE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_bw(const char* label, size_t bytes, double seconds) {
    double gb_s = static_cast<double>(bytes) / seconds / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(2) << label << ": " << gb_s << " GB/s  (" << (bytes / 1024 / 1024)
              << " MB in " << (seconds * 1000.0) << " ms)\n";
}

// ---------------------------------------------------------------------------
// Kernel launcher
// ---------------------------------------------------------------------------

static void launch_puller_kernel(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    H2DSocket& socket,
    uint32_t dram_base_addr) {
    const uint32_t cfg_addr = socket.get_config_buffer_address();

    // Catch overlaps between the fixed ping-pong buffers and the socket config.
    // If the config address falls inside [L1_BUF_A_ADDR, L1_BUF_B_ADDR+CHUNK_SIZE),
    // the kernel's DRAM reads would silently corrupt socket metadata.
    TT_FATAL(
        cfg_addr < L1_BUF_A_ADDR || cfg_addr >= L1_BUF_B_ADDR + CHUNK_SIZE,
        "Socket config addr 0x{:x} overlaps ping-pong buffers [0x{:x}, 0x{:x}) — "
        "adjust L1_BUF_A_ADDR",
        cfg_addr,
        L1_BUF_A_ADDR,
        L1_BUF_B_ADDR + CHUNK_SIZE);

    auto program = CreateProgram();

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_to_dram_puller.cpp",
        recv_core.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {
                    cfg_addr,
                    static_cast<uint32_t>(L1_BUF_A_ADDR),
                    static_cast<uint32_t>(L1_BUF_B_ADDR),
                    static_cast<uint32_t>(CHUNK_SIZE),
                    static_cast<uint32_t>(TOTAL_BYTES),
                },
        });

    // Only dram_base_addr is a runtime arg: InterleavedAddrGen inside the
    // kernel reads the firmware-global num_dram_banks, which is authoritative
    // after harvesting.  Passing bank count from the host was the bug that
    // caused noc_async_read_barrier hangs in dram_stream_sender (see its header).
    SetRuntimeArgs(program, kernel, recv_core.core_coord, {dram_base_addr});

    auto mesh_workload = MeshWorkload();
    mesh_workload.add_program(MeshCoordinateRange(recv_core.device_coord), std::move(program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
}

// ---------------------------------------------------------------------------
// Single correctness + bandwidth run
// ---------------------------------------------------------------------------

static void run_pcie_to_dram(
    const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoreCoord& recv_core, bool verify_data) {
    // 1. DRAM destination buffer.
    //    The kernel uses InterleavedAddrGen to stripe writes across DRAM banks.
    auto dram_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = TOTAL_BYTES},
        DeviceLocalBufferConfig{
            .page_size = CHUNK_SIZE,
            .buffer_type = BufferType::DRAM,
        },
        mesh_device.get());

    // 2. H2D socket in DEVICE_PULL mode.
    //    The host writes data to a page-locked ring buffer; the device kernel
    //    pulls each chunk via a PCIe NOC read (noc_read_with_state).
    //    BufferType::L1 is required by H2DSocket (the socket L1 FIFO is used
    //    for flow-control arithmetic only — the kernel reads into its own L1
    //    ping-pong buffers and writes directly to DRAM).
    auto socket = H2DSocket(mesh_device, recv_core, BufferType::L1, FIFO_SIZE, H2DMode::DEVICE_PULL);
    socket.set_page_size(CHUNK_SIZE);

    // 3. Source data (host side).
    std::vector<uint32_t> src_data(TOTAL_BYTES / sizeof(uint32_t));
    std::iota(src_data.begin(), src_data.end(), 0u);

    // 4. Dispatch kernel BEFORE feeding data so the device is ready to receive
    //    the first socket_wait_for_pages notification.
    launch_puller_kernel(mesh_device, recv_core, socket, static_cast<uint32_t>(dram_buf->address()));

    // 5. Feed data through the socket one chunk at a time.
    //    H2DSocket::write() for DEVICE_PULL mode:
    //      a) blocks if the ring is full (reserve_bytes spins on bytes_acked),
    //      b) memcpy's the chunk into the host pinned ring buffer,
    //      c) writes bytes_sent to device L1 via TLB write (notify_receiver).
    //    The device kernel wakes on (c) and issues the PCIe NOC read.
    const size_t num_chunks = TOTAL_BYTES / CHUNK_SIZE;
    const uint32_t* src_ptr = src_data.data();

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_chunks; ++i) {
        // write() signature takes void* — the data is not modified.
        socket.write(const_cast<uint32_t*>(src_ptr + i * CHUNK_SIZE / sizeof(uint32_t)), 1);
    }

    // 6. Wait until the device has ACKed every byte from PCIe
    //    (bytes_acked == bytes_sent in host pinned memory).
    socket.barrier();

    // 7. Wait for kernel completion including the drain-step DRAM write.
    //    socket.barrier() returns after the last socket_notify_sender, but the
    //    final DRAM write happens in the drain step AFTER that notify.
    //    Finish() blocks until all previously enqueued commands complete.
    Finish(mesh_device->mesh_command_queue());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    print_bw("PCIe->DRAM single-core", TOTAL_BYTES, elapsed);

    // 8. Optional correctness check: read DRAM back and compare.
    //    ReadShard is synchronous; it also acts as an additional kernel fence
    //    (command queue FIFO guarantees the read executes after the kernel).
    if (verify_data) {
        std::vector<uint32_t> dst_data(TOTAL_BYTES / sizeof(uint32_t));
        ReadShard(mesh_device->mesh_command_queue(), dst_data, dram_buf, recv_core.device_coord);
        EXPECT_EQ(src_data, dst_data);
    }
}

// ---------------------------------------------------------------------------
// Bandwidth sweep: num_runs iterations, no verification
// ---------------------------------------------------------------------------

static void bench_pcie_to_dram(
    const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoreCoord& recv_core, int num_runs = 5) {
    const size_t num_chunks = TOTAL_BYTES / CHUNK_SIZE;
    // Fixed fill pattern — verification not needed in the benchmark path.
    const std::vector<uint32_t> src_data(TOTAL_BYTES / sizeof(uint32_t), 0xDEADBEEF);

    double best_bw = 0.0;
    for (int run = 0; run < num_runs; ++run) {
        auto dram_buf = MeshBuffer::create(
            ReplicatedBufferConfig{.size = TOTAL_BYTES},
            DeviceLocalBufferConfig{
                .page_size = CHUNK_SIZE,
                .buffer_type = BufferType::DRAM,
            },
            mesh_device.get());

        auto socket = H2DSocket(mesh_device, recv_core, BufferType::L1, FIFO_SIZE, H2DMode::DEVICE_PULL);
        socket.set_page_size(CHUNK_SIZE);

        launch_puller_kernel(mesh_device, recv_core, socket, static_cast<uint32_t>(dram_buf->address()));

        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < num_chunks; ++i) {
            socket.write(const_cast<uint32_t*>(src_data.data() + i * CHUNK_SIZE / sizeof(uint32_t)), 1);
        }
        socket.barrier();
        Finish(mesh_device->mesh_command_queue());

        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        double bw = static_cast<double>(TOTAL_BYTES) / s / (1024.0 * 1024.0 * 1024.0);
        best_bw = std::max(best_bw, bw);
        std::cout << "  run " << run << ": " << std::fixed << std::setprecision(2) << bw << " GB/s\n";
    }
    std::cout << "  best: " << std::fixed << std::setprecision(2) << best_bw << " GB/s\n";
}

// ---------------------------------------------------------------------------
// GTest — GenericMeshDeviceFixture works on any topology (1x1 P150 included)
// ---------------------------------------------------------------------------

using PCIeToDRAMFixture = GenericMeshDeviceFixture;

// Correctness: verify every byte written by the host arrives in DRAM intact.
TEST_F(PCIeToDRAMFixture, PCIeToDRAM_SingleCore_Correctness) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    MeshCoreCoord recv{MeshCoordinate(0, 0), CoreCoord(0, 0)};
    run_pcie_to_dram(mesh_device_, recv, /*verify_data=*/true);
}

// Bandwidth: 5 runs of 64 MB, no verification — establishes single-core baseline.
// Expected: ~30–100 MB/s (RTT-limited, single outstanding PCIe read per chunk).
// Compare against DRAM->PCIe streaming from test_dram_stream.cpp and against
// the 140-core grid result once Phase 3 is implemented.
TEST_F(PCIeToDRAMFixture, PCIeToDRAM_SingleCore_Bandwidth) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    MeshCoreCoord recv{MeshCoordinate(0, 0), CoreCoord(0, 0)};
    std::cout << "\n=== PCIe->DRAM Single-Core Bandwidth"
              << " (chunk=" << CHUNK_SIZE / 1024 << " KB"
              << ", total=" << TOTAL_BYTES / 1024 / 1024 << " MB) ===\n";
    bench_pcie_to_dram(mesh_device_, recv, 5);
}

}  // namespace tt::tt_metal::distributed
