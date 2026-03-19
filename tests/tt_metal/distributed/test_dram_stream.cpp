// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "gmock/gmock.h"

namespace tt::tt_metal::distributed {

// ---------------------------------------------------------------------------
// Configuration — tune these to explore the throughput vs efficiency trade-off
// ---------------------------------------------------------------------------

// Chunk size per ping-pong buffer. Must be PCIe-aligned (multiples of the HAL
// alignment, typically 32 or 64 bytes). Larger = better PCIe TLP efficiency.
static constexpr size_t CHUNK_SIZE = 256 * 1024;  // 256 KB

// Socket FIFO must hold >= bandwidth * round_trip_latency (BDP).
// 4x chunk gives enough slack that the device rarely stalls on flow control.
static constexpr size_t FIFO_SIZE = 4 * CHUNK_SIZE;

// Total transfer: large enough to reach steady-state throughput.
static constexpr size_t TOTAL_BYTES = 256 * 1024 * 1024;  // 256 MB

// L1 addresses for the two ping-pong buffers inside the kernel.
// Chosen to avoid the socket config buffer and standard runtime allocations.
// Adjust if you see L1 allocation conflicts at runtime.
static constexpr uint32_t L1_BUF_A_ADDR = 512 * 1024;
static constexpr uint32_t L1_BUF_B_ADDR = 512 * 1024 + CHUNK_SIZE;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
static void print_bw(const char* label, size_t bytes, double seconds) {
    double gb_s = (static_cast<double>(bytes) / seconds) / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(2) << label << ": " << gb_s << " GB/s  (" << (bytes / 1024 / 1024)
              << " MB in " << (seconds * 1000.0) << " ms)\n";
}

// ---------------------------------------------------------------------------
// Helper: build and launch the streaming kernel for one run
// ---------------------------------------------------------------------------
static void launch_stream_kernel(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& sender_core,
    D2HSocket& output_socket,
    uint32_t dram_base_addr) {
    // Verify the socket config address does NOT overlap the ping-pong buffers.
    // If it did, the DRAM reads would silently corrupt the socket metadata.
    uint32_t cfg_addr = static_cast<uint32_t>(output_socket.get_config_buffer_address());
    TT_FATAL(
        cfg_addr < L1_BUF_A_ADDR || cfg_addr >= L1_BUF_B_ADDR + CHUNK_SIZE,
        "Socket config addr 0x{:x} overlaps with L1 ping-pong buffers [0x{:x}, 0x{:x})",
        cfg_addr,
        L1_BUF_A_ADDR,
        L1_BUF_B_ADDR + CHUNK_SIZE);

    auto stream_program = CreateProgram();

    auto kernel = CreateKernel(
        stream_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/dram_stream_sender.cpp",
        sender_core.core_coord,
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

    // Only dram_base_addr is needed: InterleavedAddrGen inside the kernel reads
    // the firmware-global num_dram_banks directly, which is authoritative after
    // harvesting. Passing bank count from the host was the root cause of the hang.
    SetRuntimeArgs(
        stream_program,
        kernel,
        sender_core.core_coord,
        {
            dram_base_addr,
        });

    auto mesh_workload = MeshWorkload();
    mesh_workload.add_program(MeshCoordinateRange(sender_core.device_coord), std::move(stream_program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
}

// ---------------------------------------------------------------------------
// Core test: correctness + bandwidth measurement
// ---------------------------------------------------------------------------
void test_dram_stream(
    const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoreCoord& sender_core, bool verify_data) {
    // 1. DRAM source buffer
    const ReplicatedBufferConfig dram_buf_cfg{.size = TOTAL_BYTES};
    const DeviceLocalBufferConfig dram_local_cfg{
        .page_size = CHUNK_SIZE,
        .buffer_type = BufferType::DRAM,
    };
    auto dram_buf = MeshBuffer::create(dram_buf_cfg, dram_local_cfg, mesh_device.get());

    std::vector<uint32_t> src_data(TOTAL_BYTES / sizeof(uint32_t));
    std::iota(src_data.begin(), src_data.end(), 0);
    WriteShard(mesh_device->mesh_command_queue(), dram_buf, src_data, sender_core.device_coord);

    // 2. D2H socket
    auto output_socket = D2HSocket(mesh_device, sender_core, FIFO_SIZE);
    output_socket.set_page_size(CHUNK_SIZE);

    // 3. dram_buf->address() is the per-bank base address for the interleaved layout.
    //    The kernel uses InterleavedAddrGen internally — no bank count needed here.
    uint32_t dram_base = static_cast<uint32_t>(dram_buf->address());

    // 4. Launch.
    //    DPRINT output from the kernel is enabled via env var at runtime:
    //      TT_METAL_DPRINT_CORES=0,0 ./distributed_unit_tests --gtest_filter=...
    launch_stream_kernel(mesh_device, sender_core, output_socket, dram_base);

    // 6. Host consumer
    const size_t num_chunks = TOTAL_BYTES / CHUNK_SIZE;
    std::vector<uint32_t> dst_data(TOTAL_BYTES / sizeof(uint32_t));
    uint32_t* dst_ptr = dst_data.data();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        output_socket.read(dst_ptr, 1);
        dst_ptr += CHUNK_SIZE / sizeof(uint32_t);
    }
    output_socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    print_bw("DRAM->PCIe stream", TOTAL_BYTES, elapsed);

    if (verify_data) {
        EXPECT_EQ(src_data, dst_data);
    }
}

// ---------------------------------------------------------------------------
// Bandwidth-only: 5 runs, single reused sink buffer, no verification
// ---------------------------------------------------------------------------
void bench_dram_stream(
    const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoreCoord& sender_core, int num_runs = 5) {
    const ReplicatedBufferConfig dram_buf_cfg{.size = TOTAL_BYTES};
    const DeviceLocalBufferConfig dram_local_cfg{
        .page_size = CHUNK_SIZE,
        .buffer_type = BufferType::DRAM,
    };
    auto dram_buf = MeshBuffer::create(dram_buf_cfg, dram_local_cfg, mesh_device.get());

    std::vector<uint32_t> src_data(TOTAL_BYTES / sizeof(uint32_t), 0xDEADBEEF);
    WriteShard(mesh_device->mesh_command_queue(), dram_buf, src_data, sender_core.device_coord);

    uint32_t dram_base = static_cast<uint32_t>(dram_buf->address());

    std::vector<uint32_t> sink(CHUNK_SIZE / sizeof(uint32_t));
    const size_t num_chunks = TOTAL_BYTES / CHUNK_SIZE;

    double best_bw = 0.0;
    for (int run = 0; run < num_runs; ++run) {
        auto output_socket = D2HSocket(mesh_device, sender_core, FIFO_SIZE);
        output_socket.set_page_size(CHUNK_SIZE);

        launch_stream_kernel(mesh_device, sender_core, output_socket, dram_base);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_chunks; ++i) {
            output_socket.read(sink.data(), 1);
        }
        output_socket.barrier();
        auto t1 = std::chrono::high_resolution_clock::now();

        double s = std::chrono::duration<double>(t1 - t0).count();
        double bw = (static_cast<double>(TOTAL_BYTES) / s) / (1024.0 * 1024.0 * 1024.0);
        best_bw = std::max(best_bw, bw);
        std::cout << "  run " << run << ": " << std::fixed << std::setprecision(2) << bw << " GB/s\n";
    }
    std::cout << "  best: " << std::fixed << std::setprecision(2) << best_bw << " GB/s\n";
}

// ---------------------------------------------------------------------------
// GTest — GenericMeshDeviceFixture adapts to whatever topology is present,
// so it runs on a single P150 (1x1) without skipping.
// ---------------------------------------------------------------------------
using DramStreamFixture = GenericMeshDeviceFixture;

TEST_F(DramStreamFixture, DramToPCIeStream_Correctness) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    MeshCoreCoord sender{MeshCoordinate(0, 0), CoreCoord(0, 0)};
    test_dram_stream(mesh_device_, sender, /*verify_data=*/true);
}

TEST_F(DramStreamFixture, DramToPCIeStream_Bandwidth) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    MeshCoreCoord sender{MeshCoordinate(0, 0), CoreCoord(0, 0)};
    std::cout << "\n=== DRAM -> PCIe Streaming Bandwidth (chunk=" << CHUNK_SIZE / 1024 << "KB) ===\n";
    bench_dram_stream(mesh_device_, sender, 5);
}

}  // namespace tt::tt_metal::distributed
