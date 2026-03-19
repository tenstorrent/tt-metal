// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Socket kernel bandwidth sweep — apples-to-apples comparison with
// test_persistent_dma.cpp (PersistentDmaFixture).
//
// Covers all four transfer modes:
//   D2H_DRAM  : DRAM → L1 (ping-pong) → PCIe → host   (dram_stream_sender.cpp)
//   D2H_L1    : L1  → PCIe → host                      (pcie_socket_sender.cpp)
//   H2D_DRAM  : host → PCIe → L1 (ping-pong) → DRAM   (pcie_to_dram_puller.cpp)
//   H2D_L1    : host → PCIe → L1                       (pcie_socket_receiver.cpp)
//
// Each mode is swept over the same size range as the persistent DMA engine:
//   256 KB / 1 MB / 4 MB / 16 MB / 64 MB  × 20 iterations
//   (L1 modes cap at 512 KB per core due to BH L1 ≈ 1.33 MB)
//
// Additional tests:
//   CoreScaling  : 1 / 4 / 8 / 16 simultaneous cores at 64 MB total
//   Bidir        : concurrent D2H_DRAM + H2D_DRAM (two threads)
//
// Output format matches PersistentDmaFixture for direct comparison.
//
// Build & run:
//   ninja -C build_Release distributed_unit_tests
//   ./build_Release/test/tt_metal/distributed/distributed_unit_tests \
//       --gtest_filter="SocketBandwidthFixture.*"
//
// Run only the summary sweep:
//   --gtest_filter="SocketBandwidthFixture.*SizeSweep*"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/host_buffer.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include "gmock/gmock.h"

namespace tt::tt_metal::distributed {

// ── Constants ──────────────────────────────────────────────────────────────

// 16 KB = NOC_MAX_BURST_SIZE on BH — optimal for both reads and writes.
static constexpr size_t CHUNK_SIZE = 16 * 1024;
// Socket FIFO: 4 chunks gives enough bandwidth-delay product slack (~64 KB)
// so the device rarely stalls on flow-control during steady-state streaming.
static constexpr size_t FIFO_SIZE = 4 * CHUNK_SIZE;
// Number of timed iterations per size point — matches PersistentDmaFixture.
static constexpr int BW_ITERS = 20;

// L1 ping-pong buffer base — safely above firmware/dispatch/socket regions.
// L1_BUF_B follows immediately; both are validated against socket config addr
// before kernel launch.
static constexpr uint32_t L1_BUF_A_ADDR = 512u * 1024u;
static constexpr uint32_t L1_BUF_B_ADDR = L1_BUF_A_ADDR + static_cast<uint32_t>(CHUNK_SIZE);

// L1 data buffer base for the L1-only tests (pcie_socket_sender / receiver).
// Must be above L1_BUF_B and socket config region.  BH L1 = 1.33 MB total;
// we stay below 1 MB to leave headroom.
static constexpr uint32_t L1_DATA_ADDR = L1_BUF_B_ADDR + static_cast<uint32_t>(CHUNK_SIZE);

// DRAM size sweep: matches test_persistent_dma.cpp DMA_FullSuite ranges.
static const std::vector<std::pair<const char*, size_t>> DRAM_SIZES = {
    {"256 KB", 256u * 1024u},
    {"  1 MB", 1u * 1024u * 1024u},
    {"  4 MB", 4u * 1024u * 1024u},
    {" 16 MB", 16u * 1024u * 1024u},
    {" 64 MB", 64u * 1024u * 1024u},
};

// L1 sweep: capped at 512 KB so the full transfer fits in a single core's L1.
static const std::vector<std::pair<const char*, size_t>> L1_SIZES = {
    {" 64 KB", 64u * 1024u},
    {"128 KB", 128u * 1024u},
    {"256 KB", 256u * 1024u},
    {"512 KB", 512u * 1024u},
};

// Core counts for scaling tests (capped at runtime against actual grid).
static const std::vector<uint32_t> CORE_COUNTS = {1, 4, 8, 16};

// ── Helpers ────────────────────────────────────────────────────────────────

static void print_table_header(const char* title) {
    std::cout << "\n┌─── " << title << " ─────────────────────────────────\n"
              << "│        size   iters    min GB/s    avg GB/s    max GB/s\n"
              << "│  -------------------------------------------------------\n";
}

static void print_table_footer() { std::cout << "+----------------------------------------------------------\n"; }

static void print_row(const char* size_label, int iters, const std::vector<double>& bw_samples) {
    double mn = *std::min_element(bw_samples.begin(), bw_samples.end());
    double sum = std::accumulate(bw_samples.begin(), bw_samples.end(), 0.0);
    double avg = sum / bw_samples.size();
    double mx = *std::max_element(bw_samples.begin(), bw_samples.end());
    std::cout << "│  " << size_label << "  " << std::setw(6) << iters << "  " << std::fixed << std::setprecision(2)
              << std::setw(10) << mn << "  " << std::setw(10) << avg << "  " << std::setw(10) << mx << "\n";
}

static inline double bw_gb_s(size_t bytes, double seconds) {
    return static_cast<double>(bytes) / seconds / (1024.0 * 1024.0 * 1024.0);
}

// Guard against socket config buffer overlapping L1 ping-pong buffers.
static void check_no_overlap(uint32_t cfg_addr, const char* ctx) {
    TT_FATAL(
        cfg_addr < L1_BUF_A_ADDR || cfg_addr >= L1_BUF_B_ADDR + static_cast<uint32_t>(CHUNK_SIZE),
        "{}: socket config 0x{:x} overlaps L1 ping-pong [0x{:x}, 0x{:x})",
        ctx,
        cfg_addr,
        L1_BUF_A_ADDR,
        L1_BUF_B_ADDR + CHUNK_SIZE);
}

// ── D2H DRAM: DRAM → socket → host ────────────────────────────────────────
//
// Uses dram_stream_sender.cpp: DRAM is read into L1 ping-pong buffers, then
// streamed to host pinned memory via PCIe NOC writes.  The host reads chunks
// from the D2HSocket FIFO.

static double run_d2h_dram_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    const CoreCoord& sender_core,
    size_t total_bytes) {
    // 1. DRAM source buffer pre-filled with known data.
    auto dram_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{.page_size = CHUNK_SIZE, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    {
        std::vector<uint32_t> fill(total_bytes / sizeof(uint32_t), 0xA5A5A5A5u);
        WriteShard(mesh_device->mesh_command_queue(), dram_buf, fill, device_coord);
        Finish(mesh_device->mesh_command_queue());
    }
    const uint32_t dram_base = static_cast<uint32_t>(dram_buf->address());

    // 2. D2H socket.
    MeshCoreCoord mesh_core{device_coord, sender_core};
    auto socket = D2HSocket(mesh_device, mesh_core, FIFO_SIZE);
    socket.set_page_size(CHUNK_SIZE);
    check_no_overlap(static_cast<uint32_t>(socket.get_config_buffer_address()), "D2H_DRAM");

    // 3. Launch kernel: dram_stream_sender.cpp
    {
        auto program = CreateProgram();
        const uint32_t cfg_addr = static_cast<uint32_t>(socket.get_config_buffer_address());
        auto kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/dram_stream_sender.cpp",
            sender_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args =
                    {
                        cfg_addr,
                        static_cast<uint32_t>(L1_BUF_A_ADDR),
                        static_cast<uint32_t>(L1_BUF_B_ADDR),
                        static_cast<uint32_t>(CHUNK_SIZE),
                        static_cast<uint32_t>(total_bytes),
                    },
            });
        SetRuntimeArgs(program, kernel, sender_core, {dram_base});
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    // 4. Timed host read.
    const size_t num_chunks = total_bytes / CHUNK_SIZE;
    std::vector<uint32_t> sink(CHUNK_SIZE / sizeof(uint32_t));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        socket.read(sink.data(), 1);
    }
    socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── D2H L1only: L1 → socket → host ────────────────────────────────────────
//
// Uses pcie_socket_sender.cpp: data pre-loaded in device L1, then NOC-written
// directly to host PCIe endpoint — no DRAM hop.  Isolates pure PCIe write BW.

static double run_d2h_l1_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    const CoreCoord& sender_core,
    size_t total_bytes) {
    TT_FATAL(
        L1_DATA_ADDR + total_bytes <= 1u * 1024u * 1024u,
        "D2H L1: total_bytes {} exceeds safe L1 region (max {} from base 0x{:x})",
        total_bytes,
        1u * 1024u * 1024u - L1_DATA_ADDR,
        L1_DATA_ADDR);

    // 1. Allocate L1 source buffer and pre-fill it.
    auto l1_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{.page_size = static_cast<uint32_t>(total_bytes), .buffer_type = BufferType::L1},
        mesh_device.get());
    // Verify the allocator gave us the address we expect (or close to it).
    const uint32_t l1_data_addr = static_cast<uint32_t>(l1_buf->address());
    {
        std::vector<uint32_t> fill(total_bytes / sizeof(uint32_t), 0xB6B6B6B6u);
        WriteShard(mesh_device->mesh_command_queue(), l1_buf, fill, device_coord);
        Finish(mesh_device->mesh_command_queue());
    }

    // 2. D2H socket.
    MeshCoreCoord mesh_core{device_coord, sender_core};
    auto socket = D2HSocket(mesh_device, mesh_core, FIFO_SIZE);
    socket.set_page_size(CHUNK_SIZE);
    const uint32_t cfg_addr = static_cast<uint32_t>(socket.get_config_buffer_address());
    TT_FATAL(
        cfg_addr < l1_data_addr || cfg_addr >= l1_data_addr + total_bytes,
        "D2H_L1: socket config 0x{:x} overlaps L1 data [0x{:x}, 0x{:x})",
        cfg_addr,
        l1_data_addr,
        l1_data_addr + total_bytes);

    // 3. Launch pcie_socket_sender.cpp
    {
        auto program = CreateProgram();
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_sender.cpp",
            sender_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args =
                    {
                        cfg_addr,
                        l1_data_addr,
                        static_cast<uint32_t>(CHUNK_SIZE),
                        static_cast<uint32_t>(total_bytes),
                    },
            });
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    // 4. Timed host read.
    const size_t num_chunks = total_bytes / CHUNK_SIZE;
    std::vector<uint32_t> sink(CHUNK_SIZE / sizeof(uint32_t));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        socket.read(sink.data(), 1);
    }
    socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── H2D DRAM: host → socket → DRAM ────────────────────────────────────────
//
// Uses pcie_to_dram_puller.cpp: device pulls from host pinned ring buffer via
// PCIe NOC reads, writes into L1 ping-pong, drains to DRAM.

static double run_h2d_dram_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    const CoreCoord& recv_core,
    size_t total_bytes) {
    // 1. DRAM destination buffer.
    auto dram_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{.page_size = CHUNK_SIZE, .buffer_type = BufferType::DRAM},
        mesh_device.get());

    // 2. H2D socket in DEVICE_PULL mode (device issues NOC reads from host ring).
    MeshCoreCoord mesh_core{device_coord, recv_core};
    auto socket = H2DSocket(mesh_device, mesh_core, BufferType::L1, FIFO_SIZE, H2DMode::DEVICE_PULL);
    socket.set_page_size(CHUNK_SIZE);
    const uint32_t cfg_addr = socket.get_config_buffer_address();
    check_no_overlap(cfg_addr, "H2D_DRAM");

    // 3. Launch pcie_to_dram_puller.cpp — kernel must be running before we push data.
    {
        auto program = CreateProgram();
        auto kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_to_dram_puller.cpp",
            recv_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args =
                    {
                        cfg_addr,
                        static_cast<uint32_t>(L1_BUF_A_ADDR),
                        static_cast<uint32_t>(L1_BUF_B_ADDR),
                        static_cast<uint32_t>(CHUNK_SIZE),
                        static_cast<uint32_t>(total_bytes),
                    },
            });
        SetRuntimeArgs(program, kernel, recv_core, {static_cast<uint32_t>(dram_buf->address())});
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    // 4. Timed host write.
    const size_t num_chunks = total_bytes / CHUNK_SIZE;
    std::vector<uint32_t> src(CHUNK_SIZE / sizeof(uint32_t), 0xC7C7C7C7u);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        socket.write(src.data(), 1);
    }
    socket.barrier();
    Finish(mesh_device->mesh_command_queue());
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── H2D L1only: host → socket → L1 ───────────────────────────────────────
//
// Uses pcie_socket_receiver.cpp: device receives via PCIe NOC reads from the
// socket FIFO and stores directly in L1 — no DRAM drain.

static double run_h2d_l1_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    const CoreCoord& recv_core,
    size_t total_bytes) {
    TT_FATAL(
        L1_DATA_ADDR + total_bytes <= 1u * 1024u * 1024u, "H2D L1: total_bytes {} exceeds safe L1 region", total_bytes);

    // Allocate an L1 destination buffer so the allocator tracks the region;
    // the kernel writes to l1_dst_addr directly via the socket.
    auto l1_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{.page_size = static_cast<uint32_t>(total_bytes), .buffer_type = BufferType::L1},
        mesh_device.get());
    const uint32_t l1_dst_addr = static_cast<uint32_t>(l1_buf->address());

    // H2D socket (DEVICE_PULL mode: device NOC-reads from pinned host ring).
    MeshCoreCoord mesh_core{device_coord, recv_core};
    auto socket = H2DSocket(mesh_device, mesh_core, BufferType::L1, FIFO_SIZE, H2DMode::DEVICE_PULL);
    socket.set_page_size(CHUNK_SIZE);
    const uint32_t cfg_addr = socket.get_config_buffer_address();
    TT_FATAL(
        cfg_addr < l1_dst_addr || cfg_addr >= l1_dst_addr + total_bytes,
        "H2D_L1: socket config 0x{:x} overlaps L1 dst [0x{:x}, 0x{:x})",
        cfg_addr,
        l1_dst_addr,
        l1_dst_addr + total_bytes);

    // Launch pcie_socket_receiver.cpp (num_iterations=1 for a single sweep).
    {
        auto program = CreateProgram();
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_receiver.cpp",
            recv_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args =
                    {
                        cfg_addr,
                        l1_dst_addr,
                        static_cast<uint32_t>(CHUNK_SIZE),
                        static_cast<uint32_t>(total_bytes),
                        1u,  // num_iterations
                    },
            });
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    // Timed host write.
    const size_t num_chunks = total_bytes / CHUNK_SIZE;
    std::vector<uint32_t> src(CHUNK_SIZE / sizeof(uint32_t), 0xD8D8D8D8u);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        socket.write(src.data(), 1);
    }
    socket.barrier();
    Finish(mesh_device->mesh_command_queue());
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── Generic sweep runner ───────────────────────────────────────────────────

using OneFn = std::function<double(size_t)>;

static void run_sweep(
    const char* title, const std::vector<std::pair<const char*, size_t>>& sizes, int iters, OneFn one_iter_fn) {
    print_table_header(title);
    for (auto& [label, bytes] : sizes) {
        // Warmup: 2 untimed iterations to load kernel binary and warm DRAM banks.
        for (int w = 0; w < 2; ++w) {
            one_iter_fn(bytes);
        }

        std::vector<double> samples;
        samples.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            samples.push_back(one_iter_fn(bytes));
        }

        print_row(label, iters, samples);
    }
    print_table_footer();
}

// ── Multi-core D2H DRAM ────────────────────────────────────────────────────
//
// N simultaneous socket senders, each on a different core.
// Each core handles total_bytes/N from its own DRAM buffer.
// Host reads round-robin from all N sockets.

static double run_d2h_dram_multicore(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t num_cores,
    size_t total_bytes) {
    TT_FATAL(total_bytes % (num_cores * CHUNK_SIZE) == 0, "total_bytes must be multiple of num_cores * CHUNK_SIZE");

    const size_t per_core_bytes = total_bytes / num_cores;

    // Allocate and fill one DRAM buffer per core.
    std::vector<std::shared_ptr<MeshBuffer>> dram_bufs;
    for (uint32_t c = 0; c < num_cores; ++c) {
        auto buf = MeshBuffer::create(
            ReplicatedBufferConfig{.size = per_core_bytes},
            DeviceLocalBufferConfig{.page_size = CHUNK_SIZE, .buffer_type = BufferType::DRAM},
            mesh_device.get());
        std::vector<uint32_t> fill(per_core_bytes / sizeof(uint32_t), 0xA5000000u | c);
        WriteShard(mesh_device->mesh_command_queue(), buf, fill, device_coord);
        dram_bufs.push_back(std::move(buf));
    }
    Finish(mesh_device->mesh_command_queue());

    // Create one D2HSocket per core.
    std::vector<std::unique_ptr<D2HSocket>> sockets;
    for (uint32_t c = 0; c < num_cores; ++c) {
        MeshCoreCoord mesh_core{device_coord, CoreCoord(c, 0)};
        auto s = std::make_unique<D2HSocket>(mesh_device, mesh_core, FIFO_SIZE);
        s->set_page_size(CHUNK_SIZE);
        check_no_overlap(static_cast<uint32_t>(s->get_config_buffer_address()), "D2H_DRAM multicore");
        sockets.push_back(std::move(s));
    }

    // Build one MeshWorkload with N kernels (one per core, possibly different
    // compile-time socket_config_addr if the allocator placed configs at
    // different L1 addresses — each CreateKernel call compiles independently).
    {
        auto program = CreateProgram();
        for (uint32_t c = 0; c < num_cores; ++c) {
            const uint32_t cfg_addr = static_cast<uint32_t>(sockets[c]->get_config_buffer_address());
            const uint32_t dram_base = static_cast<uint32_t>(dram_bufs[c]->address());
            auto kh = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/dram_stream_sender.cpp",
                CoreCoord(c, 0),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args =
                        {
                            cfg_addr,
                            static_cast<uint32_t>(L1_BUF_A_ADDR),
                            static_cast<uint32_t>(L1_BUF_B_ADDR),
                            static_cast<uint32_t>(CHUNK_SIZE),
                            static_cast<uint32_t>(per_core_bytes),
                        },
                });
            SetRuntimeArgs(program, kh, CoreCoord(c, 0), {dram_base});
        }
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    // Read from all sockets in round-robin until all chunks arrive.
    const size_t chunks_per_core = per_core_bytes / CHUNK_SIZE;
    std::vector<uint32_t> sink(CHUNK_SIZE / sizeof(uint32_t));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunks_per_core; ++i) {
        for (uint32_t c = 0; c < num_cores; ++c) {
            sockets[c]->read(sink.data(), 1);
        }
    }
    for (uint32_t c = 0; c < num_cores; ++c) {
        sockets[c]->barrier();
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── Multi-core H2D DRAM ────────────────────────────────────────────────────

static double run_h2d_dram_multicore(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t num_cores,
    size_t total_bytes) {
    TT_FATAL(total_bytes % (num_cores * CHUNK_SIZE) == 0, "total_bytes must be multiple of num_cores * CHUNK_SIZE");

    const size_t per_core_bytes = total_bytes / num_cores;

    // One DRAM destination buffer per core.
    std::vector<std::shared_ptr<MeshBuffer>> dram_bufs;
    for (uint32_t c = 0; c < num_cores; ++c) {
        dram_bufs.push_back(MeshBuffer::create(
            ReplicatedBufferConfig{.size = per_core_bytes},
            DeviceLocalBufferConfig{.page_size = CHUNK_SIZE, .buffer_type = BufferType::DRAM},
            mesh_device.get()));
    }

    // One H2D socket per core (DEVICE_PULL mode).
    std::vector<std::unique_ptr<H2DSocket>> sockets;
    for (uint32_t c = 0; c < num_cores; ++c) {
        MeshCoreCoord mesh_core{device_coord, CoreCoord(c, 0)};
        auto s = std::make_unique<H2DSocket>(mesh_device, mesh_core, BufferType::L1, FIFO_SIZE, H2DMode::DEVICE_PULL);
        s->set_page_size(CHUNK_SIZE);
        check_no_overlap(s->get_config_buffer_address(), "H2D_DRAM multicore");
        sockets.push_back(std::move(s));
    }

    // Launch all N kernels in one workload.
    {
        auto program = CreateProgram();
        for (uint32_t c = 0; c < num_cores; ++c) {
            const uint32_t cfg_addr = sockets[c]->get_config_buffer_address();
            const uint32_t dram_base = static_cast<uint32_t>(dram_bufs[c]->address());
            auto kh = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_to_dram_puller.cpp",
                CoreCoord(c, 0),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args =
                        {
                            cfg_addr,
                            static_cast<uint32_t>(L1_BUF_A_ADDR),
                            static_cast<uint32_t>(L1_BUF_B_ADDR),
                            static_cast<uint32_t>(CHUNK_SIZE),
                            static_cast<uint32_t>(per_core_bytes),
                        },
                });
            SetRuntimeArgs(program, kh, CoreCoord(c, 0), {dram_base});
        }
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    // Feed all sockets in round-robin.
    const size_t chunks_per_core = per_core_bytes / CHUNK_SIZE;
    std::vector<uint32_t> src(CHUNK_SIZE / sizeof(uint32_t), 0xC7C7C7C7u);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunks_per_core; ++i) {
        for (uint32_t c = 0; c < num_cores; ++c) {
            sockets[c]->write(src.data(), 1);
        }
    }
    for (uint32_t c = 0; c < num_cores; ++c) {
        sockets[c]->barrier();
    }
    Finish(mesh_device->mesh_command_queue());
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── GTest fixture ──────────────────────────────────────────────────────────

using SocketBandwidthFixture = GenericMeshDeviceFixture;

// ─────────────────────────────────────────────────────────────────────────
// D2H DRAM size sweep
// ─────────────────────────────────────────────────────────────────────────
TEST_F(SocketBandwidthFixture, D2H_DRAM_SizeSweep) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported";
    }

    const auto device_coord = MeshCoordinate(0, 0);
    const CoreCoord core(0, 0);

    run_sweep("Socket D2H_DRAM (1 core)", DRAM_SIZES, BW_ITERS, [&](size_t bytes) {
        return run_d2h_dram_once(mesh_device_, device_coord, core, bytes);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// D2H L1only size sweep
// ─────────────────────────────────────────────────────────────────────────
TEST_F(SocketBandwidthFixture, D2H_L1only_SizeSweep) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported";
    }

    const auto device_coord = MeshCoordinate(0, 0);
    const CoreCoord core(0, 0);

    run_sweep("Socket D2H_L1only (1 core, no DRAM)", L1_SIZES, BW_ITERS, [&](size_t bytes) {
        return run_d2h_l1_once(mesh_device_, device_coord, core, bytes);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// H2D DRAM size sweep
// ─────────────────────────────────────────────────────────────────────────
TEST_F(SocketBandwidthFixture, H2D_DRAM_SizeSweep) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported";
    }

    const auto device_coord = MeshCoordinate(0, 0);
    const CoreCoord core(0, 0);

    run_sweep("Socket H2D_DRAM (1 core)", DRAM_SIZES, BW_ITERS, [&](size_t bytes) {
        return run_h2d_dram_once(mesh_device_, device_coord, core, bytes);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// H2D L1only size sweep
// ─────────────────────────────────────────────────────────────────────────
TEST_F(SocketBandwidthFixture, H2D_L1only_SizeSweep) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported";
    }

    const auto device_coord = MeshCoordinate(0, 0);
    const CoreCoord core(0, 0);

    run_sweep("Socket H2D_L1only (1 core, no DRAM drain)", L1_SIZES, BW_ITERS, [&](size_t bytes) {
        return run_h2d_l1_once(mesh_device_, device_coord, core, bytes);
    });
}

// ─────────────────────────────────────────────────────────────────────────
// D2H DRAM core-count scaling (fixed 64 MB total)
// ─────────────────────────────────────────────────────────────────────────
TEST_F(SocketBandwidthFixture, D2H_DRAM_CoreScaling) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported";
    }

    const auto device_coord = MeshCoordinate(0, 0);
    const auto grid = mesh_device_->get_device(device_coord)->compute_with_storage_grid_size();
    const uint32_t max_cores = static_cast<uint32_t>(grid.x);  // use a single row

    constexpr size_t kTotal = 64u * 1024u * 1024u;  // 64 MB total

    std::cout << "\n┌─── Socket D2H_DRAM Core Scaling (64 MB total) ─────────────\n"
              << "│  cores    min GB/s    avg GB/s    max GB/s\n"
              << "│  -----------------------------------------------\n";

    for (uint32_t nc : CORE_COUNTS) {
        if (nc > max_cores) {
            std::cout << "│  " << std::setw(5) << nc << "  (skipped — grid only " << max_cores << " wide)\n";
            continue;
        }
        if (kTotal % (nc * CHUNK_SIZE) != 0) {
            continue;
        }

        // Warmup.
        for (int w = 0; w < 2; ++w) {
            run_d2h_dram_multicore(mesh_device_, device_coord, nc, kTotal);
        }

        std::vector<double> samples;
        for (int i = 0; i < BW_ITERS; ++i) {
            samples.push_back(run_d2h_dram_multicore(mesh_device_, device_coord, nc, kTotal));
        }

        double mn = *std::min_element(samples.begin(), samples.end());
        double avg = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        double mx = *std::max_element(samples.begin(), samples.end());
        std::cout << "│  " << std::setw(5) << nc << "  " << std::fixed << std::setprecision(2) << std::setw(10) << mn
                  << "  " << std::setw(10) << avg << "  " << std::setw(10) << mx << "\n";
    }
    std::cout << "+-----------------------------------------------------------\n";
}

// ─────────────────────────────────────────────────────────────────────────
// H2D DRAM core-count scaling (fixed 64 MB total)
// ─────────────────────────────────────────────────────────────────────────
TEST_F(SocketBandwidthFixture, H2D_DRAM_CoreScaling) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported";
    }

    const auto device_coord = MeshCoordinate(0, 0);
    const auto grid = mesh_device_->get_device(device_coord)->compute_with_storage_grid_size();
    const uint32_t max_cores = static_cast<uint32_t>(grid.x);

    constexpr size_t kTotal = 64u * 1024u * 1024u;

    std::cout << "\n┌─── Socket H2D_DRAM Core Scaling (64 MB total) ─────────────\n"
              << "│  cores    min GB/s    avg GB/s    max GB/s\n"
              << "│  -----------------------------------------------\n";

    for (uint32_t nc : CORE_COUNTS) {
        if (nc > max_cores) {
            std::cout << "│  " << std::setw(5) << nc << "  (skipped — grid only " << max_cores << " wide)\n";
            continue;
        }
        if (kTotal % (nc * CHUNK_SIZE) != 0) {
            continue;
        }

        for (int w = 0; w < 2; ++w) {
            run_h2d_dram_multicore(mesh_device_, device_coord, nc, kTotal);
        }

        std::vector<double> samples;
        for (int i = 0; i < BW_ITERS; ++i) {
            samples.push_back(run_h2d_dram_multicore(mesh_device_, device_coord, nc, kTotal));
        }

        double mn = *std::min_element(samples.begin(), samples.end());
        double avg = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        double mx = *std::max_element(samples.begin(), samples.end());
        std::cout << "│  " << std::setw(5) << nc << "  " << std::fixed << std::setprecision(2) << std::setw(10) << mn
                  << "  " << std::setw(10) << avg << "  " << std::setw(10) << mx << "\n";
    }
    std::cout << "+-----------------------------------------------------------\n";
}

// ─────────────────────────────────────────────────────────────────────────
// Bidirectional: simultaneous D2H_DRAM + H2D_DRAM on separate cores
//
// Two threads: one feeds the H2D socket, one reads from the D2H socket.
// Both run in parallel to match the PersistentDmaFixture.Bidirectional_BW test.
// D2H uses core (0,0), H2D uses core (1,0).
// ─────────────────────────────────────────────────────────────────────────
TEST_F(SocketBandwidthFixture, Bidir_DRAM) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported";
    }

    const auto device_coord = MeshCoordinate(0, 0);
    const auto grid = mesh_device_->get_device(device_coord)->compute_with_storage_grid_size();
    if (static_cast<uint32_t>(grid.x) < 2) {
        GTEST_SKIP() << "Need at least 2 cores for bidir test";
    }

    const std::vector<std::pair<const char*, size_t>> bidir_sizes = {
        {"256 KB", 256u * 1024u},
        {"  1 MB", 1u * 1024u * 1024u},
        {"  4 MB", 4u * 1024u * 1024u},
        {" 16 MB", 16u * 1024u * 1024u},
        {" 64 MB", 64u * 1024u * 1024u},
    };

    std::cout << "\n┌─── Socket Bidir_DRAM (D2H core0 + H2D core1 simultaneous) ──\n"
              << "│        size   iters    D2H GB/s    H2D GB/s  Total GB/s\n"
              << "│  -------------------------------------------------------\n";

    for (auto& [label, bytes] : bidir_sizes) {
        std::vector<double> d2h_bws, h2d_bws;

        // Warmup.
        for (int w = 0; w < 2; ++w) {
            run_d2h_dram_once(mesh_device_, device_coord, CoreCoord(0, 0), bytes);
            run_h2d_dram_once(mesh_device_, device_coord, CoreCoord(1, 0), bytes);
        }

        for (int it = 0; it < BW_ITERS; ++it) {
            // Set up D2H resources.
            auto d2h_dram = MeshBuffer::create(
                ReplicatedBufferConfig{.size = bytes},
                DeviceLocalBufferConfig{.page_size = CHUNK_SIZE, .buffer_type = BufferType::DRAM},
                mesh_device_.get());
            {
                std::vector<uint32_t> fill(bytes / sizeof(uint32_t), 0xA5A5A5A5u);
                WriteShard(mesh_device_->mesh_command_queue(), d2h_dram, fill, device_coord);
                Finish(mesh_device_->mesh_command_queue());
            }
            auto d2h_socket = D2HSocket(mesh_device_, MeshCoreCoord{device_coord, CoreCoord(0, 0)}, FIFO_SIZE);
            d2h_socket.set_page_size(CHUNK_SIZE);
            check_no_overlap(static_cast<uint32_t>(d2h_socket.get_config_buffer_address()), "Bidir D2H");

            // Set up H2D resources.
            auto h2d_dram = MeshBuffer::create(
                ReplicatedBufferConfig{.size = bytes},
                DeviceLocalBufferConfig{.page_size = CHUNK_SIZE, .buffer_type = BufferType::DRAM},
                mesh_device_.get());
            auto h2d_socket = H2DSocket(
                mesh_device_,
                MeshCoreCoord{device_coord, CoreCoord(1, 0)},
                BufferType::L1,
                FIFO_SIZE,
                H2DMode::DEVICE_PULL);
            h2d_socket.set_page_size(CHUNK_SIZE);
            check_no_overlap(h2d_socket.get_config_buffer_address(), "Bidir H2D");

            // Launch both kernels in one workload so they start simultaneously.
            {
                auto program = CreateProgram();
                const uint32_t d2h_cfg = static_cast<uint32_t>(d2h_socket.get_config_buffer_address());
                const uint32_t d2h_base = static_cast<uint32_t>(d2h_dram->address());
                auto d2h_kh = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/socket/dram_stream_sender.cpp",
                    CoreCoord(0, 0),
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = {d2h_cfg, L1_BUF_A_ADDR, L1_BUF_B_ADDR, (uint32_t)CHUNK_SIZE, (uint32_t)bytes},
                    });
                SetRuntimeArgs(program, d2h_kh, CoreCoord(0, 0), {d2h_base});

                const uint32_t h2d_cfg = h2d_socket.get_config_buffer_address();
                const uint32_t h2d_base = static_cast<uint32_t>(h2d_dram->address());
                auto h2d_kh = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_to_dram_puller.cpp",
                    CoreCoord(1, 0),
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = {h2d_cfg, L1_BUF_A_ADDR, L1_BUF_B_ADDR, (uint32_t)CHUNK_SIZE, (uint32_t)bytes},
                    });
                SetRuntimeArgs(program, h2d_kh, CoreCoord(1, 0), {h2d_base});

                auto wl = MeshWorkload();
                wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
                EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), wl, false);
            }

            // Drive D2H (read) and H2D (write) simultaneously from two threads.
            const size_t num_chunks = bytes / CHUNK_SIZE;
            std::vector<uint32_t> sink(CHUNK_SIZE / sizeof(uint32_t));
            std::vector<uint32_t> src(CHUNK_SIZE / sizeof(uint32_t), 0xC7C7C7C7u);

            double d2h_s = 0, h2d_s = 0;
            std::thread d2h_thread([&] {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < num_chunks; ++i) {
                    d2h_socket.read(sink.data(), 1);
                }
                d2h_socket.barrier();
                auto t1 = std::chrono::high_resolution_clock::now();
                d2h_s = std::chrono::duration<double>(t1 - t0).count();
            });
            std::thread h2d_thread([&] {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < num_chunks; ++i) {
                    h2d_socket.write(src.data(), 1);
                }
                h2d_socket.barrier();
                Finish(mesh_device_->mesh_command_queue());
                auto t1 = std::chrono::high_resolution_clock::now();
                h2d_s = std::chrono::duration<double>(t1 - t0).count();
            });
            d2h_thread.join();
            h2d_thread.join();

            d2h_bws.push_back(bw_gb_s(bytes, d2h_s));
            h2d_bws.push_back(bw_gb_s(bytes, h2d_s));
        }

        double d2h_avg = std::accumulate(d2h_bws.begin(), d2h_bws.end(), 0.0) / d2h_bws.size();
        double h2d_avg = std::accumulate(h2d_bws.begin(), h2d_bws.end(), 0.0) / h2d_bws.size();
        std::cout << "│  " << label << "  " << std::setw(6) << BW_ITERS << "  " << std::fixed << std::setprecision(2)
                  << std::setw(10) << d2h_avg << "  " << std::setw(10) << h2d_avg << "  " << std::setw(10)
                  << (d2h_avg + h2d_avg) << "\n";
    }
    std::cout << "+----------------------------------------------------------\n";
}

}  // namespace tt::tt_metal::distributed
