// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// socket_bw_bench — parametrizable socket bandwidth benchmark
//
// Standalone CLI executable covering all four socket transfer modes:
//   D2H_DRAM  : DRAM → L1 ping-pong → PCIe → host   (dram_stream_sender)
//   D2H_L1    : L1   → PCIe → host                  (pcie_socket_sender)
//   H2D_DRAM  : host → PCIe → L1 ping-pong → DRAM   (pcie_to_dram_puller)
//   H2D_L1    : host → PCIe → L1                    (pcie_socket_receiver)
//
// Build:
//   ninja -C build_Release socket_bw_bench
//
// Usage examples:
//   ./socket_bw_bench                          # all modes, default sizes, 1 core
//   ./socket_bw_bench --h2d --dram --cores 4 --iters 5
//   ./socket_bw_bench --dram --cores 1,4,8 --csv > results.csv
//   ./socket_bw_bench --bidir --dram --iters 10 --verbose
//   ./socket_bw_bench --l1 --sizes 64k,128k,256k

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include "impl/context/metal_context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

namespace tt::tt_metal::distributed {

// ── Constants ──────────────────────────────────────────────────────────────

// L1 ping-pong buffer base — safely above firmware/dispatch/socket regions.
static constexpr uint32_t L1_BUF_A_BASE = 512u * 1024u;  // 0x80000

// ── Helpers ────────────────────────────────────────────────────────────────

static inline double bw_gb_s(size_t bytes, double seconds) {
    return static_cast<double>(bytes) / seconds / (1024.0 * 1024.0 * 1024.0);
}

// Parse a size string with optional k/m/g suffix (case-insensitive).
// Returns 0 on parse failure.
static size_t parse_size(const std::string& s) {
    if (s.empty()) {
        return 0;
    }
    char* end = nullptr;
    unsigned long long val = strtoull(s.c_str(), &end, 10);
    if (end == s.c_str()) {
        return 0;
    }
    std::string suffix(end);
    if (suffix.empty() || suffix == "b" || suffix == "B") {
        return static_cast<size_t>(val);
    }
    char c = (char)tolower((unsigned char)suffix[0]);
    if (c == 'k') {
        return static_cast<size_t>(val) * 1024ULL;
    }
    if (c == 'm') {
        return static_cast<size_t>(val) * 1024ULL * 1024ULL;
    }
    if (c == 'g') {
        return static_cast<size_t>(val) * 1024ULL * 1024ULL * 1024ULL;
    }
    return 0;
}

// Split a comma-separated string of size tokens.
static std::vector<size_t> parse_size_list(const std::string& s) {
    std::vector<size_t> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            size_t v = parse_size(tok);
            if (v > 0) {
                out.push_back(v);
            }
        }
    }
    return out;
}

// Split a comma-separated string of unsigned integers.
static std::vector<uint32_t> parse_uint_list(const std::string& s) {
    std::vector<uint32_t> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(static_cast<uint32_t>(std::stoul(tok)));
        }
    }
    return out;
}

// ── BenchConfig ────────────────────────────────────────────────────────────

struct BenchConfig {
    bool dir_d2h = false;
    bool dir_h2d = false;
    bool dir_bidir = false;

    bool mode_dram = false;
    bool mode_l1 = false;

    std::vector<size_t> sizes_dram;  // user-supplied or default
    std::vector<size_t> sizes_l1;    // user-supplied or default (capped at runtime)

    std::vector<uint32_t> core_counts = {1};
    uint32_t core_row = 0;

    int warmup = 2;
    int iters = 20;
    size_t chunk_size = 16 * 1024;  // 16 KB default
    uint32_t fifo_depth = 4;        // fifo_size = fifo_depth * chunk_size

    bool csv = false;
    bool json_out = false;
    bool no_table = false;
    bool verbose = false;
};

// ── Result struct ──────────────────────────────────────────────────────────

struct Result {
    std::string label;      // e.g. "D2H_DRAM"
    std::string direction;  // "d2h" / "h2d" / "bidir_d2h" / "bidir_h2d"
    std::string mode;       // "dram" / "l1"
    uint32_t cores;
    size_t bytes;
    int iters;
    std::vector<double> samples;  // bandwidth samples in GB/s

    double min_bw() const { return *std::min_element(samples.begin(), samples.end()); }
    double avg_bw() const { return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(); }
    double max_bw() const { return *std::max_element(samples.begin(), samples.end()); }
};

// ── run_once functions ─────────────────────────────────────────────────────
//
// Each returns measured bandwidth in GB/s for a single transfer of total_bytes.
// chunk_size and fifo_size are runtime-configurable.
//
// L1 buffer layout (computed from chunk_size):
//   [L1_BUF_A_BASE]                  — ping buffer
//   [L1_BUF_A_BASE + chunk_size]     — pong buffer (= L1_BUF_B_ADDR)
//   [L1_BUF_A_BASE + 2*chunk_size]   — L1 data region for L1-only tests

static double run_d2h_dram_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    const CoreCoord& sender_core,
    size_t total_bytes,
    size_t chunk_size,
    size_t fifo_size) {
    const uint32_t l1_buf_a = L1_BUF_A_BASE;
    const uint32_t l1_buf_b = l1_buf_a + static_cast<uint32_t>(chunk_size);

    auto dram_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{.page_size = chunk_size, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    {
        std::vector<uint32_t> fill(total_bytes / sizeof(uint32_t), 0xA5A5A5A5u);
        WriteShard(mesh_device->mesh_command_queue(), dram_buf, fill, device_coord);
        Finish(mesh_device->mesh_command_queue());
    }
    const uint32_t dram_base = static_cast<uint32_t>(dram_buf->address());

    MeshCoreCoord mesh_core{device_coord, sender_core};
    auto socket = D2HSocket(mesh_device, mesh_core, fifo_size);
    socket.set_page_size(chunk_size);
    const uint32_t cfg_addr = static_cast<uint32_t>(socket.get_config_buffer_address());

    {
        auto program = CreateProgram();
        auto kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/dram_stream_sender.cpp",
            sender_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args =
                    {cfg_addr,
                     l1_buf_a,
                     l1_buf_b,
                     static_cast<uint32_t>(chunk_size),
                     static_cast<uint32_t>(total_bytes)},
            });
        SetRuntimeArgs(program, kernel, sender_core, {dram_base});
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t num_chunks = total_bytes / chunk_size;
    std::vector<uint32_t> sink(chunk_size / sizeof(uint32_t));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        socket.read(sink.data(), 1);
    }
    socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_d2h_l1_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    const CoreCoord& sender_core,
    size_t total_bytes,
    size_t chunk_size,
    size_t fifo_size) {
    auto l1_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{.page_size = static_cast<uint32_t>(total_bytes), .buffer_type = BufferType::L1},
        mesh_device.get());
    {
        std::vector<uint32_t> fill(total_bytes / sizeof(uint32_t), 0xB6B6B6B6u);
        WriteShard(mesh_device->mesh_command_queue(), l1_buf, fill, device_coord);
        Finish(mesh_device->mesh_command_queue());
    }
    const uint32_t actual_l1_addr = static_cast<uint32_t>(l1_buf->address());

    MeshCoreCoord mesh_core{device_coord, sender_core};
    auto socket = D2HSocket(mesh_device, mesh_core, fifo_size);
    socket.set_page_size(chunk_size);
    const uint32_t cfg_addr = static_cast<uint32_t>(socket.get_config_buffer_address());

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
                    {cfg_addr, actual_l1_addr, static_cast<uint32_t>(chunk_size), static_cast<uint32_t>(total_bytes)},
            });
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t num_chunks = total_bytes / chunk_size;
    std::vector<uint32_t> sink(chunk_size / sizeof(uint32_t));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        socket.read(sink.data(), 1);
    }
    socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_h2d_dram_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    const CoreCoord& recv_core,
    size_t total_bytes,
    size_t chunk_size,
    size_t fifo_size) {
    const uint32_t l1_buf_a = L1_BUF_A_BASE;
    const uint32_t l1_buf_b = l1_buf_a + static_cast<uint32_t>(chunk_size);

    auto dram_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{.page_size = chunk_size, .buffer_type = BufferType::DRAM},
        mesh_device.get());

    MeshCoreCoord mesh_core{device_coord, recv_core};
    auto socket = H2DSocket(mesh_device, mesh_core, BufferType::L1, fifo_size, H2DMode::DEVICE_PULL);
    socket.set_page_size(chunk_size);
    const uint32_t cfg_addr = socket.get_config_buffer_address();

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
                    {cfg_addr,
                     l1_buf_a,
                     l1_buf_b,
                     static_cast<uint32_t>(chunk_size),
                     static_cast<uint32_t>(total_bytes)},
            });
        SetRuntimeArgs(program, kernel, recv_core, {static_cast<uint32_t>(dram_buf->address())});
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t num_chunks = total_bytes / chunk_size;
    std::vector<uint32_t> src(chunk_size / sizeof(uint32_t), 0xC7C7C7C7u);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        socket.write(src.data(), 1);
    }
    socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();
    // Drain CQ after timing: device has already acknowledged all writes via barrier().
    Finish(mesh_device->mesh_command_queue());

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_h2d_l1_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    const CoreCoord& recv_core,
    size_t total_bytes,
    size_t chunk_size,
    size_t fifo_size) {
    auto l1_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{.page_size = static_cast<uint32_t>(total_bytes), .buffer_type = BufferType::L1},
        mesh_device.get());
    const uint32_t l1_dst_addr = static_cast<uint32_t>(l1_buf->address());

    MeshCoreCoord mesh_core{device_coord, recv_core};
    auto socket = H2DSocket(mesh_device, mesh_core, BufferType::L1, fifo_size, H2DMode::DEVICE_PULL);
    socket.set_page_size(chunk_size);
    const uint32_t cfg_addr = socket.get_config_buffer_address();

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
                    {cfg_addr,
                     l1_dst_addr,
                     static_cast<uint32_t>(chunk_size),
                     static_cast<uint32_t>(total_bytes),
                     1u},  // num_iterations
            });
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t num_chunks = total_bytes / chunk_size;
    std::vector<uint32_t> src(chunk_size / sizeof(uint32_t), 0xD8D8D8D8u);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_chunks; ++i) {
        socket.write(src.data(), 1);
    }
    socket.barrier();
    auto t1 = std::chrono::high_resolution_clock::now();
    Finish(mesh_device->mesh_command_queue());

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── Multi-core run functions ───────────────────────────────────────────────
//
// Each spawns N cores in a single row, one socket per core.
// Host drives all sockets round-robin.  Total bytes are split evenly.

static double run_d2h_dram_multicore(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t num_cores,
    uint32_t core_row,
    size_t total_bytes,
    size_t chunk_size,
    size_t fifo_size) {
    const uint32_t l1_buf_a = L1_BUF_A_BASE;
    const uint32_t l1_buf_b = l1_buf_a + static_cast<uint32_t>(chunk_size);
    const size_t per_core_bytes = total_bytes / num_cores;

    std::vector<std::shared_ptr<MeshBuffer>> dram_bufs;
    for (uint32_t c = 0; c < num_cores; ++c) {
        auto buf = MeshBuffer::create(
            ReplicatedBufferConfig{.size = per_core_bytes},
            DeviceLocalBufferConfig{.page_size = chunk_size, .buffer_type = BufferType::DRAM},
            mesh_device.get());
        std::vector<uint32_t> fill(per_core_bytes / sizeof(uint32_t), 0xA5000000u | c);
        WriteShard(mesh_device->mesh_command_queue(), buf, fill, device_coord);
        dram_bufs.push_back(std::move(buf));
    }
    Finish(mesh_device->mesh_command_queue());

    std::vector<std::unique_ptr<D2HSocket>> sockets;
    for (uint32_t c = 0; c < num_cores; ++c) {
        MeshCoreCoord mesh_core{device_coord, CoreCoord(c, core_row)};
        auto s = std::make_unique<D2HSocket>(mesh_device, mesh_core, fifo_size);
        s->set_page_size(chunk_size);
        sockets.push_back(std::move(s));
    }

    {
        auto program = CreateProgram();
        for (uint32_t c = 0; c < num_cores; ++c) {
            const uint32_t cfg_addr = static_cast<uint32_t>(sockets[c]->get_config_buffer_address());
            const uint32_t dram_base = static_cast<uint32_t>(dram_bufs[c]->address());
            auto kh = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/dram_stream_sender.cpp",
                CoreCoord(c, core_row),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args =
                        {cfg_addr,
                         l1_buf_a,
                         l1_buf_b,
                         static_cast<uint32_t>(chunk_size),
                         static_cast<uint32_t>(per_core_bytes)},
                });
            SetRuntimeArgs(program, kh, CoreCoord(c, core_row), {dram_base});
        }
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t chunks_per_core = per_core_bytes / chunk_size;
    std::vector<uint32_t> sink(chunk_size / sizeof(uint32_t));

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

static double run_h2d_dram_multicore(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t num_cores,
    uint32_t core_row,
    size_t total_bytes,
    size_t chunk_size,
    size_t fifo_size) {
    const uint32_t l1_buf_a = L1_BUF_A_BASE;
    const uint32_t l1_buf_b = l1_buf_a + static_cast<uint32_t>(chunk_size);
    const size_t per_core_bytes = total_bytes / num_cores;

    std::vector<std::shared_ptr<MeshBuffer>> dram_bufs;
    for (uint32_t c = 0; c < num_cores; ++c) {
        dram_bufs.push_back(MeshBuffer::create(
            ReplicatedBufferConfig{.size = per_core_bytes},
            DeviceLocalBufferConfig{.page_size = chunk_size, .buffer_type = BufferType::DRAM},
            mesh_device.get()));
    }

    std::vector<std::unique_ptr<H2DSocket>> sockets;
    for (uint32_t c = 0; c < num_cores; ++c) {
        MeshCoreCoord mesh_core{device_coord, CoreCoord(c, core_row)};
        auto s = std::make_unique<H2DSocket>(mesh_device, mesh_core, BufferType::L1, fifo_size, H2DMode::DEVICE_PULL);
        s->set_page_size(chunk_size);
        sockets.push_back(std::move(s));
    }

    {
        auto program = CreateProgram();
        for (uint32_t c = 0; c < num_cores; ++c) {
            const uint32_t cfg_addr = sockets[c]->get_config_buffer_address();
            const uint32_t dram_base = static_cast<uint32_t>(dram_bufs[c]->address());
            auto kh = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_to_dram_puller.cpp",
                CoreCoord(c, core_row),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args =
                        {cfg_addr,
                         l1_buf_a,
                         l1_buf_b,
                         static_cast<uint32_t>(chunk_size),
                         static_cast<uint32_t>(per_core_bytes)},
                });
            SetRuntimeArgs(program, kh, CoreCoord(c, core_row), {dram_base});
        }
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t chunks_per_core = per_core_bytes / chunk_size;
    std::vector<uint32_t> src(chunk_size / sizeof(uint32_t), 0xC7C7C7C7u);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunks_per_core; ++i) {
        for (uint32_t c = 0; c < num_cores; ++c) {
            sockets[c]->write(src.data(), 1);
        }
    }
    for (uint32_t c = 0; c < num_cores; ++c) {
        sockets[c]->barrier();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    Finish(mesh_device->mesh_command_queue());

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_d2h_l1_multicore(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t num_cores,
    uint32_t core_row,
    size_t total_bytes,
    size_t chunk_size,
    size_t fifo_size) {
    const size_t per_core_bytes = total_bytes / num_cores;

    std::vector<std::shared_ptr<MeshBuffer>> l1_bufs;
    for (uint32_t c = 0; c < num_cores; ++c) {
        auto buf = MeshBuffer::create(
            ReplicatedBufferConfig{.size = per_core_bytes},
            DeviceLocalBufferConfig{.page_size = static_cast<uint32_t>(per_core_bytes), .buffer_type = BufferType::L1},
            mesh_device.get());
        std::vector<uint32_t> fill(per_core_bytes / sizeof(uint32_t), 0xB6000000u | c);
        WriteShard(mesh_device->mesh_command_queue(), buf, fill, device_coord);
        l1_bufs.push_back(std::move(buf));
    }
    Finish(mesh_device->mesh_command_queue());

    std::vector<std::unique_ptr<D2HSocket>> sockets;
    for (uint32_t c = 0; c < num_cores; ++c) {
        MeshCoreCoord mesh_core{device_coord, CoreCoord(c, core_row)};
        auto s = std::make_unique<D2HSocket>(mesh_device, mesh_core, fifo_size);
        s->set_page_size(chunk_size);
        sockets.push_back(std::move(s));
    }

    {
        auto program = CreateProgram();
        for (uint32_t c = 0; c < num_cores; ++c) {
            const uint32_t cfg_addr = static_cast<uint32_t>(sockets[c]->get_config_buffer_address());
            const uint32_t l1_src_addr = static_cast<uint32_t>(l1_bufs[c]->address());
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_sender.cpp",
                CoreCoord(c, core_row),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args =
                        {cfg_addr,
                         l1_src_addr,
                         static_cast<uint32_t>(chunk_size),
                         static_cast<uint32_t>(per_core_bytes)},
                });
        }
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t chunks_per_core = per_core_bytes / chunk_size;
    std::vector<uint32_t> sink(chunk_size / sizeof(uint32_t));

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

static double run_h2d_l1_multicore(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t num_cores,
    uint32_t core_row,
    size_t total_bytes,
    size_t chunk_size,
    size_t fifo_size) {
    const size_t per_core_bytes = total_bytes / num_cores;

    std::vector<std::shared_ptr<MeshBuffer>> l1_bufs;
    for (uint32_t c = 0; c < num_cores; ++c) {
        l1_bufs.push_back(MeshBuffer::create(
            ReplicatedBufferConfig{.size = per_core_bytes},
            DeviceLocalBufferConfig{.page_size = static_cast<uint32_t>(per_core_bytes), .buffer_type = BufferType::L1},
            mesh_device.get()));
    }

    std::vector<std::unique_ptr<H2DSocket>> sockets;
    for (uint32_t c = 0; c < num_cores; ++c) {
        MeshCoreCoord mesh_core{device_coord, CoreCoord(c, core_row)};
        auto s = std::make_unique<H2DSocket>(mesh_device, mesh_core, BufferType::L1, fifo_size, H2DMode::DEVICE_PULL);
        s->set_page_size(chunk_size);
        sockets.push_back(std::move(s));
    }

    {
        auto program = CreateProgram();
        for (uint32_t c = 0; c < num_cores; ++c) {
            const uint32_t cfg_addr = sockets[c]->get_config_buffer_address();
            const uint32_t l1_dst_addr = static_cast<uint32_t>(l1_bufs[c]->address());
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_receiver.cpp",
                CoreCoord(c, core_row),
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args =
                        {cfg_addr,
                         l1_dst_addr,
                         static_cast<uint32_t>(chunk_size),
                         static_cast<uint32_t>(per_core_bytes),
                         1u},  // num_iterations
                });
        }
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t chunks_per_core = per_core_bytes / chunk_size;
    std::vector<uint32_t> src(chunk_size / sizeof(uint32_t), 0xD8D8D8D8u);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunks_per_core; ++i) {
        for (uint32_t c = 0; c < num_cores; ++c) {
            sockets[c]->write(src.data(), 1);
        }
    }
    for (uint32_t c = 0; c < num_cores; ++c) {
        sockets[c]->barrier();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    Finish(mesh_device->mesh_command_queue());

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── Bidirectional runs ─────────────────────────────────────────────────────
//
// D2H on core (0, core_row) + H2D on core (1, core_row), simultaneous threads.
// Returns {d2h_bw, h2d_bw} for a single iteration.

static std::pair<double, double> run_bidir_dram_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t core_row,
    size_t bytes,
    size_t chunk_size,
    size_t fifo_size) {
    const uint32_t l1_buf_a = L1_BUF_A_BASE;
    const uint32_t l1_buf_b = l1_buf_a + static_cast<uint32_t>(chunk_size);

    // D2H DRAM buffer.
    auto d2h_dram = MeshBuffer::create(
        ReplicatedBufferConfig{.size = bytes},
        DeviceLocalBufferConfig{.page_size = chunk_size, .buffer_type = BufferType::DRAM},
        mesh_device.get());
    {
        std::vector<uint32_t> fill(bytes / sizeof(uint32_t), 0xA5A5A5A5u);
        WriteShard(mesh_device->mesh_command_queue(), d2h_dram, fill, device_coord);
        Finish(mesh_device->mesh_command_queue());
    }

    auto d2h_socket = D2HSocket(mesh_device, MeshCoreCoord{device_coord, CoreCoord(0, core_row)}, fifo_size);
    d2h_socket.set_page_size(chunk_size);
    const uint32_t d2h_cfg = static_cast<uint32_t>(d2h_socket.get_config_buffer_address());

    // H2D DRAM buffer.
    auto h2d_dram = MeshBuffer::create(
        ReplicatedBufferConfig{.size = bytes},
        DeviceLocalBufferConfig{.page_size = chunk_size, .buffer_type = BufferType::DRAM},
        mesh_device.get());

    auto h2d_socket = H2DSocket(
        mesh_device,
        MeshCoreCoord{device_coord, CoreCoord(1, core_row)},
        BufferType::L1,
        fifo_size,
        H2DMode::DEVICE_PULL);
    h2d_socket.set_page_size(chunk_size);
    const uint32_t h2d_cfg = h2d_socket.get_config_buffer_address();

    // Launch both kernels simultaneously.
    {
        auto program = CreateProgram();
        auto d2h_kh = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/dram_stream_sender.cpp",
            CoreCoord(0, core_row),
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args =
                    {d2h_cfg, l1_buf_a, l1_buf_b, static_cast<uint32_t>(chunk_size), static_cast<uint32_t>(bytes)},
            });
        SetRuntimeArgs(program, d2h_kh, CoreCoord(0, core_row), {static_cast<uint32_t>(d2h_dram->address())});

        auto h2d_kh = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_to_dram_puller.cpp",
            CoreCoord(1, core_row),
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args =
                    {h2d_cfg, l1_buf_a, l1_buf_b, static_cast<uint32_t>(chunk_size), static_cast<uint32_t>(bytes)},
            });
        SetRuntimeArgs(program, h2d_kh, CoreCoord(1, core_row), {static_cast<uint32_t>(h2d_dram->address())});

        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t num_chunks = bytes / chunk_size;
    std::vector<uint32_t> sink(chunk_size / sizeof(uint32_t));
    std::vector<uint32_t> src(chunk_size / sizeof(uint32_t), 0xC7C7C7C7u);

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
        auto t1 = std::chrono::high_resolution_clock::now();
        h2d_s = std::chrono::duration<double>(t1 - t0).count();
        Finish(mesh_device->mesh_command_queue());
    });
    d2h_thread.join();
    h2d_thread.join();

    return {bw_gb_s(bytes, d2h_s), bw_gb_s(bytes, h2d_s)};
}

static std::pair<double, double> run_bidir_l1_once(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t core_row,
    size_t bytes,
    size_t chunk_size,
    size_t fifo_size) {
    // D2H L1: core (0, core_row)
    auto d2h_l1 = MeshBuffer::create(
        ReplicatedBufferConfig{.size = bytes},
        DeviceLocalBufferConfig{.page_size = static_cast<uint32_t>(bytes), .buffer_type = BufferType::L1},
        mesh_device.get());
    {
        std::vector<uint32_t> fill(bytes / sizeof(uint32_t), 0xB6B6B6B6u);
        WriteShard(mesh_device->mesh_command_queue(), d2h_l1, fill, device_coord);
        Finish(mesh_device->mesh_command_queue());
    }
    const uint32_t d2h_l1_addr = static_cast<uint32_t>(d2h_l1->address());

    auto d2h_socket = D2HSocket(mesh_device, MeshCoreCoord{device_coord, CoreCoord(0, core_row)}, fifo_size);
    d2h_socket.set_page_size(chunk_size);
    const uint32_t d2h_cfg = static_cast<uint32_t>(d2h_socket.get_config_buffer_address());

    // H2D L1: core (1, core_row)
    auto h2d_l1 = MeshBuffer::create(
        ReplicatedBufferConfig{.size = bytes},
        DeviceLocalBufferConfig{.page_size = static_cast<uint32_t>(bytes), .buffer_type = BufferType::L1},
        mesh_device.get());
    const uint32_t h2d_l1_addr = static_cast<uint32_t>(h2d_l1->address());

    auto h2d_socket = H2DSocket(
        mesh_device,
        MeshCoreCoord{device_coord, CoreCoord(1, core_row)},
        BufferType::L1,
        fifo_size,
        H2DMode::DEVICE_PULL);
    h2d_socket.set_page_size(chunk_size);
    const uint32_t h2d_cfg = h2d_socket.get_config_buffer_address();

    {
        auto program = CreateProgram();
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_sender.cpp",
            CoreCoord(0, core_row),
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {d2h_cfg, d2h_l1_addr, static_cast<uint32_t>(chunk_size), static_cast<uint32_t>(bytes)},
            });
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_socket_receiver.cpp",
            CoreCoord(1, core_row),
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args =
                    {h2d_cfg, h2d_l1_addr, static_cast<uint32_t>(chunk_size), static_cast<uint32_t>(bytes), 1u},
            });
        auto wl = MeshWorkload();
        wl.add_program(MeshCoordinateRange(device_coord), std::move(program));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), wl, false);
    }

    const size_t num_chunks = bytes / chunk_size;
    std::vector<uint32_t> sink(chunk_size / sizeof(uint32_t));
    std::vector<uint32_t> src(chunk_size / sizeof(uint32_t), 0xD8D8D8D8u);

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
        auto t1 = std::chrono::high_resolution_clock::now();
        h2d_s = std::chrono::duration<double>(t1 - t0).count();
        Finish(mesh_device->mesh_command_queue());
    });
    d2h_thread.join();
    h2d_thread.join();

    return {bw_gb_s(bytes, d2h_s), bw_gb_s(bytes, h2d_s)};
}

// ── Output formatters ──────────────────────────────────────────────────────

using OneFn = std::function<double(size_t)>;

static std::string fmt_bytes(size_t bytes) {
    std::ostringstream ss;
    if (bytes >= 1024 * 1024) {
        ss << std::setw(4) << (bytes / (1024 * 1024)) << " MB";
    } else {
        ss << std::setw(4) << (bytes / 1024) << " KB";
    }
    return ss.str();
}

static void print_table(const std::vector<Result>& results) {
    if (results.empty()) {
        return;
    }
    std::cout << "\n┌─── Socket Bandwidth Results ──────────────────────────────────────────────\n"
              << "│  label              mode  cores       size   iters    min GB/s    avg GB/s    max GB/s\n"
              << "│  ──────────────────────────────────────────────────────────────────────────────────\n";
    for (const auto& r : results) {
        std::cout << "│  " << std::left << std::setw(18) << r.label << "  " << std::setw(4) << r.mode << "  "
                  << std::right << std::setw(5) << r.cores << "  " << std::setw(8) << fmt_bytes(r.bytes) << "  "
                  << std::setw(6) << r.iters << "  " << std::fixed << std::setprecision(2) << std::setw(10)
                  << r.min_bw() << "  " << std::setw(10) << r.avg_bw() << "  " << std::setw(10) << r.max_bw() << "\n";
    }
    std::cout << "└───────────────────────────────────────────────────────────────────────────\n";
}

static void print_csv(const std::vector<Result>& results) {
    std::cout << "label,direction,mode,cores,bytes,iters,min_gb_s,avg_gb_s,max_gb_s\n";
    for (const auto& r : results) {
        std::cout << r.label << "," << r.direction << "," << r.mode << "," << r.cores << "," << r.bytes << ","
                  << r.iters << std::fixed << std::setprecision(4) << "," << r.min_bw() << "," << r.avg_bw() << ","
                  << r.max_bw() << "\n";
    }
}

static void print_json(const std::vector<Result>& results) {
    std::cout << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "  {\n"
                  << "    \"label\": \"" << r.label << "\",\n"
                  << "    \"direction\": \"" << r.direction << "\",\n"
                  << "    \"mode\": \"" << r.mode << "\",\n"
                  << "    \"cores\": " << r.cores << ",\n"
                  << "    \"bytes\": " << r.bytes << ",\n"
                  << "    \"iters\": " << r.iters << ",\n"
                  << std::fixed << std::setprecision(4) << "    \"min_gb_s\": " << r.min_bw() << ",\n"
                  << "    \"avg_gb_s\": " << r.avg_bw() << ",\n"
                  << "    \"max_gb_s\": " << r.max_bw() << "\n"
                  << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    std::cout << "]\n";
}

static void print_verbose_samples(const Result& r) {
    std::cout << "[" << r.label << " " << fmt_bytes(r.bytes) << "] samples:";
    for (double s : r.samples) {
        std::cout << std::fixed << std::setprecision(2) << " " << s;
    }
    std::cout << " GB/s\n";
}

// ── Argument parsing ───────────────────────────────────────────────────────

static void usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [OPTIONS]\n"
                 "\n"
                 "Directions (default: all):\n"
                 "  --d2h                  Enable D2H tests\n"
                 "  --h2d                  Enable H2D tests\n"
                 "  --bidir                Enable bidirectional (D2H core0 + H2D core1)\n"
                 "\n"
                 "Modes (default: both):\n"
                 "  --dram                 Enable DRAM-backed transfers\n"
                 "  --l1                   Enable L1-only transfers\n"
                 "\n"
                 "Transfer sizes (default: sweep 256K,1M,4M,16M,64M for DRAM; 64K,128K,256K,448K for L1):\n"
                 "  --sizes 256k,1m,4m     Comma-separated list (k/m/g suffixes supported)\n"
                 "\n"
                 "Core scaling:\n"
                 "  --cores 1,4,8          Comma-separated core counts (default: 1)\n"
                 "  --core-row 0           Which logical row to place kernels on (default: 0)\n"
                 "\n"
                 "Iteration control:\n"
                 "  --iters 20             Timed iterations per size point (default: 20)\n"
                 "  --warmup 2             Warmup iterations before timing (default: 2)\n"
                 "\n"
                 "Tuning:\n"
                 "  --chunk-size 16384     DMA chunk size in bytes (default: 16384 = 16 KB)\n"
                 "  --fifo-depth 4         Socket FIFO depth in chunks (default: 4 → 64 KB FIFO)\n"
                 "\n"
                 "Output:\n"
                 "  --csv                  Emit CSV rows to stdout\n"
                 "  --json                 Emit JSON array to stdout\n"
                 "  --no-table             Suppress the ASCII table\n"
                 "  --verbose              Print per-iteration samples\n"
                 "\n"
                 "  -h, --help             Print this usage\n";
}

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;

    bool explicit_dir = false;
    bool explicit_mode = false;
    bool user_sizes = false;

    static const struct option long_opts[] = {
        {"d2h", no_argument, nullptr, 'd'},
        {"h2d", no_argument, nullptr, 'H'},
        {"bidir", no_argument, nullptr, 'b'},
        {"dram", no_argument, nullptr, 'D'},
        {"l1", no_argument, nullptr, 'l'},
        {"sizes", required_argument, nullptr, 's'},
        {"cores", required_argument, nullptr, 'c'},
        {"core-row", required_argument, nullptr, 'r'},
        {"iters", required_argument, nullptr, 'i'},
        {"warmup", required_argument, nullptr, 'w'},
        {"chunk-size", required_argument, nullptr, 'C'},
        {"fifo-depth", required_argument, nullptr, 'f'},
        {"csv", no_argument, nullptr, 1},
        {"json", no_argument, nullptr, 2},
        {"no-table", no_argument, nullptr, 3},
        {"verbose", no_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    int opt, idx = 0;
    while ((opt = getopt_long(argc, argv, "h", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'd':
                cfg.dir_d2h = true;
                explicit_dir = true;
                break;
            case 'H':
                cfg.dir_h2d = true;
                explicit_dir = true;
                break;
            case 'b':
                cfg.dir_bidir = true;
                explicit_dir = true;
                break;
            case 'D':
                cfg.mode_dram = true;
                explicit_mode = true;
                break;
            case 'l':
                cfg.mode_l1 = true;
                explicit_mode = true;
                break;
            case 's': {
                auto v = parse_size_list(optarg);
                if (!v.empty()) {
                    cfg.sizes_dram = v;
                    cfg.sizes_l1 = v;
                    user_sizes = true;
                }
                break;
            }
            case 'c': cfg.core_counts = parse_uint_list(optarg); break;
            case 'r': cfg.core_row = static_cast<uint32_t>(std::stoul(optarg)); break;
            case 'i': cfg.iters = std::stoi(optarg); break;
            case 'w': cfg.warmup = std::stoi(optarg); break;
            case 'C': cfg.chunk_size = parse_size(optarg); break;
            case 'f': cfg.fifo_depth = static_cast<uint32_t>(std::stoul(optarg)); break;
            case 1: cfg.csv = true; break;
            case 2: cfg.json_out = true; break;
            case 3: cfg.no_table = true; break;
            case 'v': cfg.verbose = true; break;
            case 'h': usage(argv[0]); exit(0);
            default: usage(argv[0]); exit(1);
        }
    }

    if (!explicit_dir) {
        cfg.dir_d2h = cfg.dir_h2d = true;
    }
    if (!explicit_mode) {
        cfg.mode_dram = cfg.mode_l1 = true;
    }
    if (cfg.core_counts.empty()) {
        cfg.core_counts = {1};
    }

    if (!user_sizes) {
        cfg.sizes_dram = {
            256u * 1024u,
            1u * 1024u * 1024u,
            4u * 1024u * 1024u,
            16u * 1024u * 1024u,
            64u * 1024u * 1024u,
        };
        cfg.sizes_l1 = {
            64u * 1024u,
            128u * 1024u,
            256u * 1024u,
            448u * 1024u,
        };
    }

    return cfg;
}

// ── main ───────────────────────────────────────────────────────────────────

}  // namespace tt::tt_metal::distributed

int main(int argc, char** argv) {
    using namespace tt::tt_metal::distributed;
    using namespace tt::tt_metal;

    BenchConfig cfg = parse_args(argc, argv);

    // ── Device setup ────────────────────────────────────────────────────────
    const auto system_mesh_shape = MetalContext::instance().get_system_mesh().shape();

    auto mesh_device = MeshDevice::create(
        MeshDeviceConfig(system_mesh_shape),
        /*l1_small_size=*/0,
        /*trace_region_size=*/0,
        /*num_cqs=*/1,
        DispatchCoreType::WORKER);

    const MeshCoordinate device_coord(0, 0);

    // ── Pinned memory check ─────────────────────────────────────────────────
    if (!experimental::GetMemoryPinningParameters(*mesh_device).can_map_to_noc) {
        std::cerr << "WARNING: Host memory pinning to NOC not supported on this machine.\n"
                  << "         Socket bandwidth tests require pinned memory — skipping all tests.\n";
        mesh_device->close();
        return 1;
    }

    // ── Compute grid limits ─────────────────────────────────────────────────
    const auto grid = mesh_device->get_device(device_coord)->compute_with_storage_grid_size();
    const uint32_t max_cols = static_cast<uint32_t>(grid.x);

    // ── Compute L1 size cap ─────────────────────────────────────────────────
    // L1 data region for L1-only tests: L1_BUF_A_BASE + 2 * chunk_size
    // Available space: l1_size_per_core - L1_DATA_ADDR
    const uint32_t l1_data_addr = L1_BUF_A_BASE + static_cast<uint32_t>(2 * cfg.chunk_size);
    const uint32_t l1_total = mesh_device->l1_size_per_core();
    const uint32_t l1_data_max = (l1_total > l1_data_addr) ? (l1_total - l1_data_addr) : 0u;

    // Filter sizes_l1: only keep those that fit in a single core's L1.
    {
        std::vector<size_t> safe;
        for (size_t s : cfg.sizes_l1) {
            if (s <= l1_data_max) {
                safe.push_back(s);
            }
        }
        if (safe.empty() && !cfg.sizes_l1.empty()) {
            std::cerr << "WARNING: All L1 sizes exceed available L1 (" << l1_data_max / 1024
                      << " KB). L1 tests skipped.\n";
        }
        cfg.sizes_l1 = safe;
    }

    const size_t fifo_size = cfg.fifo_depth * cfg.chunk_size;

    std::cout << "socket_bw_bench  chunk=" << (cfg.chunk_size / 1024) << "KB"
              << "  fifo_depth=" << cfg.fifo_depth << "  iters=" << cfg.iters << "  warmup=" << cfg.warmup
              << "  cores=" << grid.x << "x" << grid.y << "  l1_safe_max=" << (l1_data_max / 1024) << "KB\n";

    std::vector<Result> all_results;

    // Helper: run warmup + timed sweep for a single (direction, mode, cores, size).
    auto collect = [&](const std::string& label,
                       const std::string& dir,
                       const std::string& mode,
                       uint32_t num_cores,
                       const std::vector<size_t>& sizes,
                       OneFn fn) {
        if (sizes.empty()) {
            return;
        }

        // For multicore: skip sizes not evenly divisible by num_cores * chunk_size.
        // Recording 0.0 as a sample would corrupt the stats.
        const size_t min_granularity = num_cores * cfg.chunk_size;
        std::vector<size_t> valid_sizes;
        for (size_t s : sizes) {
            if (s % min_granularity == 0) {
                valid_sizes.push_back(s);
            }
        }
        if (valid_sizes.empty()) {
            std::cerr << "NOTE: " << label << " — no sizes divisible by " << (min_granularity / 1024)
                      << " KB (cores=" << num_cores << " chunk=" << (cfg.chunk_size / 1024) << "KB), skipping.\n";
            return;
        }

        // Warmup (use first valid size).
        for (int w = 0; w < cfg.warmup; ++w) {
            fn(valid_sizes[0]);
        }

        for (size_t bytes : valid_sizes) {
            Result r;
            r.label = label;
            r.direction = dir;
            r.mode = mode;
            r.cores = num_cores;
            r.bytes = bytes;
            r.iters = cfg.iters;
            r.samples.reserve(cfg.iters);
            for (int it = 0; it < cfg.iters; ++it) {
                r.samples.push_back(fn(bytes));
            }
            if (cfg.verbose) {
                print_verbose_samples(r);
            }
            all_results.push_back(std::move(r));
        }
    };

    // ── D2H sweeps ──────────────────────────────────────────────────────────
    if (cfg.dir_d2h) {
        for (uint32_t nc : cfg.core_counts) {
            if (nc > max_cols) {
                std::cerr << "WARNING: --cores " << nc << " > grid width " << max_cols << ", skipping.\n";
                continue;
            }
            if (cfg.mode_dram) {
                const std::string lbl = "D2H_DRAM_" + std::to_string(nc) + "c";
                collect(lbl, "d2h", "dram", nc, cfg.sizes_dram, [&](size_t bytes) -> double {
                    if (nc == 1) {
                        return run_d2h_dram_once(
                            mesh_device, device_coord, CoreCoord(0, cfg.core_row), bytes, cfg.chunk_size, fifo_size);
                    }
                    return run_d2h_dram_multicore(
                        mesh_device, device_coord, nc, cfg.core_row, bytes, cfg.chunk_size, fifo_size);
                });
            }
            if (cfg.mode_l1) {
                const std::string lbl = "D2H_L1_" + std::to_string(nc) + "c";
                collect(lbl, "d2h", "l1", nc, cfg.sizes_l1, [&](size_t bytes) -> double {
                    if (nc == 1) {
                        return run_d2h_l1_once(
                            mesh_device, device_coord, CoreCoord(0, cfg.core_row), bytes, cfg.chunk_size, fifo_size);
                    }
                    return run_d2h_l1_multicore(
                        mesh_device, device_coord, nc, cfg.core_row, bytes, cfg.chunk_size, fifo_size);
                });
            }
        }
    }

    // ── H2D sweeps ──────────────────────────────────────────────────────────
    if (cfg.dir_h2d) {
        for (uint32_t nc : cfg.core_counts) {
            if (nc > max_cols) {
                std::cerr << "WARNING: --cores " << nc << " > grid width " << max_cols << ", skipping.\n";
                continue;
            }
            if (cfg.mode_dram) {
                const std::string lbl = "H2D_DRAM_" + std::to_string(nc) + "c";
                collect(lbl, "h2d", "dram", nc, cfg.sizes_dram, [&](size_t bytes) -> double {
                    if (nc == 1) {
                        return run_h2d_dram_once(
                            mesh_device, device_coord, CoreCoord(0, cfg.core_row), bytes, cfg.chunk_size, fifo_size);
                    }
                    return run_h2d_dram_multicore(
                        mesh_device, device_coord, nc, cfg.core_row, bytes, cfg.chunk_size, fifo_size);
                });
            }
            if (cfg.mode_l1) {
                const std::string lbl = "H2D_L1_" + std::to_string(nc) + "c";
                collect(lbl, "h2d", "l1", nc, cfg.sizes_l1, [&](size_t bytes) -> double {
                    if (nc == 1) {
                        return run_h2d_l1_once(
                            mesh_device, device_coord, CoreCoord(0, cfg.core_row), bytes, cfg.chunk_size, fifo_size);
                    }
                    return run_h2d_l1_multicore(
                        mesh_device, device_coord, nc, cfg.core_row, bytes, cfg.chunk_size, fifo_size);
                });
            }
        }
    }

    // ── Bidirectional sweeps ─────────────────────────────────────────────────
    if (cfg.dir_bidir) {
        if (max_cols < 2) {
            std::cerr << "WARNING: Bidirectional test needs ≥2 cores; grid width=" << max_cols << "\n";
        } else {
            auto run_bidir = [&](const std::string& mode_str,
                                 const std::vector<size_t>& sizes,
                                 std::function<std::pair<double, double>(size_t)> fn_once) {
                if (sizes.empty()) {
                    return;
                }
                // Warmup.
                for (int w = 0; w < cfg.warmup; ++w) {
                    fn_once(sizes[0]);
                }

                for (size_t bytes : sizes) {
                    Result rd, rh;
                    rd.label = "Bidir_D2H_" + mode_str;
                    rd.direction = "bidir_d2h";
                    rd.mode = mode_str;
                    rd.cores = 2;
                    rd.bytes = bytes;
                    rd.iters = cfg.iters;

                    rh.label = "Bidir_H2D_" + mode_str;
                    rh.direction = "bidir_h2d";
                    rh.mode = mode_str;
                    rh.cores = 2;
                    rh.bytes = bytes;
                    rh.iters = cfg.iters;

                    for (int it = 0; it < cfg.iters; ++it) {
                        auto [d, h] = fn_once(bytes);
                        rd.samples.push_back(d);
                        rh.samples.push_back(h);
                    }
                    if (cfg.verbose) {
                        print_verbose_samples(rd);
                        print_verbose_samples(rh);
                    }
                    all_results.push_back(std::move(rd));
                    all_results.push_back(std::move(rh));
                }
            };

            if (cfg.mode_dram) {
                run_bidir("dram", cfg.sizes_dram, [&](size_t bytes) {
                    return run_bidir_dram_once(
                        mesh_device, device_coord, cfg.core_row, bytes, cfg.chunk_size, fifo_size);
                });
            }
            if (cfg.mode_l1) {
                run_bidir("l1", cfg.sizes_l1, [&](size_t bytes) {
                    return run_bidir_l1_once(mesh_device, device_coord, cfg.core_row, bytes, cfg.chunk_size, fifo_size);
                });
            }
        }
    }

    // ── Output ───────────────────────────────────────────────────────────────
    if (!cfg.no_table) {
        print_table(all_results);
    }
    if (cfg.csv) {
        print_csv(all_results);
    }
    if (cfg.json_out) {
        print_json(all_results);
    }

    // ── Teardown ─────────────────────────────────────────────────────────────
    mesh_device->close();
    return 0;
}
