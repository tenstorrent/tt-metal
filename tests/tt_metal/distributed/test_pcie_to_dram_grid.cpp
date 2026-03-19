// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase 2 test: 4-core PCIe → DRAM grid
//
// Validates pcie_to_dram_grid_puller.cpp running on 4 simultaneous cores with
// no socket flow-control overhead.  The host pre-fills a pinned buffer, then
// all 4 kernels pull concurrently; a done-flag per core signals completion.
//
// Compared to Phase 1 (single-core, socket protocol):
//   Phase 1 best: ~7.7 GB/s  (limited by per-chunk socket round-trip)
//   Phase 2 goal: ~30–62 GB/s (4 outstanding PCIe reads cover BDP ≈ 31 KB)
//
// Build & run:
//   ninja -C build_Release distributed_unit_tests
//   ./build_Release/test/tt_metal/distributed/distributed_unit_tests \
//       --gtest_filter="PCIeGridFixture.*"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/sub_device.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_align.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "gmock/gmock.h"

namespace tt::tt_metal::distributed {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static constexpr uint32_t NUM_CORES = 4;
static constexpr size_t CHUNK_SIZE = 16 * 1024;           // 16 KB = NOC max burst
static constexpr size_t TOTAL_BYTES = 256 * 1024 * 1024;  // 256 MB

static_assert(TOTAL_BYTES % (NUM_CORES * CHUNK_SIZE) == 0);

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

// Page-aligned mmap helper.  Returns the pointer and its alloc size.
static std::pair<void*, size_t> mmap_alloc(size_t bytes) {
    size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    size_t alloc = (bytes + page - 1) & ~(page - 1);
    void* p = mmap(nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TT_FATAL(p != MAP_FAILED, "mmap failed for {} bytes", alloc);
    return {p, alloc};
}

// ---------------------------------------------------------------------------
// Main test logic
// ---------------------------------------------------------------------------

static void run_grid(const std::shared_ptr<MeshDevice>& mesh_device, bool verify_data, int num_bw_runs = 1) {
    auto device_coord = MeshCoordinate(0, 0);
    auto device_id = mesh_device->get_device(device_coord)->id();

    MeshCoordinateRangeSet device_range;
    device_range.merge(MeshCoordinateRange(device_coord));

    const size_t per_core_bytes = TOTAL_BYTES / NUM_CORES;
    const uint32_t chunks_per_core = static_cast<uint32_t>(per_core_bytes / CHUNK_SIZE);
    const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));

    // ── 1. Allocate + pin source buffer ─────────────────────────────────────
    auto [src_raw, src_alloc] = mmap_alloc(TOTAL_BYTES);
    auto src_owner = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(src_raw), [src_alloc](uint32_t* p) { munmap(p, src_alloc); });
    uint32_t* src = src_owner.get();

    // Fill with sequential data (i) so each word matches its offset.
    std::iota(src, src + TOTAL_BYTES / sizeof(uint32_t), 0u);

    HostBuffer src_hbuf(tt::stl::Span<uint32_t>(src, TOTAL_BYTES / sizeof(uint32_t)), MemoryPin(src_owner));
    auto src_pinned = experimental::PinnedMemory::Create(*mesh_device, device_range, src_hbuf, /*map_to_noc=*/true);
    auto src_noc = src_pinned->get_noc_addr(device_id);
    TT_FATAL(src_noc.has_value(), "Failed to get NOC address for source buffer");

    const uint32_t pcie_xy_enc = src_noc->pcie_xy_enc;
    const uint64_t src_phys_base = src_noc->addr;

    // ── 2. Allocate + pin done-flags (separate page per risk §16.3) ─────────
    // Each core writes one uint32_t; we allocate an entire page to prevent any
    // chance of the driver rejecting an overlapping-page pin.
    auto [flags_raw, flags_alloc] = mmap_alloc(page_sz);
    auto flags_owner = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(flags_raw), [flags_alloc](uint32_t* p) { munmap(p, flags_alloc); });
    uint32_t* flags = flags_owner.get();
    std::memset(flags, 0, page_sz);

    HostBuffer flags_hbuf(tt::stl::Span<uint32_t>(flags, page_sz / sizeof(uint32_t)), MemoryPin(flags_owner));
    auto flags_pinned = experimental::PinnedMemory::Create(*mesh_device, device_range, flags_hbuf, /*map_to_noc=*/true);
    auto flags_noc = flags_pinned->get_noc_addr(device_id);
    TT_FATAL(flags_noc.has_value(), "Failed to get NOC address for done-flags buffer");
    const uint64_t flags_phys_base = flags_noc->addr;

    // ── 3. DRAM destination buffer ───────────────────────────────────────────
    auto dram_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = TOTAL_BYTES},
        DeviceLocalBufferConfig{
            .page_size = CHUNK_SIZE,
            .buffer_type = BufferType::DRAM,
        },
        mesh_device.get());
    const uint32_t dram_base = static_cast<uint32_t>(dram_buf->address());

    // ── 4. Build 4-core program ──────────────────────────────────────────────
    // 4 cores in a row: logical (0,0)–(3,0).
    CoreRange cores(CoreCoord(0, 0), CoreCoord(NUM_CORES - 1, 0));

    auto program = CreateProgram();
    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_to_dram_grid_puller.cpp",
        CoreRangeSet(cores),
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {
                    static_cast<uint32_t>(L1_BUF_A_ADDR),
                    static_cast<uint32_t>(L1_BUF_B_ADDR),
                    static_cast<uint32_t>(CHUNK_SIZE),
                },
        });

    for (uint32_t c = 0; c < NUM_CORES; ++c) {
        const uint64_t pcie_src = src_phys_base + static_cast<uint64_t>(c) * per_core_bytes;
        const uint64_t done_flag = flags_phys_base + c * sizeof(uint32_t);

        SetRuntimeArgs(
            program,
            kernel,
            CoreCoord(c, 0),
            {
                pcie_xy_enc,
                static_cast<uint32_t>(pcie_src & 0xFFFF'FFFF),
                static_cast<uint32_t>(pcie_src >> 32),
                chunks_per_core,
                dram_base,
                c * chunks_per_core,  // dram_page_start: each core's pages are contiguous
                static_cast<uint32_t>(done_flag & 0xFFFF'FFFF),
                static_cast<uint32_t>(done_flag >> 32),
            });
    }

    auto mesh_workload = MeshWorkload();
    mesh_workload.add_program(MeshCoordinateRange(device_coord), std::move(program));

    // ── 5. Run (possibly multiple times for bandwidth averaging) ─────────────
    double best_bw = 0.0;
    for (int run = 0; run < num_bw_runs; ++run) {
        // Reset done flags before each run.
        std::memset(flags, 0, NUM_CORES * sizeof(uint32_t));

        auto t0 = std::chrono::high_resolution_clock::now();

        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

        // Spin until all cores set their done flag.
        // Volatile ensures compiler re-reads; x86 TSO makes PCIe DMA writes
        // visible without explicit fence (PCIe is cache-coherent on this platform).
        volatile uint32_t* vflags = flags;
        for (uint32_t c = 0; c < NUM_CORES; ++c) {
            while (!vflags[c]) {
                std::atomic_thread_fence(std::memory_order_acquire);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        double bw = static_cast<double>(TOTAL_BYTES) / s / (1024.0 * 1024.0 * 1024.0);
        best_bw = std::max(best_bw, bw);

        if (num_bw_runs > 1) {
            std::cout << "  run " << run << ": " << std::fixed << std::setprecision(2) << bw << " GB/s\n";
        } else {
            print_bw("PCIe->DRAM 4-core grid", TOTAL_BYTES, s);
        }
    }
    if (num_bw_runs > 1) {
        std::cout << "  best: " << std::fixed << std::setprecision(2) << best_bw << " GB/s\n";
    }

    // Finish() ensures DRAM writes complete before ReadShard.
    Finish(mesh_device->mesh_command_queue());

    // ── 6. Correctness check ─────────────────────────────────────────────────
    if (verify_data) {
        std::vector<uint32_t> expected(TOTAL_BYTES / sizeof(uint32_t));
        std::iota(expected.begin(), expected.end(), 0u);

        std::vector<uint32_t> got(TOTAL_BYTES / sizeof(uint32_t));
        ReadShard(mesh_device->mesh_command_queue(), got, dram_buf, device_coord);

        EXPECT_EQ(expected, got);
    }
}

// ---------------------------------------------------------------------------
// GTest
// ---------------------------------------------------------------------------

using PCIeGridFixture = GenericMeshDeviceFixture;

TEST_F(PCIeGridFixture, PCIeGrid_4Core_Correctness) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    run_grid(mesh_device_, /*verify_data=*/true, /*num_bw_runs=*/1);
}

TEST_F(PCIeGridFixture, PCIeGrid_4Core_Bandwidth) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    std::cout << "\n=== PCIe->DRAM 4-Core Grid Bandwidth"
              << " (chunk=" << CHUNK_SIZE / 1024 << " KB"
              << ", total=" << TOTAL_BYTES / 1024 / 1024 << " MB"
              << ", cores=" << NUM_CORES << ") ===\n";
    run_grid(mesh_device_, /*verify_data=*/false, /*num_bw_runs=*/5);
}

}  // namespace tt::tt_metal::distributed
