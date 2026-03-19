// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase 3 test: full Tensix grid PCIe → DRAM
//
// Scales to ALL available Tensix cores on the chip (queried at runtime via
// compute_with_storage_grid_size()).  On this P150/BH with harvesting mask
//   tensix: 0xc0 (rows 6 & 7 harvested), pcie: 0x2 (PCIe port 1 harvested)
// that is a 14×8 = 112-core grid, all routing through the single surviving
// PCIe tile at NOC0 (2,0).
//
// Phase summary:
//   Phase 1:  1-core, socket flow-control         → ~7.7 GB/s
//   Phase 2:  4-core, no socket, done-flags        → ~19.9 GB/s
//   Phase 3: all-core (~112), no socket, done-flags → target: ~32 GB/s
//             (bandwidth-delay product at 32 GB/s × 500 ns = 16 KB;
//              112 × 16 KB = 1.75 MB >> BDP → PCIe should saturate)
//
// Note on dual-PCIe: if both PCIe tiles were available (pcie: 0x0), the
// optimal split would be logical X ≤ 6 → PCIe(2,0), X ≥ 7 → PCIe(11,0),
// with two separate PinnedMemory source buffers and two pcie_xy_enc values.
// That optimisation is deferred to Phase 4.
//
// Build & run:
//   ninja -C build_Release distributed_unit_tests
//   ./build_Release/test/tt_metal/distributed/distributed_unit_tests \
//       --gtest_filter="FullGridFixture.*"

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

// MAP_HUGE_* may not be defined in older glibc <sys/mman.h>; derive from the
// kernel's MAP_HUGE_SHIFT encoding (log2 of page size in bits 26..31 of flags).
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif
#include <vector>

#include "gmock/gmock.h"

namespace tt::tt_metal::distributed {

// ---------------------------------------------------------------------------
// Helpers (fg_ prefix avoids ODR violations with other test TUs in the same
// distributed_unit_tests binary)
// ---------------------------------------------------------------------------

// Try hugepages in order of preference: 1 GB → 2 MB → 4 KB.
// This system has 8 × 1 GB hugepages pre-allocated, so the 440 MB source
// buffer will map as a single IOMMU TLB entry, eliminating TLB-miss overhead.
static std::pair<void*, size_t> fg_mmap_alloc(size_t bytes) {
    // 1 GB hugepages
    {
        constexpr size_t kGB = 1ull << 30;
        const size_t alloc = (bytes + kGB - 1) & ~(kGB - 1);
        void* p = mmap(
            nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
        if (p != MAP_FAILED) {
            return {p, alloc};
        }
    }
    // 2 MB hugepages
    {
        constexpr size_t k2MB = 2ull << 20;
        const size_t alloc = (bytes + k2MB - 1) & ~(k2MB - 1);
        void* p = mmap(
            nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            return {p, alloc};
        }
    }
    // 4 KB fallback
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t alloc = (bytes + page - 1) & ~(page - 1);
    void* p = mmap(nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TT_FATAL(p != MAP_FAILED, "mmap failed for {} bytes", alloc);
    return {p, alloc};
}

// ---------------------------------------------------------------------------
// Core test driver
// ---------------------------------------------------------------------------

// run_full_grid() allocates buffers once and optionally loops for BW
// measurement.  chunk_size is a kernel compile-time arg; different values
// produce different kernel binaries (Metal caches them).
static void run_full_grid(
    const std::shared_ptr<MeshDevice>& mesh_device,
    uint32_t chunk_size,
    uint32_t per_core_chunks,
    bool verify_data,
    int num_bw_runs = 1) {
    const auto device_coord = MeshCoordinate(0, 0);
    const auto device_id = mesh_device->get_device(device_coord)->id();

    // ── Grid dimensions (post-harvesting) ────────────────────────────────
    const auto grid = mesh_device->get_device(device_coord)->compute_with_storage_grid_size();
    const uint32_t ncols = static_cast<uint32_t>(grid.x);
    const uint32_t nrows = static_cast<uint32_t>(grid.y);
    const uint32_t ncores = ncols * nrows;

    const size_t per_core_bytes = static_cast<size_t>(per_core_chunks) * chunk_size;
    const size_t total_bytes = static_cast<size_t>(ncores) * per_core_bytes;

    std::cout << "  grid=" << ncols << "×" << nrows << " (" << ncores << " cores)"
              << "  chunk=" << chunk_size / 1024 << " KB"
              << "  per-core=" << per_core_bytes / 1024 << " KB"
              << "  total=" << total_bytes / (1024 * 1024) << " MB\n";

    MeshCoordinateRangeSet device_range;
    device_range.merge(MeshCoordinateRange(device_coord));

    const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));

    // ── 1. Source buffer (pre-filled; not modified between BW runs) ──────
    auto [src_raw, src_alloc] = fg_mmap_alloc(total_bytes);
    auto src_owner = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(src_raw), [src_alloc](uint32_t* p) { munmap(p, src_alloc); });
    uint32_t* src = src_owner.get();
    std::iota(src, src + total_bytes / sizeof(uint32_t), 0u);

    HostBuffer src_hbuf(tt::stl::Span<uint32_t>(src, total_bytes / sizeof(uint32_t)), MemoryPin(src_owner));
    auto src_pinned = experimental::PinnedMemory::Create(*mesh_device, device_range, src_hbuf, /*map_to_noc=*/true);
    auto src_noc = src_pinned->get_noc_addr(device_id);
    TT_FATAL(src_noc.has_value(), "Failed to get NOC address for source buffer");

    const uint32_t pcie_xy_enc = src_noc->pcie_xy_enc;
    const uint64_t src_phys_base = src_noc->addr;

    std::cout << "  pcie_xy_enc=0x" << std::hex << pcie_xy_enc << std::dec << "\n";

    // ── 2. Done-flag buffer (one uint32_t per core, page-aligned) ────────
    // Rounded up to the nearest page so the IOMMU region is clean.
    const size_t flags_bytes = ((ncores * sizeof(uint32_t) + page_sz - 1) / page_sz) * page_sz;
    auto [flags_raw, flags_alloc] = fg_mmap_alloc(flags_bytes);
    auto flags_owner = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(flags_raw), [flags_alloc](uint32_t* p) { munmap(p, flags_alloc); });
    uint32_t* flags = flags_owner.get();
    std::memset(flags, 0, flags_bytes);

    HostBuffer flags_hbuf(tt::stl::Span<uint32_t>(flags, flags_bytes / sizeof(uint32_t)), MemoryPin(flags_owner));
    auto flags_pinned = experimental::PinnedMemory::Create(*mesh_device, device_range, flags_hbuf, /*map_to_noc=*/true);
    auto flags_noc = flags_pinned->get_noc_addr(device_id);
    TT_FATAL(flags_noc.has_value(), "Failed to get NOC address for done-flags buffer");
    const uint64_t flags_phys_base = flags_noc->addr;

    // ── 3. DRAM destination buffer ───────────────────────────────────────
    auto dram_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{
            .page_size = chunk_size,
            .buffer_type = BufferType::DRAM,
        },
        mesh_device.get());
    const uint32_t dram_base = static_cast<uint32_t>(dram_buf->address());

    // ── 4. Build program for the full grid ───────────────────────────────
    // L1 ping-pong buffers: placed at 512 KB (safely above firmware/dispatch
    // regions).  buf_b follows immediately after buf_a.
    const uint32_t l1_buf_a = 512u * 1024u;
    const uint32_t l1_buf_b = l1_buf_a + chunk_size;

    CoreRange cores(CoreCoord(0, 0), CoreCoord(ncols - 1, nrows - 1));
    auto program = CreateProgram();
    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_to_dram_grid_puller.cpp",
        CoreRangeSet(cores),
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {l1_buf_a, l1_buf_b, chunk_size},
        });

    // Per-core runtime args: each core gets its own slice of [src, flags, DRAM].
    // Linear index c = y * ncols + x assigns contiguous page ranges to each core.
    for (uint32_t y = 0; y < nrows; ++y) {
        for (uint32_t x = 0; x < ncols; ++x) {
            const uint32_t c = y * ncols + x;
            const uint64_t pcie_src = src_phys_base + static_cast<uint64_t>(c) * per_core_bytes;
            const uint64_t done_flag = flags_phys_base + static_cast<uint64_t>(c) * sizeof(uint32_t);

            SetRuntimeArgs(
                program,
                kernel,
                CoreCoord(x, y),
                {
                    pcie_xy_enc,
                    static_cast<uint32_t>(pcie_src & 0xFFFF'FFFF),
                    static_cast<uint32_t>(pcie_src >> 32),
                    per_core_chunks,  // num_chunks for this core
                    dram_base,
                    c * per_core_chunks,  // dram_page_start
                    static_cast<uint32_t>(done_flag & 0xFFFF'FFFF),
                    static_cast<uint32_t>(done_flag >> 32),
                });
        }
    }

    auto mesh_workload = MeshWorkload();
    mesh_workload.add_program(MeshCoordinateRange(device_coord), std::move(program));

    // ── 5. Warmup: one untimed dispatch to upload kernel binary to all cores
    //    and open DRAM row buffers.  Without this the first measured run is
    //    ~30% slower due to per-core instruction-memory upload over the NOC.
    {
        std::memset(flags, 0, ncores * sizeof(uint32_t));
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);
        volatile uint32_t* vf = flags;
        for (uint32_t c = 0; c < ncores; ++c) {
            while (!vf[c]) {
                std::atomic_thread_fence(std::memory_order_acquire);
            }
        }
    }

    // ── 6. Run (optionally multiple times for BW averaging) ──────────────
    double best_bw = 0.0;
    for (int run = 0; run < num_bw_runs; ++run) {
        // Reset done flags before each run (source data is unchanged).
        std::memset(flags, 0, ncores * sizeof(uint32_t));

        auto t0 = std::chrono::high_resolution_clock::now();

        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

        // Spin until every core has written its done flag via PCIe NOC write.
        // volatile + acquire fence is sufficient on x86 with coherent IOMMU PCIe.
        volatile uint32_t* vflags = flags;
        for (uint32_t c = 0; c < ncores; ++c) {
            while (!vflags[c]) {
                std::atomic_thread_fence(std::memory_order_acquire);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        const double bw = static_cast<double>(total_bytes) / s / (1024.0 * 1024.0 * 1024.0);
        best_bw = std::max(best_bw, bw);

        if (num_bw_runs > 1) {
            std::cout << "  run " << run << ": " << std::fixed << std::setprecision(2) << bw << " GB/s\n";
        } else {
            std::cout << "  BW: " << std::fixed << std::setprecision(2) << bw << " GB/s  ("
                      << total_bytes / (1024 * 1024) << " MB in " << std::setprecision(1) << s * 1000.0 << " ms)\n";
        }
    }
    if (num_bw_runs > 1) {
        std::cout << "  best: " << std::fixed << std::setprecision(2) << best_bw << " GB/s\n";
    }

    // Finish() ensures all DRAM writes complete before ReadShard.
    Finish(mesh_device->mesh_command_queue());

    // ── 7. Correctness check ─────────────────────────────────────────────
    if (verify_data) {
        std::vector<uint32_t> expected(total_bytes / sizeof(uint32_t));
        std::iota(expected.begin(), expected.end(), 0u);

        std::vector<uint32_t> got(total_bytes / sizeof(uint32_t));
        ReadShard(mesh_device->mesh_command_queue(), got, dram_buf, device_coord);

        EXPECT_EQ(expected, got);
    }
}

// ---------------------------------------------------------------------------
// GTest
// ---------------------------------------------------------------------------

using FullGridFixture = GenericMeshDeviceFixture;

// Correctness: small per-core allocation (64 KB × ncores) for fast verification.
TEST_F(FullGridFixture, FullGrid_Correctness) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    std::cout << "\n=== PCIe->DRAM Full Grid Correctness (chunk=16 KB, per-core=64 KB) ===\n";
    run_full_grid(
        mesh_device_,
        /*chunk_size=*/16 * 1024,
        /*per_core_chunks=*/4,
        /*verify_data=*/true,
        /*num_bw_runs=*/1);
}

// Bandwidth: 5 runs × all cores × 4 MB/core = ~448 MB per run.
// Expected steady-state: depends on core count and PCIe tile availability.
// With 112 cores all routing through one PCIe tile: ~32 GB/s theoretical max.
TEST_F(FullGridFixture, FullGrid_Bandwidth) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    std::cout << "\n=== PCIe->DRAM Full Grid Bandwidth (chunk=16 KB, per-core=4 MB) ===\n";
    run_full_grid(
        mesh_device_,
        /*chunk_size=*/16 * 1024,
        /*per_core_chunks=*/256,
        /*verify_data=*/false,
        /*num_bw_runs=*/5);
}

// Chunk size sweep: tests 4 KB / 8 KB / 16 KB with constant 4 MB per core.
// 16 KB = NOC_MAX_BURST_SIZE and should be optimal for PCIe RTT hiding.
// Smaller chunks probe whether shorter bursts help or hurt.
TEST_F(FullGridFixture, FullGrid_ChunkSweep) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    std::cout << "\n=== PCIe->DRAM Full Grid Chunk Sweep (per-core=4 MB, 3 runs each) ===\n";
    // Keep per-core data constant at 4 MB across all chunk sizes so the
    // per-run transfer volume stays comparable.
    for (const uint32_t csz : {4u * 1024u, 8u * 1024u, 16u * 1024u}) {
        const uint32_t pcc = (4u * 1024u * 1024u) / csz;
        std::cout << "\n-- chunk=" << csz / 1024 << " KB --\n";
        run_full_grid(mesh_device_, csz, pcc, /*verify_data=*/false, /*num_bw_runs=*/3);
    }
}

}  // namespace tt::tt_metal::distributed
