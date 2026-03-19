// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// D2H full-grid test: DRAM → full Tensix grid → host pinned memory
//
// Mirror of test_pcie_to_dram_full_grid.cpp for the device-to-host direction.
// Uses dram_to_pcie_grid_pusher.cpp on every available Tensix core.
//
// Each core:
//   1. Reads its DRAM slice into L1 (double-buffered, noc_async_read)
//   2. Writes L1 chunks to host pinned memory via PCIe NOC writes
//      (noc_wwrite_with_state — posted TLPs, no completion round-trip)
//   3. Signals done by writing 1 to a per-core flag in a second pinned region
//
// D2H write advantage over H2D reads:
//   PCIe writes are "posted" (fire-and-forget, no completion TLP), so each
//   core's PCIe write latency is lower than the read round-trip.  The actual
//   bottleneck is the single surviving PCIe tile (port 1 harvested on this BH).
//   TTNN's from_device() achieved only ~1.4 GB/s (host-initiated DMA reads);
//   device-initiated PCIe writes should reach the same ~20–32 GB/s ceiling as H2D.
//
// Phase summary (same chip, BH with pcie: 0x2):
//   TTNN from_device():      ~1.4 GB/s  (host-side DMA reads, no parallelism)
//   D2H full-grid (target): ~20–32 GB/s (device-side PCIe writes, all cores)
//
// Build & run:
//   ninja -C build_Release distributed_unit_tests
//   ./build_Release/test/tt_metal/distributed/distributed_unit_tests \
//       --gtest_filter="D2HFullGridFixture.*"

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
// Helpers (d2h_ prefix avoids ODR violations with other TUs in the binary)
// ---------------------------------------------------------------------------

// Try hugepages in order of preference: 1 GB → 2 MB → 4 KB.
static std::pair<void*, size_t> d2h_mmap_alloc(size_t bytes) {
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

static void run_d2h_full_grid(
    const std::shared_ptr<MeshDevice>& mesh_device,
    uint32_t chunk_size,
    uint32_t per_core_chunks,
    bool verify_data,
    int num_bw_runs = 1,
    size_t target_total_bytes = 0) {
    const auto device_coord = MeshCoordinate(0, 0);
    const auto device_id = mesh_device->get_device(device_coord)->id();

    // ── Grid dimensions (post-harvesting) ────────────────────────────────
    const auto grid = mesh_device->get_device(device_coord)->compute_with_storage_grid_size();
    const uint32_t ncols = static_cast<uint32_t>(grid.x);
    const uint32_t nrows = static_cast<uint32_t>(grid.y);
    const uint32_t ncores = ncols * nrows;

    // Override per_core_chunks if a total target size was requested.
    if (target_total_bytes > 0) {
        per_core_chunks = static_cast<uint32_t>((target_total_bytes / ncores) / chunk_size);
        TT_FATAL(
            per_core_chunks > 0, "target_total_bytes {} too small for grid {}x{}", target_total_bytes, ncols, nrows);
    }

    const size_t per_core_bytes = static_cast<size_t>(per_core_chunks) * chunk_size;
    const size_t total_bytes = static_cast<size_t>(ncores) * per_core_bytes;

    std::cout << "  grid=" << ncols << "×" << nrows << " (" << ncores << " cores)"
              << "  chunk=" << chunk_size / 1024 << " KB"
              << "  per-core=" << per_core_bytes / 1024 << " KB"
              << "  total=" << total_bytes / (1024 * 1024) << " MB\n";

    MeshCoordinateRangeSet device_range;
    device_range.merge(MeshCoordinateRange(device_coord));

    const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));

    // ── 1. DRAM source buffer — fill once with sequential data ───────────
    // WriteShard transfers the host vector into the device DRAM buffer.
    auto dram_buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = total_bytes},
        DeviceLocalBufferConfig{
            .page_size = chunk_size,
            .buffer_type = BufferType::DRAM,
        },
        mesh_device.get());
    const uint32_t dram_base = static_cast<uint32_t>(dram_buf->address());

    {
        std::vector<uint32_t> src_data(total_bytes / sizeof(uint32_t));
        std::iota(src_data.begin(), src_data.end(), 0u);
        WriteShard(mesh_device->mesh_command_queue(), dram_buf, src_data, device_coord);
        Finish(mesh_device->mesh_command_queue());
    }

    // ── 2. Destination pinned buffer (device writes to this) ─────────────
    auto [dst_raw, dst_alloc] = d2h_mmap_alloc(total_bytes);
    auto dst_owner = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(dst_raw), [dst_alloc](uint32_t* p) { munmap(p, dst_alloc); });
    uint32_t* dst = dst_owner.get();
    std::memset(dst, 0, total_bytes);

    HostBuffer dst_hbuf(tt::stl::Span<uint32_t>(dst, total_bytes / sizeof(uint32_t)), MemoryPin(dst_owner));
    auto dst_pinned = experimental::PinnedMemory::Create(*mesh_device, device_range, dst_hbuf, /*map_to_noc=*/true);
    auto dst_noc = dst_pinned->get_noc_addr(device_id);
    TT_FATAL(dst_noc.has_value(), "Failed to get NOC address for destination buffer");

    const uint32_t pcie_xy_enc = dst_noc->pcie_xy_enc;
    const uint64_t dst_phys_base = dst_noc->addr;

    std::cout << "  pcie_xy_enc=0x" << std::hex << pcie_xy_enc << std::dec << "\n";

    // ── 3. Done-flag buffer (one uint32_t per core, page-aligned) ─────────
    const size_t flags_bytes = ((ncores * sizeof(uint32_t) + page_sz - 1) / page_sz) * page_sz;
    auto [flags_raw, flags_alloc] = d2h_mmap_alloc(flags_bytes);
    auto flags_owner = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(flags_raw), [flags_alloc](uint32_t* p) { munmap(p, flags_alloc); });
    uint32_t* flags = flags_owner.get();
    std::memset(flags, 0, flags_bytes);

    HostBuffer flags_hbuf(tt::stl::Span<uint32_t>(flags, flags_bytes / sizeof(uint32_t)), MemoryPin(flags_owner));
    auto flags_pinned = experimental::PinnedMemory::Create(*mesh_device, device_range, flags_hbuf, /*map_to_noc=*/true);
    auto flags_noc = flags_pinned->get_noc_addr(device_id);
    TT_FATAL(flags_noc.has_value(), "Failed to get NOC address for done-flags buffer");
    const uint64_t flags_phys_base = flags_noc->addr;

    // ── 4. Build program for the full grid ────────────────────────────────
    // L1 ping-pong buffers at 512 KB (safely above firmware/dispatch regions).
    const uint32_t l1_buf_a = 512u * 1024u;
    const uint32_t l1_buf_b = l1_buf_a + chunk_size;

    CoreRange cores(CoreCoord(0, 0), CoreCoord(ncols - 1, nrows - 1));
    auto program = CreateProgram();
    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/dram_to_pcie_grid_pusher.cpp",
        CoreRangeSet(cores),
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {l1_buf_a, l1_buf_b, chunk_size},
        });

    // Per-core runtime args.
    // Linear index c = y * ncols + x assigns contiguous DRAM page ranges,
    // and contiguous destination windows in the host pinned buffer.
    for (uint32_t y = 0; y < nrows; ++y) {
        for (uint32_t x = 0; x < ncols; ++x) {
            const uint32_t c = y * ncols + x;
            const uint64_t pcie_dst = dst_phys_base + static_cast<uint64_t>(c) * per_core_bytes;
            const uint64_t done_flag = flags_phys_base + static_cast<uint64_t>(c) * sizeof(uint32_t);

            SetRuntimeArgs(
                program,
                kernel,
                CoreCoord(x, y),
                {
                    pcie_xy_enc,
                    static_cast<uint32_t>(pcie_dst & 0xFFFF'FFFF),
                    static_cast<uint32_t>(pcie_dst >> 32),
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

    // ── 5. Warmup: one untimed dispatch to upload kernel binary to all cores.
    //    Without this, the first timed run is ~30% slower due to per-core
    //    instruction-memory upload over the NOC.
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

    // ── 6. Timed BW runs ─────────────────────────────────────────────────
    double best_bw = 0.0;
    for (int run = 0; run < num_bw_runs; ++run) {
        // Reset done flags; destination data is overwritten each run so no reset needed.
        std::memset(flags, 0, ncores * sizeof(uint32_t));

        auto t0 = std::chrono::high_resolution_clock::now();

        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, false);

        // Spin until every core has written its done flag via PCIe NOC write.
        // PCIe spec §2.4.1: posted writes from the same requester are strongly
        // ordered, so observing done_flag==1 guarantees all preceding data writes
        // have been committed to host memory.
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

    // ── 7. Correctness check ─────────────────────────────────────────────
    if (verify_data) {
        std::vector<uint32_t> expected(total_bytes / sizeof(uint32_t));
        std::iota(expected.begin(), expected.end(), 0u);

        // dst[] was written by device PCIe NOC writes; compare directly.
        const uint32_t* got = dst;
        EXPECT_EQ(std::vector<uint32_t>(got, got + total_bytes / sizeof(uint32_t)), expected);
    }
}

// ---------------------------------------------------------------------------
// GTest
// ---------------------------------------------------------------------------

using D2HFullGridFixture = GenericMeshDeviceFixture;

// Correctness: small per-core allocation (64 KB × ncores) for fast verification.
TEST_F(D2HFullGridFixture, D2HFullGrid_Correctness) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    std::cout << "\n=== DRAM->PCIe Full Grid Correctness (chunk=16 KB, per-core=64 KB) ===\n";
    run_d2h_full_grid(
        mesh_device_,
        /*chunk_size=*/16 * 1024,
        /*per_core_chunks=*/4,
        /*verify_data=*/true,
        /*num_bw_runs=*/1);
}

// Bandwidth: 5 runs × all cores × 4 MB/core.
// Baseline: TTNN from_device() ~1.4 GB/s; expected: ~20–32 GB/s.
TEST_F(D2HFullGridFixture, D2HFullGrid_Bandwidth) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    std::cout << "\n=== DRAM->PCIe Full Grid Bandwidth (chunk=16 KB, per-core=4 MB) ===\n";
    std::cout << "  (TTNN from_device() baseline: ~1.4 GB/s)\n";
    run_d2h_full_grid(
        mesh_device_,
        /*chunk_size=*/16 * 1024,
        /*per_core_chunks=*/256,
        /*verify_data=*/false,
        /*num_bw_runs=*/5);
}

// Chunk size sweep: 4 KB / 8 KB / 16 KB at constant 4 MB per core.
// PCIe writes are posted so smaller chunks have lower per-chunk overhead
// than H2D reads; 16 KB is still expected to be optimal.
TEST_F(D2HFullGridFixture, D2HFullGrid_ChunkSweep) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    std::cout << "\n=== DRAM->PCIe Full Grid Chunk Sweep (per-core=4 MB, 3 runs each) ===\n";
    for (const uint32_t csz : {4u * 1024u, 8u * 1024u, 16u * 1024u}) {
        const uint32_t pcc = (4u * 1024u * 1024u) / csz;
        std::cout << "\n-- chunk=" << csz / 1024 << " KB --\n";
        run_d2h_full_grid(mesh_device_, csz, pcc, /*verify_data=*/false, /*num_bw_runs=*/3);
    }
}

// Large transfer: ~30 GB spread across all DRAM banks and all Tensix cores.
// per_core_chunks is derived at runtime from the grid size so the total stays
// close to 30 GB regardless of harvesting.
//
// Requirements:
//   • Sufficient DRAM capacity (BH has 8 channels; confirm ≥ 30 GB available).
//   • ~30 GB of host hugepage memory mappable through the IOMMU for NOC writes.
//   • Run with enough hugepages reserved:
//       echo 30 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
TEST_F(D2HFullGridFixture, D2HFullGrid_30GB) {
    if (!experimental::GetMemoryPinningParameters(*mesh_device_).can_map_to_noc) {
        GTEST_SKIP() << "Host memory pinning to NOC not supported on this system";
    }
    constexpr size_t kTarget = 30ULL * 1024 * 1024 * 1024;
    std::cout << "\n=== DRAM->PCIe Full Grid ~30 GB (chunk=16 KB, 3 runs) ===\n";
    run_d2h_full_grid(
        mesh_device_,
        /*chunk_size=*/16 * 1024,
        /*per_core_chunks=*/0,
        /*verify_data=*/false,
        /*num_bw_runs=*/3,
        /*target_total_bytes=*/kTarget);
}

}  // namespace tt::tt_metal::distributed
