// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Persistent DMA engine — tests
//
// Test suite:
//   PersistentDmaFixture.D2H_Bandwidth        — D2H BW (440 MB, 5 runs)
//   PersistentDmaFixture.H2D_Bandwidth        — H2D BW (440 MB, 5 runs)
//   PersistentDmaFixture.Bidirectional_BW     — concurrent D2H + H2D
//   PersistentDmaFixture.Latency_D2H          — per-transfer latency sweep
//   PersistentDmaFixture.Latency_H2D          — per-transfer latency sweep
//
// Build & run:
//   ninja -C build_Release distributed_unit_tests
//   ./build_Release/test/tt_metal/distributed/distributed_unit_tests \
//       --gtest_filter="PersistentDmaFixture.*"

#include "dma_engine.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include <tt-metalium/tt_align.hpp>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <thread>
#include <vector>

#include "gmock/gmock.h"

namespace tt::tt_metal::distributed {

// (DmaEngine, pdma_mmap_alloc, and all DMA includes are in dma_engine.hpp)

// ── Test fixture ──────────────────────────────────────────────────────────

class PersistentDmaFixture : public GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        GenericMeshDeviceFixture::SetUp();
        // Probe whether PinnedMemory actually works on this system by trying
        // to create a small one.  GetMemoryPinningParameters().can_map_to_noc
        // is overly conservative — it returns false when IOMMU is disabled,
        // even though Blackhole's 64-bit PCIe addressing works without IOMMU.
        try {
            auto probe = make_pinned(kDmaChunkSize);
            (void)probe;
        } catch (...) {
            GTEST_SKIP() << "PinnedMemory with NOC mapping not available on this system";
        }
        // NOTE: engine_ is NOT created here.  Each test must call start_engine()
        // AFTER any WriteShard()+Finish() calls that initialize DRAM buffers.
        // Calling Finish() while persistent kernels are running on the same CQ
        // would deadlock because those kernels never complete.
    }

    void TearDown() override {
        engine_.reset();  // sends EXIT, waits for kernels (no-op if never started)
        GenericMeshDeviceFixture::TearDown();
    }

    // Call this after all DRAM initialization (WriteShard+Finish) is complete.
    void start_engine() { engine_ = std::make_unique<DmaEngine>(mesh_device_); }

    // Allocate a pinned host buffer and return (ptr, phys_addr, owner)
    struct PinnedBuf {
        std::shared_ptr<uint32_t[]> owner;
        uint32_t* ptr;
        uint64_t phys;
        size_t bytes;
    };

    PinnedBuf make_pinned(size_t bytes) {
        MeshCoordinateRangeSet device_range;
        device_range.merge(MeshCoordinateRange(MeshCoordinate(0, 0)));

        auto [raw, alloc] = pdma_mmap_alloc(bytes);
        auto owner =
            std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(raw), [alloc](uint32_t* p) { munmap(p, alloc); });
        std::memset(raw, 0, bytes);

        HostBuffer hbuf(tt::stl::Span<uint32_t>(owner.get(), bytes / sizeof(uint32_t)), MemoryPin(owner));
        auto pinned = experimental::PinnedMemory::Create(*mesh_device_, device_range, hbuf, /*map_to_noc=*/true);
        auto noc = pinned->get_noc_addr(mesh_device_->get_device(MeshCoordinate(0, 0))->id());
        TT_FATAL(noc.has_value(), "Failed to get NOC address for host buffer");
        // Keep pinned alive by storing it — slightly awkward but correct
        pinned_bufs_.push_back(std::move(pinned));

        return PinnedBuf{owner, owner.get(), noc->addr, bytes};
    }

    // Allocate a DRAM buffer of given size
    std::shared_ptr<MeshBuffer> make_dram(size_t bytes) {
        return MeshBuffer::create(
            ReplicatedBufferConfig{.size = bytes},
            DeviceLocalBufferConfig{
                .page_size = kDmaChunkSize,
                .buffer_type = BufferType::DRAM,
            },
            mesh_device_.get());
    }

    std::unique_ptr<DmaEngine> engine_;
    std::vector<std::shared_ptr<experimental::PinnedMemory>> pinned_bufs_;
};

// ── Helpers ───────────────────────────────────────────────────────────────

// Round total_bytes down to the nearest multiple of kDmaNumCores * kDmaChunkSize
static size_t align_transfer(size_t bytes) {
    const size_t align = kDmaNumCores * kDmaChunkSize;
    return (bytes / align) * align;
}

// Flush a host buffer from all CPU caches (L1/L2/LLC).
//
// Pre-DMA use: evict stale data so DDIO write-allocates fresh LLC lines.
// Post-DMA use: evict any speculatively prefetched stale lines from L1/L2
//   that DDIO (non-inclusive LLC) may not have snooped; forces subsequent
//   reads to LLC/DRAM where the committed DMA data resides.
static void clflush_range(const void* ptr, size_t bytes) {
    const char* p = static_cast<const char*>(ptr);
    for (size_t i = 0; i < bytes; i += 64) {
        _mm_clflush(p + i);
    }
    _mm_sfence();
}

// ── PCIe Write Smoke ──────────────────────────────────────────────────────
//
// Minimal isolated test of noc_wwrite_with_state:
//   1. Allocate one cache line (64 B) of pinned host memory.
//   2. Launch a single BRISC kernel that writes 0xDEADBEEF to it.
//   3. Wait for the kernel to exit (blocking Finish).
//   4. Verify the value.
//
// No command ring, no double-buffering, no DmaEngine.
// If this passes but D2H_Bandwidth fails, the problem is in the transfer
// logic, not the PCIe write primitive.
// If this fails, the basic noc_wwrite_with_state path is broken.

TEST_F(PersistentDmaFixture, PCIe_Write_Smoke) {
    std::cout << "\n=== PCIe Write Smoke Test ===\n";

    constexpr size_t kBufBytes = 64u;  // one cache line
    constexpr uint32_t kMagic = 0xDEADBEEFu;
    const CoreCoord smoke_core(0, 0);

    // Allocate pinned memory
    MeshCoordinateRangeSet device_range;
    device_range.merge(MeshCoordinateRange(MeshCoordinate(0, 0)));

    auto [raw, alloc] = pdma_mmap_alloc(kBufBytes);
    std::shared_ptr<uint32_t[]> owner(static_cast<uint32_t*>(raw), [alloc](uint32_t* p) { munmap(p, alloc); });
    std::memset(raw, 0, alloc);

    HostBuffer hbuf(tt::stl::Span<uint32_t>(owner.get(), alloc / sizeof(uint32_t)), MemoryPin(owner));
    auto pinned = experimental::PinnedMemory::Create(*mesh_device_, device_range, hbuf, /*map_to_noc=*/true);
    auto noc = pinned->get_noc_addr(mesh_device_->get_device(MeshCoordinate(0, 0))->id());
    ASSERT_TRUE(noc.has_value()) << "Failed to get NOC address for smoke-test buffer";

    std::cout << "  IOVA=0x" << std::hex << noc->addr << " pcie_xy_enc=0x" << noc->pcie_xy_enc << std::dec << "\n";

    // Launch one-shot BRISC kernel
    auto program = CreateProgram();
    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/socket/pcie_write_smoke.cpp",
        CoreRangeSet(CoreRange(smoke_core)),
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        });

    SetRuntimeArgs(
        program,
        kernel,
        smoke_core,
        {
            noc->pcie_xy_enc,
            static_cast<uint32_t>(noc->addr & 0xFFFF'FFFFu),
            static_cast<uint32_t>(noc->addr >> 32),
        });

    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(MeshCoordinate(0, 0)), std::move(program));
    // blocking=true: waits for kernel to exit before returning
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);

    // Check pinned memory — the kernel wrote kMagic to all 16 words
    const uint32_t* buf = owner.get();
    std::cout << "  pinned[0]=0x" << std::hex << buf[0] << " (expected 0x" << kMagic << ")\n" << std::dec;
    EXPECT_EQ(buf[0], kMagic) << "noc_wwrite_with_state did not deliver data to host pinned memory\n"
                              << "IOVA=0x" << std::hex << noc->addr << " pcie_xy_enc=0x" << noc->pcie_xy_enc;
}

// ── D2H Single-Chunk ──────────────────────────────────────────────────────
//
// Exercises the persistent D2H kernel with num_chunks=1 per core.
// total_bytes = kDmaNumCores * kDmaChunkSize = 256 KB.
// With num_chunks=1, the double-buffer loop is skipped entirely:
//   kernel path = prime_read → drain_write → done_flag
// This is the minimal "real transfer" test for the persistent engine.

TEST_F(PersistentDmaFixture, D2H_SingleChunk) {
    const size_t total_bytes = kDmaNumCores * kDmaChunkSize;

    std::cout << "\n=== D2H Single-Chunk (" << kDmaNumCores << " cores × 1 chunk × 16 KB =" << total_bytes / 1024
              << " KB) ===\n";

    auto dram = make_dram(total_bytes);
    {
        std::vector<uint32_t> src(total_bytes / sizeof(uint32_t));
        std::iota(src.begin(), src.end(), 0u);
        WriteShard(mesh_device_->mesh_command_queue(), dram, src, MeshCoordinate(0, 0));
        Finish(mesh_device_->mesh_command_queue());
    }

    start_engine();

    // Allocate data + done flags in ONE pinned buffer (same IOMMU mapping).
    // Done flags are cache-line spaced (kFlagRegionBytes from DmaEngine).
    constexpr size_t kFlagRegion = DmaEngine::kFlagRegionBytes;
    auto dst = make_pinned(total_bytes + kFlagRegion);
    const uint64_t flags_phys = dst.phys + total_bytes;
    volatile uint32_t* flags_ptr = dst.ptr + total_bytes / sizeof(uint32_t);
    std::cout << "  dst.phys=0x" << std::hex << dst.phys << " flags_phys=0x" << flags_phys << std::dec
              << " dram_addr=0x" << std::hex << dram->address() << std::dec << "\n";

    auto run_d2h = [&](int run_num, uint32_t batch, const char* label) {
        std::memset(dst.ptr, 0xAA + run_num, total_bytes);
        clflush_range(dst.ptr, total_bytes);

        auto t0 = std::chrono::high_resolution_clock::now();
        engine_->transfer_d2h(
            static_cast<uint32_t>(dram->address()), dst.phys, total_bytes, flags_phys, flags_ptr, batch);
        auto t1 = std::chrono::high_resolution_clock::now();

        for (uint32_t i = 0; i < total_bytes / sizeof(uint32_t); ++i) {
            ASSERT_EQ(dst.ptr[i], i) << "D2H data mismatch (run " << run_num << " " << label << ") at word " << i;
        }
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        const double bw = total_bytes / (us * 1e-6) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  run " << run_num << " PASS (" << label << ")  " << std::fixed << std::setprecision(0) << us
                  << " µs  " << std::setprecision(2) << bw << " GB/s\n";
    };

    // Correctness: 5 serial + 5 parallel (verify data each time)
    std::cout << "  --- serial (batch=1) ---\n";
    for (int r = 1; r <= 5; ++r) {
        std::string lbl = "serial-" + std::to_string(r);
        run_d2h(r, 1, lbl.c_str());
    }
    std::cout << "  --- parallel (batch=0, all 16 cores) ---\n";
    for (int r = 6; r <= 10; ++r) {
        std::string lbl = "parallel-" + std::to_string(r);
        run_d2h(r, 0, lbl.c_str());
    }
}

// ── D2H Bandwidth ─────────────────────────────────────────────────────────

TEST_F(PersistentDmaFixture, D2H_Bandwidth) {
    constexpr size_t kPerCore = 4u * 1024u * 1024u;                      // 4 MB/core
    const size_t total_bytes = align_transfer(kPerCore * kDmaNumCores);  // 64 MB

    std::cout << "\n=== Persistent D2H Bandwidth"
              << " (" << kDmaNumCores << " cores × " << kPerCore / (1024 * 1024)
              << " MB =" << total_bytes / (1024 * 1024) << " MB) ===\n";

    // Source: DRAM buffer filled with sequential data.
    // Initialize BEFORE starting the engine so that Finish() doesn't race
    // with persistent kernels running on the same CQ.
    auto dram = make_dram(total_bytes);
    {
        std::vector<uint32_t> src(total_bytes / sizeof(uint32_t));
        std::iota(src.begin(), src.end(), 0u);
        WriteShard(mesh_device_->mesh_command_queue(), dram, src, MeshCoordinate(0, 0));
        Finish(mesh_device_->mesh_command_queue());
    }

    start_engine();

    // Destination: pinned host buffer (data + done flags in same mapping)
    constexpr size_t kFlagRegion = DmaEngine::kFlagRegionBytes;
    std::cout << "  [D2H_BW] allocating pinned host buffer...\n";
    auto dst_buf = make_pinned(total_bytes + kFlagRegion);
    const uint64_t flags_phys = dst_buf.phys + total_bytes;
    volatile uint32_t* flags_ptr = dst_buf.ptr + total_bytes / sizeof(uint32_t);
    std::cout << "  [D2H_BW] dst_phys=0x" << std::hex << dst_buf.phys << " dram_addr=0x"
              << static_cast<uint32_t>(dram->address()) << std::dec << "\n";

    // Warmup
    std::cout << "  [D2H_BW] starting warmup transfer\n";
    engine_->transfer_d2h(static_cast<uint32_t>(dram->address()), dst_buf.phys, total_bytes, flags_phys, flags_ptr);

    // Timed runs
    constexpr int kRuns = 5;
    double best_bw = 0.0;
    for (int r = 0; r < kRuns; ++r) {
        std::memset(dst_buf.ptr, 0, total_bytes);
        auto t0 = std::chrono::high_resolution_clock::now();
        engine_->transfer_d2h(static_cast<uint32_t>(dram->address()), dst_buf.phys, total_bytes, flags_phys, flags_ptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        const double bw = total_bytes / s / (1024.0 * 1024.0 * 1024.0);
        best_bw = std::max(best_bw, bw);
        std::cout << "  run " << r << ": " << std::fixed << std::setprecision(2) << bw << " GB/s\n";
    }
    std::cout << "  best: " << std::fixed << std::setprecision(2) << best_bw << " GB/s\n";
}

// ── D2H Stress ────────────────────────────────────────────────────────────
//
// Hammers 100 back-to-back D2H transfers with NO gap.  Verifies data on
// every transfer and prints per-transfer bandwidth so you can see if
// performance drops across iterations (thermal, IOMMU, NOC congestion).

TEST_F(PersistentDmaFixture, D2H_Stress) {
    constexpr size_t kPerCore = 4u * 1024u * 1024u;                      // 4 MB/core
    const size_t total_bytes = align_transfer(kPerCore * kDmaNumCores);  // 64 MB
    constexpr int kIters = 100;

    std::cout << "\n=== Persistent D2H Stress Test ==="
              << "\n  " << kDmaNumCores << " cores × " << kPerCore / (1024 * 1024)
              << " MB = " << total_bytes / (1024 * 1024) << " MB per transfer"
              << "\n  " << kIters << " back-to-back transfers, data verified every iteration\n";

    auto dram = make_dram(total_bytes);
    {
        std::vector<uint32_t> src(total_bytes / sizeof(uint32_t));
        std::iota(src.begin(), src.end(), 0u);
        WriteShard(mesh_device_->mesh_command_queue(), dram, src, MeshCoordinate(0, 0));
        Finish(mesh_device_->mesh_command_queue());
    }

    start_engine();

    constexpr size_t kFlagRegion = DmaEngine::kFlagRegionBytes;
    auto dst_buf = make_pinned(total_bytes + kFlagRegion);
    const uint64_t flags_phys = dst_buf.phys + total_bytes;
    volatile uint32_t* flags_ptr = dst_buf.ptr + total_bytes / sizeof(uint32_t);

    // Warmup (1 transfer)
    engine_->transfer_d2h(static_cast<uint32_t>(dram->address()), dst_buf.phys, total_bytes, flags_phys, flags_ptr);

    double min_bw = 1e9, max_bw = 0.0, sum_bw = 0.0;
    int verify_fails = 0;

    auto wall_t0 = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < kIters; ++r) {
        std::memset(dst_buf.ptr, 0xCC, total_bytes);
        clflush_range(dst_buf.ptr, total_bytes);

        auto t0 = std::chrono::high_resolution_clock::now();
        engine_->transfer_d2h(static_cast<uint32_t>(dram->address()), dst_buf.phys, total_bytes, flags_phys, flags_ptr);
        auto t1 = std::chrono::high_resolution_clock::now();

        const double s = std::chrono::duration<double>(t1 - t0).count();
        const double bw = total_bytes / s / (1024.0 * 1024.0 * 1024.0);
        min_bw = std::min(min_bw, bw);
        max_bw = std::max(max_bw, bw);
        sum_bw += bw;

        // Spot-check data (first, middle, last words)
        bool ok = (dst_buf.ptr[0] == 0u) &&
                  (dst_buf.ptr[total_bytes / sizeof(uint32_t) / 2] ==
                   static_cast<uint32_t>(total_bytes / sizeof(uint32_t) / 2)) &&
                  (dst_buf.ptr[total_bytes / sizeof(uint32_t) - 1] ==
                   static_cast<uint32_t>(total_bytes / sizeof(uint32_t) - 1));
        if (!ok) {
            verify_fails++;
        }

        if (r < 10 || r % 10 == 9) {
            std::cout << "  [" << std::setw(3) << r << "] " << std::fixed << std::setprecision(2) << bw << " GB/s"
                      << (ok ? "" : " DATA MISMATCH") << "\n";
        }
    }

    auto wall_t1 = std::chrono::high_resolution_clock::now();
    const double wall_s = std::chrono::duration<double>(wall_t1 - wall_t0).count();
    const double agg_bw = (static_cast<double>(total_bytes) * kIters) / wall_s / (1024.0 * 1024.0 * 1024.0);

    std::cout << "\n  --- Summary (" << kIters << " transfers) ---\n"
              << "  min: " << std::fixed << std::setprecision(2) << min_bw << " GB/s\n"
              << "  max: " << max_bw << " GB/s\n"
              << "  avg: " << (sum_bw / kIters) << " GB/s\n"
              << "  aggregate (incl. memset+clflush): " << agg_bw << " GB/s\n"
              << "  wall time: " << std::setprecision(1) << wall_s << " s\n"
              << "  data errors: " << verify_fails << "/" << kIters << "\n";

    EXPECT_EQ(verify_fails, 0) << "Data verification failed on " << verify_fails << " transfers";
}

// ── H2D Bandwidth ─────────────────────────────────────────────────────────

TEST_F(PersistentDmaFixture, H2D_Bandwidth) {
    constexpr size_t kPerCore = 4u * 1024u * 1024u;
    const size_t total_bytes = align_transfer(kPerCore * kDmaNumCores);

    std::cout << "\n=== Persistent H2D Bandwidth"
              << " (" << kDmaNumCores << " cores × " << kPerCore / (1024 * 1024)
              << " MB =" << total_bytes / (1024 * 1024) << " MB) ===\n";

    // No DRAM WriteShard needed — DRAM is the destination.
    // Start engine before allocating buffers.
    start_engine();

    auto src_buf = make_pinned(total_bytes);
    {
        uint32_t* p = src_buf.ptr;
        for (size_t i = 0; i < total_bytes / sizeof(uint32_t); ++i) {
            p[i] = i;
        }
    }

    auto dram = make_dram(total_bytes);

    // Warmup
    engine_->transfer_h2d(src_buf.phys, static_cast<uint32_t>(dram->address()), total_bytes);

    constexpr int kRuns = 5;
    double best_bw = 0.0;
    for (int r = 0; r < kRuns; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        engine_->transfer_h2d(src_buf.phys, static_cast<uint32_t>(dram->address()), total_bytes);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        const double bw = total_bytes / s / (1024.0 * 1024.0 * 1024.0);
        best_bw = std::max(best_bw, bw);
        std::cout << "  run " << r << ": " << std::fixed << std::setprecision(2) << bw << " GB/s\n";
    }
    std::cout << "  best: " << std::fixed << std::setprecision(2) << best_bw << " GB/s\n";
}

// ── H2D Debug — per-phase timing + kernel cycle breakdown ─────────────────
//
// Runs one H2D Pull and one H2D Push at three sizes (256 KB, 4 MB, 64 MB),
// printing:
//   Host side: cmd_send_us, poll_us, poll_iterations, µs/BAR-read
//   Kernel:    pcie_read_cycles, dram_write_cycles, spin_wait_cycles, batch_count
//              → implied PCIe read BW/core, efficiency breakdown
//
// Also measures raw BAR write bandwidth (bar_memcpy to one core without a
// kernel in the loop) to isolate the push-path bottleneck.
//
// Run:
//   --gtest_filter="PersistentDmaFixture.H2D_Debug"

TEST_F(PersistentDmaFixture, DISABLED_H2D_Debug) { GTEST_SKIP() << "Debug test removed — use DMA_FullSuite instead"; }

// ── Bidirectional Bandwidth ───────────────────────────────────────────────
//
// D2H and H2D run simultaneously on the same 16 cores (BRISC + NCRISC).
// Both threads submit to the DmaEngine concurrently then wait.

TEST_F(PersistentDmaFixture, Bidirectional_BW) {
    constexpr size_t kPerCore = 4u * 1024u * 1024u;
    const size_t total_bytes = align_transfer(kPerCore * kDmaNumCores);

    std::cout << "\n=== Persistent Bidirectional BW (D2H + H2D simultaneously) ===\n";
    std::cout << "  " << total_bytes / (1024 * 1024) << " MB each direction, " << kDmaNumCores << " cores\n";

    // D2H: DRAM → host.  Initialize BEFORE start_engine() to avoid Finish() deadlock.
    auto d2h_src_dram = make_dram(total_bytes);
    {
        std::vector<uint32_t> init(total_bytes / sizeof(uint32_t));
        std::iota(init.begin(), init.end(), 0u);
        WriteShard(mesh_device_->mesh_command_queue(), d2h_src_dram, init, MeshCoordinate(0, 0));
        Finish(mesh_device_->mesh_command_queue());
    }

    start_engine();

    constexpr size_t kFlagRegion = DmaEngine::kFlagRegionBytes;
    auto d2h_dst = make_pinned(total_bytes + kFlagRegion);
    const uint64_t d2h_flags_phys = d2h_dst.phys + total_bytes;
    volatile uint32_t* d2h_flags_ptr = d2h_dst.ptr + total_bytes / sizeof(uint32_t);

    // H2D: host → DRAM
    auto h2d_src = make_pinned(total_bytes);
    {
        uint32_t* p = h2d_src.ptr;
        for (size_t i = 0; i < total_bytes / sizeof(uint32_t); ++i) {
            p[i] = i;
        }
    }
    auto h2d_dst_dram = make_dram(total_bytes);

    // Warmup both directions sequentially
    engine_->transfer_d2h(
        static_cast<uint32_t>(d2h_src_dram->address()), d2h_dst.phys, total_bytes, d2h_flags_phys, d2h_flags_ptr);
    engine_->transfer_h2d(h2d_src.phys, static_cast<uint32_t>(h2d_dst_dram->address()), total_bytes);

    constexpr int kRuns = 3;
    double best_bidir_bw = 0.0;

    for (int r = 0; r < kRuns; ++r) {
        std::memset(d2h_dst.ptr, 0, total_bytes);

        auto t0 = std::chrono::high_resolution_clock::now();

        std::thread d2h_thread([&] {
            engine_->transfer_d2h(
                static_cast<uint32_t>(d2h_src_dram->address()),
                d2h_dst.phys,
                total_bytes,
                d2h_flags_phys,
                d2h_flags_ptr);
        });
        std::thread h2d_thread(
            [&] { engine_->transfer_h2d(h2d_src.phys, static_cast<uint32_t>(h2d_dst_dram->address()), total_bytes); });
        d2h_thread.join();
        h2d_thread.join();

        auto t1 = std::chrono::high_resolution_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        // Bidirectional BW = sum of both directions
        const double bidir_bw = 2.0 * total_bytes / s / (1024.0 * 1024.0 * 1024.0);
        const double each_bw = total_bytes / s / (1024.0 * 1024.0 * 1024.0);
        best_bidir_bw = std::max(best_bidir_bw, bidir_bw);

        std::cout << "  run " << r << ": " << std::fixed << std::setprecision(2) << each_bw << " GB/s each dir,  "
                  << bidir_bw << " GB/s aggregate\n";
    }
    std::cout << "  best aggregate: " << std::fixed << std::setprecision(2) << best_bidir_bw << " GB/s\n";
}

// ── D2H Latency sweep ─────────────────────────────────────────────────────
//
// Measures per-transfer latency for small to large transfers.
// Demonstrates that persistent kernels amortize the fixed overhead much better
// than re-launching (non-persistent fixed cost: ~80 µs; persistent: ~5 µs).

TEST_F(PersistentDmaFixture, Latency_D2H) {
    std::cout << "\n=== Persistent D2H Latency Sweep ===\n";
    std::cout << "  (non-persistent fixed overhead ~80 µs; persistent target < 10 µs)\n\n";

    // Largest size we'll test (must be multiple of kDmaNumCores * kDmaChunkSize)
    const size_t kMaxBytes = align_transfer(64u * 1024u * 1024u);  // 64 MB

    // Fill DRAM BEFORE start_engine() to avoid Finish() deadlock.
    auto dram = make_dram(kMaxBytes);
    {
        std::vector<uint32_t> src(kMaxBytes / sizeof(uint32_t));
        std::iota(src.begin(), src.end(), 0u);
        WriteShard(mesh_device_->mesh_command_queue(), dram, src, MeshCoordinate(0, 0));
        Finish(mesh_device_->mesh_command_queue());
    }

    start_engine();

    constexpr size_t kFlagRegion = DmaEngine::kFlagRegionBytes;
    auto dst_buf = make_pinned(kMaxBytes + kFlagRegion);
    const uint64_t flags_phys = dst_buf.phys + kMaxBytes;
    volatile uint32_t* flags_ptr = dst_buf.ptr + kMaxBytes / sizeof(uint32_t);

    // Warmup with max size
    engine_->transfer_d2h(static_cast<uint32_t>(dram->address()), dst_buf.phys, kMaxBytes, flags_phys, flags_ptr);

    // Sweep transfer sizes
    const std::vector<size_t> sizes = {
        align_transfer(kDmaNumCores * kDmaChunkSize),  //   256 KB (1 chunk/core)
        align_transfer(1u * 1024u * 1024u),            //     1 MB
        align_transfer(4u * 1024u * 1024u),            //     4 MB
        align_transfer(16u * 1024u * 1024u),           //    16 MB
        align_transfer(64u * 1024u * 1024u),           //    64 MB
    };

    std::cout << std::setw(12) << "size" << std::setw(14) << "latency (µs)" << std::setw(14) << "BW (GB/s)" << "\n";
    std::cout << std::string(40, '-') << "\n";

    constexpr int kReps = 10;
    for (size_t sz : sizes) {
        double total_s = 0.0;
        for (int r = 0; r < kReps; ++r) {
            std::memset(dst_buf.ptr, 0, sz);
            auto t0 = std::chrono::high_resolution_clock::now();
            engine_->transfer_d2h(static_cast<uint32_t>(dram->address()), dst_buf.phys, sz, flags_phys, flags_ptr);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_s += std::chrono::duration<double>(t1 - t0).count();
        }
        const double avg_s = total_s / kReps;
        const double avg_us = avg_s * 1e6;
        const double bw = sz / avg_s / (1024.0 * 1024.0 * 1024.0);

        std::cout << std::setw(10) << sz / 1024 << " KB" << std::setw(14) << std::fixed << std::setprecision(1)
                  << avg_us << std::setw(14) << std::fixed << std::setprecision(2) << bw << "\n";
    }
}

// ── H2D Latency sweep ─────────────────────────────────────────────────────

TEST_F(PersistentDmaFixture, Latency_H2D) {
    std::cout << "\n=== Persistent H2D Latency Sweep ===\n\n";

    const size_t kMaxBytes = align_transfer(64u * 1024u * 1024u);

    start_engine();

    auto src_buf = make_pinned(kMaxBytes);
    {
        uint32_t* p = src_buf.ptr;
        for (size_t i = 0; i < kMaxBytes / sizeof(uint32_t); ++i) {
            p[i] = i;
        }
    }
    auto dram = make_dram(kMaxBytes);

    engine_->transfer_h2d(src_buf.phys, static_cast<uint32_t>(dram->address()), kMaxBytes);

    const std::vector<size_t> sizes = {
        align_transfer(kDmaNumCores * kDmaChunkSize),
        align_transfer(1u * 1024u * 1024u),
        align_transfer(4u * 1024u * 1024u),
        align_transfer(16u * 1024u * 1024u),
        align_transfer(64u * 1024u * 1024u),
    };

    std::cout << std::setw(12) << "size" << std::setw(14) << "latency (µs)" << std::setw(14) << "BW (GB/s)" << "\n";
    std::cout << std::string(40, '-') << "\n";

    constexpr int kReps = 10;
    for (size_t sz : sizes) {
        double total_s = 0.0;
        for (int r = 0; r < kReps; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            engine_->transfer_h2d(src_buf.phys, static_cast<uint32_t>(dram->address()), sz);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_s += std::chrono::duration<double>(t1 - t0).count();
        }
        const double avg_s = total_s / kReps;
        const double avg_us = avg_s * 1e6;
        const double bw = sz / avg_s / (1024.0 * 1024.0 * 1024.0);

        std::cout << std::setw(10) << sz / 1024 << " KB" << std::setw(14) << std::fixed << std::setprecision(1)
                  << avg_us << std::setw(14) << std::fixed << std::setprecision(2) << bw << "\n";
    }
}

// ── Comprehensive DMA Test Suite ──────────────────────────────────────────
//
// Sweeps over:
//   Direction:   D2H, H2D
//   Memory path: DRAM (interleaved across banks), L1-only (pure PCIe BW)
//   Buffer size: 256 KB, 1 MB, 4 MB, 16 MB, 64 MB
//   Iterations:  configurable (default 20)
//
// For DRAM paths, data is verified on every iteration.
// For L1-only paths, only bandwidth is measured (data content undefined).
//
// Run:
//   ./build_Release/.../distributed_unit_tests --gtest_filter="*DMA_FullSuite*"

TEST_F(PersistentDmaFixture, DMA_FullSuite) {
    enum class Dir { D2H, H2D_PULL };
    struct TestConfig {
        const char* label;
        Dir dir;
        uint32_t opcode;
        bool verify_data;
    };

    const std::vector<TestConfig> configs = {
        {"D2H_DRAM", Dir::D2H, DMA_OP_TRANSFER, true},
        {"D2H_L1only", Dir::D2H, DMA_OP_L1_ONLY, false},
        {"H2D_Pull_DRAM", Dir::H2D_PULL, DMA_OP_TRANSFER, true},
        {"H2D_Pull_L1", Dir::H2D_PULL, DMA_OP_L1_ONLY, false},
    };

    const std::vector<size_t> raw_sizes = {
        kDmaNumCores * kDmaChunkSize,  //   256 KB
        1u * 1024u * 1024u,            //     1 MB
        4u * 1024u * 1024u,            //     4 MB
        16u * 1024u * 1024u,           //    16 MB
        64u * 1024u * 1024u,           //    64 MB
    };

    constexpr int kIters = 20;

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n"
              << "║           Persistent DMA Full Test Suite                     ║\n"
              << "║  D2H + H2D(pull) × DRAM/L1 × 5 sizes × " << kIters << " iters        ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n\n";

    const size_t kMaxBytes = align_transfer(64u * 1024u * 1024u);

    // Pre-fill DRAM for D2H data verification (must happen before start_engine)
    auto dram_d2h_src = make_dram(kMaxBytes);
    {
        std::vector<uint32_t> init(kMaxBytes / sizeof(uint32_t));
        std::iota(init.begin(), init.end(), 0u);
        WriteShard(mesh_device_->mesh_command_queue(), dram_d2h_src, init, MeshCoordinate(0, 0));
        Finish(mesh_device_->mesh_command_queue());
    }

    start_engine();

    // Allocate pinned buffers for D2H destination + done flags
    constexpr size_t kFlagRegion = DmaEngine::kFlagRegionBytes;
    auto d2h_dst = make_pinned(kMaxBytes + kFlagRegion);
    const uint64_t d2h_flags_phys = d2h_dst.phys + kMaxBytes;
    volatile uint32_t* d2h_flags_ptr = d2h_dst.ptr + kMaxBytes / sizeof(uint32_t);

    // Pinned source for H2D pull path (device reads from host via PCIe)
    auto h2d_pull_src = make_pinned(kMaxBytes);
    std::iota(h2d_pull_src.ptr, h2d_pull_src.ptr + kMaxBytes / sizeof(uint32_t), 0u);

    // DRAM destination for H2D (and readback verification)
    auto dram_h2d_dst = make_dram(kMaxBytes);

    int total_errors = 0;

    for (const auto& cfg : configs) {
        std::cout << "┌─── " << cfg.label << " ─────────────────────────────────\n";
        std::cout << "│  " << std::setw(10) << "size" << std::setw(8) << "iters" << std::setw(12) << "min GB/s"
                  << std::setw(12) << "avg GB/s" << std::setw(12) << "max GB/s" << std::setw(8) << "errors" << "\n";
        std::cout << "│  " << std::string(62, '-') << "\n";

        for (size_t raw_sz : raw_sizes) {
            const size_t sz = align_transfer(raw_sz);
            if (sz == 0) {
                continue;
            }

            double min_bw = 1e9, max_bw = 0.0, sum_bw = 0.0;
            int verify_fails = 0;

            auto run_h2d = [&](size_t xfer_sz) {
                engine_->transfer_h2d(
                    h2d_pull_src.phys, static_cast<uint32_t>(dram_h2d_dst->address()), xfer_sz, cfg.opcode);
            };

            auto run_warmup = [&] {
                if (cfg.dir == Dir::D2H) {
                    std::memset(d2h_dst.ptr, 0, sz);
                    clflush_range(d2h_dst.ptr, sz);
                    engine_->transfer_d2h(
                        static_cast<uint32_t>(dram_d2h_src->address()),
                        d2h_dst.phys,
                        sz,
                        d2h_flags_phys,
                        d2h_flags_ptr,
                        0,
                        cfg.opcode);
                } else {
                    run_h2d(sz);
                }
            };
            run_warmup();

            for (int r = 0; r < kIters; ++r) {
                if (cfg.dir == Dir::D2H) {
                    std::memset(d2h_dst.ptr, 0xCC, sz);
                    clflush_range(d2h_dst.ptr, sz);
                }

                auto t0 = std::chrono::high_resolution_clock::now();

                if (cfg.dir == Dir::D2H) {
                    engine_->transfer_d2h(
                        static_cast<uint32_t>(dram_d2h_src->address()),
                        d2h_dst.phys,
                        sz,
                        d2h_flags_phys,
                        d2h_flags_ptr,
                        0,
                        cfg.opcode);
                } else {
                    run_h2d(sz);
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                const double s = std::chrono::duration<double>(t1 - t0).count();
                const double bw = sz / s / (1024.0 * 1024.0 * 1024.0);
                min_bw = std::min(min_bw, bw);
                max_bw = std::max(max_bw, bw);
                sum_bw += bw;

                if (cfg.verify_data) {
                    if (cfg.dir == Dir::D2H) {
                        const uint32_t nwords = static_cast<uint32_t>(sz / sizeof(uint32_t));
                        bool ok = (d2h_dst.ptr[0] == 0u) && (d2h_dst.ptr[nwords / 2] == nwords / 2) &&
                                  (d2h_dst.ptr[nwords - 1] == nwords - 1);
                        if (!ok) {
                            verify_fails++;
                        }
                    } else {
                        // Round-trip: D2H readback to verify H2D wrote correctly.
                        std::memset(d2h_dst.ptr, 0, sz);
                        clflush_range(d2h_dst.ptr, sz);
                        engine_->transfer_d2h(
                            static_cast<uint32_t>(dram_h2d_dst->address()),
                            d2h_dst.phys,
                            sz,
                            d2h_flags_phys,
                            d2h_flags_ptr);
                        const uint32_t nwords = static_cast<uint32_t>(sz / sizeof(uint32_t));
                        bool ok = (d2h_dst.ptr[0] == 0u) && (d2h_dst.ptr[nwords / 2] == nwords / 2) &&
                                  (d2h_dst.ptr[nwords - 1] == nwords - 1);
                        if (!ok) {
                            verify_fails++;
                        }
                    }
                }
            }

            const char* unit = "KB";
            double display_sz = static_cast<double>(sz) / 1024.0;
            if (display_sz >= 1024.0) {
                display_sz /= 1024.0;
                unit = "MB";
            }

            std::cout << "│  " << std::setw(7) << std::fixed << std::setprecision(0) << display_sz << " " << unit
                      << std::setw(8) << kIters << std::setw(12) << std::setprecision(2) << min_bw << std::setw(12)
                      << (sum_bw / kIters) << std::setw(12) << max_bw << std::setw(8) << verify_fails << "\n";

            total_errors += verify_fails;
        }
        std::cout << "+-" << std::string(65, '-') << "\n\n";
    }

    // ── Bidirectional: D2H + H2D Pull running simultaneously ──────────────
    // PCIe Gen4 x16 is full-duplex: posted writes (D2H) and non-posted reads
    // (H2D Pull) travel on independent channels.  BRISC handles D2H, NCRISC
    // handles H2D Pull — both on the same 16 cores but different processors.
    {
        std::cout << "┌─── Bidir_DRAM (D2H + H2D_Pull simultaneous) ─────────────\n";
        std::cout << "│  " << std::setw(10) << "size" << std::setw(8) << "iters" << std::setw(12) << "D2H GB/s"
                  << std::setw(12) << "H2D GB/s" << std::setw(12) << "Total GB/s" << "\n";
        std::cout << "│  " << std::string(62, '-') << "\n";

        for (size_t raw_sz : raw_sizes) {
            const size_t sz = align_transfer(raw_sz);
            if (sz == 0) {
                continue;
            }

            double sum_d2h = 0.0, sum_h2d = 0.0;

            // Warmup
            std::memset(d2h_dst.ptr, 0, sz);
            clflush_range(d2h_dst.ptr, sz);
            engine_->transfer_d2h(
                static_cast<uint32_t>(dram_d2h_src->address()),
                d2h_dst.phys,
                sz,
                d2h_flags_phys,
                d2h_flags_ptr,
                0,
                DMA_OP_TRANSFER);
            engine_->transfer_h2d(
                h2d_pull_src.phys, static_cast<uint32_t>(dram_h2d_dst->address()), sz, DMA_OP_TRANSFER);

            for (int r = 0; r < kIters; ++r) {
                std::memset(d2h_dst.ptr, 0xCC, sz);
                clflush_range(d2h_dst.ptr, sz);

                std::atomic<double> d2h_bw{0.0}, h2d_bw{0.0};

                auto t0 = std::chrono::high_resolution_clock::now();

                std::thread h2d_thread([&] {
                    auto th0 = std::chrono::high_resolution_clock::now();
                    engine_->transfer_h2d(
                        h2d_pull_src.phys, static_cast<uint32_t>(dram_h2d_dst->address()), sz, DMA_OP_TRANSFER);
                    auto th1 = std::chrono::high_resolution_clock::now();
                    double s = std::chrono::duration<double>(th1 - th0).count();
                    h2d_bw.store(sz / s / (1024.0 * 1024.0 * 1024.0), std::memory_order_relaxed);
                });

                engine_->transfer_d2h(
                    static_cast<uint32_t>(dram_d2h_src->address()),
                    d2h_dst.phys,
                    sz,
                    d2h_flags_phys,
                    d2h_flags_ptr,
                    0,
                    DMA_OP_TRANSFER);
                auto t1 = std::chrono::high_resolution_clock::now();
                double d2h_s = std::chrono::duration<double>(t1 - t0).count();
                d2h_bw.store(sz / d2h_s / (1024.0 * 1024.0 * 1024.0), std::memory_order_relaxed);

                h2d_thread.join();

                sum_d2h += d2h_bw.load(std::memory_order_relaxed);
                sum_h2d += h2d_bw.load(std::memory_order_relaxed);
            }

            const char* unit = "KB";
            double display_sz = static_cast<double>(sz) / 1024.0;
            if (display_sz >= 1024.0) {
                display_sz /= 1024.0;
                unit = "MB";
            }

            const double avg_d2h = sum_d2h / kIters;
            const double avg_h2d = sum_h2d / kIters;
            std::cout << "│  " << std::setw(7) << std::fixed << std::setprecision(0) << display_sz << " " << unit
                      << std::setw(8) << kIters << std::setw(12) << std::setprecision(2) << avg_d2h << std::setw(12)
                      << avg_h2d << std::setw(12) << (avg_d2h + avg_h2d) << "\n";
        }
        std::cout << "+-" << std::string(65, '-') << "\n\n";
    }

    std::cout << "══ Total data verification errors: " << total_errors << " ══\n";
    EXPECT_EQ(total_errors, 0) << "Data verification failed in DMA_FullSuite";
}

}  // namespace tt::tt_metal::distributed
