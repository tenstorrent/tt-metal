// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// dma_engine.hpp — DmaEngine class + hugepage allocator for persistent DMA.
//
// Included by both test_persistent_dma.cpp (gtest fixture) and
// persistent_dma_bench.cpp (standalone CLI benchmark).

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>  // full Cluster definition for get_static_tlb_writer etc.
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/tlb.hpp>
#include "tt_metal/tt_metal/test_kernels/misc/socket/persistent_dma_cmd.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <emmintrin.h>  // _mm_stream_si128, _mm_sfence
#include <immintrin.h>  // _mm_clflush
#include <iostream>
#include <numeric>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace tt::tt_metal::distributed {

// ── Hugepage allocator ─────────────────────────────────────────────────────

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

static std::pair<void*, size_t> pdma_mmap_alloc(size_t bytes) {
    constexpr size_t kGB = 1ull << 30;
    constexpr size_t k2MB = 2ull << 20;
    {
        const size_t alloc = (bytes + kGB - 1) & ~(kGB - 1);
        void* p = mmap(
            nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
        if (p != MAP_FAILED) {
            return {p, alloc};
        }
    }
    {
        const size_t alloc = (bytes + k2MB - 1) & ~(k2MB - 1);
        void* p = mmap(
            nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            return {p, alloc};
        }
    }
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t alloc = (bytes + page - 1) & ~(page - 1);
    void* p = mmap(nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TT_FATAL(p != MAP_FAILED, "mmap failed for {} bytes", alloc);
    return {p, alloc};
}

// ── DmaEngineConfig ────────────────────────────────────────────────────────
//
// All parameters that control the DMA engine's resource usage.  Every field
// has a safe default matching the original hardcoded values.
//
// chunk_size   — size of each NOC transfer unit in bytes.
//               Must be a power of 2 and ≤ 16 KB.  The NOC max burst for PCIe
//               *reads* (noc_read_with_state, used by the H2D kernel) on
//               Blackhole is 16 KB; exceeding it hangs the read barrier.
//               Smaller = lower per-transfer granularity; larger = fewer TLPs.
//
// h2d_num_bufs — number of L1 buffers in the H2D pool.  More buffers keep
//               more PCIe read requests in flight, improving H2D bandwidth.
//               Constraint: h2d_buf_start + h2d_num_bufs × chunk_size ≤ ~1 MB.
//
// ring_depth   — number of command slots in each direction's L1 ring.
//               Must be a power of 2.  The host currently issues one command
//               at a time, so 4 is always sufficient.

struct DmaEngineConfig {
    uint32_t num_cores = kDmaNumCores;
    uint32_t chunk_size = 16u * 1024u;  // 16 KB default
    uint32_t h2d_num_bufs = 8u;
    uint32_t ring_depth = 4u;
};

// ── DmaEngine ──────────────────────────────────────────────────────────────
//
// Manages up to kDmaNumCores (16) persistent DMA cores.  Kernels are launched
// once at construction and run until destruction (EXIT opcode).
//
// Thread safety: single-threaded use only in the current implementation.

class DmaEngine {
public:
    // ── Construction / destruction ────────────────────────────────────────

    // Primary constructor — accepts full config.
    explicit DmaEngine(std::shared_ptr<MeshDevice> mesh_device, DmaEngineConfig cfg = {}) :
        mesh_device_(std::move(mesh_device)),
        device_coord_(0, 0),
        num_cores_(cfg.num_cores),
        chunk_size_(cfg.chunk_size),
        h2d_num_bufs_(cfg.h2d_num_bufs),
        ring_depth_(cfg.ring_depth) {
        TT_FATAL(
            num_cores_ >= 1 && num_cores_ <= kDmaNumCores,
            "num_cores {} out of range [1, {}]",
            num_cores_,
            kDmaNumCores);
        // 16 KB is the NOC max burst size for PCIe reads (noc_read_with_state)
        // on Blackhole.  Larger values cause the read barrier to hang.
        // DRAM writes (noc_async_write) can handle larger bursts internally, but
        // keeping chunk_size ≤ 16 KB ensures both directions work correctly.
        TT_FATAL(
            chunk_size_ >= 1024 && chunk_size_ <= 16u * 1024u && (chunk_size_ & (chunk_size_ - 1)) == 0,
            "chunk_size {} must be a power of 2 in [1024, 16384] "
            "(NOC PCIe-read max burst on Blackhole is 16 KB)",
            chunk_size_);
        TT_FATAL(h2d_num_bufs_ >= 1 && h2d_num_bufs_ <= 64, "h2d_num_bufs {} out of range [1, 64]", h2d_num_bufs_);
        TT_FATAL(
            ring_depth_ >= 1 && ring_depth_ <= 16 && (ring_depth_ & (ring_depth_ - 1)) == 0,
            "ring_depth {} must be a power of 2 in [1, 16]",
            ring_depth_);

        // L1 safety: ensure buffers don't overflow the usable L1 region.
        // kDmaBufBase = 0x80400; D2H uses 2 × chunk_size, H2D uses h2d_num_bufs × chunk_size.
        const uint32_t l1_buf_end = kDmaBufBase + (2u + h2d_num_bufs_) * chunk_size_;
        TT_FATAL(
            l1_buf_end <= 1u << 20,
            "Config overflows L1: kDmaBufBase=0x{:x} + (2+{}) × {} = 0x{:x} > 1 MB",
            kDmaBufBase,
            h2d_num_bufs_,
            chunk_size_,
            l1_buf_end);

        device_ = mesh_device_->get_device(device_coord_);

        const auto grid = device_->compute_with_storage_grid_size();
        const uint32_t need_cols = (num_cores_ <= kDmaCoreCols) ? num_cores_ : kDmaCoreCols;
        const uint32_t need_rows = (num_cores_ + kDmaCoreCols - 1) / kDmaCoreCols;
        TT_FATAL(
            static_cast<uint32_t>(grid.x) >= need_cols && static_cast<uint32_t>(grid.y) >= need_rows,
            "Grid {}x{} too small for {} DMA cores (need {}x{})",
            grid.x,
            grid.y,
            num_cores_,
            need_cols,
            need_rows);

        setup_pcie_encoding();
        setup_bar_pointers();
        launch_kernels();

        std::cout << "[DmaEngine] " << num_cores_ << " persistent DMA cores ready"
                  << "  chunk=" << chunk_size_ / 1024 << "KB"
                  << "  h2d_bufs=" << h2d_num_bufs_ << "  ring=" << ring_depth_ << "\n";
        std::cout << "[DmaEngine] pcie_xy_enc=0x" << std::hex << pcie_xy_enc_ << std::dec << "\n";
    }

    // Backward-compat shim: callers that only pass num_cores.
    explicit DmaEngine(std::shared_ptr<MeshDevice> mesh_device, uint32_t num_cores) :
        DmaEngine(std::move(mesh_device), DmaEngineConfig{.num_cores = num_cores}) {}

    ~DmaEngine() {
        stop_kernels();
        std::cout << "[DmaEngine] stopped\n";
    }

    // ── D2H transfer: DRAM → host ─────────────────────────────────────────

    static constexpr size_t kFlagStride = 64;  // cache-line size
    // Maximum flag region for kDmaNumCores (backward compat for test code).
    static constexpr size_t kFlagRegionBytes = kDmaNumCores * kFlagStride;

    // Returns the flag region size for the active num_cores_ cores.
    size_t flag_region_bytes() const { return static_cast<size_t>(num_cores_) * kFlagStride; }

    void transfer_d2h(
        uint32_t dram_addr,
        uint64_t dst_phys,
        size_t total_bytes,
        uint64_t done_flags_phys,
        volatile uint32_t* done_flags_ptr,
        uint32_t batch_size = 0,
        uint32_t opcode = DMA_OP_TRANSFER) {
        TT_FATAL(
            total_bytes % (num_cores_ * chunk_size_) == 0,
            "total_bytes {} must be a multiple of {} (num_cores * chunk_size)",
            total_bytes,
            num_cores_ * chunk_size_);

        const uint32_t per_core_bytes = static_cast<uint32_t>(total_bytes / num_cores_);
        const uint32_t per_core_chunks = per_core_bytes / chunk_size_;

        for (uint32_t c = 0; c < num_cores_; ++c) {
            const uint64_t core_dst = dst_phys + static_cast<uint64_t>(c) * per_core_bytes;
            const uint64_t flag_phys = done_flags_phys + c * kFlagStride;
            const uint32_t page_start = c * per_core_chunks;
            send_cmd_d2h(c, dram_addr, page_start, core_dst, per_core_chunks, flag_phys, opcode);
        }

        uint32_t done_mask = 0;
        const uint32_t full_mask = (num_cores_ == 32) ? 0xFFFFFFFFu : ((1u << num_cores_) - 1u);
        const auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(10);
        while (done_mask != full_mask) {
            for (uint32_t c = 0; c < num_cores_; ++c) {
                if (done_mask & (1u << c)) {
                    continue;
                }
                if (bar_read32(c, kDmaD2HCompletionSeq) == d2h_seq_[c]) {
                    done_mask |= (1u << c);
                }
            }
            if (std::chrono::high_resolution_clock::now() > deadline) {
                for (uint32_t c = 0; c < num_cores_; ++c) {
                    if (done_mask & (1u << c)) {
                        continue;
                    }
                    auto* flag = reinterpret_cast<volatile uint32_t*>(
                        reinterpret_cast<volatile char*>(done_flags_ptr) + c * kFlagStride);
                    dump_d2h_timeout(c, flag, done_flags_ptr);
                }
                TT_FATAL(false, "DmaEngine D2H timeout (done_mask=0x{:x})", done_mask);
            }
        }
    }

    // ── H2D transfer: host → DRAM ─────────────────────────────────────────

    void transfer_h2d(
        uint64_t src_phys,
        uint32_t dram_addr,
        size_t total_bytes,
        uint32_t opcode = DMA_OP_TRANSFER,
        bool dbg_timing = false) {
        TT_FATAL(
            total_bytes % (num_cores_ * chunk_size_) == 0,
            "total_bytes {} must be a multiple of {} (num_cores * chunk_size)",
            total_bytes,
            num_cores_ * chunk_size_);

        const uint32_t per_core_bytes = static_cast<uint32_t>(total_bytes / num_cores_);
        const uint32_t per_core_chunks = per_core_bytes / chunk_size_;

        auto t_cmd0 = std::chrono::high_resolution_clock::now();
        for (uint32_t c = 0; c < num_cores_; ++c) {
            const uint64_t core_src = src_phys + static_cast<uint64_t>(c) * per_core_bytes;
            const uint32_t page_start = c * per_core_chunks;
            send_cmd_h2d(c, core_src, dram_addr, page_start, per_core_chunks, opcode);
        }
        auto t_cmd1 = std::chrono::high_resolution_clock::now();

        uint64_t total_poll_iters = 0;
        const auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(10);
        const uint32_t full_mask = (num_cores_ == 32) ? 0xFFFFFFFFu : ((1u << num_cores_) - 1u);
        uint32_t done_mask = 0;
        while (done_mask != full_mask) {
            for (uint32_t c = 0; c < num_cores_; ++c) {
                if (done_mask & (1u << c)) {
                    continue;
                }
                ++total_poll_iters;
                if (bar_read32(c, kDmaH2DCompletionSeq) == h2d_seq_[c]) {
                    done_mask |= (1u << c);
                }
            }
            if (std::chrono::high_resolution_clock::now() > deadline) {
                for (uint32_t c = 0; c < num_cores_; ++c) {
                    if (done_mask & (1u << c)) {
                        continue;
                    }
                    std::cerr << "[DmaEngine] H2D TIMEOUT on core " << c
                              << "  completion_seq=" << bar_read32(c, kDmaH2DCompletionSeq)
                              << " (expecting=" << h2d_seq_[c] << ")"
                              << "  progress=" << bar_read32(c, kDmaH2DProgress) << "\n";
                }
                TT_FATAL(false, "DmaEngine H2D pull timeout (done_mask=0x{:x})", done_mask);
            }
        }
        auto t_poll1 = std::chrono::high_resolution_clock::now();

        if (dbg_timing) {
            using us = std::chrono::duration<double, std::micro>;
            const double cmd_us = us(t_cmd1 - t_cmd0).count();
            const double poll_us = us(t_poll1 - t_cmd1).count();
            const double total_us = us(t_poll1 - t_cmd0).count();
            const double bw = total_bytes / (total_us * 1e-6) / (1024.0 * 1024.0 * 1024.0);
            std::cerr << "[H2D_PULL_TIMING]"
                      << " size=" << total_bytes / 1024 << " KB"
                      << "  cmd=" << std::fixed << std::setprecision(1) << cmd_us << " µs"
                      << "  poll=" << poll_us << " µs"
                      << "  poll_iters=" << total_poll_iters << "  µs/poll=" << std::setprecision(2)
                      << poll_us / total_poll_iters << "  total=" << total_us << " µs"
                      << "  BW=" << std::setprecision(2) << bw << " GB/s\n";
        }
    }

    // ── H2D host-push transfer: host BAR-writes to L1, kernel drains to DRAM

    void transfer_h2d_push(
        const uint32_t* src_ptr, uint32_t dram_addr, size_t total_bytes, uint32_t opcode = DMA_OP_H2D_PUSH) {
        TT_FATAL(
            total_bytes % (num_cores_ * chunk_size_) == 0,
            "total_bytes {} must be a multiple of {} (num_cores * chunk_size)",
            total_bytes,
            num_cores_ * chunk_size_);

        const uint32_t per_core_bytes = static_cast<uint32_t>(total_bytes / num_cores_);
        const uint32_t per_core_chunks = per_core_bytes / chunk_size_;
        const uint32_t B = h2d_num_bufs_;
        const uint32_t num_batches = (per_core_chunks + B - 1u) / B;

        for (uint32_t c = 0; c < num_cores_; ++c) {
            bar_write32(c, kDmaH2DPushSeq, 0u);
            bar_write32(c, kDmaH2DPushAck, 0u);
        }
        _mm_sfence();

        for (uint32_t c = 0; c < num_cores_; ++c) {
            const uint32_t page_start = c * per_core_chunks;
            send_cmd_h2d(c, 0, dram_addr, page_start, per_core_chunks, opcode);
        }

        uint32_t push_count[kDmaNumCores] = {};
        const uint32_t full_mask = (num_cores_ == 32) ? 0xFFFFFFFFu : ((1u << num_cores_) - 1u);

        for (uint32_t b = 0; b < num_batches; ++b) {
            const uint32_t chunk_start = b * B;

            if (b > 0) {
                const auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(10);
                uint32_t done_mask = 0;
                while (done_mask != full_mask) {
                    for (uint32_t c = 0; c < num_cores_; ++c) {
                        if (done_mask & (1u << c)) {
                            continue;
                        }
                        if (bar_read32(c, kDmaH2DPushAck) >= push_count[c]) {
                            done_mask |= (1u << c);
                        }
                    }
                    if (std::chrono::high_resolution_clock::now() > deadline) {
                        TT_FATAL(false, "H2D push ack timeout batch {}", b);
                    }
                }
            }

            for (uint32_t c = 0; c < num_cores_; ++c) {
                const uint32_t remaining = per_core_chunks - chunk_start;
                const uint32_t this_batch = (remaining < B) ? remaining : B;
                const uint32_t this_bytes = this_batch * chunk_size_;

                const void* src =
                    reinterpret_cast<const uint8_t*>(src_ptr) + c * per_core_bytes + chunk_start * chunk_size_;
                bar_memcpy(c, h2d_buf_start_addr(), src, this_bytes);
            }

            _mm_sfence();
            for (uint32_t c = 0; c < num_cores_; ++c) {
                push_count[c]++;
                bar_write32(c, kDmaH2DPushSeq, push_count[c]);
            }
            _mm_sfence();
        }

        const auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(10);
        uint32_t done_mask = 0;
        while (done_mask != full_mask) {
            for (uint32_t c = 0; c < num_cores_; ++c) {
                if (done_mask & (1u << c)) {
                    continue;
                }
                if (bar_read32(c, kDmaH2DCompletionSeq) == h2d_seq_[c]) {
                    done_mask |= (1u << c);
                }
            }
            if (std::chrono::high_resolution_clock::now() > deadline) {
                TT_FATAL(false, "H2D push completion timeout");
            }
        }
    }

    // ── H2D via kernel DMA engine: bypasses CPU MMIO, uses HW DMA ────────

    void transfer_h2d_dma(
        const void* src_ptr, uint32_t dram_addr, size_t total_bytes, uint32_t opcode = DMA_OP_H2D_PUSH) {
        TT_FATAL(dma_available_, "Kernel DMA batch not available (probe failed)");
        TT_FATAL(
            total_bytes % (num_cores_ * chunk_size_) == 0,
            "total_bytes {} must be a multiple of {} (num_cores * chunk_size)",
            total_bytes,
            num_cores_ * chunk_size_);

        const uint32_t per_core_bytes = static_cast<uint32_t>(total_bytes / num_cores_);
        const uint32_t per_core_chunks = per_core_bytes / chunk_size_;
        const uint32_t B = h2d_num_bufs_;
        const uint32_t num_batches = (per_core_chunks + B - 1u) / B;
        for (uint32_t c = 0; c < num_cores_; ++c) {
            bar_write32(c, kDmaH2DPushSeq, 0u);
            bar_write32(c, kDmaH2DPushAck, 0u);
        }
        _mm_sfence();

        for (uint32_t c = 0; c < num_cores_; ++c) {
            const uint32_t page_start = c * per_core_chunks;
            send_cmd_h2d(c, 0, dram_addr, page_start, per_core_chunks, opcode);
        }

        uint32_t pin_handle = pci_dev_->dma_pin_buffer(src_ptr, total_bytes);

        uint32_t push_count[kDmaNumCores] = {};
        std::vector<tt::umd::PCIDevice::DmaBatchEntry> entries(num_cores_);
        const uint32_t full_mask = (num_cores_ == 32) ? 0xFFFFFFFFu : ((1u << num_cores_) - 1u);

        for (uint32_t b = 0; b < num_batches; ++b) {
            const uint32_t chunk_start = b * B;

            if (b > 0) {
                const auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(10);
                uint32_t done_mask = 0;
                while (done_mask != full_mask) {
                    for (uint32_t c = 0; c < num_cores_; ++c) {
                        if (done_mask & (1u << c)) {
                            continue;
                        }
                        if (bar_read32(c, kDmaH2DPushAck) >= push_count[c]) {
                            done_mask |= (1u << c);
                        }
                    }
                    if (std::chrono::high_resolution_clock::now() > deadline) {
                        pci_dev_->dma_unpin_buffer(pin_handle);
                        TT_FATAL(false, "H2D DMA push ack timeout batch {}", b);
                    }
                }
            }

            const uint32_t remaining = per_core_chunks - chunk_start;
            const uint32_t this_batch = (remaining < B) ? remaining : B;
            const uint32_t this_bytes = this_batch * chunk_size_;

            for (uint32_t c = 0; c < num_cores_; ++c) {
                entries[c].device_addr = dma_dev_addr_[c];
                entries[c].size = this_bytes;
                entries[c].host_offset =
                    static_cast<uint64_t>(c) * per_core_bytes + static_cast<uint64_t>(chunk_start) * chunk_size_;
            }

            pci_dev_->kernel_dma_batch_transfer_pinned(pin_handle, total_bytes, entries.data(), num_cores_, true);

            for (uint32_t c = 0; c < num_cores_; ++c) {
                push_count[c]++;
                bar_write32(c, kDmaH2DPushSeq, push_count[c]);
            }
            _mm_sfence();
        }

        pci_dev_->dma_unpin_buffer(pin_handle);

        const auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(10);
        uint32_t done_mask = 0;
        while (done_mask != full_mask) {
            for (uint32_t c = 0; c < num_cores_; ++c) {
                if (done_mask & (1u << c)) {
                    continue;
                }
                if (bar_read32(c, kDmaH2DCompletionSeq) == h2d_seq_[c]) {
                    done_mask |= (1u << c);
                }
            }
            if (std::chrono::high_resolution_clock::now() > deadline) {
                TT_FATAL(false, "H2D DMA push completion timeout");
            }
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────────
    uint32_t num_cores() const { return num_cores_; }
    uint32_t chunk_size() const { return chunk_size_; }
    uint32_t h2d_num_bufs() const { return h2d_num_bufs_; }
    uint32_t ring_depth() const { return ring_depth_; }
    uint32_t pcie_xy_enc() const { return pcie_xy_enc_; }
    IDevice* device_for_debug() const { return device_; }
    CoreCoord core_for_debug(uint32_t c) const { return dma_core(c); }
    bool has_kernel_dma() const { return dma_available_; }

    // Returns the active config as a struct.
    DmaEngineConfig config() const { return {num_cores_, chunk_size_, h2d_num_bufs_, ring_depth_}; }

private:
    // ── Internal helpers ──────────────────────────────────────────────────

    void dump_d2h_timeout(uint32_t c, volatile uint32_t* flag, volatile uint32_t* done_flags_ptr) {
        std::cerr << "[DmaEngine] D2H TIMEOUT on core " << c << " (done_flag=" << *flag << " expected=" << d2h_seq_[c]
                  << ")\n";
        for (uint32_t k = 0; k < num_cores_; ++k) {
            auto* kf = reinterpret_cast<volatile uint32_t*>(
                reinterpret_cast<volatile char*>(done_flags_ptr) + k * kFlagStride);
            std::vector<uint32_t> prog, cseq, diag;
            tt::tt_metal::detail::ReadFromDeviceL1(device_, dma_core(k), kDmaD2HProgress, sizeof(uint32_t), prog);
            tt::tt_metal::detail::ReadFromDeviceL1(device_, dma_core(k), kDmaD2HCompletionSeq, sizeof(uint32_t), cseq);
            tt::tt_metal::detail::ReadFromDeviceL1(
                device_, dma_core(k), kDmaD2HDiagBase, kDmaD2HDiagWords * sizeof(uint32_t), diag);
            std::cerr << "  core " << k << ": done_flag=" << *kf << " progress=" << (prog.empty() ? -1 : (int)prog[0])
                      << " cseq=" << (cseq.empty() ? -1 : (int)cseq[0]) << "\n";
            if (diag.size() >= kDmaD2HDiagWords) {
                std::cerr << "    diag: flag_addr=0x" << std::hex << (static_cast<uint64_t>(diag[1]) << 32 | diag[0])
                          << " seq=" << std::dec << diag[2] << " cmd#=" << diag[3] << " sw_acked=" << diag[4]
                          << " hw_acked=" << diag[5] << " sw_issued=" << diag[6] << " hw_sent=" << diag[7] << "\n";
            }
        }
    }

    CoreCoord dma_core(uint32_t c) const { return CoreCoord(c % kDmaCoreCols, c / kDmaCoreCols); }

    static void nt_memcpy(volatile void* __restrict dst, const void* __restrict src, size_t bytes) {
        auto* d = reinterpret_cast<__m128i*>(const_cast<void*>(dst));
        auto* s = reinterpret_cast<const __m128i*>(src);
        const size_t n = bytes / sizeof(__m128i);
        for (size_t i = 0; i < n; i += 4) {
            __m128i v0 = _mm_load_si128(s + i + 0);
            __m128i v1 = _mm_load_si128(s + i + 1);
            __m128i v2 = _mm_load_si128(s + i + 2);
            __m128i v3 = _mm_load_si128(s + i + 3);
            _mm_stream_si128(d + i + 0, v0);
            _mm_stream_si128(d + i + 1, v1);
            _mm_stream_si128(d + i + 2, v2);
            _mm_stream_si128(d + i + 3, v3);
        }
        _mm_sfence();
    }

    struct WriterLayout {
        void* base;
        size_t tlb_size;
    };

    void setup_bar_pointers() {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

        for (uint32_t c = 0; c < num_cores_; ++c) {
            auto vcore = device_->virtual_core_from_logical_core(dma_core(c), CoreType::WORKER);
            tt_cxy_pair target(device_->id(), vcore.x, vcore.y);

            vcore_xy_[c] = tt_xy_pair(vcore.x, vcore.y);

            auto writer = cluster.get_static_tlb_writer(target);
            auto* layout = reinterpret_cast<WriterLayout*>(&writer);
            bar_base_[c] = static_cast<char*>(layout->base);
            bar_size_[c] = layout->tlb_size;
            const uint32_t h2d_buf_end = h2d_buf_start_addr() + h2d_num_bufs_ * chunk_size_;
            TT_FATAL(
                bar_size_[c] >= h2d_buf_end,
                "TLB window too small for core {}: {} bytes (need >= 0x{:x} for chunk={} bufs={})",
                c,
                bar_size_[c],
                h2d_buf_end,
                chunk_size_,
                h2d_num_bufs_);

            auto tlb_data_opt = cluster.get_tlb_data(target);
            TT_FATAL(tlb_data_opt.has_value(), "No TLB data for core {}", c);
            auto [tlb_offset, tlb_size] = tlb_data_opt.value();
            dma_dev_addr_[c] = tlb_offset + (h2d_buf_start_addr() % tlb_size);
        }

        pci_dev_ = cluster.get_pci_device(device_->id());
        TT_FATAL(pci_dev_ != nullptr, "Failed to get PCIDevice for chip {}", device_->id());

        constexpr size_t kWcTlbSize = 2u * 1024u * 1024u;
        const uint32_t h2d_start = h2d_buf_start_addr();
        const uint32_t aligned_base = h2d_start & ~(static_cast<uint32_t>(kWcTlbSize) - 1u);
        const uint32_t offset_within_tlb = h2d_start - aligned_base;

        for (uint32_t c = 0; c < num_cores_; ++c) {
            wc_tlb_[c] = pci_dev_->allocate_tlb(kWcTlbSize, tt::umd::TlbMapping::WC);

            tt::umd::tlb_data cfg{};
            cfg.local_offset = aligned_base;
            cfg.x_end = vcore_xy_[c].x;
            cfg.y_end = vcore_xy_[c].y;
            cfg.noc_sel = 0;
            cfg.ordering = tt::umd::tlb_data::Relaxed;
            cfg.static_vc = false;
            wc_tlb_[c]->configure(cfg);

            wc_write_base_[c] = wc_tlb_[c]->get_base() + offset_within_tlb;
        }

        constexpr size_t kProbeBytes = 128u * 1024u;
        auto* probe_buf = static_cast<uint8_t*>(aligned_alloc(64, kProbeBytes));
        TT_FATAL(probe_buf != nullptr, "aligned_alloc failed for probe buffer");
        std::memset(probe_buf, 0xA5, kProbeBytes);

        {
            auto t0 = std::chrono::high_resolution_clock::now();
            nt_memcpy(wc_write_base_[0], probe_buf, kProbeBytes);
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double gbps = (kProbeBytes / us) / 1e3;
            std::cout << "[DmaEngine] WC NT-store probe: " << kProbeBytes / 1024 << " KB in " << std::fixed
                      << std::setprecision(1) << us << " µs = " << std::setprecision(2) << gbps << " GB/s\n";
        }

        if (pci_dev_->has_kernel_dma_batch()) {
            std::cout << "[DmaEngine] DMA dev_addr[] = {";
            for (uint32_t c = 0; c < num_cores_; ++c) {
                if (c) {
                    std::cout << ", ";
                }
                std::cout << "0x" << std::hex << dma_dev_addr_[c] << std::dec;
            }
            std::cout << "}\n";

            const size_t total_probe = num_cores_ * kProbeBytes;
            auto [dma_raw, dma_alloc] = pdma_mmap_alloc(total_probe);
            auto* dma_probe_buf = static_cast<uint8_t*>(dma_raw);
            std::memset(dma_probe_buf, 0xB5, total_probe);

            bool dma_ok = false;
            auto* staging = pci_dev_->get_dma_staging_buffer();

            if (staging && staging->size >= total_probe) {
                std::memcpy(staging->host_ptr, dma_probe_buf, total_probe);

                std::vector<tt::umd::PCIDevice::DmaBatchEntry> entries(num_cores_);
                for (uint32_t c = 0; c < num_cores_; ++c) {
                    entries[c].device_addr = dma_dev_addr_[c];
                    entries[c].size = static_cast<uint32_t>(kProbeBytes);
                    entries[c].host_offset = c * kProbeBytes;
                }

                try {
                    pci_dev_->kernel_dma_batch_transfer_pinned(
                        staging->pin_handle, staging->size, entries.data(), num_cores_, true);

                    auto t0 = std::chrono::high_resolution_clock::now();
                    pci_dev_->kernel_dma_batch_transfer_pinned(
                        staging->pin_handle, staging->size, entries.data(), num_cores_, true);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                    double gbps = (total_probe / us) / 1e3;
                    std::cout << "[DmaEngine] Kernel DMA batch probe (staging): " << num_cores_ << "×"
                              << kProbeBytes / 1024 << " KB in " << std::fixed << std::setprecision(1) << us
                              << " µs = " << std::setprecision(2) << gbps << " GB/s\n";
                    dma_ok = true;
                } catch (const std::exception& e) {
                    std::cerr << "[DmaEngine] DMA batch (staging) FAILED: " << e.what() << "\n";
                }

                if (!dma_ok) {
                    try {
                        tt::umd::PCIDevice::DmaBatchEntry single;
                        single.device_addr = dma_dev_addr_[0];
                        single.size = static_cast<uint32_t>(kProbeBytes);
                        single.host_offset = 0;
                        pci_dev_->kernel_dma_batch_transfer_pinned(
                            staging->pin_handle, staging->size, &single, 1, true);
                        std::cout << "[DmaEngine] Single-entry DMA OK (core 0)\n";
                        dma_ok = true;
                    } catch (const std::exception& e) {
                        std::cerr << "[DmaEngine] Single-entry DMA FAILED: " << e.what() << "\n";
                    }
                }
            } else {
                std::cerr << "[DmaEngine] No staging buffer (staging="
                          << (staging ? std::to_string(staging->size) : "null") << " need=" << total_probe << ")\n";

                std::vector<tt::umd::PCIDevice::DmaBatchEntry> entries(num_cores_);
                for (uint32_t c = 0; c < num_cores_; ++c) {
                    entries[c].device_addr = dma_dev_addr_[c];
                    entries[c].size = static_cast<uint32_t>(kProbeBytes);
                    entries[c].host_offset = c * kProbeBytes;
                }
                try {
                    pci_dev_->kernel_dma_batch_transfer(dma_probe_buf, total_probe, entries.data(), num_cores_, true);
                    std::cout << "[DmaEngine] Unpinned DMA batch OK\n";
                    dma_ok = true;
                } catch (const std::exception& e) {
                    std::cerr << "[DmaEngine] Unpinned DMA batch FAILED: " << e.what() << "\n";
                }
            }
            dma_available_ = dma_ok;
            munmap(dma_raw, dma_alloc);
        } else {
            std::cout << "[DmaEngine] Kernel DMA batch not supported by KMD\n";
            dma_available_ = false;
        }

        std::free(probe_buf);
    }

    // ── L1 address helpers (runtime-config-aware) ─────────────────────────
    // D2H and H2D ring base addresses depend on ring_depth_.
    uint32_t d2h_cmd_base() const { return kDmaUserBase; }
    uint32_t h2d_cmd_base() const { return kDmaUserBase + ring_depth_ * kDmaSlotBytes; }
    // Data buffer layout: D2H ping, D2H pong, then h2d_num_bufs_ H2D bufs.
    uint32_t d2h_buf_ping() const { return kDmaBufBase; }
    uint32_t d2h_buf_pong() const { return kDmaBufBase + chunk_size_; }
    uint32_t h2d_buf_start_addr() const { return kDmaBufBase + 2u * chunk_size_; }
    uint32_t h2d_buf_addr(uint32_t i) const { return h2d_buf_start_addr() + i * chunk_size_; }

    void bar_write32(uint32_t core, uint32_t l1_addr, uint32_t val) {
        *reinterpret_cast<volatile uint32_t*>(bar_base_[core] + l1_addr) = val;
    }

    uint32_t bar_read32(uint32_t core, uint32_t l1_addr) {
        return *reinterpret_cast<volatile uint32_t*>(bar_base_[core] + l1_addr);
    }

    void bar_memcpy(uint32_t core, uint32_t l1_addr, const void* src, size_t bytes) {
        uint32_t offset = l1_addr - h2d_buf_start_addr();
        nt_memcpy(wc_write_base_[core] + offset, src, bytes);
    }

    void send_cmd_d2h(
        uint32_t c,
        uint32_t dram_base,
        uint32_t page_start,
        uint64_t dst_phys,
        uint32_t num_chunks,
        uint64_t done_flag_phys,
        uint32_t opcode = DMA_OP_TRANSFER) {
        d2h_seq_[c]++;
        const uint32_t slot_addr = d2h_cmd_base() + d2h_slot_[c] * kDmaSlotBytes;
        d2h_slot_[c] = (d2h_slot_[c] + 1u) & (ring_depth_ - 1u);

        bar_write32(c, slot_addr + 1 * 4, opcode);
        bar_write32(c, slot_addr + 2 * 4, dram_base);
        bar_write32(c, slot_addr + 3 * 4, page_start);
        bar_write32(c, slot_addr + 4 * 4, static_cast<uint32_t>(dst_phys));
        bar_write32(c, slot_addr + 5 * 4, static_cast<uint32_t>(dst_phys >> 32));
        bar_write32(c, slot_addr + 6 * 4, num_chunks);
        bar_write32(c, slot_addr + 7 * 4, static_cast<uint32_t>(done_flag_phys));
        bar_write32(c, slot_addr + 8 * 4, static_cast<uint32_t>(done_flag_phys >> 32));
        bar_write32(c, slot_addr + 9 * 4, 0u);
        bar_write32(c, slot_addr + 10 * 4, 0u);
        bar_write32(c, slot_addr + 11 * 4, 0u);

        _mm_sfence();
        bar_write32(c, slot_addr, d2h_seq_[c]);
    }

    void send_cmd_h2d(
        uint32_t c,
        uint64_t src_phys,
        uint32_t dram_base,
        uint32_t page_start,
        uint32_t num_chunks,
        uint32_t opcode = DMA_OP_TRANSFER) {
        h2d_seq_[c]++;
        const uint32_t slot_addr = h2d_cmd_base() + h2d_slot_[c] * kDmaSlotBytes;
        h2d_slot_[c] = (h2d_slot_[c] + 1u) & (ring_depth_ - 1u);

        bar_write32(c, slot_addr + 1 * 4, opcode);
        bar_write32(c, slot_addr + 2 * 4, static_cast<uint32_t>(src_phys));
        bar_write32(c, slot_addr + 3 * 4, static_cast<uint32_t>(src_phys >> 32));
        bar_write32(c, slot_addr + 4 * 4, dram_base);
        bar_write32(c, slot_addr + 5 * 4, page_start);
        bar_write32(c, slot_addr + 6 * 4, num_chunks);
        bar_write32(c, slot_addr + 7 * 4, 0u);
        bar_write32(c, slot_addr + 8 * 4, 0u);
        bar_write32(c, slot_addr + 9 * 4, 0u);
        bar_write32(c, slot_addr + 10 * 4, 0u);
        bar_write32(c, slot_addr + 11 * 4, 0u);

        _mm_sfence();
        bar_write32(c, slot_addr, h2d_seq_[c]);
    }

    void send_exit_all() {
        for (uint32_t c = 0; c < num_cores_; ++c) {
            // D2H EXIT
            {
                d2h_seq_[c]++;
                const uint32_t slot_addr = d2h_cmd_base() + d2h_slot_[c] * kDmaSlotBytes;
                d2h_slot_[c] = (d2h_slot_[c] + 1u) & (ring_depth_ - 1u);
                for (uint32_t w = 1; w < kDmaSlotBytes / 4; ++w) {
                    bar_write32(c, slot_addr + w * 4, (w == 1) ? DMA_OP_EXIT : 0u);
                }
                _mm_sfence();
                bar_write32(c, slot_addr, d2h_seq_[c]);
            }
            // H2D EXIT
            {
                h2d_seq_[c]++;
                const uint32_t slot_addr = h2d_cmd_base() + h2d_slot_[c] * kDmaSlotBytes;
                h2d_slot_[c] = (h2d_slot_[c] + 1u) & (ring_depth_ - 1u);
                for (uint32_t w = 1; w < kDmaSlotBytes / 4; ++w) {
                    bar_write32(c, slot_addr + w * 4, (w == 1) ? DMA_OP_EXIT : 0u);
                }
                _mm_sfence();
                bar_write32(c, slot_addr, h2d_seq_[c]);
            }
        }
    }

    void setup_pcie_encoding() {
        MeshCoordinateRangeSet device_range;
        device_range.merge(MeshCoordinateRange(device_coord_));

        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        auto [raw, alloc] = pdma_mmap_alloc(page);
        auto owner =
            std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(raw), [alloc](uint32_t* p) { munmap(p, alloc); });

        HostBuffer hbuf(tt::stl::Span<uint32_t>(owner.get(), alloc / sizeof(uint32_t)), MemoryPin(owner));
        auto pinned = experimental::PinnedMemory::Create(*mesh_device_, device_range, hbuf, /*map_to_noc=*/true);

        auto noc = pinned->get_noc_addr(device_->id());
        TT_FATAL(noc.has_value(), "Failed to probe PCIe XY encoding");
        pcie_xy_enc_ = noc->pcie_xy_enc;
    }

    void launch_kernels() {
        // Build CoreRangeSet for the active num_cores_ cores.
        // Supported core counts (1, 4, 8, 16) all form clean rectangles:
        //   ≤ kDmaCoreCols:  single row  (0,0)–(num_cores_-1, 0)
        //   multiple of kDmaCoreCols: full rows (0,0)–(kDmaCoreCols-1, rows-1)
        CoreRangeSet dma_core_set;
        const uint32_t full_rows = num_cores_ / kDmaCoreCols;
        const uint32_t remainder = num_cores_ % kDmaCoreCols;
        if (full_rows > 0) {
            dma_core_set = dma_core_set.merge(
                CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(kDmaCoreCols - 1, full_rows - 1))));
        }
        if (remainder > 0) {
            dma_core_set = dma_core_set.merge(
                CoreRangeSet(CoreRange(CoreCoord(0, full_rows), CoreCoord(remainder - 1, full_rows))));
        }

        // Zero both command rings and the ready flags that follow them.
        // Layout: D2H ring (ring_depth_ slots) | H2D ring (ring_depth_ slots) | flags (32 B)
        const uint32_t kClearBytes = ring_depth_ * kDmaSlotBytes * 2 + 32u;
        for (uint32_t c = 0; c < num_cores_; ++c) {
            for (uint32_t off = 0; off < kClearBytes; off += 4) {
                bar_write32(c, d2h_cmd_base() + off, 0u);
            }
        }

        auto program = CreateProgram();

        // Pass runtime config to kernels as preprocessor defines.  This lets a
        // single kernel source file serve multiple chunk_size/h2d_num_bufs/ring_depth
        // configs without source modification.  Metal's kernel cache keys on these.
        const std::map<std::string, std::string> kernel_defines = {
            {"KDMA_CHUNK_SIZE", std::to_string(chunk_size_)},
            {"KDMA_H2D_NUM_BUFS", std::to_string(h2d_num_bufs_)},
            {"KDMA_RING_DEPTH", std::to_string(ring_depth_)},
        };

        auto d2h_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/persistent_d2h_pusher.cpp",
            dma_core_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {},
                .defines = kernel_defines,
            });

        auto h2d_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/socket/persistent_h2d_puller.cpp",
            dma_core_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = {},
                .defines = kernel_defines,
            });

        for (uint32_t c = 0; c < num_cores_; ++c) {
            SetRuntimeArgs(program, d2h_kernel, dma_core(c), {pcie_xy_enc_});
            SetRuntimeArgs(program, h2d_kernel, dma_core(c), {pcie_xy_enc_});
        }

        workload_ = MeshWorkload();
        workload_.add_program(MeshCoordinateRange(device_coord_), std::move(program));

        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload_, /*blocking=*/false);

        std::cout << "[DmaEngine] waiting for kernels to boot...\n";
        for (uint32_t c = 0; c < num_cores_; ++c) {
            std::vector<uint32_t> val;
            const auto boot_deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(30);
            do {
                val.clear();
                tt::tt_metal::detail::ReadFromDeviceL1(device_, dma_core(c), kDmaD2HReadyFlag, sizeof(uint32_t), val);
                if (std::chrono::high_resolution_clock::now() > boot_deadline) {
                    std::vector<uint32_t> ring0;
                    tt::tt_metal::detail::ReadFromDeviceL1(device_, dma_core(c), d2h_cmd_base(), kDmaSlotBytes, ring0);
                    std::cerr << "[DmaEngine] D2H kernel BOOT TIMEOUT on core " << c << "\n  D2HReadyFlag=0x"
                              << std::hex << kDmaD2HReadyFlag
                              << " ring_slot0[0]=" << (ring0.empty() ? 0xDEAD : ring0[0]) << std::dec << "\n";
                    TT_FATAL(false, "D2H kernel boot timeout on core {}", c);
                }
            } while (val.empty() || val[0] != 1u);

            const auto h2d_boot_deadline = std::chrono::high_resolution_clock::now() + std::chrono::seconds(30);
            do {
                val.clear();
                tt::tt_metal::detail::ReadFromDeviceL1(device_, dma_core(c), kDmaH2DReadyFlag, sizeof(uint32_t), val);
                if (std::chrono::high_resolution_clock::now() > h2d_boot_deadline) {
                    std::cerr << "[DmaEngine] H2D kernel BOOT TIMEOUT on core " << c << "\n  H2DReadyFlag=0x"
                              << std::hex << kDmaH2DReadyFlag << std::dec << "\n";
                    TT_FATAL(false, "H2D kernel boot timeout on core {}", c);
                }
            } while (val.empty() || val[0] != 1u);
        }
    }

    void stop_kernels() {
        send_exit_all();
        Finish(mesh_device_->mesh_command_queue());
    }

    // ── State ─────────────────────────────────────────────────────────────

    std::shared_ptr<MeshDevice> mesh_device_;
    IDevice* device_;
    MeshCoordinate device_coord_;
    uint32_t num_cores_;
    uint32_t chunk_size_;
    uint32_t h2d_num_bufs_;
    uint32_t ring_depth_;

    uint32_t pcie_xy_enc_ = 0;

    uint32_t d2h_seq_[kDmaNumCores] = {};
    uint32_t h2d_seq_[kDmaNumCores] = {};
    uint32_t d2h_slot_[kDmaNumCores] = {};
    uint32_t h2d_slot_[kDmaNumCores] = {};

    char* bar_base_[kDmaNumCores] = {};
    size_t bar_size_[kDmaNumCores] = {};

    tt_xy_pair vcore_xy_[kDmaNumCores] = {};

    std::unique_ptr<tt::umd::TlbHandle> wc_tlb_[kDmaNumCores];
    uint8_t* wc_write_base_[kDmaNumCores] = {};

    tt::umd::PCIDevice* pci_dev_ = nullptr;
    uint32_t dma_dev_addr_[kDmaNumCores] = {};
    bool dma_available_ = false;

    MeshWorkload workload_;
};

}  // namespace tt::tt_metal::distributed
