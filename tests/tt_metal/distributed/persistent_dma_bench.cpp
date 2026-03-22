// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// persistent_dma_bench — parametrizable persistent DMA bandwidth benchmark
//
// Mirrors socket_bw_bench interface so the two can be compared side-by-side.
// Uses DmaEngine: kernels launch once, spin on an L1 command ring, and accept
// new transfers via direct BAR writes (~100 ns overhead vs ~80 µs for socket).
//
// Build:
//   ninja -C build_Release persistent_dma_bench
//
// Usage examples:
//   ./persistent_dma_bench                          # all modes, default sizes, 16 cores
//   ./persistent_dma_bench --dram --cores 1,4,8,16 --iters 20
//   ./persistent_dma_bench --d2h --dram --csv > pdma_d2h.csv
//   ./persistent_dma_bench --bidir --iters 10 --verbose

#include "dma_engine.hpp"

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

// ── Helpers ────────────────────────────────────────────────────────────────

static inline double bw_gb_s(size_t bytes, double seconds) {
    return static_cast<double>(bytes) / seconds / (1024.0 * 1024.0 * 1024.0);
}

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

static std::string fmt_bytes(size_t b) {
    std::ostringstream os;
    if (b >= 1024ULL * 1024ULL * 1024ULL) {
        os << (b / (1024ULL * 1024ULL * 1024ULL)) << "G";
    } else if (b >= 1024ULL * 1024ULL) {
        os << (b / (1024ULL * 1024ULL)) << "M";
    } else if (b >= 1024ULL) {
        os << (b / 1024ULL) << "K";
    } else {
        os << b << "B";
    }
    return os.str();
}

// Flush a host buffer from all CPU caches (L1/L2/LLC).
static void clflush_range(const void* ptr, size_t bytes) {
    const char* p = static_cast<const char*>(ptr);
    for (size_t i = 0; i < bytes; i += 64) {
        _mm_clflush(p + i);
    }
    _mm_sfence();
}

// ── BenchConfig ────────────────────────────────────────────────────────────

struct BenchConfig {
    bool dir_d2h = false;
    bool dir_h2d = false;
    bool dir_h2d_push = false;
    bool dir_h2d_dma = false;
    bool dir_d2h_dma = false;
    bool dir_bidir = false;

    bool dir_d2h_scatter_l1 = false;
    bool dir_h2d_scatter_l1 = false;
    bool dir_d2h_interleaved = false;
    bool dir_h2d_interleaved = false;

    bool mode_dram = false;
    bool mode_l1 = false;

    std::vector<size_t> sizes;  // user-supplied or default
    std::vector<uint32_t> core_counts = {16};

    int warmup = 2;
    int iters = 20;

    // DmaEngine knobs
    uint32_t chunk_size = 16u * 1024u;  // 16 KB default (may be lowered for WH)
    uint32_t h2d_num_bufs = 8u;
    uint32_t ring_depth = 4u;
    bool user_chunk_size = false;  // true if --chunk-size was explicitly passed

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
    // DmaEngine config for this result row
    uint32_t chunk_size;
    uint32_t h2d_num_bufs;
    uint32_t ring_depth;
    std::vector<double> samples;

    double min_bw() const { return *std::min_element(samples.begin(), samples.end()); }
    double avg_bw() const { return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(); }
    double max_bw() const { return *std::max_element(samples.begin(), samples.end()); }
};

// ── Pinned host buffer helper ───────────────────────────────────────────────

struct PinnedHostBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    uint64_t phys = 0;
    // Lifetime: owner keeps the mmap alive; keeper keeps the IOMMU mapping alive.
    // Both must outlive any device transfer using this buffer.
    std::shared_ptr<uint32_t[]> owner;                   // munmap on destruction
    std::shared_ptr<experimental::PinnedMemory> keeper;  // IOMMU unmap on destruction
};

static PinnedHostBuf make_pinned_buf(const std::shared_ptr<MeshDevice>& mesh_device, IDevice* device, size_t bytes) {
    MeshCoordinateRangeSet device_range;
    device_range.merge(MeshCoordinateRange(MeshCoordinate(0, 0)));

    auto [raw, alloc] = pdma_mmap_alloc(bytes);
    std::memset(raw, 0, bytes);

    auto owner = std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(raw), [alloc](uint32_t* p) { munmap(p, alloc); });

    HostBuffer hbuf(tt::stl::Span<uint32_t>(owner.get(), bytes / sizeof(uint32_t)), MemoryPin(owner));
    auto pinned = experimental::PinnedMemory::Create(*mesh_device, device_range, hbuf, /*map_to_noc=*/true);
    auto noc = pinned->get_noc_addr(device->id());
    TT_FATAL(noc.has_value(), "Failed to get NOC address for host buffer");

    PinnedHostBuf out;
    out.ptr = raw;
    out.bytes = bytes;
    out.phys = noc->addr;
    out.owner = std::move(owner);    // keep mmap alive
    out.keeper = std::move(pinned);  // keep IOMMU mapping alive
    return out;
}

// ── DRAM buffer helper ─────────────────────────────────────────────────────

static std::shared_ptr<MeshBuffer> make_dram_buf(
    const std::shared_ptr<MeshDevice>& mesh_device, size_t bytes, uint32_t chunk_size = 16u * 1024u) {
    return MeshBuffer::create(
        ReplicatedBufferConfig{.size = bytes},
        DeviceLocalBufferConfig{
            .page_size = chunk_size,
            .buffer_type = BufferType::DRAM,
        },
        mesh_device.get());
}

// ── run_once helpers ───────────────────────────────────────────────────────

static double run_d2h_once(
    DmaEngine& engine,
    uint32_t num_cores,
    std::shared_ptr<MeshBuffer>& dram_buf,
    PinnedHostBuf& dst_buf,  // data + flag region, phys addresses valid
    uint64_t flags_phys,
    volatile uint32_t* flags_ptr,
    size_t total_bytes,
    uint32_t opcode) {
    clflush_range(dst_buf.ptr, total_bytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_d2h(
        static_cast<uint32_t>(dram_buf->address()),
        dst_buf.phys,
        total_bytes,
        flags_phys,
        flags_ptr,
        /*batch_size=*/0,
        opcode);
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_h2d_once(
    DmaEngine& engine,
    uint32_t num_cores,
    PinnedHostBuf& src_buf,
    std::shared_ptr<MeshBuffer>& dram_buf,
    size_t total_bytes,
    uint32_t opcode) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_h2d(src_buf.phys, static_cast<uint32_t>(dram_buf->address()), total_bytes, opcode);
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_h2d_push_once(
    DmaEngine& engine,
    const void* src_ptr,
    std::shared_ptr<MeshBuffer>& dram_buf,
    size_t total_bytes,
    uint32_t opcode) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_h2d_push(
        static_cast<const uint32_t*>(src_ptr), static_cast<uint32_t>(dram_buf->address()), total_bytes, opcode);
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_h2d_dma_once(DmaEngine& engine, const void* src_ptr, size_t total_bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_h2d_hw_dma(src_ptr, total_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_h2d_dma_zc_once(DmaEngine& engine, const void* src_ptr, size_t total_bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_h2d_hw_dma_zerocopy(src_ptr, total_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_d2h_dma_once(DmaEngine& engine, void* dst_ptr, size_t total_bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_d2h_hw_dma(dst_ptr, total_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_d2h_dma_zc_once(DmaEngine& engine, size_t total_bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_d2h_hw_dma_zerocopy(total_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();

    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// ── Scattered L1 / Interleaved DRAM run_once helpers ────────────────────

static double run_d2h_scatter_l1_once(
    DmaEngine& engine, const std::vector<CoreCoord>& cores, uint32_t l1_offset, size_t shard_size, size_t total_bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_d2h_scattered_l1_zerocopy(cores, l1_offset, shard_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_h2d_scatter_l1_once(
    DmaEngine& engine, const std::vector<CoreCoord>& cores, uint32_t l1_offset, size_t shard_size, size_t total_bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_h2d_scattered_l1_zerocopy(cores, l1_offset, shard_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_d2h_interleaved_once(
    DmaEngine& engine, uint32_t num_banks, uint32_t dram_offset, size_t total_bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_d2h_interleaved_dram_zerocopy(num_banks, dram_offset, total_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();
    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

static double run_h2d_interleaved_once(
    DmaEngine& engine, uint32_t num_banks, uint32_t dram_offset, size_t total_bytes) {
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.transfer_h2d_interleaved_dram_zerocopy(num_banks, dram_offset, total_bytes);
    auto t1 = std::chrono::high_resolution_clock::now();
    return bw_gb_s(total_bytes, std::chrono::duration<double>(t1 - t0).count());
}

// Returns {d2h_bw, h2d_bw} for a single bidirectional iteration.
static std::pair<double, double> run_bidir_once(
    DmaEngine& engine,
    uint32_t num_cores,
    std::shared_ptr<MeshBuffer>& d2h_dram,
    PinnedHostBuf& d2h_dst,
    uint64_t d2h_flags_phys,
    volatile uint32_t* d2h_flags_ptr,
    PinnedHostBuf& h2d_src,
    std::shared_ptr<MeshBuffer>& h2d_dram,
    size_t total_bytes,
    uint32_t opcode) {
    double d2h_s = 0, h2d_s = 0;

    std::thread d2h_thread([&] {
        clflush_range(d2h_dst.ptr, total_bytes);
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.transfer_d2h(
            static_cast<uint32_t>(d2h_dram->address()),
            d2h_dst.phys,
            total_bytes,
            d2h_flags_phys,
            d2h_flags_ptr,
            /*batch_size=*/0,
            opcode);
        auto t1 = std::chrono::high_resolution_clock::now();
        d2h_s = std::chrono::duration<double>(t1 - t0).count();
    });
    std::thread h2d_thread([&] {
        auto t0 = std::chrono::high_resolution_clock::now();
        engine.transfer_h2d(h2d_src.phys, static_cast<uint32_t>(h2d_dram->address()), total_bytes, opcode);
        auto t1 = std::chrono::high_resolution_clock::now();
        h2d_s = std::chrono::duration<double>(t1 - t0).count();
    });
    d2h_thread.join();
    h2d_thread.join();

    return {bw_gb_s(total_bytes, d2h_s), bw_gb_s(total_bytes, h2d_s)};
}

// ── Output ─────────────────────────────────────────────────────────────────

static void print_table(const std::vector<Result>& results) {
    std::cout << "┌───────────────────────────────────────────────────────────────────────────\n"
              << "│  persistent_dma_bench results\n"
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
    std::cout << "label,direction,mode,cores,bytes,iters,"
              << "chunk_size,h2d_num_bufs,ring_depth,"
              << "min_gb_s,avg_gb_s,max_gb_s\n";
    for (const auto& r : results) {
        std::cout << r.label << "," << r.direction << "," << r.mode << "," << r.cores << "," << r.bytes << ","
                  << r.iters << "," << r.chunk_size << "," << r.h2d_num_bufs << "," << r.ring_depth << std::fixed
                  << std::setprecision(4) << "," << r.min_bw() << "," << r.avg_bw() << "," << r.max_bw() << "\n";
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
                  << "    \"chunk_size\": " << r.chunk_size << ",\n"
                  << "    \"h2d_num_bufs\": " << r.h2d_num_bufs << ",\n"
                  << "    \"ring_depth\": " << r.ring_depth << ",\n"
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
                 "Directions (default: all except h2d-push):\n"
                 "  --d2h                  Enable D2H tests (DRAM→host)\n"
                 "  --h2d                  Enable H2D tests (host→DRAM, device-pull via NOC reads)\n"
                 "  --h2d-push             Enable H2D push tests (host→DRAM, host-push via WC BAR writes)\n"
                 "  --h2d-dma              Enable H2D DMA engine tests (WH only, DesignWare eDMA)\n"
                 "  --d2h-dma              Enable D2H DMA engine tests (WH only, DesignWare eDMA)\n"
                 "  --d2h-scatter-l1       Enable D2H from scattered L1 across tensix cores (eDMA + TLB retarget)\n"
                 "  --h2d-scatter-l1       Enable H2D to scattered L1 across tensix cores (eDMA + TLB retarget)\n"
                 "  --d2h-interleaved      Enable D2H from interleaved DRAM (eDMA cycling 6 banks)\n"
                 "  --h2d-interleaved      Enable H2D to interleaved DRAM (eDMA cycling 6 banks)\n"
                 "  --bidir                Enable bidirectional (concurrent D2H + H2D on same cores)\n"
                 "\n"
                 "Modes (default: both):\n"
                 "  --dram                 DRAM-backed transfers (DMA_OP_TRANSFER)\n"
                 "  --l1                   L1-only pure-PCIe transfers (DMA_OP_L1_ONLY)\n"
                 "\n"
                 "Transfer sizes (default: 256K,1M,4M,16M,64M):\n"
                 "  --sizes 256k,1m,4m     Comma-separated list (k/m/g suffixes supported)\n"
                 "\n"
                 "Core scaling (default: 16):\n"
                 "  --cores 1,4,8,16       Comma-separated core counts (≤16, rebuilds DmaEngine)\n"
                 "\n"
                 "DmaEngine knobs (rebuild engine for each value):\n"
                 "  --chunk-size 8192      Transfer chunk in bytes; power of 2 (default: 8KB on WH, 16KB on BH)\n"
                 "                         NOTE: NOC PCIe-read burst limit is 8 KB on Wormhole, 16 KB on Blackhole.\n"
                 "  --h2d-bufs 8           H2D PCIe read pipeline depth (default 8)\n"
                 "  --ring-depth 4         Command ring slots per direction; power of 2, [1,16] (default 4)\n"
                 "\n"
                 "Iteration control:\n"
                 "  --iters 20             Timed iterations per size point (default: 20)\n"
                 "  --warmup 2             Warmup iterations before timing (default: 2)\n"
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
        {"h2d-push", no_argument, nullptr, 'P'},
        {"h2d-dma", no_argument, nullptr, 13},
        {"d2h-dma", no_argument, nullptr, 14},
        {"d2h-scatter-l1", no_argument, nullptr, 15},
        {"h2d-scatter-l1", no_argument, nullptr, 16},
        {"d2h-interleaved", no_argument, nullptr, 17},
        {"h2d-interleaved", no_argument, nullptr, 18},
        {"bidir", no_argument, nullptr, 'b'},
        {"dram", no_argument, nullptr, 'D'},
        {"l1", no_argument, nullptr, 'l'},
        {"sizes", required_argument, nullptr, 's'},
        {"cores", required_argument, nullptr, 'c'},
        {"iters", required_argument, nullptr, 'i'},
        {"warmup", required_argument, nullptr, 'w'},
        {"chunk-size", required_argument, nullptr, 10},
        {"h2d-bufs", required_argument, nullptr, 11},
        {"ring-depth", required_argument, nullptr, 12},
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
            case 'P':
                cfg.dir_h2d_push = true;
                explicit_dir = true;
                break;
            case 13:
                cfg.dir_h2d_dma = true;
                explicit_dir = true;
                break;
            case 14:
                cfg.dir_d2h_dma = true;
                explicit_dir = true;
                break;
            case 15:
                cfg.dir_d2h_scatter_l1 = true;
                explicit_dir = true;
                break;
            case 16:
                cfg.dir_h2d_scatter_l1 = true;
                explicit_dir = true;
                break;
            case 17:
                cfg.dir_d2h_interleaved = true;
                explicit_dir = true;
                break;
            case 18:
                cfg.dir_h2d_interleaved = true;
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
                    cfg.sizes = v;
                    user_sizes = true;
                }
                break;
            }
            case 'c': cfg.core_counts = parse_uint_list(optarg); break;
            case 'i': cfg.iters = std::stoi(optarg); break;
            case 'w': cfg.warmup = std::stoi(optarg); break;
            case 10:
                cfg.chunk_size = static_cast<uint32_t>(std::stoul(optarg));
                cfg.user_chunk_size = true;
                break;
            case 11: cfg.h2d_num_bufs = static_cast<uint32_t>(std::stoul(optarg)); break;
            case 12: cfg.ring_depth = static_cast<uint32_t>(std::stoul(optarg)); break;
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
        cfg.core_counts = {16};
    }

    if (!user_sizes) {
        cfg.sizes = {
            256u * 1024u,
            1u * 1024u * 1024u,
            4u * 1024u * 1024u,
            16u * 1024u * 1024u,
            64u * 1024u * 1024u,
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
    const auto system_mesh_shape = distributed::SystemMesh::instance().shape();

    auto mesh_device = MeshDevice::create(
        MeshDeviceConfig(system_mesh_shape),
        /*l1_small_size=*/0,
        /*trace_region_size=*/0,
        /*num_cqs=*/1,
        DispatchCoreType::WORKER);

    const MeshCoordinate device_coord(0, 0);
    IDevice* device = mesh_device->get_device(device_coord);

    // ── Arch-aware defaults ──────────────────────────────────────────────────
    const bool is_wh = (device->arch() == tt::ARCH::WORMHOLE_B0);
    if (!cfg.user_chunk_size) {
        cfg.chunk_size = is_wh ? 8u * 1024u : 16u * 1024u;
    }

    // ── Pinned memory check ─────────────────────────────────────────────────
    // Attempt a probe allocation rather than relying on can_map_to_noc
    // (which is overly conservative when IOMMU is disabled but BH 64-bit
    // PCIe addressing still works).
    try {
        auto probe = make_pinned_buf(mesh_device, device, cfg.chunk_size);
        (void)probe;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: PinnedMemory with NOC mapping not available: " << e.what() << "\n"
                  << "       persistent_dma_bench requires pinned host memory.\n";
        mesh_device->close();
        return 1;
    }

    std::cout << "persistent_dma_bench"
              << "  arch=" << (is_wh ? "WH" : "BH") << "  iters=" << cfg.iters << "  warmup=" << cfg.warmup
              << "  chunk=" << cfg.chunk_size / 1024 << "KB"
              << "  h2d_bufs=" << cfg.h2d_num_bufs << "  ring=" << cfg.ring_depth << "  cores=";
    for (size_t i = 0; i < cfg.core_counts.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << cfg.core_counts[i];
    }
    std::cout << "\n";

    std::vector<Result> all_results;

    // ── Per core-count sweep ─────────────────────────────────────────────────
    for (uint32_t num_cores : cfg.core_counts) {
        if (num_cores == 0 || num_cores > kDmaNumCores) {
            std::cerr << "WARNING: skipping invalid core count " << num_cores << " (must be 1–" << kDmaNumCores
                      << ")\n";
            continue;
        }

        // Determine valid sizes for this core count.
        std::vector<size_t> valid_sizes;
        for (size_t s : cfg.sizes) {
            if (s % (static_cast<size_t>(num_cores) * cfg.chunk_size) == 0) {
                valid_sizes.push_back(s);
            }
        }
        if (valid_sizes.empty()) {
            std::cerr << "NOTE: no sizes divisible by " << num_cores << " * " << cfg.chunk_size << " — skipping "
                      << num_cores << " cores\n";
            continue;
        }

        size_t max_bytes = *std::max_element(valid_sizes.begin(), valid_sizes.end());

        // ── Allocate DRAM buffers BEFORE creating DmaEngine ──────────────────
        // Finish() must not run while persistent kernels are active on the CQ.
        std::shared_ptr<MeshBuffer> d2h_dram, h2d_dram;

        if (cfg.dir_d2h || cfg.dir_bidir) {
            // L1-only mode never reads DRAM — allocate a minimal dummy buffer to
            // satisfy the command struct (address is passed but kernel skips DRAM).
            const size_t d2h_dram_alloc = cfg.mode_dram ? max_bytes : cfg.chunk_size;
            d2h_dram = make_dram_buf(mesh_device, d2h_dram_alloc, cfg.chunk_size);
            if (cfg.mode_dram) {
                // Fill with sequential data so D2H transfers have valid content.
                std::vector<uint32_t> fill(max_bytes / sizeof(uint32_t));
                std::iota(fill.begin(), fill.end(), 0u);
                WriteShard(mesh_device->mesh_command_queue(), d2h_dram, fill, device_coord);
            }
        }
        if (cfg.dir_h2d || cfg.dir_h2d_push || cfg.dir_h2d_dma || cfg.dir_bidir) {
            const size_t h2d_dram_alloc = cfg.mode_dram ? max_bytes : cfg.chunk_size;
            h2d_dram = make_dram_buf(mesh_device, h2d_dram_alloc, cfg.chunk_size);
        }
        Finish(mesh_device->mesh_command_queue());

        // ── Create DmaEngine (launches kernels) ──────────────────────────────
        DmaEngine engine(
            mesh_device,
            DmaEngineConfig{
                .num_cores = num_cores,
                .chunk_size = cfg.chunk_size,
                .h2d_num_bufs = cfg.h2d_num_bufs,
                .ring_depth = cfg.ring_depth,
                .enable_h2d_push = cfg.dir_h2d_push,
            });

        // ── Allocate pinned host buffers ─────────────────────────────────────
        const size_t flag_region = engine.flag_region_bytes();

        PinnedHostBuf d2h_dst, h2d_src;
        uint64_t d2h_flags_phys = 0;
        volatile uint32_t* d2h_flags_ptr = nullptr;

        if (cfg.dir_d2h || cfg.dir_bidir) {
            d2h_dst = make_pinned_buf(mesh_device, device, max_bytes + flag_region);
            d2h_flags_phys = d2h_dst.phys + max_bytes;
            d2h_flags_ptr = reinterpret_cast<volatile uint32_t*>(static_cast<uint8_t*>(d2h_dst.ptr) + max_bytes);
        }
        if (cfg.dir_h2d || cfg.dir_bidir) {
            h2d_src = make_pinned_buf(mesh_device, device, max_bytes);
            uint32_t* p = static_cast<uint32_t*>(h2d_src.ptr);
            for (size_t i = 0; i < max_bytes / sizeof(uint32_t); ++i) {
                p[i] = i;
            }
        }

        // ── H2D push host buffer (regular aligned malloc, not pinned) ────────
        void* push_buf = nullptr;
        std::vector<size_t> push_valid_sizes;
        if (cfg.dir_h2d_push && engine.has_h2d_push()) {
            const uint32_t pnc = engine.push_num_cores();
            for (size_t s : cfg.sizes) {
                if (s % (static_cast<size_t>(pnc) * cfg.chunk_size) == 0) {
                    push_valid_sizes.push_back(s);
                }
            }
            if (!push_valid_sizes.empty()) {
                size_t push_max = *std::max_element(push_valid_sizes.begin(), push_valid_sizes.end());
                push_buf = aligned_alloc(64, push_max);
                TT_FATAL(push_buf != nullptr, "aligned_alloc failed for push buffer");
                auto* pp = static_cast<uint32_t*>(push_buf);
                for (size_t i = 0; i < push_max / sizeof(uint32_t); ++i) {
                    pp[i] = i;
                }
            }
        }

        // ── collect() helper ─────────────────────────────────────────────────

        auto collect = [&](const std::string& label,
                           const std::string& dir,
                           const std::string& mode_str,
                           uint32_t opcode,
                           bool is_d2h,
                           bool is_bidir) {
            for (size_t total_bytes : valid_sizes) {
                if (cfg.verbose) {
                    std::cout << "  [" << label << " " << fmt_bytes(total_bytes) << " cores=" << num_cores << "]\n";
                }

                // Warmup
                for (int w = 0; w < cfg.warmup; ++w) {
                    if (is_bidir) {
                        run_bidir_once(
                            engine,
                            num_cores,
                            d2h_dram,
                            d2h_dst,
                            d2h_flags_phys,
                            d2h_flags_ptr,
                            h2d_src,
                            h2d_dram,
                            total_bytes,
                            opcode);
                    } else if (is_d2h) {
                        run_d2h_once(
                            engine, num_cores, d2h_dram, d2h_dst, d2h_flags_phys, d2h_flags_ptr, total_bytes, opcode);
                    } else {
                        run_h2d_once(engine, num_cores, h2d_src, h2d_dram, total_bytes, opcode);
                    }
                }

                // Timed iterations
                if (is_bidir) {
                    // Bidir produces two Result records (d2h + h2d).
                    std::vector<double> d2h_samples, h2d_samples;
                    for (int it = 0; it < cfg.iters; ++it) {
                        auto [d2h_bw, h2d_bw] = run_bidir_once(
                            engine,
                            num_cores,
                            d2h_dram,
                            d2h_dst,
                            d2h_flags_phys,
                            d2h_flags_ptr,
                            h2d_src,
                            h2d_dram,
                            total_bytes,
                            opcode);
                        d2h_samples.push_back(d2h_bw);
                        h2d_samples.push_back(h2d_bw);
                    }
                    Result rd2h{
                        label + "_D2H",
                        "bidir_d2h",
                        mode_str,
                        num_cores,
                        total_bytes,
                        cfg.iters,
                        cfg.chunk_size,
                        cfg.h2d_num_bufs,
                        cfg.ring_depth,
                        d2h_samples};
                    Result rh2d{
                        label + "_H2D",
                        "bidir_h2d",
                        mode_str,
                        num_cores,
                        total_bytes,
                        cfg.iters,
                        cfg.chunk_size,
                        cfg.h2d_num_bufs,
                        cfg.ring_depth,
                        h2d_samples};
                    if (cfg.verbose) {
                        print_verbose_samples(rd2h);
                        print_verbose_samples(rh2d);
                    }
                    all_results.push_back(std::move(rd2h));
                    all_results.push_back(std::move(rh2d));
                } else {
                    std::vector<double> samples;
                    for (int it = 0; it < cfg.iters; ++it) {
                        double bw = is_d2h ? run_d2h_once(
                                                 engine,
                                                 num_cores,
                                                 d2h_dram,
                                                 d2h_dst,
                                                 d2h_flags_phys,
                                                 d2h_flags_ptr,
                                                 total_bytes,
                                                 opcode)
                                           : run_h2d_once(engine, num_cores, h2d_src, h2d_dram, total_bytes, opcode);
                        samples.push_back(bw);
                    }
                    Result r{
                        label,
                        is_d2h ? "d2h" : "h2d",
                        mode_str,
                        num_cores,
                        total_bytes,
                        cfg.iters,
                        cfg.chunk_size,
                        cfg.h2d_num_bufs,
                        cfg.ring_depth,
                        samples};
                    if (cfg.verbose) {
                        print_verbose_samples(r);
                        if (!is_d2h) {
                            engine.dump_h2d_timing(total_bytes);
                        }
                    }
                    all_results.push_back(std::move(r));
                }
            }
        };

        // ── Run selected directions × modes ──────────────────────────────────
        if (cfg.dir_d2h) {
            if (cfg.mode_dram) {
                collect(
                    "D2H_DRAM",
                    "d2h",
                    "dram",
                    DMA_OP_TRANSFER,
                    /*is_d2h=*/true,
                    /*is_bidir=*/false);
            }
            if (cfg.mode_l1) {
                collect(
                    "D2H_L1",
                    "d2h",
                    "l1",
                    DMA_OP_L1_ONLY,
                    /*is_d2h=*/true,
                    /*is_bidir=*/false);
            }
        }
        if (cfg.dir_h2d) {
            if (cfg.mode_dram) {
                collect(
                    "H2D_DRAM",
                    "h2d",
                    "dram",
                    DMA_OP_TRANSFER,
                    /*is_d2h=*/false,
                    /*is_bidir=*/false);
            }
            if (cfg.mode_l1) {
                collect(
                    "H2D_L1",
                    "h2d",
                    "l1",
                    DMA_OP_L1_ONLY,
                    /*is_d2h=*/false,
                    /*is_bidir=*/false);
            }
        }
        if (cfg.dir_h2d_push && engine.has_h2d_push() && push_buf && !push_valid_sizes.empty()) {
            const uint32_t pnc = engine.push_num_cores();
            auto run_push_sizes = [&](const std::string& label, const std::string& mode_str, uint32_t opcode) {
                for (size_t total_bytes : push_valid_sizes) {
                    if (cfg.verbose) {
                        std::cout << "  [" << label << " " << fmt_bytes(total_bytes) << " push_cores=" << pnc << "]\n";
                    }
                    for (int w = 0; w < cfg.warmup; ++w) {
                        run_h2d_push_once(engine, push_buf, h2d_dram, total_bytes, opcode);
                    }
                    std::vector<double> samples;
                    for (int it = 0; it < cfg.iters; ++it) {
                        samples.push_back(run_h2d_push_once(engine, push_buf, h2d_dram, total_bytes, opcode));
                    }
                    Result r{
                        label,
                        "h2d_push",
                        mode_str,
                        pnc,
                        total_bytes,
                        cfg.iters,
                        cfg.chunk_size,
                        cfg.h2d_num_bufs,
                        cfg.ring_depth,
                        samples};
                    if (cfg.verbose) {
                        print_verbose_samples(r);
                    }
                    all_results.push_back(std::move(r));
                }
            };
            if (cfg.mode_dram) {
                run_push_sizes("H2D_PUSH_DRAM", "dram", DMA_OP_H2D_PUSH);
            }
            if (cfg.mode_l1) {
                run_push_sizes("H2D_PUSH_L1", "l1", DMA_OP_H2D_PUSH);
            }
        }
        if (push_buf) {
            std::free(push_buf);
            push_buf = nullptr;
        }

        // ── H2D eDMA engine (WH DesignWare PCIe DMA via BAR2) ─────────────
        if (cfg.dir_h2d_dma && engine.has_hw_dma()) {
            // eDMA writes to core 0's L1 via TLB-translated AXI address.
            // For bandwidth testing we overwrite the same L1 area repeatedly;
            // the goal is measuring raw eDMA throughput, not data correctness.
            void* dma_src = aligned_alloc(64, max_bytes);
            TT_FATAL(dma_src != nullptr, "aligned_alloc failed for DMA source buffer");
            auto* dp = static_cast<uint32_t*>(dma_src);
            for (size_t i = 0; i < max_bytes / sizeof(uint32_t); ++i) {
                dp[i] = i;
            }

            auto run_dma_sizes = [&](const std::string& label, const std::string& mode_str) {
                for (size_t total_bytes : valid_sizes) {
                    if (cfg.verbose) {
                        std::cout << "  [" << label << " " << fmt_bytes(total_bytes) << "]\n";
                    }
                    for (int w = 0; w < cfg.warmup; ++w) {
                        run_h2d_dma_once(engine, dma_src, total_bytes);
                    }
                    std::vector<double> samples;
                    for (int it = 0; it < cfg.iters; ++it) {
                        samples.push_back(run_h2d_dma_once(engine, dma_src, total_bytes));
                    }
                    Result r{
                        label,
                        "h2d_dma",
                        mode_str,
                        1,
                        total_bytes,
                        cfg.iters,
                        cfg.chunk_size,
                        cfg.h2d_num_bufs,
                        cfg.ring_depth,
                        samples};
                    if (cfg.verbose) {
                        print_verbose_samples(r);
                    }
                    all_results.push_back(std::move(r));
                }
            };
            run_dma_sizes("H2D_EDMA", "l1");

            // Zero-copy variant: raw eDMA throughput without memcpy overhead.
            auto run_dma_zc_sizes = [&](const std::string& label, const std::string& mode_str) {
                for (size_t total_bytes : valid_sizes) {
                    if (cfg.verbose) {
                        std::cout << "  [" << label << " " << fmt_bytes(total_bytes) << "]\n";
                    }
                    for (int w = 0; w < cfg.warmup; ++w) {
                        run_h2d_dma_zc_once(engine, dma_src, total_bytes);
                    }
                    std::vector<double> samples;
                    for (int it = 0; it < cfg.iters; ++it) {
                        samples.push_back(run_h2d_dma_zc_once(engine, dma_src, total_bytes));
                    }
                    Result r{
                        label,
                        "h2d_dma_zc",
                        mode_str,
                        1,
                        total_bytes,
                        cfg.iters,
                        cfg.chunk_size,
                        cfg.h2d_num_bufs,
                        cfg.ring_depth,
                        samples};
                    if (cfg.verbose) {
                        print_verbose_samples(r);
                    }
                    all_results.push_back(std::move(r));
                }
            };
            run_dma_zc_sizes("H2D_EDMA_ZC", "l1");

            std::free(dma_src);
        } else if (cfg.dir_h2d_dma && !engine.has_hw_dma()) {
            std::cerr << "WARNING: --h2d-dma requested but HW eDMA engine not available.\n"
                      << "         This feature requires Wormhole with an allocated DMA buffer.\n";
        }

        // ── D2H eDMA engine (WH DesignWare PCIe DMA via BAR2) ─────────────
        if (cfg.dir_d2h_dma && engine.has_hw_dma()) {
            void* dma_dst = aligned_alloc(64, max_bytes);
            TT_FATAL(dma_dst != nullptr, "aligned_alloc failed for DMA destination buffer");
            std::memset(dma_dst, 0, max_bytes);

            auto run_d2h_dma_sizes = [&](const std::string& label, const std::string& mode_str) {
                for (size_t total_bytes : valid_sizes) {
                    if (cfg.verbose) {
                        std::cout << "  [" << label << " " << fmt_bytes(total_bytes) << "]\n";
                    }
                    for (int w = 0; w < cfg.warmup; ++w) {
                        run_d2h_dma_once(engine, dma_dst, total_bytes);
                    }
                    std::vector<double> samples;
                    for (int it = 0; it < cfg.iters; ++it) {
                        samples.push_back(run_d2h_dma_once(engine, dma_dst, total_bytes));
                    }
                    Result r{
                        label,
                        "d2h_dma",
                        mode_str,
                        1,
                        total_bytes,
                        cfg.iters,
                        cfg.chunk_size,
                        cfg.h2d_num_bufs,
                        cfg.ring_depth,
                        samples};
                    if (cfg.verbose) {
                        print_verbose_samples(r);
                    }
                    all_results.push_back(std::move(r));
                }
            };
            run_d2h_dma_sizes("D2H_EDMA", "l1");

            auto run_d2h_dma_zc_sizes = [&](const std::string& label, const std::string& mode_str) {
                for (size_t total_bytes : valid_sizes) {
                    if (cfg.verbose) {
                        std::cout << "  [" << label << " " << fmt_bytes(total_bytes) << "]\n";
                    }
                    for (int w = 0; w < cfg.warmup; ++w) {
                        run_d2h_dma_zc_once(engine, total_bytes);
                    }
                    std::vector<double> samples;
                    for (int it = 0; it < cfg.iters; ++it) {
                        samples.push_back(run_d2h_dma_zc_once(engine, total_bytes));
                    }
                    Result r{
                        label,
                        "d2h_dma_zc",
                        mode_str,
                        1,
                        total_bytes,
                        cfg.iters,
                        cfg.chunk_size,
                        cfg.h2d_num_bufs,
                        cfg.ring_depth,
                        samples};
                    if (cfg.verbose) {
                        print_verbose_samples(r);
                    }
                    all_results.push_back(std::move(r));
                }
            };
            run_d2h_dma_zc_sizes("D2H_EDMA_ZC", "l1");

            std::free(dma_dst);
        } else if (cfg.dir_d2h_dma && !engine.has_hw_dma()) {
            std::cerr << "WARNING: --d2h-dma requested but HW eDMA engine not available.\n"
                      << "         This feature requires Wormhole with an allocated DMA buffer.\n";
        }

        // ── Scattered L1 benchmarks (eDMA + TLB retarget) ───────────────
        if ((cfg.dir_d2h_scatter_l1 || cfg.dir_h2d_scatter_l1) && engine.has_hw_dma()) {
            // Build list of worker cores to scatter across.
            // Use the compute grid — same cores that DMA engine uses, but
            // we target a low L1 offset (0x0) that doesn't conflict with
            // the persistent kernel's region at kDmaUserBase (0x80000).
            const auto grid = device->compute_with_storage_grid_size();
            const uint32_t scatter_cores = std::min(num_cores, static_cast<uint32_t>(grid.x * grid.y));
            std::vector<CoreCoord> scatter_core_list;
            scatter_core_list.reserve(scatter_cores);
            for (uint32_t r = 0; r < static_cast<uint32_t>(grid.y) && scatter_core_list.size() < scatter_cores; ++r) {
                for (uint32_t c = 0; c < static_cast<uint32_t>(grid.x) && scatter_core_list.size() < scatter_cores;
                     ++c) {
                    scatter_core_list.emplace_back(c, r);
                }
            }

            // H2D writes must avoid kernel firmware (~0x0–0x20000) AND the DMA
            // engine's command/buffer region (0x80000+).  The gap 0x20000–0x80000
            // (384KB) is safe unused L1.  D2H reads are non-destructive but we
            // use the same offset so both directions can share the same size list.
            constexpr uint32_t kScatterL1Offset = 0x20000u;
            constexpr uint32_t kL1UserBase = 0x80000u;                        // kDmaUserBase
            constexpr size_t kMaxShardSize = kL1UserBase - kScatterL1Offset;  // 384KB

            // Filter sizes: total must be divisible by scatter_cores AND
            // per-core shard must fit within L1 from the offset.
            std::vector<size_t> scatter_sizes;
            for (size_t s : cfg.sizes) {
                if (s % scatter_cores != 0) {
                    continue;
                }
                size_t shard = s / scatter_cores;
                if (shard > kMaxShardSize) {
                    if (cfg.verbose) {
                        std::cerr << "NOTE: " << fmt_bytes(s) << " skipped for scatter L1 — shard " << fmt_bytes(shard)
                                  << " exceeds L1 capacity " << fmt_bytes(kMaxShardSize) << " from offset 0x"
                                  << std::hex << kScatterL1Offset << std::dec << "\n";
                    }
                    continue;
                }
                scatter_sizes.push_back(s);
            }

            if (scatter_sizes.empty()) {
                std::cerr << "NOTE: no sizes divisible by " << scatter_cores
                          << " scatter cores — skipping scatter L1\n";
            } else {
                auto run_scatter = [&](const std::string& label, const std::string& dir, bool is_d2h) {
                    for (size_t total_bytes : scatter_sizes) {
                        const size_t shard_size = total_bytes / scatter_cores;
                        if (cfg.verbose) {
                            std::cout << "  [" << label << " " << fmt_bytes(total_bytes)
                                      << " scatter_cores=" << scatter_cores << " shard=" << fmt_bytes(shard_size)
                                      << "]\n";
                        }
                        for (int w = 0; w < cfg.warmup; ++w) {
                            if (is_d2h) {
                                run_d2h_scatter_l1_once(
                                    engine, scatter_core_list, kScatterL1Offset, shard_size, total_bytes);
                            } else {
                                run_h2d_scatter_l1_once(
                                    engine, scatter_core_list, kScatterL1Offset, shard_size, total_bytes);
                            }
                        }
                        std::vector<double> samples;
                        for (int it = 0; it < cfg.iters; ++it) {
                            if (is_d2h) {
                                samples.push_back(run_d2h_scatter_l1_once(
                                    engine, scatter_core_list, kScatterL1Offset, shard_size, total_bytes));
                            } else {
                                samples.push_back(run_h2d_scatter_l1_once(
                                    engine, scatter_core_list, kScatterL1Offset, shard_size, total_bytes));
                            }
                        }
                        Result r{
                            label,
                            dir,
                            "l1",
                            scatter_cores,
                            total_bytes,
                            cfg.iters,
                            cfg.chunk_size,
                            cfg.h2d_num_bufs,
                            cfg.ring_depth,
                            samples};
                        if (cfg.verbose) {
                            print_verbose_samples(r);
                        }
                        all_results.push_back(std::move(r));
                    }
                };

                if (cfg.dir_d2h_scatter_l1) {
                    run_scatter("D2H_SCATTER_L1", "d2h_scatter_l1", /*is_d2h=*/true);
                }
                if (cfg.dir_h2d_scatter_l1) {
                    run_scatter("H2D_SCATTER_L1", "h2d_scatter_l1", /*is_d2h=*/false);
                }
            }
        } else if ((cfg.dir_d2h_scatter_l1 || cfg.dir_h2d_scatter_l1) && !engine.has_hw_dma()) {
            std::cerr << "WARNING: --d2h-scatter-l1/--h2d-scatter-l1 requested but HW eDMA not available.\n";
        }

        // ── Interleaved DRAM benchmarks (eDMA cycling across banks) ─────
        if ((cfg.dir_d2h_interleaved || cfg.dir_h2d_interleaved) && engine.has_hw_dma()) {
            // WH has 6 DRAM banks (logical coords (0,0)–(5,0))
            constexpr uint32_t kNumDramBanks = 6;
            constexpr uint32_t kDramOffset = 0x100000;  // 1MB into each bank (avoid addr 0)

            // Filter sizes: total must be divisible by num_banks
            std::vector<size_t> interleaved_sizes;
            for (size_t s : cfg.sizes) {
                if (s % kNumDramBanks == 0) {
                    interleaved_sizes.push_back(s);
                }
            }

            if (interleaved_sizes.empty()) {
                std::cerr << "NOTE: no sizes divisible by " << kNumDramBanks
                          << " DRAM banks — skipping interleaved DRAM\n";
            } else {
                auto run_interleaved = [&](const std::string& label, const std::string& dir, bool is_d2h) {
                    for (size_t total_bytes : interleaved_sizes) {
                        if (cfg.verbose) {
                            std::cout << "  [" << label << " " << fmt_bytes(total_bytes) << " banks=" << kNumDramBanks
                                      << " per_bank=" << fmt_bytes(total_bytes / kNumDramBanks) << "]\n";
                        }
                        for (int w = 0; w < cfg.warmup; ++w) {
                            if (is_d2h) {
                                run_d2h_interleaved_once(engine, kNumDramBanks, kDramOffset, total_bytes);
                            } else {
                                run_h2d_interleaved_once(engine, kNumDramBanks, kDramOffset, total_bytes);
                            }
                        }
                        std::vector<double> samples;
                        for (int it = 0; it < cfg.iters; ++it) {
                            if (is_d2h) {
                                samples.push_back(
                                    run_d2h_interleaved_once(engine, kNumDramBanks, kDramOffset, total_bytes));
                            } else {
                                samples.push_back(
                                    run_h2d_interleaved_once(engine, kNumDramBanks, kDramOffset, total_bytes));
                            }
                        }
                        Result r{
                            label,
                            dir,
                            "dram",
                            kNumDramBanks,
                            total_bytes,
                            cfg.iters,
                            cfg.chunk_size,
                            cfg.h2d_num_bufs,
                            cfg.ring_depth,
                            samples};
                        if (cfg.verbose) {
                            print_verbose_samples(r);
                        }
                        all_results.push_back(std::move(r));
                    }
                };

                if (cfg.dir_d2h_interleaved) {
                    run_interleaved("D2H_INTLV_DRAM", "d2h_interleaved", /*is_d2h=*/true);
                }
                if (cfg.dir_h2d_interleaved) {
                    run_interleaved("H2D_INTLV_DRAM", "h2d_interleaved", /*is_d2h=*/false);
                }
            }
        } else if ((cfg.dir_d2h_interleaved || cfg.dir_h2d_interleaved) && !engine.has_hw_dma()) {
            std::cerr << "WARNING: --d2h-interleaved/--h2d-interleaved requested but HW eDMA not available.\n";
        }

        if (cfg.dir_bidir) {
            if (cfg.mode_dram) {
                collect(
                    "BIDIR_DRAM",
                    "bidir",
                    "dram",
                    DMA_OP_TRANSFER,
                    /*is_d2h=*/false,
                    /*is_bidir=*/true);
            }
            if (cfg.mode_l1) {
                collect(
                    "BIDIR_L1",
                    "bidir",
                    "l1",
                    DMA_OP_L1_ONLY,
                    /*is_d2h=*/false,
                    /*is_bidir=*/true);
            }
        }

        // DmaEngine destructor sends EXIT to all cores and waits for Finish().
    }

    // ── Output ───────────────────────────────────────────────────────────────
    if (!all_results.empty()) {
        if (!cfg.no_table) {
            print_table(all_results);
        }
        if (cfg.csv) {
            print_csv(all_results);
        }
        if (cfg.json_out) {
            print_json(all_results);
        }
    } else {
        std::cerr << "No results collected (check --cores / --sizes alignment).\n";
    }

    mesh_device->close();
    return 0;
}
