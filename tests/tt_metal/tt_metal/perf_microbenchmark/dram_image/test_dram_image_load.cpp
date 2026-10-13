// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host -> device DRAM bulk-load bandwidth for weight images, three ways:
//   enqueue : EnqueueWriteMeshBuffer from an ordinary host vector (host pushes through the hugepage).
//   pinned  : enqueue_write_shards from NOC-mapped pinned memory (the prefetcher pulls over PCIe).
//   pull    : a kernel on N Tensix cores pulls 64 KB pages from the pinned buffer over PCIe and writes them
//             into the interleaved DRAM buffer itself.
// Every mode is verified by reading the DRAM buffer back and comparing with the host image.
//
// The image lives in 1 GB hugepages (one PinnedMemory each): without an IOMMU only physically contiguous memory pins.
//
// Usage: test_dram_image_load [--size-mb 2048] [--page-kb 64] [--cores 32] [--iters 3] [--modes enqueue,pinned,pull]

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>
#include "tt_metal/impl/dispatch/vector_aligned.hpp"

using namespace tt;
using namespace tt::tt_metal;
using Clock = std::chrono::steady_clock;

namespace {

struct Opt {
    uint64_t size_mb = 2048;
    uint32_t page_kb = 64;
    uint32_t cores = 32;
    uint32_t iters = 3;
    std::string modes = "enqueue,pinned,pull";
};

Opt parse(int argc, char** argv) {
    Opt o;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--size-mb") {
            o.size_mb = std::stoull(v);
        } else if (k == "--page-kb") {
            o.page_kb = std::stoul(v);
        } else if (k == "--cores") {
            o.cores = std::stoul(v);
        } else if (k == "--iters") {
            o.iters = std::stoul(v);
        } else if (k == "--modes") {
            o.modes = v;
        } else {
            TT_THROW("unknown arg {}", k);
        }
    }
    return o;
}

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// cheap position-dependent pattern, filled in parallel
uint32_t pat(uint64_t i, uint32_t salt) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t(salt) << 32);
    x ^= x >> 29;
    return uint32_t(x);
}

template <typename F>
void parallel_for(uint64_t n, F f) {
    unsigned t = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned k = 0; k < t; k++) {
        th.emplace_back([&, k] {
            uint64_t lo = n * k / t, hi = n * (k + 1) / t;
            f(lo, hi);
        });
    }
    for (auto& x : th) {
        x.join();
    }
}

[[maybe_unused]] bool verify(
    distributed::MeshDevice& mesh,
    std::shared_ptr<distributed::MeshBuffer>& buf,
    const uint32_t* ref,
    uint64_t words,
    const char* tag) {
    std::vector<uint32_t> back;
    auto t0 = Clock::now();
    distributed::EnqueueReadMeshBuffer(mesh.mesh_command_queue(), back, buf, true);
    auto t1 = Clock::now();
    std::atomic<uint64_t> bad{0}, first_bad{~0ull};
    parallel_for(words, [&](uint64_t lo, uint64_t hi) {
        for (uint64_t i = lo; i < hi; i++) {
            if (back[i] != ref[i]) {
                bad++;
                uint64_t cur = first_bad.load();
                while (i < cur && !first_bad.compare_exchange_weak(cur, i)) {
                }
            }
        }
    });
    fmt::print(
        "  [{}] verify: readback {:.2f} GB/s, {} bad words{}\n",
        tag,
        words * 4 / secs(t0, t1) / 1e9,
        bad.load(),
        bad ? fmt::format(" (first at byte {:#x})", first_bad.load() * 4) : "");
    return bad == 0;
}

}  // namespace

namespace {
constexpr uint64_t GB = 1ull << 30;

// one 1 GB hugetlbfs page, mapped and populated (physically contiguous, so pinnable without an IOMMU)
uint32_t* map_hugepage_1g(int idx) {
    std::string path = fmt::format("/dev/hugepages-1G/dram_image_{}_{}", getpid(), idx);
    int fd = open(path.c_str(), O_CREAT | O_RDWR, 0600);
    TT_FATAL(fd >= 0, "open {} failed", path);
    void* p = mmap(nullptr, GB, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    unlink(path.c_str());  // the mapping keeps the page
    TT_FATAL(p != MAP_FAILED, "mmap of a 1 GB hugepage failed (free hugepages?)");
    return static_cast<uint32_t*>(p);
}
}  // namespace

int main(int argc, char** argv) {
    Opt o = parse(argc, argv);
    const uint64_t page = uint64_t(o.page_kb) * 1024;
    TT_FATAL(GB % page == 0, "page must divide 1 GB");
    const uint32_t n_chunks = std::max<uint64_t>(1, o.size_mb / 1024);
    const uint64_t size = n_chunks * GB;
    const uint64_t words = size / 4, chunk_words = GB / 4;
    const uint32_t num_pages = size / page, chunk_pages = GB / page;

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    const distributed::MeshCoordinate coord(0, 0);
    const distributed::MeshCoordinateRangeSet here(distributed::MeshCoordinateRange(coord, coord));
    IDevice* dev = mesh->get_device(0, 0);

    auto pp = experimental::GetMemoryPinningParameters(*mesh);
    fmt::print(
        "pinning: max_pins={} max_total={:.1f} GB can_map_to_noc={}\n",
        pp.max_pins,
        pp.max_total_pin_size / 1e9,
        pp.can_map_to_noc);
    fmt::print(
        "image {} x 1 GB hugepages, page {} KB ({} pages), cores {}, grid {}x{}\n",
        n_chunks,
        o.page_kb,
        num_pages,
        o.cores,
        dev->compute_with_storage_grid_size().x,
        dev->compute_with_storage_grid_size().y);

    std::vector<uint32_t*> chunk(n_chunks);
    for (uint32_t c = 0; c < n_chunks; c++) {
        chunk[c] = map_hugepage_1g(c);
    }
    auto t0 = Clock::now();
    for (uint32_t c = 0; c < n_chunks; c++) {
        parallel_for(chunk_words, [&](uint64_t lo, uint64_t hi) {
            for (uint64_t i = lo; i < hi; i++) {
                chunk[c][i] = pat(c * chunk_words + i, 7);
            }
        });
    }
    fmt::print("host fill {:.2f} s\n", secs(t0, Clock::now()));

    distributed::DeviceLocalBufferConfig local{.page_size = page, .buffer_type = BufferType::DRAM, .bottom_up = true};
    distributed::ReplicatedBufferConfig global{.size = size};
    auto buf = distributed::MeshBuffer::create(global, local, mesh.get());
    fmt::print("DRAM buffer at {:#x}\n", buf->address());

    auto has = [&](const char* m) { return o.modes.find(m) != std::string::npos; };
    auto check = [&](const char* tag) {
        std::vector<uint32_t> back;
        auto a = Clock::now();
        distributed::EnqueueReadMeshBuffer(cq, back, buf, true);
        double rs = secs(a, Clock::now());
        std::atomic<uint64_t> bad{0};
        for (uint32_t c = 0; c < n_chunks; c++) {
            parallel_for(chunk_words, [&](uint64_t lo, uint64_t hi) {
                uint64_t b = 0;
                for (uint64_t i = lo; i < hi; i++) {
                    b += back[c * chunk_words + i] != chunk[c][i];
                }
                bad += b;
            });
        }
        fmt::print("  [{}] verify: readback {:.2f} GB/s, {} bad words\n", tag, size / rs / 1e9, bad.load());
    };
    auto zero_dram = [&] {
        std::vector<uint32_t> z(words, 0);
        distributed::EnqueueWriteMeshBuffer(cq, buf, z, true);
    };
    auto report = [&](const char* tag, uint32_t it, double s) {
        fmt::print("{:8s} iter {}: {:.3f} s  {:.2f} GB/s\n", tag, it, s, size / s / 1e9);
    };

    // ---- enqueue: ordinary host push from a pageable vector
    if (has("enqueue")) {
        std::vector<uint32_t> src(words);
        for (uint32_t c = 0; c < n_chunks; c++) {
            std::memcpy(src.data() + c * chunk_words, chunk[c], GB);
        }
        for (uint32_t it = 0; it < o.iters; it++) {
            auto a = Clock::now();
            distributed::EnqueueWriteMeshBuffer(cq, buf, src, true);
            report("enqueue", it, secs(a, Clock::now()));
        }
        check("enqueue");
    }

    // ---- pin every chunk
    std::vector<std::shared_ptr<experimental::PinnedMemory>> pin(n_chunks);
    if (has("pinned") || has("pull")) {
        auto a = Clock::now();
        for (uint32_t c = 0; c < n_chunks; c++) {
            HostBuffer hb(ttsl::Span<uint32_t>(chunk[c], chunk_words), MemoryPin(std::shared_ptr<void>()));
            pin[c] = experimental::PinnedMemory::Create(*mesh, here, hb, true);
        }
        fmt::print("pin {} GB: {:.3f} s\n", n_chunks, secs(a, Clock::now()));
    }

    // ---- pinned: the dispatch prefetcher pulls each chunk over PCIe
    if (has("pinned")) {
        zero_dram();
        for (uint32_t it = 0; it < o.iters; it++) {
            std::vector<distributed::ShardDataTransfer> trs;
            auto a = Clock::now();
            for (uint32_t c = 0; c < n_chunks; c++) {
                distributed::ShardDataTransfer tr{coord};
                tr.host_data(chunk[c]).region(BufferRegion(c * GB, GB));
                experimental::ShardDataTransferSetPinnedMemory(tr, pin[c]);
                cq.enqueue_write_shards(buf, {tr}, false);
            }
            distributed::Finish(cq);
            report("pinned", it, secs(a, Clock::now()));
        }
        check("pinned");
    }

    // ---- pull: our multi-core kernel, one workload per chunk
    if (has("pull")) {
        zero_dram();
        const uint32_t burst = 16384;
        TT_FATAL(page % burst == 0, "page must be a multiple of {}", burst);
        auto grid = dev->compute_with_storage_grid_size();
        TT_FATAL(o.cores <= grid.x * grid.y, "too many cores");
        std::vector<CoreCoord> cores;
        std::set<CoreRange> ranges;
        for (uint32_t i = 0; i < o.cores; i++) {
            cores.push_back({i % grid.x, i / grid.x});
            ranges.insert(CoreRange(cores.back(), cores.back()));
        }
        CoreRangeSet core_set(ranges);

        std::vector<distributed::MeshWorkload> wls(n_chunks);
        for (uint32_t c = 0; c < n_chunks; c++) {
            auto na = pin[c]->get_noc_addr(dev->id());
            TT_FATAL(na.has_value() && na->device_id == dev->id(), "chunk {} not NOC-mapped for this chip", c);
            if (c == 0) {
                fmt::print("chunk 0 NOC addr {:#x} xy_enc {:#x}\n", na->addr, na->pcie_xy_enc);
            }
            Program prog = CreateProgram();
            CreateCircularBuffer(
                prog,
                core_set,
                CircularBufferConfig(2 * uint32_t(page), {{0, tt::DataFormat::RawUInt32}})
                    .set_page_size(0, uint32_t(page)));
            std::vector<uint32_t> ct = {na->pcie_xy_enc, uint32_t(page), burst};
            TensorAccessorArgs(*buf->get_reference_buffer()).append_to(ct);
            auto k = CreateKernel(
                prog,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dram_image/kernels/pcie_pull_to_dram.cpp",
                core_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct});
            const uint32_t page_off = c * chunk_pages;
            for (uint32_t idx = 0; idx < o.cores; idx++) {
                uint32_t mine = idx < chunk_pages ? (chunk_pages - idx + o.cores - 1) / o.cores : 0;
                SetRuntimeArgs(
                    prog,
                    k,
                    cores[idx],
                    {uint32_t(na->addr & 0xffffffffu),
                     uint32_t(na->addr >> 32),
                     uint32_t(buf->address()),
                     page_off + idx,
                     o.cores,
                     mine,
                     0u,
                     page_off});
            }
            wls[c].add_program(distributed::MeshCoordinateRange(coord, coord), std::move(prog));
        }
        for (uint32_t it = 0; it < o.iters + 1; it++) {
            auto a = Clock::now();
            for (auto& wl : wls) {
                distributed::EnqueueMeshWorkload(cq, wl, false);
            }
            distributed::Finish(cq);
            report(it == 0 ? "pull(c)" : "pull", it, secs(a, Clock::now()));
        }
        check("pull");
    }
    fmt::print("closing\n");
    pin.clear();  // unpin while the device is still open
    buf.reset();
    mesh->close();
    fmt::print("closed\n");
    for (auto* p : chunk) {
        munmap(p, GB);
    }
    return 0;
}
