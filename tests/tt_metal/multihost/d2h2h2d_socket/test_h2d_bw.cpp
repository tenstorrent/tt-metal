// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-device leg alone: one rank, no peer, no H2H. This host fills the RX rings in
// place of a peer's RMA, so the only PCIe traffic is the device pulling frames out of them.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tt_metal/distributed/host_h2d_leg.hpp"
#include "tt_metal/distributed/host_l1_map.hpp"
#include "tt_metal/distributed/host_region.hpp"
#include <tt-metalium/experimental/sockets/host_uva_frame.hpp>
#include <tt-metalium/experimental/sockets/host_uva_layout.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace mh = tt::tt_metal::distributed::multihost;

namespace {

constexpr int kDeviceId = 0;

// A stalled run must fail rather than spin: the receiver kernel exits only once it has seen
// `iters` frames, so Finish() is the other unbounded wait.
constexpr auto kStall = std::chrono::seconds(30);

struct Options {
    uint32_t payload = 16384;
    uint32_t cores = 4;
    uint32_t ring = 8;
    uint32_t iters = 20000;
    uint32_t warmup_pct = 10;
};

// strtoul, not stoul: an uncaught exception here aborts before the device is torn down,
// which leaves the pinned region and the shm rings behind.
bool parse_u32(const char* text, uint32_t& out) {
    errno = 0;
    char* end = nullptr;
    const unsigned long v = std::strtoul(text, &end, 10);
    if (end == text || errno == ERANGE) {
        return false;
    }
    uint64_t mult = 1;
    if (*end != '\0' && end[1] == '\0') {
        switch (*end) {
            case 'K': mult = 1024ull; break;
            case 'M': mult = 1ull << 20; break;
            default: return false;
        }
    } else if (*end != '\0') {
        return false;
    }
    const uint64_t scaled = static_cast<uint64_t>(v) * mult;
    if (scaled == 0 || scaled > UINT32_MAX) {
        return false;
    }
    out = static_cast<uint32_t>(scaled);
    return true;
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_next = i + 1 < argc;
        if (a == "--payload" && has_next && parse_u32(argv[++i], o.payload)) {
            continue;
        }
        if (a == "--cores" && has_next && parse_u32(argv[++i], o.cores)) {
            continue;
        }
        if (a == "--ring" && has_next && parse_u32(argv[++i], o.ring)) {
            continue;
        }
        if (a == "--iters" && has_next && parse_u32(argv[++i], o.iters)) {
            continue;
        }
        if (a == "--warmup-pct" && has_next && parse_u32(argv[++i], o.warmup_pct)) {
            continue;
        }
        std::cerr << "usage: " << argv[0]
                  << " [--payload B] [--cores N] [--ring frames] [--iters N] [--warmup-pct P]\n";
        return false;
    }
    return o.warmup_pct < 100;
}

uint64_t ns_since(std::chrono::steady_clock::time_point t) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count());
}

uint64_t pct(const std::vector<uint64_t>& v, double p) {
    const size_t i = static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5);
    return v[i];
}

struct CoreState {
    uint64_t published = 0;
    uint64_t drained = 0;
    std::vector<std::chrono::steady_clock::time_point> at;
};

}  // namespace

int main(int argc, char** argv) {
    mh::DistributedContext::create(argc, argv);
    Options o;
    if (!parse(argc, argv, o)) {
        return 2;
    }
    if (*mh::DistributedContext::get_current_world()->size() != 1) {
        std::cerr << "error: this leg is local to one host; run it without mpirun or with `-n 1`\n";
        return 2;
    }

    const uint32_t page = tt_uva_frame_page_size(o.payload);
    if (static_cast<uint64_t>(o.ring) * page > kArenaBytes) {
        std::cerr << "error: ring " << o.ring << " x page " << page << " B exceeds the " << (kArenaBytes >> 10)
                  << " KiB arena; lower --ring or --payload\n";
        return 2;
    }

    auto mesh = distributed::MeshDevice::create_unit_mesh(kDeviceId);
    IDevice* device = mesh->get_devices().front();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_width = static_cast<uint32_t>(grid.x);
    if (o.cores == 0 || o.cores > grid_width * static_cast<uint32_t>(grid.y) || o.cores > kProvisionedCores) {
        std::cerr << "error: --cores " << o.cores << " does not fit this grid\n";
        return 2;
    }

    const uint32_t l1_base = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const L1MapNew l1 = L1MapNew::compute(l1_base, static_cast<uint32_t>(device->l1_size_per_core()), o.payload, false);
    if (const std::string e = l1.fits(o.payload); !e.empty()) {
        std::cerr << "error: " << e << "\n";
        return 2;
    }

    // The leg first, then the pin: H2DLeg MAP_FIXEDs its rings over the RX arenas, and a pin
    // taken before that would go on naming the pages it replaced.
    std::string err;
    H2DLeg::Config hc;
    hc.cores = o.cores;
    hc.grid_width = grid_width;
    hc.page_bytes = page;
    hc.ring_pages = o.ring;
    // Maps the region, sized for these cores: the leg MAP_FIXEDs its rings onto it below,
    // so it has to exist before H2DLeg::create(). reserved_base() throws on failure.
    HostRegion& region = HostRegion::storage();
    uint8_t* region_base = nullptr;
    std::unique_ptr<H2DLeg> h2d;
    try {
        region_base = region.reserved_base(o.cores);
        hc.alias_region_base = region_base;
        h2d = H2DLeg::create(mesh, hc, err);
    } catch (const std::exception& ex) {
        std::cerr << "host region unavailable: " << ex.what() << "\n";
        return 1;
    }
    if (!h2d) {
        std::cerr << "h2d bringup failed: " << err << "\n";
        return 1;
    }

    try {
        // try to alias the h2d rings with the region reserved above; this gets registered w/RDMA via MPI_Windows
        region.provision(
            mesh,
            /*chip=*/0,
            o.cores,
            HostTopology{0, 1, 1},
            HostRegion::Grid{grid_width, static_cast<uint32_t>(grid.y)});
        if (const std::string e = region.verify_header(); !e.empty()) {
            std::cerr << "region header check failed: " << e << "\n";
            region.release();
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "host region unavailable: " << ex.what() << "\n";
        return 1;
    }

    // Filled ONCE, not per frame: the device only reads these pages, so republishing a slot
    // needs no rewrite and the timed loop carries no host memcpy.
    for (uint32_t c = 0; c < o.cores; ++c) {
        for (uint32_t s = 0; s < o.ring; ++s) {
            uint8_t* const slot = region_base + rx_slot_offset(c, s, page);
            std::memset(slot, 0, page);
            FrameTrailer* const t = reinterpret_cast<FrameTrailer*>(slot + o.payload);
            t->guard = tt_uva_frame_guard(kFrameVersion);
            t->length = o.payload;
            t->origin = tt_uva_t6_global_selector(0, 0, c, 1);
            // The kernel exits on this word reaching `iters`, so every frame must carry the
            // ADD-1. apply_signal() is the only thing on this leg that reads the trailer.
            t->sig_off = l1.dest_word_addr - l1.l1_base;
            t->sig_val = 1;
            t->sig_op = kSignalAdd;
        }
    }

    CoreRangeSet cores;
    std::vector<CoreCoord> core_list;
    for (uint32_t i = 0; i < o.cores; ++i) {
        const CoreCoord c{i % grid_width, i / grid_width};
        core_list.push_back(c);
        cores = cores.merge(CoreRangeSet(CoreRange(c, c)));
    }

    Program program = CreateProgram();
    auto receiver = CreateKernel(
        program,
        TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_signal.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::NOC_1,
            // verify 0: the slots are never rewritten, so a payload check would only compare
            // the same bytes repeatedly and would put a compare loop in the pull path.
            .compile_args = {
                l1.deliver_addr,
                page,
                l1.dest_word_addr,
                o.iters,
                l1.l1_base,
                l1.l1_size,
                o.payload,
                0u,
                l1.verify_addr}});
    const std::vector<uint32_t> cfg = h2d->config_addresses();
    for (uint32_t i = 0; i < o.cores; ++i) {
        SetRuntimeArgs(program, receiver, core_list[i], {cfg[i], 1u});
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/false);

    const uint64_t total = static_cast<uint64_t>(o.cores) * o.iters;
    const uint64_t warmup = total * o.warmup_pct / 100;

    std::vector<CoreState> state(o.cores);
    for (auto& s : state) {
        s.at.resize(o.ring);
    }
    std::vector<uint64_t> rt_ns;
    rt_ns.reserve(total - warmup);
    uint64_t frames = 0;
    bool ok = true;

    auto t0 = std::chrono::steady_clock::now();
    auto deadline = t0 + kStall;

    while (frames < total && ok) {
        for (uint32_t c = 0; c < o.cores; ++c) {
            while (state[c].published < o.iters) {
                DeliverTask t;
                t.core = c;
                t.slot = static_cast<uint32_t>(state[c].published % o.ring);
                t.page_offset = rx_slot_offset(c, t.slot, page);
                t.page_bytes = page;
                t.length = o.payload;
                // publish() only advances bytes_sent -- the bytes are already in the ring,
                // which is why this leg needs no copy and the pre-fill above is enough.
                if (!h2d->publish(t)) {
                    break;
                }
                state[c].at[t.slot] = std::chrono::steady_clock::now();
                ++state[c].published;
            }
        }

        for (uint32_t c = 0; c < o.cores; ++c) {
            const uint32_t n = h2d->drained(c);
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t slot = static_cast<uint32_t>(state[c].drained % o.ring);
                ++frames;
                if (frames == warmup) {
                    t0 = std::chrono::steady_clock::now();
                }
                if (frames > warmup) {
                    rt_ns.push_back(ns_since(state[c].at[slot]));
                }
                ++state[c].drained;
            }
            if (n != 0) {
                deadline = std::chrono::steady_clock::now() + kStall;
            }
        }

        if (const std::string e = h2d->first_error(); !e.empty()) {
            std::cerr << "h2d: " << e << "\n";
            ok = false;
        } else if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "stalled at " << frames << " of " << total << " frames\n";
            ok = false;
        }
    }
    const uint64_t window_ns = ns_since(t0);

    if (ok) {
        Finish(mesh->mesh_command_queue());
    }

    if (ok) {
        std::sort(rt_ns.begin(), rt_ns.end());
        const double secs = static_cast<double>(window_ns) / 1e9;
        const double gb = static_cast<double>(total - warmup) * o.payload / 1e9;
        std::cout << std::fixed << std::setprecision(2);
        if (secs > 0.0) {
            std::cout << "bandwidth " << (gb / secs) << " GB/s\n";
        }
        // publish -> drained: this host pokes bytes_sent, the device pulls the page and acks,
        // and this host sees the ack. A full round trip on one clock.
        if (!rt_ns.empty()) {
            std::cout << "latency " << (static_cast<double>(pct(rt_ns, 0.50)) / 1e3) << " us publish->drained\n";
        }
    }
    std::cout << (ok ? "PASS\n" : "FAIL\n");

    // Unpin BEFORE the leg's destructor puts anonymous pages back over the arenas: the pin
    // must not still name the pages being swapped out.
    region.release();
    h2d.reset();
    return ok ? 0 : 1;
}
