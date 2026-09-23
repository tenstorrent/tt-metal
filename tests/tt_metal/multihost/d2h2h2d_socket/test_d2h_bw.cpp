// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The device-to-host leg alone: one rank, no peer, no H2H. The sink retires on sight, so a
// frame's slot is freed by this host seeing it rather than by a transport draining it.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#include "tt_metal/distributed/host_d2h_leg.hpp"
#include "tt_metal/distributed/host_l1_map.hpp"
#include "tt_metal/distributed/host_region.hpp"
#include <tt-metalium/experimental/sockets/host_uva_frame.hpp>
#include <tt-metalium/experimental/sockets/host_uva_layout.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace mh = tt::tt_metal::distributed::multihost;

namespace {

constexpr int kDeviceId = 0;

// A stalled run must fail rather than spin: the kernel parks in socket_barrier when its
// pages go unacked, and Finish() would then never return.
constexpr auto kStall = std::chrono::seconds(30);

struct Options {
    uint32_t payload = 16384;
    uint32_t cores = 4;
    uint32_t ring = 8;
    uint32_t iters = 20000;
    uint32_t warmup_pct = 10;
};

// strtoul, not stoul: an uncaught exception here aborts the rank before the device is torn
// down, which leaves the pinned region and the shm rings behind.
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
        // Not parse_u32: its blanket zero-rejection would refuse 0, which means
        // "measure the whole run" and is a setting a caller legitimately wants.
        if (a == "--warmup-pct" && has_next) {
            const std::string v = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long pct = std::strtoul(v.c_str(), &end, 10);
            if (end != v.c_str() && *end == '\0' && errno != ERANGE && pct < 100) {
                o.warmup_pct = static_cast<uint32_t>(pct);
                continue;
            }
            std::cerr << "error: --warmup-pct must be 0..99\n";
            return false;
        }
        std::cerr << "usage: " << argv[0]
                  << " [--payload B] [--cores N] [--ring frames] [--iters N] [--warmup-pct P]\n";
        return false;
    }
    return true;
}

uint64_t pct(const std::vector<uint64_t>& v, double p) {
    const size_t i = static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5);
    return v[i];
}

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
    // RingAlias refuses an overlay wider than an arena, but it refuses it after the sockets
    // are built. Cheaper to say so here, in terms of the knob the caller turned.
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

    // The leg first, then the pin: D2HLeg MAP_FIXEDs its rings over the arenas, and a pin
    // taken before that would go on naming the pages it replaced.
    std::string err;
    D2HLeg::Config dc;
    dc.cores = o.cores;
    dc.grid_width = grid_width;
    dc.payload_bytes = o.payload;
    dc.ring_pages = o.ring;
    dc.consumed_addr = 0;  // no far device, so nothing credits and tt_uva_sync() is unused
    // Maps the region, sized for these cores: the leg MAP_FIXEDs its rings onto it below,
    // so it has to exist before D2HLeg::create().
    HostRegion& region = HostRegion::storage();
    std::unique_ptr<D2HLeg> d2h;
    try {
        dc.alias_region_base = region.reserved_base(o.cores);
        d2h = D2HLeg::create(mesh, dc, err);
    } catch (const std::exception& ex) {
        std::cerr << "host region unavailable: " << ex.what() << "\n";
        return 1;
    }
    if (!d2h) {
        std::cerr << "d2h bringup failed: " << err << "\n";
        return 1;
    }
    try {
        // try to alias the d2h rings with the region reserved above; this gets registered w/RDMA via MPI_Windows
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

    CoreRangeSet cores;
    std::vector<CoreCoord> core_list;
    for (uint32_t i = 0; i < o.cores; ++i) {
        const CoreCoord c{i % grid_width, i / grid_width};
        core_list.push_back(c);
        cores = cores.merge(CoreRangeSet(CoreRange(c, c)));
    }

    // 64-bit: iters and warmup_pct are both user-supplied uint32, and the 32-bit product
    // wraps well inside the range the test accepts, reporting a wrong rate as a clean PASS.
    const uint32_t warmup_iters =
        static_cast<uint32_t>(static_cast<uint64_t>(o.iters) * o.warmup_pct / 100);
    const uint64_t total = static_cast<uint64_t>(o.cores) * o.iters;
    // Per core, then scaled: the kernel stamps steady state at its own warmup_iters, so a
    // differently-rounded host figure would divide the wrong frame count by that window.
    const uint64_t warmup = static_cast<uint64_t>(o.cores) * warmup_iters;

    Program program = CreateProgram();
    // sig_addr 0 selects the unsignalled tt_uva_put, so there is no receiver kernel and
    // nothing on the device waits on anything this host does except the FIFO ack.
    auto sender = CreateKernel(
        program,
        TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_put.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
            .compile_args = {
                l1.stage_addr,
                page,
                l1.payload_addr,
                o.payload,
                o.iters,
                grid_width,
                0u,
                0u,
                1u,
                0u,
                l1.l1_base,
                0u,
                l1.verify_addr,
                warmup_iters}});
    const std::vector<uint32_t> cfg = d2h->config_addresses();
    for (uint32_t i = 0; i < o.cores; ++i) {
        // Nothing routes on dst here, so it names this host: the bytes go to the socket's
        // own FIFO either way, and the trailer is only read locally.
        SetRuntimeArgs(program, sender, core_list[i], {cfg[i], tt_uva_t6_global_selector(0, 0, i, 1), 0u});
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/false);

    const double ns_per_cycle = 1000.0 / static_cast<double>(device->get_clock_rate_mhz());

    // This host still drains -- nothing moves otherwise -- but it times nothing. Every
    // number reported below was measured on the device and is only carried here.
    std::vector<uint32_t> pending(o.cores, 0);
    std::vector<uint64_t> issue_cycles;
    std::vector<uint64_t> stall_cycles;
    // Capped: total scales with cores x iters, and the 32-bit wrap that used to bound
    // this is gone. An uncapped reserve throws bad_alloc with the region still pinned.
    constexpr uint64_t kMaxSamples = 4u << 20;
    const size_t samples = static_cast<size_t>(std::min<uint64_t>(total - warmup, kMaxSamples));
    issue_cycles.reserve(samples);
    stall_cycles.reserve(samples);
    uint64_t frames = 0;
    bool ok = true;
    // Per core, not against the global count: poll() drains one core fully before moving
    // on, so a global gate admits a fast core's ramp and drops its steady-state samples.
    std::vector<uint32_t> seen(o.cores, 0);

    // Hoisted: the lambda captures more than the small-buffer holds, so building it inside
    // the loop put an operator new/delete pair on every pass of the drain spin.
    const D2HLeg::Sink sink = [&](const SendTask& t) {
        ++pending[t.core];
        ++frames;
        // Matches the kernel's own `i == warmup_iters` stamp.
        if (seen[t.core]++ >= warmup_iters) {
            issue_cycles.push_back(tt_uva_frame_elapsed_issue(t.elapsed));
            stall_cycles.push_back(tt_uva_frame_elapsed_stall(t.elapsed));
        }
        return true;
    };

    auto deadline = std::chrono::steady_clock::now() + kStall;

    while (frames < total && ok) {
        d2h->poll(sink);

        // Batched per pass, not per frame: each retire is a PCIe write to the device, and
        // one per frame would put this host's overhead inside the number being measured.
        for (uint32_t c = 0; c < o.cores; ++c) {
            if (pending[c] != 0) {
                d2h->retire(c, pending[c]);
                pending[c] = 0;
                deadline = std::chrono::steady_clock::now() + kStall;
            }
        }
        if (const std::string e = d2h->first_error(); !e.empty()) {
            std::cerr << "d2h: " << e << "\n";
            ok = false;
        } else if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "stalled at " << frames << " of " << total << " frames\n";
            ok = false;
        }
    }

    // Only after the FIFO has drained: the kernel parks in socket_barrier until its pages
    // are acked, so Finish() before that is the other unbounded wait.
    if (ok) {
        Finish(mesh->mesh_command_queue());
    }

    // The loop window each core timed on its own wall clock. That clock is chip-global, so
    // the earliest begin and the latest end bound one window covering every core.
    uint64_t begin = UINT64_MAX;
    uint64_t end = 0;
    if (ok) {
        for (uint32_t i = 0; i < o.cores; ++i) {
            std::vector<uint32_t> r;
            slow_dispatch::ReadFromL1(*mesh, core_list[i], l1.verify_addr, 7 * sizeof(uint32_t), r);
            if (r.size() < 7 || r[4] != o.iters) {
                std::cerr << "core " << i << " reported " << (r.size() > 4 ? r[4] : 0) << " of " << o.iters
                          << " iterations\n";
                ok = false;
                break;
            }
            // r[5..6], not r[0..1]: the steady-state stamp, so the ramp is outside the window.
            begin = std::min(begin, static_cast<uint64_t>(r[5]) | (static_cast<uint64_t>(r[6]) << 32));
            end = std::max(end, static_cast<uint64_t>(r[2]) | (static_cast<uint64_t>(r[3]) << 32));
        }
    }

    if (ok && end > begin) {
        std::sort(issue_cycles.begin(), issue_cycles.end());
        std::sort(stall_cycles.begin(), stall_cycles.end());
        const double secs = static_cast<double>(end - begin) * ns_per_cycle / 1e9;
        const double gb = static_cast<double>(total - warmup) * o.payload / 1e9;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "bandwidth " << (gb / secs) << " GB/s\n";
        // Both stamped in stage(): issue is the payload write and its barrier, slot wait is
        // the block in socket_reserve_pages -- this host's turnaround, seen from the device.
        if (!issue_cycles.empty()) {
            std::cout << "latency " << (static_cast<double>(pct(issue_cycles, 0.50)) * ns_per_cycle / 1e3)
                      << " us issue, " << (static_cast<double>(pct(stall_cycles, 0.50)) * ns_per_cycle / 1e3)
                      << " us slot wait\n";
        }
    }
    std::cout << (ok ? "PASS\n" : "FAIL\n");

    // Unpin BEFORE the leg's destructor puts anonymous pages back over the arenas: the pin
    // must not still name the pages being swapped out.
    region.release();
    d2h.reset();
    return ok ? 0 : 1;
}
