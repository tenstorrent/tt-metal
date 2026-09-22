// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Moves --volume bytes from rank 0's Tensix cores to rank 1's, in --payload chunks, over
// chip -> host -> host -> chip. Rank 0 sends, rank 1 receives.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <tt-metalium/experimental/sockets/internal/host_rdma_window.hpp>

#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#include <tt-metalium/experimental/sockets/d2h2h2d_socket.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
namespace mh = tt::tt_metal::distributed::multihost;

namespace {

// Not a CLI knob: the point of the run is volume, and more cores only changes how it is
// divided. Raise it here when the shape needs to change.
constexpr uint32_t kCores = 4;
constexpr int kDeviceId = 0;

struct Options {
    uint64_t volume = 1ull << 30;
    uint32_t pct_steady = 10;
    uint32_t payload = 16384;
    uint32_t verify = 1;  // --no-verify for a bandwidth run
};

// strtoull, not stoull: parse() runs with MPI already up, where an uncaught exception aborts
// this rank and strands its peers in whatever collective they had reached.
bool parse_u64(const char* text, uint64_t& out, bool allow_suffix) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(text, &end, 10);
    if (end == text || errno == ERANGE) {
        return false;
    }
    uint64_t mult = 1;
    if (allow_suffix && *end != '\0' && end[1] == '\0') {
        switch (*end) {
            case 'K': mult = 1024ull; break;
            case 'M': mult = 1ull << 20; break;
            case 'G': mult = 1ull << 30; break;
            default: return false;
        }
        ++end;
    }
    if (*end != '\0' || v > UINT64_MAX / mult) {
        return false;
    }
    out = static_cast<uint64_t>(v) * mult;
    return true;
}

bool parse(int argc, char** argv, Options& o) {
    const auto bad = [&argv]() {
        std::cerr << "usage: " << argv[0] << " --volume N[K|M|G] --pct-steady 0..99 --payload N\n";
        return false;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_next = i + 1 < argc;
        uint64_t v = 0;
        if (a == "--volume" && has_next) {
            if (!parse_u64(argv[++i], o.volume, true) || o.volume == 0) {
                return bad();
            }
        } else if (a == "--pct-steady" && has_next) {
            if (!parse_u64(argv[++i], v, false) || v > 99) {
                return bad();
            }
            o.pct_steady = static_cast<uint32_t>(v);
        } else if (a == "--payload" && has_next) {
            if (!parse_u64(argv[++i], v, false) || v == 0 || v > UINT32_MAX) {
                return bad();
            }
            o.payload = static_cast<uint32_t>(v);
        } else if (a == "--no-verify") {
            o.verify = 0;
        } else {
            return bad();
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    mh::DistributedContext::create(argc, argv);
    Options o;
    if (!parse(argc, argv, o)) {
        return 2;
    }

    const mh::ContextPtr world = mh::DistributedContext::get_current_world();
    const uint32_t rank = static_cast<uint32_t>(*world->rank());
    const uint32_t ranks = static_cast<uint32_t>(*world->size());
    if (ranks != 2) {
        std::cerr << "error: this test needs exactly 2 ranks; launch with `mpirun -n 2`\n";
        return 2;
    }

    // Padding, not a separate phase: timing is not collected yet, so --pct-steady only
    // lengthens the run by the share that will later be discarded.
    const uint64_t measured = std::max<uint64_t>(1, o.volume / (static_cast<uint64_t>(kCores) * o.payload));
    const uint32_t iters = static_cast<uint32_t>(measured + measured * o.pct_steady / 100);
    const uint64_t msgs = static_cast<uint64_t>(kCores) * iters;

    // A unit mesh is opened against a world of one, or every rank claims every device.
    {
        const mh::ContextPtr solo = world->split(mh::Color{static_cast<int>(rank)}, mh::Key{0});
        mh::DistributedContext::set_current_world(solo);
    }
    auto mesh = distributed::MeshDevice::create_unit_mesh(kDeviceId);
    IDevice* device = mesh->get_devices().front();
    mh::DistributedContext::set_current_world(world);

    const CoreCoord grid = device->compute_with_storage_grid_size();

    D2H2H2DSocket::Config cfg;
    cfg.topo = HostTopology{rank, ranks, 1};
    cfg.chip = 0;
    cfg.cores = kCores;
    cfg.grid_width = static_cast<uint32_t>(grid.x);
    cfg.grid_height = static_cast<uint32_t>(grid.y);
    cfg.payload_bytes = o.payload;

    std::string err;
    std::unique_ptr<D2H2H2DSocket> sock = D2H2H2DSocket::create(mesh, device, cfg, err);
    if (!sock) {
        std::cerr << "socket bringup failed: " << err << "\n";
        return 1;
    }
    const L1MapNew& l1 = sock->l1();
    std::cout << "rank " << rank << ": " << (o.volume >> 20) << " MiB over " << kCores << " cores x " << o.payload
              << " B => " << iters << " iters\n  " << sock->describe() << "\n  " << l1.describe() << "\n";

    CoreRangeSet cores;
    std::vector<CoreCoord> core_list;
    for (uint32_t i = 0; i < kCores; ++i) {
        const CoreCoord c{i % cfg.grid_width, i / cfg.grid_width};
        core_list.push_back(c);
        cores = cores.merge(CoreRangeSet(CoreRange(c, c)));
    }

    const bool sending = rank == 0;
    const std::vector<uint32_t> send_cfg = sock->d2h().config_addresses();
    Program program = CreateProgram();

    auto sender = CreateKernel(
        program,
        TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_put.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_0,
            .compile_args = {
                l1.stage_addr,
                sock->d2h().page_size(),
                l1.payload_addr,
                o.payload,
                sending ? iters : 0u,
                cfg.grid_width,
                rank,
                cfg.chip,
                cfg.topo.chips_per_host,
                l1.consumed_addr,
                l1.l1_base,
                l1.dest_word_addr,
                0u,
                0u}});
    for (uint32_t i = 0; i < kCores; ++i) {
        SetRuntimeArgs(
            program, sender, core_list[i], {send_cfg[i], tt_uva_t6_global_selector(1 - rank, cfg.chip, i, 1), 0u});
    }

    const std::vector<uint32_t> recv_cfg = sock->h2d().config_addresses();
    auto receiver = CreateKernel(
        program,
        TT_DIRECT_KERNEL_DIR "/kernels/test_kernel_signal.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::NOC_1,
            // dest_word is the app's signal word here: 8 bytes the L1 map budgets and
            // nothing else reads. ADD-1 per frame, so `iters` is this core's whole run.
            .compile_args = {
                l1.deliver_addr,
                sock->d2h().page_size(),
                l1.dest_word_addr,
                iters,
                l1.l1_base,
                l1.l1_size,
                o.payload,
                o.verify,
                l1.verify_addr}});
    for (uint32_t i = 0; i < kCores; ++i) {
        SetRuntimeArgs(program, receiver, core_list[i], {recv_cfg[i], sending ? 0u : 1u});
    }
    MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    // Non-blocking, so the poll loop below can run: a receiver exits once its signal reaches
    // `iters`, and only this loop can feed it there.
    EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/false);

    // The socket owns no thread, so the run IS this loop. `retired`, not `sent`: stopping at
    // QUEUED leaves the tail un-put and the kernel parked in socket_barrier, hanging Finish().
    const auto done = [&] {
        const auto& c = sock->counters();
        return sending ? c.retired >= msgs : c.drained >= msgs;
    };
    // Re-armed on any forward motion, so a long run never trips it and a stall always does.
    // Must precede Finish(), which is the other unbounded wait once a kernel is parked.
    constexpr auto kStall = std::chrono::seconds(30);
    auto deadline = std::chrono::steady_clock::now() + kStall;
    uint64_t last_moved = 0;
    bool stalled = false;
    while (!done() && !sock->failed()) {
        sock->poll();
        const auto& p = sock->counters();
        const uint64_t moved = p.sent + p.retired + p.received + p.drained;
        if (moved != last_moved) {
            last_moved = moved;
            deadline = std::chrono::steady_clock::now() + kStall;
        } else if (std::chrono::steady_clock::now() > deadline) {
            std::cerr << "rank " << rank << ": stalled " << kStall.count() << " s\n  " << sock->describe() << "\n";
            stalled = true;
            break;
        }
    }

    // Collective verdict BEFORE the collective calls below: done() is asymmetric, so one rank
    // finishing while the other stalls would leave the healthy one hung in MPI_Barrier.
    std::string agree_err;
    if (!RdmaWindow::agree(!sock->failed() && !stalled && done(), agree_err)) {
        std::cerr << "rank " << rank << ": " << (agree_err.empty() ? sock->first_error() : agree_err) << "\n";
        std::cout << "FAIL\n";
        return 1;
    }

    Finish(mesh->mesh_command_queue());
    if (const std::string be = sock->barrier(); !be.empty()) {
        std::cerr << "barrier: " << be << "\n";
    }

    uint64_t bad_cores = 0;
    if (!sending && o.verify != 0) {
        for (uint32_t i = 0; i < kCores; ++i) {
            std::vector<uint32_t> r;
            slow_dispatch::ReadFromL1(*mesh, core_list[i], l1.verify_addr, 2 * sizeof(uint32_t), r);
            if (r.size() < 2 || r[0] != 0 || r[1] != iters) {
                std::cerr << "rank " << rank << ": core " << i << " landed " << (r.size() > 1 ? r[1] : 0) << " of "
                          << iters << ", " << (r.empty() ? 0 : r[0]) << " corrupt\n";
                ++bad_cores;
            }
        }
    }

    const auto& c = sock->counters();
    const bool ok = !sock->failed() && bad_cores == 0 && (sending ? c.retired >= msgs : c.drained >= msgs);
    std::cout << "rank " << rank << ": sent " << c.sent << " received " << c.received << " drained " << c.drained
              << " of " << msgs << "\n";
    if (!ok) {
        std::cout << "  first error: " << sock->first_error() << "\n";
    }
    std::cout << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
