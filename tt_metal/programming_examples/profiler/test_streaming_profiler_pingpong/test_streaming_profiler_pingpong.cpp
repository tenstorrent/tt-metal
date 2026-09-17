// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Acceptance test for the per-tile wall-clock offsets: pairs of worker cores on every chip, each pair in one row or
// one column, exchange NoC atomics with a zone stamped at each end, both directions, alternating NoCs. NoC 0 runs +x/+y
// and NoC 1 the reverse, so A->B on NoC 0 and B->A on NoC 1 cross the same links between the two tiles when A is the
// lower coordinate; on the host timeline those two one-way times must agree, and half their difference is the offset
// error between the tiles. The other two arcs cross the ring's wrap link, where an atomic on NoC 1 measured ~20 ns
// slower than on NoC 0 along x, so they are reported only as a NoC diagnostic. Diagonal pairs would traverse different
// rows and columns in each direction and are not used. Run with TT_METAL_STREAMING_PROFILER=1.
//
//   test_streaming_profiler_pingpong [--rounds N] [--settle-ms M]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;
namespace sp = tt::tt_metal::experimental::streaming_profiler;

namespace {
constexpr uint32_t kFlagAddr = 0x170000;  // L1 scratch the flags live in, above anything the program allocates

struct Stamp {
    uint32_t chip;
    CoreCoord core;
    bool tx;
    sp::Zone zone;
};

double mean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) {
        s += x;
    }
    return v.empty() ? NAN : s / static_cast<double>(v.size());
}
double stdev(const std::vector<double>& v) {
    const double m = mean(v);
    double s = 0;
    for (double x : v) {
        s += (x - m) * (x - m);
    }
    return v.size() > 1 ? std::sqrt(s / static_cast<double>(v.size() - 1)) : NAN;
}
}  // namespace

int main(int argc, char** argv) {
    uint32_t rounds = 2000, settle_ms = 1500;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--rounds")) {
            rounds = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--settle-ms")) {
            settle_ms = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
        }
    }
    std::mutex mu;
    std::vector<Stamp> stamps;
    const auto sub = sp::RegisterCallback("pingpong", [&](const sp::Batch<sp::RecordType::Zones>& b) {
        std::lock_guard<std::mutex> g(mu);
        for (const sp::Zone& z : b.zones()) {
            const std::string_view name = z.site().name;
            if (name == "PP_TX" || name == "PP_RX") {
                stamps.push_back(Stamp{z.core().chip_id, z.core().logical, name == "PP_TX", z});
            }
        }
    });

    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x < 6 || grid.y < 6) {
        std::fprintf(stderr, "grid %zux%zu too small\n", grid.x, grid.y);
        return 1;
    }
    // Three columns and three rows across the grid; A is the lower coordinate along the pair's axis, and no core
    // serves two pairs.
    const std::vector<std::pair<CoreCoord, CoreCoord>> pairs = {
        {CoreCoord(0, 1), CoreCoord(0, grid.y - 1)},
        {CoreCoord(grid.x / 2, 0), CoreCoord(grid.x / 2, grid.y - 1)},
        {CoreCoord(grid.x - 1, 1), CoreCoord(grid.x - 1, grid.y - 1)},
        {CoreCoord(1, 0), CoreCoord(grid.x - 1, 0)},
        {CoreCoord(0, grid.y / 2), CoreCoord(grid.x - 2, grid.y / 2)},
        {CoreCoord(2, grid.y - 2), CoreCoord(grid.x - 2, grid.y - 2)},
    };
    Program program = CreateProgram();
    std::vector<CoreRange> ranges;
    for (const auto& [a, b] : pairs) {
        ranges.emplace_back(a, a);
        ranges.emplace_back(b, b);
    }
    const auto kid = CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_streaming_profiler_pingpong/kernels/pingpong_dm.cpp",
        CoreRangeSet(ranges),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    for (const auto& [a, b] : pairs) {
        const CoreCoord va = mesh_device->worker_core_from_logical_core(a);
        const CoreCoord vb = mesh_device->worker_core_from_logical_core(b);
        SetRuntimeArgs(
            program, kid, a, {0u, static_cast<uint32_t>(vb.x), static_cast<uint32_t>(vb.y), kFlagAddr, rounds});
        SetRuntimeArgs(
            program, kid, b, {1u, static_cast<uint32_t>(va.x), static_cast<uint32_t>(va.y), kFlagAddr, rounds});
    }
    std::vector<uint32_t> zero = {0, 0};
    for (IDevice* d : mesh_device->get_devices()) {
        for (const auto& [a, b] : pairs) {
            detail::WriteToDeviceL1(d, a, kFlagAddr, zero);
            detail::WriteToDeviceL1(d, b, kFlagAddr, zero);
        }
    }
    // The clock trackers and the host line need a moment before records can be placed.
    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    std::printf(
        "[pingpong] %zu pairs x %u rounds on %zu chips\n", pairs.size(), rounds, mesh_device->get_devices().size());
    std::fflush(stdout);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);
    std::printf("[pingpong] workload done\n");
    std::fflush(stdout);
    // A core whose wait gave up left the round it wanted at flag + 4.
    for (IDevice* d : mesh_device->get_devices()) {
        for (const auto& [a, b] : pairs) {
            for (const CoreCoord& c : {a, b}) {
                std::vector<uint32_t> words(2, 0);
                detail::ReadFromDeviceL1(d, c, kFlagAddr, 8, words);
                if (words[1] != 0) {
                    std::printf(
                        "[pingpong] chip %d core (%zu,%zu) gave up waiting for round %u\n",
                        d->id(),
                        c.x,
                        c.y,
                        words[1]);
                }
            }
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    mesh_device->close();

    // Per (chip, core): TX and RX stamps in arrival order, which is round order within a lane.
    std::map<std::pair<uint32_t, CoreCoord>, std::pair<std::vector<sp::Zone>, std::vector<sp::Zone>>> by_core;
    {
        std::lock_guard<std::mutex> g(mu);
        for (const Stamp& s : stamps) {
            auto& e = by_core[{s.chip, s.core}];
            (s.tx ? e.first : e.second).push_back(s.zone);
        }
    }
    auto tsc_of = [](const sp::Zone& z) { return z.start_time().time_since_epoch().count(); };
    auto ns_of = [](int64_t ticks) {
        return static_cast<double>(ticks) / 10.0;  // host_clock units to ns
    };
    double worst = 0;
    std::printf(
        "pair                 chip rounds   direct arcs A->B(noc0) B->A(noc1) ns   offset err ns   jitter ns   "
        "wrap arcs A->B(noc1) B->A(noc0) ns   their half-difference ns\n");
    std::vector<uint32_t> chips;
    for (const auto& [k, _] : by_core) {
        if (chips.empty() || chips.back() != k.first) {
            chips.push_back(k.first);
        }
    }
    for (uint32_t chip : chips) {
        for (const auto& [a, b] : pairs) {
            auto ia = by_core.find({chip, a}), ib = by_core.find({chip, b});
            if (ia == by_core.end() || ib == by_core.end()) {
                continue;
            }
            const auto& [tx_a, rx_a] = ia->second;
            const auto& [tx_b, rx_b] = ib->second;
            const size_t n = std::min({tx_a.size(), rx_a.size(), tx_b.size(), rx_b.size()});
            if (n != rounds || tx_a.size() != rx_b.size() || tx_b.size() != rx_a.size()) {
                std::printf(
                    "(%zu,%zu)-(%zu,%zu) chip %u: stamps %zu/%zu/%zu/%zu of %u rounds, skipped\n",
                    a.x,
                    a.y,
                    b.x,
                    b.y,
                    chip,
                    tx_a.size(),
                    rx_b.size(),
                    tx_b.size(),
                    rx_a.size(),
                    rounds);
                continue;
            }
            // fwd[noc]: A->B one-way on that NoC; bwd[noc]: B->A. Rounds alternate NoC in pairs, so a round's two
            // messages share a NoC and each direction gets both NoCs over the run.
            std::vector<double> fwd[2], bwd[2];
            for (size_t k = 0; k < n; k++) {
                const int64_t ta = tsc_of(tx_a[k]), rb = tsc_of(rx_b[k]), tb = tsc_of(tx_b[k]), ra = tsc_of(rx_a[k]);
                if (ta == 0 || rb == 0 || tb == 0 || ra == 0) {
                    continue;
                }
                const uint32_t noc = ((k + 1) >> 1) & 1u;
                fwd[noc].push_back(ns_of(rb - ta));
                bwd[noc].push_back(ns_of(ra - tb));
            }
            if (fwd[0].empty() || bwd[1].empty()) {
                std::printf("(%zu,%zu)-(%zu,%zu) chip %u: no placeable rounds\n", a.x, a.y, b.x, b.y, chip);
                continue;
            }
            // The direct arcs: the same links in the two directions. Jitter is the per-message scatter of each.
            const double e = (mean(fwd[0]) - mean(bwd[1])) / 2;
            const double e_wrap = (mean(fwd[1]) - mean(bwd[0])) / 2;
            worst = std::max(worst, std::fabs(e));
            std::printf(
                "(%2zu,%2zu)-(%2zu,%2zu)        %2u  %5zu        %7.1f    %7.1f            %+6.2f       %.1f / %.1f    "
                "   %7.1f    %7.1f                %+6.2f\n",
                a.x,
                a.y,
                b.x,
                b.y,
                chip,
                fwd[0].size() + fwd[1].size(),
                mean(fwd[0]),
                mean(bwd[1]),
                e,
                stdev(fwd[0]),
                stdev(bwd[1]),
                mean(fwd[1]),
                mean(bwd[0]),
                e_wrap);
        }
    }
    std::printf("worst offset error over the direct arcs of all pairs and chips: %.2f ns\n", worst);
    sp::UnregisterCallback(sub);
    return 0;
}
