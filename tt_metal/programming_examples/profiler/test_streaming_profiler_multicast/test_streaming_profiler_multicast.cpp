// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Core-to-core check of the device-local timeline, independent of the mirrored reads that set the per-tile clock
// offsets: every Tensix core on every chip stamps a zone when a broadcast lands, and a broadcast reaches each core a
// fixed number of router hops after it leaves, 9 cycles a hop (BlackholeA0/NoC README.md, measured 9.00 on this box).
// Two sources take turns, at the two corners of the worker grid, each at the start corner of its NoC's rectangle, so
// no route wraps and a core's hop count is its Manhattan distance from the source along that NoC's directions.
// Every core stamps every round it receives, so a core's arrivals in tick order are its rounds in order, and a round's
// NoC is its parity (multicast_dm.cpp). Per round, a core's placed arrival less its hops is the round's common
// time plus its own error; the mean over the rounds is the error of that core's clock on the timeline, and a chip's
// worst is the spread of those errors over its cores. Pollers see an arrival up to a loop late; the sources' random pad
// spreads that uniformly, so it is common to all cores, and it dithers the whole-tick stamps so the mean resolves
// below a tick where a median would not. A clock's error shows on both NoCs alike, so the covariance of a chip's two
// per-core error sets is the clock errors' variance, apart from the measurement noise each NoC has on its own. Because
// the two sources sit at opposite corners, a core's two hop counts sum to a constant, so a clock gradient along x+y
// looks the same as equal and opposite per-hop errors on the two NoCs. Exits nonzero if any core is missing an arrival
// or has one too many, or any record was dropped or unplaced. Run with TT_METAL_STREAMING_PROFILER=1.
//
//   test_streaming_profiler_multicast [--rounds N] [--settle-ms M] [--arrivals FILE]
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
constexpr uint32_t kFlagAddr = 0x170000;    // L1 scratch above anything the program allocates
constexpr uint32_t kValuesAddr = 0x148000;  // the round numbers the sources broadcast, 16 B apart, up to kMaxRounds
constexpr uint32_t kMaxRounds = 8000;
constexpr double kCyclesPerHop = 9.0;

struct Arrival {
    uint32_t chip;
    CoreCoord logical, physical;
    int64_t host;   // 0.1 ns
    int64_t ticks;  // the core's own wall clock
};
}  // namespace

int main(int argc, char** argv) {
    uint32_t rounds = 4000, settle_ms = 1500;
    const char* arrivals_path = nullptr;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--rounds")) {
            rounds = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--settle-ms")) {
            settle_ms = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--arrivals")) {
            arrivals_path = argv[i + 1];
        }
    }
    rounds = std::clamp(rounds, 4u, kMaxRounds) & ~1u;
    std::mutex mu;
    std::vector<Arrival> arrivals;
    uint64_t dropped_bytes = 0, unplaced = 0;
    const auto sub = sp::RegisterCallback("multicast", [&](const sp::Batch<sp::RecordType::Zones>& b) {
        std::lock_guard<std::mutex> g(mu);
        dropped_bytes += b.dropped_bytes();
        for (const sp::Zone& z : b.zones()) {
            if (std::string_view(z.site().name) != "MC_RX") {
                continue;
            }
            if (z.start_time().time_since_epoch().count() == 0) {
                unplaced++;
                continue;
            }
            arrivals.push_back(Arrival{
                z.core().chip_id,
                z.core().logical,
                z.core().physical,
                z.start_time().time_since_epoch().count(),
                static_cast<int64_t>(z.start_timestamp())});
        }
    });

    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const CoreCoord lo(0, 0), hi(grid.x - 1, grid.y - 1);
    const CoreCoord vlo = mesh_device->worker_core_from_logical_core(lo);
    const CoreCoord vhi = mesh_device->worker_core_from_logical_core(hi);
    const uint32_t num_dests = static_cast<uint32_t>(grid.x * grid.y - 1);
    Program program = CreateProgram();
    const CoreRangeSet all(CoreRange(lo, hi));
    const auto kid = CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_streaming_profiler_multicast/kernels/multicast_dm.cpp",
        all,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            const CoreCoord c(x, y);
            const uint32_t role = c == lo ? 0u : (c == hi ? 1u : 2u);
            // A NoC 1 rectangle runs from its start corner at the high coordinates down to the low ones.
            const CoreCoord& s = role == 1 ? vhi : vlo;
            const CoreCoord& e = role == 1 ? vlo : vhi;
            SetRuntimeArgs(
                program,
                kid,
                c,
                {role,
                 static_cast<uint32_t>(s.x),
                 static_cast<uint32_t>(s.y),
                 static_cast<uint32_t>(e.x),
                 static_cast<uint32_t>(e.y),
                 num_dests,
                 kFlagAddr,
                 kValuesAddr,
                 rounds});
        }
    }
    std::vector<uint32_t> zero = {0, 0};
    for (IDevice* d : mesh_device->get_devices()) {
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                detail::WriteToDeviceL1(d, CoreCoord(x, y), kFlagAddr, zero);
            }
        }
    }
    const CoreCoord noc_grid = mesh_device->get_devices().front()->grid_size();
    const size_t num_chips = mesh_device->get_devices().size();
    // The link solves' windows fill before the rounds start.
    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    std::printf(
        "[multicast] %zux%zu Tensix cores x %u rounds (%u a NoC) on %zu chips; NoC grid %zux%zu\n",
        grid.x,
        grid.y,
        rounds,
        rounds / 2,
        mesh_device->get_devices().size(),
        noc_grid.x,
        noc_grid.y);
    std::fflush(stdout);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);
    uint32_t gave_up = 0;
    for (IDevice* d : mesh_device->get_devices()) {
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                std::vector<uint32_t> words(2, 0);
                detail::ReadFromDeviceL1(d, CoreCoord(x, y), kFlagAddr, 8, words);
                if (words[1] != 0) {
                    std::printf(
                        "[multicast] chip %d core (%u,%u) gave up waiting for round %u (flag %u)\n",
                        d->id(),
                        x,
                        y,
                        words[1],
                        words[0]);
                    gave_up++;
                }
            }
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    mesh_device->close();
    sp::UnregisterCallback(sub);

    std::map<uint32_t, std::vector<Arrival>> by_chip;
    {
        std::lock_guard<std::mutex> g(mu);
        for (const Arrival& a : arrivals) {
            by_chip[a.chip].push_back(a);
        }
    }
    if (FILE* f = arrivals_path != nullptr ? std::fopen(arrivals_path, "w") : nullptr) {
        std::fprintf(f, "chip,x,y,phys_x,phys_y,host_0p1ns,ticks\n");
        for (const Arrival& a : arrivals) {
            std::fprintf(
                f,
                "%u,%zu,%zu,%zu,%zu,%lld,%lld\n",
                a.chip,
                a.logical.x,
                a.logical.y,
                a.physical.x,
                a.physical.y,
                static_cast<long long>(a.host),
                static_cast<long long>(a.ticks));
        }
        std::fclose(f);
    }
    const int W = static_cast<int>(noc_grid.x), H = static_cast<int>(noc_grid.y);
    using CoreKey = std::pair<size_t, size_t>;
    const CoreKey lo_key{lo.x, lo.y}, hi_key{hi.x, hi.y};
    using Round = std::map<CoreKey, const Arrival*>;
    struct Lane {
        CoreCoord src;
        std::vector<Round> rounds;
    };
    std::map<std::pair<uint32_t, uint32_t>, Lane> by_lane;
    bool incomplete = by_chip.size() != num_chips;
    for (auto& [chip, v] : by_chip) {
        std::map<CoreKey, std::vector<const Arrival*>> by_core;
        for (const Arrival& a : v) {
            by_core[CoreKey{a.logical.x, a.logical.y}].push_back(&a);
        }
        bool counts_ok = by_core.size() == grid.x * grid.y;
        for (auto& [c, list] : by_core) {
            std::sort(list.begin(), list.end(), [](const Arrival* a, const Arrival* b) { return a->ticks < b->ticks; });
            const size_t want = c == lo_key || c == hi_key ? rounds / 2 : rounds;
            if (list.size() != want) {
                std::printf(
                    "[multicast] chip %u core (%zu,%zu): %zu arrivals, expected %zu\n",
                    chip,
                    c.first,
                    c.second,
                    list.size(),
                    want);
                counts_ok = false;
            }
        }
        if (!counts_ok) {
            incomplete = true;
            continue;
        }
        Lane& odd = by_lane[{chip, 0u}];
        Lane& even = by_lane[{chip, 1u}];
        odd.src = by_core.at(lo_key).front()->physical;
        even.src = by_core.at(hi_key).front()->physical;
        odd.rounds.resize(rounds / 2);
        even.rounds.resize(rounds / 2);
        for (const auto& [c, list] : by_core) {
            for (size_t k = 0; k < list.size(); k++) {
                const size_t r = c == lo_key ? 2 * k + 2 : (c == hi_key ? 2 * k + 1 : k + 1);
                ((r & 1u) ? odd : even).rounds[(r - 1) / 2][c] = list[k];
            }
        }
    }
    std::map<std::pair<uint32_t, CoreKey>, double> err_by_noc[2];
    double worst_span_ns = 0.0, worst_span_ticks = 0.0;
    std::printf(
        "chip noc cores rounds  ns/tick   placed: span ns (ticks)  noise ns  rms ns   raw clocks: span ticks\n");
    for (const auto& [key, lane] : by_lane) {
        const auto [chip, noc] = key;
        const std::vector<Round>& rounds = lane.rounds;
        const CoreCoord src = lane.src;
        std::vector<CoreKey> cores;
        std::vector<double> hops;
        for (const auto& [c, a] : rounds.front()) {
            const CoreCoord p = a->physical;
            const int dx = noc == 0 ? (static_cast<int>(p.x) - static_cast<int>(src.x) + W) % W
                                    : (static_cast<int>(src.x) - static_cast<int>(p.x) + W) % W;
            const int dy = noc == 0 ? (static_cast<int>(p.y) - static_cast<int>(src.y) + H) % H
                                    : (static_cast<int>(src.y) - static_cast<int>(p.y) + H) % H;
            cores.push_back(c);
            hops.push_back(static_cast<double>(dx + dy));
        }
        const size_t m = cores.size(), n = rounds.size();
        // e[i][k]: core i's arrival in round k less its hops, less the round's mean over the cores, in ns and in raw
        // ticks. A round's ns per tick is the reference core's between this round and the next.
        std::vector<std::vector<double>> e(m, std::vector<double>(n)), raw(m, std::vector<double>(n));
        double tau_sum = 0.0;
        for (size_t k = 0; k < n; k++) {
            const size_t k2 = k + 1 < n ? k + 1 : k - 1;
            const Arrival* a = rounds[k].at(cores[0]);
            const Arrival* b = rounds[k2].at(cores[0]);
            const double tau =
                (static_cast<double>(b->host - a->host) / 10.0) / static_cast<double>(b->ticks - a->ticks);
            tau_sum += tau;
            double mh = 0.0, mr = 0.0;
            for (size_t i = 0; i < m; i++) {
                const Arrival* x = rounds[k].at(cores[i]);
                e[i][k] = static_cast<double>(x->host) / 10.0 - kCyclesPerHop * hops[i] * tau;
                raw[i][k] = static_cast<double>(x->ticks - a->ticks) - kCyclesPerHop * hops[i];
                mh += e[i][k];
                mr += raw[i][k];
            }
            for (size_t i = 0; i < m; i++) {
                e[i][k] -= mh / static_cast<double>(m);
                raw[i][k] -= mr / static_cast<double>(m);
            }
        }
        const double tau = tau_sum / static_cast<double>(n);
        const auto median = [](std::vector<double> v) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        };
        std::vector<double> E(m);
        double lo_e = 1e30, hi_e = -1e30, noise = 0.0, rms = 0.0, raw_lo = 1e30, raw_hi = -1e30;
        for (size_t i = 0; i < m; i++) {
            double sum = 0.0, sum2 = 0.0;
            for (const double x : e[i]) {
                sum += x;
                sum2 += x * x;
            }
            E[i] = sum / static_cast<double>(n);
            noise = std::max(noise, std::sqrt(std::max(sum2 / static_cast<double>(n) - E[i] * E[i], 0.0) / n));
            const double r_i = median(raw[i]);
            lo_e = std::min(lo_e, E[i]);
            hi_e = std::max(hi_e, E[i]);
            rms += E[i] * E[i];
            raw_lo = std::min(raw_lo, r_i);
            raw_hi = std::max(raw_hi, r_i);
            err_by_noc[noc][{chip, cores[i]}] = E[i];
        }
        rms = std::sqrt(rms / static_cast<double>(m));
        worst_span_ns = std::max(worst_span_ns, hi_e - lo_e);
        worst_span_ticks = std::max(worst_span_ticks, (hi_e - lo_e) / tau);
        std::printf(
            "%4u %3u %5zu  %5zu  %.4f          %6.3f (%5.2f)     %6.3f   %6.3f           %6.2f\n",
            chip,
            noc,
            m,
            n,
            tau,
            hi_e - lo_e,
            (hi_e - lo_e) / tau,
            noise,
            rms,
            raw_hi - raw_lo);
    }
    std::map<uint32_t, std::vector<std::pair<double, double>>> both;
    for (const auto& [k, e0] : err_by_noc[0]) {
        const auto it = err_by_noc[1].find(k);
        if (it != err_by_noc[1].end()) {
            both[k.first].emplace_back(e0, it->second);
        }
    }
    double worst_avg_span = 0.0, worst_split = 0.0, worst_clock_rms = 0.0;
    for (const auto& [chip, pairs] : both) {
        double lo_a = 1e30, hi_a = -1e30, m0 = 0.0, m1 = 0.0, cov = 0.0;
        for (const auto& [e0, e1] : pairs) {
            lo_a = std::min(lo_a, (e0 + e1) / 2);
            hi_a = std::max(hi_a, (e0 + e1) / 2);
            worst_split = std::max(worst_split, std::fabs((e0 - e1) / 2));
            m0 += e0;
            m1 += e1;
        }
        m0 /= static_cast<double>(pairs.size());
        m1 /= static_cast<double>(pairs.size());
        for (const auto& [e0, e1] : pairs) {
            cov += (e0 - m0) * (e1 - m1);
        }
        worst_avg_span = std::max(worst_avg_span, hi_a - lo_a);
        worst_clock_rms = std::max(worst_clock_rms, std::sqrt(std::max(cov / static_cast<double>(pairs.size()), 0.0)));
    }
    std::printf(
        "worst core-to-core error of the device-local timeline (max minus min over a chip's cores): %.3f ns (%.2f "
        "ticks) on one NoC, %.3f ns on the two NoCs' mean; NoC half-difference worst %.3f ns; the clock errors, the "
        "part common to both NoCs, rms %.3f ns on the worst chip. %u cores gave up, %llu records unplaced, %llu bytes "
        "dropped%s\n",
        worst_span_ns,
        worst_span_ticks,
        worst_avg_span,
        worst_split,
        worst_clock_rms,
        gave_up,
        static_cast<unsigned long long>(unplaced),
        static_cast<unsigned long long>(dropped_bytes),
        incomplete ? "; SOME CHIPS MISSING OR WITH WRONG ARRIVAL COUNTS" : "");
    return gave_up == 0 && !incomplete && unplaced == 0 && dropped_bytes == 0 && !by_lane.empty() ? 0 : 1;
}
