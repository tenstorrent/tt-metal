// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Core-to-core check of the device-local timeline, independent of the mirrored reads that set the per-tile clock
// offsets: every Tensix core on every chip stamps a zone when a broadcast lands, and a broadcast reaches each core a
// fixed number of router hops after it leaves, 9 cycles a hop (BlackholeA0/NoC README.md, measured 9.00 on this box).
// Two sources take turns, at the two corners of the worker grid, each at the start corner of its NoC's rectangle, so
// no route wraps and a core's hop count is its Manhattan distance from the source along that NoC's directions.
// Arrivals are grouped into rounds by host time, a round's NoC is the one whose source did not stamp it, and only
// rounds every receiver stamped once are used. Per round, a core's placed arrival less its hops is the round's common
// time plus its own error; the mean over the rounds is the error of that core's clock on the timeline, and a chip's
// worst is the spread of those errors over its cores. Pollers see an arrival up to a loop late; the sources' random pad
// spreads that uniformly, so it is common to all cores, and it dithers the whole-tick stamps so the mean resolves
// below a tick where a median would not. A clock's error shows on both NoCs alike, so the covariance of a chip's two
// per-core error sets is the clock errors' variance, apart from the measurement noise each NoC has on its own. Exits
// nonzero if any lane is short of rounds or any record was dropped or unplaced. Run with TT_METAL_STREAMING_PROFILER=1.
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
constexpr double kRoundGapNs = 5000.0;  // rounds are ~30 us apart and a round's arrivals span well under 1 us
constexpr double kOutlierNs = 20.0;

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
    rounds = std::min(rounds, kMaxRounds) & ~1u;
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
    const size_t receivers = grid.x * grid.y - 1;
    using CoreKey = std::pair<size_t, size_t>;
    const CoreKey lo_key{lo.x, lo.y}, hi_key{hi.x, hi.y};
    struct Lane {
        std::vector<std::map<CoreKey, const Arrival*>> rounds;
        size_t seen = 0;
    };
    std::map<std::pair<uint32_t, uint32_t>, Lane> by_lane;
    std::map<std::pair<uint32_t, CoreKey>, CoreCoord> phys_of;
    size_t unattributed = 0;
    for (auto& [chip, v] : by_chip) {
        std::sort(v.begin(), v.end(), [](const Arrival& a, const Arrival& b) { return a.host < b.host; });
        for (const Arrival& a : v) {
            phys_of[{chip, CoreKey{a.logical.x, a.logical.y}}] = a.physical;
        }
        for (size_t i = 0; i < v.size();) {
            size_t j = i + 1;
            while (j < v.size() && static_cast<double>(v[j].host - v[j - 1].host) / 10.0 < kRoundGapNs) {
                j++;
            }
            std::map<CoreKey, const Arrival*> r;
            bool dup = false;
            for (size_t k = i; k < j; k++) {
                dup |= !r.emplace(CoreKey{v[k].logical.x, v[k].logical.y}, &v[k]).second;
            }
            i = j;
            const bool has_lo = r.contains(lo_key), has_hi = r.contains(hi_key);
            if (has_lo == has_hi) {
                unattributed++;
                continue;
            }
            Lane& lane = by_lane[{chip, has_hi ? 0u : 1u}];
            lane.seen++;
            if (!dup && r.size() == receivers) {
                lane.rounds.push_back(std::move(r));
            }
        }
    }
    std::map<std::pair<uint32_t, CoreKey>, double> err_by_noc[2];
    bool incomplete = unattributed != 0 || by_lane.size() != 2 * by_chip.size();
    double worst_span_ns = 0.0, worst_span_ticks = 0.0;
    FILE* csv = std::fopen("multicast_cores.csv", "w");
    if (csv != nullptr) {
        std::fprintf(csv, "chip,noc,x,y,phys_x,phys_y,hops,err_ns,err_ticks,raw_err_ticks,jitter_ns,outliers\n");
    }
    std::printf(
        "chip noc cores rounds(kept/seen)  ns/tick   placed: span ns (ticks)  noise ns  rms ns  jitter ns  "
        "outliers   raw clocks: span ticks\n");
    for (auto& [key, lane] : by_lane) {
        const auto [chip, noc] = key;
        const std::vector<std::map<CoreKey, const Arrival*>>& rounds = lane.rounds;
        const size_t seen = lane.seen;
        const auto src_it = phys_of.find({chip, noc == 0 ? lo_key : hi_key});
        if (rounds.size() < 3 || src_it == phys_of.end()) {
            std::printf(
                "%4u %3u: %zu complete rounds of %zu, source %s; skipped\n",
                chip,
                noc,
                rounds.size(),
                seen,
                src_it == phys_of.end() ? "not found" : "found");
            incomplete = true;
            continue;
        }
        const CoreCoord src = src_it->second;
        incomplete |= rounds.size() < seen * 9 / 10;
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
        double lo_e = 1e30, hi_e = -1e30, noise = 0.0, rms = 0.0, jit = 0.0, raw_lo = 1e30, raw_hi = -1e30;
        size_t outliers = 0;
        for (size_t i = 0; i < m; i++) {
            double sum = 0.0, sum2 = 0.0;
            for (const double x : e[i]) {
                sum += x;
                sum2 += x * x;
            }
            E[i] = sum / static_cast<double>(n);
            noise = std::max(noise, std::sqrt(std::max(sum2 / static_cast<double>(n) - E[i] * E[i], 0.0) / n));
            std::vector<double> dev(n);
            size_t out_i = 0;
            for (size_t k = 0; k < n; k++) {
                dev[k] = std::fabs(e[i][k] - E[i]);
                out_i += dev[k] > kOutlierNs;
            }
            const double j_i = 1.4826 * median(dev);
            const double r_i = median(raw[i]);
            outliers += out_i;
            lo_e = std::min(lo_e, E[i]);
            hi_e = std::max(hi_e, E[i]);
            rms += E[i] * E[i];
            jit += j_i * j_i;
            raw_lo = std::min(raw_lo, r_i);
            raw_hi = std::max(raw_hi, r_i);
            err_by_noc[noc][{chip, cores[i]}] = E[i];
            if (csv != nullptr) {
                std::fprintf(
                    csv,
                    "%u,%u,%zu,%zu,%zu,%zu,%.0f,%.3f,%.3f,%.3f,%.3f,%zu\n",
                    chip,
                    noc,
                    cores[i].first,
                    cores[i].second,
                    phys_of[{chip, cores[i]}].x,
                    phys_of[{chip, cores[i]}].y,
                    hops[i],
                    E[i],
                    E[i] / tau,
                    r_i,
                    j_i,
                    out_i);
            }
        }
        rms = std::sqrt(rms / static_cast<double>(m));
        jit = std::sqrt(jit / static_cast<double>(m));
        worst_span_ns = std::max(worst_span_ns, hi_e - lo_e);
        worst_span_ticks = std::max(worst_span_ticks, (hi_e - lo_e) / tau);
        std::printf(
            "%4u %3u %5zu   %5zu/%-5zu       %.4f          %6.3f (%5.2f)     %6.3f   %6.3f   %6.2f   %8zu"
            "           %6.2f\n",
            chip,
            noc,
            m,
            n,
            seen,
            tau,
            hi_e - lo_e,
            (hi_e - lo_e) / tau,
            noise,
            rms,
            jit,
            outliers,
            raw_hi - raw_lo);
    }
    if (csv != nullptr) {
        std::fclose(csv);
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
        "part common to both NoCs, rms %.3f ns on the worst chip. The two sources sit at opposite corners, so a core's "
        "two hop counts sum to a constant: a clock gradient along x+y and equal and "
        "opposite per-hop errors on the two NoCs look the same. %u cores gave up, %zu rounds with no NoC, %llu records "
        "unplaced, %llu bytes dropped%s; per-core table in multicast_cores.csv\n",
        worst_span_ns,
        worst_span_ticks,
        worst_avg_span,
        worst_split,
        worst_clock_rms,
        gave_up,
        unattributed,
        static_cast<unsigned long long>(unplaced),
        static_cast<unsigned long long>(dropped_bytes),
        incomplete ? "; SOME CHIP/NOC LANES SKIPPED OR SHORT OF ROUNDS" : "");
    return gave_up == 0 && !incomplete && unplaced == 0 && dropped_bytes == 0 && !by_lane.empty() ? 0 : 1;
}
