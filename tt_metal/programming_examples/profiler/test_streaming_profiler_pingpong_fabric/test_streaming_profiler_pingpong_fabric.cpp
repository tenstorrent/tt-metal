// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Cross-chip acceptance test for the d2d sync: on every pair of chips joined by an ethernet link, one worker core on
// each end exchanges fabric atomics with a zone stamped at each end, both directions. On the host timeline the two
// one-way times must agree; half their difference is the placement error between the two chips as seen by tensix
// cores through the fabric, independent of the ethernet stamping the sync itself is built on (the link's own
// direction asymmetry is common to both). Run with TT_METAL_STREAMING_PROFILER=1.
//
//   test_streaming_profiler_pingpong_fabric [--rounds N] [--settle-ms M]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

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
struct Pair {
    uint32_t chip_a, chip_b;  // chip_a < chip_b; A has role 0
    CoreCoord core_a, core_b;
    uint32_t link_a, link_b;  // the fabric link index each end sends on: the two ends of one cable
    int dx, dy;               // both workers' NoC-0 offset from their own end of that cable
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
    const auto sub = sp::RegisterCallback("pingpong-fabric", [&](const sp::Batch<sp::RecordType::Zones>& b) {
        std::lock_guard<std::mutex> g(mu);
        for (const sp::Zone& z : b.zones()) {
            const std::string_view name = z.site().name;
            if (name == "PP_TX" || name == "PP_RX") {
                stamps.push_back(Stamp{z.core().chip_id, z.core().logical, name == "PP_TX", z});
            }
        }
    });

    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);

    // Chips by mesh coordinate, their fabric node ids, and every ethernet-linked pair.
    std::map<uint32_t, distributed::MeshCoordinate> coord_of;
    std::map<uint32_t, tt::tt_fabric::FabricNodeId> node_of;
    for (const auto& c : distributed::MeshCoordinateRange(mesh_device->shape())) {
        IDevice* d = mesh_device->get_device(c);
        coord_of.emplace(d->id(), c);
        node_of.emplace(d->id(), tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(d->id()));
    }
    // A packet's local NoC hops, worker to link core at the sender and link core to worker at the receiver, only
    // cancel between the two directions when both workers sit at the same NoC offset from their own end of one
    // cable. So each pair is placed by the link core: the first forwarding link at A names a channel, its peer
    // channel at B is looked up and B is made to send on that same cable, and the workers are the cores at the
    // smallest common offset from the two link cores.
    auto& cluster = MetalContext::instance().get_cluster();
    auto& control_plane = MetalContext::instance().get_control_plane();
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    std::map<uint32_t, std::map<std::pair<int, int>, CoreCoord>> workers_at;  // chip -> NoC-0 (x, y) -> logical
    for (const auto& [chip, _] : node_of) {
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                const CoreCoord l(x, y);
                const CoreCoord p = cluster.get_physical_coordinate_from_logical_coordinates(chip, l, CoreType::WORKER);
                workers_at[chip][{static_cast<int>(p.x), static_cast<int>(p.y)}] = l;
            }
        }
    }
    auto channels_toward = [&](uint32_t chip, uint32_t peer) {
        const auto dir = control_plane.get_forwarding_direction(node_of.at(chip), node_of.at(peer));
        return control_plane.get_active_fabric_eth_channels_in_direction(node_of.at(chip), dir.value());
    };
    std::vector<std::pair<int, int>> offsets;
    for (int dy = -12; dy <= 12; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            if (dy != 0) {
                offsets.emplace_back(dx, dy);
            }
        }
    }
    std::sort(offsets.begin(), offsets.end(), [](const auto& p, const auto& q) {
        return std::tuple(std::abs(p.first) + std::abs(p.second), std::abs(p.first), p.second < 0) <
               std::tuple(std::abs(q.first) + std::abs(q.second), std::abs(q.first), q.second < 0);
    });
    std::map<uint32_t, std::set<CoreCoord>> used;
    std::vector<Pair> pairs;
    for (const auto& [a, na] : node_of) {
        for (const auto& [b, nb] : node_of) {
            if (a >= b || tt::tt_fabric::get_neighbor_eth_directions(na, nb).empty()) {
                continue;
            }
            const auto links_ab = tt::tt_fabric::get_forwarding_link_indices(na, nb);
            if (links_ab.empty()) {
                std::fprintf(stderr, "no fabric link from chip %u to chip %u\n", a, b);
                return 1;
            }
            const uint32_t link_a = links_ab.front();
            const uint32_t chan_a = channels_toward(a, b).at(link_a);
            const auto& soc_a = cluster.get_soc_desc(a);
            const auto& soc_b = cluster.get_soc_desc(b);
            const auto eth_a = soc_a.get_eth_core_for_channel(chan_a, CoordSystem::LOGICAL);
            const auto [peer_chip, eth_b] = cluster.get_connected_ethernet_core({a, CoreCoord(eth_a.x, eth_a.y)});
            if (peer_chip != b) {
                std::fprintf(stderr, "chip %u channel %u lands on chip %u, not chip %u\n", a, chan_a, peer_chip, b);
                return 1;
            }
            const uint32_t chan_b = static_cast<uint32_t>(soc_b.logical_eth_core_to_chan_map.at(eth_b));
            const auto chans_b = channels_toward(b, a);
            const auto links_ba = tt::tt_fabric::get_forwarding_link_indices(nb, na);
            std::optional<uint32_t> link_b;
            for (uint32_t i = 0; i < chans_b.size(); i++) {
                if (chans_b[i] == chan_b && std::find(links_ba.begin(), links_ba.end(), i) != links_ba.end()) {
                    link_b = i;
                }
            }
            if (!link_b) {
                std::fprintf(stderr, "chip %u channel %u does not forward back to chip %u\n", b, chan_b, a);
                return 1;
            }
            const auto ra = soc_a.get_eth_core_for_channel(chan_a, CoordSystem::NOC0);
            const auto rb = soc_b.get_eth_core_for_channel(chan_b, CoordSystem::NOC0);
            bool placed = false;
            for (const auto& [dx, dy] : offsets) {
                const auto wa = workers_at[a].find({static_cast<int>(ra.x) + dx, static_cast<int>(ra.y) + dy});
                const auto wb = workers_at[b].find({static_cast<int>(rb.x) + dx, static_cast<int>(rb.y) + dy});
                if (wa == workers_at[a].end() || wb == workers_at[b].end() || used[a].count(wa->second) ||
                    used[b].count(wb->second)) {
                    continue;
                }
                pairs.push_back(Pair{a, b, wa->second, wb->second, link_a, *link_b, dx, dy});
                used[a].insert(wa->second);
                used[b].insert(wb->second);
                std::printf(
                    "[pingpong-fabric] chip %u worker (%zu,%zu) <-> chip %u worker (%zu,%zu): link cores NoC-0 "
                    "(%zu,%zu) and (%zu,%zu), workers at offset (%+d,%+d) from them, channels %u and %u\n",
                    a,
                    wa->second.x,
                    wa->second.y,
                    b,
                    wb->second.x,
                    wb->second.y,
                    static_cast<size_t>(ra.x),
                    static_cast<size_t>(ra.y),
                    static_cast<size_t>(rb.x),
                    static_cast<size_t>(rb.y),
                    dx,
                    dy,
                    chan_a,
                    chan_b);
                placed = true;
                break;
            }
            if (!placed) {
                std::fprintf(stderr, "no common worker offset from the link cores of chips %u and %u\n", a, b);
                return 1;
            }
        }
    }

    // One program per chip: each core's kernel carries its peer and a fabric connection toward the peer's chip.
    std::map<uint32_t, Program> programs;
    for (const auto& [chip, _] : node_of) {
        programs.emplace(chip, CreateProgram());
    }
    auto arm = [&](uint32_t chip,
                   const CoreCoord& core,
                   uint32_t role,
                   uint32_t peer_chip,
                   const CoreCoord& peer,
                   uint32_t link_idx) {
        Program& program = programs.at(chip);
        const auto kid = CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_streaming_profiler_pingpong_fabric/kernels/"
            "pingpong_fabric_dm.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        const CoreCoord vpeer = mesh_device->worker_core_from_logical_core(peer);
        const tt::tt_fabric::FabricNodeId& src = node_of.at(chip);
        const tt::tt_fabric::FabricNodeId& dst = node_of.at(peer_chip);
        std::vector<uint32_t> args = {
            role,
            static_cast<uint32_t>(vpeer.x),
            static_cast<uint32_t>(vpeer.y),
            kFlagAddr,
            rounds,
            static_cast<uint32_t>(dst.chip_id),
            static_cast<uint32_t>(dst.mesh_id.get())};
        tt::tt_fabric::append_fabric_connection_rt_args(src, dst, link_idx, program, core, args);
        SetRuntimeArgs(program, kid, core, args);
    };
    for (const Pair& p : pairs) {
        arm(p.chip_a, p.core_a, 0, p.chip_b, p.core_b, p.link_a);
        arm(p.chip_b, p.core_b, 1, p.chip_a, p.core_a, p.link_b);
    }
    std::vector<uint32_t> zero = {0, 0};
    for (const Pair& p : pairs) {
        detail::WriteToDeviceL1(mesh_device->get_device(coord_of.at(p.chip_a)), p.core_a, kFlagAddr, zero);
        detail::WriteToDeviceL1(mesh_device->get_device(coord_of.at(p.chip_b)), p.core_b, kFlagAddr, zero);
    }
    // The clock trackers and the host line need a moment before records can be placed.
    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    std::printf("[pingpong-fabric] %zu linked pairs x %u rounds on %zu chips\n", pairs.size(), rounds, node_of.size());
    std::fflush(stdout);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    for (auto& [chip, program] : programs) {
        const auto& c = coord_of.at(chip);
        workload.add_program(distributed::MeshCoordinateRange(c, c), std::move(program));
    }
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);
    std::printf("[pingpong-fabric] workload done\n");
    std::fflush(stdout);
    for (const Pair& p : pairs) {
        for (const auto& [chip, core] : {std::pair{p.chip_a, p.core_a}, std::pair{p.chip_b, p.core_b}}) {
            std::vector<uint32_t> words(2, 0);
            detail::ReadFromDeviceL1(mesh_device->get_device(coord_of.at(chip)), core, kFlagAddr, 8, words);
            if (words[1] != 0) {
                std::printf(
                    "[pingpong-fabric] chip %u core (%zu,%zu) gave up waiting for round %u\n",
                    chip,
                    core.x,
                    core.y,
                    words[1]);
            }
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    mesh_device->close();

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
    std::printf(
        "pair        rounds   one-way A->B ns (p10 / min / mean / sd)   one-way B->A ns (p10 / min / mean / sd)   "
        "offset error B-A ns (p10, mean)\n");
    // Half the difference of the two directions' 10th percentiles: the latency floor is the same path both ways,
    // while the mean carries router queueing tails of hundreds of ns that differ by direction.
    auto p10 = [](std::vector<double> v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 10, v.end());
        return v[v.size() / 10];
    };
    std::map<std::pair<uint32_t, uint32_t>, double> err;  // (a,b) -> estimated placement error of B relative to A
    std::vector<double> errs, errs_mean;
    for (const Pair& p : pairs) {
        auto ia = by_core.find({p.chip_a, p.core_a}), ib = by_core.find({p.chip_b, p.core_b});
        if (ia == by_core.end() || ib == by_core.end()) {
            std::printf("chip %u - chip %u: no stamps\n", p.chip_a, p.chip_b);
            continue;
        }
        const auto& [tx_a, rx_a] = ia->second;
        const auto& [tx_b, rx_b] = ib->second;
        const size_t n = std::min({tx_a.size(), rx_a.size(), tx_b.size(), rx_b.size()});
        if (n != rounds) {
            std::printf(
                "chip %u - chip %u: stamps %zu/%zu/%zu/%zu of %u rounds, skipped\n",
                p.chip_a,
                p.chip_b,
                tx_a.size(),
                rx_b.size(),
                tx_b.size(),
                rx_a.size(),
                rounds);
            continue;
        }
        std::vector<double> fwd, bwd;
        for (size_t k = 0; k < n; k++) {
            const int64_t ta = tsc_of(tx_a[k]), rb = tsc_of(rx_b[k]), tb = tsc_of(tx_b[k]), ra = tsc_of(rx_a[k]);
            if (ta == 0 || rb == 0 || tb == 0 || ra == 0) {
                continue;
            }
            fwd.push_back(ns_of(rb - ta));
            bwd.push_back(ns_of(ra - tb));
        }
        if (fwd.empty()) {
            std::printf("chip %u - chip %u: no placeable rounds\n", p.chip_a, p.chip_b);
            continue;
        }
        const double e = (p10(fwd) - p10(bwd)) / 2;
        const double e_mean = (mean(fwd) - mean(bwd)) / 2;
        err[{p.chip_a, p.chip_b}] = e;
        errs.push_back(e);
        errs_mean.push_back(e_mean);
        std::printf(
            "chip %u - chip %u  %5zu   %7.1f / %7.1f / %7.1f / %6.1f        %7.1f / %7.1f / %7.1f / %6.1f        "
            "%+6.2f  (%+6.2f)\n",
            p.chip_a,
            p.chip_b,
            fwd.size(),
            p10(fwd),
            *std::min_element(fwd.begin(), fwd.end()),
            mean(fwd),
            stdev(fwd),
            p10(bwd),
            *std::min_element(bwd.begin(), bwd.end()),
            mean(bwd),
            stdev(bwd),
            e,
            e_mean);
    }
    for (const auto& [name, v] : {std::pair{"p10", &errs}, std::pair{"mean", &errs_mean}}) {
        if (v->empty()) {
            continue;
        }
        double ss = 0, worst = 0;
        for (double e : *v) {
            ss += e * e;
            worst = std::max(worst, std::fabs(e));
        }
        std::printf(
            "offset error over %zu links (%s): rms %.2f ns, worst %.2f ns\n",
            v->size(),
            name,
            std::sqrt(ss / static_cast<double>(v->size())),
            worst);
    }
    // Closures around 4-cycles of links: a per-link bias common to the sync and this test cancels around a loop
    // only if it is a real clock offset; a path asymmetry does not.
    auto signed_err = [&](uint32_t a, uint32_t b, double& out) {
        if (auto it = err.find({std::min(a, b), std::max(a, b)}); it != err.end()) {
            out = a < b ? it->second : -it->second;
            return true;
        }
        return false;
    };
    std::vector<uint32_t> chips;
    for (const auto& [chip, _] : node_of) {
        chips.push_back(chip);
    }
    std::set<std::vector<uint32_t>> seen;
    for (uint32_t a : chips) {
        for (uint32_t b : chips) {
            for (uint32_t c : chips) {
                for (uint32_t d : chips) {
                    if (a >= b || a >= c || a >= d || b == c || b == d || c == d || b > d) {
                        continue;
                    }
                    double e1, e2, e3, e4;
                    if (signed_err(a, b, e1) && signed_err(b, c, e2) && signed_err(c, d, e3) && signed_err(d, a, e4)) {
                        std::printf("loop %u-%u-%u-%u closes to %+.2f ns\n", a, b, c, d, e1 + e2 + e3 + e4);
                    }
                }
            }
        }
    }
    sp::UnregisterCallback(sub);
    return 0;
}
