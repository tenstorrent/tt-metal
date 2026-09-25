// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Cross-chip acceptance test for the d2d sync: on every pair of chips joined by an ethernet link, one worker core on
// each end exchanges fabric atomics with a zone stamped at each end, both directions. On the host timeline the two
// one-way times must agree; half their difference is the placement error between the two chips as seen by tensix
// cores through the fabric, independent of the ethernet stamping the sync itself is built on (the link's own
// direction asymmetry is common to both). Exits nonzero if any worker gave up or any pair is short of placed stamps.
// Run with TT_METAL_STREAMING_PROFILER=1.
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

struct Pair {
    uint32_t chip_a, chip_b;  // chip_a < chip_b; A has role 0
    CoreCoord core_a, core_b;
    uint32_t link_a, link_b;  // the fabric link index each end sends on: the two ends of one cable
};
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
    std::map<std::pair<uint32_t, CoreCoord>, std::pair<std::vector<sp::Zone>, std::vector<sp::Zone>>> by_core;
    const auto sub = sp::RegisterCallback("pingpong-fabric", [&](const sp::Batch<sp::RecordType::Zones>& b) {
        std::lock_guard<std::mutex> g(mu);
        for (const sp::Zone& z : b.zones()) {
            const std::string_view name = z.site().name;
            if (name == "PP_TX" || name == "PP_RX") {
                auto& e = by_core[{z.core().chip_id, z.core().logical}];
                (name == "PP_TX" ? e.first : e.second).push_back(z);
            }
        }
    });

    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);

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
    const auto offset_key = [](int dx, int dy) {
        return std::tuple(std::abs(dx) + std::abs(dy), std::abs(dx), dy < 0);
    };
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
            std::optional<std::tuple<int, int, CoreCoord, CoreCoord>> best;
            for (const auto& [pa, la] : workers_at[a]) {
                const int dx = pa.first - static_cast<int>(ra.x), dy = pa.second - static_cast<int>(ra.y);
                const auto wb = workers_at[b].find({static_cast<int>(rb.x) + dx, static_cast<int>(rb.y) + dy});
                if (wb == workers_at[b].end() || used[a].count(la) || used[b].count(wb->second)) {
                    continue;
                }
                if (!best || offset_key(dx, dy) < offset_key(std::get<0>(*best), std::get<1>(*best))) {
                    best.emplace(dx, dy, la, wb->second);
                }
            }
            if (!best) {
                std::fprintf(stderr, "no common worker offset from the link cores of chips %u and %u\n", a, b);
                return 1;
            }
            const auto& [dx, dy, wa, wb] = *best;
            pairs.push_back(Pair{a, b, wa, wb, link_a, *link_b});
            used[a].insert(wa);
            used[b].insert(wb);
            std::printf(
                "[pingpong-fabric] chip %u worker (%zu,%zu) <-> chip %u worker (%zu,%zu): link cores NoC-0 "
                "(%zu,%zu) and (%zu,%zu), workers at offset (%+d,%+d) from them, channels %u and %u\n",
                a,
                wa.x,
                wa.y,
                b,
                wb.x,
                wb.y,
                static_cast<size_t>(ra.x),
                static_cast<size_t>(ra.y),
                static_cast<size_t>(rb.x),
                static_cast<size_t>(rb.y),
                dx,
                dy,
                chan_a,
                chan_b);
        }
    }

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
    // The link solves' windows fill before the exchange starts.
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
    uint32_t failures = 0;
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
                failures++;
            }
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    mesh_device->close();
    sp::UnregisterCallback(sub);

    const auto unplaced = [](const std::vector<sp::Zone>& v) {
        return std::count_if(
            v.begin(), v.end(), [](const sp::Zone& z) { return z.start_time().time_since_epoch().count() == 0; });
    };
    const auto one_way_ns = [](const sp::Zone& tx, const sp::Zone& rx) {
        return std::chrono::duration<double, std::nano>(rx.start_time() - tx.start_time()).count();
    };
    std::printf("pair        rounds   one-way A->B ns (p10)   one-way B->A ns (p10)   offset error B-A ns (p10)\n");
    // Half the difference of the two directions' 10th percentiles: the latency floor is the same path both ways,
    // while the mean carries router queueing tails of hundreds of ns that differ by direction.
    auto p10 = [](std::vector<double> v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 10, v.end());
        return v[v.size() / 10];
    };
    std::vector<double> errs;
    for (const Pair& p : pairs) {
        auto ia = by_core.find({p.chip_a, p.core_a}), ib = by_core.find({p.chip_b, p.core_b});
        if (ia == by_core.end() || ib == by_core.end()) {
            std::printf("chip %u - chip %u: no stamps\n", p.chip_a, p.chip_b);
            failures++;
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
            failures++;
            continue;
        }
        const auto n_unplaced = unplaced(tx_a) + unplaced(rx_a) + unplaced(tx_b) + unplaced(rx_b);
        if (n_unplaced != 0) {
            std::printf(
                "chip %u - chip %u: %lld unplaced stamps, skipped\n",
                p.chip_a,
                p.chip_b,
                static_cast<long long>(n_unplaced));
            failures++;
            continue;
        }
        std::vector<double> fwd, bwd;
        for (size_t k = 0; k < n; k++) {
            fwd.push_back(one_way_ns(tx_a[k], rx_b[k]));
            bwd.push_back(one_way_ns(tx_b[k], rx_a[k]));
        }
        const double f10 = p10(fwd), b10 = p10(bwd), e = (f10 - b10) / 2;
        errs.push_back(e);
        std::printf(
            "chip %u - chip %u  %5zu   %7.1f                 %7.1f                 %+6.2f\n",
            p.chip_a,
            p.chip_b,
            fwd.size(),
            f10,
            b10,
            e);
    }
    if (!errs.empty()) {
        double ss = 0, worst = 0;
        for (double e : errs) {
            ss += e * e;
            worst = std::max(worst, std::fabs(e));
        }
        std::printf(
            "offset error over %zu links (p10): rms %.2f ns, worst %.2f ns\n",
            errs.size(),
            std::sqrt(ss / static_cast<double>(errs.size())),
            worst);
    }
    return failures == 0 ? 0 : 1;
}
