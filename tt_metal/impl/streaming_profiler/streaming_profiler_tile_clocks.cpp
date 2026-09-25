// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_tile_clocks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "impl/kernels/kernel.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

const TileClock* TileClocks::find(CoreType type, const CoreCoord& logical) const {
    for (const TileClock& t : tiles) {
        if (t.type == type && t.logical == logical) {
            return &t;
        }
    }
    return nullptr;
}

namespace {

constexpr uint32_t kReps = 256;

struct Node {
    CoreType type;
    bool active_eth = false;
    CoreCoord logical, virt, phys;
    uint32_t scratch = 0;       // the kernel's L1 scratch, as the tile addresses it
    uint64_t host_scratch = 0;  // the same scratch as the host addresses it over the NoC (tagged on a DRAM tile)
};

// One tile's reading of a partner over the NoC that runs straight from the tile to it: the median and the quartile
// spread of 2 * (partner wall - bracket midpoint) in the clocks' low words, the median round trip, in ticks, over
// the network's reads, and one coarse whole-clock difference that places the median in its 2^32-tick turn.
struct Reading {
    uint32_t s, t, noc;
    int32_t median2 = 0, spread2 = 0, rtt = 0;
    int64_t coarse = 0;
    int64_t whole2() const {
        const double turns = (static_cast<double>(coarse) - median2 / 2.0) / 4294967296.0;
        return median2 + 2 * std::llround(turns) * int64_t{4294967296};
    }
};

// NoC 0 runs towards higher raw coordinates, NoC 1 towards lower.
bool aligned(const CoreCoord& a, const CoreCoord& b) { return a != b && (a.x == b.x || a.y == b.y); }
bool upward(const CoreCoord& a, const CoreCoord& b) { return a.x == b.x ? b.y > a.y : b.x > a.x; }

uint32_t packed_xy(const CoreCoord& c) { return (static_cast<uint32_t>(c.y) << 16) | static_cast<uint32_t>(c.x); }

// Solves the normal equations N x = r by Gaussian elimination with partial pivoting; a pivot of zero leaves that
// unknown at 0.
std::vector<double> solve_normal(std::vector<std::vector<double>> N, std::vector<double> r) {
    const size_t n = r.size();
    std::vector<double> x(n, 0.0);
    for (size_t c = 0; c < n; c++) {
        size_t piv = c;
        for (size_t k = c + 1; k < n; k++) {
            if (std::fabs(N[k][c]) > std::fabs(N[piv][c])) {
                piv = k;
            }
        }
        std::swap(N[c], N[piv]);
        std::swap(r[c], r[piv]);
        if (std::fabs(N[c][c]) < 1e-9) {
            continue;
        }
        for (size_t k = 0; k < n; k++) {
            if (k == c || N[k][c] == 0.0) {
                continue;
            }
            const double f = N[k][c] / N[c][c];
            for (size_t j = c; j < n; j++) {
                N[k][j] -= f * N[c][j];
            }
            r[k] -= f * r[c];
        }
    }
    for (size_t c = 0; c < n; c++) {
        if (std::fabs(N[c][c]) >= 1e-9) {
            x[c] = r[c] / N[c][c];
        }
    }
    return x;
}

// Every tile with a RISC: the whole Tensix grid (the dispatch cores included), every eth core, every DRAM core Metal
// may place a kernel on. Where a kernel scratches, and zeroes on exit: a compute core in the user L1 the allocator
// hands out (nothing is allocated yet); a dispatch core in the profiler ring space past its control vector (drained
// by nobody); an eth or DRAM core at the bottom of its unreserved region, below the pusher's and the link ends'
// carves and under the relay's staging, which the relay fills before it sends. A ring the profiler decodes is not
// usable even zeroed: the pusher's and the relay's first frames broke with the scratch there.
std::vector<Node> enumerate_nodes(IDevice* device, ContextId ctx) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto& soc = cluster.get_soc_desc(chip);
    std::vector<Node> nodes;
    const auto add = [&](CoreType type, const CoreCoord& logical, bool active_eth, uint32_t scratch, uint64_t host) {
        nodes.push_back(Node{
            .type = type,
            .active_eth = active_eth,
            .logical = logical,
            .virt = cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, type),
            .phys = cluster.get_physical_coordinate_from_logical_coordinates(chip, logical, type, /*no_warn=*/true),
            .scratch = scratch,
            .host_scratch = host});
    };
    const auto ring_space = [&](HalProgrammableCoreType t) {
        TT_FATAL(
            hal.get_dev_size(t, HalL1MemAddrType::PROFILER) >=
                kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE + kernel_profiler::kTileNetScratchBytes,
            "streaming profiler: a profiler L1 region cannot hold the tile clock scratch");
        return static_cast<uint32_t>(hal.get_dev_addr(t, HalL1MemAddrType::PROFILER)) +
               kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE;
    };
    const uint32_t user_l1 = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t dispatch_scratch = ring_space(HalProgrammableCoreType::TENSIX);
    const CoreCoord compute = device->compute_with_storage_grid_size();
    const CoreCoord grid = soc.get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            const bool is_compute = x < compute.x && y < compute.y;
            const uint32_t scratch = is_compute ? user_l1 : dispatch_scratch;
            add(CoreType::WORKER, CoreCoord{x, y}, false, scratch, scratch);
        }
    }
    const auto sorted = [](const std::unordered_set<CoreCoord>& cores) {
        std::vector<CoreCoord> out(cores.begin(), cores.end());
        std::sort(out.begin(), out.end(), [](const CoreCoord& a, const CoreCoord& b) {
            return a.y != b.y ? a.y < b.y : a.x < b.x;
        });
        return out;
    };
    if (hal.has_programmable_core_type(HalProgrammableCoreType::IDLE_ETH)) {
        const uint32_t scratch = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
        for (const CoreCoord& l : sorted(device->get_inactive_ethernet_cores())) {
            add(CoreType::ETH, l, false, scratch, scratch);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::ACTIVE_ETH)) {
        const uint32_t scratch = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
        for (const CoreCoord& l : sorted(device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false))) {
            add(CoreType::ETH, l, true, scratch, scratch);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        // A DRAM core's physical coordinate from the cluster is its translated one; the NoC 0 grid position that
        // says which Tensix row it shares comes from the SoC descriptor, in the same order.
        const uint32_t scratch = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        const uint64_t host = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        const std::vector<CoreCoord> logical = soc.get_metal_dram_cores(CoordSystem::LOGICAL);
        const std::vector<CoreCoord> noc0 = soc.get_metal_dram_cores(CoordSystem::NOC0);
        TT_FATAL(logical.size() == noc0.size(), "streaming profiler: DRAM core lists disagree");
        for (size_t i = 0; i < logical.size(); i++) {
            add(CoreType::DRAM, logical[i], false, scratch, host);
            nodes.back().phys = noc0[i];
        }
    }
    return nodes;
}

const char* kind_name(CoreType t) { return t == CoreType::WORKER ? "Tensix" : t == CoreType::DRAM ? "DRAM" : "eth"; }

// The kernel on every initiator, one program per kind of core, then the go to one initiator at a time so nothing
// else is on the NoC while it reads. Each initiator reads every node in its row and column. Every kernel is released
// at the end, and on any failure, so no tile is left waiting on its go word.
std::vector<Reading> read_network(
    IDevice* device,
    ContextId ctx,
    const std::vector<Node>& nodes,
    const std::vector<uint32_t>& initiators,
    std::vector<Reading>* both_nocs = nullptr) {
    auto& cluster = MetalContext::instance(ctx).get_cluster();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const uint32_t nonce = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 0x10u;
    std::vector<Reading> readings;
    std::vector<std::vector<uint32_t>> args(nodes.size());
    std::vector<std::vector<Reading*>> dest(nodes.size());
    std::vector<std::pair<uint32_t, size_t>> dest_idx;
    for (uint32_t s : initiators) {
        args[s] = {nodes[s].scratch, kReps, nonce, 0};
        for (uint32_t t = 0; t < nodes.size(); t++) {
            if (!aligned(nodes[s].phys, nodes[t].phys)) {
                continue;
            }
            const bool up = upward(nodes[s].phys, nodes[t].phys);
            args[s].push_back((up ? 0u : 1u << 31) | packed_xy(nodes[t].virt));
            readings.push_back(Reading{s, t, up ? 0u : 1u});
            dest_idx.emplace_back(s, readings.size() - 1);
        }
    }
    // One idle eth tile reads every eth tile in its row on the other NoC too: around the ring the two NoCs' hops
    // cancel, and one initiator's own latency is common to every target.
    if (both_nocs != nullptr) {
        both_nocs->clear();
        const auto q = std::find_if(initiators.begin(), initiators.end(), [&](uint32_t i) {
            return nodes[i].type == CoreType::ETH && !nodes[i].active_eth;
        });
        if (q != initiators.end()) {
            for (uint32_t t = 0; t < nodes.size(); t++) {
                if (nodes[t].type != CoreType::ETH || t == *q || !aligned(nodes[*q].phys, nodes[t].phys)) {
                    continue;
                }
                const bool up = upward(nodes[*q].phys, nodes[t].phys);
                args[*q].push_back((up ? 1u << 31 : 0u) | packed_xy(nodes[t].virt));
                both_nocs->push_back(Reading{*q, t, up ? 1u : 0u});
            }
        }
    }
    for (const auto& [i, k] : dest_idx) {
        dest[i].push_back(&readings[k]);
    }
    if (both_nocs != nullptr) {
        for (Reading& r : *both_nocs) {
            dest[r.s].push_back(&r);
        }
    }
    for (uint32_t s : initiators) {
        args[s][3] = static_cast<uint32_t>(args[s].size() - 4);
        TT_FATAL(
            args[s][3] <= kernel_profiler::kTileNetMaxPartners,
            "streaming profiler: tile ({},{}) has {} row and column partners, the table holds {}",
            nodes[s].logical.x,
            nodes[s].logical.y,
            args[s][3],
            kernel_profiler::kTileNetMaxPartners);
    }
    enum Kind { kTensix, kIdleEth, kActiveEth, kDram, kKinds };
    const auto kind_of = [](const Node& n) {
        return n.type == CoreType::WORKER ? kTensix
               : n.type == CoreType::DRAM ? kDram
               : n.active_eth             ? kActiveEth
                                          : kIdleEth;
    };
    std::vector<std::set<CoreRange>> ranges(kKinds);
    for (uint32_t s : initiators) {
        ranges[kind_of(nodes[s])].insert(CoreRange(nodes[s].logical, nodes[s].logical));
    }
    std::vector<Program> programs;
    std::vector<KernelHandle> kids;
    std::vector<int> program_of(kKinds, -1);
    const char* src = "tt_metal/tools/profiler/sync/tile_sync.cpp";
    for (int k = 0; k < kKinds; k++) {
        if (ranges[k].empty()) {
            continue;
        }
        program_of[k] = static_cast<int>(programs.size());
        programs.push_back(CreateProgram());
        Program& p = programs.back();
        const CoreRangeSet cores(ranges[k]);
        switch (static_cast<Kind>(k)) {
            case kTensix:
                kids.push_back(CreateKernel(
                    p,
                    src,
                    cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}));
                break;
            case kIdleEth:
                kids.push_back(
                    CreateKernel(p, src, cores, EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::RISCV_0_default}));
                break;
            case kActiveEth:
                kids.push_back(CreateKernel(p, src, cores, EthernetConfig{.noc = NOC::RISCV_0_default}));
                break;
            case kDram: kids.push_back(CreateKernel(p, src, cores, DramConfig{.noc = NOC::NOC_0})); break;
            case kKinds: break;
        }
    }
    const auto table_of = [&](uint32_t s) { return nodes[s].host_scratch + kernel_profiler::kTileNetTable; };
    // Fast dispatch's go signal reaches every core of the grid, and a host-launched kernel leaves its launch slot
    // valid on exit, so the slots go back to the firmware's initial message once the kernels have exited.
    const auto& hal = MetalContext::instance(ctx).hal();
    const auto core_type_of = [](const Node& n) {
        return n.type == CoreType::WORKER ? HalProgrammableCoreType::TENSIX
               : n.type == CoreType::DRAM ? HalProgrammableCoreType::DRAM
               : n.active_eth             ? HalProgrammableCoreType::ACTIVE_ETH
                                          : HalProgrammableCoreType::IDLE_ETH;
    };
    const auto invalidate_launch = [&](const Node& n) {
        const HalProgrammableCoreType t = core_type_of(n);
        auto msg = hal.get_dev_msgs_factory(t).create<dev_msgs::launch_msg_t>();
        const uint64_t addr = t == HalProgrammableCoreType::DRAM ? hal.get_dev_noc_addr(t, HalL1MemAddrType::LAUNCH)
                                                                 : hal.get_dev_addr(t, HalL1MemAddrType::LAUNCH);
        cluster.write_core(msg.data(), static_cast<uint32_t>(msg.size()), tt_cxy_pair(chip, n.virt), addr);
    };
    // The firmware seeds its profiler ring position from the control vector at the first launch of its session,
    // which this is; a tail left by the previous session would put it thousands of words ahead of the relay's head.
    const auto zero_control = [&](const Node& n) {
        const HalProgrammableCoreType t = core_type_of(n);
        const std::vector<uint8_t> zero(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
        const uint64_t addr = t == HalProgrammableCoreType::DRAM ? hal.get_dev_noc_addr(t, HalL1MemAddrType::PROFILER)
                                                                 : hal.get_dev_addr(t, HalL1MemAddrType::PROFILER);
        cluster.write_core(zero.data(), static_cast<uint32_t>(zero.size()), tt_cxy_pair(chip, n.virt), addr);
    };
    const uint32_t zero[2] = {0, 0};
    for (uint32_t s : initiators) {
        const int pi = program_of[kind_of(nodes[s])];
        SetRuntimeArgs(programs[pi], kids[pi], nodes[s].logical, args[s]);
        cluster.write_core(zero, sizeof(zero), tt_cxy_pair(chip, nodes[s].virt), table_of(s));
        zero_control(nodes[s]);
    }
    for (Program& p : programs) {
        detail::CompileProgram(device, p, /*force_slow_dispatch=*/true);
        detail::WriteRuntimeArgsToDevice(device, p, /*force_slow_dispatch=*/true);
        detail::LaunchProgram(device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    }
    const auto release = [&]() {
        for (uint32_t s : initiators) {
            const uint32_t go = kernel_profiler::kTileNetGoExit;
            cluster.write_core(&go, sizeof(go), tt_cxy_pair(chip, nodes[s].virt), table_of(s));
        }
        for (Program& p : programs) {
            detail::WaitProgramDone(device, p, false);
        }
        for (uint32_t s : initiators) {
            invalidate_launch(nodes[s]);
        }
    };
    try {
        for (uint32_t s : initiators) {
            const uint32_t n = args[s][3];
            const tt_cxy_pair core(chip, nodes[s].virt);
            std::vector<uint32_t> t(kernel_profiler::TILE_NET_OUT_0 + kernel_profiler::TILE_NET_OUT_WORDS * n, 0);
            const auto await = [&](uint32_t ready, const char* what) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                for (;;) {
                    cluster.read_core(t.data(), static_cast<uint32_t>(t.size() * sizeof(uint32_t)), core, table_of(s));
                    if (t[kernel_profiler::TILE_NET_READY] == ready) {
                        return;
                    }
                    TT_FATAL(
                        std::chrono::steady_clock::now() < deadline,
                        "streaming profiler: device {} {} tile ({},{}) did not {} within 2 s",
                        chip,
                        kind_name(nodes[s].type),
                        nodes[s].logical.x,
                        nodes[s].logical.y,
                        what);
                }
            };
            await(nonce, "come up for its tile clock reads");
            const uint32_t go = kernel_profiler::kTileNetGoMeasure;
            cluster.write_core(&go, sizeof(go), core, table_of(s));
            await(~nonce, "finish its tile clock reads");
            for (uint32_t m = 0; m < n; m++) {
                const uint32_t w = kernel_profiler::TILE_NET_OUT_0 + kernel_profiler::TILE_NET_OUT_WORDS * m;
                Reading& r = *dest[s][m];
                r.median2 = static_cast<int32_t>(t[w]);
                r.spread2 = static_cast<int32_t>(t[w + 1]);
                r.rtt = static_cast<int32_t>(t[w + 2]);
                r.coarse = static_cast<int64_t>((uint64_t{t[w + 4]} << 32) | t[w + 3]);
            }
        }
    } catch (...) {
        release();
        throw;
    }
    release();
    return readings;
}

void measure_chip(IDevice* device, ContextId ctx) {
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<Node> nodes = enumerate_nodes(device, ctx);
    std::vector<uint32_t> all(nodes.size());
    for (uint32_t i = 0; i < nodes.size(); i++) {
        all[i] = i;
    }
    std::vector<Reading> both;
    const std::vector<Reading> readings = read_network(device, ctx, nodes, all, &both);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // A pair: the lower node's NoC 0 reading of the higher and the higher's NoC 1 reading of the lower. Half their
    // difference is the offset; their round trips must agree.
    std::map<std::pair<uint32_t, uint32_t>, const Reading*> by_pair;
    for (const Reading& r : readings) {
        by_pair[{r.s, r.t}] = &r;
    }
    struct Pair {
        uint32_t lo, hi;
        double offset, rtt_diff;
    };
    std::vector<Pair> pairs;
    // An active eth tile's own reads leave its NIU ~6 cycles later than an idle eth tile's (their mirrored round
    // trips differ by 6.0 ticks), which biases a mirrored pair with one by a tick and a half; those tiles are placed
    // from the both-NoC reads instead.
    for (const Reading& r : readings) {
        if (r.noc != 0 || nodes[r.s].active_eth || nodes[r.t].active_eth) {
            continue;
        }
        const auto m = by_pair.find({r.t, r.s});
        if (m == by_pair.end()) {
            continue;
        }
        pairs.push_back(Pair{
            r.s,
            r.t,
            static_cast<double>(r.whole2() - m->second->whole2()) / 4.0,
            static_cast<double>(r.rtt - m->second->rtt)});
    }
    // x[i] = tile i's wall tick minus tile 0's; x[0] = 0.
    const size_t n = nodes.size();
    std::vector<std::vector<double>> N(n - 1, std::vector<double>(n - 1, 0.0));
    std::vector<double> rhs(n - 1, 0.0);
    const auto col = [](uint32_t i) { return static_cast<int32_t>(i) - 1; };
    for (const Pair& p : pairs) {
        const int32_t cols[2] = {col(p.hi), col(p.lo)};
        const double coef[2] = {1.0, -1.0};
        for (int i = 0; i < 2; i++) {
            if (cols[i] < 0) {
                continue;
            }
            rhs[cols[i]] += coef[i] * p.offset;
            for (int j = 0; j < 2; j++) {
                if (cols[j] >= 0) {
                    N[cols[i]][cols[j]] += coef[i] * coef[j];
                }
            }
        }
    }
    std::vector<double> x = solve_normal(N, rhs);
    {
        // est(t): the both-NoC initiator's two readings of eth tile t averaged, t's offset from it plus a constant
        // common to every target. Active eth tiles take the idle eth targets' mean placement through it.
        std::map<uint32_t, std::pair<const Reading*, const Reading*>> of_q;
        for (const Reading& r : both) {
            of_q[r.t].second = &r;
        }
        for (const Reading& r : readings) {
            if (!both.empty() && r.s == both.front().s && of_q.contains(r.t)) {
                of_q[r.t].first = &r;
            }
        }
        const auto est = [&](uint32_t t) {
            const auto& [a, b] = of_q.at(t);
            return static_cast<double>(a->whole2() + b->whole2()) / 4.0;
        };
        double shift = 0.0;
        size_t refs = 0;
        for (const auto& [t, ab] : of_q) {
            if (ab.first != nullptr && ab.second != nullptr && !nodes[t].active_eth && t != 0) {
                shift += x[t - 1] - est(t);
                refs++;
            }
        }
        if (refs != 0) {
            shift /= static_cast<double>(refs);
            for (const auto& [t, ab] : of_q) {
                if (ab.first != nullptr && ab.second != nullptr && nodes[t].active_eth) {
                    x[t - 1] = shift + est(t);
                }
            }
        }
    }
    const auto x_of = [&](uint32_t i) { return i == 0 ? 0.0 : x[i - 1]; };

    double closure_ss = 0.0, closure_worst = 0.0, rtt_ss = 0.0, rtt_worst = 0.0;
    std::map<std::pair<CoreType, CoreType>, std::pair<double, size_t>> rtt_by_kind;
    for (const Pair& p : pairs) {
        const double r = p.offset - (x_of(p.hi) - x_of(p.lo));
        closure_ss += r * r;
        closure_worst = std::max(closure_worst, std::fabs(r));
        rtt_ss += p.rtt_diff * p.rtt_diff;
        rtt_worst = std::max(rtt_worst, std::fabs(p.rtt_diff));
        auto key = std::minmax(nodes[p.lo].type, nodes[p.hi].type);
        auto& acc = rtt_by_kind[{key.first, key.second}];
        acc.first += p.rtt_diff;
        acc.second++;
    }
    std::vector<double> spreads;
    for (const Reading& r : readings) {
        spreads.push_back(r.spread2 / 2.0);
    }
    std::sort(spreads.begin(), spreads.end());
    double frac_worst = 0.0;
    struct Span {
        double lo = 0.0, hi = 0.0;
        size_t count = 0;
    };
    std::map<CoreType, Span> spans;
    TileClocks clocks;
    for (uint32_t i = 0; i < n; i++) {
        frac_worst = std::max(frac_worst, std::fabs(x_of(i) - std::llround(x_of(i))));
        Span& sp = spans[nodes[i].type];
        sp.lo = sp.count == 0 ? x_of(i) : std::min(sp.lo, x_of(i));
        sp.hi = sp.count == 0 ? x_of(i) : std::max(sp.hi, x_of(i));
        sp.count++;
        clocks.tiles.push_back(
            TileClock{nodes[i].type, nodes[i].logical, nodes[i].virt, nodes[i].phys, std::llround(x_of(i))});
    }
    std::string kinds;
    for (const auto& [key, acc] : rtt_by_kind) {
        kinds += fmt::format(
            "{}{}-{} {:+.2f}",
            kinds.empty() ? "" : ", ",
            kind_name(key.first),
            kind_name(key.second),
            acc.first / acc.second);
    }
    const double ns_per_tick =
        1e3 / static_cast<double>(MetalContext::instance(ctx).get_cluster().get_device_aiclk(chip));
    std::string span_text;
    for (const auto& [type, sp] : spans) {
        span_text += fmt::format(
            "{}{} {} tiles {:+.0f}..{:+.0f} ticks ({:.1f} ns wide)",
            span_text.empty() ? "" : "; ",
            sp.count,
            kind_name(type),
            sp.lo,
            sp.hi,
            (sp.hi - sp.lo) * ns_per_tick);
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] Device {}: tile clocks against the first Tensix tile: {}; {} mirrored pairs x 2 x {} "
        "reads in {:.0f} ms; closures {:.2f} ticks rms, {:.2f} worst; offsets {:.2f} ticks from an integer at worst; "
        "mirrored round trips disagree {:.2f} ticks rms, {:.1f} worst (mean by pair: {}); sample quartile spread "
        "{:.1f} ticks median, {:.1f} worst",
        chip,
        span_text,
        pairs.size(),
        kReps,
        ms,
        pairs.empty() ? 0.0 : std::sqrt(closure_ss / static_cast<double>(pairs.size())),
        closure_worst,
        frac_worst,
        pairs.empty() ? 0.0 : std::sqrt(rtt_ss / static_cast<double>(pairs.size())),
        rtt_worst,
        kinds,
        spreads.empty() ? 0.0 : spreads[spreads.size() / 2],
        spreads.empty() ? 0.0 : spreads.back());
    service().set_tile_clocks(chip, std::move(clocks));
}

}  // namespace

void measure_tile_clocks(IDevice* device, ContextId ctx) {
    const auto& mc = MetalContext::instance(ctx);
    if (!mc.rtoptions().get_streaming_profiler_enabled() || mc.hal().get_arch() == tt::ARCH::QUASAR ||
        service().tile_clocks(static_cast<uint32_t>(device->id())) != nullptr) {
        return;
    }
    try {
        measure_chip(device, ctx);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: tile clocks not measured ({}); its lanes are placed as if every tile "
            "kept the pusher's clock",
            device->id(),
            e.what());
    }
}

}  // namespace tt::tt_metal::streaming_profiler
