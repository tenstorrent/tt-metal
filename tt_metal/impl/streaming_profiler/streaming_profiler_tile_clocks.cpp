// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_tile_clocks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "hostdev/streaming_profiler_sync.h"
#include "impl/kernels/kernel.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"
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
    HalProgrammableCoreType core;
    CoreCoord logical, virt, phys;
    uint32_t scratch = 0;       // the kernel's L1 scratch, as the tile addresses it
    uint64_t host_scratch = 0;  // the same scratch as the host addresses it over the NoC (tagged on a DRAM tile)
};

// One tile's reading of a partner over the NoC that runs straight from the tile to it: the median of
// 2 * (partner wall - bracket midpoint) in the clocks' low words, the median round trip, in ticks, over the network's
// reads, and one coarse whole-clock difference that places the median in its 2^32-tick turn. A both-NoC reading runs
// on the other NoC and names its twin, the same initiator's reading of the same target on the direct NoC.
struct Reading {
    uint32_t s, t, noc;
    bool both = false;
    uint32_t twin = 0;
    int32_t median2 = 0, rtt = 0;
    int64_t coarse = 0;
    int64_t whole2() const {
        return 2 * coarse + static_cast<int32_t>(static_cast<uint32_t>(median2) - static_cast<uint32_t>(2 * coarse));
    }
};

// NoC 0 runs towards higher raw coordinates, NoC 1 towards lower.
bool aligned(const CoreCoord& a, const CoreCoord& b) { return a != b && (a.x == b.x || a.y == b.y); }
bool upward(const CoreCoord& a, const CoreCoord& b) { return a.x == b.x ? b.y > a.y : b.x > a.x; }

// Solves N x = r in place for a symmetric positive definite N by Cholesky, leaving x in r. Returns the first row whose
// pivot is not positive, or r.size() when every pivot is. On a grounded Laplacian a pivot at rounding level means
// that row's node has no path of edges to the ground.
size_t cholesky_solve(std::vector<std::vector<double>>& N, std::vector<double>& r) {
    const size_t n = r.size();
    for (size_t j = 0; j < n; j++) {
        double d = N[j][j];
        for (size_t k = 0; k < j; k++) {
            d -= N[j][k] * N[j][k];
        }
        if (d <= 1e-9) {
            return j;
        }
        N[j][j] = std::sqrt(d);
        for (size_t i = j + 1; i < n; i++) {
            double v = N[i][j];
            for (size_t k = 0; k < j; k++) {
                v -= N[i][k] * N[j][k];
            }
            N[i][j] = v / N[j][j];
        }
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < i; k++) {
            r[i] -= N[i][k] * r[k];
        }
        r[i] /= N[i][i];
    }
    for (size_t i = n; i-- > 0;) {
        for (size_t k = i + 1; k < n; k++) {
            r[i] -= N[k][i] * r[k];
        }
        r[i] /= N[i][i];
    }
    return n;
}

// Every tile with a RISC: the whole Tensix grid (the dispatch cores included), every eth core, every DRAM core Metal
// may place a kernel on that is no DRAM view's endpoint. Where a kernel scratches, and zeroes on exit: a compute core
// in the user L1 the allocator hands out (nothing is allocated yet); a dispatch core in the profiler ring space past
// its control vector (drained by nobody); an eth or DRAM core at the bottom of its unreserved region, below the
// pusher's and the link ends' carves and under the relay's staging, which the relay fills before it sends. A ring the
// profiler decodes is not usable even zeroed: the pusher's and the relay's first frames broke with the scratch there.
std::vector<Node> enumerate_nodes(IDevice* device, ContextId ctx) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto& soc = cluster.get_soc_desc(chip);
    std::vector<Node> nodes;
    const auto add =
        [&](CoreType type, HalProgrammableCoreType core, const CoreCoord& logical, uint32_t scratch, uint64_t host) {
            nodes.push_back(Node{
                .type = type,
                .core = core,
                .logical = logical,
                .virt = cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, type),
                .phys = cluster.get_physical_coordinate_from_logical_coordinates(chip, logical, type, /*no_warn=*/true),
                .scratch = scratch,
                .host_scratch = host});
        };
    const auto ring_space = [&](HalProgrammableCoreType t) {
        TT_FATAL(
            hal.get_dev_size(t, HalL1MemAddrType::PROFILER) >=
                kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE + sizeof(kernel_profiler::TileNetScratch),
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
            add(CoreType::WORKER, HalProgrammableCoreType::TENSIX, CoreCoord{x, y}, scratch, scratch);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::IDLE_ETH)) {
        const uint32_t scratch = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
        for (const CoreCoord& l : sorted_yx(device->get_inactive_ethernet_cores())) {
            add(CoreType::ETH, HalProgrammableCoreType::IDLE_ETH, l, scratch, scratch);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::ACTIVE_ETH)) {
        const uint32_t scratch = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
        for (const CoreCoord& l : sorted_yx(device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false))) {
            add(CoreType::ETH, HalProgrammableCoreType::ACTIVE_ETH, l, scratch, scratch);
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
            // Firmware holds a DRAM view endpoint's NIU in NOC2AXI: a read it issues there never goes out, and one
            // arriving there goes to GDDR, so the tile has no mirrored pair to take part in.
            if (soc.get_dram_endpoint_noc_mask(soc.get_physical_dram_core_from_logical(logical[i])) != 0) {
                continue;
            }
            add(CoreType::DRAM, HalProgrammableCoreType::DRAM, logical[i], scratch, host);
            nodes.back().phys = noc0[i];
        }
    }
    return nodes;
}

const char* kind_name(CoreType t) { return t == CoreType::WORKER ? "Tensix" : t == CoreType::DRAM ? "DRAM" : "eth"; }

// The kernel on every node, one program per kind of core, then the go to one node at a time so nothing else is on the
// NoC while it reads. Each node reads every node in its row and column. Every kernel is released at the end, and on
// any failure, so no tile is left waiting on its go word. A source's readings are contiguous, in its partner table's
// order.
std::vector<Reading> read_network(IDevice* device, ContextId ctx, const std::vector<Node>& nodes) {
    auto& cluster = MetalContext::instance(ctx).get_cluster();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const uint32_t nonce = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 0x10u;
    // One idle eth tile reads every eth tile in its row on the other NoC too: around the ring the two NoCs' hops
    // cancel, and one initiator's own latency is common to every target.
    const auto q = std::find_if(
        nodes.begin(), nodes.end(), [](const Node& n) { return n.core == HalProgrammableCoreType::IDLE_ETH; });
    const auto qi = static_cast<uint32_t>(q - nodes.begin());
    std::vector<Reading> readings;
    std::vector<std::vector<uint32_t>> args(nodes.size());
    std::vector<size_t> first(nodes.size());
    for (uint32_t s = 0; s < nodes.size(); s++) {
        args[s] = {nodes[s].scratch, kReps, nonce, 0};
        first[s] = readings.size();
        for (uint32_t t = 0; t < nodes.size(); t++) {
            if (!aligned(nodes[s].phys, nodes[t].phys)) {
                continue;
            }
            const bool up = upward(nodes[s].phys, nodes[t].phys);
            args[s].push_back((up ? 0u : 1u << 31) | packed_xy(nodes[t].virt));
            readings.push_back(Reading{s, t, up ? 0u : 1u});
        }
        if (s != qi) {
            continue;
        }
        const size_t direct_end = readings.size();
        for (size_t i = first[s]; i < direct_end; i++) {
            const uint32_t t = readings[i].t;
            if (nodes[t].type != CoreType::ETH) {
                continue;
            }
            const bool up = readings[i].noc == 0;
            args[s].push_back((up ? 1u << 31 : 0u) | packed_xy(nodes[t].virt));
            readings.push_back(Reading{s, t, up ? 1u : 0u, true, static_cast<uint32_t>(i)});
        }
    }
    for (uint32_t s = 0; s < nodes.size(); s++) {
        args[s][3] = static_cast<uint32_t>(args[s].size() - 4);
        TT_FATAL(
            args[s][3] <= kernel_profiler::kTileNetMaxPartners,
            "streaming profiler: tile ({},{}) has {} row and column partners, the table holds {}",
            nodes[s].logical.x,
            nodes[s].logical.y,
            args[s][3],
            kernel_profiler::kTileNetMaxPartners);
    }
    std::map<HalProgrammableCoreType, std::set<CoreRange>> ranges;
    for (const Node& n : nodes) {
        ranges[n.core].insert(CoreRange(n.logical, n.logical));
    }
    const char* src = "tt_metal/tools/profiler/sync/tile_sync.cpp";
    const auto create_kernel = [&](Program& p, HalProgrammableCoreType core, const CoreRangeSet& cores) {
        switch (core) {
            case HalProgrammableCoreType::TENSIX:
                return CreateKernel(
                    p,
                    src,
                    cores,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            case HalProgrammableCoreType::IDLE_ETH:
                return CreateKernel(p, src, cores, EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::RISCV_0_default});
            case HalProgrammableCoreType::ACTIVE_ETH:
                return CreateKernel(p, src, cores, EthernetConfig{.noc = NOC::RISCV_0_default});
            case HalProgrammableCoreType::DRAM: return CreateKernel(p, src, cores, DramConfig{.noc = NOC::NOC_0});
            case HalProgrammableCoreType::DISPATCH:
            case HalProgrammableCoreType::COUNT: break;
        }
        TT_THROW("Unreachable");
    };
    std::vector<Program> programs;
    std::vector<KernelHandle> kids;
    std::map<HalProgrammableCoreType, size_t> program_of;
    for (const auto& [core, cores] : ranges) {
        program_of[core] = programs.size();
        programs.push_back(CreateProgram());
        kids.push_back(create_kernel(programs.back(), core, CoreRangeSet(cores)));
    }
    const auto table_of = [&](uint32_t s) {
        return nodes[s].host_scratch + offsetof(kernel_profiler::TileNetScratch, table);
    };
    // Fast dispatch's go signal reaches every core of the grid, and a host-launched kernel leaves its launch slot
    // valid on exit, so the slots go back to the firmware's initial message once the kernels have exited.
    const auto& hal = MetalContext::instance(ctx).hal();
    const auto host_addr = [&](HalProgrammableCoreType t, HalL1MemAddrType a) {
        return t == HalProgrammableCoreType::DRAM ? hal.get_dev_noc_addr(t, a) : hal.get_dev_addr(t, a);
    };
    const auto invalidate_launch = [&](const Node& n) {
        auto msg = hal.get_dev_msgs_factory(n.core).create<dev_msgs::launch_msg_t>();
        cluster.write_core(
            msg.data(),
            static_cast<uint32_t>(msg.size()),
            tt_cxy_pair(chip, n.virt),
            host_addr(n.core, HalL1MemAddrType::LAUNCH));
    };
    // The firmware seeds its profiler ring position from the control vector at the first launch of its session,
    // which this is; a tail left by the previous session would put it thousands of words ahead of the relay's head.
    const auto zero_control = [&](const Node& n) {
        const std::vector<uint8_t> zero(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
        cluster.write_core(
            zero.data(),
            static_cast<uint32_t>(zero.size()),
            tt_cxy_pair(chip, n.virt),
            host_addr(n.core, HalL1MemAddrType::PROFILER));
    };
    const uint32_t zero[2] = {0, 0};
    for (uint32_t s = 0; s < nodes.size(); s++) {
        const size_t pi = program_of.at(nodes[s].core);
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
        for (uint32_t s = 0; s < nodes.size(); s++) {
            const uint32_t go = kernel_profiler::kTileNetGoExit;
            cluster.write_core(&go, sizeof(go), tt_cxy_pair(chip, nodes[s].virt), table_of(s));
        }
        for (Program& p : programs) {
            detail::WaitProgramDone(device, p, false);
        }
        for (const Node& n : nodes) {
            invalidate_launch(n);
        }
    };
    try {
        for (uint32_t s = 0; s < nodes.size(); s++) {
            const uint32_t n = args[s][3];
            const tt_cxy_pair core(chip, nodes[s].virt);
            kernel_profiler::TileNetTable t{};
            const auto table_bytes = static_cast<uint32_t>(
                offsetof(kernel_profiler::TileNetTable, partner) + n * sizeof(kernel_profiler::TileNetPartner));
            const auto await = [&](uint32_t ready, const char* what) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                for (;;) {
                    cluster.read_core(&t, table_bytes, core, table_of(s));
                    if (t.ready == ready) {
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
                const kernel_profiler::TileNetPartner& p = t.partner[m];
                Reading& r = readings[first[s] + m];
                r.median2 = p.median2;
                r.rtt = p.rtt;
                r.coarse = static_cast<int64_t>((uint64_t{p.coarse_hi} << 32) | p.coarse_lo);
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
    const std::vector<Reading> readings = read_network(device, ctx, nodes);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const auto is_active_eth = [&](uint32_t i) { return nodes[i].core == HalProgrammableCoreType::ACTIVE_ETH; };

    // A pair: the lower node's NoC 0 reading of the higher and the higher's NoC 1 reading of the lower. Half their
    // difference is the offset; their round trips must agree.
    std::map<std::pair<uint32_t, uint32_t>, const Reading*> by_pair;
    for (const Reading& r : readings) {
        if (!r.both) {
            by_pair[{r.s, r.t}] = &r;
        }
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
        if (r.both || r.noc != 0 || is_active_eth(r.s) || is_active_eth(r.t)) {
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
    // x[i] = tile i's wall tick minus tile 0's. The unknowns are every tile but tile 0 and the active eth tiles, and
    // the normal equations are the pair graph's Laplacian with those rows and columns removed.
    const size_t n = nodes.size();
    std::vector<int64_t> col(n, -1);
    std::vector<uint32_t> node_of;
    for (uint32_t i = 1; i < n; i++) {
        if (!is_active_eth(i)) {
            col[i] = static_cast<int64_t>(node_of.size());
            node_of.push_back(i);
        }
    }
    std::vector<std::vector<double>> N(node_of.size(), std::vector<double>(node_of.size(), 0.0));
    std::vector<double> rhs(node_of.size(), 0.0);
    for (const Pair& p : pairs) {
        const int64_t hi = col[p.hi], lo = col[p.lo];
        if (hi >= 0) {
            N[hi][hi] += 1.0;
            rhs[hi] += p.offset;
        }
        if (lo >= 0) {
            N[lo][lo] += 1.0;
            rhs[lo] -= p.offset;
        }
        if (hi >= 0 && lo >= 0) {
            N[hi][lo] -= 1.0;
            N[lo][hi] -= 1.0;
        }
    }
    const size_t bad = cholesky_solve(N, rhs);
    TT_FATAL(
        bad == rhs.size(),
        "streaming profiler: device {} {} tile ({},{}) has no chain of mirrored pairs to the first Tensix tile",
        chip,
        kind_name(nodes[node_of[bad]].type),
        nodes[node_of[bad]].logical.x,
        nodes[node_of[bad]].logical.y);
    std::vector<double> x(n, 0.0);
    for (size_t c = 0; c < node_of.size(); c++) {
        x[node_of[c]] = rhs[c];
    }

    // est[t]: the both-NoC initiator's two readings of eth tile t averaged, t's offset from it plus a constant common
    // to every target. Active eth tiles take the idle eth targets' mean placement through it.
    std::vector<std::optional<double>> est(n);
    for (const Reading& r : readings) {
        if (r.both) {
            est[r.t] = static_cast<double>(readings[r.twin].whole2() + r.whole2()) / 4.0;
        }
    }
    double shift = 0.0;
    size_t refs = 0;
    for (uint32_t t = 1; t < n; t++) {
        if (est[t] && !is_active_eth(t)) {
            shift += x[t] - *est[t];
            refs++;
        }
    }
    for (uint32_t t = 0; t < n; t++) {
        if (!is_active_eth(t)) {
            continue;
        }
        TT_FATAL(
            refs != 0 && est[t],
            "streaming profiler: device {} active eth tile ({},{}) has no both-NoC reading to place it from",
            chip,
            nodes[t].logical.x,
            nodes[t].logical.y);
        x[t] = shift / static_cast<double>(refs) + *est[t];
    }

    double closure_ss = 0.0, closure_worst = 0.0, rtt_ss = 0.0, rtt_worst = 0.0;
    for (const Pair& p : pairs) {
        const double r = p.offset - (x[p.hi] - x[p.lo]);
        closure_ss += r * r;
        closure_worst = std::max(closure_worst, std::fabs(r));
        rtt_ss += p.rtt_diff * p.rtt_diff;
        rtt_worst = std::max(rtt_worst, std::fabs(p.rtt_diff));
    }
    TileClocks clocks;
    for (uint32_t i = 0; i < n; i++) {
        clocks.tiles.push_back(TileClock{nodes[i].type, nodes[i].logical, std::llround(x[i])});
    }
    const double pair_count = static_cast<double>(std::max<size_t>(pairs.size(), 1));
    log_info(
        tt::LogMetal,
        "[streaming profiler] Device {}: tile clocks from {} mirrored pairs x 2 x {} reads in {:.0f} ms; closures "
        "{:.2f} ticks rms, {:.2f} worst; mirrored round trips disagree {:.2f} ticks rms, {:.1f} worst",
        chip,
        pairs.size(),
        kReps,
        ms,
        std::sqrt(closure_ss / pair_count),
        closure_worst,
        std::sqrt(rtt_ss / pair_count),
        rtt_worst);
    service().set_tile_clocks(chip, std::move(clocks));
}

}  // namespace

void measure_tile_clocks(IDevice* device, ContextId ctx) {
    const auto& mc = MetalContext::instance(ctx);
    if (!mc.rtoptions().get_streaming_profiler_enabled() || mc.hal().get_arch() != tt::ARCH::BLACKHOLE ||
        service().tile_clocks(static_cast<uint32_t>(device->id())) != nullptr) {
        return;
    }
    measure_chip(device, ctx);
}

}  // namespace tt::tt_metal::streaming_profiler
