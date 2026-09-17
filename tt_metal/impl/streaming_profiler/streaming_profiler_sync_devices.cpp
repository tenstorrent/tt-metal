// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/streaming_profiler/streaming_profiler_d2d_sync.hpp"
#include "impl/streaming_profiler/streaming_profiler_link_sync.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {
constexpr uint32_t kLinkSyncChannels = 1;
constexpr uint32_t kLinkSyncSamples = 240;
constexpr uint32_t kLinkSyncSampleSize = 16;

// One idle-eth core's reading of one tile: the core's wall tick minus the tile's, each NoC's read round trip, and
// the NoC 0 minus NoC 1 reading in half ticks.
struct TileReading {
    int64_t value = 0;
    uint32_t rtt0 = 0, rtt1 = 0;
    int32_t ddiff = 0;
};

// Solves the normal equations N x = r by Gaussian elimination with partial pivoting; a pivot of zero leaves that
// unknown at 0 (an unobserved tile).
std::vector<double> solve_normal(std::vector<std::vector<double>> N, std::vector<double> r) {
    const size_t n = r.size();
    std::vector<double> x(n, 0.0);
    std::vector<size_t> col(n);
    for (size_t i = 0; i < n; i++) {
        col[i] = i;
    }
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

}  // namespace

KernelHandle create_pusher_kernel(Program& program, const EthL1& l1, const CoreCoords& core, bool measure_only) {
    return CreateKernel(
        program,
        "tt_metal/tools/profiler/sync/eth_clock_pusher.cpp",
        core.logical,
        EthernetConfig{
            .eth_mode = Eth::IDLE,
            .noc = NOC::RISCV_0_default,
            .processor = DataMovementProcessor::RISCV_0,
            .compile_args = {
                kEthPointUs * 50u,
                l1.cfg,
                l1.stage,
                l1.ctrl,
                packed_xy(core.virt),
                l1.scratch,
                l1.table,
                measure_only ? 1u : 0u,
                l1.ring}});
}

SyncDevices::SyncDevices(
    ContextId context_id, const EthL1& eth_l1, uint32_t aeth_unreserved, uint32_t aeth_unres_size) :
    context_id_(context_id), eth_l1_(eth_l1), aeth_unreserved_(aeth_unreserved), aeth_unres_size_(aeth_unres_size) {}

SyncDevices::~SyncDevices() = default;

uint32_t SyncDevices::add_device(Device d) {
    devices_.push_back(DeviceState{.d = std::move(d)});
    return static_cast<uint32_t>(devices_.size() - 1);
}

void SyncDevices::truncate(uint32_t n) {
    if (devices_.size() > n) {
        devices_.resize(n);
    }
}

void SyncDevices::start_probe() {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    for (uint32_t di = 0; di < devices_.size(); di++) {
        if (!devices_[di].d.eth.empty()) {
            host_probe_ = std::make_shared<HostProbe>(
                cluster,
                devices_[di].d.chip_id,
                service().sync().map(),
                MetalContext::instance(context_id_).rtoptions().get_streaming_profiler_d2d_csv_path());
            root_dev_ = di;
            break;
        }
    }
}

void SyncDevices::stop(tt::Cluster& cluster) {
    if (host_probe_) {
        host_probe_->stop();
    }
    stop_links(cluster);
}

// Every idle eth core reads every Tensix tile and every other idle eth tile, one core at a time so nothing else is
// on the NoC, and the tiles' offsets are solved together: each reading says tile - source, the pusher fixes the
// origin. Three numbers per chip come out of it: how far apart the tiles' clocks are, how well the sources agree on
// each tile (the placement's likely error), and what the round trips bound the error to with no symmetry assumed:
// on the torus a read's request and response together cover whole rings at a fixed latency per hop
// (RoutingPaths.md, README.md of the NoC ISA docs), so a round trip minus its rings is the two ends' own handling,
// and the register sample lies somewhere inside the far end's share of it.
std::vector<double> SyncDevices::solve_tiles(uint32_t di, const char* when) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const Device& d = devices_[di].d;
    const uint32_t nt = static_cast<uint32_t>(d.tensix.size());
    const uint32_t ns = static_cast<uint32_t>(d.eth.size());
    const uint32_t nh = ns - 1;
    TT_FATAL(
        nt + nh <= kEthTableMaxTiles,
        "streaming profiler: device {} has {} tiles to read, the tile table holds {}",
        d.chip_id,
        nt + nh,
        kEthTableMaxTiles);
    // One core's table over `count` tiles; false until written.
    auto read_table = [&](const CoreCoord& virt, uint32_t count, std::vector<TileReading>& out, TileReading& loop) {
        std::vector<uint32_t> t(kernel_profiler::eth_tile_out_word(count, count), 0);
        cluster.read_core(
            t.data(), static_cast<uint32_t>(t.size() * sizeof(uint32_t)), tt_cxy_pair(d.chip_id, virt), eth_l1_.table);
        if (t[kernel_profiler::ETH_TILE_READY] != (kernel_profiler::kEthTileReadyWord | count)) {
            return false;
        }
        loop.rtt0 = t[kernel_profiler::ETH_TILE_LOOP_RTT];
        loop.rtt1 = t[kernel_profiler::ETH_TILE_LOOP_RTT1];
        loop.ddiff = static_cast<int32_t>(t[kernel_profiler::ETH_TILE_LOOP_DDIFF]);
        out.resize(count);
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t w = kernel_profiler::eth_tile_out_word(count, i);
            out[i].value = static_cast<int64_t>((static_cast<uint64_t>(t[w + 1]) << 32) | t[w]);
            out[i].rtt0 = t[w + 2];
            out[i].rtt1 = t[w + 3];
            out[i].ddiff = static_cast<int32_t>(t[w + 4]);
        }
        return true;
    };
    // Sources: the pusher, then the helpers; a source's tile list is the Tensix tiles, then the other sources in
    // source order. Unknowns: the Tensix tiles, then the helpers; the pusher is the origin.
    std::vector<CoreCoord> src_logical, src_virt, src_phys;
    for (const CoreCoords& h : d.eth) {
        src_logical.push_back(h.logical);
        src_virt.push_back(h.virt);
        src_phys.push_back(h.phys);
    }
    std::vector<std::vector<TileReading>> readings(ns);
    std::vector<TileReading> loops(ns);
    for (uint32_t s = 0; s < ns; s++) {
        std::vector<uint32_t> table(kernel_profiler::ETH_TILE_XY_0, 0);
        for (const CoreCoords& c : d.tensix) {
            table.push_back(packed_xy(c.virt));
        }
        for (uint32_t o = 0; o < ns; o++) {
            if (o != s) {
                table.push_back(packed_xy(src_virt[o]));
            }
        }
        table[kernel_profiler::ETH_TILE_N] = nt + nh;
        cluster.write_core(
            table.data(),
            static_cast<uint32_t>(table.size() * sizeof(uint32_t)),
            tt_cxy_pair(d.chip_id, src_virt[s]),
            eth_l1_.table);
        Program p = CreateProgram();
        const KernelHandle kid = create_pusher_kernel(p, eth_l1_, d.eth[s], /*measure_only=*/true);
        SetRuntimeArgs(p, kid, src_logical[s], std::vector<uint32_t>{0});
        detail::CompileProgram(d.device, p, /*force_slow_dispatch=*/true);
        detail::WriteRuntimeArgsToDevice(d.device, p, /*force_slow_dispatch=*/true);
        detail::LaunchProgram(d.device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!read_table(src_virt[s], nt + nh, readings[s], loops[s])) {
            TT_FATAL(
                std::chrono::steady_clock::now() < deadline,
                "streaming profiler: device {} idle eth ({},{}) tile table not written within 2 s",
                d.chip_id,
                src_logical[s].x,
                src_logical[s].y);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        detail::WaitProgramDone(d.device, p, false);
    }

    // Observations: for source s and its i-th listed tile, value = source wall - tile wall = x_tile - x_source
    // with x = pusher wall - tile wall (x_pusher = 0). Unknown index: Tensix i -> i, helper h -> nt + h.
    struct Obs {
        int32_t s, t;  // unknown indices; -1 is the pusher
        uint32_t src;  // source order
        TileReading r;
        CoreCoord from, to;  // raw NoC 0 coordinates
    };
    std::vector<Obs> obs;
    const auto unknown_of_source = [&](uint32_t s) { return s == 0 ? -1 : static_cast<int32_t>(nt + s - 1); };
    for (uint32_t s = 0; s < ns; s++) {
        uint32_t listed = 0;
        for (uint32_t i = 0; i < nt; i++, listed++) {
            obs.push_back(Obs{
                unknown_of_source(s), static_cast<int32_t>(i), s, readings[s][listed], src_phys[s], d.tensix[i].phys});
        }
        for (uint32_t o = 0; o < ns; o++) {
            if (o == s) {
                continue;
            }
            obs.push_back(
                Obs{unknown_of_source(s), unknown_of_source(o), s, readings[s][listed++], src_phys[s], src_phys[o]});
        }
    }
    // A read within the source's column or row covers one ring, the rest two; the one-ring reads sit a fixed few
    // ticks off the two-ring ones, so each of those two paths gets an unknown of its own, found from the tiles
    // both kinds of source read, instead of pulling on the tiles' offsets.
    const size_t nu = nt + nh + 2;
    const size_t kCol = nt + nh, kRow = nt + nh + 1;
    const auto path_of = [&](const Obs& o) -> int32_t {
        if (o.from.x == o.to.x) {
            return static_cast<int32_t>(kCol);
        }
        return o.from.y == o.to.y ? static_cast<int32_t>(kRow) : -1;
    };
    std::vector<std::vector<double>> N(nu, std::vector<double>(nu, 0.0));
    std::vector<double> rhs(nu, 0.0);
    for (const Obs& o : obs) {
        const double v = static_cast<double>(o.r.value);
        const int32_t cols[3] = {o.t, o.s, path_of(o)};
        const double coef[3] = {1.0, -1.0, 1.0};
        for (int i = 0; i < 3; i++) {
            if (cols[i] < 0) {
                continue;
            }
            rhs[cols[i]] += coef[i] * v;
            for (int j = 0; j < 3; j++) {
                if (cols[j] >= 0) {
                    N[cols[i]][cols[j]] += coef[i] * coef[j];
                }
            }
        }
    }
    const std::vector<double> x = solve_normal(N, rhs);
    const auto x_of = [&](int32_t u) { return u < 0 ? 0.0 : x[static_cast<size_t>(u)]; };
    const auto fit_of = [&](const Obs& o) {
        const int32_t pth = path_of(o);
        return x_of(o.t) - x_of(o.s) + (pth >= 0 ? x[static_cast<size_t>(pth)] : 0.0);
    };
    if (const std::string& csv = MetalContext::instance(context_id_).rtoptions().get_streaming_profiler_d2d_csv_path();
        !csv.empty()) {
        const std::string path = fmt::format("{}.tiles.{}.chip{}.csv", csv, when, d.chip_id);
        if (std::FILE* f = std::fopen(path.c_str(), "w"); f != nullptr) {
            std::fprintf(f, "src,src_x,src_y,tile,tile_x,tile_y,value,solved,residual,rtt0,rtt1,ddiff\n");
            for (const Obs& o : obs) {
                std::fprintf(
                    f,
                    "%u,%u,%u,%d,%u,%u,%lld,%.2f,%.2f,%u,%u,%d\n",
                    o.src,
                    static_cast<uint32_t>(o.from.x),
                    static_cast<uint32_t>(o.from.y),
                    o.t,
                    static_cast<uint32_t>(o.to.x),
                    static_cast<uint32_t>(o.to.y),
                    static_cast<long long>(o.r.value),
                    fit_of(o),
                    static_cast<double>(o.r.value) - fit_of(o),
                    o.r.rtt0,
                    o.r.rtt1,
                    o.r.ddiff);
            }
            std::fclose(f);
        }
    }
    const double ns_per_tick = 1.0 / d.frequency_ghz;

    const auto [xlo, xhi] = std::minmax_element(x.begin(), x.begin() + nt);
    double ss = 0.0, worst = 0.0;
    for (const Obs& o : obs) {
        const double res = static_cast<double>(o.r.value) - fit_of(o);
        ss += res * res;
        worst = std::max(worst, std::fabs(res));
    }
    const double rms = obs.empty() ? 0.0 : std::sqrt(ss / static_cast<double>(obs.size()));
    // A read's round trip is its rings (x ring, y ring, both, or neither for the loopback read),
    // each a fixed transit, plus the two ends' handling. The ring transits come from the round trips by class;
    // what remains of a read is its ends, and the sample lies inside the far end's part, so a placement is off by
    // at most half of (this read's ends + the source's loopback ends), whatever the split.
    const auto& soc = cluster.get_soc_desc(d.chip_id);
    const uint32_t W = static_cast<uint32_t>(soc.grid_size.x), H = static_cast<uint32_t>(soc.grid_size.y);
    double sum_row = 0.0, sum_col = 0.0, sum_gen = 0.0;
    uint32_t n_row = 0, n_col = 0, n_gen = 0;
    for (const Obs& o : obs) {
        const double ends = static_cast<double>(o.r.rtt0) - static_cast<double>(loops[o.src].rtt0);
        const bool same_col = o.from.x == o.to.x, same_row = o.from.y == o.to.y;
        if (same_col) {
            sum_col += ends;
            n_col++;
        } else if (same_row) {
            sum_row += ends;
            n_row++;
        } else {
            sum_gen += ends;
            n_gen++;
        }
    }
    // Ring transits from the class means, over the loopback: y ring from same-column reads, x ring from
    // same-row reads, and the general class checks their sum (its excess is the turns).
    const double y_ring = n_col ? sum_col / n_col : 0.0;
    const double x_ring = n_row ? sum_row / n_row : 0.0;
    const double turns = n_gen ? sum_gen / n_gen - x_ring - y_ring : 0.0;
    double bound = 0.0, transit_rms = 0.0;
    for (const Obs& o : obs) {
        const bool same_col = o.from.x == o.to.x, same_row = o.from.y == o.to.y;
        const double transit =
            (same_col ? 0.0 : x_ring) + (same_row ? 0.0 : y_ring) + (same_col || same_row ? 0.0 : turns);
        const double ends = static_cast<double>(o.r.rtt0) - transit;
        const double resid = ends - static_cast<double>(loops[o.src].rtt0);
        transit_rms += resid * resid;
        bound = std::max(bound, 0.5 * (ends + static_cast<double>(loops[o.src].rtt0)));
    }
    transit_rms = obs.empty() ? 0.0 : std::sqrt(transit_rms / static_cast<double>(obs.size()));
    // The two NoCs' split of each reading against the routes: on NoC 0 a request runs right then down, its
    // response on round the same way, so the request is ahead of the midpoint by half the hop imbalance; NoC 1
    // mirrors it. Per hop and per wrap link fitted, the residual is what the routes do not explain.
    {
        std::vector<std::vector<double>> M(4, std::vector<double>(4, 0.0));
        std::vector<double> mr(4, 0.0);
        std::vector<std::array<double, 4>> rows;
        std::vector<double> vals;
        for (const Obs& o : obs) {
            const int64_t dxr = (static_cast<int64_t>(o.to.x) - static_cast<int64_t>(o.from.x) + W) % W;
            const int64_t dyr = (static_cast<int64_t>(o.to.y) - static_cast<int64_t>(o.from.y) + H) % H;
            const std::array<double, 4> a = {
                dxr ? static_cast<double>(2 * dxr - static_cast<int64_t>(W)) : 0.0,
                dyr ? static_cast<double>(2 * dyr - static_cast<int64_t>(H)) : 0.0,
                dxr ? (o.to.x < o.from.x ? 1.0 : -1.0) : 0.0,
                dyr ? (o.to.y < o.from.y ? 1.0 : -1.0) : 0.0};
            const double v = 0.5 * static_cast<double>(o.r.ddiff);  // NoC 0 minus NoC 1 midpoint error, ticks
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    M[i][j] += a[i] * a[j];
                }
                mr[i] += a[i] * v;
            }
            rows.push_back(a);
            vals.push_back(v);
        }
        const std::vector<double> fit = solve_normal(M, mr);
        double rss = 0.0, rworst = 0.0;
        for (size_t k = 0; k < rows.size(); k++) {
            double pred = 0.0;
            for (size_t i = 0; i < 4; i++) {
                pred += rows[k][i] * fit[i];
            }
            const double res = vals[k] - pred;
            rss += res * res;
            rworst = std::max(rworst, std::fabs(res));
        }
        // The split only sees the two NoCs' per-hop costs summed; their difference is the two round trips'
        // difference, both covering the same rings.
        double d_sum = 0.0;
        int32_t d_worst = 0;
        for (const Obs& o : obs) {
            const int32_t d = static_cast<int32_t>(o.r.rtt1) - static_cast<int32_t>(o.r.rtt0);
            d_sum += d;
            d_worst = std::abs(d) > std::abs(d_worst) ? d : d_worst;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] Device {}: tile reads on the two NoCs split as the torus routes say: {:.2f} ticks "
            "per x hop, {:.2f} per y hop, wrap links {:+.1f} x {:+.1f} y; unexplained {:.1f} ticks rms, {:.1f} worst; "
            "NoC 1 round trips against NoC 0: {:+.2f} ticks mean, {:+d} worst",
            d.chip_id,
            fit[0],
            fit[1],
            fit[2],
            fit[3],
            rows.empty() ? 0.0 : std::sqrt(rss / static_cast<double>(rows.size())),
            rworst,
            obs.empty() ? 0.0 : d_sum / static_cast<double>(obs.size()),
            d_worst);
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] Device {} at {}: {} tiles' wall clocks span {:.0f} ticks ({:.1f} ns), solved from {} "
        "idle "
        "eth sources over {} reads; the sources disagree by {:.2f} ticks rms, {:.1f} worst ({:.2f} ns rms); one-ring "
        "reads sit {:+.1f} (column) {:+.1f} (row) ticks off two-ring ones; a placement is off by at most {:.1f} ticks "
        "({:.1f} ns) from the reads' own ends (rings: x {:.0f}, y {:.0f}, turns {:.0f} ticks; ends model residual "
        "{:.1f} ticks rms)",
        d.chip_id,
        when,
        nt,
        *xhi - *xlo,
        (*xhi - *xlo) * ns_per_tick,
        ns,
        obs.size(),
        rms,
        worst,
        rms * ns_per_tick,
        x[kCol],
        x[kRow],
        bound,
        bound * ns_per_tick,
        x_ring,
        y_ring,
        turns,
        transit_rms);
    return x;
}

void SyncDevices::measure_tiles(uint32_t di, CaptureContext::Device& cap) {
    DeviceState& st = devices_[di];
    const std::vector<double> x = solve_tiles(di, "arm");
    std::vector<int64_t>& off = cap.tile_offset;
    for (size_t i = 0; i < st.d.tensix.size(); i++) {
        off[i] = std::llround(x[i]);
    }
    st.tile_solution = x;
    st.tile_solved_at = std::chrono::steady_clock::now();
}

// The same measurement once the capture's kernels are all stopped and the NoC is quiet again, against the table the
// capture ran with. Every tile's clock ticks on the one AICLK, so the skew between tiles must not have moved through
// the workload's clock steps, and their offset to the eth clock must not have moved either; either would mean the
// table placed the records after the move wrong.
void SyncDevices::recheck_tiles(uint32_t di) {
    const DeviceState& st = devices_[di];
    const Device& d = st.d;
    const size_t nt = d.tensix.size();
    if (st.tile_solution.size() < nt) {
        return;
    }
    const std::vector<double> x = solve_tiles(di, "capture end");
    double common = 0.0;
    for (size_t i = 0; i < nt; i++) {
        common += x[i] - st.tile_solution[i];
    }
    common /= static_cast<double>(nt);
    double ss = 0.0, worst = 0.0;
    for (size_t i = 0; i < nt; i++) {
        const double d = (x[i] - st.tile_solution[i]) - common;
        ss += d * d;
        worst = std::max(worst, std::fabs(d));
    }
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - st.tile_solved_at).count();
    const bool moved = worst > 1.0 || std::fabs(common) > 1.0;
    const std::string line = fmt::format(
        "[streaming profiler] Device {}: tile clocks re-read {:.1f} s after arm: tile-to-tile skew moved {:.2f} ticks "
        "rms, {:.1f} worst; the eth-minus-tensix offset moved {:+.0f} ticks in common (0 while every tile's clock "
        "runs with the eth's)",
        d.chip_id,
        elapsed_s,
        std::sqrt(ss / static_cast<double>(nt)),
        worst,
        common);
    if (moved) {
        log_warning(tt::LogMetal, "{}: the table the capture ran with did not hold", line);
    } else {
        log_info(tt::LogMetal, "{}", line);
    }
}

void SyncDevices::plan_links() {
    auto& mc = MetalContext::instance(context_id_);
    fabric_link_sync_ = mc.get_fabric_config() != tt_fabric::FabricConfig::DISABLED;
    if (!link_sync::enabled()) {
        log_info(tt::LogMetal, "[streaming profiler] link sync left out (TT_METAL_STREAMING_PROFILER_LINK_SYNC=0)");
        return;
    }
    auto& cluster = mc.get_cluster();
    for (size_t a = 0; a < devices_.size(); a++) {
        const uint32_t chip_a = devices_[a].d.chip_id;
        for (size_t b = a + 1; b < devices_.size(); b++) {
            const uint32_t chip_b = devices_[b].d.chip_id;
            for (const link_sync::Link& link : link_sync::links_between(cluster, chip_a, chip_b)) {
                const bool flip = link.chip_a != chip_a;  // the lower chip sends
                links_.push_back(CaptureContext::Link{
                    .dev_a = static_cast<uint32_t>(flip ? b : a),
                    .dev_b = static_cast<uint32_t>(flip ? a : b),
                    .chip_a = link.chip_a,
                    .chip_b = link.chip_b,
                    .eth_a = link.eth_a,
                    .eth_b = link.eth_b});
            }
        }
    }
    if (links_.empty() && devices_.size() > 1) {
        log_warning(tt::LogMetal, "[streaming profiler] link sync: no eth connection between the local devices");
    }
}

// Launching this during boot() instead deadlocked: the FIFO filled with no reader, the pusher parked in
// socket_reserve_pages, and the armed sync kernels wedged an eth core (a board reset).
void SyncDevices::launch_links() {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    // The stop/done words sit at the top of the active eth core's UNRESERVED region, clear of the sync kernel's eth
    // channels (which start at its base) and its profiler ring, past the frame slots of a router-hosted end, which
    // keeps its diagnostics at the same place (link_sync::kL1Bytes, kCtlOffset).
    const uint32_t stop_addr = aeth_unreserved_ + aeth_unres_size_ - link_sync::kL1Bytes + link_sync::kCtlOffset;
    for (const CaptureContext::Link& L : links_) {
        IDevice* dev_a = devices_[L.dev_a].d.device;
        IDevice* dev_b = devices_[L.dev_b].d.device;
        const CoreCoord virt_a =
            cluster.get_virtual_coordinate_from_logical_coordinates(L.chip_a, L.eth_a, CoreType::ETH);
        const CoreCoord virt_b =
            cluster.get_virtual_coordinate_from_logical_coordinates(L.chip_b, L.eth_b, CoreType::ETH);
        if (fabric_link_sync_) {
            // The routers on this link run the two ends (fabric_erisc_router.cpp, LINK_SYNC_ROLE); nothing to
            // launch: the sender waits for the run word, and their diagnostics are read where the resident kernels
            // leave theirs.
            link_syncs_.push_back(ResidentSync{
                .dev_a = dev_a,
                .dev_b = dev_b,
                .virt_a = virt_a,
                .virt_b = virt_b,
                .chip_a = L.chip_a,
                .chip_b = L.chip_b,
                .stop_a = stop_addr,
                .stop_b = stop_addr});
            log_info(
                tt::LogMetal,
                "[streaming profiler] link sync {} eth({},{}) -> {} eth({},{}): in the fabric routers at {} Hz",
                L.chip_a,
                L.eth_a.x,
                L.eth_a.y,
                L.chip_b,
                L.eth_b.x,
                L.eth_b.y,
                50'000'000u / link_sync::kPaceTicks);
            continue;
        }
        const uint32_t zero[2] = {0, 0};  // stop + done, clear before launch
        cluster.write_core(zero, sizeof(zero), tt_cxy_pair(L.chip_a, virt_a), stop_addr);
        cluster.write_core(zero, sizeof(zero), tt_cxy_pair(L.chip_b, virt_b), stop_addr);
        const std::vector<uint32_t> ct = {kLinkSyncChannels, kLinkSyncSamples, kLinkSyncSampleSize};
        auto ps = std::make_unique<Program>(CreateProgram());
        auto pr = std::make_unique<Program>(CreateProgram());
        const auto kid_s = CreateKernel(
            *ps,
            "tt_metal/tools/profiler/sync/sync_device_kernel_sender.cpp",
            L.eth_a,
            EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = ct});
        const auto kid_r = CreateKernel(
            *pr,
            "tt_metal/tools/profiler/sync/sync_device_kernel_receiver.cpp",
            L.eth_b,
            EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = ct});
        // The stop word and pace ride as RUNTIME args (positional compile args past index 2 do not reach an eth
        // kernel here). Sender: {stop_addr, pace}; receiver: {stop_addr}.
        SetRuntimeArgs(*ps, kid_s, L.eth_a, {stop_addr, link_sync::kPaceTicks});
        SetRuntimeArgs(*pr, kid_r, L.eth_b, {stop_addr});
        try {
            detail::CompileProgram(dev_a, *ps, /*force_slow_dispatch=*/true);
            detail::CompileProgram(dev_b, *pr, /*force_slow_dispatch=*/true);
        } catch (const std::exception& ex) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] link sync {}<->{}: sync kernels failed to compile ({}); pair skipped",
                L.chip_a,
                L.chip_b,
                ex.what());
            continue;
        }
        detail::WriteRuntimeArgsToDevice(dev_a, *ps, /*force_slow_dispatch=*/true);
        detail::WriteRuntimeArgsToDevice(dev_b, *pr, /*force_slow_dispatch=*/true);
        detail::LaunchProgram(dev_a, *ps, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
        detail::LaunchProgram(dev_b, *pr, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
        link_syncs_.push_back(ResidentSync{
            .ps = std::move(ps),
            .pr = std::move(pr),
            .dev_a = dev_a,
            .dev_b = dev_b,
            .virt_a = virt_a,
            .virt_b = virt_b,
            .chip_a = L.chip_a,
            .chip_b = L.chip_b,
            .stop_a = stop_addr,
            .stop_b = stop_addr});
        log_info(
            tt::LogMetal,
            "[streaming profiler] link sync {} eth({},{}) -> {} eth({},{}): RESIDENT at {} Hz",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            50'000'000u / link_sync::kPaceTicks);
    }
    // Every planned end is in place (launched here, or a router that has been waiting): let the senders go.
    for (const ResidentSync& r : link_syncs_) {
        cluster.write_core(&link_sync::kCtlRun, sizeof(uint32_t), tt_cxy_pair(r.chip_a, r.virt_a), r.stop_a);
    }
}

void SyncDevices::stop_links(tt::Cluster& cluster) {
    const auto poll_done = [&](uint32_t chip, const CoreCoord& virt, uint32_t done_addr, const char* which) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        for (;;) {
            uint32_t done = 0;
            cluster.read_core(&done, sizeof(done), tt_cxy_pair(chip, virt), done_addr);
            if (done != 0) {
                return;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] resident link sync {} on chip {} did not confirm stop within 2 s",
                    which,
                    chip);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    for (const ResidentSync& r : link_syncs_) {
        const bool resident = r.ps != nullptr;
        // Sender first: its current round still completes off the live receiver, then it stops between rounds; a
        // resident sender exits, a router-hosted one goes quiet.
        cluster.write_core(&link_sync::kCtlStop, sizeof(uint32_t), tt_cxy_pair(r.chip_a, r.virt_a), r.stop_a);
        if (resident) {
            poll_done(r.chip_a, r.virt_a, r.stop_a + 4, "sender");
        }
        // What each end left past its control word (eth_ptp::StopDiag): rounds, the timer word (0 no hardware path,
        // 1 ran, 2 never acknowledged its rate, in which case that end emitted no hardware stamps), wall cycles
        // inside bursts, wall cycles and refclk ticks of the run, the longest burst in wall cycles, then rounds
        // dropped, bursts with a hand-off beyond the frames, bursts whose frames did not all hand off or stamp in
        // time, rounds with the ingress count off, and waits for a frame or echo given up.
        struct StopDiag {
            uint32_t rounds, timer, hold_lo, hold_hi, wall_lo, wall_hi, ref_lo, ref_hi, hold_max, drop[5];
        };
        static_assert(sizeof(StopDiag) == 14 * sizeof(uint32_t));
        StopDiag da{}, db{};
        cluster.read_core(&da, sizeof(da), tt_cxy_pair(r.chip_a, r.virt_a), r.stop_a + 8);
        // Now the receiver sees no further frame; a resident one exits on its stop word.
        if (resident) {
            cluster.write_core(&link_sync::kCtlStop, sizeof(uint32_t), tt_cxy_pair(r.chip_b, r.virt_b), r.stop_b);
            poll_done(r.chip_b, r.virt_b, r.stop_b + 4, "receiver");
        }
        cluster.read_core(&db, sizeof(db), tt_cxy_pair(r.chip_b, r.virt_b), r.stop_b + 8);
        const auto u64 = [](uint32_t lo, uint32_t hi) { return static_cast<double>((uint64_t{hi} << 32) | lo); };
        const auto pct = [&](const StopDiag& d) {
            const double wall = u64(d.wall_lo, d.wall_hi);
            return wall == 0.0 ? 0.0 : 100.0 * u64(d.hold_lo, d.hold_hi) / wall;
        };
        const auto longest_us = [&](const StopDiag& d) {
            const double wall = u64(d.wall_lo, d.wall_hi);
            return wall == 0.0 ? 0.0 : d.hold_max * (u64(d.ref_lo, d.ref_hi) * 20.0 / wall) / 1000.0;
        };
        log_info(
            tt::LogMetal,
            "[streaming profiler] link sync chip {} -> chip {}: {} rounds over {:.0f} ms; core time in bursts: "
            "sender {:.2f} % (longest {:.2f} us), receiver {:.2f} % (longest {:.2f} us)",
            r.chip_a,
            r.chip_b,
            da.rounds,
            u64(da.ref_lo, da.ref_hi) / 50'000.0,
            pct(da),
            longest_us(da),
            pct(db),
            longest_us(db));
        for (const auto& [chip, name, d] :
             {std::tuple{r.chip_a, "sender", &da}, std::tuple{r.chip_b, "receiver", &db}}) {
            if (d->drop[0] != 0 || d->drop[1] != 0 || d->drop[4] != 0) {
                log_info(
                    tt::LogMetal,
                    "[streaming profiler] link sync chip {} {} dropped {} hardware rounds: bursts with a hand-off "
                    "beyond "
                    "the frames {}, bursts whose frames did not all stamp in time {}, ingress count off {}, waits "
                    "given "
                    "up {}",
                    chip,
                    name,
                    d->drop[0],
                    d->drop[1],
                    d->drop[2],
                    d->drop[3],
                    d->drop[4]);
            }
        }
        for (const auto& [chip, word] : {std::pair{r.chip_a, da.timer}, std::pair{r.chip_b, db.timer}}) {
            if (word == 2) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] link sync chip {}: the 1588 timer never acknowledged its rate; this end sent "
                    "no stamps and the link is not solved",
                    chip);
            }
        }
    }
    link_syncs_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
