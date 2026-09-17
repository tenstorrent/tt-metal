// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <x86intrin.h>

#include <fmt/format.h>
#include <numa.h>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_engine.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

// The PCIe tile's SII register block in its own address space, and the count-from-reset timer inside it
// (Blackhole PCIE_SS spec, tables 4 and 12).
constexpr uint64_t kSiiBase = 0xFFFFFFFFF0000000ull;
constexpr uint32_t kCfrLo = 0xA8, kCfrHi = 0xAC;
constexpr size_t kTlbBytes = 2 * 1024 * 1024;
constexpr uint32_t kBurstReads = 1000;
// Reads within this much of the burst's tightest round trip carry the least queueing on either leg.
constexpr double kRttSlackNs = 50.0;
constexpr size_t kWindowBursts = 10;
constexpr size_t kWindowPairs = 20;
// A steady pair off the segment by more than the pair bracket can explain is a slew step.
constexpr double kSteadyKinkNs = 40.0;

}  // namespace

int64_t clock_ns(clockid_t id) {
    timespec ts{};
    clock_gettime(id, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

int64_t tsc_now() noexcept {
    _mm_lfence();
    const int64_t t = static_cast<int64_t>(__rdtsc());
    _mm_lfence();
    return t;
}

double tsc_ticks_per_ns() {
    static const double rate = [] {
        const int64_t t0 = tsc_now(), r0 = clock_ns(CLOCK_MONOTONIC_RAW);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int64_t r1 = clock_ns(CLOCK_MONOTONIC_RAW), t1 = tsc_now();
        return static_cast<double>(t1 - t0) / static_cast<double>(r1 - r0);
    }();
    return rate;
}

double units_per_tsc() {
    static const double u = 10.0 / tsc_ticks_per_ns();
    return u;
}

namespace {
// Double-buffered under a generation, so a reader that sees the new generation sees the whole segment.
struct SteadySlots {
    SteadySegment seg[2];
    std::atomic<uint32_t> gen{0};
};
SteadySlots g_steady;
}  // namespace

void SteadyView::set(const SteadySegment& segment) noexcept {
    const uint32_t g = g_steady.gen.load(std::memory_order_relaxed);
    g_steady.seg[(g + 1) & 1] = segment;
    g_steady.gen.store(g + 1, std::memory_order_release);
}

int64_t SteadyView::mono_ns(int64_t tsc) noexcept {
    thread_local uint32_t gen = ~0u;
    thread_local SteadySegment seg;
    const uint32_t g = g_steady.gen.load(std::memory_order_acquire);
    if (g != gen) {
        seg = g_steady.seg[g & 1];
        gen = g;
    }
    if (!seg.ok) {
        // The stand-in pair is taken once per thread and generation.
        seg = SteadySegment{static_cast<int64_t>(__rdtsc()), clock_ns(CLOCK_MONOTONIC), 1.0 / tsc_ticks_per_ns(), true};
    }
    return seg.mono_of(tsc);
}

HostProbe::HostProbe(tt::Cluster& cluster, uint32_t chip_id, ClockMap& map) :
    cluster_(cluster), chip_id_(chip_id), map_(map) {
    ticks_per_ns_ = tsc_ticks_per_ns();
    const auto pcie =
        cluster_.get_driver()->get_soc_descriptor(chip_id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    if (pcie.empty()) {
        log_warning(
            tt::LogMetal, "[streaming profiler] host probe: chip {} has no PCIe tile in its descriptor", chip_id);
        return;
    }
    pcie_x_ = static_cast<uint32_t>(pcie.front().x);
    pcie_y_ = static_cast<uint32_t>(pcie.front().y);
    auto* tlb_manager = cluster_.get_driver()->get_chip(chip_id)->get_tlb_manager();
    const tt_xy_pair xy(pcie_x_, pcie_y_);
    try {
        if (!tlb_manager->is_tlb_mapped(xy)) {
            tlb_manager->configure_tlb(xy, kTlbBytes, kSiiBase, tt::umd::tlb_data::Strict);
        }
        if (tlb_manager->is_tlb_mapped(xy, kSiiBase + kCfrLo, 4)) {
            window_ = tlb_manager->get_tlb_window(xy);
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[streaming profiler] host probe: no static TLB for the PCIe tile: {}", e.what());
    }
    // The LO read latches HI; the host is this instance's only reader, so the pair is consistent.
    cfr_lo_last_ = read_cfr_lo();
    uint32_t hi = 0;
    cluster_.read_reg(&hi, tt_cxy_pair(chip_id_, CoreCoord(pcie_x_, pcie_y_)), kSiiBase + kCfrHi);
    cfr_hi_ = hi;
    thread_ = std::thread(&HostProbe::run, this);
}

HostProbe::~HostProbe() { stop(); }

void HostProbe::stop() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

HostLine HostProbe::line() const {
    std::lock_guard<std::mutex> g(mu_);
    return line_;
}

SteadySegment HostProbe::steady() const {
    std::lock_guard<std::mutex> g(mu_);
    return steady_;
}

uint32_t HostProbe::read_cfr_lo() {
    if (window_ != nullptr) {
        return window_->read32(kCfrLo);
    }
    uint32_t v = 0;
    cluster_.read_reg(&v, tt_cxy_pair(chip_id_, CoreCoord(pcie_x_, pcie_y_)), kSiiBase + kCfrLo);
    return v;
}

bool HostProbe::burst(BurstPoint& out) {
    struct Read {
        int64_t mid, rtt;
        uint64_t refclk;
    };
    std::vector<Read> reads;
    reads.reserve(kBurstReads);
    for (uint32_t i = 0; i < kBurstReads; i++) {
        const int64_t t0 = tsc_now();
        const uint32_t lo = read_cfr_lo();
        const int64_t t1 = tsc_now();
        if (lo < cfr_lo_last_) {
            cfr_hi_++;
        }
        cfr_lo_last_ = lo;
        reads.push_back(Read{t0 + (t1 - t0) / 2, t1 - t0, (static_cast<uint64_t>(cfr_hi_) << 32) | lo});
    }
    int64_t rtt_min = reads.front().rtt;
    for (const Read& r : reads) {
        rtt_min = std::min(rtt_min, r.rtt);
    }
    const int64_t cut = rtt_min + static_cast<int64_t>(kRttSlackNs * ticks_per_ns_);
    const int64_t tsc_ref = reads.front().mid;
    const uint64_t ref_ref = reads.front().refclk;
    double st = 0.0, sr = 0.0;
    uint32_t n = 0;
    for (const Read& r : reads) {
        if (r.rtt <= cut) {
            st += static_cast<double>(r.mid - tsc_ref);
            sr += static_cast<double>(r.refclk - ref_ref);
            n++;
        }
    }
    reads_ += reads.size();
    kept_ += n;
    if (n == 0) {
        return false;
    }
    out = BurstPoint{static_cast<double>(tsc_ref) + st / n, static_cast<double>(ref_ref) + sr / n};
    return true;
}

// Least squares of the burst points, centred on their means so the ~1e12-tick magnitudes cancel.
void HostProbe::refit() {
    const size_t n = points_.size();
    HostLine l;
    l.bursts = static_cast<uint32_t>(n);
    if (n >= 2) {
        double mt = 0.0, mr = 0.0;
        for (const BurstPoint& p : points_) {
            mt += p.tsc;
            mr += p.refclk;
        }
        mt /= n;
        mr /= n;
        double srr = 0.0, srt = 0.0;
        for (const BurstPoint& p : points_) {
            srr += (p.refclk - mr) * (p.refclk - mr);
            srt += (p.refclk - mr) * (p.tsc - mt);
        }
        if (srr > 0.0) {
            l.b = srt / srr;
            l.a = mt - l.b * mr;
            double rss = 0.0;
            for (const BurstPoint& p : points_) {
                const double e = p.tsc - l.tsc_of(p.refclk);
                rss += e * e;
            }
            l.sigma_ns = n > 2 ? std::sqrt(rss / (n - 2)) / ticks_per_ns_ : 0.0;
            l.ok = true;
        }
    }
    std::lock_guard<std::mutex> g(mu_);
    line_ = l;
}

void HostProbe::steady_pair() {
    int64_t best_gap = INT64_MAX, best_tsc = 0, best_mono = 0;
    for (int i = 0; i < 16; i++) {
        const int64_t t0 = tsc_now();
        const int64_t m = clock_ns(CLOCK_MONOTONIC);
        const int64_t t1 = tsc_now();
        if (t1 - t0 < best_gap) {
            best_gap = t1 - t0;
            best_tsc = t0 + (t1 - t0) / 2;
            best_mono = m;
        }
    }
    SteadySegment cur = steady();
    if (cur.ok) {
        const double resid = static_cast<double>(best_mono - cur.mono_of(best_tsc));
        if (std::abs(resid) > kSteadyKinkNs) {
            log_debug(tt::LogMetal, "[streaming profiler] host probe: steady_clock slew step of {:+.0f} ns", resid);
            pairs_.clear();
        }
    }
    pairs_.emplace_back(best_tsc, best_mono);
    while (pairs_.size() > kWindowPairs) {
        pairs_.pop_front();
    }
    SteadySegment s;
    const size_t n = pairs_.size();
    if (n >= 2) {
        double mt = 0.0, mm = 0.0;
        for (const auto& [t, m] : pairs_) {
            mt += static_cast<double>(t - pairs_.front().first);
            mm += static_cast<double>(m - pairs_.front().second);
        }
        mt /= n;
        mm /= n;
        double stt = 0.0, stm = 0.0;
        for (const auto& [t, m] : pairs_) {
            const double dt = static_cast<double>(t - pairs_.front().first) - mt;
            stt += dt * dt;
            stm += dt * (static_cast<double>(m - pairs_.front().second) - mm);
        }
        if (stt > 0.0) {
            s.ns_per_tick = stm / stt;
            s.tsc0 = pairs_.front().first + static_cast<int64_t>(mt);
            s.mono0 = pairs_.front().second + static_cast<int64_t>(mm);
            s.ok = true;
        }
    } else {
        s.tsc0 = best_tsc;
        s.mono0 = best_mono;
        s.ns_per_tick = 1.0 / ticks_per_ns_;
        s.ok = true;
    }
    {
        std::lock_guard<std::mutex> g(mu_);
        steady_ = s;
    }
    SteadyView::set(s);
}

void HostProbe::run() {
    set_os_thread_name("sp-hostprobe");
    // The read's round trip is 90 ns longer from the other socket, and the extra sits on one leg, so a thread that
    // wanders between sockets shifts the bracket's midpoint by tens of ns burst to burst.
    if (numa_available() != -1) {
        numa_run_on_node(static_cast<int>(cluster_.get_numa_node_for_device(chip_id_)));
    }
    // The first bursts come quickly so a line exists before records need it; then one every 100 ms, a cadence the
    // refclk period's ramp on the TSC (~0.05 ppm/s) keeps the line's prediction within tens of ns of.
    static constexpr int64_t kSchedule[] = {0, 100, 200, 300};
    const auto t_start = std::chrono::steady_clock::now();
    size_t k = 0;
    int64_t next_pair_ms = 0;
    const auto elapsed_ms = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start)
            .count();
    };
    const auto take = [&] {
        BurstPoint p{};
        if (!burst(p)) {
            return;
        }
        points_.push_back(p);
        while (points_.size() > kWindowBursts) {
            points_.pop_front();
        }
        refit();
        bursts_++;
        const HostLine l = line();
        // A node per burst: the line as it stands, at the burst's own instant. Records placed between two
        // bursts run on the newer node's tangent; those placed later interpolate between the nodes.
        if (l.ok && l.bursts >= 3) {
            map_.append_host(HostNode{.at = p.refclk, .value = l.tsc_of(p.refclk), .tangent = l.b});
        }
    };
    while (!stop_.load(std::memory_order_acquire)) {
        const int64_t now_ms = elapsed_ms();
        const int64_t due = k < std::size(kSchedule) ? kSchedule[k]
                                                     : kSchedule[std::size(kSchedule) - 1] +
                                                           100 * static_cast<int64_t>(k + 1 - std::size(kSchedule));
        if (now_ms >= due) {
            take();
            k++;
        }
        if (now_ms >= next_pair_ms) {
            steady_pair();
            next_pair_ms = now_ms + 100;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // A capture shorter than the schedule's third burst would end without a line: the bursts still owed come now,
    // and the last spans the capture with the first.
    for (int tries = 0; tries < 8; tries++) {
        if (const HostLine l = line(); l.ok && l.bursts >= 3) {
            break;
        }
        take();
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] host probe chip {}: {} bursts, {} of {} reads kept; final refclk period {:.6f} ns, "
        "residual {:.1f} ns; steady_clock {:.9f} ns per tick",
        chip_id_,
        bursts_,
        kept_,
        reads_,
        line().ok ? line().b / ticks_per_ns_ : 0.0,
        line().sigma_ns,
        steady().ns_per_tick);
}

namespace link_sync {

bool enabled() {
    const auto& rtoptions = MetalContext::instance().rtoptions();
    return rtoptions.get_streaming_profiler_enabled() && rtoptions.get_streaming_profiler_link_sync_enabled();
}

namespace {
// Whether an eth core can carry the sync: any connected core without fabric; with fabric, only a core the topology
// gave a router, since the router is what runs the link end.
bool eligible(const tt::Cluster& cluster, uint32_t chip, const CoreCoord& eth_logical) {
    auto& mc = MetalContext::instance();
    if (mc.get_fabric_config() == tt_fabric::FabricConfig::DISABLED) {
        return true;
    }
    const auto& cp = mc.get_control_plane();
    const auto node = cp.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(chip));
    const auto& soc = cluster.get_soc_desc(static_cast<ChipId>(chip));
    for (const auto& [chan, direction] : cp.get_active_fabric_eth_channels(node)) {
        if (soc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL) == eth_logical) {
            return true;
        }
    }
    return false;
}
}  // namespace

std::vector<Link> links_between(const tt::Cluster& cluster, uint32_t chip_x, uint32_t chip_y) {
    std::vector<Link> out;
    const uint32_t lo = std::min(chip_x, chip_y), hi = std::max(chip_x, chip_y);
    if (lo == hi) {
        return out;
    }
    const auto by_peer = cluster.get_ethernet_cores_grouped_by_connected_chips(static_cast<ChipId>(lo));
    const auto it = by_peer.find(static_cast<ChipId>(hi));
    if (it == by_peer.end()) {
        return out;
    }
    for (const CoreCoord& eth_a : it->second) {
        const CoreCoord eth_b =
            std::get<1>(cluster.get_connected_ethernet_core(std::make_tuple(static_cast<ChipId>(lo), eth_a)));
        if (eligible(cluster, lo, eth_a) && eligible(cluster, hi, eth_b)) {
            out.push_back(Link{.chip_a = lo, .chip_b = hi, .eth_a = eth_a, .eth_b = eth_b});
        }
    }
    return out;
}

Role role_of(const tt::Cluster& cluster, uint32_t chip, const CoreCoord& eth_logical) {
    const auto by_peer = cluster.get_ethernet_cores_grouped_by_connected_chips(static_cast<ChipId>(chip));
    for (const auto& [peer, cores] : by_peer) {
        if (std::find(cores.begin(), cores.end(), eth_logical) == cores.end()) {
            continue;
        }
        std::vector<Link> links = links_between(cluster, chip, static_cast<uint32_t>(peer));
        for (const Link& link : links) {
            if (link.chip_a == chip && link.eth_a == eth_logical) {
                return Role::Sender;
            }
            if (link.chip_b == chip && link.eth_b == eth_logical) {
                return Role::Receiver;
            }
        }
        return Role::None;
    }
    return Role::None;
}

}  // namespace link_sync

namespace {
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
            host_probe_ = std::make_shared<HostProbe>(cluster, devices_[di].d.chip_id, service().sync().map());
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

// One idle-eth core's reading of one tile: the core's wall tick minus the tile's.
struct SyncDevices::TileReading {
    int64_t value = 0;
};

// One reading as an equation: value = x[t] - x[s] + x[path], x being pusher wall minus tile wall. -1 is the pusher
// (the origin, x = 0) or, for the path, a two-ring read.
struct SyncDevices::TileObs {
    int32_t s, t;
    uint32_t src;  // source order
    TileReading r;
    CoreCoord from, to;  // raw NoC 0 coordinates
};

// The unknowns: the Tensix tiles, the helper sources, then two path terms. A read within the source's column or row
// covers one ring, the rest two; the one-ring reads sit a fixed few ticks off the two-ring ones, so each of those two
// paths gets an unknown of its own, found from the tiles both kinds of source read, instead of pulling on the tiles'
// offsets.
struct SyncDevices::TileUnknowns {
    size_t n_tiles, n_helpers;
    size_t col() const { return n_tiles + n_helpers; }
    size_t row() const { return n_tiles + n_helpers + 1; }
    size_t count() const { return n_tiles + n_helpers + 2; }
    int32_t path_of(const TileObs& o) const {
        if (o.from.x == o.to.x) {
            return static_cast<int32_t>(col());
        }
        return o.from.y == o.to.y ? static_cast<int32_t>(row()) : -1;
    }
    double fit_of(const TileObs& o, const std::vector<double>& x) const {
        const auto x_of = [&](int32_t u) { return u < 0 ? 0.0 : x[static_cast<size_t>(u)]; };
        return x_of(o.t) - x_of(o.s) + x_of(path_of(o));
    }
};

std::vector<SyncDevices::TileObs> SyncDevices::read_tiles(uint32_t di) {
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
    auto read_table = [&](const CoreCoord& virt, uint32_t count, std::vector<TileReading>& out) {
        std::vector<uint32_t> t(kernel_profiler::eth_tile_out_word(count, count), 0);
        cluster.read_core(
            t.data(), static_cast<uint32_t>(t.size() * sizeof(uint32_t)), tt_cxy_pair(d.chip_id, virt), eth_l1_.table);
        if (t[kernel_profiler::ETH_TILE_READY] != (kernel_profiler::kEthTileReadyWord | count)) {
            return false;
        }
        out.resize(count);
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t w = kernel_profiler::eth_tile_out_word(count, i);
            out[i].value = static_cast<int64_t>((static_cast<uint64_t>(t[w + 1]) << 32) | t[w]);
        }
        return true;
    };
    // Sources: the pusher, then the helpers; a source's tile list is the Tensix tiles, then the other sources in
    // source order.
    std::vector<std::vector<TileReading>> readings(ns);
    for (uint32_t s = 0; s < ns; s++) {
        const CoreCoords& src = d.eth[s];
        std::vector<uint32_t> table(kernel_profiler::ETH_TILE_XY_0, 0);
        for (const CoreCoords& c : d.tensix) {
            table.push_back(packed_xy(c.virt));
        }
        for (uint32_t o = 0; o < ns; o++) {
            if (o != s) {
                table.push_back(packed_xy(d.eth[o].virt));
            }
        }
        table[kernel_profiler::ETH_TILE_N] = nt + nh;
        cluster.write_core(
            table.data(),
            static_cast<uint32_t>(table.size() * sizeof(uint32_t)),
            tt_cxy_pair(d.chip_id, src.virt),
            eth_l1_.table);
        Program p = CreateProgram();
        const KernelHandle kid = create_pusher_kernel(p, eth_l1_, src, /*measure_only=*/true);
        SetRuntimeArgs(p, kid, src.logical, std::vector<uint32_t>{0});
        detail::CompileProgram(d.device, p, /*force_slow_dispatch=*/true);
        detail::WriteRuntimeArgsToDevice(d.device, p, /*force_slow_dispatch=*/true);
        detail::LaunchProgram(d.device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!read_table(src.virt, nt + nh, readings[s])) {
            TT_FATAL(
                std::chrono::steady_clock::now() < deadline,
                "streaming profiler: device {} idle eth ({},{}) tile table not written within 2 s",
                d.chip_id,
                src.logical.x,
                src.logical.y);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        detail::WaitProgramDone(d.device, p, false);
    }
    // Unknown index: Tensix i -> i, helper h -> nt + h, the pusher -1.
    const auto unknown_of_source = [&](uint32_t s) { return s == 0 ? -1 : static_cast<int32_t>(nt + s - 1); };
    std::vector<TileObs> obs;
    for (uint32_t s = 0; s < ns; s++) {
        uint32_t listed = 0;
        for (uint32_t i = 0; i < nt; i++, listed++) {
            obs.push_back(TileObs{
                unknown_of_source(s),
                static_cast<int32_t>(i),
                s,
                readings[s][listed],
                d.eth[s].phys,
                d.tensix[i].phys});
        }
        for (uint32_t o = 0; o < ns; o++) {
            if (o != s) {
                obs.push_back(TileObs{
                    unknown_of_source(s),
                    unknown_of_source(o),
                    s,
                    readings[s][listed++],
                    d.eth[s].phys,
                    d.eth[o].phys});
            }
        }
    }
    return obs;
}

std::vector<double> SyncDevices::solve_tiles(uint32_t di) {
    const Device& d = devices_[di].d;
    const std::vector<TileObs> obs = read_tiles(di);
    const TileUnknowns u{d.tensix.size(), d.eth.size() - 1};
    std::vector<std::vector<double>> N(u.count(), std::vector<double>(u.count(), 0.0));
    std::vector<double> rhs(u.count(), 0.0);
    for (const TileObs& o : obs) {
        const double v = static_cast<double>(o.r.value);
        const int32_t cols[3] = {o.t, o.s, u.path_of(o)};
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
    log_tile_fit(d, obs, x, u);
    return x;
}

// Two numbers per chip: how far apart the tiles' clocks are, and how well the sources agree on each tile (the
// placement's likely error).
void SyncDevices::log_tile_fit(
    const Device& d, const std::vector<TileObs>& obs, const std::vector<double>& x, const TileUnknowns& u) const {
    const double ns_per_tick = 1.0 / d.frequency_ghz;
    const auto [xlo, xhi] = std::minmax_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(u.n_tiles));
    double ss = 0.0, worst = 0.0;
    for (const TileObs& o : obs) {
        const double res = static_cast<double>(o.r.value) - u.fit_of(o, x);
        ss += res * res;
        worst = std::max(worst, std::fabs(res));
    }
    const double rms = obs.empty() ? 0.0 : std::sqrt(ss / static_cast<double>(obs.size()));
    log_info(
        tt::LogMetal,
        "[streaming profiler] Device {}: {} tiles' wall clocks span {:.0f} ticks ({:.1f} ns), solved from {} idle eth "
        "sources over {} reads; the sources disagree by {:.2f} ticks rms, {:.1f} worst ({:.2f} ns rms); one-ring "
        "reads sit {:+.1f} (column) {:+.1f} (row) ticks off two-ring ones",
        d.chip_id,
        u.n_tiles,
        *xhi - *xlo,
        (*xhi - *xlo) * ns_per_tick,
        d.eth.size(),
        obs.size(),
        rms,
        worst,
        rms * ns_per_tick,
        x[u.col()],
        x[u.row()]);
}

void SyncDevices::measure_tiles(uint32_t di, CaptureContext::Device& cap) {
    const std::vector<double> x = solve_tiles(di);
    for (size_t i = 0; i < devices_[di].d.tensix.size(); i++) {
        cap.tile_offset[i] = std::llround(x[i]);
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
    // An end's core as the decoder numbers it: the roster is the workers, the pusher, then its linked cores.
    const auto core_of = [&](size_t di, const CoreCoord& eth) -> int64_t {
        const Device& d = devices_[di].d;
        for (size_t i = 0; i < d.linked.size(); i++) {
            if (d.linked[i].logical == eth) {
                return static_cast<int64_t>(d.tensix.size() + 1 + i);
            }
        }
        return -1;
    };
    for (size_t a = 0; a < devices_.size(); a++) {
        const uint32_t chip_a = devices_[a].d.chip_id;
        for (size_t b = a + 1; b < devices_.size(); b++) {
            const uint32_t chip_b = devices_[b].d.chip_id;
            for (const link_sync::Link& link : link_sync::links_between(cluster, chip_a, chip_b)) {
                const bool flip = link.chip_a != chip_a;  // the lower chip sends
                const size_t dev_a = flip ? b : a, dev_b = flip ? a : b;
                const int64_t core_a = core_of(dev_a, link.eth_a), core_b = core_of(dev_b, link.eth_b);
                if (core_a < 0 || core_b < 0) {
                    log_info(
                        tt::LogMetal,
                        "[streaming profiler] link sync {} eth({},{}) -> {} eth({},{}): an end is outside the pusher's "
                        "roster (a dispatch tunnel's core), link not used",
                        link.chip_a,
                        link.eth_a.x,
                        link.eth_a.y,
                        link.chip_b,
                        link.eth_b.x,
                        link.eth_b.y);
                    continue;
                }
                links_.push_back(CaptureContext::Link{
                    .dev_a = static_cast<uint32_t>(dev_a),
                    .dev_b = static_cast<uint32_t>(dev_b),
                    .chip_a = link.chip_a,
                    .chip_b = link.chip_b,
                    .core_a = static_cast<uint32_t>(core_a),
                    .core_b = static_cast<uint32_t>(core_b),
                    .eth_a = link.eth_a,
                    .eth_b = link.eth_b});
            }
        }
    }
    if (links_.empty() && devices_.size() > 1) {
        log_warning(tt::LogMetal, "[streaming profiler] link sync: no eth connection between the local devices");
    }
}

bool SyncDevices::launch_link_ends(const CaptureContext::Link& L, uint32_t link_l1, ResidentSync& out) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const uint32_t zero[2] = {0, 0};  // stop + done, clear before launch
    cluster.write_core(zero, sizeof(zero), tt_cxy_pair(L.chip_a, out.virt_a), out.stop_a);
    cluster.write_core(zero, sizeof(zero), tt_cxy_pair(L.chip_b, out.virt_b), out.stop_b);
    auto ps = std::make_unique<Program>(CreateProgram());
    auto pr = std::make_unique<Program>(CreateProgram());
    const auto kid_s = CreateKernel(
        *ps,
        "tt_metal/tools/profiler/sync/eth_ptp_link_sender.cpp",
        L.eth_a,
        EthernetConfig{.noc = NOC::RISCV_0_default});
    const auto kid_r = CreateKernel(
        *pr,
        "tt_metal/tools/profiler/sync/eth_ptp_link_receiver.cpp",
        L.eth_b,
        EthernetConfig{.noc = NOC::RISCV_0_default});
    SetRuntimeArgs(*ps, kid_s, L.eth_a, {link_l1, kernel_profiler::kLinkSyncPaceTicks});
    SetRuntimeArgs(*pr, kid_r, L.eth_b, {link_l1});
    try {
        detail::CompileProgram(out.dev_a, *ps, /*force_slow_dispatch=*/true);
        detail::CompileProgram(out.dev_b, *pr, /*force_slow_dispatch=*/true);
    } catch (const std::exception& ex) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] link sync {}<->{}: sync kernels failed to compile ({}); pair skipped",
            L.chip_a,
            L.chip_b,
            ex.what());
        return false;
    }
    detail::WriteRuntimeArgsToDevice(out.dev_a, *ps, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(out.dev_b, *pr, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(out.dev_a, *ps, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(out.dev_b, *pr, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    out.ps = std::move(ps);
    out.pr = std::move(pr);
    return true;
}

// Launching this during boot() instead deadlocked: the FIFO filled with no reader, the pusher parked in
// socket_reserve_pages, and the armed sync kernels wedged an eth core (a board reset).
void SyncDevices::launch_links() {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    // The link's L1 is the top of the active eth core's UNRESERVED region, the same place whether a resident kernel
    // or a router runs the end (the routers' config leaves it clear).
    const uint32_t link_l1 = aeth_unreserved_ + aeth_unres_size_ - kernel_profiler::kLinkSyncL1Bytes;
    const uint32_t stop_addr = link_l1 + kernel_profiler::kLinkSyncCtlOffset;
    for (const CaptureContext::Link& L : links_) {
        ResidentSync r{
            .dev_a = devices_[L.dev_a].d.device,
            .dev_b = devices_[L.dev_b].d.device,
            .virt_a = cluster.get_virtual_coordinate_from_logical_coordinates(L.chip_a, L.eth_a, CoreType::ETH),
            .virt_b = cluster.get_virtual_coordinate_from_logical_coordinates(L.chip_b, L.eth_b, CoreType::ETH),
            .chip_a = L.chip_a,
            .chip_b = L.chip_b,
            .stop_a = stop_addr,
            .stop_b = stop_addr};
        // With fabric the routers on this link run the two ends (fabric_erisc_router.cpp, LINK_SYNC_ROLE): nothing
        // to launch, the sender waits for the run word, and their diagnostics are read where the resident kernels
        // leave theirs.
        if (!fabric_link_sync_ && !launch_link_ends(L, link_l1, r)) {
            continue;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] link sync {} eth({},{}) -> {} eth({},{}): {} at {} Hz",
            L.chip_a,
            L.eth_a.x,
            L.eth_a.y,
            L.chip_b,
            L.eth_b.x,
            L.eth_b.y,
            fabric_link_sync_ ? "in the fabric routers" : "RESIDENT",
            kernel_profiler::kEthRefclkHz / kernel_profiler::kLinkSyncPaceTicks);
        link_syncs_.push_back(std::move(r));
    }
    // Every planned end is in place (launched here, or a router that has been waiting): let the senders go.
    for (const ResidentSync& r : link_syncs_) {
        cluster.write_core(
            &kernel_profiler::kLinkSyncCtlRun, sizeof(uint32_t), tt_cxy_pair(r.chip_a, r.virt_a), r.stop_a);
    }
}

namespace {

// What each end leaves past its control word (eth_ptp::StopDiag): rounds, the timer word (0 no hardware path, 1 ran,
// 2 never acknowledged its rate, in which case that end emitted no hardware stamps), wall cycles inside bursts, wall
// cycles and refclk ticks of the run, the longest burst in wall cycles, then rounds dropped, bursts with a hand-off
// beyond the frames, bursts whose frames did not all hand off or stamp in time, rounds with the ingress count off,
// and waits for a frame or echo given up.
struct StopDiag {
    uint32_t rounds, timer, hold_lo, hold_hi, wall_lo, wall_hi, ref_lo, ref_hi, hold_max, drop[5];
    double wall() const { return static_cast<double>((uint64_t{wall_hi} << 32) | wall_lo); }
    double refclk() const { return static_cast<double>((uint64_t{ref_hi} << 32) | ref_lo); }
    // The share of the core's time inside bursts, and the longest burst in us.
    double hold_pct() const {
        return wall() == 0.0 ? 0.0 : 100.0 * static_cast<double>((uint64_t{hold_hi} << 32) | hold_lo) / wall();
    }
    double longest_us() const { return wall() == 0.0 ? 0.0 : hold_max * (refclk() * 20.0 / wall()) / 1000.0; }
};
static_assert(sizeof(StopDiag) == 14 * sizeof(uint32_t));

StopDiag read_stop_diag(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint32_t ctl) {
    StopDiag d{};
    cluster.read_core(&d, sizeof(d), tt_cxy_pair(chip, virt), ctl + 8);
    return d;
}

void log_link_diag(uint32_t chip_a, uint32_t chip_b, const StopDiag& da, const StopDiag& db) {
    log_info(
        tt::LogMetal,
        "[streaming profiler] link sync chip {} -> chip {}: {} rounds over {:.0f} ms; core time in bursts: "
        "sender {:.2f} % (longest {:.2f} us), receiver {:.2f} % (longest {:.2f} us)",
        chip_a,
        chip_b,
        da.rounds,
        da.refclk() / 50'000.0,
        da.hold_pct(),
        da.longest_us(),
        db.hold_pct(),
        db.longest_us());
    for (const auto& [chip, name, d] : {std::tuple{chip_a, "sender", &da}, std::tuple{chip_b, "receiver", &db}}) {
        if (d->drop[0] != 0 || d->drop[1] != 0 || d->drop[4] != 0) {
            log_info(
                tt::LogMetal,
                "[streaming profiler] link sync chip {} {} dropped {} hardware rounds: bursts with a hand-off beyond "
                "the frames {}, bursts whose frames did not all stamp in time {}, ingress count off {}, waits given "
                "up {}",
                chip,
                name,
                d->drop[0],
                d->drop[1],
                d->drop[2],
                d->drop[3],
                d->drop[4]);
        }
        if (d->timer == 2) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] link sync chip {}: the 1588 timer never acknowledged its rate; this end sent no "
                "stamps and the link is not solved",
                chip);
        }
    }
}

}  // namespace

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
        cluster.write_core(
            &kernel_profiler::kLinkSyncCtlStop, sizeof(uint32_t), tt_cxy_pair(r.chip_a, r.virt_a), r.stop_a);
        if (resident) {
            poll_done(r.chip_a, r.virt_a, r.stop_a + 4, "sender");
        }
        const StopDiag da = read_stop_diag(cluster, r.chip_a, r.virt_a, r.stop_a);
        // Now the receiver sees no further frame; a resident one exits on its stop word.
        if (resident) {
            cluster.write_core(
                &kernel_profiler::kLinkSyncCtlStop, sizeof(uint32_t), tt_cxy_pair(r.chip_b, r.virt_b), r.stop_b);
            poll_done(r.chip_b, r.virt_b, r.stop_b + 4, "receiver");
        }
        log_link_diag(r.chip_a, r.chip_b, da, read_stop_diag(cluster, r.chip_b, r.virt_b, r.stop_b));
    }
    link_syncs_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
