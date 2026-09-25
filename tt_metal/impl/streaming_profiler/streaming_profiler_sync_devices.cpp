// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <x86intrin.h>

#include <fmt/format.h>
#include <numa.h>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/types/io_window_config.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include <tt-metalium/allocator.hpp>
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
#include "impl/streaming_profiler/streaming_profiler_tile_clocks.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

// The PCIe tile's SII register block in its own address space, and the count-from-reset timer inside it
// (Blackhole PCIE_SS spec, tables 4 and 12).
constexpr uint64_t kSiiBase = 0xFFFFFFFFF0000000ull;
constexpr uint32_t kCfrLo = 0xA8, kCfrHi = 0xAC;
constexpr uint32_t kBurstReads = 1000;
constexpr size_t kWindowBursts = 10;
constexpr uint32_t kLinkCtl = offsetof(kernel_profiler::LinkSyncL1, ctl);
constexpr uint32_t kLinkDone = offsetof(kernel_profiler::LinkSyncL1, done);
constexpr uint32_t kLinkDiag = offsetof(kernel_profiler::LinkSyncL1, diag);
// The pusher closes a window into a point at least once a millisecond.
constexpr uint32_t kEthPointTicks = kernel_profiler::kEthRefclkHz / 1000;

int64_t clock_ns(clockid_t id) {
    timespec ts{};
    clock_gettime(id, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

// TSC ticks per nanosecond, measured once per process against CLOCK_MONOTONIC_RAW.
double tsc_ticks_per_ns() {
    static const double rate = [] {
        const int64_t t0 = tsc_now(), r0 = clock_ns(CLOCK_MONOTONIC_RAW);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int64_t r1 = clock_ns(CLOCK_MONOTONIC_RAW), t1 = tsc_now();
        return static_cast<double>(t1 - t0) / static_cast<double>(r1 - r0);
    }();
    return rate;
}

CoreCoord eth_virt(const tt::Cluster& cluster, uint32_t chip, const CoreCoord& eth_logical) {
    return cluster.get_virtual_coordinate_from_logical_coordinates(chip, eth_logical, CoreType::ETH);
}

}  // namespace

int64_t tsc_now() noexcept {
    _mm_lfence();
    const int64_t t = static_cast<int64_t>(__rdtsc());
    _mm_lfence();
    return t;
}

int64_t units_per_tsc_q32() {
    static const int64_t q = std::llround(10.0 / tsc_ticks_per_ns() * 4294967296.0);
    return q;
}

double units_per_tsc() { return static_cast<double>(units_per_tsc_q32()) / 4294967296.0; }

int64_t units_of_tsc(int64_t tsc) {
    return static_cast<int64_t>((static_cast<__int128>(tsc) * units_per_tsc_q32() + (__int128{1} << 31)) >> 32);
}

int64_t tsc_of_units(int64_t units) {
    const __int128 q = units_per_tsc_q32();
    return static_cast<int64_t>(((static_cast<__int128>(units) << 32) + q / 2) / q);
}

int64_t steady_mono_ns(int64_t tsc) noexcept {
    int64_t ns = 0;
    if (service().sync().map().steady_ns(tsc, ns)) {
        return ns;
    }
    thread_local const int64_t stand_in_tsc = tsc_now(), stand_in_mono = clock_ns(CLOCK_MONOTONIC);
    return stand_in_mono + std::llround(static_cast<double>(tsc - stand_in_tsc) / tsc_ticks_per_ns());
}

HostProbe::HostProbe(tt::Cluster& cluster, uint32_t chip_id, ClockMap& map) :
    cluster_(cluster), chip_id_(chip_id), map_(map) {
    ticks_per_ns_ = tsc_ticks_per_ns();
    const auto pcie =
        cluster_.get_driver()->get_soc_descriptor(chip_id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    TT_FATAL(!pcie.empty(), "streaming profiler: host probe chip {} has no PCIe tile in its descriptor", chip_id);
    // WC like every other UMD window: a read is uncached under either caching type, and tsc_now()'s fences order it.
    window_ = cluster_.get_driver()->create_io_window(
        chip_id,
        pcie.front(),
        kSiiBase,
        tt::umd::HostIoWindowConfig{.mapping = tt::umd::HostMemoryCaching::WC, .size = kCfrHi + sizeof(uint32_t)});
    // The LO read latches HI; the host is this instance's only reader, so the pair is consistent.
    cfr_lo_last_ = window_->read32(kCfrLo);
    cfr_hi_ = window_->read32(kCfrHi);
    map_.set_bases(static_cast<int64_t>((static_cast<uint64_t>(cfr_hi_) << 32) | cfr_lo_last_), tsc_now());
    thread_ = std::thread(&HostProbe::run, this);
}

HostProbe::~HostProbe() { stop(); }

void HostProbe::stop() {
    {
        std::lock_guard lock(stop_mutex_);
        stop_ = true;
    }
    stop_cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
    window_.reset();  // the mapping must not outlive the device
}

HostProbe::BurstPoint HostProbe::burst() {
    struct Read {
        int64_t mid, rtt;
        uint64_t refclk;
    };
    std::vector<Read> reads;
    reads.reserve(kBurstReads);
    for (uint32_t i = 0; i < kBurstReads; i++) {
        const int64_t t0 = tsc_now();
        const uint32_t lo = window_->read32(kCfrLo);
        const int64_t t1 = tsc_now();
        if (lo < cfr_lo_last_) {
            cfr_hi_++;
        }
        cfr_lo_last_ = lo;
        reads.push_back(Read{t0 + (t1 - t0) / 2, t1 - t0, (static_cast<uint64_t>(cfr_hi_) << 32) | lo});
    }
    // The faster half of the reads carry the least queueing on either leg.
    std::vector<int64_t> rtts;
    rtts.reserve(reads.size());
    for (const Read& r : reads) {
        rtts.push_back(r.rtt);
    }
    rtt_floor_ = std::min(rtt_floor_, *std::min_element(rtts.begin(), rtts.end()));
    std::nth_element(rtts.begin(), rtts.begin() + rtts.size() / 2, rtts.end());
    const int64_t cut = rtts[rtts.size() / 2];
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
    return BurstPoint{
        static_cast<double>(tsc_ref - map_.tsc_base()) + st / n,
        static_cast<double>(static_cast<int64_t>(ref_ref) - map_.root_base()) + sr / n};
}

// Least squares of the burst points, centred on their means.
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
        l.b = srt / srr;
        l.a = mt - l.b * mr;
        double rss = 0.0;
        for (const BurstPoint& p : points_) {
            const double e = p.tsc - l.tsc_of(p.refclk);
            rss += e * e;
        }
        l.sigma_ns = n > 2 ? std::sqrt(rss / (n - 2)) / ticks_per_ns_ : 0.0;
    }
    line_ = l;
}

// CLOCK_MONOTONIC is slewed, never stepped, so the series is the pairs themselves, linear between them.
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
    ns_per_tick_ = pair_tsc_ != 0 && best_tsc > pair_tsc_
                       ? static_cast<double>(best_mono - pair_mono_) / static_cast<double>(best_tsc - pair_tsc_)
                       : 1.0 / ticks_per_ns_;
    map_.append_steady(best_tsc, best_mono, ns_per_tick_);
    pair_tsc_ = best_tsc;
    pair_mono_ = best_mono;
}

void HostProbe::run() {
    set_os_thread_name("sp-hostprobe");
    // The read's round trip is 90 ns longer from the other socket, and the extra sits on one leg, so a thread that
    // wanders between sockets shifts the bracket's midpoint by tens of ns burst to burst.
    if (numa_available() != -1) {
        numa_run_on_node(static_cast<int>(cluster_.get_numa_node_for_device(chip_id_)));
    }
    // A burst every 100 ms from the start, a cadence the refclk period's ramp on the TSC (~0.05 ppm/s) keeps the line's
    // prediction within tens of ns of.
    const auto t_start = std::chrono::steady_clock::now();
    const auto take = [&] {
        const BurstPoint p = burst();
        points_.push_back(p);
        while (points_.size() > kWindowBursts) {
            points_.pop_front();
        }
        refit();
        bursts_++;
        // A node per burst: the line as it stands, at the burst's own instant. Records placed between two
        // bursts run on the newer node's tangent; those placed later interpolate between the nodes.
        if (line_.bursts >= 3) {
            map_.append_host(HostNode{.at = p.refclk, .value = line_.tsc_of(p.refclk), .tangent = line_.b});
        }
    };
    std::unique_lock lock(stop_mutex_);
    for (int64_t k = 1; !stop_; k++) {
        take();
        steady_pair();
        stop_cv_.wait_until(lock, t_start + std::chrono::milliseconds(100 * k), [&] { return stop_; });
    }
    lock.unlock();
    // A capture shorter than three bursts would end without a line: the bursts still owed come now, and the last
    // spans the capture with the first.
    while (points_.size() < 3) {
        take();
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] host probe chip {}: {} bursts, {} of {} reads kept, read round trip floor {:.0f} ns; "
        "final refclk period {:.6f} ns, residual {:.1f} ns; steady_clock {:.9f} ns per tick",
        chip_id_,
        bursts_,
        kept_,
        reads_,
        static_cast<double>(rtt_floor_) / ticks_per_ns_,
        line_.b / ticks_per_ns_,
        line_.sigma_ns,
        ns_per_tick_);
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

uint32_t l1_addr(const Hal& hal) {
    return hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) +
           hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) -
           kernel_profiler::kLinkSyncL1Bytes;
}

}  // namespace link_sync

std::vector<CoreCoord> sorted_yx(const std::unordered_set<CoreCoord>& cores) {
    std::vector<CoreCoord> out(cores.begin(), cores.end());
    std::sort(out.begin(), out.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y != b.y ? a.y < b.y : a.x < b.x;
    });
    return out;
}

KernelHandle create_pusher_kernel(Program& program, const EthL1& l1, const CoreCoords& core, const CoreCoord& arc) {
    return CreateKernel(
        program,
        "tt_metal/tools/profiler/sync/eth_clock_pusher.cpp",
        core.logical,
        EthernetConfig{
            .eth_mode = Eth::IDLE,
            .noc = NOC::RISCV_0_default,
            .processor = DataMovementProcessor::RISCV_0,
            .compile_args = {kEthPointTicks, l1.ctrl, l1.pll, l1.sync_ring, packed_xy(arc)}});
}

KernelHandle create_drainer_kernel(
    Program& program, const EthL1& l1, const CoreCoords& core, const CoreCoords& pusher) {
    return CreateKernel(
        program,
        "tt_metal/tools/profiler/sync/eth_clock_drainer.cpp",
        core.logical,
        EthernetConfig{
            .eth_mode = Eth::IDLE,
            .noc = NOC::RISCV_0_default,
            .processor = DataMovementProcessor::RISCV_0,
            .compile_args = {
                l1.cfg,
                l1.sync_cfg,
                l1.stage,
                l1.ctrl,
                l1.scratch,
                packed_xy(pusher.virt),
                l1.ctrl,
                l1.sync_ring,
                l1.link_ring}});
}

SyncDevices::SyncDevices(ContextId context_id) : context_id_(context_id) {}

SyncDevices::~SyncDevices() = default;

uint32_t SyncDevices::add_device(Device d) {
    devices_.push_back(std::move(d));
    return static_cast<uint32_t>(devices_.size() - 1);
}

void SyncDevices::truncate(uint32_t n) {
    if (devices_.size() > n) {
        devices_.resize(n);
    }
}

void SyncDevices::start_probe() {
    host_probe_ = std::make_unique<HostProbe>(
        MetalContext::instance(context_id_).get_cluster(), devices_.front().chip_id, service().sync().map());
}

void SyncDevices::stop(tt::Cluster& cluster) {
    if (host_probe_) {
        host_probe_->stop();
    }
    stop_links(cluster);
}

void SyncDevices::measure_tiles(uint32_t di, CaptureContext::Device& cap) {
    const Device& d = devices_[di];
    const TileClocks* clocks = service().tile_clocks(d.chip_id);
    TT_FATAL(clocks != nullptr, "streaming profiler: device {} has no tile clocks", d.chip_id);
    const auto offset_of = [&](CoreType type, const CoreCoord& logical) {
        const TileClock* t = clocks->find(type, logical);
        TT_FATAL(
            t != nullptr,
            "streaming profiler: device {} tile ({},{}) is not in its tile clocks",
            d.chip_id,
            logical.x,
            logical.y);
        return t->offset;
    };
    const int64_t pusher = offset_of(CoreType::ETH, d.pusher.logical);
    size_t i = 0;
    const auto place = [&](CoreType type, const CoreCoord& logical) {
        cap.tile_offset[i++] = pusher - offset_of(type, logical);
    };
    for (const CoreCoords& c : d.tensix) {
        place(CoreType::WORKER, c.logical);
    }
    cap.tile_offset[i++] = 0;
    for (const CoreCoords& c : d.linked) {
        place(CoreType::ETH, c.logical);
    }
    TT_FATAL(i == cap.tile_offset.size(), "streaming profiler: device {} roster and tile offsets disagree", d.chip_id);
    cap.drainer_offset = pusher - offset_of(CoreType::ETH, d.drainer.logical);
}

void SyncDevices::plan_links() {
    auto& mc = MetalContext::instance(context_id_);
    fabric_link_sync_ = mc.get_fabric_config() != tt_fabric::FabricConfig::DISABLED;
    if (fabric_link_sync_) {  // TEMP: the chip -> fabric node map for the global-timeline CCL check
        const auto& cp = mc.get_control_plane();
        for (const Device& d : devices_) {
            const auto fn = cp.get_fabric_node_id_from_physical_chip_id(d.chip_id);
            const auto shape = cp.get_physical_mesh_shape(fn.mesh_id);
            log_info(
                tt::LogMetal,
                "[streaming profiler] TEMP FABRICMAP chip {} mesh {} fabric_chip {} shape {}x{}",
                d.chip_id,
                *fn.mesh_id,
                fn.chip_id,
                shape[0],
                shape.dims() > 1 ? shape[1] : 1);
        }
    }
    if (!link_sync::enabled()) {
        log_info(tt::LogMetal, "[streaming profiler] link sync left out (TT_METAL_STREAMING_PROFILER_LINK_SYNC=0)");
        return;
    }
    auto& cluster = mc.get_cluster();
    // An end's core as the decoder numbers it: the roster is the workers, the pusher, then its linked cores.
    const auto core_of = [&](size_t di, const CoreCoord& eth) {
        const Device& d = devices_[di];
        const auto it =
            std::find_if(d.linked.begin(), d.linked.end(), [&](const CoreCoords& c) { return c.logical == eth; });
        TT_FATAL(
            it != d.linked.end(),
            "streaming profiler: device {} link sync end eth({},{}) is not among the pusher's linked cores",
            d.chip_id,
            eth.x,
            eth.y);
        return static_cast<uint32_t>(d.tensix.size() + 1 + (it - d.linked.begin()));
    };
    for (size_t a = 0; a < devices_.size(); a++) {
        const uint32_t chip_a = devices_[a].chip_id;
        for (size_t b = a + 1; b < devices_.size(); b++) {
            const uint32_t chip_b = devices_[b].chip_id;
            for (const link_sync::Link& link : link_sync::links_between(cluster, chip_a, chip_b)) {
                const bool flip = link.chip_a != chip_a;  // the lower chip sends
                const size_t dev_a = flip ? b : a, dev_b = flip ? a : b;
                links_.push_back(CaptureContext::Link{
                    .dev_a = static_cast<uint32_t>(dev_a),
                    .dev_b = static_cast<uint32_t>(dev_b),
                    .chip_a = link.chip_a,
                    .chip_b = link.chip_b,
                    .core_a = core_of(dev_a, link.eth_a),
                    .core_b = core_of(dev_b, link.eth_b),
                    .eth_a = link.eth_a,
                    .eth_b = link.eth_b});
            }
        }
    }
    if (links_.empty() && devices_.size() > 1) {
        log_warning(tt::LogMetal, "[streaming profiler] link sync: no eth connection between the local devices");
    }
}

std::pair<std::unique_ptr<Program>, std::unique_ptr<Program>> SyncDevices::launch_link_ends(
    const CaptureContext::Link& L) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    IDevice* dev_a = devices_[L.dev_a].device;
    IDevice* dev_b = devices_[L.dev_b].device;
    const uint32_t zero[2] = {0, 0};
    static_assert(kLinkDone == kLinkCtl + sizeof(uint32_t));
    const tt_cxy_pair end_a(L.chip_a, eth_virt(cluster, L.chip_a, L.eth_a));
    const tt_cxy_pair end_b(L.chip_b, eth_virt(cluster, L.chip_b, L.eth_b));
    cluster.write_core(zero, sizeof(zero), end_a, *link_l1_ + kLinkCtl);
    cluster.write_core(zero, sizeof(zero), end_b, *link_l1_ + kLinkCtl);
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
    SetRuntimeArgs(*ps, kid_s, L.eth_a, {*link_l1_});
    SetRuntimeArgs(*pr, kid_r, L.eth_b, {*link_l1_});
    detail::CompileProgram(dev_a, *ps, /*force_slow_dispatch=*/true);
    detail::CompileProgram(dev_b, *pr, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(dev_a, *ps, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(dev_b, *pr, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(dev_a, *ps, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(dev_b, *pr, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    return {std::move(ps), std::move(pr)};
}

void SyncDevices::launch_links() {
    auto& mc = MetalContext::instance(context_id_);
    auto& cluster = mc.get_cluster();
    link_l1_ = link_sync::l1_addr(mc.hal());
    for (const CaptureContext::Link& L : links_) {
        // With fabric the routers on this link run the two ends (fabric_erisc_router.cpp, LINK_SYNC_ROLE): nothing
        // to launch, the sender waits for the run word, and their diagnostics are read where the resident kernels
        // leave theirs.
        if (!fabric_link_sync_) {
            resident_.push_back(launch_link_ends(L));
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
    }
    // Every planned end is in place (launched here, or a router that has been waiting): let the senders go.
    for (const CaptureContext::Link& L : links_) {
        cluster.write_core(
            &kernel_profiler::kLinkSyncCtlRun,
            sizeof(uint32_t),
            tt_cxy_pair(L.chip_a, eth_virt(cluster, L.chip_a, L.eth_a)),
            *link_l1_ + kLinkCtl);
    }
}

namespace {

using kernel_profiler::LinkSyncDiag;

LinkSyncDiag read_link_diag(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint32_t link_l1) {
    LinkSyncDiag d{};
    cluster.read_core(&d, sizeof(d), tt_cxy_pair(chip, virt), link_l1 + kLinkDiag);
    return d;
}

void log_link_diag(uint32_t chip_a, uint32_t chip_b, const LinkSyncDiag& da, const LinkSyncDiag& db) {
    for (const auto& [chip, name, d] : {std::tuple{chip_a, "sender", &da}, std::tuple{chip_b, "receiver", &db}}) {
        if (d->rounds_lost != 0 || d->bursts_mismatched != 0 || d->frames_unstamped != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] link sync chip {} {}: left out: {} rounds not recorded, {} bursts whose ingress "
                "stamps did not match their frames, {} frames without an egress stamp",
                chip,
                name,
                d->rounds_lost,
                d->bursts_mismatched,
                d->frames_unstamped);
        }
        if (d->timer == kernel_profiler::kLinkSyncTimerNoRate) {
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
    if (!link_l1_) {
        return;
    }
    const uint32_t link_l1 = *link_l1_;
    const bool resident = !resident_.empty();
    for (const CaptureContext::Link& L : links_) {
        const CoreCoord virt_a = eth_virt(cluster, L.chip_a, L.eth_a);
        const CoreCoord virt_b = eth_virt(cluster, L.chip_b, L.eth_b);
        // Sender first: its current round still completes off the live receiver, then it stops between rounds; a
        // resident sender exits, a router-hosted one goes quiet.
        cluster.write_core(
            &kernel_profiler::kLinkSyncCtlStop, sizeof(uint32_t), tt_cxy_pair(L.chip_a, virt_a), link_l1 + kLinkCtl);
        if (resident) {
            poll_done(L.chip_a, virt_a, link_l1 + kLinkDone, "sender");
        }
        const LinkSyncDiag da = read_link_diag(cluster, L.chip_a, virt_a, link_l1);
        // Now the receiver sees no further frame; a resident one exits on its stop word.
        if (resident) {
            cluster.write_core(
                &kernel_profiler::kLinkSyncCtlStop,
                sizeof(uint32_t),
                tt_cxy_pair(L.chip_b, virt_b),
                link_l1 + kLinkCtl);
            poll_done(L.chip_b, virt_b, link_l1 + kLinkDone, "receiver");
        }
        log_link_diag(L.chip_a, L.chip_b, da, read_link_diag(cluster, L.chip_b, virt_b, link_l1));
    }
    resident_.clear();
    link_l1_.reset();
}

}  // namespace tt::tt_metal::streaming_profiler
