// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_device.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <set>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tracy/Tracy.hpp>
#include <tracy/TracyTTDevice.hpp>

#include <chrono>
#include <x86intrin.h>
#include <thread>

#include "impl/dispatch/dispatch_core_manager.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>  // MeshCoreCoord
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/types/tlb.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig (a DRISC kernel is not in the public headers yet)
#include "jit_build/build_env_manager.hpp"
#include "llrt/tt_cluster.hpp"
#include "hostdev/streaming_profiler_common.h"

#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_receiver.hpp"
#include "impl/streaming_profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

namespace pz = tt::tt_metal::profiler;

namespace {

// Views 7 and 2 last: a roster truncated to fewer relays sheds the bring-up-fragile views first.
constexpr std::array<uint32_t, 8> kRelayBankRoster = {5u, 6u, 4u, 1u, 0u, 3u, 7u, 2u};

// Staging slots per relay, capped at what a DRISC's L1 fits.
constexpr uint32_t kMaxStageSlots = 7;

}  // namespace

// Host<->device clock sync: the Tensix wall clock is read over NoC bracketed by host clock reads (Cristian's
// algorithm; the midpoint cancels the round trip to first order). Without it device zones anchor at the time
// the first marker was consumed and lag the host zones by the drain latency.
struct StreamingProfilerSync {
    double frequency = 0.0;  // device cycles per nanosecond (GHz)
    uint64_t device_at_anchor = 0;
    int64_t host_anchor = 0;
    bool valid = false;
};

// spacing_us lengthens the regression baseline: back-to-back samples span only ~360 us and the fitted
// frequency then carries ~1e-4 of error, which grows with time since the anchor.
StreamingProfilerSync sync_device_clock(
    tt::Cluster& cluster, uint32_t chip_id, const CoreCoord& worker, uint32_t spacing_us = 0) {
    // RISCV_DEBUG_REG_WALL_CLOCK_L/H are Tensix debug registers by spec, but a DRAM tile answers them too, which
    // is what allows a per-relay anchor.
    constexpr uint64_t kWallClockL = 0xFFB121F0ULL;
    constexpr uint64_t kWallClockH = 0xFFB121F8ULL;
    constexpr uint32_t kSamples = 100;
    struct S {
        int64_t host_mid;
        uint64_t dev;
        int64_t rt;
    };
    std::vector<S> samples;
    samples.reserve(kSamples);
    const tt_cxy_pair target(chip_id, worker);
    for (uint32_t i = 0; i < kSamples; i++) {
        uint32_t lo = 0, hi = 0;
        const int64_t t0 = tracy::Profiler::GetTime();
        cluster.read_reg(&lo, target, kWallClockL);  // reading L latches H
        cluster.read_reg(&hi, target, kWallClockH);
        const int64_t t1 = tracy::Profiler::GetTime();
        samples.push_back(S{(t0 + t1) / 2, (static_cast<uint64_t>(hi) << 32) | lo, t1 - t0});
        if (spacing_us != 0 && i + 1 < kSamples) {
            std::this_thread::sleep_for(std::chrono::microseconds(spacing_us));
        }
    }
    // Drop NoC/PCIe-contended outliers: keep samples whose round-trip is within 1.5x the median.
    std::vector<int64_t> rts;
    rts.reserve(samples.size());
    for (const auto& s : samples) {
        rts.push_back(s.rt);
    }
    std::sort(rts.begin(), rts.end());
    const int64_t rt_cut = rts[rts.size() / 2] + rts[rts.size() / 2] / 2;

    // Centered least squares: centering avoids cancellation at absolute-timestamp magnitudes.
    double hx = 0, dy = 0;
    uint32_t n = 0;
    for (const auto& s : samples) {
        if (s.rt > rt_cut) {
            continue;
        }
        hx += static_cast<double>(s.host_mid);
        dy += static_cast<double>(s.dev);
        n++;
    }
    StreamingProfilerSync out;
    if (n < 2) {
        return out;
    }
    hx /= n;
    dy /= n;
    double num = 0, den = 0;
    for (const auto& s : samples) {
        if (s.rt > rt_cut) {
            continue;
        }
        const double ddx = static_cast<double>(s.host_mid) - hx;
        const double ddy = static_cast<double>(s.dev) - dy;
        num += ddx * ddy;
        den += ddx * ddx;
    }
    if (std::abs(den) < 1e-10) {
        return out;
    }
    const double slope = num / den;  // device cycles per host tick
#ifdef TRACY_ENABLE
    const double ns_per_tick = TracyGetTimerMul();
#else
    const double ns_per_tick = 1.0;
#endif
    out.frequency = slope / (ns_per_tick > 0.0 ? ns_per_tick : 1.0);  // cycles per ns
    // Anchor on the sample mean: extrapolating an intercept to host_time=0 turns a tiny slope error into a huge
    // offset.
    out.host_anchor = static_cast<int64_t>(hx);
    out.device_at_anchor = static_cast<uint64_t>(dy);
    out.valid = out.frequency > 0.0;
    return out;
}

// Bring-up runs several MMIO paths and a hang in any of them reports only "MMIO per-op timeout"; this names
// the stall site.
thread_local std::string g_bringup_step = "(not started)";

Devices::DeviceCtx::DeviceCtx() = default;
Devices::DeviceCtx::~DeviceCtx() = default;
Devices::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;

const std::string& bringup_step() { return g_bringup_step; }

Devices::~Devices() = default;

std::vector<ReceiverDeviceConfig> Devices::boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& rtopts = MetalContext::instance(context_id).rtoptions();

    if (cluster.arch() != tt::ARCH::BLACKHOLE) {
        log_debug(tt::LogMetal, "[streaming profiler] not Blackhole; skipping relay capture.");
        return {};
    }

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx ctx;
        ctx.chip_id = static_cast<uint32_t>(mesh_device->get_device(coord)->id());
        if (!boot_device(mesh_device, ctx, coord)) {
            continue;  // boot logs its own reason; degrade to no-capture for this device
        }
        double freq = cluster.get_device_aiclk(ctx.chip_id) / 1000.0;
        if (freq <= 0.0) {
            freq = 1.0;
        }
        StreamingProfilerSync sync;
        if (!ctx.core_virt.empty()) {
            const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
            // 500 us spacing spans ~50 ms of baseline instead of ~360 us; this is the one frequency every context on
            // the chip uses.
            sync = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/500);
        }
        // The sync anchors against Tracy's timer; the public clock is std::chrono::steady_clock, so pin the two
        // together here, once, and express the anchor in steady_clock terms.
        const int64_t steady_now =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        const int64_t tracy_now = tracy::Profiler::GetTime();
        if (sync.valid) {
            ctx.clock_synced = true;
            ctx.freq_ghz = sync.frequency;
            ctx.anchor_ticks = sync.device_at_anchor;
            ctx.anchor_host_ns =
                steady_now -
                static_cast<int64_t>(static_cast<double>(tracy_now - sync.host_anchor) * TracyGetTimerMul());
        } else {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {} clock sync FAILED; falling back to first-marker anchoring "
                "(device zones will lag the host zones by the drain latency)",
                ctx.chip_id);
            ctx.freq_ghz = freq;
            ctx.anchor_ticks = 0;
            ctx.anchor_host_ns = steady_now;
        }
        ctx.active = true;
        devices_.push_back(std::move(ctx));
    }

    // After devices_ is stable: socket ownership moves into the receiver, and the lane tables are flattened so
    // no consumer does a per-record hash lookup.
    std::vector<ReceiverDeviceConfig> rdevs;
    if (!devices_.empty()) {
        for (auto& ctx : devices_) {
            auto& rd = rdevs.emplace_back();
            rd.chip_id = ctx.chip_id;
            rd.num_cores = ctx.nl / kNRisc;
            rd.core_of_xy = ctx.core_of_xy;
            rd.clock = {
                .frequency_ghz = ctx.freq_ghz,
                .anchor_ticks = ctx.anchor_ticks,
                .anchor_host_ns = ctx.anchor_host_ns,
                .chip_id = ctx.chip_id,
                .measured = ctx.clock_synced};
            rd.lane_table.reserve(ctx.nl);
            for (uint32_t ci = 0; ci < rd.num_cores; ci++) {
                const auto [lx, ly] = ctx.core_logical[ci];
                const auto [px, py] = ctx.core_physical[ci];
                for (uint32_t r = 0; r < kNRisc; r++) {
                    rd.lane_table.push_back(streaming_profiler::LaneInfo{
                        ctx.chip_id,
                        static_cast<uint16_t>(lx),
                        static_cast<uint16_t>(ly),
                        static_cast<uint16_t>(px),
                        static_cast<uint16_t>(py),
                        static_cast<uint8_t>(r)});
                }
            }
            for (uint32_t sk = 0; sk < ctx.n_drisc; sk++) {
                if (ctx.sockets[sk] != nullptr) {
                    TT_FATAL(sk == rd.sockets.size(), "sockets must form a contiguous prefix");
                    rd.sockets.push_back(std::move(ctx.sockets[sk]));
                }
            }
        }
    }
    if (!devices_.empty()) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] active on {} device(s){}",
            devices_.size(),
            rtopts.get_streaming_profiler_tracy_enabled() ? " with the Tracy sink" : "");
    }
    return rdevs;
}

// Stream mode (1) or NOC2AXI (0) for every relay's NIU, in one launch, run to completion. D2HSocket
// construction writes its config into DRISC L1 from the host, which only lands once the NIU terminates
// inbound traffic at L1. One launch: every LaunchProgram carries a dram_barrier that MMIO-polls a core in
// every DRAM channel, and a barrier that reaches a core already in stream mode never completes.
void Devices::set_drisc_niu_mode(IDevice* device, const std::vector<CoreCoord>& drisc_logicals, uint32_t stream) {
    if (drisc_logicals.empty()) {
        return;
    }
    std::set<CoreRange> ranges;
    for (const auto& c : drisc_logicals) {
        ranges.insert(CoreRange(c, c));
    }
    Program p = CreateProgram();
    CreateKernel(
        p,
        "tt_metal/tools/profiler/kernels/drisc_niu_mode.cpp",
        CoreRangeSet(ranges),
        DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
    const std::string who = fmt::format("niu-mode[{} cores]->{}", drisc_logicals.size(), stream);
    g_bringup_step = who + ":CompileProgram";
    detail::CompileProgram(device, p, /*force_slow_dispatch=*/true);
    g_bringup_step = who + ":WriteRuntimeArgs";
    detail::WriteRuntimeArgsToDevice(device, p, /*force_slow_dispatch=*/true);
    // Launch and wait split so a failure names which half stalled; a stall on the first label means a core was
    // already in stream mode when this run began.
    g_bringup_step = who + ":LaunchProgram(dram_barrier,no-wait)";
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    g_bringup_step = who + ":WaitProgramDone(poll-after-flip)";
    detail::WaitProgramDone(device, p);
    g_bringup_step = who + ":done";
}

// Producers boot unarmed (enumerate_worker_grid() clears PROFILER_ARMED on every Tensix core of the device) and
// only block on a full ring once armed here, so a core no relay drains can never wedge device close. Arming follows the
// relays coming up; a relay that fails leaves the whole device unarmed and its markers are overwritten instead.
void Devices::arm_producers(DeviceCtx& ctx) {
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const size_t n = ctx.core_virt.size();
    uint32_t one = 1;
    for (size_t ci = 0; ci < n; ci++) {
        const auto [vx, vy] = ctx.core_virt[ci];
        cluster.write_core(
            &one,
            sizeof(uint32_t),
            tt_cxy_pair(ctx.chip_id, CoreCoord{vx, vy}),
            prof_l1 + kernel_profiler::PROFILER_ARMED * sizeof(uint32_t));
    }
}

void Devices::report_unarmed(uint32_t device_id) {
    log_warning(
        tt::LogMetal,
        "[streaming profiler] Device {}: no DRISC relay -- producers stay unarmed (markers are DROPPED, but the "
        "workload will not stall waiting for a consumer)",
        device_id);
}

// Head is relay-written and tail producer-written, so head == tail on every RISC means everything published
// was consumed. Must precede the quiesce: dispatch cores emit zones through device close and would park in
// ring_ensure_room against stopped relays.
bool Devices::wait_producer_rings_drained(DeviceCtx& ctx, std::chrono::milliseconds budget) {
    if (ctx.core_virt.empty()) {
        return true;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const size_t n = ctx.core_virt.size();
    std::vector<uint8_t> drained(n, 0);
    std::vector<uint32_t> ht(2 * kernel_profiler::PROFILER_SPSC_MAX_RISC, 0);
    const auto dl = std::chrono::steady_clock::now() + budget;
    size_t pending = n;
    while (pending != 0 && std::chrono::steady_clock::now() < dl) {
        pending = 0;
        for (size_t ci = 0; ci < n; ci++) {
            if (drained[ci] != 0) {
                continue;
            }
            const auto [vx, vy] = ctx.core_virt[ci];
            cluster.read_core(
                ht.data(),
                static_cast<uint32_t>(ht.size() * sizeof(uint32_t)),
                tt_cxy_pair(ctx.chip_id, CoreCoord{vx, vy}),
                prof_l1);
            bool empty = true;
            for (uint32_t r = 0; r < kNRisc; r++) {
                if (ht[kernel_profiler::SPSC_RING_HEAD_0 + r] != ht[kernel_profiler::SPSC_RING_TAIL_0 + r]) {
                    empty = false;
                    break;
                }
            }
            drained[ci] = empty ? 1u : 0u;
            pending += empty ? 0u : 1u;
        }
    }
    return pending == 0;
}

// The only path that drops a marker: a producer still publishing after the drain budget, unblocked so device
// close does not wedge in wait_until_cores_done().
void Devices::disarm_producer_backpressure(DeviceCtx& ctx) {
    if (ctx.core_virt.empty()) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const size_t n = ctx.core_virt.size();
    uint32_t zero = 0;
    for (size_t ci = 0; ci < n; ci++) {
        const auto [vx, vy] = ctx.core_virt[ci];
        cluster.write_core(
            &zero,
            sizeof(uint32_t),
            tt_cxy_pair(ctx.chip_id, CoreCoord{vx, vy}),
            prof_l1 + kernel_profiler::PROFILER_ARMED * sizeof(uint32_t));
    }
}

namespace {

// A static TLB window skips UMD's per-access reconfigure on the socket's ack write (171 vs 382 ns). Metal
// maps a window per DRAM channel only on the channel's preferred worker endpoint port (configure_static_tlbs
// -> ddr_to_noc0) and the relay sits on the unused port, so it maps its own: 2 MB at address 0 spans the whole
// 128 KB DRISC L1. Best-effort: windows are finite, and losing the race costs only the ~210 ns.
void configure_relay_static_tlb(tt::Cluster& cluster, uint32_t device_id, const CoreCoord& drisc_virtual, uint32_t d) {
    if (cluster.is_mock_or_emulated()) {
        return;
    }
    auto* tlb_manager = cluster.get_driver()->get_chip(device_id)->get_tlb_manager();
    const tt_xy_pair tlb_core(drisc_virtual.x, drisc_virtual.y);
    if (tlb_manager->is_tlb_mapped(tlb_core)) {
        return;
    }
    try {
        g_bringup_step = fmt::format("relay {}: configure static TLB", d);
        tlb_manager->configure_tlb(tlb_core, /*tlb_size=*/2 * 1024 * 1024, /*address=*/0, tt::umd::tlb_data::Strict);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] could not configure a static TLB for DRISC core ({}, {}): {} "
            "-- the socket ack write stays on the dynamic path",
            tlb_core.x,
            tlb_core.y,
            e.what());
    }
}

// A resident relay launches fire-and-forget, so a core that never leaves reset produces no error and the
// workload wedges on full rings; the heartbeat must leave 0 and then advance.
bool relay_heartbeat_advanced(
    tt::Cluster& cluster,
    uint32_t device_id,
    const CoreCoord& drisc_virtual,
    uint64_t hb_addr,
    uint64_t stop_addr,
    uint32_t d) {
    const tt_cxy_pair core(device_id, drisc_virtual);
    uint32_t hb0 = 0, hb1 = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        cluster.read_core(&hb0, sizeof(hb0), core, hb_addr);
        if (hb0 != 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // A single sample cannot tell a dead relay from a slow one; 200 ms is ~6000 idle sweeps.
    if (hb0 != 0) {
        const auto adv_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        do {
            cluster.read_core(&hb1, sizeof(hb1), core, hb_addr);
            if (hb1 != hb0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } while (std::chrono::steady_clock::now() < adv_deadline);
    }
    if (hb0 != 0 && hb1 != hb0) {
        return true;
    }
    uint32_t stopw = 0;
    cluster.read_core(&stopw, sizeof(stopw), core, stop_addr);
    log_warning(
        tt::LogMetal,
        "[streaming profiler] Device {}: relay {} FAILED TO START (heartbeat {} -> {} after "
        "launch, stop word {}). The producers would block forever on a full ring and wedge the "
        "workload, so capture is disabled for this run instead.",
        device_id,
        d,
        hb0,
        hb1,
        stopw);
    return false;
}

}  // namespace

// Core identity is not on the wire: the producing core writes it into SPSC_CORE_XY and these maps resolve it.
void Devices::enumerate_worker_grid(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx, BootPlan& plan) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t device_id = ctx.chip_id;

    plan.prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    // The poll list defines the drained set; a producer outside it fills its ring, blocks forever, and takes the
    // host down in wait_until_cores_done. The relay lives on a DRAM core, so the full grid is polled.
    const uint32_t gx = static_cast<uint32_t>(grid.x);
    const uint32_t gy = static_cast<uint32_t>(grid.y);
    plan.num_cores = static_cast<uint64_t>(gx) * gy;
    ctx.nl = static_cast<uint32_t>(plan.num_cores) * kNRisc;
    ctx.core_logical.resize(plan.num_cores);
    ctx.core_physical.resize(plan.num_cores);
    ctx.core_virt.resize(plan.num_cores);
    plan.coords.assign(plan.num_cores, 0);
    plan.zero_ctrl.assign(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);

    // Row-major enumeration is what makes a relay's band a contiguous run of core indices.
    for (uint32_t idx = 0; idx < plan.num_cores; idx++) {
        const uint32_t lx = idx % gx, ly = idx / gx;
        CoreCoord v =
            cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{lx, ly}, CoreType::WORKER);
        const uint32_t vx = static_cast<uint32_t>(v.x), vy = static_cast<uint32_t>(v.y);
        plan.coords[idx] = (vx & 0xFFFFu) | ((vy & 0xFFFFu) << 16);
        ctx.core_of_xy[plan.coords[idx]] = idx;
        cluster.write_core(
            plan.zero_ctrl.data(), (uint32_t)plan.zero_ctrl.size(), tt_cxy_pair(device_id, v), plan.prof_l1);
        ctx.core_logical[idx] = {lx, ly};
        const CoreCoord p = cluster.get_physical_coordinate_from_logical_coordinates(
            device_id, CoreCoord{lx, ly}, CoreType::WORKER, /*no_warn=*/true);
        ctx.core_physical[idx] = {static_cast<uint32_t>(p.x), static_cast<uint32_t>(p.y)};
        ctx.core_virt[idx] = {vx, vy};
    }

    // Producers boot unarmed. Upstream does this in BRISC firmware at boot; this tree keeps hw/firmware byte-for-byte
    // main, so the host does it here, before any relay launches, on every Tensix core of the device and not just the
    // compute grid zeroed above: a dispatch core's ring is drained by nobody and its L1 survives device re-init, so a
    // stale arm there would park its firmware zone in the stall path and wedge wait_until_cores_done() at close.
    // arm_producers() then sets the flag only on the cores the relays serve, once every relay is up.
    {
        const CoreCoord tensix_grid = cluster.get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
        uint32_t zero = 0;
        for (uint32_t ly = 0; ly < static_cast<uint32_t>(tensix_grid.y); ly++) {
            for (uint32_t lx = 0; lx < static_cast<uint32_t>(tensix_grid.x); lx++) {
                const CoreCoord v = cluster.get_virtual_coordinate_from_logical_coordinates(
                    device_id, CoreCoord{lx, ly}, CoreType::WORKER);
                cluster.write_core(
                    &zero,
                    sizeof(uint32_t),
                    tt_cxy_pair(device_id, v),
                    plan.prof_l1 + kernel_profiler::PROFILER_ARMED * sizeof(uint32_t));
            }
        }
    }
}

// Relay count, each relay's DRAM view and core, then every relay's NIU into stream mode. False: no relay can
// run on this device.
bool Devices::choose_relay_banks(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx, BootPlan& plan) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& rtopts = MetalContext::instance(context_id).rtoptions();
    const uint32_t device_id = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(device_id);

    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    if (nbanks == 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: no DRAM views to host a relay -- the streaming profiler is OFF "
            "for this device.",
            device_id);
        return false;
    }

    const uint32_t view_cap = std::min<uint32_t>(kMaxRelays, nbanks);
    static_assert(kMaxRelays == 8, "rtoptions bounds TT_METAL_STREAMING_PROFILER_NRELAYS at 8");
    const uint32_t requested = rtopts.get_streaming_profiler_num_relays();
    if (requested == 0) {
        ctx.n_drisc = view_cap;
    } else if (requested > view_cap) {
        ctx.n_drisc = view_cap;
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: TT_METAL_STREAMING_PROFILER_NRELAYS={} exceeds this part's {} DRAM "
            "views (one relay each); CLAMPED to {} relays",
            device_id,
            requested,
            nbanks,
            ctx.n_drisc);
    } else {
        ctx.n_drisc = requested;
    }

    plan.banks.clear();
    for (const uint32_t b : kRelayBankRoster) {
        if (b < nbanks) {
            plan.banks.push_back(b);
        }
    }
    TT_FATAL(
        plan.banks.size() >= ctx.n_drisc,
        "streaming profiler needs {} relay banks but only {} usable DRAM views are in the roster (part has {} "
        "views)",
        ctx.n_drisc,
        plan.banks.size(),
        nbanks);
    plan.banks.resize(ctx.n_drisc);

    // Picked up front so that every relay's NIU flips in one launch (see set_drisc_niu_mode).
    plan.relay_cores.clear();
    for (uint32_t d = 0; d < ctx.n_drisc; d++) {
        plan.relay_cores.push_back(mesh_device->impl().pick_unused_dram_logical_core(ctx.device, plan.banks[d]));
    }
    // pick_unused_dram_logical_core() reserves per view and cannot see two views resolving to one physical port
    // (views 0 and 7 have both come back as NoC core 0-0); two relays on one L1 would silently overlap, so refuse.
    for (uint32_t a = 0; a < plan.relay_cores.size(); a++) {
        for (uint32_t b = a + 1; b < plan.relay_cores.size(); b++) {
            TT_FATAL(
                plan.relay_cores[a] != plan.relay_cores[b],
                "streaming profiler: DRISC {} (DRAM view {}) and DRISC {} (DRAM view {}) both resolve to logical "
                "DRAM core ({},{}). Two resident relay kernels cannot share a core.",
                a,
                plan.banks[a],
                b,
                plan.banks[b],
                plan.relay_cores[a].x,
                plan.relay_cores[a].y);
        }
    }
    // Cluster::dram_barrier syncs subchannel 0 of every channel and every LaunchProgram carries one; a relay
    // resident there is in stream mode, where a DRAM-range address no longer forwards to GDDR. Reported, not
    // fatal: it usually works, and this is the explanation for a later MMIO timeout.
    std::vector<uint32_t> collide;
    for (int ch = 0; ch < soc.get_num_dram_channels(); ch++) {
        const CoreCoord bar = soc.get_dram_core_for_channel(ch, 0, CoordSystem::LOGICAL);
        for (uint32_t d = 0; d < plan.relay_cores.size(); d++) {
            if (plan.relay_cores[d] == bar) {
                collide.push_back(d);
            }
        }
    }
    if (!collide.empty()) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] {} of {} relays sit on a dram_barrier target core (subchannel 0 "
            "of their channel). Every LaunchProgram barriers those cores while they are in stream "
            "mode; a 60-70 ms MMIO timeout at bring-up or weight upload has this as a candidate.",
            collide.size(),
            plan.relay_cores.size());
    }
    set_drisc_niu_mode(ctx.device, plan.relay_cores, 1);
    return true;
}

// One replicated mesh buffer with one interleaved page per bank reserves the same window in every bank of
// every device. Mesh-level because the lock-step allocator never sees a device-local Buffer::create and
// would hand the region out again.
void Devices::reserve_spool(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx, BootPlan& plan) {
    const auto context_id = mesh_device->impl().get_context_id();
    const auto& rtopts = MetalContext::instance(context_id).rtoptions();
    const uint32_t spool_mb = rtopts.get_streaming_profiler_spool_mb();

    plan.spool_bytes = spool_mb * (1u << 20);
    plan.spool_addr = 0;
    if (plan.spool_bytes != 0 && spool_buffer_ == nullptr) {
        const uint32_t nbanks_dram = ctx.device->allocator()->get_num_banks(BufferType::DRAM);
        try {
            spool_buffer_ = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{static_cast<DeviceAddr>(nbanks_dram) * plan.spool_bytes},
                distributed::DeviceLocalBufferConfig{.page_size = plan.spool_bytes, .buffer_type = BufferType::DRAM},
                mesh_device.get());
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] could not reserve {} MiB/bank of DRAM for the GDDR spool ({}); falling "
                "back to direct push",
                spool_mb,
                e.what());
        }
    }
    if (spool_buffer_ != nullptr) {
        plan.spool_addr = static_cast<uint32_t>(spool_buffer_->address());
    } else {
        plan.spool_bytes = 0;
    }
}

bool Devices::launch_relay(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord,
    const BootPlan& plan,
    uint32_t d) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& hal = MetalContext::instance(context_id).hal();
    const auto& rtopts = MetalContext::instance(context_id).rtoptions();
    const uint32_t device_id = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(device_id);

    // Contiguous bands in the host's grid order: a core belongs to exactly one relay, and the integer prefix
    // split assigns every core once.
    const uint32_t lo = static_cast<uint32_t>((plan.num_cores * d) / ctx.n_drisc);
    const uint32_t hi = static_cast<uint32_t>((plan.num_cores * (d + 1)) / ctx.n_drisc);
    // One 64-byte record per core (landed tails, head mirror, wire XY) and byte-indexed core lists in the kernel.
    constexpr uint32_t kMaxCores = 128;
    constexpr uint32_t kScratchBytes = kMaxCores * 64;
    if (hi - lo > kMaxCores) {
        log_error(
            tt::LogMetal,
            "[streaming profiler] Device {}: relay {} would own {} cores but a relay holds at most {}; set "
            "TT_METAL_STREAMING_PROFILER_NRELAYS to at least {} for this {}-core grid",
            device_id,
            d,
            hi - lo,
            kMaxCores,
            (plan.num_cores + kMaxCores - 1) / kMaxCores,
            plan.num_cores);
        return false;
    }
    const uint32_t my_cores = hi - lo;
    if (my_cores == 0) {
        return true;
    }
    ctx.drisc_logical[d] = mesh_device->impl().pick_unused_dram_logical_core(ctx.device, plan.banks[d]);
    const CoreCoord translated = soc.dram_bank_endpoint_coords.at(ctx.drisc_logical[d].x).at(ctx.drisc_logical[d].y);
    const tt::umd::CoreCoord phys = soc.translate_coord_to(
        tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);
    const CoreCoord drisc_phys{phys.x, phys.y};
    ctx.drisc_virtual[d] = ctx.device->virtual_core_from_logical_core(ctx.drisc_logical[d], CoreType::DRAM);
    ctx.drisc_l1_base[d] = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    ctx.drisc_l1_noc[d] = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    constexpr uint32_t kCfgReserve = 8 * 1024;
    // done(64) + stop(64).
    constexpr uint32_t kMiscBytes = 1024;
    const uint32_t fixed = kCfgReserve + kScratchBytes + kMiscBytes;
    const uint32_t nstage = std::min(region > fixed ? (region - fixed) / plan.slot_bytes : 0u, kMaxStageSlots);
    if (nstage == 0) {
        log_warning(tt::LogMetal, "[streaming profiler] Device {}: DRISC L1 too small; skipping", device_id);
        return false;
    }
    const uint32_t stage_base = ctx.drisc_l1_base[d];
    const uint32_t core_records = stage_base + nstage * plan.slot_bytes;
    ctx.done_addr[d] = core_records + kScratchBytes;
    ctx.stop_addr[d] = ctx.done_addr[d] + kernel_profiler::kRelayCtrlWordStride;
    const uint32_t cfg_l1 = ctx.drisc_l1_base[d] + region - kCfgReserve;
    TT_FATAL(
        ctx.stop_addr[d] + kernel_profiler::kRelayCtrlWordStride <= cfg_l1,
        "DRISC L1 layout overlaps the socket config");

    configure_relay_static_tlb(cluster, device_id, ctx.drisc_virtual[d], d);

    const uint32_t sk = d;
    try {
        g_bringup_step = fmt::format("relay {}: D2HSocket construct (writes config into DRISC L1)", d);
        ctx.sockets[sk] = std::make_unique<distributed::D2HSocket>(
            mesh_device,
            distributed::MeshCoreCoord{coord, CoreCoord(drisc_phys.x, drisc_phys.y)},
            (rtopts.get_streaming_profiler_fifo_mb() << 20) / kPageSize * kPageSize,
            distributed::D2HSocket::ExternalConfigBuffer{
                .address = cfg_l1, .sender_core_type = HalProgrammableCoreType::DRAM});
        ctx.sockets[sk]->set_page_size(kPageSize);

        // Zero the relay core's own profiler ring: the relay is built with PROFILE_KERNEL, firmware writes zone
        // markers into this ring on every launch, nothing drains it, and the SPSC backend blocks on a full ring, so
        // after ~74 launches in one reset window the RISC wedges in firmware init.
        const uint64_t relay_prof_l1 = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER);
        cluster.write_core(
            plan.zero_ctrl.data(),
            (uint32_t)plan.zero_ctrl.size(),
            tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
            relay_prof_l1);

        // A stale done or heartbeat from the previous run reads as this run's live state.
        uint32_t zero3[13] = {};
        cluster.write_core(
            zero3,
            sizeof(zero3),
            tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
            ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
        // Teardown leaves stop at 1 or 2 and the relay loop exits on nonzero stop.
        uint32_t zero4[4] = {};
        cluster.write_core(
            zero4,
            sizeof(zero4),
            tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
            ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));

        ctx.relay_program[d] = std::make_unique<Program>(CreateProgram());
        const std::unordered_map<std::string, uint32_t> cargs = {
            {"stage_base", stage_base},
            {"n_stage", nstage},
            {"core_records", core_records},
            {"done_addr", ctx.done_addr[d]},
            {"stop_addr", ctx.stop_addr[d]},
            {"socket_config_addr", ctx.sockets[sk]->get_config_buffer_address()},
            {"max_cores", kMaxCores},
            // d&2 splits the pushers across two of the four unicast request VCs.
            {"write_vc", (d & 2u) ? 0u : 1u},
            // The bounce slots cost a staging generation, so a smaller L1 falls back to direct push rather than
            // tripping the kernel's geometry static_asserts.
            {"spool_base", plan.spool_addr},
            {"spool_bytes", nstage >= 7u ? plan.spool_bytes : 0u}};
        if (plan.spool_bytes != 0 && nstage < 7u) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: only {} staging slots fit, too few for the spool's bounce "
                "buffers; relay {} runs direct push",
                device_id,
                nstage,
                d);
        }
        TT_FATAL(
            my_cores * 32u <= plan.slot_bytes,
            "CV-first tails staging ({} cores x 32 B) does not fit inside the slot past the pipeline",
            my_cores);
        auto relay_id = CreateKernel(
            *ctx.relay_program[d],
            "tt_metal/tools/profiler/kernels/streaming_profiler_relay.cpp",
            ctx.drisc_logical[d],
            // NOC 1 egress runs ~2x the service interval of NOC 0, so a relay parked there takes essentially every
            // profiler stall.
            DramConfig{
                .noc = NOC::NOC_0, .defines = {{"STREAMING_PROFILER_RELAY_KERNEL", "1"}}, .named_compile_args = cargs});
        std::vector<uint32_t> rt = {my_cores, static_cast<uint32_t>(plan.prof_l1)};
        // Reversed: launch order follows global index, so the slice's last-launched cores land in the first-chunk
        // slots, which are serviced first.
        rt.insert(
            rt.end(),
            plan.coords.rbegin() + (plan.coords.size() - hi),
            plan.coords.rbegin() + (plan.coords.size() - lo));
        SetRuntimeArgs(*ctx.relay_program[d], relay_id, ctx.drisc_logical[d], rt);

        detail::CompileProgram(ctx.device, *ctx.relay_program[d], /*force_slow_dispatch=*/true);
        detail::WriteRuntimeArgsToDevice(ctx.device, *ctx.relay_program[d], /*force_slow_dispatch=*/true);
        g_bringup_step = fmt::format("relay {}: relay kernel LaunchProgram", d);
        detail::LaunchProgram(
            ctx.device, *ctx.relay_program[d], /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

        g_bringup_step = fmt::format("relay {}: heartbeat verify", d);
        const uint64_t hb_addr = ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]) + 4;
        const uint64_t stop_noc = ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]);
        if (!relay_heartbeat_advanced(cluster, device_id, ctx.drisc_virtual[d], hb_addr, stop_noc, d)) {
            ctx.relay_program[d].reset();
            ctx.sockets[sk].reset();
            return false;
        }
    } catch (const std::exception& e) {
        // A code-region overflow fails the load, not the start, and the run then exits 0 with every marker dropped.
        const std::string what = e.what();
        const bool elf_too_big = what.find("overflows region") != std::string::npos;
        log_error(
            tt::LogMetal,
            "[streaming profiler] Device {}: DRISC {} FAILED TO LOAD{} -- THIS CAPTURE WILL BE EMPTY. No "
            "device zones will be produced and the run will still exit 0.{} ({})",
            device_id,
            d,
            elf_too_big ? " (relay kernel ELF EXCEEDS THE DRISC CODE REGION)" : "",
            elf_too_big ? " Reduce relay-kernel code: a u64 division anywhere in the kernel costs a "
                          "956 B soft-div."
                        : "",
            what);
        ctx.relay_program[d].reset();
        ctx.sockets[sk].reset();
        return false;
    }

    return true;
}

bool Devices::boot_device(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord) {
    const auto context_id = mesh_device->impl().get_context_id();
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t device_id = ctx.chip_id;

    // The relay is a DRISC: one DM RISC-V on a DRAM core, which today exists only on Blackhole.
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: no DRAM programmable cores (card FW below the DRISC gate?)",
            device_id);
        report_unarmed(device_id);
        return false;
    }

    BootPlan plan;
    // Mirrors the kernel's kSlotWords.
    plan.slot_bytes = kernel_profiler::spsc_span_slot_words(kNRisc) * sizeof(uint32_t);
    enumerate_worker_grid(mesh_device, ctx, plan);

    ctx.device = mesh_device->get_device(coord);
    if (!choose_relay_banks(mesh_device, ctx, plan)) {
        report_unarmed(device_id);
        return false;
    }
    reserve_spool(mesh_device, ctx, plan);

    for (uint32_t d = 0; d < ctx.n_drisc; d++) {
        if (!launch_relay(mesh_device, ctx, coord, plan, d)) {
            report_unarmed(device_id);
            return false;
        }
    }
    arm_producers(ctx);
    return true;
}

void Devices::quiesce(Receiver* receiver) {
    for (auto& ctx : devices_) {
        if (!wait_producer_rings_drained(ctx, std::chrono::seconds(2))) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: producers still publishing after the 2 s drain budget -- "
                "unblocking ring back-pressure so device close cannot wedge; markers still in flight on those "
                "cores are DROPPED",
                ctx.chip_id);
            disarm_producer_backpressure(ctx);
        }
    }

    for (auto& ctx : devices_) {
        auto& cluster = MetalContext::instance().get_cluster();
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            if (ctx.relay_program[d] == nullptr) {
                continue;
            }
            const tt_cxy_pair drisc(ctx.chip_id, ctx.drisc_virtual[d]);
            uint32_t quiesce = kernel_profiler::kRelayStopQuiesce;
            cluster.write_core(
                &quiesce, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            uint32_t done = 0;
            bool drained = false;
            while (std::chrono::steady_clock::now() < deadline) {
                cluster.read_core(
                    &done, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
                if ((done & kernel_profiler::kRelayDoneMask) == kernel_profiler::kRelayDoneWord) {
                    break;
                }
                if (!drained && receiver != nullptr &&
                    (done & kernel_profiler::kRelayDoneMask) == kernel_profiler::kRelayDrainedWord) {
                    receiver->notify_producers_drained(static_cast<uint32_t>(&ctx - devices_.data()), d);
                    drained = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if ((done & kernel_profiler::kRelayDoneMask) != kernel_profiler::kRelayDoneWord) {
                log_warning(
                    tt::LogMetal, "[streaming profiler] Device {}: DRISC relay did not acknowledge stop", ctx.chip_id);
            } else if (receiver != nullptr) {
                // done follows the relay's socket barrier, so the host has already acked every byte this socket will
                // carry.
                receiver->notify_producers_done(static_cast<uint32_t>(&ctx - devices_.data()), d);
            }
            // Release restores the NIU; NOC2AXI takes this L1 out of the host's view, so it comes last.
            uint32_t release = kernel_profiler::kRelayStopRelease;
            cluster.write_core(
                &release, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
        }
    }
}

void Devices::verify(Receiver& receiver) {
    for (auto& ctx : devices_) {
        verify_completeness(ctx, static_cast<uint32_t>(&ctx - devices_.data()), receiver);
    }
}

// One MMIO pass per worker core: the producer-owned stall counters, and each lane's tail against the
// receiver's consumed-words mirror.
void Devices::verify_completeness(DeviceCtx& ctx, uint32_t device_index, Receiver& receiver) {
    if (ctx.core_virt.empty()) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    std::vector<uint32_t> heads = receiver.final_lane_heads(device_index);
    std::vector<uint32_t> cv(kernel_profiler::SPSC_CONTROL_END, 0);
    uint64_t total = 0, worst = 0, cores_hit = 0;
    uint64_t stranded_words = 0, stranded_lanes = 0, checked_lanes = 0;
    uint32_t worst_lane = 0, worst_lane_words = 0;
    uint64_t risc_total[kNRisc] = {};
    struct CoreStall {
        uint32_t count, vx, vy, idx;
    };
    std::vector<CoreStall> stalled_cores;
    for (size_t ci = 0; ci < ctx.core_virt.size(); ci++) {
        const auto [vx, vy] = ctx.core_virt[ci];
        cluster.read_core(
            cv.data(),
            kernel_profiler::SPSC_CONTROL_END * sizeof(uint32_t),
            tt_cxy_pair(ctx.chip_id, CoreCoord{vx, vy}),
            prof_l1);
        uint64_t core_total = 0;
        for (uint32_t r = 0; r < kernel_profiler::SPSC_STALL_COUNT_MAX; r++) {
            core_total += cv[kernel_profiler::SPSC_STALL_COUNT_0 + r];
            if (r < kNRisc) {
                risc_total[r] += cv[kernel_profiler::SPSC_STALL_COUNT_0 + r];
            }
        }
        total += core_total;
        worst = std::max(worst, core_total);
        cores_hit += (core_total != 0) ? 1 : 0;
        if (core_total != 0) {
            stalled_cores.push_back({static_cast<uint32_t>(core_total), vx, vy, static_cast<uint32_t>(ci)});
        }
        if (heads.empty()) {
            continue;
        }
        for (uint32_t r = 0; r < kNRisc; r++) {
            const uint32_t lane = static_cast<uint32_t>(ci) * kNRisc + r;
            const uint32_t tail = cv[kernel_profiler::SPSC_RING_TAIL_0 + r];
            const int32_t left = static_cast<int32_t>(tail - (lane < heads.size() ? heads[lane] : 0));
            checked_lanes++;
            if (left > 0) {
                stranded_lanes++;
                stranded_words += static_cast<uint32_t>(left);
                if (static_cast<uint32_t>(left) > worst_lane_words) {
                    worst_lane_words = static_cast<uint32_t>(left);
                    worst_lane = lane;
                }
            }
        }
    }
    if (total != 0) {
        std::sort(stalled_cores.begin(), stalled_cores.end(), [](const CoreStall& a, const CoreStall& b) {
            return a.count > b.count;
        });
        std::string top;
        for (size_t i = 0; i < stalled_cores.size(); i++) {
            const auto& c = stalled_cores[i];
            top += fmt::format("{}({},{})#{}={}", i != 0 ? " " : "", c.vx, c.vy, c.idx, c.count);
        }
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: {} profiler stalls on {} of {} cores (worst core {}); by RISC BR {} | NC "
            "{} | "
            "T0 {} | T1 {} | T2 {}; top cores (virt x,y)=count: {}",
            ctx.chip_id,
            total,
            cores_hit,
            ctx.core_virt.size(),
            worst,
            risc_total[0],
            risc_total[1],
            risc_total[2],
            risc_total[3],
            risc_total[4],
            top);
    }
    if (heads.empty()) {
        return;
    }
    if (stranded_lanes != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] COMPLETENESS: device {} -- {}/{} lanes fully drained; {} lanes stranded {} "
            "words (worst lane {}: {}) <<< stop-path sweep-to-empty contract violated; the capture tail is "
            "incomplete",
            ctx.chip_id,
            checked_lanes - stranded_lanes,
            checked_lanes,
            stranded_lanes,
            stranded_words,
            worst_lane,
            worst_lane_words);
    }
}

}  // namespace tt::tt_metal::streaming_profiler
