// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_device.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <tracy/Tracy.hpp>
#include <tracy/TracyTTDevice.hpp>

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
#include "llrt/tt_cluster.hpp"
#include "hostdev/streaming_profiler_common.h"

namespace tt::tt_metal::streaming_profiler {

namespace {

// Views 7 and 2 last: a roster truncated to fewer relays sheds the bring-up-fragile views first.
constexpr std::array<uint32_t, 8> kRelayBankRoster = {5u, 6u, 4u, 1u, 0u, 3u, 7u, 2u};
// Staging slots per relay, capped at what a DRISC's L1 fits; the spool's bounce slots need all of them, direct push
// the relay's two generations of two slots.
constexpr uint32_t kMaxStageSlots = 7;
constexpr uint32_t kMinStageSlots = 4;
// One 128-byte record per core (control-vector words 12..31, head mirror, wire XY; a power of two keeps the kernel's
// record addressing a shift) and byte-indexed core lists in the kernel. 72 covers two relays on a 140-core grid; a
// relay handed more cores refuses the capture with a message.
constexpr uint32_t kMaxRelayCores = 72;
constexpr uint32_t kScratchBytes = kMaxRelayCores * 128;
static_assert(kScratchBytes % 64 == 0);
constexpr uint32_t kCfgReserve = 8 * 1024;
constexpr uint32_t kMiscBytes = 1024;  // done(64) + stop(64), with headroom
constexpr uint32_t kPageSize = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
constexpr uint32_t kNRisc = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;

// Bring-up runs several MMIO paths and a hang in any of them reports only "MMIO per-op timeout"; this names
// the stall site.

int64_t steady_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

uint32_t packed_xy(const CoreCoord& c) {
    return (static_cast<uint32_t>(c.x) & 0xFFFFu) | ((static_cast<uint32_t>(c.y) & 0xFFFFu) << 16);
}

// Host<->device clock sync: the Tensix wall clock is read over NoC bracketed by host clock reads (Cristian's
// algorithm; the midpoint cancels the round trip to first order). Samples 500 us apart: back to back they span only
// ~360 us and the fitted frequency then carries ~1e-4 of error, which grows with time since the anchor; spaced they
// span ~50 ms.
DeviceClock sync_device_clock(tt::Cluster& cluster, uint32_t chip_id, const CoreCoord& worker) {
    constexpr uint32_t kSpacingUs = 500;
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
        if (i + 1 < kSamples) {
            std::this_thread::sleep_for(std::chrono::microseconds(kSpacingUs));
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
    std::erase_if(samples, [rt_cut](const S& s) { return s.rt > rt_cut; });

    DeviceClock out{.chip_id = chip_id};
    // Centered least squares: centering avoids cancellation at absolute-timestamp magnitudes.
    double hx = 0, dy = 0;
    for (const auto& s : samples) {
        hx += static_cast<double>(s.host_mid);
        dy += static_cast<double>(s.dev);
    }
    hx /= static_cast<double>(samples.size());
    dy /= static_cast<double>(samples.size());
    double num = 0, den = 0;
    for (const auto& s : samples) {
        const double ddx = static_cast<double>(s.host_mid) - hx;
        const double ddy = static_cast<double>(s.dev) - dy;
        num += ddx * ddy;
        den += ddx * ddx;
    }
    const double slope = num / den;  // device cycles per host tick
#ifdef TRACY_ENABLE
    const double ns_per_tick = TracyGetTimerMul() > 0.0 ? TracyGetTimerMul() : 1.0;
#else
    const double ns_per_tick = 1.0;
#endif
    out.frequency_ghz = slope / ns_per_tick;
    // Anchor on the sample mean: extrapolating an intercept to host_time=0 turns a tiny slope error into a huge
    // offset. The samples bracket Tracy's timer while the public clock is std::chrono::steady_clock, so the anchor
    // is re-expressed in steady_clock terms here, once.
    out.anchor_ticks = static_cast<uint64_t>(dy);
    out.anchor_host_ns =
        steady_now_ns() -
        static_cast<int64_t>(static_cast<double>(tracy::Profiler::GetTime() - static_cast<int64_t>(hx)) * ns_per_tick);
    return out;
}

// A static TLB window skips UMD's per-access reconfigure on the socket's ack write (171 vs 382 ns). Metal
// maps a window per DRAM channel only on the channel's preferred worker endpoint port (configure_static_tlbs
// -> ddr_to_noc0) and the relay sits on the unused port, so it maps its own: 2 MB at address 0 spans the whole
// 128 KB DRISC L1. Best-effort: windows are finite, and losing the race costs only the ~210 ns.
void configure_relay_static_tlb(tt::Cluster& cluster, uint32_t device_id, const CoreCoord& drisc_virtual) {
    if (cluster.is_mock_or_emulated()) {
        return;
    }
    auto* tlb_manager = cluster.get_driver()->get_chip(device_id)->get_tlb_manager();
    const tt_xy_pair tlb_core(drisc_virtual.x, drisc_virtual.y);
    if (tlb_manager->is_tlb_mapped(tlb_core)) {
        return;
    }
    try {
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
// workload wedges on full rings; the heartbeat counts sweeps, and two of them prove the loop runs.
bool relay_heartbeat_advanced(
    tt::Cluster& cluster, uint32_t device_id, const CoreCoord& drisc_virtual, uint64_t hb_addr, uint32_t d) {
    const tt_cxy_pair core(device_id, drisc_virtual);
    uint32_t hb = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        cluster.read_core(&hb, sizeof(hb), core, hb_addr);
        if (hb >= 2) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    log_warning(
        tt::LogMetal,
        "[streaming profiler] Device {}: relay {} FAILED TO START (heartbeat {} after launch). The producers would "
        "block forever on a full ring and wedge the workload, so capture is disabled for this run instead.",
        device_id,
        d,
        hb);
    return false;
}

// Every relay's NIU into stream mode, in one launch, run to completion. D2HSocket construction writes its
// config into DRISC L1 from the host, which only lands once the NIU terminates inbound traffic at L1. One
// launch: every LaunchProgram carries a dram_barrier that MMIO-polls a core in every DRAM channel, and a
// barrier that reaches a core already in stream mode never completes.
void set_drisc_niu_stream_mode(IDevice* device, const std::vector<CoreCoord>& drisc_logicals) {
    std::set<CoreRange> ranges;
    for (const auto& c : drisc_logicals) {
        ranges.insert(CoreRange(c, c));
    }
    Program p = CreateProgram();
    CreateKernel(
        p,
        "tt_metal/tools/profiler/kernels/drisc_niu_mode.cpp",
        CoreRangeSet(ranges),
        DramConfig{.noc = NOC::NOC_0, .compile_args = {1u}});
    const std::string who = fmt::format("niu-mode[{} cores]", drisc_logicals.size());
    detail::CompileProgram(device, p, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(device, p, /*force_slow_dispatch=*/true);
    // Launch and wait split so a failure names which half stalled; a stall on the first label means a core was
    // already in stream mode when this run began.
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    detail::WaitProgramDone(device, p);
}

}  // namespace

Devices::DeviceCtx::DeviceCtx() = default;
Devices::DeviceCtx::~DeviceCtx() = default;
Devices::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;


Devices::~Devices() = default;

std::vector<CapturedDevice> Devices::boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    context_id_ = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const auto& hal = MetalContext::instance(context_id_).hal();
    const auto& rtopts = MetalContext::instance(context_id_).rtoptions();

    if (cluster.arch() != tt::ARCH::BLACKHOLE) {
        log_debug(tt::LogMetal, "[streaming profiler] not Blackhole; skipping relay capture.");
        return {};
    }
    // The relay is a DRISC: one DM RISC-V on a DRAM core, which today exists only on Blackhole.
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] no DRAM programmable cores (card FW below the DRISC gate?); producers stay unarmed "
            "and markers are DROPPED");
        return {};
    }
    prof_l1_ = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    drisc_l1_base_ = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    drisc_l1_noc_ = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    slot_bytes_ = kernel_profiler::spsc_span_slot_words(kNRisc) * sizeof(uint32_t);
    const uint32_t region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t fixed = kCfgReserve + kScratchBytes + kMiscBytes;
    l1_.n_stage = std::min(region > fixed ? (region - fixed) / slot_bytes_ : 0u, kMaxStageSlots);
    if (l1_.n_stage < kMinStageSlots) {
        log_warning(tt::LogMetal, "[streaming profiler] DRISC L1 too small for a relay; skipping");
        return {};
    }
    l1_.stage_base = drisc_l1_base_;
    l1_.core_records = l1_.stage_base + l1_.n_stage * slot_bytes_;
    l1_.done = l1_.core_records + kScratchBytes;
    l1_.stop = l1_.done + kernel_profiler::kRelayCtrlWordStride;
    l1_.cfg = drisc_l1_base_ + region - kCfgReserve;
    TT_FATAL(l1_.stop + kernel_profiler::kRelayCtrlWordStride <= l1_.cfg, "DRISC L1 layout overlaps the socket config");

    std::vector<CapturedDevice> out;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx ctx;
        ctx.device = mesh_device->get_device(coord);
        ctx.chip_id = static_cast<uint32_t>(ctx.device->id());
        if (!boot_device(mesh_device, ctx, coord)) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: no DRISC relay -- producers stay unarmed (markers are DROPPED, but "
                "the workload will not stall waiting for a consumer)",
                ctx.chip_id);
            continue;
        }
        ctx.out.chip_id = ctx.chip_id;
        ctx.out.numa_node =
            static_cast<int>(MetalContext::instance(context_id_).get_cluster().get_numa_node_for_device(ctx.chip_id));
        out.push_back(std::move(ctx.out));
        devices_.push_back(std::move(ctx));
    }
    if (!devices_.empty()) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] active on {} device(s){}",
            devices_.size(),
            rtopts.get_streaming_profiler_tracy_enabled() ? " with the Tracy sink" : "");
    }
    return out;
}

bool Devices::boot_device(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord) {
    enumerate_worker_grid(mesh_device, ctx);
    if (!choose_relay_cores(mesh_device, ctx)) {
        return false;
    }
    reserve_spool(mesh_device, ctx.device);
    for (uint32_t d = 0; d < ctx.n_relays; d++) {
        if (!launch_relay(mesh_device, ctx, coord, d)) {
            return false;
        }
    }
    set_producers_armed(ctx, true);

    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    ctx.out.clock = sync_device_clock(cluster, ctx.chip_id, ctx.cores.front().virt);
    TT_FATAL(
        ctx.out.clock.frequency_ghz > 0.0,
        "streaming profiler: device {} wall clock did not advance during clock sync",
        ctx.chip_id);
    return true;
}

void Devices::write_ctrl_word(const DeviceCtx& ctx, const CoreCoord& virt, uint32_t index, uint32_t value) {
    MetalContext::instance(context_id_)
        .get_cluster()
        .write_core(&value, sizeof(value), tt_cxy_pair(ctx.chip_id, virt), prof_l1_ + index * sizeof(uint32_t));
}

// Core identity is not in the packets: the relay stamps each frame with the NoC coordinate the host seeded it with,
// and core_xy resolves it to the core index the lanes are numbered by.
void Devices::enumerate_worker_grid(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const uint32_t chip = ctx.chip_id;
    // The poll list defines the drained set; a producer outside it fills its ring, blocks forever, and takes the
    // host down in wait_until_cores_done. The relay lives on a DRAM core, so the full grid is polled.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const std::vector<uint8_t> zero_ctrl(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
    CaptureContext::Device& cap = ctx.out.ctx;
    for (uint32_t ly = 0; ly < grid.y; ly++) {
        for (uint32_t lx = 0; lx < grid.x; lx++) {
            const CoreCoord logical{lx, ly};
            const WorkerCore c{
                logical,
                cluster.get_physical_coordinate_from_logical_coordinates(
                    chip, logical, CoreType::WORKER, /*no_warn=*/true),
                cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, CoreType::WORKER)};
            cluster.write_core(
                zero_ctrl.data(), static_cast<uint32_t>(zero_ctrl.size()), tt_cxy_pair(chip, c.virt), prof_l1_);
            cap.core_xy.push_back(packed_xy(c.virt));
            for (uint32_t r = 0; r < kNRisc; r++) {
                cap.lanes.push_back(experimental::streaming_profiler::Core{
                    .logical = logical,
                    .physical = c.physical,
                    .chip_id = chip,
                    .risc = static_cast<experimental::streaming_profiler::Risc>(r)});
            }
            ctx.cores.push_back(c);
        }
    }

    // Producers boot unarmed. Upstream does this in BRISC firmware at boot; this tree keeps hw/firmware byte-for-byte
    // main, so the host does it here, before any relay launches, on every Tensix core of the device and not just the
    // compute grid zeroed above: a dispatch core's ring is drained by nobody and its L1 survives device re-init, so a
    // stale arm there would park its firmware zone in the stall path and wedge wait_until_cores_done() at close.
    // set_producers_armed() then sets the flag only on the cores the relays serve, once every relay is up.
    const CoreCoord tensix_grid = cluster.get_soc_desc(chip).get_grid_size(CoreType::TENSIX);
    for (uint32_t ly = 0; ly < tensix_grid.y; ly++) {
        for (uint32_t lx = 0; lx < tensix_grid.x; lx++) {
            const CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(chip, CoreCoord{lx, ly}, CoreType::WORKER);
            write_ctrl_word(ctx, v, kernel_profiler::PROFILER_ARMED, 0);
        }
    }
}

bool Devices::choose_relay_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const auto& rtopts = MetalContext::instance(context_id_).rtoptions();
    const uint32_t chip = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(chip);

    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    if (nbanks == 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: no DRAM views to host a relay -- the streaming profiler is OFF "
            "for this device.",
            chip);
        return false;
    }

    const uint32_t view_cap = std::min<uint32_t>(kMaxRelays, nbanks);
    static_assert(kMaxRelays == 8, "rtoptions bounds TT_METAL_STREAMING_PROFILER_NRELAYS at 8");
    const uint32_t requested = rtopts.get_streaming_profiler_num_relays();
    ctx.n_relays = requested == 0 ? view_cap : std::min(requested, view_cap);
    if (requested > view_cap) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: TT_METAL_STREAMING_PROFILER_NRELAYS={} exceeds this part's {} DRAM "
            "views (one relay each); CLAMPED to {} relays",
            chip,
            requested,
            nbanks,
            ctx.n_relays);
    }

    std::vector<uint32_t> banks;
    for (const uint32_t b : kRelayBankRoster) {
        if (b < nbanks) {
            banks.push_back(b);
        }
    }
    TT_FATAL(
        banks.size() >= ctx.n_relays,
        "streaming profiler needs {} relay banks but only {} usable DRAM views are in the roster (part has {} "
        "views)",
        ctx.n_relays,
        banks.size(),
        nbanks);

    // Picked up front so that every relay's NIU flips in one launch (see set_drisc_niu_stream_mode).
    std::vector<CoreCoord> relay_cores;
    for (uint32_t d = 0; d < ctx.n_relays; d++) {
        ctx.relays[d].logical = mesh_device->impl().pick_unused_dram_logical_core(ctx.device, banks[d]);
        relay_cores.push_back(ctx.relays[d].logical);
    }
    // pick_unused_dram_logical_core() reserves per view and cannot see two views resolving to one physical port
    // (views 0 and 7 have both come back as NoC core 0-0); two relays on one L1 would silently overlap, so refuse.
    for (uint32_t a = 0; a < ctx.n_relays; a++) {
        for (uint32_t b = a + 1; b < ctx.n_relays; b++) {
            TT_FATAL(
                relay_cores[a] != relay_cores[b],
                "streaming profiler: DRISC {} (DRAM view {}) and DRISC {} (DRAM view {}) both resolve to logical "
                "DRAM core ({},{}). Two resident relay kernels cannot share a core.",
                a,
                banks[a],
                b,
                banks[b],
                relay_cores[a].x,
                relay_cores[a].y);
        }
    }
    // Cluster::dram_barrier syncs subchannel 0 of every channel and every LaunchProgram carries one; a relay
    // resident there is in stream mode, where a DRAM-range address no longer forwards to GDDR. Reported, not
    // fatal: it usually works, and this is the explanation for a later MMIO timeout.
    uint32_t collide = 0;
    for (int ch = 0; ch < soc.get_num_dram_channels(); ch++) {
        const CoreCoord bar = soc.get_dram_core_for_channel(ch, 0, CoordSystem::LOGICAL);
        collide += static_cast<uint32_t>(std::count(relay_cores.begin(), relay_cores.end(), bar));
    }
    if (collide != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] {} of {} relays sit on a dram_barrier target core (subchannel 0 "
            "of their channel). Every LaunchProgram barriers those cores while they are in stream "
            "mode; a 60-70 ms MMIO timeout at bring-up or weight upload has this as a candidate.",
            collide,
            ctx.n_relays);
    }
    set_drisc_niu_stream_mode(ctx.device, relay_cores);
    return true;
}

// One replicated mesh buffer with one interleaved page per bank reserves the same window in every bank of
// every device. Mesh-level because the lock-step allocator never sees a device-local Buffer::create and
// would hand the region out again.
void Devices::reserve_spool(const std::shared_ptr<distributed::MeshDevice>& mesh_device, IDevice* device) {
    const uint32_t spool_mb = MetalContext::instance(context_id_).rtoptions().get_streaming_profiler_spool_mb();
    if (spool_mb == 0 || spool_buffer_ != nullptr) {
        return;
    }
    const uint32_t bytes = spool_mb * (1u << 20);
    const uint32_t nbanks_dram = device->allocator()->get_num_banks(BufferType::DRAM);
    try {
        spool_buffer_ = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{static_cast<DeviceAddr>(nbanks_dram) * bytes},
            distributed::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = BufferType::DRAM},
            mesh_device.get());
        spool_addr_ = static_cast<uint32_t>(spool_buffer_->address());
        spool_bytes_ = bytes;
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] could not reserve {} MiB/bank of DRAM for the GDDR spool ({}); falling "
            "back to direct push",
            spool_mb,
            e.what());
    }
}

bool Devices::launch_relay(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord,
    uint32_t d) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    const auto& hal = MetalContext::instance(context_id_).hal();
    const auto& rtopts = MetalContext::instance(context_id_).rtoptions();
    const uint32_t chip = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(chip);
    Relay& relay = ctx.relays[d];

    // Contiguous bands in the host's grid order: a core belongs to exactly one relay, and the integer prefix
    // split assigns every core once.
    const uint32_t num_cores = static_cast<uint32_t>(ctx.cores.size());
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(num_cores) * d) / ctx.n_relays);
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(num_cores) * (d + 1)) / ctx.n_relays);
    const uint32_t my_cores = hi - lo;
    if (my_cores > kMaxRelayCores) {
        log_error(
            tt::LogMetal,
            "[streaming profiler] Device {}: relay {} would own {} cores but a relay holds at most {}; set "
            "TT_METAL_STREAMING_PROFILER_NRELAYS to at least {} for this {}-core grid",
            chip,
            d,
            my_cores,
            kMaxRelayCores,
            (num_cores + kMaxRelayCores - 1) / kMaxRelayCores,
            num_cores);
        return false;
    }
    if (my_cores == 0) {
        return true;
    }
    const CoreCoord translated = soc.dram_bank_endpoint_coords.at(relay.logical.x).at(relay.logical.y);
    const tt::umd::CoreCoord phys = soc.translate_coord_to(
        tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);
    relay.virt = ctx.device->virtual_core_from_logical_core(relay.logical, CoreType::DRAM);
    const tt_cxy_pair drisc(chip, relay.virt);

    configure_relay_static_tlb(cluster, chip, relay.virt);

    try {
        auto socket = std::make_unique<distributed::D2HSocket>(
            mesh_device,
            distributed::MeshCoreCoord{coord, CoreCoord(phys.x, phys.y)},
            (rtopts.get_streaming_profiler_fifo_mb() << 20) / kPageSize * kPageSize,
            distributed::D2HSocket::ExternalConfigBuffer{
                .address = l1_.cfg, .sender_core_type = HalProgrammableCoreType::DRAM},
            distributed::D2HSocket::ProcessScope::InProcess);
        socket->set_page_size(kPageSize);

        // Zero the relay core's own profiler ring: the relay is built with PROFILE_KERNEL, firmware writes zone
        // markers into this ring on every launch, nothing drains it, and the SPSC backend blocks on a full ring, so
        // after ~74 launches in one reset window the RISC wedges in firmware init.
        const std::vector<uint8_t> zero_ctrl(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
        cluster.write_core(
            zero_ctrl.data(),
            static_cast<uint32_t>(zero_ctrl.size()),
            drisc,
            hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER));

        // A stale done, heartbeat or stop word from the previous run reads as this run's live state (teardown leaves
        // stop at 1 or 2, and the relay loop exits on nonzero stop).
        uint32_t zero_words[2 * kernel_profiler::kRelayCtrlWordStride / sizeof(uint32_t)] = {};
        cluster.write_core(zero_words, sizeof(zero_words), drisc, relay_noc_addr(l1_.done));

        auto program = std::make_unique<Program>(CreateProgram());
        const std::unordered_map<std::string, uint32_t> cargs = {
            {"stage_base", l1_.stage_base},
            {"n_stage", l1_.n_stage},
            {"core_records", l1_.core_records},
            {"done_addr", l1_.done},
            {"stop_addr", l1_.stop},
            {"socket_config_addr", socket->get_config_buffer_address()},
            {"max_cores", kMaxRelayCores},
            // d&2 splits the pushers across two of the four unicast request VCs.
            {"write_vc", (d & 2u) ? 0u : 1u},
            // The bounce slots cost a staging generation, so a smaller L1 falls back to direct push rather than
            // tripping the kernel's geometry static_asserts.
            {"spool_base", spool_addr_},
            {"spool_bytes", l1_.n_stage >= kMaxStageSlots ? spool_bytes_ : 0u}};
        if (spool_bytes_ != 0 && l1_.n_stage < kMaxStageSlots) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: only {} staging slots fit, too few for the spool's bounce "
                "buffers; relay {} runs direct push",
                chip,
                l1_.n_stage,
                d);
        }
        auto relay_id = CreateKernel(
            *program,
            "tt_metal/tools/profiler/kernels/streaming_profiler_relay.cpp",
            relay.logical,
            // NOC 1 egress runs ~2x the service interval of NOC 0, so a relay parked there takes essentially every
            // profiler stall.
            DramConfig{
                .noc = NOC::NOC_0, .defines = {{"STREAMING_PROFILER_RELAY_KERNEL", "1"}}, .named_compile_args = cargs});
        std::vector<uint32_t> rt = {my_cores, static_cast<uint32_t>(prof_l1_)};
        // Reversed: launch order follows global index, so the slice's last-launched cores land in the first-chunk
        // slots, which are serviced first.
        for (uint32_t ci = hi; ci-- > lo;) {
            rt.push_back(ctx.out.ctx.core_xy[ci]);
        }
        SetRuntimeArgs(*program, relay_id, relay.logical, rt);

        detail::CompileProgram(ctx.device, *program, /*force_slow_dispatch=*/true);
        detail::WriteRuntimeArgsToDevice(ctx.device, *program, /*force_slow_dispatch=*/true);
        detail::LaunchProgram(ctx.device, *program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

        if (!relay_heartbeat_advanced(cluster, chip, relay.virt, relay_noc_addr(l1_.done) + 4, d)) {
            return false;
        }
        TT_FATAL(d == ctx.out.sockets.size(), "sockets must form a contiguous prefix");
        ctx.out.sockets.push_back(std::move(socket));
        relay.program = std::move(program);
    } catch (const std::exception& e) {
        // A code-region overflow fails the load, not the start, and the run then exits 0 with every marker dropped.
        log_error(
            tt::LogMetal,
            "[streaming profiler] Device {}: DRISC {} FAILED TO LOAD -- THIS CAPTURE WILL BE EMPTY; the run will still "
            "exit 0 ({})",
            chip,
            d,
            e.what());
        return false;
    }
    return true;
}

// Producers boot unarmed (enumerate_worker_grid() clears PROFILER_ARMED on every Tensix core of the device) and
// only block on a full ring once armed, so a core no relay drains can never wedge device close. Arming follows the
// relays coming up; a relay that fails leaves the whole device unarmed and its markers are overwritten instead.
void Devices::set_producers_armed(const DeviceCtx& ctx, bool armed) {
    for (const WorkerCore& c : ctx.cores) {
        write_ctrl_word(ctx, c.virt, kernel_profiler::PROFILER_ARMED, armed ? 1u : 0u);
    }
}

void Devices::quiesce(const RelayStateFn& on_state) {
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    for (uint32_t di = 0; di < devices_.size(); di++) {
        const DeviceCtx& ctx = devices_[di];
        const auto write_stop = [&](uint32_t d, uint32_t word) {
            cluster.write_core(
                &word, sizeof(word), tt_cxy_pair(ctx.chip_id, ctx.relays[d].virt), relay_noc_addr(l1_.stop));
        };
        for (uint32_t d = 0; d < ctx.n_relays; d++) {
            const tt_cxy_pair drisc(ctx.chip_id, ctx.relays[d].virt);
            write_stop(d, kernel_profiler::kRelayStopQuiesce);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            bool drained = false;
            for (;;) {
                uint32_t state = 0;
                cluster.read_core(&state, sizeof(state), drisc, relay_noc_addr(l1_.done));
                state &= kernel_profiler::kRelayDoneMask;
                if (state == kernel_profiler::kRelayDoneWord) {
                    break;
                }
                if (!drained && on_state && state == kernel_profiler::kRelayDrainedWord) {
                    on_state(di, d, RelayState::Drained);
                    drained = true;
                }
                TT_FATAL(
                    std::chrono::steady_clock::now() < deadline,
                    "streaming profiler: device {} relay {} did not finish within 10 s of its stop (state {:#x})",
                    ctx.chip_id,
                    d,
                    state);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (on_state) {
                // done follows the relay's socket barrier, so the host has already acked every byte this socket will
                // carry.
                on_state(di, d, RelayState::Done);
            }
        }
        // Nothing drains the rings any more: a producer blocked on a full one is released and overwrites from here on.
        set_producers_armed(ctx, false);
        // Release restores the NIU; NOC2AXI takes this L1 out of the host's view, so it comes last.
        for (uint32_t d = 0; d < ctx.n_relays; d++) {
            write_stop(d, kernel_profiler::kRelayStopRelease);
        }
    }
}

// One MMIO pass per worker core: the producer-owned stall counters, and each lane's tail against the consumed-words
// mirror.
void Devices::verify_completeness(uint32_t device_index) {
    const DeviceCtx& ctx = devices_[device_index];
    auto& cluster = MetalContext::instance(context_id_).get_cluster();
    std::vector<uint32_t> cv(kernel_profiler::SPSC_CONTROL_END, 0);
    uint64_t total = 0, stranded_words = 0, stranded_lanes = 0;
    std::vector<std::pair<uint32_t, uint32_t>> stalled;  // (stalls, core index)
    for (size_t ci = 0; ci < ctx.cores.size(); ci++) {
        cluster.read_core(
            cv.data(),
            kernel_profiler::SPSC_CONTROL_END * sizeof(uint32_t),
            tt_cxy_pair(ctx.chip_id, ctx.cores[ci].virt),
            prof_l1_);
        uint32_t core_total = 0;
        for (uint32_t r = 0; r < kernel_profiler::SPSC_STALL_COUNT_MAX; r++) {
            core_total += cv[kernel_profiler::SPSC_STALL_COUNT_0 + r];
        }
        total += core_total;
        if (core_total != 0) {
            stalled.emplace_back(core_total, static_cast<uint32_t>(ci));
        }
        for (uint32_t r = 0; r < kNRisc; r++) {
            const int32_t left = static_cast<int32_t>(
                cv[kernel_profiler::SPSC_RING_TAIL_0 + r] - cv[kernel_profiler::SPSC_RING_HEAD_0 + r]);
            if (left > 0) {
                stranded_lanes++;
                stranded_words += static_cast<uint32_t>(left);
            }
        }
    }
    if (total != 0) {
        std::sort(stalled.begin(), stalled.end(), std::greater<>());
        std::string top;
        for (const auto& [count, ci] : stalled) {
            const CoreCoord& v = ctx.cores[ci].virt;
            top += fmt::format("{}({},{})#{}={}", top.empty() ? "" : " ", v.x, v.y, ci, count);
        }
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: {} profiler stalls on {} of {} cores; (virt x,y)#index=stalls: {}",
            ctx.chip_id,
            total,
            stalled.size(),
            ctx.cores.size(),
            top);
    }
    if (stranded_lanes != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: {} words on {} lanes were published after the relay's last sweep and are "
            "not in the capture",
            ctx.chip_id,
            stranded_words,
            stranded_lanes);
    }
}

}  // namespace tt::tt_metal::streaming_profiler
