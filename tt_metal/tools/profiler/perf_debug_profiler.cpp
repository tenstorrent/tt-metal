// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_profiler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <span>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tracy/Tracy.hpp>
#include <common/TracyTTDeviceData.hpp>  // tracy::RiscType X280_RD0/X280_RELAY0 lanes

#include <chrono>
#include <thread>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>  // MeshCoreCoord
#include <umd/device/types/core_coordinates.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig (a DRISC kernel is not in the public headers yet)
#include "jit_build/build_env_manager.hpp"
#include "llrt/tt_cluster.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tools/profiler/x280_driver.hpp"
#include "tools/profiler/x280_profzone_boot.hpp"
#include "tools/profiler/x280_profzone_decode.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"
#include "tools/profiler/perf_debug_profiler_packets.hpp"
#include "impl/profiler/profiler.hpp"  // generateZoneSourceLocationsHashes (zone hash -> name)
#include "prof_packet.h"
#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal {

namespace pz = tt::tt_metal::profiler;

// Host-only record type for a PP_DATA payload continuation. Never appears on the wire, so it only has to
// avoid the codes prof_packet.h actually uses (0,1,2,5..11); 31 is the top of the 5-bit type field.
constexpr uint32_t kRecDataCont = 31u;
// Largest payload the 7-bit wire size field can express, in uint64s -- bounds the consumer's scratch.
constexpr uint32_t kMaxEventValues = 64;

// pimpl so the BroadcastRing header stays out of perf_debug_profiler.hpp.
struct RecRingHolder {
    tt::tt_metal::BroadcastRing<PerfDebugRec> ring;
    explicit RecRingHolder(size_t cap) : ring(cap) {}
};

// Ring capacity in RECORDS (rounded up to a power of two by BroadcastRing), TT_METAL_PERF_DEBUG_RING_RECS.
// Default 4M == test_x280_realprof's --mqcap default (~96 MB at 24 B/Rec). A lagging consumer DROPS rather
// than back-pressuring, so this sizes the burst it can absorb, not a correctness bound.
size_t ring_capacity_recs() {
    static const size_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_RING_RECS");
        if (s == nullptr || *s == '\0') {
            return static_cast<size_t>(4u << 20);
        }
        return static_cast<size_t>(std::strtoull(s, nullptr, 10));
    }();
    return v;
}

namespace {
// TT_METAL_PERF_DEBUG_DRISC_GAP: fixed inter-sweep gap in device cycles for the DRISC drainer. 0 (default)
// means continuous sweeping -- peak throughput. Bulk reads cost 41x the NoC bytes of the old poll, so a
// non-zero gap is how that traffic gets bounded once peak is no longer the goal; this is the knob a pacing
// controller would drive.
uint32_t drisc_gap_cycles() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_DRISC_GAP");
        return (s == nullptr || *s == '\0') ? 0u : static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    }();
    return v;
}

// Read once: profile the X280 drain harts as well as the worker kernels.
bool hart_zones_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_HART_ZONES");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return on;
}
}  // namespace

// Per-read page cap, overridable at runtime for tuning: TT_METAL_PERF_DEBUG_MAX_PAGES (0 = uncapped, take
// whatever the FIFO holds). The compiled default came from the synthetic benchmark; on high-volume real models
// (UFLD-v2: ~99M markers) the busier socket pins at the cap on every read, which is a suspect for the relay
// sitting in HOST-WAIT.
uint32_t max_pages_per_read(uint32_t compiled_default) {
    static const uint32_t v = [compiled_default] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_MAX_PAGES");
        if (s == nullptr || *s == '\0') {
            return compiled_default;
        }
        return static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_STALL_ONLY=1: decode far enough to COUNT PRODUCER STALL ZONES and nothing else --
// no record building, no BroadcastRing publish, no Tracy. The packet walk is still the real one (a raw scan
// for the 0x7FFF pattern would false-positive on timestamp words that happen to equal 32767), so the count
// is exact; what is skipped is the ~24 B record per marker and the publish.
//
// This exists to measure the KNEE without the host being the thing under test: producer stalls are the knee
// metric, and we measured that host-side per-marker work is what feeds back into the DRISC's credit wait and
// makes stall counts swing between 0 and ~1,000 for the same configuration.
bool stall_only() {
    static const bool on = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_STALL_ONLY");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return on;
}

// TT_METAL_PERF_DEBUG_NO_DECODE=1: the reader thread does read() + ack and NOTHING else -- no decode, no
// publish. Deliberately a measurement tool, not a mode: it isolates whether the DRISC's credit wait is
// caused by per-marker host work sharing the ack path. The DRISC's own phase counters live in device L1 and
// are read at teardown, so they stay valid with this on; the host-derived stats (marker count, stall zones)
// do NOT, since those are produced by the decode being skipped.
bool decode_disabled() {
    static const bool off = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_NO_DECODE");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return off;
}

// TT_METAL_PERF_DEBUG_NO_TRACY=1: drain and decode EXACTLY as normal (markers and stall zones are still
// counted) but skip the Tracy push. Isolates the cost of the sink from the cost of read+decode -- if the
// relay stops host-waiting with this on, the Tracy push is provably the bottleneck.
bool tracy_push_disabled() {
    static const bool off = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_NO_TRACY");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return off;
}

// ---- Host<->device clock sync -------------------------------------------------------------------------
// Restores what the legacy RealtimeProfilerManager used to provide before perf_debug gated it off. Without a
// real sync, AddDevice() can only guess: it anchors "device time 0" at the host time the FIRST MARKER was
// CONSUMED, which is later than when the device produced it by the whole drain+decode+ring latency -- so
// every device zone sits shifted right of the host CPU zones.
//
// Done host-side on purpose: the RT manager synced via a device kernel + dispatch handshake (and a reserved
// tensix core), which is exactly the baggage perf_debug exists to avoid. Instead read the Tensix WALL CLOCK
// (the very counter kernel markers timestamp with) straight over NoC, bracketed by host clock reads --
// Cristian's algorithm: the midpoint of the bracket cancels the round-trip to first order.
struct PerfDebugSync {
    double frequency = 0.0;  // device cycles per nanosecond (GHz)
    uint64_t device_at_anchor = 0;
    int64_t host_anchor = 0;
    bool valid = false;
};

PerfDebugSync sync_device_clock(tt::Cluster& cluster, uint32_t chip_id, const CoreCoord& worker) {
    // RISCV_DEBUG_REG_WALL_CLOCK_L/H. Reading L atomically LATCHES H, so read L then H (H's own latency is
    // irrelevant). Same registers the X280 firmware co-samples in calibrate().
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
        cluster.read_reg(&lo, target, kWallClockL);  // latches H
        cluster.read_reg(&hi, target, kWallClockH);
        const int64_t t1 = tracy::Profiler::GetTime();
        samples.push_back(S{(t0 + t1) / 2, (static_cast<uint64_t>(hi) << 32) | lo, t1 - t0});
    }
    // Drop NoC/PCIe-contended outliers: keep samples whose round-trip is within 1.5x the median.
    std::vector<int64_t> rts;
    rts.reserve(samples.size());
    for (const auto& s : samples) {
        rts.push_back(s.rt);
    }
    std::sort(rts.begin(), rts.end());
    const int64_t rt_cut = rts[rts.size() / 2] + rts[rts.size() / 2] / 2;

    // Centered least squares (centering avoids catastrophic cancellation at absolute-timestamp magnitudes).
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
    PerfDebugSync out;
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
    // Anchor on the sample mean (self-consistent: device time AT that host time), rather than extrapolating
    // an intercept back to host_time=0 where a tiny slope error becomes a huge offset.
    out.host_anchor = static_cast<int64_t>(hx);
    out.device_at_anchor = static_cast<uint64_t>(dy);
    out.valid = out.frequency > 0.0;
    return out;
}

PerfDebugProfiler::DeviceCtx::DeviceCtx() = default;
PerfDebugProfiler::DeviceCtx::~DeviceCtx() = default;
PerfDebugProfiler::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;

PerfDebugProfiler::PerfDebugProfiler(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    try {
        start(mesh_device);
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[perf-debug profiler] init failed ({}); disabled for this session.", e.what());
        stop();
    }
}

PerfDebugProfiler::~PerfDebugProfiler() { stop(); }

void PerfDebugProfiler::start(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();

    if (cluster.arch() != tt::ARCH::BLACKHOLE) {
        log_debug(tt::LogMetal, "[perf-debug profiler] not Blackhole; skipping X280 capture.");
        return;
    }

    tracy_ = std::make_unique<PerfDebugTracyHandler>();
    // NOTE: zone names are loaded LAZILY on the first drain (see drain_loop), NOT here -- at start()
    // (MeshDevice bring-up) the workload's kernels have not been JIT-compiled yet, so their zone-source-
    // location entries are not in the log and every name would fall back to "Zone_<hash>".

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx ctx;
        ctx.chip_id = static_cast<uint32_t>(mesh_device->get_device(coord)->id());
        if (!boot_device(mesh_device, ctx)) {
            continue;  // boot logs its own reason; degrade to no-capture for this device
        }
        // Tracy: anchor + pre-create the per-core contexts (off the drain hot path). Freq = device
        // aiclk in GHz (cycles/ns), matching the standard DeviceProfiler.
        // Anchor the device timeline with a REAL host<->device clock sync (see sync_device_clock). Falls back
        // to the old guess -- aiclk + "device 0 == now" -- only if the sync cannot fit a line, in which case
        // device zones are placed relative to the first marker CONSUMED and so lag the host zones.
        double freq = cluster.get_device_aiclk(ctx.chip_id) / 1000.0;
        if (freq <= 0.0) {
            freq = 1.0;
        }
        PerfDebugSync sync;
        if (!ctx.core_virt.empty()) {
            const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
            sync = sync_device_clock(cluster, ctx.chip_id, w);
        }
        if (sync.valid) {
            ctx.synced = true;
            tracy_->AddDevice(
                ctx.chip_id, sync.host_anchor, static_cast<double>(sync.device_at_anchor), sync.frequency);
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] Device {} clock sync: frequency={:.6f} GHz (aiclk reports {:.6f}), "
                "device_time_at_anchor={} cycles",
                ctx.chip_id,
                sync.frequency,
                freq,
                sync.device_at_anchor);
        } else {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] Device {} clock sync FAILED; falling back to first-marker anchoring "
                "(device zones will lag the host zones by the drain latency)",
                ctx.chip_id);
            tracy_->AddDevice(ctx.chip_id, tracy::Profiler::GetTime(), 0.0, freq);
        }
        // NOTE: per-core Tracy contexts are created LAZILY on each core's first zone (HandleWorkerZone ->
        // GetOrCreateContext). We deliberately do NOT pre-create the full worker grid here: only ~16 of
        // ~110 cores typically run the workload, and pre-creating all of them litters the capture with
        // empty (count=0) contexts that read as "cores not showing up". The per-zone mutex+lookup cost is
        // identical either way; lazy creation just avoids minting dead contexts.
        ctx.active = true;
        devices_.push_back(std::move(ctx));
    }

    // Spawn AFTER devices_ is stable (the threads index into it). ONE writer (read+decode+publish) and one
    // consumer (ring -> Tracy), matching test_x280_realprof: the slow Tracy sink is off the drain path.
    if (!devices_.empty()) {
        const uint32_t cap = max_pages_per_read(kMaxPagesPerRead);
        // Records per read: a page holds at most page_words/2 two-word markers.
        const size_t recs_per_page = (kPageSize / sizeof(uint32_t)) / 2;
        read_chunk_recs_ = cap ? static_cast<size_t>(cap) * recs_per_page : static_cast<size_t>(kHRingWords);
        ring_ = std::make_unique<RecRingHolder>(ring_capacity_recs());
        for (uint32_t s = 0; s < kNSockets; s++) {
            writers_.emplace_back(&PerfDebugProfiler::writer_thread, this, s);
            decoders_.emplace_back(&PerfDebugProfiler::decoder_thread, this, s);
        }
        consumers_.emplace_back(&PerfDebugProfiler::consumer_thread, this);
    }
    if (!devices_.empty()) {
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] active on {} device(s): DRISC drain -> {} MiB D2H socket -> Tracy",
            devices_.size(),
            (static_cast<uint64_t>(kHRingWords) * 4) / (1024 * 1024));
    }
}

// Put a DRISC's NIU into stream mode (1) or back to NOC2AXI (0). Its own program, run to completion:
// D2HSocket construction writes the config into DRISC L1 from the host, which only lands once the NIU
// terminates inbound traffic at L1. Launched outside the command queue like the drainer itself.
void PerfDebugProfiler::set_drisc_niu_mode(IDevice* device, const CoreCoord& drisc_logical, uint32_t stream) {
    Program p = CreateProgram();
    CreateKernel(
        p,
        "tt_metal/tools/profiler/kernels/drisc_niu_mode.cpp",
        drisc_logical,
        DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
    detail::CompileProgram(device, p, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(device, p, /*force_slow_dispatch=*/true);
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);
}

// Producers are armed by TT_METAL_DEVICE_PROFILER, not by us, and a lossless producer BLOCKS on a full
// ring. So whenever the drainer fails to come up, the workload does not merely lose its capture -- it
// wedges, with 550 lanes parked in ring_ensure_room and nothing on the other end. PROFILER_TERMINATE
// exists precisely for this: while set, the producer stops blocking and proceeds.
//
// Learned the hard way, twice in one session from unrelated causes: card FW below the DRISC gate, and a
// drain kernel that failed to JIT-compile. Both looked like a hung test.
void PerfDebugProfiler::disarm_producers(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t device_id) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    uint32_t one = 1;
    uint32_t n = 0;
    for (uint32_t ly = 0; ly < static_cast<uint32_t>(grid.y); ly++) {
        for (uint32_t lx = 0; lx < static_cast<uint32_t>(grid.x); lx++) {
            const CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{lx, ly}, CoreType::WORKER);
            cluster.write_core(
                &one,
                sizeof(uint32_t),
                tt_cxy_pair(device_id, v),
                prof_l1 + kernel_profiler::PROFILER_TERMINATE * sizeof(uint32_t));
            n++;
        }
    }
    log_warning(
        tt::LogMetal,
        "[perf-debug profiler] Device {}: no DRISC drainer -- disarmed ring back-pressure on {} cores "
        "(markers are DROPPED, but the workload will not stall waiting for a consumer)",
        device_id,
        n);
}

bool PerfDebugProfiler::boot_device(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t device_id = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(device_id);

    // The drainer is a DRISC: one DM RISC-V on a DRAM core. Nothing else here is Blackhole-specific, but
    // that is the only place they exist today.
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] Device {}: no DRAM programmable cores (card FW below the DRISC gate?)",
            device_id);
        disarm_producers(mesh_device, device_id);
        return false;
    }

    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t gx = static_cast<uint32_t>(grid.x), gy = static_cast<uint32_t>(grid.y);
    const uint64_t num_cores = static_cast<uint64_t>(gx) * gy;
    ctx.nl = static_cast<uint32_t>(num_cores) * kNRisc;
    ctx.core_virt.resize(num_cores);

    // Pre-zero every core's profiler control vector (heads and tails start clean) and build the maps the
    // HOST owns: core index -> virtual (x,y), which is the drainer's poll list and Tracy's view, and the
    // inverse packed (y<<16)|x -> core index, which is the one thing the drainer does not put on the wire.
    // Identity travels in the payload instead, written by the producing core into SPSC_CORE_XY.
    std::vector<uint32_t> coords(num_cores, 0);
    std::vector<uint8_t> zero_ctrl(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
    for (uint32_t ly = 0; ly < gy; ly++) {
        for (uint32_t lx = 0; lx < gx; lx++) {
            const uint32_t idx = ly * gx + lx;
            CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{lx, ly}, CoreType::WORKER);
            const uint32_t vx = static_cast<uint32_t>(v.x), vy = static_cast<uint32_t>(v.y);
            coords[idx] = (vx & 0xFFFFu) | ((vy & 0xFFFFu) << 16);
            cluster.write_core(zero_ctrl.data(), (uint32_t)zero_ctrl.size(), tt_cxy_pair(device_id, v), prof_l1);
            const CoreCoord noc0 = cluster.get_physical_coordinate_from_logical_coordinates(
                device_id, CoreCoord{lx, ly}, CoreType::WORKER, /*no_warn=*/true);
            ctx.core_virt[idx] = {vx, vy};
            ctx.virt_to_noc0[(static_cast<uint64_t>(vx) << 32) | vy] = {
                static_cast<uint32_t>(noc0.x), static_cast<uint32_t>(noc0.y)};
        }
    }

    ctx.device = mesh_device->get_devices().front();
    const distributed::MeshCoordinate scoord = *distributed::MeshCoordinateRange(mesh_device->shape()).begin();

    // ---- bring up each DRISC over a disjoint slice of the grid ----
    //
    // The split is contiguous rather than interleaved: each drainer's coords list is a run of the same
    // grid order the host uses everywhere else, so a core belongs to exactly one drainer and neither can
    // see the other's rings. Nothing is shared on the device -- separate L1, separate socket, separate
    // head mirrors -- so the two drain loops never interact.
    for (uint32_t d = 0; d < kNSockets; d++) {
        const uint32_t lo = static_cast<uint32_t>((num_cores * d) / kNSockets);
        const uint32_t hi = static_cast<uint32_t>((num_cores * (d + 1)) / kNSockets);
        const uint32_t my_cores = hi - lo;
        if (my_cores == 0) {
            continue;
        }
        ctx.drisc_logical[d] = mesh_device->impl().pick_unused_dram_logical_core(d);
        const CoreCoord translated =
            soc.dram_bank_endpoint_coords.at(ctx.drisc_logical[d].x).at(ctx.drisc_logical[d].y);
        const tt::umd::CoreCoord drisc_phys = soc.translate_coord_to(
            tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED),
            CoordSystem::NOC0);
        ctx.drisc_virtual[d] = ctx.device->virtual_core_from_logical_core(ctx.drisc_logical[d], CoreType::DRAM);
        ctx.drisc_l1_base[d] = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        ctx.drisc_l1_noc[d] = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

        const uint32_t span_bytes = (kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE +
                                     kNRisc * kernel_profiler::PROFILER_L1_VECTOR_SIZE) *
                                    sizeof(uint32_t);
        const uint32_t slot_bytes = kernel_profiler::SPSC_SPAN_PREFIX_WORDS * sizeof(uint32_t) + span_bytes;
        const uint32_t region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        constexpr uint32_t kCfgReserve = 8 * 1024;
        constexpr uint32_t kScratchBytes = 128 * 32;
        constexpr uint32_t kMiscBytes = 512;
        const uint32_t fixed = kCfgReserve + kScratchBytes + kMiscBytes;
        const uint32_t nstage = region > fixed ? (region - fixed) / slot_bytes : 0;
        if (nstage == 0) {
            log_warning(tt::LogMetal, "[perf-debug profiler] Device {}: DRISC L1 too small; skipping", device_id);
            disarm_producers(mesh_device, device_id);
            return false;
        }
        const uint32_t stage_base = ctx.drisc_l1_base[d];
        const uint32_t head_scratch = stage_base + nstage * slot_bytes;
        ctx.done_addr[d] = head_scratch + kScratchBytes;
        ctx.stop_addr[d] = ctx.done_addr[d] + 64;
        ctx.results_addr[d] = ctx.stop_addr[d] + 64;
        const uint32_t cfg_l1 = ctx.drisc_l1_base[d] + region - kCfgReserve;
        TT_FATAL(ctx.results_addr[d] + 128 <= cfg_l1, "DRISC L1 layout overlaps the socket config");

        // Stream mode first: the socket config is written from the host and only lands in L1 once the NIU
        // stops forwarding inbound DRAM-range addresses to GDDR. The kernel restores it on the host's word.
        set_drisc_niu_mode(ctx.device, ctx.drisc_logical[d], 1);

        try {
            ctx.sockets[d] = std::make_unique<distributed::D2HSocket>(
                mesh_device,
                distributed::MeshCoreCoord{scoord, CoreCoord(drisc_phys.x, drisc_phys.y)},
                static_cast<uint32_t>((static_cast<uint64_t>(kHRingWords) * 4 / kPageSize) * kPageSize),
                distributed::D2HSocket::ExternalConfigBuffer{.address = cfg_l1, .sender_is_l2cpu = true});
            ctx.sockets[d]->set_page_size(kPageSize);
            ctx.decode[d] = std::make_unique<pz::ProfzoneDecodeState>();
            ctx.decode[d]->reset(ctx.nl);
            for (uint32_t c = 0; c < num_cores; c++) {
                ctx.decode[d]->core_of_xy[coords[c]] = c;  // full map: lane ids stay global across drainers
            }

            uint32_t zero = 0;
            cluster.write_core(
                &zero,
                sizeof(uint32_t),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));

            ctx.drain_program[d] = std::make_unique<Program>(CreateProgram());
            auto drain_id = CreateKernel(
                *ctx.drain_program[d],
                "tt_metal/tools/profiler/kernels/drisc_profiler_drain.cpp",
                ctx.drisc_logical[d],
                DramConfig{
                    .noc = NOC::NOC_0,
                    .compile_args = {
                        stage_base,
                        nstage,
                        head_scratch,
                        ctx.results_addr[d],
                        ctx.done_addr[d],
                        ctx.stop_addr[d],
                        ctx.sockets[d]->get_config_buffer_address(),
                        0xFFFFFFFFu,
                        128,
                        drisc_gap_cycles()}});
            std::vector<uint32_t> rt = {my_cores, static_cast<uint32_t>(prof_l1)};
            rt.insert(rt.end(), coords.begin() + lo, coords.begin() + hi);
            SetRuntimeArgs(*ctx.drain_program[d], drain_id, ctx.drisc_logical[d], rt);

            detail::CompileProgram(ctx.device, *ctx.drain_program[d], /*force_slow_dispatch=*/true);
            detail::WriteRuntimeArgsToDevice(ctx.device, *ctx.drain_program[d], /*force_slow_dispatch=*/true);
            detail::LaunchProgram(
                ctx.device, *ctx.drain_program[d], /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] Device {}: DRISC {} failed to start ({}); continuing without capture",
                device_id,
                d,
                e.what());
            ctx.drain_program[d].reset();
            ctx.sockets[d].reset();
            disarm_producers(mesh_device, device_id);
            return false;
        }

        log_info(
            tt::LogMetal,
            "[perf-debug profiler] Device {}: DRISC {} resident on logical ({},{}) [noc0 ({},{})], cores "
            "[{},{}) of {}, {} staging slots x {} B",
            device_id,
            d,
            ctx.drisc_logical[d].x,
            ctx.drisc_logical[d].y,
            drisc_phys.x,
            drisc_phys.y,
            lo,
            hi,
            num_cores,
            nstage,
            slot_bytes);
    }

    return true;
}

// ONE read+decode pass over (ctx, sock_idx): pages -> profzone_decode -> PerfDebugRec -> BroadcastRing.
// Returns true if it moved data. Deliberately does NOT touch Tracy: the sink lives on the consumer thread so
// a slow Tracy push can never back-pressure the FIFO -> relay -> reader -> worker cores. (Measured: with the
// push inline, UFLD-v2 held relay0 in HOST-WAIT 15.85 s of a 19 s run and stalled producers 826x; with the
// push removed, 0 stalls. This is the same structure test_x280_realprof uses.)
// Decode + publish, OFF the reader thread.
//
// MEASURED: with this inline, the reader's ack rate was gated by per-marker work -- read 3.8 ms, decode
// 13.5 ms, publish 8.7 ms, so 85% of host time sat between one ack and the next. At delay 300 that produced
// 17,366 producer stalls where the same device code with a minimal decode produced 0. The copy was never the
// problem (15% of host work, 15-19 GB/s); the interpretation was.
//
// Sequential by construction: ProfzoneDecodeState carries sticky timer highs, the packet residual and the
// per-lane head mirror across buffers, so buffers for one socket MUST be decoded in arrival order. That is
// why there is one decoder per socket rather than a pool.
void PerfDebugProfiler::decode_and_publish(DeviceCtx& ctx, uint32_t sock_idx, std::vector<uint32_t>& buf) {
    DeviceCtx::SockState& ss = ctx.sock_state[sock_idx];
    pz::ProfzoneDecodeState& st = *ctx.decode[sock_idx];
    static const bool ddbg = (std::getenv("TT_PERF_DEBUG_ZONE_DUMP") != nullptr);
    (void)ddbg;
    const auto t_dec_all = std::chrono::steady_clock::now();
    if (stall_only()) {
        pz::profzone_decode(
            st, buf.data(), buf.size(), ctx.nl,
            [&](uint32_t, uint32_t type, uint32_t hash, uint64_t, uint32_t) {
                if (hash == 0x7FFFu && type == PP_ZONE_START) {
                    ss.stall++;
                    w_stalls_++;
                }
                ss.emit++;
            });
        w_decode_ns_ +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_dec_all)
                .count();
        return;
    }
    ss.batch.resize(read_chunk_recs_);  // upper bound on records from one read (words >= records)
    PerfDebugRec* bcur = ss.batch.data();
    PerfDebugRec* const bend = ss.batch.data() + ss.batch.size();
    const uint32_t dev_idx = static_cast<uint32_t>(&ctx - devices_.data());

    ZoneScopedNC("decode", 0x8E44AD);  // purple: pages -> records. With the sink decoupled this plus sock-read
                                       // is the writer's whole job; if it ever fills the thread, the DRAIN is
                                       // the wall (not Tracy) -- the opposite of the UFLD-v2 case.
    const auto t_dec0 = std::chrono::steady_clock::now();
    pz::profzone_decode(
        st,
        buf.data(),
        buf.size(),
        ctx.nl,
        [&](uint32_t lane, uint32_t type, uint32_t hash, uint64_t ts, uint32_t prog) {
            // X280 wire codes, NOT hostdevcommon PacketTypes: the two sources never co-exist and share no
            // decode, so never compare a wire type against a PacketTypes value (they agree at 0/1 only by
            // history). PP_DATA events arrive on the emit_data sink below; PP_ZONE_TOTAL is not a duration.
            if (type != PP_ZONE_START && type != PP_ZONE_END) {
                return;
            }
            ss.emit++;
            if (hash == 0x7FFFu && type == PP_ZONE_START) {
                ss.stall++;  // PROFILER_STALL_ZONE_ID: a producer RISC blocked on a FULL ring. Non-zero means
                             // the capture PERTURBED the workload (kernels elongated); still lossless.
            }
            if (lane / kNRisc >= ctx.core_virt.size() || bcur >= bend) {
                return;
            }
            // Rebase to the first device ts this run sees, so zones land near the Tracy context origin
            // instead of ~device-wall-clock ticks into the timeline.
            if (ctx.marker_ts_base == 0) {
                ctx.marker_ts_base = ts;
            }
            // Pack the device index into the high bits of lane: one ring serves every (device, socket), and
            // the consumer must know which DeviceCtx to resolve coords against. lane = core*NRISC+risc fits
            // comfortably in 24 bits (110 cores * 5 = 550), so the top 8 are free. Keeps PerfDebugRec
            // byte-identical to test_x280_realprof's Rec.
            *bcur++ = PerfDebugRec{ts, (dev_idx << 24) | lane, type, hash, prog};
        },
        // X280 drain-hart spans (bcfg.hartzones). Accumulate; they are placed at stop(), which reads the
        // rdcycle->Tensix calibration. Written only by this (single) writer thread.
        [&](uint32_t hart, uint32_t meta, uint64_t rdc) {
            if (hart < ctx.hz_raw.size()) {
                ctx.hz_raw[hart].push_back(DeviceCtx::HZMark{rdc, meta});
            }
        },
        // PP_DATA point events (DeviceTimestampedData / DeviceRecordEvent). The payload cannot fit in a
        // 24-byte PerfDebugRec, and WIDENING the Rec would cost ~67% more ring bytes on a path that carries
        // ~99M records for a single UFLD-v2 run -- so the payload rides CONTINUATION records instead: one
        // primary (type PP_DATA, zone = id | size<<20) followed by one record per uint64. Events are rare
        // next to zones, so the common path pays nothing and Rec stays byte-identical to the harness Rec.
        [&](uint32_t lane,
            uint32_t type,
            uint32_t id,
            uint64_t ts,
            uint32_t prog,
            const uint32_t* payload,
            uint32_t n) {
            ss.emit++;
            if (lane / kNRisc >= ctx.core_virt.size()) {
                return;
            }
            const uint32_t cont = (n + 1u) / 2u;  // 2 payload words == one uint64 == one continuation rec
            if (bcur + 1 + cont > bend) {
                return;  // batch full: drop this event rather than emit a primary with no payload
            }
            if (ctx.marker_ts_base == 0) {
                ctx.marker_ts_base = ts;
            }
            // `type` (PP_DATA vs PP_EVENT) rides the record so the consumer knows whether the id is a
            // compile-time tag it may name-resolve.
            *bcur++ = PerfDebugRec{ts, (dev_idx << 24) | lane, type, id | (n << 20), prog};
            for (uint32_t k = 0; k < cont; k++) {
                // The producer writes each uint64 hi-word first (see timeStampedData), so recombine in
                // that order and hand the consumer a finished value.
                const uint64_t hi = payload[2 * k];
                const uint64_t lo = (2 * k + 1 < n) ? payload[2 * k + 1] : 0u;
                *bcur++ = PerfDebugRec{(hi << 32) | lo, (dev_idx << 24) | lane, kRecDataCont, 0, prog};
            }
        });

    w_decode_ns_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_dec0).count();

    const size_t bn = static_cast<size_t>(bcur - ss.batch.data());
    w_recs_ += bn;
    if (bn != 0 && ring_) {
        ZoneScopedNC("publish", 0xE67E22);  // orange: publish this read's records to the BroadcastRing
        const auto t_p0 = std::chrono::steady_clock::now();
        ring_->ring.writer().publish_batch(std::span<const PerfDebugRec>(ss.batch.data(), bn));
        w_publish_ns_ +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_p0).count();
    }
}

bool PerfDebugProfiler::drain_pass(DeviceCtx& ctx, uint32_t sock_idx) {
    distributed::D2HSocket* sock = ctx.sockets[sock_idx].get();
    if (sock == nullptr) {
        return false;
    }
    DeviceCtx::SockState& ss = ctx.sock_state[sock_idx];
    // No decode state here any more -- this thread only reads and acks; ProfzoneDecodeState belongs to the
    // decoder thread, which is the only place the stream is interpreted.
    const uint32_t page_words = kPageSize / sizeof(uint32_t);
    static const bool ddbg = (std::getenv("TT_PERF_DEBUG_ZONE_DUMP") != nullptr);

    // Both of these touch the device/FIFO state and run on EVERY pass for EVERY socket, so they set the
    // writer's loop period. They were previously outside any zone, which is how a run could show 79 ms of
    // writer zones inside a 5,433 ms span with the time unattributable.
    uint32_t fifo_pages;
    uint32_t np;
    {
        ZoneScopedNC("sock-poll", 0x16A085);  // teal: "is there anything to read?" -- pure overhead when empty
        const auto t0 = std::chrono::steady_clock::now();
        fifo_pages = sock->get_fifo_curr_size() / sock->get_page_size();
        np = sock->pages_available();
        w_poll_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
                          .count();
        w_polls_++;
    }
    if (np == 0) {
        return false;
    }
    if (np >= fifo_pages) {
        np = fifo_pages - 1u;  // never read more than the FIFO holds (pages_available can spike)
    }
    const uint32_t cap = max_pages_per_read(kMaxPagesPerRead);
    if (cap != 0 && np > cap) {
        np = cap;
    }
    if (ddbg && ss.iters < 40) {
        log_info(tt::LogMetal, "[drain sock={}] iter={} np={} fifo_pages={}", sock_idx, ss.iters, np, fifo_pages);
    }
    ss.iters++;
    ss.pages += np;
    // Take a buffer from the pool. If the decoder has fallen behind and the pool is dry, still read (the
    // FIFO must keep draining or the DRISC stalls) but discard the data and count it.
    std::vector<uint32_t> pooled;
    bool discard = false;
    {
        std::lock_guard<std::mutex> lk(dq_[sock_idx].m);
        DecodeQueue& q = dq_[sock_idx];
        if (!q.free_bufs.empty()) {
            pooled = std::move(q.free_bufs.back());
            q.free_bufs.pop_back();
        } else if (q.allocated < kMaxPooledBufs) {
            q.allocated++;
        } else {
            q.dropped++;
            discard = true;
        }
    }
    std::vector<uint32_t>& dst = discard ? ss.buf : pooled;
    {
        ZoneScopedNC("buf-resize", 0xD35400);
        dst.resize(static_cast<size_t>(np) * page_words);
    }
    {
        ZoneScopedNC("sock-read", 0x27AE60);  // green: pulls pages AND acks the sender -- the critical path
        const auto t0 = std::chrono::steady_clock::now();
        sock->read(dst.data(), np);
        w_read_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
                          .count();
        w_reads_++;
        w_bytes_ += static_cast<uint64_t>(np) * kPageSize;
    }
    if (decode_disabled() || discard) {
        return true;
    }
    // Hand the raw buffer to the decoder and go straight back to polling. This is the whole point: the ack
    // (issued inside sock->read above) is no longer behind 85% of host work.
    {
        std::lock_guard<std::mutex> lk(dq_[sock_idx].m);
        dq_[sock_idx].work.push_back(DecodeItem{&ctx, sock_idx, std::move(pooled)});
    }
    dq_[sock_idx].cv.notify_one();
    return true;
}

// The single writer thread: round-robin every (device, socket); each drain_pass publishes its own read as one
// data-driven batch, then wake readers once per sweep. Idle sweeps back off. Mirrors test_x280_realprof.
void PerfDebugProfiler::decoder_thread(uint32_t sock_idx) {
    tracy::SetThreadName("perf-debug-decoder");
    DecodeQueue& q = dq_[sock_idx];
    for (;;) {
        DecodeItem item;
        {
            std::unique_lock<std::mutex> lk(q.m);
            q.cv.wait(lk, [&q] { return !q.work.empty() || q.quit; });
            if (q.work.empty()) {
                if (q.quit) {
                    return;
                }
                continue;
            }
            q.max_depth = std::max<uint64_t>(q.max_depth, q.work.size());
            item = std::move(q.work.front());
            q.work.pop_front();
        }
        decode_and_publish(*item.ctx, item.sock, item.buf);
        std::lock_guard<std::mutex> lk(q.m);
        item.buf.clear();
        q.free_bufs.push_back(std::move(item.buf));
    }
}

void PerfDebugProfiler::writer_thread(uint32_t sock_idx) {
    tracy::SetThreadName("x280-writer");
    // Startup accounting: the gap between this thread entering and its FIRST successful drain is the window
    // in which the D2H FIFO can fill unserviced -- which back-pressures relay -> reader -> worker rings and
    // stalls every producing RISC once. Reported so it is a number rather than an inference.
    const auto t_writer_entry = std::chrono::steady_clock::now();
    // Wall time of this thread's whole loop. Without it, phase totals cannot distinguish "saturated" from
    // "mostly waiting for the device" -- the exact mistake made when host work time was compared against the
    // DRISC's busy time as though they shared a window.
    const auto t_wall0 = t_writer_entry;
    bool first_data_seen = false;
    auto watchdog = std::chrono::steady_clock::now();
    auto backoff = std::chrono::microseconds(50);
    // Drain-to-empty on stop: stop() sets P_STOP first, so the X280 stops producing; keep reading until every
    // socket has been empty for a sustained window, else the tail of the run is lost. Deadline backstops it.
    constexpr uint32_t kQuiesceEmpties = 200;
    std::chrono::steady_clock::time_point drain_deadline{};
    bool deadline_set = false;
    for (;;) {
        const bool stopping = stop_.load(std::memory_order_acquire);
        if (stopping && !deadline_set) {
            drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            deadline_set = true;
        }
        bool any = false, all_done = true;
        for (auto& ctx : devices_) {
            {
                const uint32_t s = sock_idx;  // one reader per socket -- they never touch each other's state
                DeviceCtx::SockState& ss = ctx.sock_state[s];
                if (ss.done) {
                    continue;
                }
                all_done = false;
                if (drain_pass(ctx, s)) {
                    any = true;
                    ss.quiesce = 0;
                } else if (
                    stopping &&
                    (++ss.quiesce >= kQuiesceEmpties || std::chrono::steady_clock::now() >= drain_deadline)) {
                    ss.done = true;  // stop signalled AND drained (or deadline) => this socket is flushed
                }
            }
        }
        if (any && !first_data_seen) {
            first_data_seen = true;
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_writer_entry).count();
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] writer: first data {:.2f} ms after thread start [large => the FIFO sat "
                "unserviced and producers will have stalled once]",
                ms);
        }
        if (any && ring_) {
            ring_->ring.writer().wake_readers();
        }
        if (all_done) {
            break;
        }
        if (any) {
            watchdog = std::chrono::steady_clock::now();
        } else {
            if (std::chrono::steady_clock::now() - watchdog > std::chrono::seconds(120)) {
                log_warning(tt::LogMetal, "[perf-debug profiler] writer WALL TIMEOUT (120 s no progress)");
                break;
            }
            // Every socket came back empty: the writer is STARVED waiting on the device. If this dominates
            // while the device shows no stalls, the host is comfortably ahead -- the healthy state.
            if (first_data_seen) {
                ZoneScopedNC("sock-idle", 0x7D6608);  // dark yellow: steady-state starvation (healthy)
                std::this_thread::sleep_for(backoff);
            } else {
                // Distinct name: idling BEFORE any data has arrived is the startup window, not steady-state
                // starvation, and only this one can leave the FIFO unserviced while producers fill rings.
                ZoneScopedNC("writer-startup-idle", 0xC0392B);  // red
                std::this_thread::sleep_for(backoff);
            }
        }
    }
    for (auto& ctx : devices_) {
        for (uint32_t s = 0; s < kNSockets; s++) {
            const DeviceCtx::SockState& ss = ctx.sock_state[s];
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] socket {} drained: {} pages, {} markers ({} reads); producer stall "
                "zones: {} [0 = X280 kept up, non-zero = capture perturbed the workload]",
                s,
                ss.pages,
                ss.emit,
                ss.iters,
                ss.stall);
        }
    }
    w_wall_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t_wall0)
                     .count();
}

// BroadcastRing reader -> Tracy. This is the slow side (~270 ns/marker measured), and it is now the ONLY
// thing that suffers when Tracy cannot keep up: it DROPS its own records (reported) instead of stalling the
// device. Runs until the writer is done and the ring is drained.
void PerfDebugProfiler::consumer_thread() {
    tracy::SetThreadName("x280-consume");
    if (!ring_) {
        return;
    }
    auto rd = ring_->ring.make_reader();
    std::vector<PerfDebugRec> scratch(read_chunk_recs_ ? read_chunk_recs_ : 65536);
    uint64_t cnt = 0;
    // PP_DATA reassembly state. Locals (not members) because each consumer thread reads the whole ring
    // independently, and because a primary record and its continuations can straddle a read batch.
    struct PendingEvent {
        bool active = false;
        uint32_t lane_full = 0;
        uint64_t ts = 0;
        uint32_t id = 0;
        uint32_t type = 0;
        uint32_t prog = 0;
        uint32_t want = 0;  // uint64s expected
        uint32_t got = 0;
        uint64_t vals[kMaxEventValues] = {};
    } pend;
    auto emit_batch = [&](std::span<PerfDebugRec> b) {
        // Names are only resolvable once the workload's kernels have JIT-compiled, which has certainly
        // happened by the time records reach us.
        std::call_once(names_once_, [this]() {
            try {
                for (auto& [h, md] : generateZoneSourceLocationsHashes()) {
                    zone_names_[h] = md.marker_name;
                }
            } catch (const std::exception& e) {
                log_warning(tt::LogMetal, "[perf-debug profiler] zone-name load failed ({})", e.what());
            }
            zone_names_[0x7FFFu] = "X280-STALL";  // PROFILER_STALL_ZONE_ID
        });
        ZoneScopedNC("tracy-emit", 0x2980B9);  // blue: pushing this batch into Tracy -- the slow side (~0.8M
                                               // rec/s). When this saturates, the RING drops; it can no longer
                                               // back-pressure the device.
        // Resolve a lane to its coords + name and push the reassembled event. Shares the coord/name
        // resolution shape with the zone path below.
        auto flush_event = [&]() {
            if (!pend.active) {
                return;
            }
            pend.active = false;
            const uint32_t dev_idx = pend.lane_full >> 24;
            const uint32_t lane = pend.lane_full & 0x00FFFFFFu;
            if (dev_idx >= devices_.size()) {
                return;
            }
            DeviceCtx& ctx = devices_[dev_idx];
            const uint32_t ci = lane / kNRisc, risc = lane % kNRisc;
            if (ci >= ctx.core_virt.size()) {
                return;
            }
            const auto [vx, vy] = ctx.core_virt[ci];
            uint32_t nx = vx, ny = vy;
            if (auto it = ctx.virt_to_noc0.find((static_cast<uint64_t>(vx) << 32) | vy); it != ctx.virt_to_noc0.end()) {
                nx = it->second.first;
                ny = it->second.second;
            }
            perf_debug::WorkerEventPacket pkt;
            pkt.chip_id = ctx.chip_id;
            pkt.core_virtual_x = vx;
            pkt.core_virtual_y = vy;
            pkt.core_noc0_x = nx;
            pkt.core_noc0_y = ny;
            pkt.risc = risc;
            pkt.id = pend.id;
            pkt.runtime_id = (pend.type == PP_EVENT);
            // A runtime id is NOT a source-location hash: looking it up would borrow an unrelated zone's
            // name (id 42 vs whatever hashes to 42). Only compile-time tags get resolved.
            if (!pkt.runtime_id) {
                if (auto it = zone_names_.find(static_cast<uint16_t>(pend.id)); it != zone_names_.end()) {
                    pkt.name = it->second;
                }
            }
            const uint64_t base = ctx.synced ? 0 : ctx.marker_ts_base;
            pkt.timestamp = (pend.ts >= base) ? (pend.ts - base) : 0;
            pkt.runtime_host_id = pend.prog;
            pkt.values = pend.vals;
            pkt.num_values = pend.got;
            if (!tracy_push_disabled()) {
                tracy_->HandleWorkerEvent(pkt);
            }
        };

        for (const auto& r : b) {
            if (r.type == kRecDataCont) {
                if (pend.active && pend.got < kMaxEventValues) {
                    pend.vals[pend.got++] = r.ts;
                }
                if (pend.active && pend.got >= pend.want) {
                    flush_event();
                }
                continue;
            }
            if (r.type == PP_DATA || r.type == PP_EVENT) {
                flush_event();  // defensive: a truncated predecessor must not absorb this event's payload
                pend = PendingEvent{};
                pend.active = true;
                pend.lane_full = r.lane;
                pend.ts = r.ts;
                pend.id = r.zone & 0xFFFFFu;
                pend.type = r.type;
                pend.prog = r.prog;
                pend.want = ((r.zone >> 20) + 1u) / 2u;  // payload words -> uint64s
                if (pend.want == 0) {
                    flush_event();  // a bare event (DeviceRecordEvent) has no continuations
                }
                continue;
            }
            if (r.type != PP_ZONE_START && r.type != PP_ZONE_END) {
                continue;  // e.g. PP_ZONE_TOTAL: an accumulated sum, not a duration on the timeline
            }
            const uint32_t dev_idx = r.lane >> 24;
            const uint32_t lane = r.lane & 0x00FFFFFFu;
            if (dev_idx >= devices_.size()) {
                continue;
            }
            DeviceCtx& ctx = devices_[dev_idx];
            const uint32_t ci = lane / kNRisc, risc = lane % kNRisc;
            if (ci >= ctx.core_virt.size()) {
                continue;
            }
            const auto [vx, vy] = ctx.core_virt[ci];
            uint32_t nx = vx, ny = vy;
            if (auto it = ctx.virt_to_noc0.find((static_cast<uint64_t>(vx) << 32) | vy); it != ctx.virt_to_noc0.end()) {
                nx = it->second.first;
                ny = it->second.second;
            }
            std::string_view name;
            if (auto it = zone_names_.find(static_cast<uint16_t>(r.zone)); it != zone_names_.end()) {
                name = it->second;
            }
            perf_debug::WorkerZonePacket pkt;
            pkt.chip_id = ctx.chip_id;
            pkt.core_virtual_x = vx;
            pkt.core_virtual_y = vy;
            pkt.core_noc0_x = nx;
            pkt.core_noc0_y = ny;
            pkt.risc = risc;
            pkt.timer_id = r.zone;
            pkt.name = name;
            // Synced: push the RAW device timestamp -- the context was anchored with a real (host, device)
            // pair, so Tracy places it exactly. Unsynced: fall back to rebasing on the first marker seen.
            const uint64_t base = ctx.synced ? 0 : ctx.marker_ts_base;
            pkt.timestamp = (r.ts >= base) ? (r.ts - base) : 0;
            pkt.is_start = (r.type == PP_ZONE_START);
            if (!tracy_push_disabled()) {
                tracy_->HandleWorkerZone(pkt);
            }
        }
        cnt += b.size();
    };
    for (;;) {
        auto tok = rd.wait_token();
        auto got = rd.read_batch(std::span<PerfDebugRec>(scratch));
        if (!got.empty()) {
            emit_batch(got);
            continue;
        }
        if (writer_done_.load(std::memory_order_acquire)) {  // writer finished -> drain the tail, then exit
            for (;;) {
                auto g = rd.read_batch(std::span<PerfDebugRec>(scratch));
                if (g.empty()) {
                    break;
                }
                emit_batch(g);
            }
            break;
        }
        {
            ZoneScopedNC("ring-wait", 0x7F8C8D);  // gray: consumer starved, waiting on the writer. Mirrors the
                                                  // harness's mq-pop-wait. Plentiful here = the sink is keeping
                                                  // up; ~absent = the sink is the bottleneck and drops loom.
            rd.wait(tok);
        }
    }
    consumed_.fetch_add(cnt);
    dropped_.fetch_add(rd.dropped());
}

void PerfDebugProfiler::stop() {
    if (stopped_.exchange(true)) {
        return;
    }
    // Tell each DRISC to quiesce, then wait for it to publish `done` -- which it does only after its
    // socket barrier, so every page is already on its way to the host when we stop reading.
    for (auto& ctx : devices_) {
        auto& cluster = MetalContext::instance().get_cluster();
        for (uint32_t d = 0; d < kNSockets; d++) {
            if (ctx.drain_program[d] == nullptr) {
                continue;
            }
                const tt_cxy_pair drisc(ctx.chip_id, ctx.drisc_virtual[d]);
            uint32_t one = 1;
            cluster.write_core(&one, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            uint32_t done = 0;
            while (std::chrono::steady_clock::now() < deadline) {
                cluster.read_core(&done, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
                if ((done & 0xFFFF0000u) == 0xD09E0000u) {
                    break;
                }
                // The writer thread is still draining, so the socket keeps emptying while we wait.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if ((done & 0xFFFF0000u) != 0xD09E0000u) {
                log_warning(
                    tt::LogMetal, "[perf-debug profiler] Device {}: DRISC drainer did not acknowledge stop", ctx.chip_id);
            }
            // The drainer's own view of the run. Host-side page and marker counts cannot distinguish a
            // bandwidth wall from a latency one; sweeps/frames/cycles can.
            std::vector<uint32_t> res(33, 0);
            cluster.read_core(
                res.data(),
                res.size() * sizeof(uint32_t),
                drisc,
                ctx.drisc_l1_noc[d] + (ctx.results_addr[d] - ctx.drisc_l1_base[d]));
            auto u64 = [&res](size_t i) { return (static_cast<uint64_t>(res[i + 1]) << 32) | res[i]; };
            const uint64_t cyc = u64(0);
            const uint64_t dw = u64(2);
            const double kCycPerUs = 1.35e3;  // the drainer stamps with the Tensix wall clock
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] DRISC: {} sweeps ({} idle), {} frames, {} pushes, {} words, {} pages, "
                "max occ {}/{}, overflows {}",
                res[4],
                res[20],
                res[6],
                res[9],
                dw,
                res[5],
                res[7],
                kernel_profiler::PROFILER_L1_VECTOR_SIZE,
                res[8]);
            // Per-phase, so a lifetime average can never again hide that empty and loaded sweeps differ by
            // orders of magnitude. `reserve` is the host FIFO credit wait: if that dominates, the DRISC is not
            // the bottleneck at all -- the host consumer is.
            const uint64_t c_read = u64(10), c_proc = u64(12), c_res = u64(14), c_wr = u64(16), c_bar = u64(18);
            const uint64_t c_idle = u64(21), c_busy = u64(23);
            const uint64_t acct = c_read + c_proc + c_res + c_wr + c_bar;
            auto pct = [cyc](uint64_t v) {
                return cyc ? (100.0 * static_cast<double>(v) / static_cast<double>(cyc)) : 0.0;
            };
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] DRISC phases of {:.1f} ms: read {:.1f}% | proc {:.1f}% | "
                "reserve(credit-wait) {:.1f}% | write {:.1f}% | wr-barrier {:.1f}% | unaccounted {:.1f}%",
                cyc / kCycPerUs / 1000.0,
                pct(c_read),
                pct(c_proc),
                pct(c_res),
                pct(c_wr),
                pct(c_bar),
                pct(cyc > acct ? cyc - acct : 0));
            const uint32_t sweeps_busy = res[4] > res[20] ? res[4] - res[20] : 0;
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] DRISC sweeps: idle {} @ {:.1f} us | busy {} @ {:.1f} us | worst sweep "
                "{:.1f} us | worst credit-wait {:.1f} us",
                res[20],
                res[20] ? (static_cast<double>(c_idle) / kCycPerUs) / res[20] : 0.0,
                sweeps_busy,
                sweeps_busy ? (static_cast<double>(c_busy) / kCycPerUs) / sweeps_busy : 0.0,
                res[25] / kCycPerUs,
                res[26] / kCycPerUs);
            // write sub-split. Exact per busy sweep: ship_run only executes when a frame is being sent.
            const uint64_t c_chunk = u64(27), c_push = u64(29), c_notify = u64(31);
            const double pu = res[9] ? static_cast<double>(res[9]) : 1.0;  // pushes
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] DRISC write split over {} pushes: noc-chunk {:.2f} us/push ({:.1f} ms) | "
                "push_pages {:.2f} us/push ({:.1f} ms) | notify {:.2f} us/push ({:.1f} ms)",
                res[9],
                (c_chunk / kCycPerUs) / pu,
                c_chunk / kCycPerUs / 1000.0,
                (c_push / kCycPerUs) / pu,
                c_push / kCycPerUs / 1000.0,
                (c_notify / kCycPerUs) / pu,
                c_notify / kCycPerUs / 1000.0);

            // Release it to restore the NIU. It cannot do that until we say so: NOC2AXI forwards inbound
            // DRAM-range addresses to GDDR, so the flip takes this L1 out of the host's view.
            uint32_t two = 2;
            cluster.write_core(&two, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
        }
    }
    stop_.store(true, std::memory_order_release);
    for (auto& w : writers_) {
        if (w.joinable()) {
            w.join();  // each drains its own socket to quiescence
        }
    }
    // ---- the knee metric, read straight out of every worker's L1 ----
    //
    // No decode required, so this is valid in NO_DECODE mode where the host does nothing but read and ack.
    // The producer increments its own counter when it blocks, so nothing downstream can lose it.
    for (auto& ctx : devices_) {
        if (ctx.core_virt.empty()) {
            continue;
        }
        auto& cluster = MetalContext::instance().get_cluster();
        const auto& hal = MetalContext::instance().hal();
        const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
        std::vector<uint32_t> cv(kernel_profiler::SPSC_CONTROL_END, 0);
        uint64_t total = 0, worst = 0, cores_hit = 0;
        for (const auto& [vx, vy] : ctx.core_virt) {
            cluster.read_core(
                cv.data(),
                kernel_profiler::SPSC_CONTROL_END * sizeof(uint32_t),
                tt_cxy_pair(ctx.chip_id, CoreCoord{vx, vy}),
                prof_l1);
            uint64_t core_total = 0;
            for (uint32_t r = 0; r < kernel_profiler::SPSC_STALL_COUNT_MAX; r++) {
                core_total += cv[kernel_profiler::SPSC_STALL_COUNT_0 + r];
            }
            total += core_total;
            worst = std::max(worst, core_total);
            cores_hit += (core_total != 0);
        }
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] Device {}: L1 STALL COUNTERS -- {} producer stalls across {} of {} cores "
            "(worst core {}) [0 = the capture did not perturb the workload]",
            ctx.chip_id,
            total,
            cores_hit,
            ctx.core_virt.size(),
            worst);
    }

    // Drain every decode queue before declaring the writers done, else the tail of the run is lost.
    for (uint32_t s = 0; s < kNSockets; s++) {
        std::lock_guard<std::mutex> lk(dq_[s].m);
        dq_[s].quit = true;
    }
    for (uint32_t s = 0; s < kNSockets; s++) {
        dq_[s].cv.notify_all();
    }
    for (auto& d : decoders_) {
        if (d.joinable()) {
            d.join();
        }
    }
    uint64_t q_dropped = 0, q_depth = 0;
    for (uint32_t s = 0; s < kNSockets; s++) {
        q_dropped += dq_[s].dropped;
        q_depth = std::max(q_depth, dq_[s].max_depth);
    }
    if (q_dropped != 0) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] decoders fell behind: {} reads discarded (pool of {} per socket "
            "exhausted); capture is incomplete but the workload was not perturbed",
            q_dropped,
            kMaxPooledBufs);
    }
    log_info(tt::LogMetal, "[perf-debug profiler] decode queues: max depth {} buffers", q_depth);
    writer_done_.store(true, std::memory_order_release);
    // Why the host is the wall. The egress-only benchmark moved bytes with a copy and no interpretation;
    // this thread also DECODES every marker and publishes every record, on the same thread that issues the
    // socket acks -- so the ack rate, and hence the sender's credit wait, is gated by per-marker work.
    if (w_reads_ != 0) {
        const double read_ms = w_read_ns_ / 1e6, dec_ms = w_decode_ns_ / 1e6, pub_ms = w_publish_ns_ / 1e6;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] host writer: {} reads, {:.1f} MB, {} records | sock-read {:.1f} ms "
            "({:.2f} GB/s) | decode {:.1f} ms ({:.1f} ns/marker) | publish {:.1f} ms",
            w_reads_,
            w_bytes_ / (1024.0 * 1024.0),
            w_recs_,
            read_ms,
            read_ms > 0 ? (w_bytes_ / 1e9) / (read_ms / 1e3) : 0.0,
            dec_ms,
            w_recs_ ? w_decode_ns_ / static_cast<double>(w_recs_) : 0.0,
            pub_ms);
        const double wall_ms = w_wall_ns_ / 1e6;
        const double work_ms = (w_read_ns_ + w_decode_ns_ + w_publish_ns_ + w_poll_ns_) / 1e6;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] host writer wall {:.1f} ms: poll {:.1f}% ({} polls) | sock-read {:.1f}% | "
            "decode {:.1f}% | publish {:.1f}% | IDLE/other {:.1f}% -- {:.0f}% busy",
            wall_ms,
            wall_ms > 0 ? 100.0 * (w_poll_ns_ / 1e6) / wall_ms : 0.0,
            w_polls_,
            wall_ms > 0 ? 100.0 * (w_read_ns_ / 1e6) / wall_ms : 0.0,
            wall_ms > 0 ? 100.0 * (w_decode_ns_ / 1e6) / wall_ms : 0.0,
            wall_ms > 0 ? 100.0 * (w_publish_ns_ / 1e6) / wall_ms : 0.0,
            wall_ms > 0 ? 100.0 * (wall_ms - work_ms) / wall_ms : 0.0,
            wall_ms > 0 ? 100.0 * work_ms / wall_ms : 0.0);
        if (stall_only()) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] STALL-ONLY decode: {} producer stall zones (knee metric; no records "
                "built, no Tracy)",
                w_stalls_);
        }
        uint64_t drift = 0;
        for (auto& ctx : devices_) {
            for (auto& d : ctx.decode) {
                if (d) {
                    drift += d->head_drift;
                }
            }
        }
        if (drift != 0) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] head-mirror drift on {} frames -- the drainer's write-back lagged a "
                "snapshot, or a frame was lost",
                drift);
        }
    }
    if (ring_) {
        ring_->ring.writer().wake_readers();  // unblock a consumer parked in wait()
    }
    for (auto& c : consumers_) {
        if (c.joinable()) {
            c.join();
        }
    }
    if (ring_) {
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] BroadcastRing: cap {} records; consumer took {} records, dropped {} "
            "[0 dropped => the Tracy sink kept up]",
            ring_capacity_recs(),
            consumed_.load(),
            dropped_.load());
    }
    push_hart_zones();  // must run BEFORE tracy_.reset() -- it creates/uses Tracy contexts
    tracy_.reset();
    devices_.clear();
}

// Map the collected X280 drain-hart spans onto the device timeline and push them to Tracy as their own
// per-hart lanes ("rd0/rd1/relay0/relay1" in the GUI, via the widened RiscType). Runs once, at stop(), after
// the drain threads have joined so hz_raw is stable.
//
// The harts timestamp themselves with rdcycle (a fixed 1 GHz counter), NOT the Tensix wall clock the kernel
// markers use, so the two cannot be compared directly. hart0 therefore co-samples both clocks at boot (that
// is what bcfg.hartzones also switches on) and the host least-squares fits tensix = a*rdcycle + b here.
// Timestamps are rebased on the harts' OWN minimum, which cancels the constant offset between the
// calibration reference core's raw wall clock and the marker timeline (the same per-node origin trick the
// standalone harness uses; without it the whole lane can land before the origin and clamp to zero).
void PerfDebugProfiler::push_hart_zones() {
    if (!tracy_ || !hart_zones_enabled()) {
        return;
    }
    for (auto& ctx : devices_) {
        if (!ctx.driver || ctx.hz_raw.empty()) {
            continue;
        }
        uint64_t total = 0;
        for (const auto& v : ctx.hz_raw) {
            total += v.size();
        }
        if (total == 0) {
            log_warning(tt::LogMetal, "[perf-debug profiler] hart zones enabled but none were captured.");
            continue;
        }
        // ---- rdcycle -> Tensix fit from hart0's boot co-samples {rdcycle_mid, tensix, noc_round_trip} ----
        const uint32_t nc = pz::kProfzoneCalibN;
        std::vector<uint64_t> raw(static_cast<size_t>(nc) * 3);
        try {
            ctx.driver->read_block(raw.data(), nc * 3 * sizeof(uint64_t), pz::kProfzoneCalibBase);
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[perf-debug profiler] hart-zone calib read failed ({})", e.what());
            continue;
        }
        std::vector<uint64_t> rts(nc);
        for (uint32_t i = 0; i < nc; i++) {
            rts[i] = raw[i * 3 + 2];
        }
        std::sort(rts.begin(), rts.end());
        const uint64_t rt_cut = rts[nc / 2] + rts[nc / 2] / 2;  // drop NoC-contended outliers
        const uint64_t x_base = raw[0], t_base = raw[1];
        double sx = 0, st = 0, sxx = 0, sxt = 0;
        uint32_t nfit = 0;
        for (uint32_t i = 0; i < nc; i++) {
            if (raw[i * 3 + 2] > rt_cut) {
                continue;
            }
            const double x = static_cast<double>(raw[i * 3 + 0] - x_base);
            const double t = static_cast<double>(raw[i * 3 + 1] - t_base);
            sx += x;
            st += t;
            sxx += x * x;
            sxt += x * t;
            nfit++;
        }
        if (nfit < 2 || (sxx * nfit - sx * sx) == 0.0) {
            log_warning(tt::LogMetal, "[perf-debug profiler] hart-zone calib unusable (nfit={})", nfit);
            continue;
        }
        const double a = (sxt * nfit - sx * st) / (sxx * nfit - sx * sx);
        const double b = (st - a * sx) / nfit;
        auto map_ts = [&](uint64_t x) -> uint64_t {
            return static_cast<uint64_t>(a * static_cast<double>(x - x_base) + b) + t_base;
        };
        // Rebase on the MARKER origin so hart spans and kernel zones share one timeline. Falls back to the
        // harts' own minimum only if no markers were captured (nothing to align to).
        uint64_t hz_min = ~0ull, hz_max = 0;
        for (const auto& v : ctx.hz_raw) {
            for (const auto& m : v) {
                const uint64_t t = map_ts(m.rdc);
                hz_min = std::min(hz_min, t);
                hz_max = std::max(hz_max, t);
            }
        }
        // Synced devices place hart spans on the same absolute Tensix timeline as the markers (no rebase).
        const uint64_t marker_origin = ctx.synced ? 0 : ctx.marker_ts_base;
        const uint64_t hz_span_start = hz_min;  // kept for the diagnostic below
        if (ctx.synced) {
            hz_min = 0;  // absolute Tensix time; the context anchor does the placement
        } else if (marker_origin != 0) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] hart zones start {:.1f} ms BEFORE the first kernel marker (X280 drains "
                "from MeshDevice bring-up); aligning both on the marker origin",
                static_cast<double>(marker_origin - std::min(marker_origin, hz_min)) / 1.35e6);
            hz_min = marker_origin;
        }
        (void)hz_span_start;
        const CoreCoord l2t = pz::x280_l2cpu_tile(0);
        static constexpr uint32_t kHartColor[4] = {0xE67E22u, 0xF1C40Fu, 0x1ABC9Cu, 0x3498DBu};
        static constexpr uint32_t kBulkColor = 0xE74C3Cu;      // reader switched to BULK
        static constexpr uint32_t kHostWaitColor = 0x34495Eu;  // relay blocked on a full host FIFO
        static constexpr uint32_t kSpscWaitColor = 0x8E44ADu;  // reader blocked on a full LIM STAGE
        // ★ Push ALL harts as ONE CHRONOLOGICALLY SORTED stream, not hart-by-hart. Every hart lane lives in
        // the SAME Tracy GPU context, and Tracy's calibrated GPU-zone path expects timestamps to arrive in
        // increasing order per context. Pushing all of hart0, then all of hart1, ... makes the context's clock
        // jump backwards once per hart, which renders the lanes as separate blocks marched across the timeline
        // even though their real windows fully overlap (verified: identical 10-bucket time histograms, all
        // harts active throughout, min/max windows within 4 ms of each other).
        struct HZItem {
            uint64_t ts;
            uint32_t hart;
            uint32_t meta;
        };
        std::vector<HZItem> ordered;
        ordered.reserve(total);
        for (uint64_t h = 0; h < ctx.hz_raw.size(); h++) {
            for (const auto& m : ctx.hz_raw[h]) {
                ordered.push_back(HZItem{map_ts(m.rdc), static_cast<uint32_t>(h), m.meta});
            }
        }
        // stable_sort: a hart's own START/END pair share no timestamp ordering guarantee otherwise, and equal
        // timestamps must keep their emission order or a zone can close before it opens.
        std::stable_sort(ordered.begin(), ordered.end(), [](const HZItem& a, const HZItem& b) { return a.ts < b.ts; });
        std::vector<uint32_t> nz_per_hart(ctx.hz_raw.size(), 0);
        // Per-hart, per-KIND accounting (0=DRAIN 1=BULK 2=HOST-WAIT 3=SPSC-WAIT). Counts alone mislead -- a few
        // very long HOST-WAITs matter more than many short drains -- so accumulate occupancy too. This is what
        // answers "is the host still the wall, or has the bottleneck moved upstream to the reader?"
        const size_t nh = ctx.hz_raw.size();
        std::vector<std::array<uint64_t, 4>> kcount(nh, {0, 0, 0, 0});
        std::vector<std::array<uint64_t, 4>> kcyc(nh, {0, 0, 0, 0});
        std::vector<std::array<uint64_t, 4>> kopen(nh, {0, 0, 0, 0});
        for (const auto& it : ordered) {
            const uint64_t h = it.hart;
            const bool is_reader = (h < kNRead);
            const std::string hname =
                is_reader ? ("X280 rd" + std::to_string(h)) : ("X280 relay" + std::to_string(h - kNRead));
            const uint32_t lane_risc = is_reader ? (static_cast<uint32_t>(tracy::RiscType::X280_RD0) + h)
                                                 : (static_cast<uint32_t>(tracy::RiscType::X280_RELAY0) + (h - kNRead));
            const uint32_t is_start = it.meta & 1u;
            const uint32_t kind = (it.meta >> 1) & 3u;
            const char* suffix = (kind == 1) ? " BULK" : (kind == 2) ? " HOST-WAIT" : (kind == 3) ? " SPSC-WAIT" : "";
            perf_debug::WorkerZonePacket pkt;
            pkt.chip_id = ctx.chip_id;
            pkt.is_x280 = true;
            pkt.color = (kind == 1)   ? kBulkColor
                        : (kind == 2) ? kHostWaitColor
                        : (kind == 3) ? kSpscWaitColor
                                      : kHartColor[h & 3];
            pkt.core_noc0_x = static_cast<uint32_t>(l2t.x);
            pkt.core_noc0_y = static_cast<uint32_t>(l2t.y);
            pkt.risc = lane_risc;
            const std::string zn = hname + suffix;
            pkt.name = zn;
            pkt.timestamp = (it.ts >= hz_min) ? (it.ts - hz_min) : 0;
            pkt.is_start = (is_start != 0u);
            tracy_->HandleWorkerZone(pkt);
            nz_per_hart[h] += is_start;
            if (is_start) {
                kcount[h][kind]++;
                kopen[h][kind] = it.ts;
            } else if (kopen[h][kind] != 0) {
                kcyc[h][kind] += (it.ts > kopen[h][kind]) ? (it.ts - kopen[h][kind]) : 0;
                kopen[h][kind] = 0;
            }
        }
        for (uint64_t h = 0; h < nz_per_hart.size(); h++) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] hart {} ({}): {} zones | DRAIN {}x/{:.0f}ms  BULK {}x/{:.0f}ms  "
                "HOST-WAIT {}x/{:.0f}ms  SPSC-WAIT {}x/{:.0f}ms",
                h,
                (h < kNRead) ? "READ" : "RELAY",
                nz_per_hart[h],
                kcount[h][0],
                static_cast<double>(kcyc[h][0]) / 1.35e6,
                kcount[h][1],
                static_cast<double>(kcyc[h][1]) / 1.35e6,
                kcount[h][2],
                static_cast<double>(kcyc[h][2]) / 1.35e6,
                kcount[h][3],
                static_cast<double>(kcyc[h][3]) / 1.35e6);
        }
        for (uint64_t h = 0; h < 0; h++) {  // (old per-hart push loop retained below only for its diagnostics)
            const bool is_reader = (h < kNRead);
            const std::string hname =
                is_reader ? ("X280 rd" + std::to_string(h)) : ("X280 relay" + std::to_string(h - kNRead));
            const uint32_t lane_risc = is_reader ? (static_cast<uint32_t>(tracy::RiscType::X280_RD0) + h)
                                                 : (static_cast<uint32_t>(tracy::RiscType::X280_RELAY0) + (h - kNRead));
            uint32_t nz = 0, n_end = 0;
            int depth_dbg = 0, max_depth_dbg = 0, unbalanced_dbg = 0;
            for (const auto& m : ctx.hz_raw[h]) {
                const uint32_t is_start = m.meta & 1u;
                const uint32_t kind = (m.meta >> 1) & 3u;  // 0=drain 1=bulk 2=hostwait 3=spscwait
                const char* suffix = (kind == 1)   ? " BULK"
                                     : (kind == 2) ? " HOST-WAIT"
                                     : (kind == 3) ? " SPSC-WAIT"
                                                   : "";
                const std::string zn = hname + suffix;
                const uint64_t ts = map_ts(m.rdc);
                perf_debug::WorkerZonePacket pkt;
                pkt.chip_id = ctx.chip_id;
                pkt.is_x280 = true;
                pkt.color = (kind == 1)   ? kBulkColor
                            : (kind == 2) ? kHostWaitColor
                            : (kind == 3) ? kSpscWaitColor
                                          : kHartColor[h & 3];
                pkt.core_noc0_x = static_cast<uint32_t>(l2t.x);
                pkt.core_noc0_y = static_cast<uint32_t>(l2t.y);
                pkt.risc = lane_risc;
                pkt.name = zn;
                pkt.timestamp = (ts >= hz_min) ? (ts - hz_min) : 0;
                pkt.is_start = (is_start != 0u);
                tracy_->HandleWorkerZone(pkt);
                nz += is_start;
                // DIAG: START/END balance. Unbalanced pairs make the Tracy handler nest ever deeper instead of
                // closing zones, which renders as giant cascading boxes. Under drain SATURATION the FW may not
                // have room to inject the closing marker, so this is the prime suspect there.
                if (is_start) {
                    depth_dbg++;
                    max_depth_dbg = std::max(max_depth_dbg, depth_dbg);
                } else {
                    n_end++;
                    if (depth_dbg == 0) {
                        unbalanced_dbg++;  // END with no open START (orphan)
                    } else {
                        depth_dbg--;
                    }
                }
            }
            // DIAG: per-hart mapped window. All four harts run CONCURRENTLY for the whole session, so these
            // windows must OVERLAP. If they are disjoint/staggered, the per-hart rdcycle counters are not on a
            // common origin and hart0's calibration cannot be applied to harts 1..3 as-is.
            uint64_t h_min = ~0ull, h_max = 0;
            for (const auto& m : ctx.hz_raw[h]) {
                const uint64_t t = map_ts(m.rdc);
                h_min = std::min(h_min, t);
                h_max = std::max(h_max, t);
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] hart {} ({}): {} zones -> Tracy; window [{:.1f} .. {:.1f}] ms rel, raw_rdc0={}",
                h,
                is_reader ? "READ" : "RELAY",
                nz,
                ctx.hz_raw[h].empty() ? 0.0 : static_cast<double>(h_min - hz_min) / 1.35e6,
                ctx.hz_raw[h].empty() ? 0.0 : static_cast<double>(h_max - hz_min) / 1.35e6,
                ctx.hz_raw[h].empty() ? 0ull : ctx.hz_raw[h].front().rdc);
            // DIAG: where in time this hart's zones actually LIVE (10 equal buckets over the whole hart-zone
            // span). If lanes look "staggered" in the GUI but their min/max windows overlap, the answer is
            // here: a lane whose zones all sit in one bucket renders as one narrow merged box at that offset.
            {
                const uint64_t span = (hz_max > hz_min) ? (hz_max - hz_min) : 1;
                uint32_t bucket[10] = {0};
                for (const auto& m : ctx.hz_raw[h]) {
                    const uint64_t t = map_ts(m.rdc);
                    const uint64_t rel = (t > hz_min) ? (t - hz_min) : 0;
                    uint32_t bi = static_cast<uint32_t>((rel * 10) / span);
                    bucket[std::min<uint32_t>(bi, 9)]++;
                }
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler]   hart {} TIME-HIST(10 buckets): {} {} {} {} {} {} {} {} {} {}",
                    h,
                    bucket[0],
                    bucket[1],
                    bucket[2],
                    bucket[3],
                    bucket[4],
                    bucket[5],
                    bucket[6],
                    bucket[7],
                    bucket[8],
                    bucket[9]);
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler]   hart {} BALANCE: starts={} ends={} left_open={} orphan_ends={} max_depth={}",
                h,
                nz,
                n_end,
                depth_dbg,
                unbalanced_dbg,
                max_depth_dbg);
        }
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] hart zones: {} spans over {:.1f} ms (a={:.5f}, {} calib samples kept)",
            total,
            static_cast<double>(hz_max - hz_min) / 1.35e6,
            a,
            nfit);
    }
}

}  // namespace tt::tt_metal
