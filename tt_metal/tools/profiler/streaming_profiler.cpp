// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/streaming_profiler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <set>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tracy/Tracy.hpp>
#include <common/TracyTTDeviceData.hpp>  // tracy::RiscType worker lanes

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
#include "hostdevcommon/profiler_common.h"

#include "tools/profiler/spsc_marker_decode.hpp"
#include "tools/profiler/streaming_profiler_consumer.hpp"
#include "tools/profiler/streaming_profiler_env.hpp"
#include "tools/profiler/streaming_profiler_tracy_handler.hpp"
#include "tools/profiler/streaming_profiler_receiver.hpp"
#include "tools/profiler/streaming_profiler_tracy_consumer.hpp"
#include "llrt/zone_meta.hpp"  // per-ELF (zone id -> source location), the streaming name source
#include "tools/profiler/spsc_packet.h"

namespace tt::tt_metal {

namespace pz = tt::tt_metal::profiler;

namespace {

// TT_METAL_STREAMING_PROFILER_NO_STATIC_TLB: skip configuring a static TLB window for the DRISC relay,
// leaving the socket's ack write on UMD's dynamic (reconfigure-per-access) path.
bool no_static_tlb() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_NO_STATIC_TLB");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_NOC forces which NIU every relay egresses on (reads take the other); unset =
// NOC 0 for all of them. Sharing one egress NIU beats spreading the relays over both: NOC 1 egress runs
// ~2x the service interval of NOC 0, so a relay parked there takes essentially every producer stall.
// The socket's NOC0-derived PCIe encoding is correct on both NoCs, because the PCIe tile lives in
// translated space and the coordinate mirroring that applies to worker coords does not apply to it.
int relay_noc_override() {
    static const int v = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_NOC");
        return (s == nullptr || *s == '\0') ? -1 : (std::strtoul(s, nullptr, 10) == 1 ? 1 : 0);
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_RESERVE_COLUMN: under slow dispatch, hold the last worker column back and
// poll only 11x10=110 instead of the full 12x10=120. Not a functional requirement -- the relay lives on a
// DRAM core, so no worker is spent on it -- but a fixed poll-list length makes sweep costs comparable
// across runs.
bool reserve_column_env() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_RESERVE_COLUMN");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return v;
}

const std::vector<uint32_t>& relay_vcs() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_RELAY_VCS");
        if (s != nullptr && *s != '\0') {
            const char* p = s;
            while (*p != '\0') {
                out.push_back(static_cast<uint32_t>(std::strtoul(p, nullptr, 10)) & 3u);
                while (*p != '\0' && *p != ',') {
                    p++;
                }
                if (*p == ',') {
                    p++;
                }
            }
        }
        return out;
    }();
    return v;
}

// DRAM view ids the relays occupy, one per relay. Views 7 and 2 sit at the end of the default roster so
// that a roster truncated to fewer relays sheds the historically bring-up-fragile views first.
const std::vector<uint32_t>& relay_banks() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_RELAY_BANKS");
        if (s != nullptr && *s != '\0') {
            const char* p = s;
            while (*p != '\0') {
                out.push_back(static_cast<uint32_t>(std::strtoul(p, nullptr, 10)));
                while (*p != '\0' && *p != ',') {
                    p++;
                }
                if (*p == ',') {
                    p++;
                }
            }
        }
        if (out.empty()) {
            out = {5u, 6u, 4u, 1u, 0u, 3u, 7u, 2u};
        }
        return out;
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_NRELAYS forces the relay count, 1..kMaxRelays; 0 = unset, boot_device then takes
// min(kMaxRelays, DRAM views). A forced value above the view count is clamped there.
uint32_t n_relays_env(uint32_t max_relays) {
    const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_NRELAYS");
    if (s == nullptr || *s == '\0') {
        return 0u;
    }
    char* end = nullptr;
    const unsigned long n = std::strtoul(s, &end, 10);
    TT_FATAL(
        end != s && *end == '\0' && n >= 1 && n <= max_relays,
        "TT_METAL_STREAMING_PROFILER_NRELAYS='{}' is not an integer in [1, {}]",
        s,
        max_relays);
    return static_cast<uint32_t>(n);
}

// TT_METAL_STREAMING_PROFILER_NSTAGE: cap on staging slots. Default 7, which is what a DRISC's L1 fits.
uint32_t nstage_cap(uint32_t computed) {
    const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_NSTAGE");
    const uint32_t cap = (s == nullptr || *s == '\0') ? 7u : static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    return (cap != 0 && computed > cap) ? cap : computed;
}

}  // namespace

// TT_METAL_STREAMING_PROFILER_SHIP_MIN_PCT: a relay defers shipping a live core until its fullest lane
// holds at least this percent of its own ring, unless the core aged out. 0 ships every live core every
// sweep; values past 50 are capped by the kernel's half-ring lane trigger. Per-lane, not per-span: the
// producer that blocks is always a lane, and a span percent under-reads a concentrated core's binding
// ring by up to kNumRisc. Default 25 -- the measured stall-free band ends between 30 and 35.
uint32_t ship_min_pct() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_SHIP_MIN_PCT");
        const uint32_t n = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 25u;
        return n > 100 ? 100u : n;
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_DRAM_MB: per-relay GDDR spool ring, in MiB. Non-zero makes each relay DMA
// frames into a ring in its own DRAM bank and forward them to the host FIFO from a non-blocking pump, so
// the service loop never touches the PCIe tile and host-side pressure lands in spool occupancy instead of
// in the sweep interval. 0 = direct push.
uint32_t dram_spool_mb() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_DRAM_MB");
        const uint32_t n = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 128u;
        // A ring beyond 4095 MiB would overflow the kernel's 32-bit ring arithmetic (a bank is 4 GiB anyway).
        return n > 4095u ? 4095u : n;
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_RELAY_SLICE_MAP: comma-separated permutation, entry d = which core band relay
// d takes. Unset = identity. Bands are row-major slices assigned with no reference to where the relay's own
// DRAM core sits, so the default pairing is arbitrary.
const std::vector<uint32_t>& relay_slice_map() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_RELAY_SLICE_MAP");
        if (s != nullptr && *s != '\0') {
            for (const char* p = s; *p != '\0';) {
                out.push_back(static_cast<uint32_t>(std::strtoul(p, nullptr, 10)));
                while (*p != '\0' && *p != ',') {
                    p++;
                }
                if (*p == ',') {
                    p++;
                }
            }
        }
        return out;
    }();
    return v;
}

// Per-relay core-count weights from TT_METAL_STREAMING_PROFILER_RELAY_WEIGHTS, indexed by relay slot (not
// bank id). Relays are not equally fast -- an even split measured a 45% spread in service interval, and the
// knee follows the slowest relay -- so weighting core count by 1/interval equalises the intervals instead
// of the counts. The weights are board-specific calibration, so the default stays even.
const std::vector<uint32_t>& relay_weights() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_RELAY_WEIGHTS");
        if (s != nullptr && *s != '\0') {
            for (const char* p = s; *p != '\0';) {
                out.push_back(static_cast<uint32_t>(std::strtoul(p, nullptr, 10)));
                while (*p != '\0' && *p != ',') {
                    p++;
                }
                if (*p == ',') {
                    p++;
                }
            }
            return out;
        }
        return out;  // empty = even split; the use site treats it that way
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_RELAY_SUBCH: comma-separated DRAM subchannel index per relay. Each DRAM view
// has three subchannels at quite different NoC positions (bank 5: NOC0 (9,2)/(9,10)/(9,3); bank 3:
// (0,5)/(0,7)/(0,6)), and pick_unused_dram_logical_core() returns whichever is first unreserved -- nothing
// picks for locality. Unset keeps that behaviour.
const std::vector<uint32_t>& relay_subchannels() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_RELAY_SUBCH");
        if (s != nullptr && *s != '\0') {
            for (const char* p = s; *p != '\0';) {
                out.push_back(static_cast<uint32_t>(std::strtoul(p, nullptr, 10)));
                while (*p != '\0' && *p != ',') {
                    p++;
                }
                if (*p == ',') {
                    p++;
                }
            }
        }
        return out;
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_RELAY_ASSIGN=xsplit: assign worker cores to relays by NoC reachability instead
// of row-major index order. Round-trip latency is position-independent on these unidirectional tori (a
// request costs dx+dy hops and its response (17-dx)+(12-dy), always 29), but link occupancy is not: reads
// ride NoC 1 (-x/-y) and DRAM sits in NOC0 columns x=0 (views D0-D3) and x=9 (views D4-D7), so a relay in
// the x=0 column reaching a worker at x=1 wraps 0->16->15->...->1 and holds ~15 links for that one read.
// Row-major bands span both halves; grouping puts each relay on the half its own column reaches cheaply:
//   x=9 column relays (bank >= 4) -> the left half  (NOC0 x < 9)
//   x=0 column relays (bank <  4) -> the right half (NOC0 x > 9)
bool relay_assign_xsplit() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_RELAY_ASSIGN");
        return s != nullptr && std::string_view(s) == "xsplit";
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_FIFO_MB: host FIFO per D2H socket, in MiB, default 64. It is the pipeline's
// only elasticity in a direct-push run, and it is plain mmap + IOMMU host RAM reached by a full 64-bit
// NoC/PCIe address, so it costs no TLB window and has no channel cap. Capped at 3.5 GiB: the socket's byte
// size and the device's wrap-safe credit arithmetic (bytes_sent - bytes_acked) are 32-bit.
uint32_t host_fifo_bytes() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_FIFO_MB");
        uint64_t mb = (s != nullptr && *s != '\0') ? std::strtoull(s, nullptr, 10) : 64ull;
        mb = std::clamp<uint64_t>(mb, 1, 3584);
        return static_cast<uint32_t>(mb << 20);
    }();
    return v;
}

// TT_METAL_STREAMING_PROFILER_TRACY=1: attach the Tracy sink. Off by default -- the primary consumers are
// the registered ones (register_consumer / the ops CSV), and Tracy is one more, expensive, consumer.
bool tracy_push_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_TRACY");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return on;
}

// Host<->device clock sync. Without one, AddDevice() can only anchor "device time 0" at the host time the
// first marker was CONSUMED, which lags production by the whole drain+decode latency and shifts every
// device zone right of the host CPU zones. Done host-side: read the Tensix wall clock (the counter kernel
// markers timestamp with) straight over NoC, bracketed by host clock reads -- Cristian's algorithm, whose
// bracket midpoint cancels the round-trip to first order.
struct StreamingProfilerSync {
    double frequency = 0.0;  // device cycles per nanosecond (GHz)
    uint64_t device_at_anchor = 0;
    int64_t host_anchor = 0;
    bool valid = false;
};

// spacing_us spreads the samples out to lengthen the regression baseline: at 0 (back-to-back) the baseline
// is only ~360 us -- 100 samples x ~3.6 us of MMIO round trip -- and with ~us of host-timestamp jitter the
// fitted frequency then carries ~1e-4 of error. That is a rate error, so it grows with time since the
// anchor and shows up as rows drifting apart rather than as a constant skew.
StreamingProfilerSync sync_device_clock(
    tt::Cluster& cluster, uint32_t chip_id, const CoreCoord& worker, uint32_t spacing_us = 0) {
    // RISCV_DEBUG_REG_WALL_CLOCK_L/H are Tensix-tile debug registers by spec, but a DRAM tile answers them
    // too (measured on all 7 views with `test_streaming_profiler_zones --clkprobe 1`), which is what makes
    // the per-core anchor below possible.
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
    // Anchor on the sample mean (self-consistent: device time AT that host time), rather than extrapolating
    // an intercept back to host_time=0 where a tiny slope error becomes a huge offset.
    out.host_anchor = static_cast<int64_t>(hx);
    out.device_at_anchor = static_cast<uint64_t>(dy);
    out.valid = out.frequency > 0.0;
    return out;
}

// Last bring-up step entered. Bring-up runs several distinct MMIO paths and a hang in any of them reports
// only "MMIO per-op timeout", so this is what names the stall site.
thread_local std::string g_bringup_step = "(not started)";

StreamingProfiler::DeviceCtx::DeviceCtx() = default;
StreamingProfiler::DeviceCtx::~DeviceCtx() = default;
StreamingProfiler::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;

StreamingProfiler::StreamingProfiler(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    try {
        start(mesh_device);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] init failed at step [{}] ({}); disabled for this session.",
            g_bringup_step,
            e.what());
        stop();
    }
}

StreamingProfiler::~StreamingProfiler() { stop(); }

void StreamingProfiler::start(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();

    if (cluster.arch() != tt::ARCH::BLACKHOLE) {
        log_debug(tt::LogMetal, "[streaming profiler] not Blackhole; skipping relay capture.");
        return;
    }

    tracy_ = std::make_unique<StreamingProfilerTracyHandler>();
    // Zone names are not loaded here or on the first drain: they arrive per-ELF as each binary is loaded
    // (llrt::ZoneMetaRegistry). At MeshDevice bring-up no workload kernel has been compiled yet, and by the
    // first drain the later kernels still have not been.

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx ctx;
        ctx.chip_id = static_cast<uint32_t>(mesh_device->get_device(coord)->id());
        if (!boot_device(mesh_device, ctx, coord)) {
            continue;  // boot logs its own reason; degrade to no-capture for this device
        }
        // Anchor the device timeline with a measured clock sync (see sync_device_clock). The fallback --
        // aiclk plus "device 0 == now" -- places device zones relative to the first marker consumed, so they
        // lag the host zones by the drain latency.
        double freq = cluster.get_device_aiclk(ctx.chip_id) / 1000.0;
        if (freq <= 0.0) {
            freq = 1.0;
        }
        StreamingProfilerSync sync;
        if (!ctx.core_virt.empty()) {
            const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
            // 100 samples x 500 us spans ~50 ms of baseline instead of ~360 us, cutting the fitted-frequency
            // error by the baseline ratio (~140x). This is the one frequency every context on this chip uses.
            sync = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/500);
        }
        if (sync.valid) {
            ctx.clock_synced = true;
            ctx.freq_ghz = sync.frequency;
            tracy_->AddDevice(
                ctx.chip_id, sync.host_anchor, static_cast<double>(sync.device_at_anchor), sync.frequency);
            log_info(
                tt::LogMetal,
                "[streaming profiler] Device {} clock sync: frequency={:.6f} GHz (aiclk reports {:.6f}), "
                "device_time_at_anchor={} cycles",
                ctx.chip_id,
                sync.frequency,
                freq,
                sync.device_at_anchor);
        } else {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {} clock sync FAILED; falling back to first-marker anchoring "
                "(device zones will lag the host zones by the drain latency)",
                ctx.chip_id);
            ctx.freq_ghz = freq;
            tracy_->AddDevice(ctx.chip_id, tracy::Profiler::GetTime(), 0.0, freq);
        }
        // A relay does not share the worker's clock origin. Both counters zero at chip reset, but the Tensix
        // domain is clocked only while out of reset (measured 1.8 s per 32 s of wall against the DRAM core's
        // 19.8 s), so a chip-wide worker anchor put DRISC rows up to 42 minutes to the right while their
        // spans stayed correct to 0.17%. The duty ratio is unpredictable from the host and board-dependent
        // (on another part the same offset is 3 us), so each relay core gets its own measured anchor.
        if (sync.valid && tracy_ != nullptr) {
            for (uint32_t d = 0; d < ctx.n_drisc; d++) {
                // Keyed on NOC0 like every other context lookup, while the register read needs the virtual
                // pair. No mapping means self-profiling is off, so this core has no Tracy row to anchor.
                const auto nit = ctx.virt_to_noc0.find(
                    (static_cast<uint64_t>(ctx.drisc_virtual[d].x) << 32) |
                    static_cast<uint64_t>(ctx.drisc_virtual[d].y));
                if (nit == ctx.virt_to_noc0.end()) {
                    continue;
                }
                const StreamingProfilerSync ds = sync_device_clock(cluster, ctx.chip_id, ctx.drisc_virtual[d]);
                if (!ds.valid) {
                    // Degrade to the worker anchor rather than dropping the rows: a misplaced row is still
                    // readable, an absent one is not.
                    log_warning(
                        tt::LogMetal,
                        "[streaming profiler] Device {} DRISC {} at NOC0 ({},{}): DRAM-core clock sync FAILED; "
                        "its zones and plots fall back to the WORKER anchor and will be shifted by the "
                        "reset->open gap",
                        ctx.chip_id,
                        d,
                        nit->second.first,
                        nit->second.second);
                    continue;
                }
                // One shared frequency, per-core anchors. Alignment is relative, so a shared rate makes
                // differential drift zero by construction and any error in it is common-mode. The cores' true
                // rates agree to ~5 ppm while their individual fits scatter over ~99 ppm, so fitting per core
                // would trade a 5 ppm physical term for a 99 ppm noise one.
                tracy_->AddCore(
                    ctx.chip_id,
                    nit->second.first,
                    nit->second.second,
                    ds.host_anchor,
                    static_cast<double>(ds.device_at_anchor),
                    sync.frequency);
                const double off_ms =
                    (static_cast<double>(ds.device_at_anchor) - static_cast<double>(sync.device_at_anchor)) /
                    (sync.frequency > 0.0 ? sync.frequency : 1.0) / 1e6;
                const double fit_ppm =
                    sync.frequency > 0.0 ? (ds.frequency - sync.frequency) / sync.frequency * 1e6 : 0.0;
                log_info(
                    tt::LogMetal,
                    "[streaming profiler] Device {} DRISC {} NOC0 ({},{}) clock sync: frequency={:.6f} GHz "
                    "(SHARED across all contexts); this core's own fit {:.6f} = {:+.1f} ppm, NOT APPLIED, "
                    "device_time_at_anchor={} cycles, offset vs worker anchor {:+.3f} ms",
                    ctx.chip_id,
                    d,
                    nit->second.first,
                    nit->second.second,
                    sync.frequency,
                    ds.frequency,
                    fit_ppm,
                    ds.device_at_anchor,
                    off_ms);
            }
        }
        // Per-core Tracy contexts are created lazily on each core's first zone: only ~16 of ~110 cores
        // typically run the workload, and pre-creating the grid litters the capture with empty contexts
        // that read as "cores not showing up".
        ctx.active = true;
        devices_.push_back(std::move(ctx));
    }

    // Built after devices_ is stable: socket ownership moves into the receiver, and the lane tables it
    // hands consumers are flattened here so no consumer ever does a per-record hash lookup.
    if (!devices_.empty()) {
        std::vector<streaming_profiler::ReceiverDeviceConfig> rdevs;
        for (auto& ctx : devices_) {
            auto& rd = rdevs.emplace_back();
            rd.chip_id = ctx.chip_id;
            rd.num_cores = ctx.nl / kNRisc;
            rd.core_of_xy = ctx.core_of_xy;
            rd.clock_synced = ctx.clock_synced;
            rd.frequency_ghz = ctx.freq_ghz;
            rd.numa_node = static_cast<int>(cluster.get_numa_node_for_device(ctx.chip_id));
            rd.lane_table.reserve(ctx.nl);
            for (uint32_t ci = 0; ci < rd.num_cores; ci++) {
                const auto [vx, vy] = ctx.core_virt[ci];
                uint32_t nx = vx, ny = vy;
                if (auto it = ctx.virt_to_noc0.find((static_cast<uint64_t>(vx) << 32) | vy);
                    it != ctx.virt_to_noc0.end()) {
                    nx = it->second.first;
                    ny = it->second.second;
                }
                const auto role = (ctx.n_worker_cores != 0 && ci >= ctx.n_worker_cores)
                                      ? streaming_profiler::StreamingProfilerLaneRole::Relay
                                      : streaming_profiler::StreamingProfilerLaneRole::Worker;
                for (uint32_t r = 0; r < kNRisc; r++) {
                    rd.lane_table.push_back(streaming_profiler::StreamingProfilerLaneInfo{
                        ctx.chip_id,
                        static_cast<uint16_t>(vx),
                        static_cast<uint16_t>(vy),
                        static_cast<uint16_t>(nx),
                        static_cast<uint16_t>(ny),
                        static_cast<uint8_t>(r),
                        role});
                }
            }
            for (uint32_t sk = 0; sk < ctx.n_drisc; sk++) {
                if (ctx.sockets[sk] != nullptr) {
                    TT_FATAL(sk == rd.sockets.size(), "sockets must form a contiguous prefix");
                    rd.sockets.push_back(std::move(ctx.sockets[sk]));
                }
            }
        }
        receiver_ = std::make_unique<streaming_profiler::StreamingProfilerReceiver>(std::move(rdevs));
        if (tracy_push_enabled()) {
            tracy_consumer_ = std::make_unique<streaming_profiler::StreamingProfilerTracyConsumer>(tracy_.get());
            // Tracy takes device zones whole (one QueueGpuZone item per zone), and the paired stream's
            // per-lane completion order is the order the Tracy server rebuilds nesting from.
            receiver_->add_consumer(
                "tracy",
                [c = tracy_consumer_.get()](const streaming_profiler::StreamingProfilerRecordBatch& b) { (*c)(b); });
        }
        streaming_profiler::attach_registered_consumers(*receiver_);
        receiver_->start();
    }
    if (!devices_.empty()) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] active on {} device(s): DRISC relay -> {} MiB D2H socket -> {}",
            devices_.size(),
            host_fifo_bytes() / (1024 * 1024),
            tracy_push_enabled() ? "registered consumers + Tracy"
                                 : "registered consumers (Tracy off; opt in with TT_METAL_STREAMING_PROFILER_TRACY=1)");
    }
}

// Put a DRISC's NIU into stream mode (1) or back to NOC2AXI (0). Run to completion because D2HSocket
// construction writes the config into DRISC L1 from the host, which only lands once the NIU terminates
// inbound traffic at L1 instead of forwarding it to GDDR.
void StreamingProfiler::set_drisc_niu_mode(IDevice* device, const CoreCoord& drisc_logical, uint32_t stream) {
    Program p = CreateProgram();
    CreateKernel(
        p,
        "tt_metal/tools/profiler/kernels/drisc_niu_mode.cpp",
        drisc_logical,
        DramConfig{.noc = NOC::NOC_0, .compile_args = {stream}});
    const std::string who = fmt::format("niu-mode({},{})->{}", drisc_logical.x, drisc_logical.y, stream);
    g_bringup_step = who + ":CompileProgram";
    detail::CompileProgram(device, p, /*force_slow_dispatch=*/true);
    g_bringup_step = who + ":WriteRuntimeArgs";
    detail::WriteRuntimeArgsToDevice(device, p, /*force_slow_dispatch=*/true);
    // Split into launch and wait so a failure names which half stalled: LaunchProgram's dram_barrier
    // MMIO-polls a core in every DRAM channel, while the completion poll necessarily runs after the flip,
    // against a core that is by definition already in stream mode.
    g_bringup_step = who + ":LaunchProgram(dram_barrier,no-wait)";
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    g_bringup_step = who + ":WaitProgramDone(poll-after-flip)";
    detail::WaitProgramDone(device, p);
    g_bringup_step = who + ":done";
}

// Flip every relay's NIU in one launch. Every LaunchProgram carries a dram_barrier, which MMIO-polls a core
// in every DRAM channel, so flipping one relay per launch makes the second flip's barrier address a core
// that is already in stream mode -- where an inbound DRAM-range address no longer means what the barrier
// assumes -- and the read never completes (~210 ms root-port completion timeout). One launch is one
// barrier, before any core is flipped. Restores (stream -> noc2axi) go through the same path.
void StreamingProfiler::set_drisc_niu_mode(
    IDevice* device, const std::vector<CoreCoord>& drisc_logicals, uint32_t stream) {
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
    // Split like the single-core path. Here the barrier runs before any core is in stream mode, so a
    // failure on the first label means a core was already in stream mode when this run began -- a restore
    // that did not complete, or a reset that did not cover it.
    g_bringup_step = who + ":LaunchProgram(dram_barrier,no-wait)";
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    g_bringup_step = who + ":WaitProgramDone(poll-after-flip)";
    detail::WaitProgramDone(device, p);
    g_bringup_step = who + ":done";
}

// Producers are armed by TT_METAL_DEVICE_PROFILER, not by us, and a lossless producer blocks on a full
// ring. So whenever the relay fails to come up the workload does not merely lose its capture, it wedges
// with every lane parked in ring_ensure_room. PROFILER_TERMINATE exists for this: while set, the producer
// stops blocking and proceeds.
void StreamingProfiler::disarm_producers(
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
        "[streaming profiler] Device {}: no DRISC relay -- disarmed ring back-pressure on {} cores "
        "(markers are DROPPED, but the workload will not stall waiting for a consumer)",
        device_id,
        n);
}

// Wait until every producer's ring is empty with the relays still running: head is relay-written and tail
// is producer-written, so head == tail on every RISC means the consumer has taken everything published.
//
// This must precede the quiesce. Dispatch cores keep emitting zones through device close, so with the
// relays already stopped they would park in ring_ensure_room for however long that takes.
bool StreamingProfiler::wait_producer_rings_drained(DeviceCtx& ctx, std::chrono::milliseconds budget) {
    if (ctx.core_virt.empty()) {
        return true;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const size_t n = ctx.n_worker_cores != 0 ? ctx.n_worker_cores : ctx.core_virt.size();
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

// Last resort, and the only path that drops a marker: a producer still publishing after the drain budget
// expired. Unblocking it keeps device close from wedging in wait_until_cores_done().
void StreamingProfiler::disarm_producer_backpressure(DeviceCtx& ctx) {
    if (ctx.core_virt.empty()) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const size_t n = ctx.n_worker_cores != 0 ? ctx.n_worker_cores : ctx.core_virt.size();
    uint32_t one = 1;
    for (size_t ci = 0; ci < n; ci++) {
        const auto [vx, vy] = ctx.core_virt[ci];
        cluster.write_core(
            &one,
            sizeof(uint32_t),
            tt_cxy_pair(ctx.chip_id, CoreCoord{vx, vy}),
            prof_l1 + kernel_profiler::PROFILER_TERMINATE * sizeof(uint32_t));
    }
}

bool StreamingProfiler::boot_device(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t device_id = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(device_id);

    const char* sd_env = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const bool slow_dispatch = sd_env != nullptr && *sd_env != '\0' && *sd_env != '0';

    // The relay is a DRISC: one DM RISC-V on a DRAM core, which today exists only on Blackhole.
    if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: no DRAM programmable cores (card FW below the DRISC gate?)",
            device_id);
        disarm_producers(mesh_device, device_id);
        return false;
    }

    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    // Slow dispatch hands the whole worker grid to compute (12x10 here) because nothing is reserved for
    // dispatch; fast dispatch reserves the rest itself and returns 11x10.
    //
    // The poll list built below defines the drained set, and a producer outside it hangs the workload:
    // producers are lossless, so an undrained one fills its ring, blocks forever in ring_ensure_room and
    // takes the host down with it in wait_until_cores_done. The relay lives on a DRAM core, so no worker is
    // spent on it and the full grid can be polled.
    const bool reserve_column = slow_dispatch && reserve_column_env();
    const uint32_t gx = static_cast<uint32_t>(grid.x) - (reserve_column ? 1u : 0u);
    const uint32_t gy = static_cast<uint32_t>(grid.y);
    const uint64_t num_cores = static_cast<uint64_t>(gx) * gy;
    ctx.nl = static_cast<uint32_t>(num_cores) * kNRisc;
    ctx.core_virt.resize(num_cores);

    // Pre-zero every core's profiler control vector and build the maps the host owns: core index ->
    // virtual (x,y), which is the relay's poll list and Tracy's view, and the inverse packed (y<<16)|x ->
    // core index. Core identity is not on the wire; it travels in the payload, written by the producing
    // core into SPSC_CORE_XY.
    std::vector<uint32_t> coords(num_cores, 0);
    std::vector<uint8_t> zero_ctrl(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
    // Enumeration order decides what a contiguous relay slice means. Default is row-major; xsplit orders by
    // (NoC half, row, column), so the first n_left entries are exactly the cores the x=9 DRAM column reaches
    // without wrapping a row.
    std::vector<std::pair<uint32_t, uint32_t>> order;  // (lx, ly) in assignment order
    order.reserve(num_cores);
    uint32_t n_left = 0;
    {
        std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> keyed;
        keyed.reserve(num_cores);
        for (uint32_t ly = 0; ly < gy; ly++) {
            for (uint32_t lx = 0; lx < gx; lx++) {
                const CoreCoord n0 = cluster.get_physical_coordinate_from_logical_coordinates(
                    device_id, CoreCoord{lx, ly}, CoreType::WORKER, /*no_warn=*/true);
                const uint32_t half = static_cast<uint32_t>(n0.x) < 9u ? 0u : 1u;
                keyed.push_back({half, {lx, ly}});
                n_left += (half == 0u) ? 1u : 0u;
            }
        }
        if (relay_assign_xsplit()) {
            std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;  // stable: row-major order preserved inside each half
            });
        } else {
            n_left = 0;  // signals "no grouping" to the assignment loop
        }
        for (const auto& k : keyed) {
            order.push_back(k.second);
        }
    }
    for (uint32_t idx = 0; idx < num_cores; idx++) {
        {
            const uint32_t lx = order[idx].first, ly = order[idx].second;
            CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, CoreCoord{lx, ly}, CoreType::WORKER);
            const uint32_t vx = static_cast<uint32_t>(v.x), vy = static_cast<uint32_t>(v.y);
            coords[idx] = (vx & 0xFFFFu) | ((vy & 0xFFFFu) << 16);
            ctx.core_of_xy[coords[idx]] = idx;
            cluster.write_core(zero_ctrl.data(), (uint32_t)zero_ctrl.size(), tt_cxy_pair(device_id, v), prof_l1);
            const CoreCoord noc0 = cluster.get_physical_coordinate_from_logical_coordinates(
                device_id, CoreCoord{lx, ly}, CoreType::WORKER, /*no_warn=*/true);
            ctx.core_virt[idx] = {vx, vy};
            ctx.virt_to_noc0[(static_cast<uint64_t>(vx) << 32) | vy] = {
                static_cast<uint32_t>(noc0.x), static_cast<uint32_t>(noc0.y)};
        }
    }

    const distributed::MeshCoordinate scoord = coord;
    ctx.device = mesh_device->get_device(coord);

    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    if (nbanks == 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Device {}: no DRAM views to host a relay -- the streaming profiler is OFF "
            "for this device.",
            device_id);
        disarm_producers(mesh_device, device_id);
        return false;
    }
    {
        const uint32_t view_cap = std::min<uint32_t>(kMaxRelays, nbanks);
        const uint32_t requested = n_relays_env(kMaxRelays);
        if (requested == 0) {
            ctx.n_drisc = view_cap;
            log_info(
                tt::LogMetal,
                "[streaming profiler] Device {}: {} relays = min({} max, {} DRAM views); override with "
                "TT_METAL_STREAMING_PROFILER_NRELAYS",
                device_id,
                ctx.n_drisc,
                kMaxRelays,
                nbanks);
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
            log_info(
                tt::LogMetal,
                "[streaming profiler] Device {}: {} relays, forced by TT_METAL_STREAMING_PROFILER_NRELAYS (part has {} "
                "DRAM views, max {})",
                device_id,
                ctx.n_drisc,
                nbanks,
                kMaxRelays);
        }
    }
    // The default roster lists all 8 views, fragile ones last; views this part lacks drop out before the
    // n_drisc prefix is taken.
    std::vector<uint32_t> banks;
    {
        const bool banks_forced = [] {
            const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_RELAY_BANKS");
            return s != nullptr && *s != '\0';
        }();
        for (const uint32_t b : relay_banks()) {
            TT_FATAL(
                !banks_forced || b < nbanks,
                "TT_METAL_STREAMING_PROFILER_RELAY_BANKS names DRAM view {} but this part has views 0..{}",
                b,
                nbanks - 1);
            if (b < nbanks) {
                banks.push_back(b);
            }
        }
        TT_FATAL(
            banks.size() >= ctx.n_drisc,
            "streaming profiler needs {} relay banks but only {} usable DRAM views are listed{} (part has {} views)",
            ctx.n_drisc,
            banks.size(),
            banks_forced ? " in TT_METAL_STREAMING_PROFILER_RELAY_BANKS" : " in the default roster",
            nbanks);
        banks.resize(ctx.n_drisc);
    }

    // Mirrors the kernel's kSlotWords.
    const uint32_t slot_bytes_all = kernel_profiler::spsc_span_slot_words(kNRisc) * sizeof(uint32_t);

    // Picked up front so that every relay's NIU flips in one launch (see set_drisc_niu_mode).
    std::vector<CoreCoord> flip_cores;
    for (uint32_t d = 0; d < ctx.n_drisc; d++) {
        flip_cores.push_back(mesh_device->impl().pick_unused_dram_logical_core(ctx.device, banks[d]));
    }
    // pick_unused_dram_logical_core() takes a DRAM view and reserves that view's worker/eth endpoints;
    // it has no idea another view can resolve to the same physical port (views 0 and 7 have both come
    // back as NoC core 0-0). Two resident relays sharing one core's L1 would overlap staging, socket
    // config and results with nothing to notice, so refuse to launch instead.
    for (uint32_t a = 0; a < flip_cores.size(); a++) {
        for (uint32_t b = a + 1; b < flip_cores.size(); b++) {
            TT_FATAL(
                flip_cores[a] != flip_cores[b],
                "streaming profiler: DRISC {} (DRAM view {}) and DRISC {} (DRAM view {}) both resolve to logical "
                "DRAM core ({},{}). Two resident relay kernels cannot share a core -- pick different banks via "
                "TT_METAL_STREAMING_PROFILER_RELAY_BANKS.",
                a,
                banks[a],
                b,
                banks[b],
                flip_cores[a].x,
                flip_cores[a].y);
        }
    }
    // Cluster::dram_barrier passes no subchannel, so LocalChip::dram_membar syncs subchannel 0 of every
    // channel, and every LaunchProgram carries one. A relay resident on such a core is in stream mode,
    // where an inbound DRAM-range address no longer forwards to GDDR, so the barrier is addressing a
    // core whose semantics changed under it. Reported rather than fatal: the configuration usually
    // works, and the point is to have the explanation available for a later MMIO timeout.
    {
        std::vector<uint32_t> collide;
        for (int ch = 0; ch < soc.get_num_dram_channels(); ch++) {
            const CoreCoord bar = soc.get_dram_core_for_channel(ch, 0, CoordSystem::LOGICAL);
            for (uint32_t d = 0; d < flip_cores.size(); d++) {
                if (flip_cores[d] == bar) {
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
                flip_cores.size());
        } else {
            log_info(
                tt::LogMetal,
                "[streaming profiler] no relay sits on a dram_barrier target core (checked {} channels "
                "against {} relays).",
                soc.get_num_dram_channels(),
                flip_cores.size());
        }
    }
    set_drisc_niu_mode(ctx.device, flip_cores, 1);

    const std::vector<uint32_t>& slice_map = relay_slice_map();
    TT_FATAL(
        slice_map.empty() || slice_map.size() >= ctx.n_drisc,
        "TT_METAL_STREAMING_PROFILER_RELAY_SLICE_MAP needs {} entries, got {}",
        ctx.n_drisc,
        slice_map.size());
    // Weighted prefix split: relay slot sl owns cores [cum[sl], cum[sl+1]) scaled to num_cores. Integer
    // math on the running sum keeps the partition exact -- every core assigned once, no rounding gap.
    const std::vector<uint32_t>& weights_env = relay_weights();
    TT_FATAL(
        weights_env.empty() || weights_env.size() >= ctx.n_drisc,
        "TT_METAL_STREAMING_PROFILER_RELAY_WEIGHTS needs {} entries (one per relay), got {}",
        ctx.n_drisc,
        weights_env.size());
    std::vector<uint64_t> wcum(ctx.n_drisc + 1, 0);
    for (uint32_t i = 0; i < ctx.n_drisc; i++) {
        const uint32_t w = weights_env.empty() ? 1u : weights_env[i];
        TT_FATAL(w != 0, "relay weight {} must be non-zero", i);
        wcum[i + 1] = wcum[i] + w;
    }
    // One replicated mesh buffer with one interleaved page per DRAM bank reserves the same
    // [address, address+spool) window in every bank of every device, so a single buffer covers every relay.
    // It must be a mesh-level buffer: MeshBuffer allocations run through the mesh lock-step allocator,
    // which never sees a device-local Buffer::create and would hand the same region out again.
    uint32_t spool_bytes = dram_spool_mb() * (1u << 20);
    uint32_t spool_addr = 0;
    if (spool_bytes != 0 && spool_buffer_ == nullptr) {
        const uint32_t nbanks_dram = ctx.device->allocator()->get_num_banks(BufferType::DRAM);
        try {
            spool_buffer_ = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{static_cast<DeviceAddr>(nbanks_dram) * spool_bytes},
                distributed::DeviceLocalBufferConfig{.page_size = spool_bytes, .buffer_type = BufferType::DRAM},
                mesh_device.get());
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] could not reserve {} MiB/bank of DRAM for the GDDR spool ({}); falling "
                "back to direct push",
                dram_spool_mb(),
                e.what());
        }
    }
    if (spool_buffer_ != nullptr) {
        spool_addr = static_cast<uint32_t>(spool_buffer_->address());
        log_info(
            tt::LogMetal,
            "[streaming profiler] GDDR spool: {} MiB per relay at bank offset 0x{:x}",
            dram_spool_mb(),
            spool_addr);
    } else {
        spool_bytes = 0;
    }
    // xsplit: a relay serves only the half its own DRAM column reaches without wrapping a row, and the
    // relays of a column split that half between them.
    std::vector<uint32_t> xs_grp(ctx.n_drisc, 0), xs_rank(ctx.n_drisc, 0);
    uint32_t xs_n[2] = {0, 0};
    for (uint32_t d = 0; d < ctx.n_drisc && d < banks.size(); d++) {
        const uint32_t g = banks[d] >= 4u ? 0u : 1u;  // views D4-D7 sit in NOC0 column x=9, D0-D3 in x=0
        xs_grp[d] = g;
        xs_rank[d] = xs_n[g]++;
    }
    // Each relay's coords are a contiguous run of the same grid order the host uses everywhere else, so a
    // core belongs to exactly one relay and nothing -- L1, socket, head mirrors -- is shared on the device.
    for (uint32_t d = 0; d < ctx.n_drisc; d++) {
        const uint32_t sl = slice_map.empty() ? d : slice_map[d];
        TT_FATAL(sl < ctx.n_drisc, "slice {} out of range for {} relays", sl, ctx.n_drisc);
        uint32_t lo = static_cast<uint32_t>((num_cores * wcum[sl]) / wcum[ctx.n_drisc]);
        uint32_t hi = static_cast<uint32_t>((num_cores * wcum[sl + 1]) / wcum[ctx.n_drisc]);
        if (n_left != 0 && xs_n[xs_grp[d]] != 0) {
            const uint32_t g = xs_grp[d];
            const uint32_t base = g == 0u ? 0u : n_left;
            const uint32_t span = g == 0u ? n_left : static_cast<uint32_t>(num_cores) - n_left;
            lo = base + static_cast<uint32_t>((static_cast<uint64_t>(span) * xs_rank[d]) / xs_n[g]);
            hi = base + static_cast<uint32_t>((static_cast<uint64_t>(span) * (xs_rank[d] + 1)) / xs_n[g]);
        }
        const uint32_t my_cores = hi - lo;
        if (my_cores == 0) {
            continue;
        }
        CoreCoord drisc_phys{};  // NOC0 coords of the relay core, for the socket and the log line
        uint32_t region = 0;     // usable L1 on the relay core
        ctx.drisc_logical[d] = mesh_device->impl().pick_unused_dram_logical_core(ctx.device, banks[d]);
        if (const auto& sub_sel = relay_subchannels(); d < sub_sel.size()) {
            // Forced placement. Validated against the same reserved set the picker honours, so a
            // requested subchannel that is a worker/eth endpoint is refused rather than silently
            // double-booking a core.
            const uint32_t nsub = soc.get_grid_size(tt::CoreType::DRAM).y;
            TT_FATAL(sub_sel[d] < nsub, "relay {} subchannel {} >= {}", d, sub_sel[d], nsub);
            const size_t chan = soc.get_channel_for_dram_view(static_cast<int>(banks[d]));
            const tt::umd::CoreCoord tc = soc.get_dram_core_for_channel(
                static_cast<int>(chan), static_cast<int>(sub_sel[d]), tt::CoordSystem::TRANSLATED);
            bool reserved = false;
            for (const auto& c : soc.dram_view_worker_cores.at(banks[d])) {
                reserved = reserved || (c.x == tc.x && c.y == tc.y);
            }
            for (const auto& c : soc.dram_view_eth_cores.at(banks[d])) {
                reserved = reserved || (c.x == tc.x && c.y == tc.y);
            }
            TT_FATAL(!reserved, "relay {} subchannel {} is a reserved worker/eth endpoint", d, sub_sel[d]);
            ctx.drisc_logical[d] =
                soc.get_logical_dram_core_for_subchannel(static_cast<int>(banks[d]), static_cast<int>(sub_sel[d]));
        }
        {
            const uint32_t nsub = soc.get_grid_size(tt::CoreType::DRAM).y;
            const size_t chan = soc.get_channel_for_dram_view(static_cast<int>(banks[d]));
            std::string cand;
            for (uint32_t sub = 0; sub < nsub; sub++) {
                const tt::umd::CoreCoord tc = soc.get_dram_core_for_channel(
                    static_cast<int>(chan), static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
                const tt::umd::CoreCoord nc = soc.translate_coord_to(tc, tt::CoordSystem::NOC0);
                cand += fmt::format(" sub{}=NOC0({},{})", sub, nc.x, nc.y);
            }
            log_info(
                tt::LogMetal,
                "[streaming profiler] relay {} bank {} chan {}: {} subchannels ->{} | chose logical ({},{})",
                d,
                banks[d],
                chan,
                nsub,
                cand,
                ctx.drisc_logical[d].x,
                ctx.drisc_logical[d].y);
        }
        const CoreCoord translated =
            soc.dram_bank_endpoint_coords.at(ctx.drisc_logical[d].x).at(ctx.drisc_logical[d].y);
        const tt::umd::CoreCoord phys = soc.translate_coord_to(
            tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);
        drisc_phys = CoreCoord{phys.x, phys.y};
        ctx.drisc_virtual[d] = ctx.device->virtual_core_from_logical_core(ctx.drisc_logical[d], CoreType::DRAM);
        log_info(
            tt::LogMetal,
            "[streaming profiler] relay {} at virtual ({},{}) owns band {} = cores [{}, {}) of {}",
            d,
            ctx.drisc_virtual[d].x,
            ctx.drisc_virtual[d].y,
            sl,
            lo,
            hi,
            num_cores);
        ctx.drisc_l1_base[d] = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        ctx.drisc_l1_noc[d] = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

        const uint32_t slot_bytes = slot_bytes_all;
        constexpr uint32_t kCfgReserve = 8 * 1024;
        // One 64-byte record per core (landed tails, head mirror, wire XY); the kernel's max_cores bound.
        constexpr uint32_t kMaxCores = 128;
        constexpr uint32_t kScratchBytes = kMaxCores * 64;
        // done(64) + stop(64) + results(256) + handshake(64).
        constexpr uint32_t kMiscBytes = 1024;
        const uint32_t fixed = kCfgReserve + kScratchBytes + kMiscBytes;
        const uint32_t nstage = nstage_cap(region > fixed ? (region - fixed) / slot_bytes : 0);
        if (nstage == 0) {
            log_warning(tt::LogMetal, "[streaming profiler] Device {}: DRISC L1 too small; skipping", device_id);
            disarm_producers(mesh_device, device_id);
            return false;
        }
        const uint32_t nstage_relay = nstage;
        const uint32_t stage_base = ctx.drisc_l1_base[d];
        const uint32_t core_records = stage_base + nstage * slot_bytes;
        ctx.done_addr[d] = core_records + kScratchBytes;
        ctx.stop_addr[d] = ctx.done_addr[d] + 64;
        const uint32_t cfg_l1 = ctx.drisc_l1_base[d] + region - kCfgReserve;
        TT_FATAL(ctx.stop_addr[d] + 64 <= cfg_l1, "DRISC L1 layout overlaps the socket config");

        // Stream mode -- already flipped for every relay by the pre-pass above -- is what makes the
        // host-written socket config land in this L1 instead of being forwarded to GDDR; the kernel
        // restores NOC2AXI on the host's word.

        // TT_METAL_STREAMING_PROFILER_NIU_TEST isolates the NIU mode flip from everything else the relay
        // does: flip, optionally restore, then bail before any socket, kernel or egress exists. The flip
        // is the only thing a relay writes that outlives the process, since NIU_CFG_0 persists until a
        // chip reset.
        //   =leave -> stay in stream mode, exactly as a run that dies before the stop=2 handshake leaves it
        //   =flip  -> restore NOC2AXI immediately (the clean-teardown control)
        const char* niu_test = std::getenv("TT_METAL_STREAMING_PROFILER_NIU_TEST");
        if (niu_test != nullptr && *niu_test != '\0') {
            const bool restore = std::string_view(niu_test) != "leave";
            if (restore) {
                set_drisc_niu_mode(ctx.device, ctx.drisc_logical[d], 0);
            }
            log_info(
                tt::LogMetal,
                "[streaming profiler] NIU TEST: DRISC {} logical ({},{}) flipped to stream mode and {} "
                "-- no socket, no kernel, no egress",
                d,
                ctx.drisc_logical[d].x,
                ctx.drisc_logical[d].y,
                restore ? "RESTORED to NOC2AXI" : "LEFT IN STREAM MODE");
            disarm_producers(mesh_device, device_id);
            return false;
        }

        // A static TLB window skips UMD's per-access TLB reconfigure on the socket's per-read ack write:
        // measured 171 ns/write static against 382 ns dynamic.
        //
        // Metal maps static windows at device init for workers/eth/dispatch and, on Blackhole, one 4 GB
        // window per DRAM channel -- but only on that channel's preferred worker endpoint port
        // (ll_api::configure_static_tlbs -> blackhole::ddr_to_noc0 takes the channel's last of 3 NoC ports).
        // The relay deliberately sits on the unused port, so its core is not in that map and the socket
        // would otherwise find no window. 2 MB at address 0 spans the DRISC's whole 128 KB L1
        // (MEM_DRISC_L1_BASE = 0), and Strict ordering matches what workers get.
        //
        // Best-effort: a window is a finite device resource, and losing the race only costs the ~210 ns.
        if (!no_static_tlb() && !cluster.is_mock_or_emulated()) {
            auto* tlb_manager = cluster.get_driver()->get_chip(device_id)->get_tlb_manager();
            const tt_xy_pair tlb_core(ctx.drisc_virtual[d].x, ctx.drisc_virtual[d].y);
            if (!tlb_manager->is_tlb_mapped(tlb_core)) {
                try {
                    g_bringup_step = fmt::format("relay {}: configure static TLB", d);
                    tlb_manager->configure_tlb(
                        tlb_core, /*tlb_size=*/2 * 1024 * 1024, /*address=*/0, tt::umd::tlb_data::Strict);
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
        }

        const uint32_t sk = d;
        try {
            {
                // sender_uses_physical_noc_addr switches the socket between "physical NoC coord + full L1
                // address" and the normal worker path (logical coord, worker-L1 semantics). The socket picks
                // the static-vs-dynamic write path by asking UMD whether this core has a window (see
                // init_sender_tlb), so the window configured just above is what puts the DRISC on the
                // static path.
                g_bringup_step = fmt::format("relay {}: D2HSocket construct (writes config into DRISC L1)", d);
                ctx.sockets[sk] = std::make_unique<distributed::D2HSocket>(
                    mesh_device,
                    distributed::MeshCoreCoord{scoord, CoreCoord(drisc_phys.x, drisc_phys.y)},
                    (host_fifo_bytes() / kPageSize) * kPageSize,
                    distributed::D2HSocket::ExternalConfigBuffer{
                        .address = cfg_l1, .sender_uses_physical_noc_addr = true});
                ctx.sockets[sk]->set_page_size(kPageSize);
                // Time pages_available() itself: off the hugepage path it reads host memory rather than
                // MMIO, so a per-poll cost inferred from anything else is not comparable.
                {
                    const auto t0 = std::chrono::steady_clock::now();
                    constexpr uint32_t kPollProbe = 2000;
                    uint32_t sink = 0;
                    for (uint32_t k = 0; k < kPollProbe; k++) {
                        sink += ctx.sockets[sk]->pages_available();
                    }
                    const double ns_per =
                        std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() /
                        kPollProbe;
                    log_info(
                        tt::LogMetal,
                        "[streaming profiler] flow-control poll probe: {:.0f} ns/call over {} calls | hugepage "
                        "path: {} | (sink {})",
                        ns_per,
                        kPollProbe,
                        ctx.sockets[sk]->is_using_hugepage() ? "YES (clflush+lfence)" : "no (mfence, host buffer)",
                        sink);
                    // socket read() ends in notify_sender(), which PCIe-writes bytes_acked to the sender
                    // core: one device write per read, and the access the poll probe above does not cover.
                    // Re-sending the current bytes_acked is idempotent, so probing it is safe.
                    {
                        const auto a0 = std::chrono::steady_clock::now();
                        constexpr uint32_t kAckProbe = 500;
                        for (uint32_t k = 0; k < kAckProbe; k++) {
                            ctx.sockets[sk]->probe_ack_write();
                        }
                        const double ack_ns =
                            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - a0).count() /
                            kAckProbe;
                        log_info(
                            tt::LogMetal,
                            "[streaming profiler] ACK-WRITE probe: {:.0f} ns/write over {} device writes | TLB "
                            "path: {}",
                            ack_ns,
                            kAckProbe,
                            ctx.sockets[sk]->has_static_tlb() ? "STATIC window" : "DYNAMIC (reconfigure per access)");
                    }
                    // A 4-byte read from the relay core is the same access wait_until_cores_done() issues per
                    // core, and the one that blows the budget in the "MMIO per-op timeout: 4B load" aborts.
                    {
                        const auto r0 = std::chrono::steady_clock::now();
                        constexpr uint32_t kRdProbe = 500;
                        const uint64_t rd_addr = ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]);
                        const tt_cxy_pair rd_core(device_id, ctx.drisc_virtual[d]);
                        uint32_t v = 0, acc = 0;
                        for (uint32_t k = 0; k < kRdProbe; k++) {
                            cluster.read_core(&v, sizeof(v), rd_core, rd_addr);
                            acc += v;
                        }
                        const double rd_ns =
                            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - r0).count() /
                            kRdProbe;
                        log_info(
                            tt::LogMetal,
                            "[streaming profiler] DEVICE-READ probe: {:.0f} ns/read over {} 4B device reads (acc {})",
                            rd_ns,
                            kRdProbe,
                            acc);
                    }
                    // Control probe: both probes above target the relay's DRAM endpoint, so on their own a
                    // slow number cannot distinguish card-wide MMIO degradation from a sick DRAM endpoint.
                    // A fixed worker read in every run separates the two.
                    {
                        const auto w0 = std::chrono::steady_clock::now();
                        constexpr uint32_t kWkProbe = 500;
                        const tt_cxy_pair wk_core(
                            device_id, CoreCoord{ctx.core_virt[0].first, ctx.core_virt[0].second});
                        uint32_t v = 0, acc = 0;
                        for (uint32_t k = 0; k < kWkProbe; k++) {
                            cluster.read_core(&v, sizeof(v), wk_core, prof_l1);
                            acc += v;
                        }
                        const double wk_ns =
                            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - w0).count() /
                            kWkProbe;
                        log_info(
                            tt::LogMetal,
                            "[streaming profiler] WORKER-READ control probe: {:.0f} ns/read over {} 4B reads from "
                            "worker virtual ({},{}) (acc {}) -- compare against DEVICE-READ above: both slow = "
                            "card-wide, worker fast = the relay's endpoint alone",
                            wk_ns,
                            kWkProbe,
                            ctx.core_virt[0].first,
                            ctx.core_virt[0].second,
                            acc);
                    }
                }
            }

            // Zero the relay core's own profiler ring. The relay kernel is built with PROFILE_KERNEL=1, so
            // the firmware writes its own zone markers into this core's ring on every launch, and this core
            // is excluded from the drained set, so nothing ever empties it. The ring is 512 words and the
            // SPSC backend blocks on a full ring rather than dropping, so after ~74 launches inside one
            // card-reset window the RISC wedges in firmware init and the relay silently never starts.
            const uint64_t relay_prof_l1 =
                hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER);
            cluster.write_core(
                zero_ctrl.data(),
                (uint32_t)zero_ctrl.size(),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                relay_prof_l1);

            // done | heartbeat and the rest of the 64 B pad: a stale value from the previous run reads as
            // this run's live state.
            uint32_t zero3[13] = {};
            cluster.write_core(
                zero3,
                sizeof(zero3),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
            // The stop word, plus the sync-event rendezvous triple (req | ack | go) sharing its 64 B pad.
            // Teardown leaves stop at 1 or 2 and the relay loop is `while (... && *stop == 0 ...)`, so a
            // stale value makes the next kernel exit after one sweep; a stale `req` parks every relay at a
            // barrier nobody is going to release.
            uint32_t zero4[4] = {};
            cluster.write_core(
                zero4,
                sizeof(zero4),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));

            ctx.relay_program[d] = std::make_unique<Program>(CreateProgram());
            // Read on the kernel side with get_named_compile_time_arg_val, so retiring one is a local edit.
            const std::unordered_map<std::string, uint32_t> cargs = {
                {"stage_base", stage_base},
                {"n_stage", nstage_relay},
                {"core_records", core_records},
                {"done_addr", ctx.done_addr[d]},
                {"stop_addr", ctx.stop_addr[d]},
                {"socket_config_addr", ctx.sockets[sk]->get_config_buffer_address()},
                {"max_cores", kMaxCores},
                // d&2 splits the pushers across two of the four unicast request VCs;
                // TT_METAL_STREAMING_PROFILER_RELAY_VCS overrides the assignment per relay.
                {"write_vc", d < relay_vcs().size() ? relay_vcs()[d] : ((d & 2u) ? 0u : 1u)},
                {"ship_min_pct", ship_min_pct()},
                // The bounce slots cost the kernel a staging generation, so the spool needs the full slot
                // count; a smaller L1 falls back to direct push rather than tripping the kernel's geometry
                // static_asserts.
                {"spool_base", spool_addr},
                {"spool_bytes", nstage_relay >= 7u ? spool_bytes : 0u}};
            if (spool_bytes != 0 && nstage_relay < 7u) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] Device {}: only {} staging slots fit, too few for the spool's bounce "
                    "buffers; relay {} runs direct push",
                    device_id,
                    nstage,
                    d);
            }
            TT_FATAL(
                my_cores * 32u <= slot_bytes_all,
                "CV-first tails staging ({} cores x 32 B) does not fit inside the slot past the pipeline",
                my_cores);
            auto relay_id = CreateKernel(
                *ctx.relay_program[d],
                "tt_metal/tools/profiler/kernels/streaming_profiler_relay.cpp",
                ctx.drisc_logical[d],
                DramConfig{
                    .noc = (relay_noc_override() < 0 ? false : relay_noc_override() == 1) ? NOC::NOC_1 : NOC::NOC_0,
                    .defines = {{"STREAMING_PROFILER_RELAY_KERNEL", "1"}},
                    .named_compile_args = cargs});
            std::vector<uint32_t> rt = {my_cores, static_cast<uint32_t>(prof_l1)};
            // Reversed: launch order follows global index, so the slice's last-launched cores land in the
            // first-chunk slots, which are read and serviced first.
            rt.insert(rt.end(), coords.rbegin() + (coords.size() - hi), coords.rbegin() + (coords.size() - lo));
            SetRuntimeArgs(*ctx.relay_program[d], relay_id, ctx.drisc_logical[d], rt);

            detail::CompileProgram(ctx.device, *ctx.relay_program[d], /*force_slow_dispatch=*/true);
            detail::WriteRuntimeArgsToDevice(ctx.device, *ctx.relay_program[d], /*force_slow_dispatch=*/true);
            g_bringup_step = fmt::format("relay {}: relay kernel LaunchProgram", d);
            detail::LaunchProgram(
                ctx.device, *ctx.relay_program[d], /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

            // A resident relay is launched fire-and-forget, so a core that fails to come out of reset
            // produces no error: the producers fill their rings, block (they are lossless), and the workload
            // wedges forever with a perfectly healthy card. Poll the heartbeat instead of assuming -- it must
            // leave 0 and then advance.
            g_bringup_step = fmt::format("relay {}: heartbeat verify", d);
            {
                const uint64_t hb_addr = ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]) + 4;
                const tt_cxy_pair core(device_id, ctx.drisc_virtual[d]);
                uint32_t hb0 = 0, hb1 = 0;
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
                while (std::chrono::steady_clock::now() < deadline) {
                    cluster.read_core(&hb0, sizeof(hb0), core, hb_addr);
                    if (hb0 != 0) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                // Poll for advance rather than sampling once: a single short sample cannot tell a dead relay
                // from a slow one. 200 ms is ~6000 idle sweeps of headroom at 30 us/sweep.
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
                if (hb0 == 0 || hb1 == hb0) {
                    uint32_t stopw = 0;
                    cluster.read_core(
                        &stopw, sizeof(stopw), core, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
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
                    ctx.relay_program[d].reset();
                    ctx.sockets[sk].reset();
                    disarm_producers(mesh_device, device_id);
                    return false;
                }
            }
        } catch (const std::exception& e) {
            // A code-region overflow makes the relay fail to load rather than fail to start, and the run
            // then exits 0 with every marker silently dropped, so name the cause at error level.
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
            disarm_producers(mesh_device, device_id);
            return false;
        }

        log_info(
            tt::LogMetal,
            "[streaming profiler] Device {}: {} {} resident on logical ({},{}) [noc0 ({},{})], cores "
            "[{},{}) of {}, {} staging slots x {} B",
            device_id,
            "DRISC relay (worker rings -> D2H socket)",
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

void StreamingProfiler::stop() {
    if (stopped_.exchange(true)) {
        return;
    }

    // Producers before consumers: let the rings empty while the relays are still draining them, so no
    // producer ever meets a stopped consumer.
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
            uint32_t one = 1;
            cluster.write_core(
                &one, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            uint32_t done = 0;
            while (std::chrono::steady_clock::now() < deadline) {
                cluster.read_core(
                    &done, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
                if ((done & 0xFFFF0000u) == 0xD09E0000u) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if ((done & 0xFFFF0000u) != 0xD09E0000u) {
                log_warning(
                    tt::LogMetal, "[streaming profiler] Device {}: DRISC relay did not acknowledge stop", ctx.chip_id);
            } else if (receiver_ != nullptr) {
                // done follows the relay's socket barrier, so the host has already acked every byte this
                // socket will ever carry and the stream can retire on one final empty check.
                receiver_->notify_producers_done(static_cast<uint32_t>(&ctx - devices_.data()), d);
            }
            // Release it to restore the NIU. It cannot do that until we say so: NOC2AXI forwards inbound
            // DRAM-range addresses to GDDR, so the flip takes this L1 out of the host's view.
            uint32_t two = 2;
            cluster.write_core(
                &two, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
        }
    }
    if (receiver_ != nullptr) {
        streaming_profiler::detach_registered_consumers();
        receiver_->shutdown();
    }
    for (auto& ctx : devices_) {
        verify_completeness(ctx, static_cast<uint32_t>(&ctx - devices_.data()));
    }
    if (receiver_ != nullptr) {
        receiver_->log_report();
        const auto zm = llrt::ZoneMetaRegistry::instance().stats();
        log_info(
            tt::LogMetal,
            "[streaming profiler] zone names: {} records from {} ELFs | id collisions {} | foreign/stale "
            "metadata sections ignored {} [collisions MUST be 0; a non-zero foreign count means the JIT cache "
            "holds ELFs from a different .tt_zone_meta layout]",
            zm.records,
            zm.elfs,
            llrt::ZoneMetaRegistry::instance().collisions(),
            zm.foreign_sections);
    }
    receiver_.reset();
    tracy_consumer_.reset();
    tracy_.reset();
    devices_.clear();
    // After the relays are quiesced, so nothing touches the spool, and while the mesh allocator is still
    // alive to take the region back.
    spool_buffer_.reset();
}

// One MMIO pass per worker core: the producer-owned stall counters, which nothing downstream can lose and
// so stay valid even under NO_DECODE, and each lane's own tail against the receiver's consumed-words
// mirror, which is the direct assertion that the stop-path sweep-to-empty held.
void StreamingProfiler::verify_completeness(DeviceCtx& ctx, uint32_t device_index) {
    if (ctx.core_virt.empty()) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    std::vector<uint32_t> heads;
    if (receiver_ != nullptr && !streaming_profiler::env_flag("TT_METAL_STREAMING_PROFILER_NO_DECODE")) {
        heads = receiver_->final_lane_heads(device_index);
    }
    std::vector<uint32_t> cv(kernel_profiler::SPSC_CONTROL_END, 0);
    uint64_t total = 0, worst = 0, cores_hit = 0;
    uint64_t stranded_words = 0, stranded_lanes = 0, checked_lanes = 0;
    uint32_t worst_lane = 0, worst_lane_words = 0;
    uint64_t risc_total[kNRisc] = {};
    struct CoreStall {
        uint32_t count, vx, vy, idx;
    };
    std::vector<CoreStall> stalled_cores;
    // Worker cores only. With DRISC self-profiling on, core_virt also holds the relay cores, and a DRAM
    // core has no producer and no stall counters -- reading the Tensix profiler address on one returns
    // whatever happens to be at that offset in DRISC L1.
    const size_t n_stall_cores = ctx.n_worker_cores != 0 ? ctx.n_worker_cores : ctx.core_virt.size();
    for (size_t ci = 0; ci < n_stall_cores; ci++) {
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
    log_info(
        tt::LogMetal,
        "[streaming profiler] Device {}: L1 STALL COUNTERS -- {} producer stalls across {} of {} cores "
        "(worst core {}) [0 stall-count = capture did not perturb]",
        ctx.chip_id,
        total,
        cores_hit,
        n_stall_cores,
        worst);
    if (total != 0) {
        std::sort(stalled_cores.begin(), stalled_cores.end(), [](const CoreStall& a, const CoreStall& b) {
            return a.count > b.count;
        });
        std::string top;
        for (size_t i = 0; i < stalled_cores.size(); i++) {
            const auto& c = stalled_cores[i];
            top += fmt::format("{}({},{})#{}={}", i != 0 ? " " : "", c.vx, c.vy, c.idx, c.count);
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] Device {}: stall breakdown by RISC -- BR {} | NC {} | T0 {} | T1 {} | T2 {}; "
            "top cores (virt x,y)=count: {}",
            ctx.chip_id,
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
    if (stranded_lanes == 0) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] COMPLETENESS: device {} -- {}/{} lanes fully drained, 0 words stranded",
            ctx.chip_id,
            checked_lanes,
            checked_lanes);
    } else {
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

}  // namespace tt::tt_metal
