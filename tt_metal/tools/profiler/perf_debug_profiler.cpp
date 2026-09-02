// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_profiler.hpp"

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
#include "tools/profiler/perf_debug_consumer.hpp"
#include "tools/profiler/perf_debug_env.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"
#include "tools/profiler/perf_debug_receiver.hpp"
#include "tools/profiler/perf_debug_tracy_consumer.hpp"
#include "llrt/zone_meta.hpp"  // per-ELF (zone id -> source location), the streaming name source
#include "tools/profiler/spsc_packet.h"

namespace tt::tt_metal {

namespace pz = tt::tt_metal::profiler;

namespace {

// TT_METAL_PERF_DEBUG_NO_STATIC_TLB: skip configuring a static TLB window for the DRISC drainer, leaving the
// socket's ack write on UMD's dynamic (reconfigure-per-access) path. Exists so static-vs-dynamic can be A/B'd
// on ONE binary -- rebuilding between arms makes every difference suspect.
bool no_static_tlb() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_NO_STATIC_TLB");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return v;
}



// ABLATION: strip the drain loop to EGRESS ONLY -- no worker reads, no per-core processing; the drainer
// re-ships pre-staged mock bytes forever. Pair with NO_DECODE=1, since the payload is mock.
//
// TT_METAL_PERF_DEBUG_NOC forces which NIU EVERY drainer egresses on (reads take the other); unset =
// NOC 0 for all six.
//
// It USED to alternate by drainer index, so each PCIe-tile NIU would carry three fillers' pushes instead
// of one carrying all six. That reasoning is sound and was measured wrong: NOC 1 egress runs ~2x the
// service interval of NOC 0 (per-filler, delay 30: 41.0-52.6 us forced to NOC 1 against 20.2-36.5 us
// forced to NOC 0), so alternating parked three of six fillers on the bad NIU. Those three then owned a
// contiguous core band and took essentially every producer stall below saturation. Forcing all six to
// NOC 0 halves stalls across the range (delay 30: 60,963 -> 35,699) even though they now share one NIU.
// The winning pairing is egress on 0 / reads on 1, which is what kReadNoc derives. The socket's NOC0-derived PCIe encoding is correct on BOTH NoCs -- the PCIe tile lives in
// translated space, so the coordinate mirroring that applies to worker coords does not apply to it
// (FINDINGS §N+12).
int drain_noc_override() {
    static const int v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_NOC");
        return (s == nullptr || *s == '\0') ? -1 : (std::strtoul(s, nullptr, 10) == 1 ? 1 : 0);
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_RESERVE_COLUMN: under slow dispatch, hold the last worker column back and poll only
// 11x10=110, instead of the full 12x10=120.
//
// Default changed 2026-08-07: the DRISC arm now polls the FULL grid. Reserving the column was only ever a
// COMPARABILITY device -- it made a DRISC (120 cores) and a Tensix (110, one of them being the drainer
// itself) sweep the same poll-list length, so a difference between the arms could not be blamed on the grid.
// It is not a functional requirement for a DRISC: that drainer lives on a DRAM core
// (pick_unused_dram_logical_core), so no worker core is ever needed for it. Full-grid coverage is the more
// useful default -- it profiles all 120 cores, and it makes `--gx 0` ("use whatever grid the device offers")
// safe under slow dispatch, which is exactly the mismatch that hung the workload for weeks (FINDINGS N+24).
//
// Set this when running the DRISC-vs-Tensix 2x2, where equal sweep cost matters more than coverage.
// Ignored on the Tensix arm, whose drainer physically lives in that column, so there it is always reserved.
bool reserve_column_env() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_RESERVE_COLUMN");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return v;
}


// DRAM banks (DRAM VIEW ids) the FILLERS occupy, one per filler. Views 7 and 2 ride at the end: the N+29
// sweep recorded view 7's spare port colliding with view 0's (both NoC core 0-0) and view 2 failing
// bringup outright, so a downgraded roster that drops trailing entries sheds the historically fragile
// views first. Both boot clean on the current UMD/soc-descriptor state, and the duplicate-core TT_FATAL
// in boot_device still checks every roster -- nothing in pick_unused_dram_logical_core() would.
const std::vector<uint32_t>& filler_vcs() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FILLER_VCS");
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

const std::vector<uint32_t>& role_filler_banks() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FILLER_BANKS");
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

// TT_METAL_PERF_DEBUG_NSTAGE: cap on staging slots. A DRISC's L1 fits 7; a Tensix's fits ~130, which would
// make the two drainers incomparable, so the Tensix path clamps to the DRISC count by default.
uint32_t nstage_cap(uint32_t computed) {
    const char* s = std::getenv("TT_METAL_PERF_DEBUG_NSTAGE");
    const uint32_t cap = (s == nullptr || *s == '\0') ? 7u : static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    return (cap != 0 && computed > cap) ? cap : computed;
}

}  // namespace

// TT_METAL_STREAMING_PROFILER_TRACY=1: attach the Tracy sink. OFF BY DEFAULT -- the streaming profiler's
// primary consumers are the registered ones (register_consumer / the ops CSV); Tracy is one more consumer
// and an expensive one, so it is opt-in rather than something every capture pays for. (This inverts the
// old TT_METAL_PERF_DEBUG_NO_TRACY.) Drain and decode run EXACTLY the same either way, which also makes
// the off state the sink-cost ablation: if the relay stops host-waiting with Tracy off, the Tracy push is
// provably the bottleneck.


// TT_METAL_PERF_DEBUG_SHIP_MIN_PCT: a FILLER defers shipping a live core until its FULLEST lane holds at
// least this percent of its own ring, unless the core aged out (serviced regardless). 0 disables (ship
// every live core every sweep). Per-lane, not per-span: the producer that blocks is always a lane, and a
// span percent under-reads a concentrated core's binding ring by up to kNumRisc x (span-5% let BRISC ride
// to a quarter ring and cost 551 stalls at delay 10; lane-5% opens the blind window at 5%). Values past
// 50 are capped by the kernel's half-ring lane trigger. Deferral only arms once the grid has grown for
// consecutive sweeps and disarms after a quiet spell (the kernel's growth-streak gate), so idle-phase
// trickle is flushed instead of parked across a burst onset. Default 25: the measured strict-0 band ends
// between 30 and 35 -- 35-48 flicker ~9% of runs at the burst knee and break at the delay where per-visit
// growth straddles the threshold (jitter lands on deferral-pre-loaded lanes), and >=50 stalls every run
// (two-visit exposure 2T+15 exceeds the usable ring).
uint32_t ship_min_pct() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_SHIP_MIN_PCT");
        const uint32_t n = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 25u;
        return n > 100 ? 100u : n;
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_DRAM_MB: per-filler GDDR spool ring, in MiB. Non-zero makes each filler ship frames
// by DMA into a ring in its own DRAM bank and forward them to the host FIFO from a non-blocking pump, so
// the service loop never touches the PCIe tile and host-side pressure lands in spool occupancy instead of
// in the sweep interval. The spool is the pipeline's burst elasticity: producers only feel back-pressure
// once it fills. 0 = the direct-push path, exactly as it was before the spool existed.
uint32_t dram_spool_mb() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_DRAM_MB");
        const uint32_t n = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 128u;
        // A ring beyond 4095 MiB would overflow the kernel's 32-bit ring arithmetic (a bank is 4 GiB anyway).
        return n > 4095u ? 4095u : n;
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_FILLER_SLICE_MAP: comma-separated permutation, entry d = which core BAND filler d
// takes. Unset = identity, which is what the code has always done -- and the bands are row-major slices of
// the worker grid assigned with no reference to where the filler's own DRAM core sits, so the pairing is
// arbitrary. Measured per-filler service intervals spread 20.1-36.6 us at delay 30 with the DRISCs scattered
// over y=12..23 while workers occupy y=2..11; this knob exists to test whether that spread is DRISC
// placement (follows the filler) or a property of the rows (follows the band).
const std::vector<uint32_t>& filler_slice_map() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FILLER_SLICE_MAP");
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

// Per-filler CORE-COUNT WEIGHTS, indexed by filler slot (not bank id). An even split makes every filler
// own the same ~22 cores, which makes the SLOWEST filler set the knee -- and they are not equally fast.
// Measured at delay 8, 130 cores, even split: per-filler service interval 19.7 / 21.1 / 22.3 / 22.8 / 26.6
// / 28.5 us, a 45% spread that follows the FILLER and not the core band (permuting bands leaves d5 slowest
// while it owns bands 5, 5, then 0). Mo's v6 roster spreads 3% over the same grid and takes 11 stalls where
// an even split takes 428 k, because a lane fills in 22.5 us -- so a 28.5 us filler stalls every ring and a
// 19.7 us one never does. Weighting core count by 1/interval equalises the intervals instead of the counts.
//
// Weights are a CALIBRATION, and the default is board-specific: they came off this part's default bank
// roster (5,6,4,1,0,3). Override with TT_METAL_PERF_DEBUG_FILLER_WEIGHTS, or pass all-equal values to get
// the old even split back.
const std::vector<uint32_t>& filler_weights() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FILLER_WEIGHTS");
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
        // Default EVEN. Weighting core counts compensates for bad filler placement rather than fixing it;
        // the placement itself is the lever (which DRAM subchannel each filler sits on). Kept as a knob so
        // the compensation can still be measured against a placement fix.
        return out;  // empty = even split; the use site treats it that way
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_FILLER_SUBCH: comma-separated DRAM SUBCHANNEL index per filler. Each DRAM view has
// three subchannels at quite different NoC positions (bank 5: NOC0 (9,2)/(9,10)/(9,3); bank 3: (0,5)/(0,7)/
// (0,6)), and pick_unused_dram_logical_core() returns whichever is simply FIRST unreserved -- nothing picks
// for locality. Unset keeps that behaviour.
//
// Placement is worth choosing because the per-filler service interval spreads 45% (19.7-28.5 us at delay 8)
// and the knee is set by the WORST filler, while a lane fills in 22.5 us. But it is chosen by MEASUREMENT,
// not by a distance model: the observed spread does not track NoC distance to the owned band (the closest
// filler, bank 3 at (0,6) owning rows 8-9, is the slowest at 36.5 us), nor column, nor staging depth
// (identical 7 slots), nor core count (equal).
const std::vector<uint32_t>& filler_subchannels() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FILLER_SUBCH");
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

// TT_METAL_PERF_DEBUG_FILLER_ASSIGN=xsplit: assign worker cores to fillers by NoC REACHABILITY instead of
// row-major index order.
//
// Round-trip LATENCY is position-independent on these NoCs -- both are unidirectional tori, so a request
// costs dx+dy hops and its response (17-dx)+(12-dy), always 29 total. LINK OCCUPANCY is not. Reads ride
// NoC 1 (-x/-y), and DRAM sits in NOC0 columns x=0 (views D0-D3) and x=9 (views D4-D7), so a filler in the
// x=0 column reaching a worker at x=1 wraps 0->16->15->...->1 and holds ~15 links for that one read, while
// reaching x=16 holds one. Row-major bands span BOTH halves, so nearly every one of a sweep's ~93 reads
// wraps a row. Grouping instead puts each filler on the half its own column reaches cheaply:
//   x=9 column fillers (bank >= 4) -> the LEFT half  (NOC0 x < 9)
//   x=0 column fillers (bank <  4) -> the RIGHT half (NOC0 x > 9)
bool filler_assign_xsplit() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FILLER_ASSIGN");
        return s != nullptr && std::string_view(s) == "xsplit";
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_FIFO_MB: host FIFO per D2H socket, in MiB. See the header comment on the default.
// Capped at 3.5 GiB: the socket's byte size and the device's wrap-safe credit arithmetic
// (reserve_pages_bounded's bytes_sent - bytes_acked) are 32-bit, so a FIFO at or past 4 GiB overflows
// them -- past this cap the knob would have to move to page units and the socket config to 64-bit.
uint32_t host_fifo_bytes() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FIFO_MB");
        uint64_t mb = (s != nullptr && *s != '\0') ? std::strtoull(s, nullptr, 10) : 64ull;
        mb = std::clamp<uint64_t>(mb, 1, 3584);
        return static_cast<uint32_t>(mb << 20);
    }();
    return v;
}

bool tracy_push_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("TT_METAL_STREAMING_PROFILER_TRACY");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return on;
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

// spacing_us spreads the samples in time to lengthen the REGRESSION BASELINE. It exists because the slope this
// function fits is baseline-limited, and at the default (0 = back-to-back) the baseline is only ~360 us -- 100
// samples x ~3.6 us of MMIO round trip -- so with ~us of host-timestamp jitter the fitted frequency carries
// ~1e-4 of error. MEASURED consequence (FINDINGS N+47): the six DRISC fits scattered over 99 ppm while the two
// clocks' TRUE rates agree to ~5 ppm, i.e. the fit noise was 20x the physical difference, and multiplied by the
// ~1.3 s between the anchor and the workload it displaced each DRISC row by 46-70 us with the sign tracking its
// own fit error (correlation 0.985). Frequency error is a RATE error: it grows with time since the anchor, which
// is why it shows up as rows drifting apart rather than as a constant skew.
PerfDebugSync sync_device_clock(
    tt::Cluster& cluster, uint32_t chip_id, const CoreCoord& worker, uint32_t spacing_us = 0) {
    // RISCV_DEBUG_REG_WALL_CLOCK_L/H. Reading L atomically LATCHES H, so read L then H (H's own latency is
    // irrelevant).
    //
    // CORRECTED: an earlier version of this comment said "the same registers the drainer firmware co-samples in
    // calibrate()". THERE IS NO SUCH FUNCTION -- `calibrate` appears nowhere in the drain kernels, in
    // tt_metal/hw, or anywhere else on the device side. The drainer does NOT co-sample two clocks; the whole
    // reason §N+46 needed a per-core HOST anchor is that no device-side rebase exists to lean on.
    //
    // Valid on a DRAM tile as well as a Tensix one, which is what makes the per-core anchor possible. That is
    // not documented anywhere -- these are Tensix-tile debug registers by spec -- so it was measured first, in
    // isolation, before this function was ever pointed at a DRAM core: see `test_perf_debug_zones --clkprobe 1`,
    // which reads all 7 DRAM views plus a worker and reports the rate and the pairwise offset. Result: readable,
    // non-zero, advancing at aiclk to 0.1 MHz on every view.
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

// BRING-UP PROGRESS MARKER.
//
// The drainer hang lands somewhere inside bring-up (FINDINGS N+32): the failure message says only
// "MMIO per-op timeout", leaving a 445 ms window that contains several distinct MMIO paths --
// dram_barrier and wait_until_cores_done (both inside set_drisc_niu_mode's LaunchProgram), the static
// TLB configure, D2HSocket construction, the drain kernel's own launch, and the heartbeat poll.
// Naming which one stalls has so far been guesswork, and three confident mechanisms have already died
// that way. This records the last step ENTERED so the next hang reports its own stall site.
thread_local std::string g_bringup_step = "(not started)";

PerfDebugProfiler::DeviceCtx::DeviceCtx() = default;
PerfDebugProfiler::DeviceCtx::~DeviceCtx() = default;
PerfDebugProfiler::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;

PerfDebugProfiler::PerfDebugProfiler(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    try {
        start(mesh_device);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] init failed at step [{}] ({}); disabled for this session.",
            g_bringup_step,
            e.what());
        stop();
    }
}

PerfDebugProfiler::~PerfDebugProfiler() { stop(); }

void PerfDebugProfiler::start(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();

    if (cluster.arch() != tt::ARCH::BLACKHOLE) {
        log_debug(tt::LogMetal, "[perf-debug profiler] not Blackhole; skipping drainer capture.");
        return;
    }

    tracy_ = std::make_unique<PerfDebugTracyHandler>();
    // NOTE: zone names are NOT loaded here, and not on the first drain either. They arrive per-ELF as each
    // kernel/firmware binary is loaded (llrt::ZoneMetaRegistry), which is the only schedule that works for
    // a model: at start() (MeshDevice bring-up) no workload kernel has been compiled yet, and by the first
    // drain the LATER kernels still have not been.

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx ctx;
        ctx.chip_id = static_cast<uint32_t>(mesh_device->get_device(coord)->id());
        if (!boot_device(mesh_device, ctx, coord)) {
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
            // LONG BASELINE, deliberately: 100 samples x 500 us spans ~50 ms instead of ~360 us, cutting the
            // fitted-frequency error by the baseline ratio (~140x). This is the ONE frequency every context on
            // this chip will use (see below), so it is worth 50 ms of a 9-12 s device open to measure it well.
            sync = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/500);
        }
        if (sync.valid) {
            ctx.clock_synced = true;
            ctx.freq_ghz = sync.frequency;
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
            ctx.freq_ghz = freq;
            tracy_->AddDevice(ctx.chip_id, tracy::Profiler::GetTime(), 0.0, freq);
        }
        // ---- PER-DRAM-CORE ANCHOR (the ORIGIN): drainers do NOT share the worker's clock origin -------------
        //
        // The sync above is measured on a TENSIX worker and registered per CHIP, so DRISC rows inherited it and
        // came out shifted right by the whole difference in banked counter totals -- measured 18.5 to 42.4 MINUTES
        // on bh-05. Spans agreed to 0.15-0.17% throughout, so only the ORIGIN was ever wrong.
        //
        // Cause is CLOCK GATING, not a different zero point: both counters zero at chip reset and neither
        // re-zeroes at device open, but the Tensix domain is clocked only while out of reset (1.8 s per 32 s of
        // wall) against the DRAM core's 19.8 s -- ~11x duty ratio, unpredictable by the host, so it must be
        // MEASURED per core. Reset discipline cannot fix it: device open (9-12 s) exceeds the workload window
        // (~753 ms). BOARD-DEPENDENT -- on bh-26 it is +0.003 ms, so do not judge this on a part where it no-ops.
        // Prerequisite verified separately (--clkprobe): a DRAM tile DOES answer RISCV_DEBUG_REG_WALL_CLOCK.
        if (sync.valid && tracy_ != nullptr) {
            for (uint32_t d = 0; d < ctx.n_drisc; d++) {
                // Keyed on NOC0, like every other context lookup; drisc_virtual is the VIRTUAL space, and the
                // register read needs the virtual pair. Absent mapping means self-profiling is off, so this
                // core has no Tracy row to anchor -- see the identical guard in the role loop below.
                const auto nit = ctx.virt_to_noc0.find(
                    (static_cast<uint64_t>(ctx.drisc_virtual[d].x) << 32) |
                    static_cast<uint64_t>(ctx.drisc_virtual[d].y));
                if (nit == ctx.virt_to_noc0.end()) {
                    continue;
                }
                const PerfDebugSync ds = sync_device_clock(cluster, ctx.chip_id, ctx.drisc_virtual[d]);
                if (!ds.valid) {
                    // Degrade to the worker anchor rather than dropping the rows: a misplaced row is still
                    // readable, an absent one is not. Loud, because it silently reinstates the whole bug.
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] Device {} DRISC {} at NOC0 ({},{}): DRAM-core clock sync FAILED; "
                        "its zones and plots fall back to the WORKER anchor and will be shifted by the "
                        "reset->open gap",
                        ctx.chip_id,
                        d,
                        nit->second.first,
                        nit->second.second);
                    continue;
                }
                // ONE FREQUENCY (slope) for every context; each core keeps its own ANCHOR (origin). Alignment is
                // RELATIVE, so a shared rate makes differential drift zero BY CONSTRUCTION and any error in it is
                // common-mode -- invisible on a timeline. Measured: the true rates agree to ~5 ppm while the
                // per-core fits scattered over ~99 ppm, so this trades a 99 ppm noise term for a 5 ppm physical
                // one (N+47; the per-core alternative costs 48-119 us and grows at 29-72 ppm, N+48).
                // Not selectable: the per-core alternative is strictly worse, so no knob chooses it. To reproduce
                // that comparison, pass `ds.frequency` here instead.
                tracy_->AddCore(
                    ctx.chip_id,
                    nit->second.first,
                    nit->second.second,
                    ds.host_anchor,
                    static_cast<double>(ds.device_at_anchor),
                    sync.frequency);
                // Log the OFFSET, not just the anchor: it is the board-dependence tell. Microseconds means the
                // part shares an origin and this changed nothing; minutes means it just fixed the capture.
                // Divided by the SHARED slope -- the one this row is actually rendered with, so the reported
                // offset cannot disagree with the mapping in force.
                const double off_ms =
                    (static_cast<double>(ds.device_at_anchor) - static_cast<double>(sync.device_at_anchor)) /
                    (sync.frequency > 0.0 ? sync.frequency : 1.0) / 1e6;
                // The core's own fit is REPORTED but never applied: its spread is the diagnostic that identified
                // this error term (measured -79.6 to +71.7 ppm across 9 runs on 2 parts).
                const double fit_ppm =
                    sync.frequency > 0.0 ? (ds.frequency - sync.frequency) / sync.frequency * 1e6 : 0.0;
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] Device {} DRISC {} NOC0 ({},{}) clock sync: frequency={:.6f} GHz "
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
        // NOTE: per-core Tracy contexts are created LAZILY on each core's first zone (HandleWorkerZone ->
        // GetOrCreateContext). We deliberately do NOT pre-create the full worker grid here: only ~16 of
        // ~110 cores typically run the workload, and pre-creating all of them litters the capture with
        // empty (count=0) contexts that read as "cores not showing up". The per-zone mutex+lookup cost is
        // identical either way; lazy creation just avoids minting dead contexts.
        ctx.active = true;
        devices_.push_back(std::move(ctx));
    }

    // Build the receiver AFTER devices_ is stable: socket ownership moves to it, and the lane tables it
    // hands consumers are flattened from the boot-time maps here so no consumer ever does a per-record
    // hash lookup.
    if (!devices_.empty()) {
        std::vector<perf_debug::ReceiverDeviceConfig> rdevs;
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
                                      ? perf_debug::PerfDebugLaneRole::Filler
                                      : perf_debug::PerfDebugLaneRole::Worker;
                for (uint32_t r = 0; r < kNRisc; r++) {
                    rd.lane_table.push_back(perf_debug::PerfDebugLaneInfo{
                        ctx.chip_id,
                        static_cast<uint16_t>(vx),
                        static_cast<uint16_t>(vy),
                        static_cast<uint16_t>(nx),
                        static_cast<uint16_t>(ny),
                        static_cast<uint8_t>(r),
                        role});
                }
            }
            for (uint32_t sk = 0; sk < kNSockets; sk++) {
                if (ctx.sockets[sk] != nullptr) {
                    TT_FATAL(sk == rd.sockets.size(), "sockets must form a contiguous prefix");
                    rd.sockets.push_back(std::move(ctx.sockets[sk]));
                }
            }
        }
        // No load_zone_names hook: names come per-ELF from llrt::ZoneMetaRegistry, which each consumer
        // mirrors lazily (the table GROWS as binaries JIT-load, so a one-shot snapshot would be taken when
        // it holds a fraction of its final size). PRODUCER-STALL and the DRISC self-zones are ordinary
        // zones with ordinary ELF records now -- nothing is registered by hand.
        receiver_ = std::make_unique<perf_debug::PerfDebugReceiver>(std::move(rdevs));
        if (tracy_push_enabled()) {
            tracy_consumer_ = std::make_unique<perf_debug::PerfDebugTracyConsumer>(tracy_.get());
            // An ordinary public consumer: Tracy takes device zones WHOLE now (one QueueGpuZone item
            // per zone), and the paired stream's per-lane completion order is exactly the order the
            // Tracy server rebuilds nesting from.
            receiver_->add_consumer(
                "tracy", [c = tracy_consumer_.get()](const perf_debug::PerfDebugRecordBatch& b) { (*c)(b); });
        }
        perf_debug::attach_registered_consumers(*receiver_);
        receiver_->start();
    }
    if (!devices_.empty()) {
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] active on {} device(s): DRISC drain -> {} MiB D2H socket -> {}",
            devices_.size(),
            host_fifo_bytes() / (1024 * 1024),
            tracy_push_enabled() ? "registered consumers + Tracy"
                                 : "registered consumers (Tracy off; opt in with TT_METAL_STREAMING_PROFILER_TRACY=1)");
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
    const std::string who = fmt::format("niu-mode({},{})->{}", drisc_logical.x, drisc_logical.y, stream);
    g_bringup_step = who + ":CompileProgram";
    detail::CompileProgram(device, p, /*force_slow_dispatch=*/true);
    g_bringup_step = who + ":WriteRuntimeArgs";
    detail::WriteRuntimeArgsToDevice(device, p, /*force_slow_dispatch=*/true);
    // This LaunchProgram is the heavyweight one: dram_barrier MMIO-polls a core in EVERY DRAM channel,
    // then wait_until_cores_done polls the launched core.
    // SPLIT ON PURPOSE, to tell the two halves of this step apart in a failure. The one-launch fix above
    // guarantees the dram_barrier runs BEFORE any core is in stream mode -- but the completion poll
    // necessarily runs AFTER, because this kernel's whole body IS the flip (drisc_niu_mode.cpp), so a
    // core that has finished is by definition already in stream mode, where "all inbound NoC traffic
    // terminates at L1" instead of being forwarded to GDDR. If the residual ~210 ms non-completing read
    // is the poll rather than the barrier, only the second label can ever appear on a failure.
    g_bringup_step = who + ":LaunchProgram(dram_barrier,no-wait)";
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    g_bringup_step = who + ":WaitProgramDone(poll-after-flip)";
    detail::WaitProgramDone(device, p);
    g_bringup_step = who + ":done";
}

// Flip every drainer's NIU in ONE launch.
//
// WHY THIS EXISTS (FINDINGS N+32/N+34). Flipping the NIUs one launch at a time hung the SECOND
// drainer's bring-up, and the instrumented repro named the site 4 times out of 4:
//
//   init failed at step [niu-mode(3,1)->1:LaunchProgram(dram_barrier+wait_until_cores_done)]
//
// Every LaunchProgram carries a `dram_barrier`, which MMIO-polls a core in EVERY DRAM channel, plus
// `wait_until_cores_done`. Done per drainer, the SECOND flip's barrier runs while the FIRST drainer is
// already resident with its DRAM core in stream mode -- and stream mode is exactly what changes the
// meaning of an inbound DRAM-range address on that core. The read never completes and the host eats a
// root-port completion timeout (~210 ms, N+31).
//
// One launch over all the cores means one barrier, and it happens BEFORE any core is in stream mode,
// so the hazardous ordering cannot arise. Restores (stream -> noc2axi) go through the same path.
void PerfDebugProfiler::set_drisc_niu_mode(
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
    // SPLIT for the same reason the single-core path above is split, and it should have been from the
    // start: the two halves fail for different reasons and only the label tells them apart. The barrier
    // runs BEFORE any core here is in stream mode (that is what one-launch buys, N+32/N+34), so a failure
    // on the first label means a core was ALREADY in stream mode when this run began -- a restore that did
    // not complete, or a reset that did not cover it. A failure on the second is the poll-after-flip
    // hazard, which is inherent to this kernel because its whole body IS the flip.
    g_bringup_step = who + ":LaunchProgram(dram_barrier,no-wait)";
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    g_bringup_step = who + ":WaitProgramDone(poll-after-flip)";
    detail::WaitProgramDone(device, p);
    g_bringup_step = who + ":done";
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

// Wait until every producer's ring is EMPTY, with the fillers still running. head is drainer-written and
// tail is producer-written, so head == tail on every RISC means the consumer has taken everything published.
//
// This has to happen before anything stops draining. Quiescing the fillers first leaves the workers' rings
// unserved while producers are still live -- dispatch cores keep emitting zones through device close, which
// is the very case PROFILER_TERMINATE was added for -- and they park in ring_ensure_room for however long
// that lasts. Measured: 100% of kimi's producer stalls sat inside the last ~130 ms of a 283 s capture, none
// anywhere else in the run. Draining first removes the window instead of dropping the markers in it.
bool PerfDebugProfiler::wait_producer_rings_drained(DeviceCtx& ctx, std::chrono::milliseconds budget) {
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

// Last resort, and the ONLY path that drops a marker: a producer still publishing after the drain budget
// expired. Unblocking it is what keeps device close from wedging in wait_until_cores_done().
void PerfDebugProfiler::disarm_producer_backpressure(DeviceCtx& ctx) {
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

bool PerfDebugProfiler::boot_device(
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

    // A Tensix drainer under FAST dispatch, which this used to refuse outright ("a resident worker program
    // cannot coexist with fast dispatch"). It can, on ONE core: dispatch_core_manager pops a worker off the
    // BACK of the dispatch pool for the real-time profiler and REMOVES it from logical_dispatch_cores, so FD
    // never allocates it. With the RT profiler off, that core is idle and a resident non-CQ program can own it.
    //
    // This matters because it is the missing cell of the 2x2. Every "Tensix survives where the DRISC hangs"
    // result compared DRISC+fast against Tensix+slow -- two variables at once -- precisely BECAUSE the Tensix
    // arm was believed to be slow-dispatch-only. It is not, and core type can finally be tested with dispatch
    // mode held fixed.

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
    // Slow dispatch hands the WHOLE worker grid to compute (12x10 here) because nothing is reserved for
    // dispatch; fast dispatch reserves the rest itself and returns 11x10.
    //
    // THE POLL LIST BUILT BELOW DEFINES THE DRAINED SET, AND A PRODUCER OUTSIDE IT HANGS THE WORKLOAD.
    // Producers are lossless: an undrained one fills its SPSC ring, blocks forever in ring_ensure_room,
    // never finishes, and the host dies in wait_until_cores_done on an uncaught 45 s throw. So the poll
    // list must cover every core the workload can land on.
    //
    // DRISC (default): poll the full 12x10. The drainer is on a DRAM core, so no worker is spent on it and
    // there is nothing to reserve -- `--gx 0` is then safe, because the producer grid and the poll list are
    // the same 120 cores by construction.
    //
    // Tensix: the drainer physically lives in the last column, so that column is ALWAYS held back and the
    // workload must be run with `--gx 11` to match. A producer placed there would both go undrained and
    // scribble on the drainer's L1 -- which is why the Tensix arm hangs on all 120 cores rather than 10.
    //
    // TT_METAL_PERF_DEBUG_RESERVE_COLUMN=1 forces the 110 reservation on the DRISC arm too, for 2x2 runs
    // where equal poll-list length (= equal idle sweep cost) matters more than coverage. See FINDINGS N+24.
    const bool reserve_column = slow_dispatch && reserve_column_env();
    const uint32_t gx = static_cast<uint32_t>(grid.x) - (reserve_column ? 1u : 0u);
    const uint32_t gy = static_cast<uint32_t>(grid.y);
    const uint64_t num_cores = static_cast<uint64_t>(gx) * gy;
    ctx.nl = static_cast<uint32_t>(num_cores) * kNRisc;
    ctx.core_virt.resize(num_cores);

    // Pre-zero every core's profiler control vector (heads and tails start clean) and build the maps the
    // HOST owns: core index -> virtual (x,y), which is the drainer's poll list and Tracy's view, and the
    // inverse packed (y<<16)|x -> core index, which is the one thing the drainer does not put on the wire.
    // Identity travels in the payload instead, written by the producing core into SPSC_CORE_XY.
    std::vector<uint32_t> coords(num_cores, 0);
    std::vector<uint8_t> zero_ctrl(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE, 0);
    // Enumeration ORDER decides what a contiguous filler slice means. Default is row-major; xsplit orders
    // by (NoC half, row, column) so the first n_left entries are exactly the cores the x=9 DRAM column
    // reaches without wrapping a row.
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
        if (filler_assign_xsplit()) {
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

    // ---- bring up each DRISC over a disjoint slice of the grid ----
    //
    // The split is contiguous rather than interleaved: each drainer's coords list is a run of the same
    // grid order the host uses everywhere else, so a core belongs to exactly one drainer and neither can
    // see the other's rings. Nothing is shared on the device -- separate L1, separate socket, separate
    // head mirrors -- so the two drain loops never interact.
    // ONE NIU FLIP FOR ALL DRAINERS, BEFORE ANY OF THEM IS IN STREAM MODE (FINDINGS N+34).
    // Doing this per drainer inside the loop below is what hung drainer 1's bring-up: each flip is a
    // LaunchProgram, every LaunchProgram carries a dram_barrier over every DRAM channel, and the second
    // one therefore barriered across drainer 0's already-stream-mode core. Picking the cores up front
    // costs a cheap repeat of the bank selection and removes the ordering entirely.
    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    if (nbanks < kNFillers) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] needs {} DRAM views (one filler each) but this part has {} -- the "
            "streaming profiler is OFF for this device.",
            kNFillers,
            nbanks);
        return false;
    }
    const auto& banks = role_filler_banks();
    TT_FATAL(banks.size() >= kNFillers, "perf-debug needs {} filler banks (got {})", kNFillers, banks.size());

    // ---- DRISC SELF-PROFILING: give each drainer a LANE BLOCK of its own ----------------------------------
    //
    // Must mirror the kernel's kSlotWords: capped slots except under self-zones, whose RAW self frame
    // needs the full span.
    const uint32_t slot_bytes_all = kernel_profiler::spsc_span_slot_words(kNRisc) * sizeof(uint32_t);

        std::vector<CoreCoord> flip_cores;
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            flip_cores.push_back(mesh_device->impl().pick_unused_dram_logical_core(ctx.device, banks[d]));
        }
        // TWO DRISCs MUST NEVER LAND ON THE SAME CORE. pick_unused_dram_logical_core() takes a DRAM VIEW and
        // reserves that view's worker/eth endpoints -- it has no idea another view may resolve to the SAME
        // physical port. The N+29 sweep records exactly that: view 0 and view 7 both come back as NoC core
        // 0-0. With six banks in play (and a filler-bank env override) it can happen, and the result would be
        // two resident kernels sharing one core's L1 -- staging, socket config, results, all overlapped, with
        // no counter that would notice. Refuse to launch instead.
        for (uint32_t a = 0; a < flip_cores.size(); a++) {
            for (uint32_t b = a + 1; b < flip_cores.size(); b++) {
                TT_FATAL(
                    flip_cores[a] != flip_cores[b],
                    "perf-debug: DRISC {} (DRAM view {}) and DRISC {} (DRAM view {}) both resolve to logical DRAM "
                    "core ({},{}). Two resident drain kernels cannot share a core -- pick different banks via "
                    "TT_METAL_PERF_DEBUG_FILLER_BANKS.",
                    a,
                    banks[a],
                    b,
                    banks[b],
                    flip_cores[a].x,
                    flip_cores[a].y);
            }
        }
        // DOES A DRAINER SIT ON A dram_barrier TARGET? Cluster::dram_barrier passes no subchannel, so
        // LocalChip::dram_membar syncs subchannel 0 of EVERY channel -- and every LaunchProgram carries one.
        // A drainer resident on such a core is in stream mode, where an inbound DRAM-range address no longer
        // forwards to GDDR, so the barrier is addressing a core whose semantics we changed. N+32 fixed the
        // ordering for the FLIP's own barrier (one launch, before any core flips); it cannot help the
        // barrier inside every later program, including weight upload, which runs with all six resident.
        // Reported rather than fatal: this configuration does usually work, and the point is to know
        // whether a bring-up or upload MMIO timeout had this available as an explanation.
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
                    "[perf-debug profiler] {} of {} drainers sit on a dram_barrier target core (subchannel 0 "
                    "of their channel). Every LaunchProgram barriers those cores while they are in stream "
                    "mode; a 60-70 ms MMIO timeout at bring-up or weight upload has this as a candidate.",
                    collide.size(),
                    flip_cores.size());
            } else {
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] no drainer sits on a dram_barrier target core (checked {} channels "
                    "against {} drainers).",
                    soc.get_num_dram_channels(),
                    flip_cores.size());
            }
        }
        set_drisc_niu_mode(ctx.device, flip_cores, 1);

    const std::vector<uint32_t>& slice_map = filler_slice_map();
    TT_FATAL(
        slice_map.empty() || slice_map.size() >= ctx.n_drisc,
        "TT_METAL_PERF_DEBUG_FILLER_SLICE_MAP needs {} entries, got {}",
        ctx.n_drisc,
        slice_map.size());
    // Weighted prefix split: filler slot sl owns cores [cum[sl], cum[sl+1]) scaled to num_cores, so a
    // slower filler owns proportionally fewer. Integer math on the running sum keeps the partition exact
    // (every core assigned once, no rounding gap) whatever the weights are.
    const std::vector<uint32_t>& weights_env = filler_weights();
    TT_FATAL(
        weights_env.empty() || weights_env.size() >= kNFillers,
        "TT_METAL_PERF_DEBUG_FILLER_WEIGHTS needs {} entries",
        kNFillers);
    std::vector<uint64_t> wcum(kNFillers + 1, 0);
    for (uint32_t i = 0; i < kNFillers; i++) {
        const uint32_t w = weights_env.empty() ? 1u : weights_env[i];
        TT_FATAL(w != 0, "filler weight {} must be non-zero", i);
        wcum[i + 1] = wcum[i] + w;
    }
    // ---- GDDR spool reservation (TT_METAL_PERF_DEBUG_DRAM_MB; 0 = direct push) ----
    // One REPLICATED mesh buffer with one interleaved page per DRAM bank: the same [address, address+spool)
    // window is reserved in every bank of every device, and each filler spools into its own bank through its
    // local DMA engine, so a single buffer covers all of them. It must be a MESH-level buffer: MeshBuffer
    // allocations run through the mesh lock-step allocator, which never sees a device-local Buffer::create
    // and would hand the same region out again.
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
                "[perf-debug profiler] could not reserve {} MiB/bank of DRAM for the GDDR spool ({}); falling "
                "back to direct push",
                dram_spool_mb(),
                e.what());
        }
    }
    if (spool_buffer_ != nullptr) {
        spool_addr = static_cast<uint32_t>(spool_buffer_->address());
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] GDDR spool: {} MiB per filler at bank offset 0x{:x}",
            dram_spool_mb(),
            spool_addr);
    } else {
        spool_bytes = 0;
    }
    // xsplit: a filler serves only the half its own DRAM column reaches without wrapping a row, and the
    // fillers of each column split that half between them.
    std::vector<uint32_t> xs_grp(ctx.n_drisc, 0), xs_rank(ctx.n_drisc, 0);
    uint32_t xs_n[2] = {0, 0};
    for (uint32_t d = 0; d < ctx.n_drisc && d < banks.size(); d++) {
        const uint32_t g = banks[d] >= 4u ? 0u : 1u;  // views D4-D7 sit in NOC0 column x=9, D0-D3 in x=0
        xs_grp[d] = g;
        xs_rank[d] = xs_n[g]++;
    }
    for (uint32_t d = 0; d < ctx.n_drisc; d++) {
        const uint32_t sl = slice_map.empty() ? d : slice_map[d];
        TT_FATAL(sl < kNFillers, "slice {} out of range for {} fillers", sl, kNFillers);
        uint32_t lo = static_cast<uint32_t>((num_cores * wcum[sl]) / wcum[kNFillers]);
        uint32_t hi = static_cast<uint32_t>((num_cores * wcum[sl + 1]) / wcum[kNFillers]);
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
        CoreCoord drisc_phys{};  // NOC0 coords of the drainer core, for the socket and the log line
        uint32_t region = 0;     // usable L1 on the drainer core
            // Host-facing duty from NoC rows y != 0 is DELIBERATE. FINDINGS N+29 measured host-facing
            // drainers hanging at 12.8% on y != 0 cores against 1.3% on the two y == 0 cores, and the
            // mover role existed to keep PCIe egress on those two -- but that sweep ran the socket's ack
            // path through UMD's dynamic per-access TLB reconfigure. Every filler now gets its own static
            // window (configured below) and the socket takes the static path through it, so watch the hang
            // rate rather than assume the old figure transfers.
            ctx.drisc_logical[d] = mesh_device->impl().pick_unused_dram_logical_core(ctx.device, banks[d]);
            if (const auto& sub_sel = filler_subchannels(); d < sub_sel.size()) {
                // Forced placement. Validated against the same reserved set the picker honours, so a
                // requested subchannel that is a worker/eth endpoint is refused rather than silently
                // double-booking a core.
                const uint32_t nsub = soc.get_grid_size(tt::CoreType::DRAM).y;
                TT_FATAL(sub_sel[d] < nsub, "filler {} subchannel {} >= {}", d, sub_sel[d], nsub);
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
                TT_FATAL(!reserved, "filler {} subchannel {} is a reserved worker/eth endpoint", d, sub_sel[d]);
                ctx.drisc_logical[d] =
                    soc.get_logical_dram_core_for_subchannel(static_cast<int>(banks[d]), static_cast<int>(sub_sel[d]));
            }
            {
                // What placement freedom exists: pick_unused_dram_logical_core returns the FIRST unreserved
                // subchannel of the view, and a view has several at different NoC coords. Log them all --
                // choosing among these is the locality lever, and nothing currently chooses.
                const uint32_t nsub = soc.get_grid_size(tt::CoreType::DRAM).y;
                const size_t chan = soc.get_channel_for_dram_view(static_cast<int>(banks[d]));
                std::string cand;
                for (uint32_t sub = 0; sub < nsub; sub++) {
                    const tt::umd::CoreCoord tc =
                        soc.get_dram_core_for_channel(static_cast<int>(chan), static_cast<int>(sub), tt::CoordSystem::TRANSLATED);
                    const tt::umd::CoreCoord nc = soc.translate_coord_to(tc, tt::CoordSystem::NOC0);
                    cand += fmt::format(" sub{}=NOC0({},{})", sub, nc.x, nc.y);
                }
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] filler {} bank {} chan {}: {} subchannels ->{} | chose logical ({},{})",
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
                tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED),
                CoordSystem::NOC0);
            drisc_phys = CoreCoord{phys.x, phys.y};
            ctx.drisc_virtual[d] = ctx.device->virtual_core_from_logical_core(ctx.drisc_logical[d], CoreType::DRAM);
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] filler {} at virtual ({},{}) owns band {} = cores [{}, {}) of {}",
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
        // done(64) + stop(64) + results(64 words = 256) + handshake(64). Was 512 when results was 48 words.
        constexpr uint32_t kMiscBytes = 1024;
        const uint32_t fixed = kCfgReserve + kScratchBytes + kMiscBytes;
        const uint32_t nstage = nstage_cap(region > fixed ? (region - fixed) / slot_bytes : 0);
        if (nstage == 0) {
            log_warning(tt::LogMetal, "[perf-debug profiler] Device {}: DRISC L1 too small; skipping", device_id);
            disarm_producers(mesh_device, device_id);
            return false;
        }
        // ---- DRISC SELF-PROFILING takes ONE staging slot, and takes it out of nstage rather than out of L1 ----
        //
        // The self frame is a full slot (prefix + control vector + five rings), so it needs somewhere slot-sized
        const uint32_t nstage_drain = nstage;
        const uint32_t stage_base = ctx.drisc_l1_base[d];
        const uint32_t core_records = stage_base + nstage * slot_bytes;
        ctx.done_addr[d] = core_records + kScratchBytes;
        ctx.stop_addr[d] = ctx.done_addr[d] + 64;
        const uint32_t cfg_l1 = ctx.drisc_l1_base[d] + region - kCfgReserve;
        TT_FATAL(ctx.stop_addr[d] + 64 <= cfg_l1, "DRISC L1 layout overlaps the socket config");

        // Stream mode first: the socket config is written from the host and only lands in L1 once the NIU
        // stops forwarding inbound DRAM-range addresses to GDDR. The kernel restores it on the host's word.
        // A Tensix NIU is already a NoC master, so this (and the kernel's restore tail) is DRISC-only.
            // NIU already flipped to stream mode for EVERY drainer by the single pre-pass above.

            // TT_METAL_PERF_DEBUG_NIU_TEST isolates the NIU mode flip from everything else the drainer does.
            // The flip is the ONLY thing either drainer path writes that outlives the process (NIU_CFG_0
            // persists until a chip reset), so it is the standing candidate for why the card stays bad after a
            // DRISC run -- but it has never been tested WITHOUT a drain underneath it. Flip, optionally
            // restore, then bail before any socket, kernel or egress exists.
            //   =leave -> stay in stream mode, exactly as a run that dies before the stop=2 handshake leaves it
            //   =flip  -> restore NOC2AXI immediately (the clean-teardown control)
            const char* niu_test = std::getenv("TT_METAL_PERF_DEBUG_NIU_TEST");
            if (niu_test != nullptr && *niu_test != '\0') {
                const bool restore = std::string_view(niu_test) != "leave";
                if (restore) {
                    set_drisc_niu_mode(ctx.device, ctx.drisc_logical[d], 0);
                }
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] NIU TEST: DRISC {} logical ({},{}) flipped to stream mode and {} "
                    "-- no socket, no kernel, no egress",
                    d,
                    ctx.drisc_logical[d].x,
                    ctx.drisc_logical[d].y,
                    restore ? "RESTORED to NOC2AXI" : "LEFT IN STREAM MODE");
                disarm_producers(mesh_device, device_id);
                return false;
            }

        // Give the DRISC drainer a STATIC TLB window, so the socket's per-read ack write skips UMD's
        // per-access TLB reconfigure -- the same path the Tensix drainer already gets for free. Measured on
        // bh-05: 171 ns/write static vs 382 ns dynamic, i.e. ~210 ns of pure reconfigure per socket read().
        //
        // Metal maps static windows at device init for workers/eth/dispatch and, on Blackhole, one 4 GB
        // window per DRAM channel -- but only on that channel's PREFERRED WORKER ENDPOINT port
        // (ll_api::configure_static_tlbs -> blackhole::ddr_to_noc0 takes the channel's last of 3 NoC ports).
        // The drainer deliberately sits on the *unused* port (pick_unused_dram_logical_core), so its core is
        // NOT in that map and the socket would otherwise have no window to find. Configure one here: 2 MB at
        // address 0 spans the DRISC's whole 128 KB L1 (MEM_DRISC_L1_BASE = 0), and Strict ordering matches
        // what workers get, so both drainers end up on an identical host write path.
        //
        // Best-effort: a window is a finite device resource, and losing this race only costs the ~210 ns.
        if (!no_static_tlb() && !cluster.is_mock_or_emulated()) {
            auto* tlb_manager = cluster.get_driver()->get_chip(device_id)->get_tlb_manager();
            const tt_xy_pair tlb_core(ctx.drisc_virtual[d].x, ctx.drisc_virtual[d].y);
            if (!tlb_manager->is_tlb_mapped(tlb_core)) {
                try {
                    g_bringup_step = fmt::format("drainer {}: configure static TLB", d);
                    tlb_manager->configure_tlb(
                        tlb_core, /*tlb_size=*/2 * 1024 * 1024, /*address=*/0, tt::umd::tlb_data::Strict);
                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] could not configure a static TLB for DRISC core ({}, {}): {} "
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
                // sender_uses_physical_noc_addr switches the socket between "physical NoC coord + full L1 addr" (DRISC,
                // drainer) and the normal worker path (logical coord, worker-L1 semantics). The socket picks the
                // static-vs-dynamic write path by ASKING UMD whether this core has a window (see init_sender_tlb),
                // so the window configured just above is what puts the DRISC on the static path.
                g_bringup_step = fmt::format("drainer {}: D2HSocket construct (writes config into DRISC L1)", d);
                ctx.sockets[sk] = std::make_unique<distributed::D2HSocket>(
                    mesh_device,
                    distributed::MeshCoreCoord{
                        scoord, CoreCoord(drisc_phys.x, drisc_phys.y)},
                    (host_fifo_bytes() / kPageSize) * kPageSize,
                    distributed::D2HSocket::ExternalConfigBuffer{
                        .address = cfg_l1, .sender_uses_physical_noc_addr = true});
                ctx.sockets[sk]->set_page_size(kPageSize);
                // MEASURE the flow-control poll directly. The per-poll cost derived from the rounded "poll X%"
                // log line differed ~10x between a fast and a degraded card, but that figure is too indirect to
                // build a diagnosis on -- and the non-hugepage path reads HOST memory, not MMIO, so a ~1 us cost
                // would not make sense there. Time the actual call and report which path it takes.
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
                        "[perf-debug profiler] flow-control poll probe: {:.0f} ns/call over {} calls | hugepage "
                        "path: {} | (sink {})",
                        ns_per,
                        kPollProbe,
                        ctx.sockets[sk]->is_using_hugepage() ? "YES (clflush+lfence)" : "no (mfence, host buffer)",
                        sink);
                    // ACK-WRITE PROBE. socket read() ends in notify_sender(), which PCIe-writes bytes_acked to
                    // the sender core -- one DEVICE write per read. This is the access the poll probe did NOT
                    // cover (that one reads host memory: 13 ns on every box measured). A fixed ~4 us/read
                    // overhead on one card is the right order for a dynamic-TLB reconfigure per access, so
                    // measure the write and say which TLB path it uses. Writing the current bytes_acked is
                    // idempotent -- it re-sends the value the socket already holds.
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
                            "[perf-debug profiler] ACK-WRITE probe: {:.0f} ns/write over {} device writes | TLB "
                            "path: {}",
                            ack_ns,
                            kAckProbe,
                            ctx.sockets[sk]->has_static_tlb() ? "STATIC window" : "DYNAMIC (reconfigure per access)");
                    }
                    // DEVICE SMALL-READ probe. 4-byte cluster.read_core from the drainer core -- the SAME access
                    // wait_until_cores_done() issues per core and the one that blows the 2 ms budget in the
                    // "MMIO per-op timeout: 4B load took N us" aborts. The write probe above and this read probe
                    // together cover both directions; the flow-control poll probe covers neither (host memory).
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
                            "[perf-debug profiler] DEVICE-READ probe: {:.0f} ns/read over {} 4B device reads (acc {})",
                            rd_ns,
                            kRdProbe,
                            acc);
                    }
                    // WORKER-CORE CONTROL PROBE -- the one that makes "degraded" mean something.
                    //
                    // Both probes above target the DRAINER core, which is a DRAM core on the DRISC arm and a
                    // worker on the Tensix arm. So the two arms have never measured the same destination, and
                    // every "the card is degraded" number ever recorded here is really "accesses to the DRAM
                    // endpoint are ~13x slower". Whether a WORKER access on that same card is also slow was never
                    // measured -- and "the Tensix arm never degrades" may just be that it never probes the
                    // endpoint that degrades.
                    //
                    // Probing a fixed worker in EVERY run settles it in one observation:
                    //   worker slow too  => card-wide MMIO degradation, the drainer core is incidental
                    //   worker fine      => the DRAM endpoint alone is sick, and the Tensix arm is BLIND to it
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
                            "[perf-debug profiler] WORKER-READ control probe: {:.0f} ns/read over {} 4B reads from "
                            "worker virtual ({},{}) (acc {}) -- compare against DEVICE-READ above: both slow = "
                            "card-wide, worker fast = the drainer's endpoint alone",
                            wk_ns,
                            kWkProbe,
                            ctx.core_virt[0].first,
                            ctx.core_virt[0].second,
                            acc);
                    }
                }
            }

            // Zero done AND the heartbeat word behind it. Zeroing only `done` leaves the PREVIOUS
            // run's heartbeat in L1, so a drainer that never starts reads as the last run's final state --
            // which is exactly how a failed start got misread as "exited and wedged in the socket tail".
            // ZERO THE DRAINER CORE'S OWN PROFILER RING -- for BOTH core types.
            //
            // The drain kernel is built with PROFILE_KERNEL=1 whichever core it lands on, so the firmware
            // writes its own zone markers (BRISC-KERNEL / DRISC-KERNEL, ~7 words) into THIS core's profiler
            // ring on every launch -- and this core is deliberately excluded from the drained core set, so
            // nothing ever empties it. The ring is 512 words and the SPSC backend BLOCKS on a full ring
            // rather than dropping, so after ~74 launches the RISC wedges in firmware init before
            // kernel_main and the drainer silently never starts.
            //
            // Measured and fixed on Tensix (6/6 at launch 74; 600 clean runs after). The DRAM path has the
            // SAME exposure -- bh_hal_dram.cpp gives DRAM cores a PROFILER region of sizeof(profiler_msg_t)
            // and drisck.cc emits DRISC-KERNEL zones -- it just was never run 74 times inside one
            // card-reset window. Zero it here for both rather than wait to rediscover it on the DRISC.
            const uint64_t drainer_prof_l1 =
                hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER);
            cluster.write_core(
                zero_ctrl.data(),
                (uint32_t)zero_ctrl.size(),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                drainer_prof_l1);

            // done | heartbeat and the rest of the 64 B pad: a stale value from the previous run reads as
            // this run's live state.
            uint32_t zero3[13] = {};
            cluster.write_core(
                zero3,
                sizeof(zero3),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
            // ALSO the stop word -- teardown leaves it at 1 (quiesce) or 2 (free the NIU), and the drain loop
            // is `while (... && *stop == 0 ...)`, so a stale value would make the next kernel exit after ONE
            // sweep while the host reports FAILED TO START. (Not the cause of the slow-dispatch wedge below --
            // that reproduces with stop=0 -- but the same class of stale-state bug as the heartbeat word.)
            // FOUR words, not one: stop plus the sync-event rendezvous triple (req | ack | go) that shares its
            // 64 B pad. A stale `req` from a previous run would make every drainer park at a barrier nobody is
            // going to release, and it would present as the workload wedging at the first sweep -- the same
            // signature as the stale-stop bug this write already existed to prevent.
            uint32_t zero4[4] = {};
            cluster.write_core(
                zero4,
                sizeof(zero4),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));

            ctx.drain_program[d] = std::make_unique<Program>(CreateProgram());
            // Compile-time arguments by NAME: the kernel reads them with
            // get_named_compile_time_arg_val, so retiring one is a local edit on both sides.
            const std::unordered_map<std::string, uint32_t> cargs = {
                {"stage_base", stage_base},
                {"n_stage", nstage_drain},
                {"core_records", core_records},
                {"done_addr", ctx.done_addr[d]},
                {"stop_addr", ctx.stop_addr[d]},
                {"socket_config_addr", ctx.sockets[sk]->get_config_buffer_address()},
                {"max_cores", kMaxCores},
                // With the egress NoC alternating on d&1, d&2 splits each NoC's pushers across two of
                // the four unicast request VCs; TT_METAL_PERF_DEBUG_FILLER_VCS (comma-separated, one
                // entry per filler) overrides the whole assignment for arbitration experiments at the
                // shared PCIe tile.
                {"write_vc", d < filler_vcs().size() ? filler_vcs()[d] : ((d & 2u) ? 0u : 1u)},
                {"ship_min_pct", ship_min_pct()},
                // The bounce slots cost the kernel a staging generation, so the spool needs the full
                // slot count; a smaller L1 falls back to direct push rather than failing the kernel's
                // geometry static_asserts.
                {"spool_base", spool_addr},
                {"spool_bytes", nstage_drain >= 7u ? spool_bytes : 0u}};
            if (spool_bytes != 0 && nstage_drain < 7u) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] Device {}: only {} staging slots fit, too few for the spool's bounce "
                    "buffers; filler {} runs direct push",
                    device_id,
                    nstage,
                    d);
            }
            TT_FATAL(
                my_cores * 32u <= slot_bytes_all,
                "CV-first tails staging ({} cores x 32 B) does not fit inside the slot past the pipeline",
                my_cores);
            auto drain_id = CreateKernel(
                *ctx.drain_program[d],
                "tt_metal/tools/profiler/kernels/drisc_profiler_filler.cpp",
                ctx.drisc_logical[d],
                DramConfig{
                    .noc = (drain_noc_override() < 0 ? false : drain_noc_override() == 1) ? NOC::NOC_1
                                                                                                  : NOC::NOC_0,
                    .defines = {{"PERF_DEBUG_DRAIN_KERNEL", "1"}},
                    .named_compile_args = cargs});
            std::vector<uint32_t> rt = {my_cores, static_cast<uint32_t>(prof_l1)};
            // Reversed: launch order follows global index, so the slice's last-launched cores (the
            // join-blind victims) land in the first-chunk slots, which are read and serviced first.
            rt.insert(rt.end(), coords.rbegin() + (coords.size() - hi), coords.rbegin() + (coords.size() - lo));
            SetRuntimeArgs(*ctx.drain_program[d], drain_id, ctx.drisc_logical[d], rt);

            detail::CompileProgram(ctx.device, *ctx.drain_program[d], /*force_slow_dispatch=*/true);
            detail::WriteRuntimeArgsToDevice(ctx.device, *ctx.drain_program[d], /*force_slow_dispatch=*/true);
            g_bringup_step = fmt::format("drainer {}: drain kernel LaunchProgram", d);
            detail::LaunchProgram(
                ctx.device, *ctx.drain_program[d], /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);

            // VERIFY THE DRAINER ACTUALLY STARTED. A resident drainer is launched fire-and-forget, so a core
            // that fails to come out of reset produces no error -- the producers simply fill their rings,
            // block (they are lossless), and the workload wedges forever with a perfectly healthy card. That
            // is the same failure the earlier drain port hit: the host checked one hart, the rest never started, and
            g_bringup_step = fmt::format("drainer {}: heartbeat verify", d);
            // the run hung. Poll the heartbeat instead of assuming: it must leave 0 and then advance.
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
                // Poll for ADVANCE rather than sampling once 2 ms later: a single short sample cannot tell a
                // dead drainer from a slow one. 200 ms is ~6000 idle sweeps of headroom at 30 us/sweep.
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
                        "[perf-debug profiler] Device {}: drainer {} FAILED TO START (heartbeat {} -> {} after "
                        "launch, stop word {}). The producers would block forever on a full ring and wedge the "
                        "workload, so capture is disabled for this run instead.",
                        device_id,
                        d,
                        hb0,
                        hb1,
                        stopw);
                    ctx.drain_program[d].reset();
                    ctx.sockets[sk].reset();
                    disarm_producers(mesh_device, device_id);
                    return false;
                }
            }
        } catch (const std::exception& e) {
            // A code-region overflow makes the drainer fail to LOAD, not merely fail to start, and the run
            // then completes with exit 0 while every marker is dropped. Measured cost of that being quiet:
            // a 101 KB capture with zero device zones that looked like a successful run, whose only tells
            // were a warning 47 lines in and the ABSENCE of the per-DRISC report. Name the cause at the
            // top, at error level, and say what it costs the capture.
            const std::string what = e.what();
            const bool elf_too_big = what.find("overflows region") != std::string::npos;
            log_error(
                tt::LogMetal,
                "[perf-debug profiler] Device {}: DRISC {} FAILED TO LOAD{} -- THIS CAPTURE WILL BE EMPTY. No "
                "device zones will be produced and the run will still exit 0.{} ({})",
                device_id,
                d,
                elf_too_big ? " (drain kernel ELF EXCEEDS THE DRISC CODE REGION)" : "",
                elf_too_big ? " Reduce drain-kernel code: a u64 division anywhere in the kernel costs a "
                              "956 B soft-div."
                            : "",
                what);
            ctx.drain_program[d].reset();
            ctx.sockets[sk].reset();
            disarm_producers(mesh_device, device_id);
            return false;
        }

        log_info(
            tt::LogMetal,
            "[perf-debug profiler] Device {}: {} {} resident on logical ({},{}) [noc0 ({},{})], cores "
            "[{},{}) of {}, {} staging slots x {} B",
            device_id,
            "DRISC FILLER (worker rings -> D2H socket)",
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

void PerfDebugProfiler::stop() {
    if (stopped_.exchange(true)) {
        return;
    }

    // BEFORE any quiesce: the drainers must still be looping to answer a rendezvous, and firing here rather
    // Producers before consumers: let the rings empty while the fillers are still draining them, so no
    // producer ever meets a stopped consumer.
    for (auto& ctx : devices_) {
        if (!wait_producer_rings_drained(ctx, std::chrono::seconds(2))) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] Device {}: producers still publishing after the 2 s drain budget -- "
                "unblocking ring back-pressure so device close cannot wedge; markers still in flight on those "
                "cores are DROPPED",
                ctx.chip_id);
            disarm_producer_backpressure(ctx);
        }
    }

    // Tell each DRISC to quiesce, then wait for it to publish `done` -- which it does only after its
    // socket barrier, so every page is already on its way to the host when we stop reading.
    for (auto& ctx : devices_) {
        auto& cluster = MetalContext::instance().get_cluster();
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            if (ctx.drain_program[d] == nullptr) {
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
                // The receiver's decode threads are still draining, so the socket keeps emptying while we wait.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if ((done & 0xFFFF0000u) != 0xD09E0000u) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] Device {}: DRISC drainer did not acknowledge stop",
                    ctx.chip_id);
            } else if (receiver_ != nullptr) {
                // done follows the drainer's socket barrier, i.e. the host has already read and acked every
                // byte this socket will ever carry -- the stream can retire itself on one final empty check.
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
        perf_debug::detach_registered_consumers();
        receiver_->shutdown();
    }
    for (auto& ctx : devices_) {
        verify_completeness(ctx, static_cast<uint32_t>(&ctx - devices_.data()));
    }
    if (receiver_ != nullptr) {
        receiver_->log_report();
        // Zone naming, from the per-ELF metadata sections. `id collisions 0` is one half of the naming
        // invariant; the other half (unnamed marker rows, MUST also be 0) is reported per consumer as each
        // one tears down, since the name mirrors live there now.
        const auto zm = llrt::ZoneMetaRegistry::instance().stats();
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] zone names: {} records from {} ELFs | id collisions {} | foreign/stale "
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
    // After the drainers are quiesced (nothing touches the spool any more) and while the mesh allocator
    // is still alive to take the region back.
    spool_buffer_.reset();
}

// One MMIO pass per worker core: the producer-owned stall counters (the knee metric -- nothing downstream
// can lose them, so they are valid even under NO_DECODE), and each lane's own tail against the receiver's
// consumed-words mirror, which is the direct assertion that the stop-path sweep-to-empty held.
void PerfDebugProfiler::verify_completeness(DeviceCtx& ctx, uint32_t device_index) {
    if (ctx.core_virt.empty()) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint64_t prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    std::vector<uint32_t> heads;
    if (receiver_ != nullptr && !perf_debug::env_flag("TT_METAL_PERF_DEBUG_NO_DECODE")) {
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
    // WORKER cores only. With DRISC self-profiling on, core_virt also holds the drainer cores, and a DRAM
    // core has no producer and no stall counters -- reading the TENSIX profiler address on one returns
    // whatever is at that offset in DRISC L1.
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
        "[perf-debug profiler] Device {}: L1 STALL COUNTERS -- {} producer stalls across {} of {} cores "
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
            "[perf-debug profiler] Device {}: stall breakdown by RISC -- BR {} | NC {} | T0 {} | T1 {} | T2 {}; "
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
            "[perf-debug profiler] COMPLETENESS: device {} -- {}/{} lanes fully drained, 0 words stranded",
            ctx.chip_id,
            checked_lanes,
            checked_lanes);
    } else {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] COMPLETENESS: device {} -- {}/{} lanes fully drained; {} lanes stranded {} "
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
