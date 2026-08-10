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
#include <optional>
#include <span>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tracy/Tracy.hpp>
#include <common/TracyTTDeviceData.hpp>  // tracy::RiscType worker lanes

#include <chrono>
#include <x86intrin.h>
#include <thread>

#include <tt-metalium/allocator.hpp>
#include "impl/dispatch/dispatch_core_manager.hpp"
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
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/types/tlb.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig (a DRISC kernel is not in the public headers yet)
#include "jit_build/build_env_manager.hpp"
#include "llrt/tt_cluster.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tools/profiler/spsc_marker_decode.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"
#include "tools/profiler/perf_debug_profiler_packets.hpp"
#include "impl/profiler/profiler.hpp"  // generateZoneSourceLocationsHashes (zone hash -> name)
#include "tools/profiler/spsc_packet.h"
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
// Default 4M == the standalone drain harness's --mqcap default (~96 MB at 24 B/Rec). A lagging consumer DROPS rather
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

// TT_METAL_PERF_DEBUG_SHIP_REPEAT: EGRESS AMPLIFIER for stress testing. N>1 makes the drainer re-send each
// staged frame N times, so egress bandwidth stops being bounded by producer rate -- the extra sends skip the
// read and process phases. Written to answer "can PCIe egress alone hang the card?" on the Tensix drainer,
// whose own ceiling is read/process (saturated at 511/512 ring occupancy while pushing only 5.2 GB/s).
// The host then receives N duplicate copies of every frame, so this is NOT a valid capture: pair it with
// TT_METAL_PERF_DEBUG_NO_DECODE=1 and read the page/byte counters, not the markers.
uint32_t ship_repeat() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_SHIP_REPEAT");
        const uint32_t n = (s == nullptr || *s == '\0') ? 1u : static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
        return n == 0 ? 1u : n;
    }();
    return v;
}

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

// TT_METAL_PERF_DEBUG_WRITER_TIMEOUT_S: how long the writer tolerates zero progress before giving up.
// Breaking out stops acking FOREVER, which converts a transient stall into a permanent deadlock (a drainer
// in socket_reserve_pages spins with no escape and never re-checks *stop), so this is a diagnostic knob:
// lower it to surface a hang quickly, and read the drainer dump it now emits on the way out.
std::chrono::seconds writer_timeout() {
    static const std::chrono::seconds v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_WRITER_TIMEOUT_S");
        const uint32_t n = (s == nullptr || *s == '\0') ? 120u : static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
        return std::chrono::seconds(n == 0 ? 120u : n);
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_WRITER_DIE_AFTER: test hook -- writer thread exits after N successful reads, so the
// "consumer vanished mid-stream" deadlock can be reproduced deliberately. 0 (default) disables it.
uint32_t writer_die_after() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_WRITER_DIE_AFTER");
        return (s == nullptr || *s == '\0') ? 0u : static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_DRAIN_TENSIX: run the drain kernel on a Tensix BRISC instead of a DRISC. Control
// path only -- see boot_device(). Requires slow dispatch.
bool drain_on_tensix() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_DRAIN_TENSIX");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_NO_NOC_INIT: do not resync the drainer's software NoC counter mirrors from hardware at
// kernel entry. That resync is what fixes the slow-dispatch wedge (a resident core's mirrors persist across
// launches, so a run that ends with writes unacked leaves the next run's write barrier unsatisfiable). This
// knob exists to bring the wedge BACK on the same binary -- a fix you cannot un-apply is a fix you cannot
// prove. Diagnostic only; it deliberately reintroduces a hang.
bool no_noc_init() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_NO_NOC_INIT");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return v;
}

// ABLATION: strip the drain loop to EGRESS ONLY. TT_METAL_PERF_DEBUG_ABLATE=1 compiles out every worker
// read and all per-core processing; the drainer re-ships the same pre-staged mock bytes forever. Purpose is to
// bisect the hang: if DRAM-core -> PCIe egress alone can hang the card, the read side is irrelevant.
// TT_METAL_PERF_DEBUG_ABLATE_SPIN is a cycle count that stands in for the sweep. IT MUST BE LARGE. A real run
// is only 1.7% duty (268 busy sweeps of 15,477) and idles ~1.7 ms between bursts, which is what lets the host
// fully drain the FIFO so the NEXT burst runs at the true ~16 GB/s. Spins of 4k-40k cycles (3-30 us) are ~57x
// too small: the loop stays permanently credit-bound at ~3.9 GB/s and the spin appears to do nothing, because
// when you are already blocked on credits a spin only displaces wait time. Use ~2.3M cycles (~1.7 ms) to
// reproduce real burst behaviour. Pair with NO_DECODE=1 -- the payload is mock.
uint32_t ablate() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ABLATE");
        return (s == nullptr || *s == '\0') ? 0u : static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    }();
    return v;
}

uint32_t ablate_spin() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ABLATE_SPIN");
        return (s == nullptr || *s == '\0') ? 0u : static_cast<uint32_t>(std::strtoul(s, nullptr, 10));
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_NOC selects which NIU the DRISC drainer EGRESSES on (reads use the other one). Default
// 0, matching every result recorded so far. Exists to test whether the hang follows the NoC rather than the
// core: if egress on NoC 1 stops hanging, NoC 0's route from the DRAM endpoint to the PCIe tile is implicated.
//
// NOT just a flag flip. On Blackhole NoC 1 MIRRORS coordinates
// (NOC_0_X_PHYS_COORD(noc, size_x, x) = noc == 0 ? x : size_x - 1 - x), while the socket's pcie_xy_enc is
// built from NOC0 coords by a NoC-agnostic hal function. Flipping NOC_INDEX alone would aim every payload
// write at the wrong tile -- which could hang the card for a reason that has nothing to do with the question.
// So the host re-encodes the PCIe tile in mirrored coords and passes it down as an override.
uint32_t drain_noc() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_NOC");
        return (s == nullptr || *s == '\0') ? 0u : (std::strtoul(s, nullptr, 10) == 1 ? 1u : 0u);
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

// DRAM banks a resident DRISC drainer may safely occupy. MEASURED, not derived -- see the call site
// and FINDINGS N+29. Drainer d takes kSafeBanks[d].
constexpr uint32_t kSafeBanks[] = {0, 3};
constexpr uint32_t kNumSafeBanks = sizeof(kSafeBanks) / sizeof(kSafeBanks[0]);

// TT_METAL_PERF_DEBUG_DRISC_BANK: DIAGNOSTIC override of the bank drainer 0 takes (drainer d then
// takes base+d). Unset = use kSafeBanks, which is what production wants. Returns -1 when unset, so
// that an explicit "=0" is distinguishable from the default.
int drisc_bank_override() {
    static const int v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_DRISC_BANK");
        return (s != nullptr && *s != '\0') ? static_cast<int>(std::strtol(s, nullptr, 10)) : -1;
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
// TT_METAL_PERF_DEBUG_FILL_PCT: target span fill for the drainer's pacing controller, as a percent of a
// span's live capacity (kNumRisc * ring words). 0 disables the loop and leaves the fixed
// TT_METAL_PERF_DEBUG_DRISC_GAP behaviour.
//
// Why it exists: the drainer ships the WHOLE span per core per frame regardless of how much is live, so
// host cost is frames x 10,560 B and the fill ratio decides bytes-per-marker. Sweeping continuously
// against slow producers returns ~37%-full spans, which is why producer stalls got WORSE as producers got
// SLOWER -- ~2x the host bytes for the same payload. Pacing holds the spans full instead.
uint32_t fill_target_pct() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FILL_PCT");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 70u;
    }();
    return v;
}

// Ceiling on the controller's gap, in DRISC cycles. Bounds worst-case capture latency: a core with data
// waits at most this long between sweeps. ~148 us at 1.35 GHz.
//
// MUST be well above the gap the controller actually wants, or it saturates and the loop never closes.
// Measured at delay 125 (120 cores, 6M markers): with a 20,000 ceiling it pinned at 20,000 and producers
// still stalled; at 50,000 it pinned at 50,000; at 100,000 it pinned at 100,000; only at 200,000 did it
// SETTLE, at 108,881 -- i.e. the true operating point is ~109k and every smaller ceiling was clipping it.
// A gap pinned exactly at this value in the results is the signature of a ceiling that is too low.
// TT_METAL_PERF_DEBUG_READ_SPLIT=1: issue the batch's span reads alternately on BOTH NoCs instead of
// only kReadNoc. The busy sweep is read-latency bound and the batch cannot grow (DRISC L1 holds 7 spans),
// so splitting is the only way to raise outstanding read transactions.
uint32_t read_split() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_READ_SPLIT");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 0u;
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_BACKOFF_US: the writer's sleep when a poll round finds every socket empty.
//
// It was a fixed 50 us. That is fine when reads are large (cap 1024 = 64 KB retires a lot per wake), but
// at a small cap each read frees only ~11 KB, so the writer drains what is visible, sleeps 50 us, and the
// FIFO refills and credit-starves the DRAINER during the nap -- which is how the device reports
// credit-wait while the writer reports 70% idle.
uint32_t writer_backoff_us() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_BACKOFF_US");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 50u;
    }();
    return v;
}

uint32_t gap_max_cycles() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_GAP_MAX");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 200000u;
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_ACK_PREDRAIN=1: issue an explicit store fence BEFORE the ack and time it separately.
// notify_sender() is a 4 B PCIe write plus an sfence, and the ACK-WRITE probe clocks that pair at ~175 ns in
// a quiet loop -- yet in the drain path it costs ~4,000 ns. The difference is that here it lands directly
// after a 65 KB memcpy, so the sfence has to drain that store backlog synchronously. If that is the whole
// story, the pre-drain absorbs the cost and the ack itself falls back to a few hundred ns.
// ---- low-overhead timing for per-read costs -------------------------------------------------------
//
// std::chrono::steady_clock::now() costs ~650 ns a call here, which is FATAL for timing events that are
// themselves a few microseconds: an EMPTY timed region measured 1,303 ns, i.e. the instrument was a third
// of the "ack" it was supposed to be measuring. rdtsc is ~20-30 cycles. Accumulate ticks, convert once at
// report time.
inline uint64_t tsc_now() { return __rdtsc(); }

double tsc_ns_per_tick() {
    static const double v = [] {
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t c0 = __rdtsc();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(20)) {
        }
        const uint64_t c1 = __rdtsc();
        const auto t1 = std::chrono::steady_clock::now();
        const double ns =
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        return (c1 > c0) ? ns / static_cast<double>(c1 - c0) : 0.0;
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_RESIZE_ZERO=1 restores the OLD buffer handling (clear() on return to the pool, exact
// resize per read), which value-initialized the whole buffer immediately before memcpy overwrote it. Kept
// so the fix can be A/B'd on silicon instead of argued from first principles.
bool resize_zero_legacy() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_RESIZE_ZERO");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return v;
}

bool ack_predrain() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ACK_PREDRAIN");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return v;
}

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
    // irrelevant). Same registers the drainer firmware co-samples in calibrate().
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
    // consumer (ring -> Tracy), matching the standalone drain harness: the slow Tracy sink is off the drain path.
    if (!devices_.empty()) {
        const uint32_t cap = max_pages_per_read(kMaxPagesPerRead);
        // Records per read: a page holds at most page_words/2 two-word markers.
        const size_t recs_per_page = (kPageSize / sizeof(uint32_t)) / 2;
        read_chunk_recs_ = cap ? static_cast<size_t>(cap) * recs_per_page : static_cast<size_t>(kHRingWords);
        ring_ = std::make_unique<RecRingHolder>(ring_capacity_recs());
        // PRE-POPULATE THE BUFFER POOL.
        //
        // The writer takes a buffer from free_bufs, or default-constructs an empty one when the pool is
        // dry -- and an empty vector then gets resize()d to the read size, which VALUE-INITIALIZES all
        // 64 KB immediately before memcpy overwrites every byte. Measured on a healthy card that cost
        // ~9.6 ms/run, 76% of the writer's entire workload, against 3.0 ms for the copy it precedes.
        //
        // Grow-only sizing alone could not fix it: the decoder runs 231-253 buffers behind, so the pool is
        // dry most of the time and nearly every read allocated a FRESH (size 0) vector. Handing out
        // already-sized buffers is what makes grow-only actually bite -- after this, resize() is a no-op
        // on the steady-state path and the zeroing happens once per buffer at startup instead of once per
        // read.
        const size_t max_read_words = static_cast<size_t>(cap ? cap : kHRingWords) * (kPageSize / sizeof(uint32_t));
        const size_t kPrefillBufs = (cap && cap < 512) ? 1536 : 320;  // small caps => far more reads in flight  // > the ~253 max queue depth observed, so the pool stays warm
        for (uint32_t s = 0; s < kNSockets; s++) {
            std::lock_guard<std::mutex> lk(dq_[s].m);
            for (size_t i = 0; i < kPrefillBufs; i++) {
                dq_[s].free_bufs.emplace_back(max_read_words, 0u);
                dq_[s].allocated++;
            }
        }
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
    g_bringup_step = who + ":LaunchProgram(dram_barrier+wait_until_cores_done)";
    detail::LaunchProgram(device, p, /*wait_until_cores_done=*/true, /*force_slow_dispatch=*/true);
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

bool PerfDebugProfiler::boot_device(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t device_id = ctx.chip_id;
    const auto& soc = cluster.get_soc_desc(device_id);

    // TT_METAL_PERF_DEBUG_DRAIN_TENSIX=1 runs the identical drain kernel on a Tensix BRISC instead of a
    // DRISC. It is a control for "does the DRAM core have anything to do with the PCIe hang", not a product
    // mode: it needs TT_METAL_SLOW_DISPATCH_MODE=1 so the dispatch row/column is free (the drainer core is
    // taken from there, leaving the producers the full compute grid) and so a resident non-CQ program is
    // legal on a worker at all.
    const bool tensix_drain = drain_on_tensix();
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
    std::optional<CoreCoord> fd_tensix_core;
    if (tensix_drain && !slow_dispatch) {
        auto rt_core = MetalContext::instance(context_id).get_dispatch_core_manager().get_reserved_realtime_profiler_core(
            device_id);
        if (!rt_core.has_value()) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] Device {}: no reserved real-time-profiler core to borrow, so a Tensix "
                "drainer has nowhere to live under fast dispatch (reservation is skipped for non-MMIO chips, "
                "ETH dispatch, fabric tensix datamover, and Quasar) -- use TT_METAL_SLOW_DISPATCH_MODE=1",
                device_id);
            disarm_producers(mesh_device, device_id);
            return false;
        }
        fd_tensix_core = CoreCoord{rt_core->x, rt_core->y};
    }

    // The drainer is a DRISC: one DM RISC-V on a DRAM core. Nothing else here is Blackhole-specific, but
    // that is the only place they exist today.
    if (!tensix_drain && !hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
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
    const bool reserve_column = slow_dispatch && (tensix_drain || reserve_column_env());
    const uint32_t gx = static_cast<uint32_t>(grid.x) - (reserve_column ? 1u : 0u);
    const uint32_t gy = static_cast<uint32_t>(grid.y);
    const uint64_t num_cores = static_cast<uint64_t>(gx) * gy;
    ctx.nl = static_cast<uint32_t>(num_cores) * kNRisc;
    if (first_ts_.size() < ctx.nl) {
        first_ts_.assign(ctx.nl, 0);  // stagger probe (see header); 0 = lane not seen yet
    }
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
    // ONE NIU FLIP FOR ALL DRAINERS, BEFORE ANY OF THEM IS IN STREAM MODE (FINDINGS N+34).
    // Doing this per drainer inside the loop below is what hung drainer 1's bring-up: each flip is a
    // LaunchProgram, every LaunchProgram carries a dram_barrier over every DRAM channel, and the second
    // one therefore barriered across drainer 0's already-stream-mode core. Picking the cores up front
    // costs a cheap repeat of the bank selection and removes the ordering entirely.
    if (!tensix_drain) {
        const uint32_t nbanks_pre = static_cast<uint32_t>(soc.get_num_dram_views());
        const int bank_ov_pre = drisc_bank_override();
        std::vector<CoreCoord> flip_cores;
        for (uint32_t d = 0; d < kNSockets; d++) {
            const uint32_t bank = (bank_ov_pre >= 0)
                                      ? static_cast<uint32_t>((static_cast<uint32_t>(bank_ov_pre) + d) % nbanks_pre)
                                      : kSafeBanks[d];
            flip_cores.push_back(mesh_device->impl().pick_unused_dram_logical_core(bank));
        }
        set_drisc_niu_mode(ctx.device, flip_cores, 1);
    }

    for (uint32_t d = 0; d < kNSockets; d++) {
        const uint32_t lo = static_cast<uint32_t>((num_cores * d) / kNSockets);
        const uint32_t hi = static_cast<uint32_t>((num_cores * (d + 1)) / kNSockets);
        const uint32_t my_cores = hi - lo;
        if (my_cores == 0) {
            continue;
        }
        CoreCoord drisc_phys{};  // NOC0 coords of the drainer core, for the socket and the log line
        uint32_t region = 0;     // usable L1 on the drainer core
        if (tensix_drain) {
            // Under slow dispatch the dispatch row/column is idle, so the drainer takes a core from there
            // and the producers keep the FULL compute grid -- the offered load is then identical to the
            // DRISC runs, which is the only way the two are comparable.
            // Column gx is the one held back above; drainer d takes row d of it.
            // Slow dispatch: the held-back column is free, drainer d takes row d of it. Fast dispatch: the
            // column belongs to dispatch, so borrow the idle RT-profiler core instead (see above). Either way
            // the drainer sits OUTSIDE the producer grid, which is what keeps the two arms comparable.
            if (fd_tensix_core.has_value()) {
                ctx.drisc_logical[d] = *fd_tensix_core;
            } else {
                TT_FATAL(
                    d < gy,
                    "drainer {} does not fit the reserved column (only {} rows)",
                    d,
                    gy);
                ctx.drisc_logical[d] = CoreCoord{gx, d};
            }
            ctx.drisc_virtual[d] = ctx.device->virtual_core_from_logical_core(ctx.drisc_logical[d], CoreType::WORKER);
            drisc_phys = cluster.get_physical_coordinate_from_logical_coordinates(
                device_id, ctx.drisc_logical[d], CoreType::WORKER, /*no_warn=*/true);
            // A Tensix's unreserved L1 belongs to the allocator, so the HAL refuses to name it (hal.hpp:705)
            // -- take the allocator's base instead and run to the top of L1. Safe to carve raw here because
            // the drainer core is outside the producer grid and this workload allocates no L1 buffers; a
            // workload that did would need a real sharded allocation on this core.
            ctx.drisc_l1_base[d] = ctx.device->allocator()->get_base_allocator_addr(HalMemType::L1);
            ctx.drisc_l1_noc[d] = ctx.drisc_l1_base[d];  // worker L1 is addressed directly, no DRAM-view offset
            region = ctx.device->l1_size_per_core() - static_cast<uint32_t>(ctx.drisc_l1_base[d]);
        } else {
            // TT_METAL_PERF_DEBUG_DRISC_BANK shifts which DRAM bank drainer d takes. DEFAULT 0, AND
            // BANK 0 IS THE ONE THAT IS KNOWN GOOD -- do not "fix" this to something else.
            //
            // It was added to test a hypothesis that turned out to be WRONG, and the measurement is worth
            // more than the hypothesis. The theory: pick_unused_dram_logical_core() reserves the bank's
            // WORKER and ETH endpoints and returns the first subchannel left, while UMD's dram_membar()
            // barriers SUBCHANNEL 0 of every channel (dram_membar(channels, subchannel = 0); Cluster::
            // dram_barrier passes no subchannel). On banks 0 and 4-7 the worker/eth endpoints are [2,1],
            // so the only free port IS subchannel 0 -- the drainer lands exactly on the core the host
            // barriers, and we put that core in stream mode. Banks 1-3 have endpoints [0,1], so the
            // drainer gets subchannel 2, which the barrier never polls. That predicted bank 0 wedges and
            // bank 1 does not.
            //
            // A DRISC DRAINER IS ONLY SAFE ON A DRAM CORE IN NoC ROW y == 0. Measured, 8 banks x 25
            // runs, randomized, resetting after every non-clean run so the runs are independent
            // (bh-26, 2026-08-08; FINDINGS N+29):
            //
            //   bank 0 -> core 0-0  y=0    0/25    bank 1 -> 0-3  y=3    3/25
            //   bank 3 -> core 9-0  y=0    0/25    bank 2 -> 0-8  y=8    2/25
            //   bank 7 -> core 0-0  y=0    1/25    bank 4 -> 9-2  y=2    3/25
            //                                      bank 5 -> 9-9  y=9    5/25
            //                                      bank 6 -> 9-5  y=5    3/25
            //   grouped: y==0  1/75 (1.3%)   vs   y!=0  16/125 (12.8%)   Fisher p ~ 0.006
            //
            // Note it is NOT the subchannel: banks 4/5/6 are subchannel 0 of their channel and still
            // hang. The failures are MMIO per-op timeouts -- the host's small reads stop completing.
            // Why row 0 is special is unproven; the soc descriptor notes that CMFW reads DRAM telemetry
            // through a particular noc0 endpoint ("to avoid SYS-1419"), which is a lead, not a finding.
            //
            // There are exactly two safe cores, 0-0 and 9-0, one per DRAM side -- which is what caps
            // kNSockets at 2. pick_unused_dram_logical_core() does NOT know any of this: it reserves
            // worker and eth endpoints only, so "unused" does not mean "safe to repurpose".
            const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
            const int bank_ov = drisc_bank_override();
            const uint32_t bank = (bank_ov >= 0) ? static_cast<uint32_t>((static_cast<uint32_t>(bank_ov) + d) % nbanks)
                                                 : kSafeBanks[d];
            TT_FATAL(
                bank_ov >= 0 || d < kNumSafeBanks,
                "perf-debug: {} DRISC drainers requested but only {} DRAM banks are known safe (row y==0). "
                "Raising kNSockets needs a bank safety sweep first -- see FINDINGS N+29.",
                kNSockets,
                kNumSafeBanks);
            ctx.drisc_logical[d] = mesh_device->impl().pick_unused_dram_logical_core(bank);
            const CoreCoord translated =
                soc.dram_bank_endpoint_coords.at(ctx.drisc_logical[d].x).at(ctx.drisc_logical[d].y);
            const tt::umd::CoreCoord phys = soc.translate_coord_to(
                tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED),
                CoordSystem::NOC0);
            drisc_phys = CoreCoord{phys.x, phys.y};
            ctx.drisc_virtual[d] = ctx.device->virtual_core_from_logical_core(ctx.drisc_logical[d], CoreType::DRAM);
            ctx.drisc_l1_base[d] = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
            ctx.drisc_l1_noc[d] = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
            region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        }

        const uint32_t span_bytes = (kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE +
                                     kNRisc * kernel_profiler::PROFILER_L1_VECTOR_SIZE) *
                                    sizeof(uint32_t);
        const uint32_t slot_bytes = kernel_profiler::SPSC_SPAN_PREFIX_WORDS * sizeof(uint32_t) + span_bytes;
        constexpr uint32_t kCfgReserve = 8 * 1024;
        constexpr uint32_t kScratchBytes = 128 * 32;
        constexpr uint32_t kMiscBytes = 512;
        const uint32_t fixed = kCfgReserve + kScratchBytes + kMiscBytes;
        const uint32_t nstage = nstage_cap(region > fixed ? (region - fixed) / slot_bytes : 0);
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
        TT_FATAL(ctx.results_addr[d] + 192 <= cfg_l1, "DRISC L1 layout overlaps the socket config");

        // Stream mode first: the socket config is written from the host and only lands in L1 once the NIU
        // stops forwarding inbound DRAM-range addresses to GDDR. The kernel restores it on the host's word.
        // A Tensix NIU is already a NoC master, so this (and the kernel's restore tail) is DRISC-only.
        if (!tensix_drain) {
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
        if (!tensix_drain && !no_static_tlb() && !cluster.is_mock_or_emulated()) {
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

        try {
            // sender_is_l2cpu switches the socket between "physical NoC coord + full L1 address" (DRISC,
            // drainer) and the normal worker path (logical coord, worker-L1 semantics). The socket picks the
            // static-vs-dynamic write path by ASKING UMD whether this core has a window (see init_sender_tlb),
            // so the window configured just above is what puts the DRISC on the static path.
            g_bringup_step = fmt::format("drainer {}: D2HSocket construct (writes config into DRISC L1)", d);
            ctx.sockets[d] = std::make_unique<distributed::D2HSocket>(
                mesh_device,
                distributed::MeshCoreCoord{
                    scoord,
                    tensix_drain ? ctx.drisc_logical[d] : CoreCoord(drisc_phys.x, drisc_phys.y)},
                static_cast<uint32_t>((static_cast<uint64_t>(kHRingWords) * 4 / kPageSize) * kPageSize),
                distributed::D2HSocket::ExternalConfigBuffer{.address = cfg_l1, .sender_is_l2cpu = !tensix_drain});
            ctx.sockets[d]->set_page_size(kPageSize);
            // MEASURE the flow-control poll directly. The per-poll cost derived from the rounded "poll X%"
            // log line differed ~10x between a fast and a degraded card, but that figure is too indirect to
            // build a diagnosis on -- and the non-hugepage path reads HOST memory, not MMIO, so a ~1 us cost
            // would not make sense there. Time the actual call and report which path it takes.
            {
                const auto t0 = std::chrono::steady_clock::now();
                constexpr uint32_t kPollProbe = 2000;
                uint32_t sink = 0;
                for (uint32_t k = 0; k < kPollProbe; k++) {
                    sink += ctx.sockets[d]->pages_available();
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
                    ctx.sockets[d]->is_using_hugepage() ? "YES (clflush+lfence)" : "no (mfence, host buffer)",
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
                        ctx.sockets[d]->probe_ack_write();
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
                        ctx.sockets[d]->has_static_tlb() ? "STATIC window" : "DYNAMIC (reconfigure per access)");
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
            ctx.decode[d] = std::make_unique<pz::SpscDecodeState>();
            ctx.decode[d]->reset(ctx.nl);
            for (uint32_t c = 0; c < num_cores; c++) {
                ctx.decode[d]->core_of_xy[coords[c]] = c;  // full map: lane ids stay global across drainers
            }

            // Zero done AND the heartbeat/phase words behind it. Zeroing only `done` leaves the PREVIOUS
            // run's hb/phase in L1, so a drainer that never starts reads as the last run's final state --
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
                tensix_drain ? prof_l1
                             : hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER);
            cluster.write_core(
                zero_ctrl.data(),
                (uint32_t)zero_ctrl.size(),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                drainer_prof_l1);

            uint32_t zero3[3] = {0, 0, 0};
            cluster.write_core(
                zero3,
                sizeof(zero3),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
            // ALSO the stop word -- teardown leaves it at 1 (quiesce) or 2 (free the NIU), and the drain loop
            // is `while (... && *stop == 0 ...)`, so a stale value would make the next kernel exit after ONE
            // sweep while the host reports FAILED TO START. (Not the cause of the slow-dispatch wedge below --
            // that reproduces with stop=0 -- but the same class of stale-state bug as the hb/phase words.)
            uint32_t zero1 = 0;
            cluster.write_core(
                &zero1,
                sizeof(zero1),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
            // Same reason, for the results block: it is published only on kernel exit, so a drainer that is
            // still running at teardown leaves the PREVIOUS run's numbers there and they read as this run's.
            // That is how a 42 s run reported "495.7 ms, credit-wait 0.1%" and hid its own credit timeouts.
            const std::vector<uint32_t> zero_res(48, 0);
            cluster.write_core(
                zero_res.data(),
                (uint32_t)(zero_res.size() * sizeof(uint32_t)),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.results_addr[d] - ctx.drisc_l1_base[d]));

            // Mirrored PCIe-tile encoding for NoC 1 (see drain_noc()). 0 => kernel uses the socket's NOC0 value.
            uint32_t pcie_enc_override = 0;
            // MEASURED WRONG, kept only as a knob: mirroring the PCIe tile for NoC 1 makes the payload land
            // somewhere else -- pages still flow (socket credits are a separate path) but decode yields ZERO
            // markers. The PCIe encoding is in TRANSLATED space (the kernel is built with PCIE_NOC_X=19,
            // PCIE_NOC_Y=24, outside the 17x12 NOC0 grid), and NOC_0_X_PHYS_COORD mirrors WORKER coordinates,
            // not this. Default is now to use the socket's own encoding on both NoCs.
            const char* mirror_env = std::getenv("TT_METAL_PERF_DEBUG_NOC_MIRROR");
            const bool want_mirror = mirror_env != nullptr && *mirror_env != '\0' && *mirror_env != '0';
            if (!tensix_drain && drain_noc() == 1 && want_mirror) {
                const auto& mmio_soc = cluster.get_soc_desc(cluster.get_associated_mmio_device(device_id));
                const auto pcie_noc0 = mmio_soc.get_cores(CoreType::PCIE, CoordSystem::NOC0).front();
                const uint32_t mx = static_cast<uint32_t>(mmio_soc.grid_size.x) - 1 - static_cast<uint32_t>(pcie_noc0.x);
                const uint32_t my = static_cast<uint32_t>(mmio_soc.grid_size.y) - 1 - static_cast<uint32_t>(pcie_noc0.y);
                pcie_enc_override = hal.noc_xy_pcie64_encoding(mx, my);
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] NoC 1 egress: PCIe tile NOC0 ({},{}) -> NOC1 ({},{}) on a {}x{} grid, "
                    "enc 0x{:x}",
                    pcie_noc0.x, pcie_noc0.y, mx, my, mmio_soc.grid_size.x, mmio_soc.grid_size.y, pcie_enc_override);
            }

            ctx.drain_program[d] = std::make_unique<Program>(CreateProgram());
            const std::vector<uint32_t> cargs = {
                stage_base,
                nstage,
                head_scratch,
                ctx.results_addr[d],
                ctx.done_addr[d],
                ctx.stop_addr[d],
                ctx.sockets[d]->get_config_buffer_address(),
                0xFFFFFFFFu,
                128,
                drisc_gap_cycles(),
                ship_repeat(),
                no_noc_init() ? 0u : 1u,
                ablate(),
                ablate_spin(),
                nstage,
                (my_cores + nstage - 1) / nstage,
                pcie_enc_override,
                fill_target_pct(),
                gap_max_cycles(),
                read_split()};
            const std::string kdrain = "tt_metal/tools/profiler/kernels/drisc_profiler_drain.cpp";
            auto drain_id =
                tensix_drain ? CreateKernel(
                                   *ctx.drain_program[d],
                                   kdrain,
                                   ctx.drisc_logical[d],
                                   DataMovementConfig{
                                       .processor = DataMovementProcessor::RISCV_0,
                                       .noc = NOC::RISCV_0_default,
                                       .compile_args = cargs,
                                       .defines = {{"DRAIN_ON_TENSIX", "1"}}})
                             : CreateKernel(
                                   *ctx.drain_program[d],
                                   kdrain,
                                   ctx.drisc_logical[d],
                                   DramConfig{.noc = drain_noc() == 1 ? NOC::NOC_1 : NOC::NOC_0, .compile_args = cargs});
            std::vector<uint32_t> rt = {my_cores, static_cast<uint32_t>(prof_l1)};
            rt.insert(rt.end(), coords.begin() + lo, coords.begin() + hi);
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
                const uint64_t hb_addr =
                    ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]) + 4;
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
                    // Report WHERE it stopped, not just that it did. `phase` is the kernel's own progress
                    // marker; without it this warning sent me chasing a stale stop word and a TLB change,
                    // when phase=11 says plainly that it is wedged in the sweep-body write barrier.
                    uint32_t st[5] = {0, 0, 0, 0, 0};
                    cluster.read_core(
                        st, sizeof(st), core, ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
                    uint32_t stopw = 0;
                    cluster.read_core(
                        &stopw,
                        sizeof(stopw),
                        core,
                        ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] Device {}: drainer {} FAILED TO START (heartbeat {} -> {} after "
                        "launch). The producers would block forever on a full ring and wedge the workload, so "
                        "capture is disabled for this run instead. State: done=0x{:x} hb={} phase={} stop={} "
                        "| write-barrier predicate: HW_ACK_RECEIVED={} vs SW_acked={} (equal => flushed; "
                        "unequal and FROZEN => the software mirror is out of sync, not a stalled NoC) "
                        "(phase: 1=INIT 2=POLL 3=RESERVE 4=WRITE 5=EXIT 6-9=write-substeps 11-13=barriers "
                        "14=socket-barrier 15=tail-barrier)",
                        device_id,
                        d,
                        hb0,
                        hb1,
                        st[0],
                        st[1],
                        st[2],
                        stopw,
                        st[3],
                        st[4]);
                    ctx.drain_program[d].reset();
                    ctx.sockets[d].reset();
                    disarm_producers(mesh_device, device_id);
                    return false;
                }
            }
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
            "[perf-debug profiler] Device {}: {} {} resident on logical ({},{}) [noc0 ({},{})], cores "
            "[{},{}) of {}, {} staging slots x {} B",
            device_id,
            tensix_drain ? "TENSIX-BRISC drainer" : "DRISC",
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

// ONE read+decode pass over (ctx, sock_idx): pages -> spsc_decode -> PerfDebugRec -> BroadcastRing.
// Returns true if it moved data. Deliberately does NOT touch Tracy: the sink lives on the consumer thread so
// a slow Tracy push can never back-pressure the FIFO -> relay -> reader -> worker cores. (Measured: with the
// push inline, UFLD-v2 held relay0 in HOST-WAIT 15.85 s of a 19 s run and stalled producers 826x; with the
// push removed, 0 stalls. This is the same structure the standalone drain harness uses.)
// Decode + publish, OFF the reader thread.
//
// MEASURED: with this inline, the reader's ack rate was gated by per-marker work -- read 3.8 ms, decode
// 13.5 ms, publish 8.7 ms, so 85% of host time sat between one ack and the next. At delay 300 that produced
// 17,366 producer stalls where the same device code with a minimal decode produced 0. The copy was never the
// problem (15% of host work, 15-19 GB/s); the interpretation was.
//
// Sequential by construction: SpscDecodeState carries sticky timer highs, the packet residual and the
// per-lane head mirror across buffers, so buffers for one socket MUST be decoded in arrival order. That is
// why there is one decoder per socket rather than a pool.
void PerfDebugProfiler::decode_and_publish(
    DeviceCtx& ctx, uint32_t sock_idx, std::vector<uint32_t>& buf, size_t words) {
    DeviceCtx::SockState& ss = ctx.sock_state[sock_idx];
    pz::SpscDecodeState& st = *ctx.decode[sock_idx];
    static const bool ddbg = (std::getenv("TT_PERF_DEBUG_ZONE_DUMP") != nullptr);
    (void)ddbg;
    const auto t_dec_all = std::chrono::steady_clock::now();
    if (stall_only()) {
        pz::spsc_decode(
            st, buf.data(), words, ctx.nl,
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
    pz::spsc_decode(
        st,
        buf.data(),
        words,
        ctx.nl,
        [&](uint32_t lane, uint32_t type, uint32_t hash, uint64_t ts, uint32_t prog) {
            // SPSC wire codes, NOT hostdevcommon PacketTypes: the two sources never co-exist and share no
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
            // byte-identical to the standalone drain harness's Rec.
            *bcur++ = PerfDebugRec{ts, (dev_idx << 24) | lane, type, hash, prog};
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
    // Stagger probe: remember each lane's FIRST marker timestamp (cheap: one compare per record).
    if (!first_ts_.empty()) {
        for (size_t k = 0; k < bn; k++) {
            const uint32_t lane = ss.batch[k].lane & 0x00FFFFFFu;
            if (lane < first_ts_.size() && first_ts_[lane] == 0) {
                first_ts_[lane] = ss.batch[k].ts;
            }
        }
    }
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
    // No decode state here any more -- this thread only reads and acks; SpscDecodeState belongs to the
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
        // A FLOW-CONTROL VIOLATION, not a quirk: pages_available() is (bytes_sent - bytes_acked)/page, so
        // exceeding the FIFO means the sender wrote more than the FIFO holds at least once. Clamping keeps
        // the read in bounds but desynchronizes host read_ptr from device write_ptr -- the host then acks
        // bytes it never consumed, over-crediting the sender permanently. Say so loudly, once per socket.
        if (!ss.overflow_reported) {
            ss.overflow_reported = true;
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] socket {}: FLOW-CONTROL VIOLATION -- pages_available={} >= fifo_pages={} "
                "(sender wrote past the FIFO). Clamping to {}; host read_ptr and device write_ptr are now "
                "desynchronized and the sender may be permanently over-credited.",
                sock_idx,
                np,
                fifo_pages,
                fifo_pages - 1u);
        }
        np = fifo_pages - 1u;
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
    const uint64_t t_pool0 = tsc_now();
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
    w_pool_ns_ += tsc_now() - t_pool0;
    std::vector<uint32_t>& dst = discard ? ss.buf : pooled;
    {
        ZoneScopedNC("buf-resize", 0xD35400);
        // GROW ONLY. std::vector::resize VALUE-INITIALIZES, so resizing to the exact read size every time
        // zeroed the whole buffer immediately before memcpy overwrote every byte of it -- 12.3 ms per run,
        // more than the copy itself (7.9 ms). Buffers are pooled, so letting them grow monotonically to the
        // largest read means the zeroing happens a handful of times instead of once per read. The valid
        // length travels in DecodeItem::words.
        const uint64_t tr = tsc_now();
        const size_t need = static_cast<size_t>(np) * page_words;
        if (resize_zero_legacy()) {
            dst.resize(need);  // legacy: exact size every read; pairs with clear() below
        } else if (dst.size() < need) {
            dst.resize(need);
        }
        w_resize_ns_ += tsc_now() - tr;
    }
    {
        // SPLIT so the byte-proportional part is visible on its own. read(notify_sender=false) is
        // wait_for_bytes + the memcpys + pop_bytes; the ack is then issued separately and timed. Same work,
        // same order, same point in time -- the ack still goes out before the buffer is handed to the
        // decoder, so the drainer's credit is released exactly as early as before.
        ZoneScopedNC("sock-read", 0x27AE60);  // green: pulls pages -- the byte-proportional stage
        const uint64_t t0 = tsc_now();
        sock->read(dst.data(), np, /*notify_sender=*/false);
        const uint64_t t1 = tsc_now();
        if (ack_predrain()) {
            asm volatile("sfence" ::: "memory");
        }
        const uint64_t t1b = tsc_now();
        sock->probe_ack_write();  // notify_sender(): one PCIe write + sfence
        const uint64_t t2 = tsc_now();
        w_read_ns_ += t1 - t0;
        w_predrain_ns_ += t1b - t1;
        w_ack_ns_ += t2 - t1b;
        w_reads_++;
        w_bytes_ += static_cast<uint64_t>(np) * kPageSize;
    }
    if (decode_disabled() || discard) {
        return true;
    }
    // Hand the raw buffer to the decoder and go straight back to polling. This is the whole point: the ack
    // (issued inside sock->read above) is no longer behind 85% of host work.
    const uint64_t t_enq0 = tsc_now();
    {
        std::lock_guard<std::mutex> lk(dq_[sock_idx].m);
        dq_[sock_idx].work.push_back(
            DecodeItem{&ctx, sock_idx, std::move(pooled), static_cast<size_t>(np) * page_words});
    }
    dq_[sock_idx].cv.notify_one();
    w_enq_ns_ += tsc_now() - t_enq0;
    return true;
}

// The single writer thread: round-robin every (device, socket); each drain_pass publishes its own read as one
// data-driven batch, then wake readers once per sweep. Idle sweeps back off. Mirrors the standalone drain harness.
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
        decode_and_publish(*item.ctx, item.sock, item.buf, item.words);
        std::lock_guard<std::mutex> lk(q.m);
        if (resize_zero_legacy()) {
            item.buf.clear();  // legacy path: forces the next resize to zero the whole buffer
        }
        q.free_bufs.push_back(std::move(item.buf));
    }
}

void PerfDebugProfiler::writer_thread(uint32_t sock_idx) {
    tracy::SetThreadName("perf-debug-writer");
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
    auto backoff = std::chrono::microseconds(writer_backoff_us());
    // Drain-to-empty on stop: stop() sets P_STOP first, so the drainer stops producing; keep reading until every
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
        // TEST HOOK: stop acking after N successful reads, to reproduce "the host consumer went away while
        // the device is mid-stream" on demand. That is the state the unbounded credit wait used to deadlock
        // in; with the bounded wait the drainer should drop frames and the workload should still finish.
        if (any && writer_die_after() != 0) {
            static uint32_t reads_done = 0;
            if (++reads_done >= writer_die_after()) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] TEST HOOK: writer exiting after {} reads; the device will lose "
                    "its credits from here on",
                    reads_done);
                break;
            }
        }
        if (all_done) {
            break;
        }
        if (any) {
            watchdog = std::chrono::steady_clock::now();
        } else {
            if (std::chrono::steady_clock::now() - watchdog > writer_timeout()) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] writer WALL TIMEOUT ({} s no progress)",
                    std::chrono::duration_cast<std::chrono::seconds>(writer_timeout()).count());
                // Before giving up -- which permanently stops acking and so permanently strands a drainer
                // that IS waiting on credits -- record what the drainer is actually doing. A starving writer
                // and a credit-blocked drainer are contradictory states (blocked implies a FULL fifo), so if
                // both appear the flow-control accounting has desynchronized.
                for (auto& c : devices_) {
                    for (uint32_t dd = 0; dd < kNSockets; dd++) {
                        dump_drainer_state(c, dd, "writer-wall-timeout");
                    }
                }
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
                "zones: {} [0 = drainer kept up, non-zero = capture perturbed the workload]",
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
    tracy::SetThreadName("perf-debug-consumer");
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
            zone_names_[0x7FFFu] = "PRODUCER-STALL";  // PROFILER_STALL_ZONE_ID
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

// Report how tightly the producer cores started, straight out of the capture: per-lane FIRST marker
// timestamp. If the cores start together the spread is small and many cores have data in the same drainer
// sweep; if they are staggered the drainer keeps finding a few cores at a time. That is exactly the
// difference between the "fast" and "degraded" profiles (57 vs 12 frames per busy sweep), and this measures
// it without any new silicon experiment.
void PerfDebugProfiler::report_lane_spread() {
    std::vector<uint64_t> seen;
    seen.reserve(first_ts_.size());
    for (uint64_t t : first_ts_) {
        if (t != 0) {
            seen.push_back(t);
        }
    }
    if (seen.size() < 2) {
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] lane-spread probe: only {} lanes had markers -- nothing to compare "
            "(decode must be ON for this probe)",
            seen.size());
        return;
    }
    std::sort(seen.begin(), seen.end());
    const uint64_t lo = seen.front(), hi = seen.back();
    const double kCycPerUs = 1.35e3;
    auto us = [&](uint64_t c) { return static_cast<double>(c) / kCycPerUs; };
    const uint64_t med = seen[seen.size() / 2];
    const uint64_t p90 = seen[(seen.size() * 9) / 10];
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] lane-spread probe: {} lanes started | first-marker spread {:.1f} us "
        "(median +{:.1f} us, p90 +{:.1f} us from first) | {:.2f} us mean gap between consecutive lane starts",
        seen.size(),
        us(hi - lo),
        us(med - lo),
        us(p90 - lo),
        us(hi - lo) / static_cast<double>(seen.size() - 1));
}

void PerfDebugProfiler::dump_drainer_state(DeviceCtx& ctx, uint32_t d, const char* why) {
    if (ctx.drain_program[d] == nullptr) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const tt_cxy_pair drisc(ctx.chip_id, ctx.drisc_virtual[d]);
    const uint64_t base = ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]);
    // done | heartbeat | phase, read as one 3-word block, twice, so a frozen heartbeat is distinguishable
    // from a slow one. 60 ms is ~2000 sweeps of headroom at the measured 27-30 us/sweep.
    uint32_t a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
    cluster.read_core(a, sizeof(a), drisc, base);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    cluster.read_core(b, sizeof(b), drisc, base);
    const bool exited = (a[0] & 0xFFFF0000u) == 0xD09E0000u;
    const char* phase_name = "?";
    switch (b[2]) {
        case 1: phase_name = "INIT"; break;
        case 2: phase_name = "POLL"; break;
        case 3: phase_name = "RESERVE(credit-wait)"; break;
        case 4: phase_name = "WRITE"; break;
        case 5: phase_name = "EXIT"; break;
        default: break;
    }
    uint32_t np = 0, fifo_pages = 0;
    if (ctx.sockets[d]) {
        np = ctx.sockets[d]->pages_available();
        fifo_pages = ctx.sockets[d]->get_fifo_curr_size() / ctx.sockets[d]->get_page_size();
    }
    log_warning(
        tt::LogMetal,
        "[perf-debug profiler] DRAINER STATE ({}) dev {} drainer {}: done=0x{:08X} ({}) | heartbeat {} -> {} "
        "({}) | phase {} ({}) | host sees {} of {} fifo pages available",
        why,
        ctx.chip_id,
        d,
        a[0],
        exited ? "KERNEL EXITED" : "still resident",
        a[1],
        b[1],
        b[1] == a[1] ? "FROZEN" : "advancing",
        b[2],
        phase_name,
        np,
        fifo_pages);
    if (b[1] == a[1] && !exited && b[2] == 3 && np == 0) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler]   => CONTRADICTION: drainer blocked on credits while host sees an EMPTY "
            "fifo. bytes_sent/bytes_acked have desynchronized; the sender is waiting for credit the host "
            "believes it already granted.");
    }
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
                // WHERE is it stuck? The phase word is live while the loop runs, so this says which part of
                // the kernel is blocking instead of leaving it to inference. POLL = sweep body (NoC reads or
                // the control-vector pass), RESERVE = credit wait (should be impossible now it is bounded),
                // WRITE = the PCIe write / push / notify / barrier, EXIT = the socket teardown tail.
                dump_drainer_state(ctx, d, "stop-not-acked");
            }
            // The drainer's own view of the run. Host-side page and marker counts cannot distinguish a
            // bandwidth wall from a latency one; sweeps/frames/cycles can.
            std::vector<uint32_t> res(48, 0);
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
                "max occ {}/{}, overflows {}, pace-gap {} cyc",
                res[4],
                res[20],
                res[6],
                res[9],
                dw,
                res[5],
                res[7],
                kernel_profiler::PROFILER_L1_VECTOR_SIZE,
                res[8],
                res[42]);
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] NoC check: runtime noc_index before Noc{{}} = {}, after = {}; "
                "compile-time NOC_INDEX = {}, read NoC = {}{}",
                res[36],
                res[37],
                res[38],
                res[39],
                res[37] != res[38] ? "  <<< MISMATCH: the default-arg write barrier watches the WRONG NoC"
                                   : "  (agree -- default-arg barrier is correct)");
            // A bounded credit wait means a wedged consumer costs FRAMES, not the workload. Never let that
            // trade happen quietly: without this line a dropped frame is indistinguishable from a clean run.
            if (res[33] != 0 || res[34] != 0) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] CREDIT WAIT TIMED OUT {}x -- dropped {} frames to keep the "
                    "workload running. The host consumer stopped acking (see the writer WALL TIMEOUT above); "
                    "capture for those frames is lost but the producers were never blocked.{}",
                    res[33],
                    res[34],
                    res[35] != 0 ? " EGRESS DECLARED DEAD: a write barrier expired, so the drainer stopped "
                                   "shipping and left its loop rather than reuse staging with writes still in "
                                   "flight."
                                 : "");
            }
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
            // proc sub-split. `proc` is the biggest busy-sweep phase, and it is two unrelated things:
            // a LOCAL scan of the staged control vectors, and a per-live-core 20 B NoC head write-back
            // (up to one issue per core per sweep). This drainer is issue-bound, so which half dominates
            // decides the optimization: batch/rate-limit the write-back, or tighten the per-RISC loops.
            const uint64_t c_ph_head = (static_cast<uint64_t>(res[41]) << 32) | res[40];
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] DRISC proc split: head-write-back {:.1f}% of busy ({:.1f}% of proc) | "
                "scan {:.1f}% of busy -- head-write-back is {} NoC issues/sweep, scan is local L1",
                pct(c_ph_head),
                c_proc ? (100.0 * static_cast<double>(c_ph_head) / static_cast<double>(c_proc)) : 0.0,
                pct(c_proc > c_ph_head ? c_proc - c_ph_head : 0),
                ctx.core_virt.size());
            // WORST-SWEEP breakdown. The knee is decided by the worst sweep, not the mean, and the worst
            // has been running ~2.5x the mean with no explanation. These are that one sweep's phases.
            {
                const double ws_tot = static_cast<double>(res[43] + res[44] + res[45] + res[46] + res[47]);
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] WORST sweep {:.1f} us = read {:.1f} + proc {:.1f} + credit-wait "
                    "{:.1f} + write {:.1f} + wr-barrier {:.1f} us (accounted {:.1f} us, {:.0f}%)",
                    res[25] / kCycPerUs,
                    res[43] / kCycPerUs,
                    res[44] / kCycPerUs,
                    res[45] / kCycPerUs,
                    res[46] / kCycPerUs,
                    res[47] / kCycPerUs,
                    ws_tot / kCycPerUs,
                    res[25] ? 100.0 * ws_tot / static_cast<double>(res[25]) : 0.0);
            }
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
    // After the decode threads are joined, so every lane's first marker has been seen.
    report_lane_spread();
    log_info(tt::LogMetal, "[perf-debug profiler] decode queues: max depth {} buffers", q_depth);
    writer_done_.store(true, std::memory_order_release);
    // Why the host is the wall. The egress-only benchmark moved bytes with a copy and no interpretation;
    // this thread also DECODES every marker and publishes every record, on the same thread that issues the
    // socket acks -- so the ack rate, and hence the sender's credit wait, is gated by per-marker work.
    if (w_reads_ != 0) {
        // w_read_ns_ holds TSC TICKS (the fine per-read timers moved to rdtsc; steady_clock cost ~650 ns a
        // call, which swamped events of a few us). decode/publish are still steady_clock nanoseconds.
        // Mixing them silently reported this same copy as "56.0 ms" here and "12.4 ms" in the split below.
        const double read_ms = w_read_ns_ * tsc_ns_per_tick() / 1e6;
        const double dec_ms = w_decode_ns_ / 1e6, pub_ms = w_publish_ns_ / 1e6;
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
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] host writer sock-read split: copy {:.1f} ms ({:.2f} GB/s, {:.1f} ns/KB) | "
            "ack {:.1f} ms ({:.0f} ns/read) | predrain {:.1f} ms ({:.0f} ns/read) | resize {:.1f} ms",
            w_read_ns_ * tsc_ns_per_tick() / 1e6,
            w_read_ns_ ? (static_cast<double>(w_bytes_) / (w_read_ns_ * tsc_ns_per_tick())) : 0.0,
            w_bytes_ ? (static_cast<double>(w_read_ns_) * tsc_ns_per_tick() * 1024.0 / static_cast<double>(w_bytes_))
                     : 0.0,
            w_ack_ns_ * tsc_ns_per_tick() / 1e6,
            w_reads_ ? (static_cast<double>(w_ack_ns_) * tsc_ns_per_tick() / w_reads_) : 0.0,
            w_predrain_ns_ * tsc_ns_per_tick() / 1e6,
            w_reads_ ? (static_cast<double>(w_predrain_ns_) * tsc_ns_per_tick() / w_reads_) : 0.0,
            w_resize_ns_ * tsc_ns_per_tick() / 1e6);
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] writer handoff: pool-acquire {:.1f} ms ({:.0f} ns/read) | enqueue+notify "
            "{:.1f} ms ({:.0f} ns/read) -- both take the SAME mutex the decoder holds while dequeuing",
            w_pool_ns_ * tsc_ns_per_tick() / 1e6,
            w_reads_ ? (static_cast<double>(w_pool_ns_) * tsc_ns_per_tick() / w_reads_) : 0.0,
            w_enq_ns_ * tsc_ns_per_tick() / 1e6,
            w_reads_ ? (static_cast<double>(w_enq_ns_) * tsc_ns_per_tick() / w_reads_) : 0.0);
        // TWO THREADS, TWO BUDGETS. decode+publish run on the DECODER thread, not the writer -- charging
        // them against the writer's wall produced "134% busy" with negative idle. And the per-read timers
        // are TSC TICKS while poll/decode/publish are steady_clock ns; mixing them inflated sock-read by
        // the ~4.5x tick ratio (4.4 ms of copy reported as 41.5% of a 48.1 ms wall).
        const double wall_ms = w_wall_ns_ / 1e6;
        const double tick_ms = tsc_ns_per_tick() / 1e6;
        const double sock_ms = (w_read_ns_ + w_ack_ns_ + w_resize_ns_) * tick_ms;  // the writer's real work
        const double dec_only_ms = w_decode_ns_ / 1e6;
        const double pub_only_ms = w_publish_ns_ / 1e6;
        const double writer_work_ms = sock_ms + (w_poll_ns_ / 1e6);
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] WRITER thread wall {:.1f} ms: poll {:.1f}% ({} polls) | sock-read {:.1f}% "
            "(copy+ack+resize) | idle {:.1f}% -- {:.0f}% busy",
            wall_ms,
            wall_ms > 0 ? 100.0 * (w_poll_ns_ / 1e6) / wall_ms : 0.0,
            w_polls_,
            wall_ms > 0 ? 100.0 * sock_ms / wall_ms : 0.0,
            wall_ms > 0 ? 100.0 * (wall_ms - writer_work_ms) / wall_ms : 0.0,
            wall_ms > 0 ? 100.0 * writer_work_ms / wall_ms : 0.0);
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] DECODER thread: decode {:.1f} ms + publish {:.1f} ms = {:.1f} ms vs writer "
            "wall {:.1f} ms -- {:.0f}% of the writer's wall; queue depth is the tell if it lags",
            dec_only_ms,
            pub_only_ms,
            dec_only_ms + pub_only_ms,
            wall_ms,
            wall_ms > 0 ? 100.0 * (dec_only_ms + pub_only_ms) / wall_ms : 0.0);
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
    tracy_.reset();
    devices_.clear();
}


}  // namespace tt::tt_metal
