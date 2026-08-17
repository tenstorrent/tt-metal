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
// avoid the codes spsc_packet.h actually uses (0,1,2,5..11); 31 is the top of the 5-bit type field.
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
// It IS just a flag flip, and the question it was built to ask is already answered -- see FINDINGS §N+12:
// egress on NoC 1 hangs identically to NoC 0 (16.0 vs 16.2 GB/s, load matched within 1%, both at run 16), so
// the route from the DRAM endpoint to the PCIe tile is dead as an explanation.
//
// This comment previously claimed the flip needed a mirrored PCIe encoding, because NoC 1 mirrors coordinates
// (NOC_0_X_PHYS_COORD(noc, size_x, x) = noc == 0 ? x : size_x - 1 - x). That was WRONG: the macro mirrors
// WORKER coords, while the PCIe tile lives in TRANSLATED space (the kernel is built with PCIE_NOC_X=19,
// PCIE_NOC_Y=24 -- both outside the 17x12 NOC0 grid), so the socket's NOC0-derived pcie_xy_enc is correct on
// BOTH NoCs. Measured: with the mirrored override, 0 markers decode from 2.37M pages; without it, 5,501,058
// decode. The override survives only behind TT_METAL_PERF_DEBUG_NOC_MIRROR=1 to keep the dead end documented.
// Watch for its signature: pages flow while zero markers decode, because socket credits advance on a
// different path than the payload writes, so a wrong payload destination looks like healthy throughput.
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

// ---- ROLE SPLIT knobs (see PerfDebugProfiler::kMaxDrisc in the header for the why) --------------------
//
// TT_METAL_PERF_DEBUG_ROLE_SPLIT=1 runs SIX DRISCs -- 4 fillers (a quarter of the worker grid each -> its own
// device DRAM ring) and 2 movers (TWO DRAM rings each -> the existing D2H socket) -- instead of 2 drainers
// each doing the whole job. Unset or 0 is today's path, bit for bit: every role-split compile arg is then 0
// and the kernel's `if constexpr` discards all of it.
// Shared truthiness for the perf-debug knobs. Tests the WHOLE value, not just the first character: the old
// `*s != '0'` idiom made `=false`, `=off` and `=no` all evaluate to ENABLED (any word not starting with '0'),
// while `=01` and `=0x1` evaluated to DISABLED. That is backwards for knobs whose purpose is keeping an
// instrument out of a production run, so falsy words are honoured and numbers decide by value.
bool env_flag(const char* name) {
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') {
        return false;
    }
    std::string v(s);
    for (char& c : v) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (v == "false" || v == "off" || v == "no" || v == "n" || v == "disable" || v == "disabled") {
        return false;
    }
    char* end = nullptr;
    const long n = std::strtol(v.c_str(), &end, 0);
    if (end != nullptr && *end == '\0') {
        return n != 0;  // "0"/"00"/"0x0" off; "01" on
    }
    return true;
}

// DEFAULT ON. The 6-DRISC split is the production shape -- it moved the knee from delay 60 to 15 (§N+40) --
// and the 2-drainer fallback is well past its own knee at the delays we care about: measured 22,761 producer
// stalls across 120 of 120 cores at delay 15 with the split off against 0 with it on. Both are lossless, so
// the old default cost workload perturbation rather than correctness. Set a falsy value to force the
// 2-drainer path; that is a bisect tool, not a production choice.
bool role_split() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ROLE_SPLIT");
        return (s == nullptr || *s == '\0') ? true : env_flag("TT_METAL_PERF_DEBUG_ROLE_SPLIT");
    }();
    return v;
}

// Per-filler DRAM ring size, in MiB. The whole reason to stage in DRAM is that this number is not capped by
// the TLB window budget the way the 12 MiB host FIFO is, so make it large enough that a host hiccup cannot
// reach the producers: 64 MiB is ~6,300 frames, roughly 115 busy sweeps of slack against the host FIFO's 21.
//
// This now sizes the HAL's DRAM PROFILER region too (perf_debug_dram_region_bytes_per_risc above), so it is
// read before any device is opened. Lowering it lowers DRAM held per bank one-for-one; 12 MiB is enough for a
// 5,000-zone/RISC capture and 64 MiB buys ~16-17k zones/RISC of runway (FINDINGS §N+39).
uint32_t role_ring_mb() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ROLE_RING_MB");
        const uint32_t n = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 64u;
        return n == 0 ? 64u : n;
    }();
    return v;
}

// DRAM banks (DRAM VIEW ids) the FILLERS occupy. Default 5, 6, 4, 1 = NoC cores 9-9, 9-5, 9-2, 0-3 -- all
// y != 0, which is legal for this role only (see the header). Two 25-run blocks on bank 5 -- N+29's WORST
// core, 5/25 for a full-job drainer -- gave 0/25 for stream-mode-held and 0/25 for filler-only duty.
//
// Views 4 and 1 were added for fillers 2 and 3. NOT view 7: the N+29 sweep records view 7's unused port as
// NoC core 0-0, the SAME core view 0 resolves to, and view 0 hosts mover 0. Two resident kernels on one core
// is not a subtle failure, but nothing in pick_unused_dram_logical_core() would have stopped it, so the
// duplicate-core TT_FATAL in boot_device now checks it explicitly for every roster.
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
            out = {5u, 6u, 4u, 1u};
        }
        return out;
    }();
    return v;
}

// ALLOCATOR bank ids the RINGS live in. Rings 0 and 1 stay on banks 1 and 2, exactly where they were measured;
// rings 2 and 3 take banks 4 and 5.
//
// The old invariant -- a ring shares a DRAM channel with NO drainer -- is now UNREACHABLE and has been
// deliberately relaxed: 6 drainer channels plus 4 rings is 10 against 7 allocator banks. What is kept is the
// part with evidence behind it: a ring is never placed on a MOVER bank (0, 3), because host-facing duty is
// where the N+29 hazard was measured. Ring traffic terminates at the channel's PREFERRED WORKER endpoint while
// a drainer sits on that channel's unused subchannel, so even a shared channel means different cores and
// different NIUs -- and the per-ring load (~1.4 GB/s written, ~1.4 GB/s read) is a rounding error against a
// GDDR channel. Measured with the overlap in place: 0 ring-room waits, 0 hs_bad, staged == moved on all four
// rings, and the knee moved the right way. Overridable with TT_METAL_PERF_DEBUG_ROLE_RING_BANKS.
const std::vector<uint32_t>& role_ring_banks() {
    static const std::vector<uint32_t> v = [] {
        std::vector<uint32_t> out;
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ROLE_RING_BANKS");
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
            out = {1u, 2u, 4u, 5u};
        }
        return out;
    }();
    return v;
}

constexpr uint32_t kRoleFull = 0, kRoleFiller = 1, kRoleMover = 2;
// Must match drisc_profiler_drain.cpp's kProbeMoverMagic and handshake offsets.
constexpr uint32_t kProbeFillerMagic = 0xF11E5A17u;
constexpr uint32_t kProbeMoverMagic = 0x5A0FE1EDu;
constexpr uint32_t kHsHead = 0, kHsTail = 16, kHsProbeF = 32, kHsProbeM = 48, kHsBytes = 64;

// ---- DRISC SELF-PROFILING knobs -----------------------------------------------------------------------
//
// TT_METAL_PERF_DEBUG_DRISC_ZONES=1 makes every drainer emit its OWN device zones, framed as worker spans and
// pushed down the path it already owns, so they land in Tracy on their own per-core row beside the workers.
// Unset or 0 is today's path bit for bit: every compile arg below is then 0 and the kernel's `if constexpr`
// discards all of it, including the staging slot it would otherwise reserve.
//
// This TRACES rather than samples. Sampling was built first and rejected on use: at ~4.5% of busy sweeps a
// drainer's row is a few disconnected zones whose gaps read as idle time when they are really uncaptured
// sweeps. Every sweep inside an active window is now instrumented instead.
bool drisc_zones() {
    static const bool v = [] {
        // NOC_FOOTPRINT IMPLIES ZONES. The per-sweep NoC series rides the self-zone marker stream, so
        // footprint-without-zones yields the out[] totals and the log block but NO plots -- a silently
        // half-working configuration. Asking for the footprint therefore turns self-profiling on too.
        // An explicit falsy DRISC_ZONES still wins, so the combination remains expressible.
        const char* z = std::getenv("TT_METAL_PERF_DEBUG_DRISC_ZONES");
        if ((z == nullptr || *z == '\0') && env_flag("TT_METAL_PERF_DEBUG_NOC_FOOTPRINT")) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] NOC_FOOTPRINT implies DRISC_ZONES (the per-sweep NoC series rides the "
                "self-zone stream) -- enabling DRISC self-profiling");
            return true;
        }
        return env_flag("TT_METAL_PERF_DEBUG_DRISC_ZONES");
    }();
    return v;
}

// How long, in MICROSECONDS of device wall clock, the capture window stays open past the last work seen. This
// is what makes coverage contiguous across a burst instead of restarting per busy sweep, and what keeps the
// drainer's ~99% idle residency (hundreds of ms to seconds, against a ~2 ms workload) out of the trace.
// In cycles on the device; a filler idles at 8.5 us + a 12.7 us pacing gap, a mover at 0.7 us, so a hold
// expressed in SWEEPS would mean two completely different durations on the two roles.
uint32_t drisc_zone_hold_cycles() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_DRISC_ZONE_HOLD_US");
        const uint32_t us = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 500u;
        return (us == 0 ? 500u : us) * 1350u;  // the drainer stamps with the 1.35 GHz Tensix wall clock
    }();
    return v;
}

// TT_METAL_PERF_DEBUG_NOC_FOOTPRINT=1 makes each drainer sample its OWN NIU master counters once per sweep
// (8 local MMIO loads, no NoC traffic of its own) and report the NoC bytes and transactions it actually issued,
// on both NoCs. This exists because every footprint figure we have is ARITHMETIC over frame counts; this is the
// hardware's own tally. Off by default: the per-sweep sample is not free on a drainer that is >99% idle, and
// N+41 measured that a mere +4% on idle-sweep cost crosses the knee at SD delay 15.
uint32_t noc_footprint() {
    static const uint32_t v = [] { return env_flag("TT_METAL_PERF_DEBUG_NOC_FOOTPRINT") ? 1u : 0u; }();
    return v;
}

// DETAIL LEVEL. 0 (default) = DRISC-SWEEP + DRISC-PACE only: the two depth-0 zones that account for a
// drainer's whole cadence with no unexplained whitespace, at ~4 markers per sweep. 1 = also the per-batch
// child phases (read / read-wait / proc / credit-wait / write / wr-barrier), ~100 markers per sweep.
uint32_t drisc_zone_detail() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_DRISC_ZONE_DETAIL");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 0u;
    }();
    return v;
}

// Frame budget per DRISC. With tracing this is a COVERAGE limit rather than a safety net: frames scale with
// how long the drainer stays busy. At detail 0 a frame holds ~63 sweeps, so 256 frames is ~16,000 sweeps --
// far more than a workload window needs. At detail 1 a frame holds ~2.5 sweeps, so the same budget is ~640.
uint32_t drisc_zone_frames() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_DRISC_ZONE_FRAMES");
        const uint32_t n = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 256u;
        return n == 0 ? 256u : n;
    }();
    return v;
}

// ---- COMMON-TRIGGER SYNC EVENT ----------------------------------------------------------------------------
//
// TT_METAL_PERF_DEBUG_SYNC_EVENT = how many triggers to fire (0 = off, the default). Each one releases every
// drainer from a rendezvous barrier at the same instant, so the spread in the resulting DRISC-SYNC zones'
// RENDERED timestamps is anchor + render error with no genuine timing difference in it.
//
// FIRE MORE THAN ONE, and space them: a frequency error is a RATE error, so it shows up as the residual
// DRIFTING across triggers rather than as a constant offset. N triggers G ms apart turn that slope into
// something directly readable -- at ~40 ppm, 1 s of separation is ~40 us of drift, far above the floor.
uint32_t sync_event_count() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_SYNC_EVENT");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 0u;
    }();
    return v;
}

// Gap between triggers, ms. Default 250: five triggers then span 1 s, which is the same order as the
// anchor-to-workload distance the rate error is multiplied by.
uint32_t sync_event_gap_ms() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_SYNC_EVENT_GAP_MS");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 250u;
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

// Declared in the header, so it lives OUTSIDE the anonymous namespace above (external linkage) while still
// seeing role_split()/role_ring_mb() from it.
//
// The HAL sizes its per-bank DRAM PROFILER region as
// `per_risc_bytes * MaxProcessorsPerCoreType * CEIL_NUM_CORES_PER_DRAM_CHANNEL` (5 * 20 = 100 on Blackhole and
// Wormhole), so the per-risc figure that yields R bytes per bank is R / 100.
//
// Nothing downstream assumes this landed on exactly ROLE_RING_MB: those multipliers live in the arch HALs and
// are not visible here, so if an arch differs the region comes out a different size and the ring adapts to
// whatever was actually reserved (frames = region_bytes / slot_bytes).
uint32_t perf_debug_dram_region_bytes_per_risc() {
    if (!role_split()) {
        return 0;
    }
    const uint64_t want_bytes = static_cast<uint64_t>(role_ring_mb()) * 1024ull * 1024ull;
    return static_cast<uint32_t>((want_bytes + 99) / 100);
}

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
        const double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
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
    // calibrate()". THERE IS NO SUCH FUNCTION -- `calibrate` appears nowhere in drisc_profiler_drain.cpp, in
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
            // LONG BASELINE, deliberately: 100 samples x 500 us spans ~50 ms instead of ~360 us, cutting the
            // fitted-frequency error by the baseline ratio (~140x). This is the ONE frequency every context on
            // this chip will use (see below), so it is worth 50 ms of a 9-12 s device open to measure it well.
            sync = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/500);
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
        //
        // The DRAINER cores are the exception, and pre-created: with self-profiling on they are GUARANTEED to
        // emit zones, so none of them can turn into a dead context, and there are only a handful.
        if (auto it = self_zone_cores_.find(ctx.chip_id); it != self_zone_cores_.end() && !it->second.empty()) {
            tracy_->PreCreateContexts(ctx.chip_id, it->second);
        }
        // Give each drainer's rows its ROLE, so a plot reads "DRISC 9-9 FILLER" instead of coordinates the
        // reader has to map back to a job by hand. String literals: the role text is interned into a plot
        // name whose pointer the SERVER dereferences, so it has to outlive everything.
        if (tracy_ != nullptr) {
            for (uint32_t d = 0; d < kMaxDrisc; d++) {
                if (ctx.drain_program[d] == nullptr) {
                    continue;
                }
                const char* role = ctx.role[d] == kRoleFiller  ? "FILLER"
                                   : ctx.role[d] == kRoleMover ? "MOVER"
                                                               : "DRAINER";
                // TRANSLATE: the handler keys on NOC0 coords (that is what a decoded event carries), while
                // drisc_virtual is the VIRTUAL space. Registering the virtual pair would look up nothing and
                // the label would silently fall back to bare coordinates -- the failure would have been an
                // absent word, not an error. virt_to_noc0 exists for exactly this reason.
                const auto nit = ctx.virt_to_noc0.find(
                    (static_cast<uint64_t>(ctx.drisc_virtual[d].x) << 32) |
                    static_cast<uint64_t>(ctx.drisc_virtual[d].y));
                if (nit == ctx.virt_to_noc0.end()) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] DRISC {} at virtual ({},{}) has no NOC0 mapping -- its plot rows "
                        "will be labelled by coordinate only, without {}",
                        d,
                        ctx.drisc_virtual[d].x,
                        ctx.drisc_virtual[d].y,
                        role);
                    continue;
                }
                tracy_->SetDriscRole(ctx.chip_id, nit->second.first, nit->second.second, role);
            }
        }
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
        pub_last_ts_.assign(static_cast<size_t>(kNSockets) * devices_.front().nl, 0);
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
        const size_t kPrefillBufs =
            (cap && cap < 512) ? 1536 : 320;  // small caps => far more reads in flight  // > the ~253 max queue depth
                                              // observed, so the pool stays warm
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
        auto rt_core = MetalContext::instance(context_id)
                           .get_dispatch_core_manager()
                           .get_reserved_realtime_profiler_core(device_id);
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
    // ---- ROLE SPLIT: decide the roster before anything is flipped or launched ----
    //
    // Roles, banks and socket ownership are all resolved up front because the NIU pre-pass below needs every
    // drainer's core in ONE launch (see its comment) and because a mover's compile args reference its
    // filler's L1, so the fillers must be set up first. The default path takes the `else` and is unchanged.
    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    // DEGRADE, don't fail, when the part cannot host the full roster. The 6-DRISC split needs kNFillers filler
    // banks plus kNSockets host-facing banks, and the fillers' RINGS need banks too; a harvested or smaller
    // part may not have them. Nothing downstream checked this -- the roster was fixed before
    // pick_unused_dram_logical_core() was ever called, so a short part would have indexed past the end of
    // kSafeBanks / the filler-bank list rather than reporting anything.
    //
    // Ladder: 6 (4 fillers + 2 movers) -> 2 full-role drainers -> 1. Each step is a configuration already
    // measured to work, just with a worse knee (§N+34: 1 drainer knee 100, 2 -> 20; §N+40: the split -> 15).
    // Losslessness never depends on the count: fewer drainers means producers stall sooner, not that markers
    // are dropped.
    const uint32_t need_split = kNFillers + kNSockets;
    bool rsplit = role_split() && !tensix_drain;
    if (rsplit && nbanks < need_split) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] role split needs {} DRAM views (4 fillers + 2 movers) but this part has {} "
            "-- falling back to {} full-role drainer(s). Capture stays LOSSLESS; the knee moves in (§N+34).",
            need_split,
            nbanks,
            std::min<uint32_t>(kNSockets, nbanks));
        rsplit = false;
    }
    // Second rung: even the 2-drainer path needs 2 host-facing banks. With one, run a single drainer over the
    // whole grid -- the original shape, knee ~100 (§N+34) but still complete.
    const uint32_t n_full = std::min<uint32_t>(kNSockets, nbanks == 0 ? 1u : nbanks);
    if (!rsplit && n_full < kNSockets) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] only {} DRAM view(s) available: running {} drainer(s) over the whole grid.",
            nbanks,
            n_full);
    }
    std::vector<uint32_t> banks;     // DRAM bank hosting DRISC d itself
    std::vector<uint32_t> ringbank;  // DRAM bank holding the ring DRISC d reads/writes (0 when unused)
    if (rsplit) {
        ctx.n_drisc = kNFillers + kNSockets;
        const auto& fb = role_filler_banks();
        const auto& rb = role_ring_banks();
        TT_FATAL(
            fb.size() >= kNFillers && rb.size() >= kNFillers,
            "perf-debug role split needs {} filler banks and {} ring banks (got {} and {})",
            kNFillers,
            kNFillers,
            fb.size(),
            rb.size());
        for (uint32_t f = 0; f < kNFillers; f++) {
            ctx.role[f] = kRoleFiller;
            ctx.sock_of[f] = kNoSocket;
            ctx.n_peer[f] = 0;
            banks.push_back(fb[f]);
            ringbank.push_back(rb[f]);
        }
        // Mover m drains fillers m, m + kNSockets, ... -- so at 4 fillers, mover 0 takes fillers 0 and 2 and
        // mover 1 takes 1 and 3. STRIDED rather than adjacent on purpose: the fillers own contiguous quarters
        // of the grid in index order, so striding gives each socket one low-half slice and one high-half
        // slice. Adjacent pairing would put both halves of the grid's busy end on one socket if the workload
        // is not uniform across the grid.
        for (uint32_t m = 0; m < kNSockets; m++) {
            const uint32_t d = kNFillers + m;
            ctx.role[d] = kRoleMover;
            ctx.sock_of[d] = m;
            ctx.n_peer[d] = kNFillers / kNSockets;
            for (uint32_t p = 0; p < ctx.n_peer[d]; p++) {
                ctx.peer_of[d][p] = m + p * kNSockets;
            }
            banks.push_back(kSafeBanks[m]);
            // A mover owns no ring of its own -- it reads its PEERS' rings, and their banks/addresses reach it
            // as compile args taken from those peers' entries. Recorded as peer 0's only for the log line.
            ringbank.push_back(rb[ctx.peer_of[d][0]]);
        }
    } else {
        ctx.n_drisc = n_full;
        const int bank_ov_pre = drisc_bank_override();
        for (uint32_t d = 0; d < n_full; d++) {
            ctx.role[d] = kRoleFull;
            ctx.sock_of[d] = d;
            banks.push_back(
                (bank_ov_pre >= 0) ? static_cast<uint32_t>((static_cast<uint32_t>(bank_ov_pre) + d) % nbanks)
                                   : kSafeBanks[d]);
            ringbank.push_back(0);
        }
    }

    // ---- DRISC SELF-PROFILING: give each drainer a LANE BLOCK of its own ----------------------------------
    //
    // A drainer's self frame is an ordinary span frame, so the host resolves it exactly like a worker's: by
    // looking up SPSC_CORE_XY in core_of_xy and turning the answer into lane = core * NRISC + risc. That means
    // the drainer cores need core indices, and the lane space has to be wide enough to hold them. Appending
    // them after the worker grid keeps every worker lane id EXACTLY where it was, which matters because
    // pub_last_ts_ / first_ts_ / con_last_ts_ are all indexed by it.
    //
    // Done HERE rather than in the per-DRISC loop below because ctx.nl is consumed by SpscDecodeState::reset()
    // inside that loop: a drainer brought up before another would otherwise get a decoder sized too small and
    // silently drop the later drainer's frames (lane >= nl is a `continue`).
    const uint32_t self_core0 = static_cast<uint32_t>(num_cores);
    const bool self_zones_on = drisc_zones() && drisc_zone_frames() != 0;
    if (self_zones_on) {
        ctx.n_worker_cores = static_cast<uint32_t>(num_cores);
        ctx.nl = (static_cast<uint32_t>(num_cores) + ctx.n_drisc) * kNRisc;
        ctx.core_virt.resize(static_cast<size_t>(num_cores) + ctx.n_drisc);
        if (first_ts_.size() < ctx.nl) {
            first_ts_.assign(ctx.nl, 0);
        }
    }

    for (uint32_t d = 0; d < ctx.n_drisc; d++) {
        ctx.dram_bank[d] = ringbank[d];
    }

    const uint32_t span_bytes_all =
        (kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE + kNRisc * kernel_profiler::PROFILER_L1_VECTOR_SIZE) *
        sizeof(uint32_t);
    const uint32_t slot_bytes_all = kernel_profiler::SPSC_SPAN_PREFIX_WORDS * sizeof(uint32_t) + span_bytes_all;
    uint32_t nstage_report = 0;  // last drainer's mapped staging-slot count, for the self-profiling log line

    // ---- REUSE the old profiler's DRAM region; do NOT allocate a second buffer ----
    //
    // The HAL reserves a PROFILER region at the same bank-relative offset in EVERY DRAM bank (just past the
    // barrier word, with DRAM_UNRESERVED starting above it). That region belongs to the push-to-DRAM profiler
    // backend, which is never the active one when we are: the streaming backend keeps markers in an L1 ring and
    // never writes DRAM. So it is dead space, and staging frames there costs nothing extra.
    //
    // What this replaces: an interleaved MeshBuffer with page_size == ring_bytes. That reservation was
    // LOCK-STEP -- the allocator holds the same offset in every bank -- so a 64 MiB ring cost 447 MiB to
    // address 127 MiB, on top of the ~32 MiB HAL region we were leaving idle. Reusing the region makes the
    // ring's size and the region's size ONE knob (perf_debug_dram_region_bytes_per_risc), and the ring becomes
    // available in every bank at a single address rather than only in banks we allocated pages for.
    //
    // Frames are still whole: capacity truncates to a multiple of the 165-page frame, so a FRAME never
    // straddles the wrap (a RUN of frames still can, and stage_run splits it).
    if (rsplit) {
        try {
            const auto& hal = MetalContext::instance().hal();
            const uint32_t region_bytes = hal.get_dev_size(HalDramMemAddrType::PROFILER);
            const uint32_t region_addr = static_cast<uint32_t>(hal.get_dev_addr(HalDramMemAddrType::PROFILER));
            ctx.dram_frames = region_bytes / slot_bytes_all;
            TT_FATAL(
                ctx.dram_frames >= 64,
                "perf-debug role split: the DRAM profiler region holds {} frames ({} B / {} B per frame), need at "
                "least 64. Raise TT_METAL_PERF_DEBUG_ROLE_RING_MB (it sizes this region) or "
                "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT.",
                ctx.dram_frames,
                region_bytes,
                slot_bytes_all);
            const uint32_t ring_bytes = ctx.dram_frames * slot_bytes_all;

            // TWO DIFFERENT BANK SPACES, and mixing them up would silently mis-address every frame.
            // A DRISC's own placement is indexed by DRAM VIEW (soc.get_num_dram_views()) because that is what
            // pick_unused_dram_logical_core takes. A ring's bank is an ALLOCATOR bank id, because that is what
            // the kernel's get_noc_addr_from_bank_id indexes (NUM_DRAM_BANKS=7 in the JIT defines). They are
            // not the same count, so validate the ring banks against the allocator.
            const uint32_t alloc_banks = ctx.device->allocator()->get_num_banks(BufferType::DRAM);
            for (uint32_t f = 0; f < kNFillers; f++) {
                TT_FATAL(
                    ringbank[f] < alloc_banks,
                    "perf-debug role split: ring bank {} does not exist (the allocator has {} DRAM banks)",
                    ringbank[f],
                    alloc_banks);
            }

            // The HAL address is CHANNEL-relative, but the kernel reaches the ring through
            // get_noc_addr_from_bank_id, which adds bank_to_dram_offset[bank] on its own. Subtract the host's
            // view of that offset so the kernel's addition lands back on the HAL address.
            //
            // PER RING, not one shared value. It used to be one compile arg with a TT_FATAL demanding every
            // ring bank have the SAME offset -- fine at two rings on a part where every offset is 0, but at
            // four rings that FATAL would kill capture on any part where they differ, for no reason: each
            // filler already carries its own (bank, addr) pair, and a mover gets its peers' pairs explicitly.
            for (uint32_t f = 0; f < kNFillers; f++) {
                const int32_t off = ctx.device->allocator()->get_bank_offset(BufferType::DRAM, ringbank[f]);
                ctx.dram_addr[f] = static_cast<uint32_t>(static_cast<int64_t>(region_addr) - off);
            }
            // Movers address their peers' rings, so copy the peer-0 pair over for the log line only.
            for (uint32_t d = kNFillers; d < ctx.n_drisc; d++) {
                ctx.dram_addr[d] = ctx.dram_addr[ctx.peer_of[d][0]];
            }

            std::string ring_desc;
            for (uint32_t f = 0; f < kNFillers; f++) {
                ring_desc += fmt::format(
                    "{}filler {} -> bank {} @ bank-relative 0x{:x}",
                    f == 0 ? "" : ", ",
                    f,
                    ringbank[f],
                    ctx.dram_addr[f]);
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] role split: {} DRAM rings of {} frames ({:.1f} MiB each) REUSING the "
                "profiler DRAM region at channel-relative 0x{:x} -- {}; of {} allocator banks. The region "
                "reserves {:.1f} MiB per bank in EVERY bank = {} MiB total, of which {} MiB carries a ring -- "
                "the other {} MiB is region in banks no ring uses, so more rings are FREE",
                kNFillers,
                ctx.dram_frames,
                ring_bytes / (1024.0 * 1024.0),
                region_addr,
                ring_desc,
                alloc_banks,
                region_bytes / (1024.0 * 1024.0),
                (static_cast<uint64_t>(region_bytes) * alloc_banks) / (1024 * 1024),
                (static_cast<uint64_t>(region_bytes) * kNFillers) / (1024 * 1024),
                (static_cast<uint64_t>(region_bytes) * (alloc_banks - kNFillers)) / (1024 * 1024));
        } catch (const std::exception& e) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] role split: could not allocate the DRAM staging rings ({}); capture is "
                "disabled for this run rather than launching fillers with nowhere to write",
                e.what());
            disarm_producers(mesh_device, device_id);
            return false;
        }
    }

    if (!tensix_drain) {
        std::vector<CoreCoord> flip_cores;
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            flip_cores.push_back(mesh_device->impl().pick_unused_dram_logical_core(banks[d]));
        }
        // TWO DRISCs MUST NEVER LAND ON THE SAME CORE. pick_unused_dram_logical_core() takes a DRAM VIEW and
        // reserves that view's worker/eth endpoints -- it has no idea another view may resolve to the SAME
        // physical port. The N+29 sweep records exactly that: view 0 and view 7 both come back as NoC core
        // 0-0. At two DRISCs the roster was hardcoded to {0, 3} and it could not happen; with six banks in
        // play (and a filler-bank env override) it can, and the result would be two resident kernels sharing
        // one core's L1 -- staging, socket config, results, handshake, all overlapped, with no counter that
        // would notice. Refuse to launch instead.
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
        set_drisc_niu_mode(ctx.device, flip_cores, 1);
    }

    for (uint32_t d = 0; d < ctx.n_drisc; d++) {
        // A MOVER has no slice of the worker grid -- it never touches a worker core. Fillers (and the default
        // full-job drainers) take the same contiguous halves as before, so the drained set is unchanged.
        const bool is_mover = ctx.role[d] == kRoleMover;
        const bool is_filler = ctx.role[d] == kRoleFiller;
        // How many DRISCs SWEEP THE GRID -- which is what the slices divide by, and it is no longer the same
        // as kNSockets. Off: 2 full-job drainers take halves. On: kNFillers fillers take kNFillers-ths, which
        // is the entire point of the change (the knee is the filler's scan over its slice, FINDINGS N+28), so
        // getting this denominator wrong would look like a working build that simply did not improve.
        const uint32_t n_slices = rsplit ? kNFillers : kNSockets;
        const uint32_t slice = is_mover ? 0u : d;
        const uint32_t lo = is_mover ? 0u : static_cast<uint32_t>((num_cores * slice) / n_slices);
        const uint32_t hi = is_mover ? 0u : static_cast<uint32_t>((num_cores * (slice + 1)) / n_slices);
        const uint32_t my_cores = hi - lo;
        if (my_cores == 0 && !is_mover) {
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
                TT_FATAL(d < gy, "drainer {} does not fit the reserved column (only {} rows)", d, gy);
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
            // ROLE-AWARE. The y==0 restriction is a property of the HOST-FACING duty, not of running a
            // resident kernel on a DRAM core: on bank 5 -- N+29's worst core at 5/25 for a full-job drainer --
            // two 25-run blocks gave 0/25 held in stream mode and 0/25 doing filler-only duty. So a FILLER may
            // sit anywhere; anything that talks to the host must come from kSafeBanks.
            const uint32_t bank = banks[d];
            const int bank_ov = drisc_bank_override();
            const uint32_t host_facing_idx = is_mover ? ctx.sock_of[d] : d;
            TT_FATAL(
                is_filler || bank_ov >= 0 || host_facing_idx < kNumSafeBanks,
                "perf-debug: {} host-facing DRISCs requested but only {} DRAM banks are known safe (row y==0). "
                "Raising the host-facing count needs a bank safety sweep first -- see FINDINGS N+29.",
                host_facing_idx + 1,
                kNumSafeBanks);
            ctx.drisc_logical[d] = mesh_device->impl().pick_unused_dram_logical_core(bank);
            const CoreCoord translated =
                soc.dram_bank_endpoint_coords.at(ctx.drisc_logical[d].x).at(ctx.drisc_logical[d].y);
            const tt::umd::CoreCoord phys = soc.translate_coord_to(
                tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED),
                CoordSystem::NOC0);
            drisc_phys = CoreCoord{phys.x, phys.y};
            ctx.drisc_virtual[d] = ctx.device->virtual_core_from_logical_core(ctx.drisc_logical[d], CoreType::DRAM);
            // This drainer's own core, as a Tracy row: virtual coords are what its self frame carries in
            // SPSC_CORE_XY, NOC0 is what a Tracy context is keyed on. Registered for BOTH maps here, where both
            // are in hand -- a DRAM core is absent from metal_soc_descriptor's profiler flat-id map (TENSIX and
            // ETH only), so nothing else would ever give it a row.
            if (self_zones_on) {
                ctx.core_virt[self_core0 + d] = {
                    static_cast<uint32_t>(ctx.drisc_virtual[d].x), static_cast<uint32_t>(ctx.drisc_virtual[d].y)};
                ctx.virt_to_noc0
                    [(static_cast<uint64_t>(ctx.drisc_virtual[d].x) << 32) |
                     static_cast<uint64_t>(ctx.drisc_virtual[d].y)] = {
                    static_cast<uint32_t>(drisc_phys.x), static_cast<uint32_t>(drisc_phys.y)};
            }
            ctx.drisc_l1_base[d] = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
            ctx.drisc_l1_noc[d] = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
            region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        }

        const uint32_t slot_bytes = slot_bytes_all;
        constexpr uint32_t kCfgReserve = 8 * 1024;
        constexpr uint32_t kScratchBytes = 128 * 32;
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
        // to live. There is no room to ADD one: a DRAM core's UNRESERVED L1 is 86 KB and 7 slots of 10,560 B
        // plus the scratch/misc/socket-config reserve already leave under 2 KB spare. So hand the kernel
        // (nstage - 1) slots and let it use index kNStage -- one past its own array -- for the self frame. L1
        // does not grow, the OFF build is untouched, and the only behavioural cost is that a MOVER's largest
        // batch drops from 7 frames to 6 (a filler's batch is bounded by kGenSlots = 3 either way, so it is
        // unaffected). That cost is real and is measured in FINDINGS rather than waved off.
        // ROLE-AWARE self-frame budget. One shared cap starved the movers: they sample every ~1.3 us against a
        // filler's ~157 us, so on ResNet-50 the 256-frame default stopped both movers ~20% into an 844 ms run
        // and they NEVER came back -- the cap is permanent, not periodic, so nothing re-arms them. Measured
        // need for full coverage of that run: filler 107 frames, mover 1,813-1,936 (~18x).
        //
        // Cost is NOT symmetric either, and the log line states it per drainer: full mover coverage was
        // 13.0-14.9% of that mover's egress against 1.5% for a filler. So this buys completeness with real
        // bytes; if that is too dear the cheaper lever is decimating the MOVER's sample interval (1.3 us
        // resolution over 844 ms is far finer than any question needs), not shrinking the cap back.
        const uint32_t self_frames_base = drisc_zones() ? drisc_zone_frames() : 0u;
        const uint32_t nstage_drain = (self_frames_base != 0 && nstage >= 3) ? nstage - 1u : nstage;
        if (self_frames_base != 0 && nstage < 3) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] Device {}: only {} staging slots fit, too few to spare one for DRISC "
                "self-profiling (needs >= 3 so kGenSlots stays >= 1); self zones are OFF for this run",
                device_id,
                nstage);
        }
        const uint32_t self_slot = nstage_drain;  // must match the kernel's kSelfSlot == kNStage
        nstage_report = nstage;
        const uint32_t stage_base = ctx.drisc_l1_base[d];
        const uint32_t head_scratch = stage_base + nstage * slot_bytes;
        ctx.done_addr[d] = head_scratch + kScratchBytes;
        ctx.stop_addr[d] = ctx.done_addr[d] + 64;
        ctx.results_addr[d] = ctx.stop_addr[d] + 64;
        // The role-split handshake block. Allocated for every role so the L1 layout (and hence every other
        // address) is identical whether the knob is on or off -- a mover reads its FILLER's block, and that
        // only works because both cores lay their L1 out the same way.
        // Sized off the SHARED constant, not a literal: this moved once already (48 -> 64 words) and a
        // hand-copied 256 here would have silently overlapped the handshake.
        ctx.hs_addr[d] = ctx.results_addr[d] + kernel_profiler::SPSC_DRAIN_RESULT_WORDS * sizeof(uint32_t);
        TT_FATAL(
            self_frames_base == 0 || self_slot < nstage,
            "perf-debug: DRISC self-profiling wants staging slot {} but only {} slots are mapped",
            self_slot,
            nstage);
        const uint32_t cfg_l1 = ctx.drisc_l1_base[d] + region - kCfgReserve;
        TT_FATAL(ctx.hs_addr[d] + kHsBytes <= cfg_l1, "DRISC L1 layout overlaps the socket config");

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

        // Socket index this DRISC owns. A FILLER owns none: it never talks to the host, which is the whole
        // point of the split. NOTE these arrays are sized kNSockets, not kMaxDrisc -- indexing them by the
        // DRISC index would run off the end once there are four DRISCs.
        const uint32_t sk = ctx.sock_of[d];
        const bool has_socket = sk != kNoSocket;
        try {
            if (has_socket) {
                // sender_uses_physical_noc_addr switches the socket between "physical NoC coord + full L1 addr" (DRISC,
                // drainer) and the normal worker path (logical coord, worker-L1 semantics). The socket picks the
                // static-vs-dynamic write path by ASKING UMD whether this core has a window (see init_sender_tlb),
                // so the window configured just above is what puts the DRISC on the static path.
                g_bringup_step = fmt::format("drainer {}: D2HSocket construct (writes config into DRISC L1)", d);
                ctx.sockets[sk] = std::make_unique<distributed::D2HSocket>(
                    mesh_device,
                    distributed::MeshCoreCoord{
                        scoord, tensix_drain ? ctx.drisc_logical[d] : CoreCoord(drisc_phys.x, drisc_phys.y)},
                    static_cast<uint32_t>((static_cast<uint64_t>(kHRingWords) * 4 / kPageSize) * kPageSize),
                    distributed::D2HSocket::ExternalConfigBuffer{
                        .address = cfg_l1, .sender_uses_physical_noc_addr = !tensix_drain});
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
                // One decode stream per SOCKET, so this follows the socket and not the DRISC. A filler produces
                // no stream to decode: its frames reach the host through its mover's socket.
                ctx.decode[sk] = std::make_unique<pz::SpscDecodeState>();
                ctx.decode[sk]->reset(ctx.nl);
                for (uint32_t c = 0; c < num_cores; c++) {
                    ctx.decode[sk]->core_of_xy[coords[c]] = c;  // full map: lane ids stay global across drainers
                }
            }  // has_socket

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

            // done | hb | phase | dbg_hw | dbg_sw | then FOUR WORDS PER PEER (probe_f-echo | probe-frame |
            // live head | live tail) -- 13 words of the 64 B pad, because a stale probe echo from the previous
            // run would pass the bring-up check that exists to catch a bad handshake. Peer 1's block was added
            // with dual-ring movers; zeroing only peer 0's would leave exactly that hole.
            uint32_t zero3[13] = {};
            cluster.write_core(
                zero3,
                sizeof(zero3),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
            // ALSO the stop word -- teardown leaves it at 1 (quiesce) or 2 (free the NIU), and the drain loop
            // is `while (... && *stop == 0 ...)`, so a stale value would make the next kernel exit after ONE
            // sweep while the host reports FAILED TO START. (Not the cause of the slow-dispatch wedge below --
            // that reproduces with stop=0 -- but the same class of stale-state bug as the hb/phase words.)
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
            // Same reason, for the results block: it is published only on kernel exit, so a drainer that is
            // still running at teardown leaves the PREVIOUS run's numbers there and they read as this run's.
            // That is how a 42 s run reported "495.7 ms, credit-wait 0.1%" and hid its own credit timeouts.
            const std::vector<uint32_t> zero_res(kernel_profiler::SPSC_DRAIN_RESULT_WORDS, 0);
            cluster.write_core(
                zero_res.data(),
                (uint32_t)(zero_res.size() * sizeof(uint32_t)),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.results_addr[d] - ctx.drisc_l1_base[d]));
            // ---- the FILLER's handshake block, planted before its kernel or its mover exists ----
            //
            // Zero head/tail (the kernel re-zeros head; tail belongs to the mover from launch onward, so it
            // must start clean or the filler reads a bogus consumed-count and overwrites live frames), then
            // plant the magic the mover has to read back. The mover echoes it into its own L1 and writes its
            // own magic here, and bring-up refuses the run if either is wrong -- a mistaken peer coordinate
            // or L1 address otherwise yields a plausible garbage `head` and silently corrupt capture.
            //
            // The planted magic is kProbeFillerMagic + THIS FILLER'S INDEX. With one magic for all fillers the
            // echo only proved the mover read SOME filler's probe word, so a mover whose peer-1 coordinate
            // named the wrong filler would have passed -- and then two movers would drain one ring while
            // another was never drained, which back-pressures a lossless producer into wedging the workload.
            if (is_filler) {
                std::vector<uint32_t> hs(kHsBytes / sizeof(uint32_t), 0);
                hs[kHsProbeF / sizeof(uint32_t)] = kProbeFillerMagic + d;
                cluster.write_core(
                    hs.data(),
                    kHsBytes,
                    tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                    ctx.drisc_l1_noc[d] + (ctx.hs_addr[d] - ctx.drisc_l1_base[d]));
            }

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
                const uint32_t mx =
                    static_cast<uint32_t>(mmio_soc.grid_size.x) - 1 - static_cast<uint32_t>(pcie_noc0.x);
                const uint32_t my =
                    static_cast<uint32_t>(mmio_soc.grid_size.y) - 1 - static_cast<uint32_t>(pcie_noc0.y);
                pcie_enc_override = hal.noc_xy_pcie64_encoding(mx, my);
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] NoC 1 egress: PCIe tile NOC0 ({},{}) -> NOC1 ({},{}) on a {}x{} grid, "
                    "enc 0x{:x}",
                    pcie_noc0.x,
                    pcie_noc0.y,
                    mx,
                    my,
                    mmio_soc.grid_size.x,
                    mmio_soc.grid_size.y,
                    pcie_enc_override);
            }

            // A MOVER reads its FILLER's handshake block, so it needs that filler's virtual NoC coords and L1
            // address. Fillers occupy indices [0, kNFillers) and are set up FIRST, which is why these are
            // already populated by the time a mover is configured.
            // Per peer slot: (virtual xy, handshake address, ring bank, ring address). Slot 1 stays all-zero
            // for a single-ring mover, and the kernel's kNPeer never reaches it.
            uint32_t peer_xy[kNPeerMax] = {};
            uint32_t peer_hs[kNPeerMax] = {};
            uint32_t peer_bank[kNPeerMax] = {};
            uint32_t peer_addr[kNPeerMax] = {};
            if (is_mover) {
                for (uint32_t pi = 0; pi < ctx.n_peer[d]; pi++) {
                    const uint32_t p = ctx.peer_of[d][pi];
                    peer_xy[pi] = (static_cast<uint32_t>(ctx.drisc_virtual[p].x) & 0xFFFFu) |
                                  ((static_cast<uint32_t>(ctx.drisc_virtual[p].y) & 0xFFFFu) << 16);
                    peer_hs[pi] = ctx.hs_addr[p];
                    peer_bank[pi] = ctx.dram_bank[p];
                    peer_addr[pi] = ctx.dram_addr[p];
                }
            }
            ctx.drain_program[d] = std::make_unique<Program>(CreateProgram());
            const std::vector<uint32_t> cargs = {
                stage_base,
                nstage_drain,  // one fewer than the slots mapped when self-profiling owns the last one
                head_scratch,
                ctx.results_addr[d],
                ctx.done_addr[d],
                ctx.stop_addr[d],
                has_socket ? ctx.sockets[sk]->get_config_buffer_address() : 0u,
                0xFFFFFFFFu,
                128,
                drisc_gap_cycles(),
                ship_repeat(),
                no_noc_init() ? 0u : 1u,
                ablate(),
                ablate_spin(),
                nstage,
                is_mover ? 1u : (my_cores + nstage - 1) / nstage,
                pcie_enc_override,
                fill_target_pct(),
                gap_max_cycles(),
                read_split(),
                // ---- role split (arg 20..31). All zero on the default path, and every use of them in the
                // kernel is behind `if constexpr`, so the emitted code is identical when the knob is off.
                // Args 21/22 are "the ring at index 0": a filler's OWN ring, or a mover's peer-0 ring.
                ctx.role[d],
                is_mover ? peer_bank[0] : ctx.dram_bank[d],
                is_mover ? peer_addr[0] : ctx.dram_addr[d],
                ctx.dram_frames,
                is_filler ? ctx.hs_addr[d] : 0u,
                peer_xy[0],
                peer_hs[0],
                // arg 27..31: the mover's peer COUNT and everything about peer 1.
                ctx.n_peer[d],
                peer_xy[1],
                peer_hs[1],
                peer_bank[1],
                peer_addr[1],
                // ---- DRISC self-profiling (arg 32..35). All zero when the knob is off. ----
                // The identity is passed in rather than read from a firmware global: SPSC_CORE_XY has to be in
                // the SAME coordinate space the worker cores stamp (virtual), because the host resolves a frame
                // to a lane through one map keyed on exactly that. A DRISC guessing its own coordinates would
                // be a second, silent way to get it wrong.
                self_frames_base != 0 ? 1u : 0u,
                drisc_zone_hold_cycles(),
                (static_cast<uint32_t>(ctx.drisc_virtual[d].x) & 0xFFFFu) |
                    ((static_cast<uint32_t>(ctx.drisc_virtual[d].y) & 0xFFFFu) << 16),
                // A MOVER gets 16x the budget: it samples every ~1.3 us against a filler's ~157 us, so one
                // shared cap covers a filler's whole run and cuts a mover off at ~20% of it. Measured need on
                // ResNet-50 trace+2cq: 107 frames for a filler, 1,813-1,936 for a mover, so 256 -> 4,096 keeps
                // comparable headroom at both ends. An explicit TT_METAL_PERF_DEBUG_DRISC_ZONE_FRAMES scales
                // both roles together.
                is_mover ? self_frames_base * 16u : self_frames_base,
                drisc_zone_detail(),
                noc_footprint(),
                // arg 38: the sync event. Gated on zones being on as well, because it rides the self-zone ring
                // and the kernel static_asserts that pairing -- passing 1 with zones off would not build.
                (sync_event_count() != 0 && self_frames_base != 0) ? 1u : 0u};
            const std::string kdrain = "tt_metal/tools/profiler/kernels/drisc_profiler_drain.cpp";
            auto drain_id =
                tensix_drain
                    ? CreateKernel(
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
                    // Report WHERE it stopped, not just that it did. `phase` is the kernel's own progress
                    // marker; without it this warning sent me chasing a stale stop word and a TLB change,
                    // when phase=11 says plainly that it is wedged in the sweep-body write barrier.
                    uint32_t st[5] = {0, 0, 0, 0, 0};
                    cluster.read_core(
                        st, sizeof(st), core, ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]));
                    uint32_t stopw = 0;
                    cluster.read_core(
                        &stopw, sizeof(stopw), core, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
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
                    if (has_socket) {
                        ctx.sockets[sk].reset();
                    }
                    disarm_producers(mesh_device, device_id);
                    return false;
                }
            }

            // ---- VERIFY THE MOVER<->FILLER HANDSHAKE, both directions, before letting data flow ----
            //
            // The heartbeat only proves the kernel is looping. It cannot see that the mover is reading the
            // WRONG L1 word: a bad peer coordinate or address returns a plausible `head`, the mover ships
            // whatever DRAM held, and the result is a capture that decodes to nonsense with every counter
            // reading clean -- exactly how the last two-thread ring bug presented (1.03M records lost, all
            // nesting corrupt, invisible everywhere). So check the planted magic in both directions and
            // refuse the run if either is wrong, rather than discover it in the data.
            // EVERY peer is checked. A dual-ring mover has two independent chances to be pointed at the wrong
            // L1, and the magics are per-peer (see the plant site) so a right-looking value from the WRONG
            // filler no longer passes.
            for (uint32_t pi = 0; is_mover && pi < ctx.n_peer[d]; pi++) {
                const uint32_t p = ctx.peer_of[d][pi];
                const uint32_t want_echo = kProbeFillerMagic + p;  // planted by filler p, read by this mover
                const uint32_t want_back = kProbeMoverMagic + pi;  // written by this mover into peer slot pi
                const tt_cxy_pair mv(device_id, ctx.drisc_virtual[d]);
                const tt_cxy_pair fl(device_id, ctx.drisc_virtual[p]);
                uint32_t echo = 0, back = 0;
                const auto pdl = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
                do {
                    cluster.read_core(
                        &echo,
                        sizeof(echo),
                        mv,
                        ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]) + 20 + 16 * pi);
                    cluster.read_core(
                        &back,
                        sizeof(back),
                        fl,
                        ctx.drisc_l1_noc[p] + (ctx.hs_addr[p] - ctx.drisc_l1_base[p]) + kHsProbeM);
                    if (echo == want_echo && back == want_back) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                } while (std::chrono::steady_clock::now() < pdl);
                if (echo != want_echo || back != want_back) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] Device {}: role-split HANDSHAKE PROBE FAILED for mover {} peer "
                        "slot {} <-> filler {}. mover read 0x{:08X} from the filler's probe word (expected "
                        "0x{:08X} -- a different filler's magic here means the peer coordinate names the WRONG "
                        "filler); the filler holds 0x{:08X} where the mover should have written 0x{:08X}. Peer "
                        "is virtual ({},{}) L1 0x{:x}. Capture is disabled rather than shipping frames read "
                        "from an address neither side agrees on.",
                        device_id,
                        d,
                        pi,
                        p,
                        echo,
                        want_echo,
                        back,
                        want_back,
                        ctx.drisc_virtual[p].x,
                        ctx.drisc_virtual[p].y,
                        ctx.hs_addr[p]);
                    ctx.drain_program[d].reset();
                    if (has_socket) {
                        ctx.sockets[sk].reset();
                    }
                    disarm_producers(mesh_device, device_id);
                    return false;
                }
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] role split: mover {} peer slot {} <-> filler {} handshake OK (peer "
                    "virtual ({},{}) L1 0x{:x}, ring bank {} @ 0x{:x}, {} frames)",
                    d,
                    pi,
                    p,
                    ctx.drisc_virtual[p].x,
                    ctx.drisc_virtual[p].y,
                    ctx.hs_addr[p],
                    ctx.dram_bank[p],
                    ctx.dram_addr[p],
                    ctx.dram_frames);
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
                elf_too_big ? " Reduce drain-kernel code or disable a feature: "
                              "TT_METAL_PERF_DEBUG_DRISC_ZONES and TT_METAL_PERF_DEBUG_NOC_FOOTPRINT together "
                              "are within 32 B of the limit."
                            : "",
                what);
            ctx.drain_program[d].reset();
            if (has_socket) {
                ctx.sockets[sk].reset();
            }
            disarm_producers(mesh_device, device_id);
            return false;
        }

        log_info(
            tt::LogMetal,
            "[perf-debug profiler] Device {}: {} {} resident on logical ({},{}) [noc0 ({},{})], cores "
            "[{},{}) of {}, {} staging slots x {} B",
            device_id,
            tensix_drain ? "TENSIX-BRISC drainer"
                         : (is_filler ? "DRISC FILLER (worker rings -> DRAM ring)"
                                      : (is_mover ? "DRISC MOVER (DRAM ring -> D2H socket)" : "DRISC")),
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

    // ---- DRISC SELF-PROFILING: teach every decoder the drainer cores' identities -------------------------
    //
    // AFTER the bring-up loop, because a drainer's virtual coords only exist once it has been placed, and a
    // mover's socket decoder is created before the last filler is placed. Every socket gets EVERY drainer's
    // entry for the same reason it gets every worker's: a frame's identity is resolved from the map, and a
    // missing entry makes the decoder skip the frame whole -- silently.
    if (self_zones_on) {
        for (uint32_t sk = 0; sk < kNSockets; sk++) {
            if (ctx.decode[sk] == nullptr) {
                continue;
            }
            for (uint32_t d = 0; d < ctx.n_drisc; d++) {
                const uint32_t xy = (static_cast<uint32_t>(ctx.drisc_virtual[d].x) & 0xFFFFu) |
                                    ((static_cast<uint32_t>(ctx.drisc_virtual[d].y) & 0xFFFFu) << 16);
                ctx.decode[sk]->core_of_xy[xy] = self_core0 + d;
            }
        }
        // Mint the drainers' Tracy contexts up front. Only these six -- pre-creating the worker grid litters a
        // capture with empty contexts (see start()), but a drainer core is guaranteed to produce zones when this
        // knob is on, and creating a context costs a GpuNewContext+Populate+name that has no business happening
        // on the drain path.
        std::vector<std::pair<uint32_t, uint32_t>> drisc_noc0;
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            const auto it = ctx.virt_to_noc0.find(
                (static_cast<uint64_t>(ctx.drisc_virtual[d].x) << 32) | static_cast<uint64_t>(ctx.drisc_virtual[d].y));
            if (it != ctx.virt_to_noc0.end()) {
                drisc_noc0.push_back(it->second);
            }
        }
        self_zone_cores_[ctx.chip_id] = std::move(drisc_noc0);
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] Device {}: DRISC SELF-PROFILING on -- {} drainer cores get lanes [{},{}) "
            "and their own Tracy rows | TRACING every sweep in a work-armed window (hold {} us past the last "
            "work), detail {} | budget {} frames/DRISC ({:.0f} KB) | a drainer's staging slot count drops "
            "{} -> {} to make room",
            device_id,
            ctx.n_drisc,
            self_core0 * kNRisc,
            ctx.nl,
            drisc_zone_hold_cycles() / 1350u,
            drisc_zone_detail() == 0 ? "0 (SWEEP + PACE only)" : "1 (full per-batch phases)",
            drisc_zone_frames(),
            drisc_zone_frames() * slot_bytes_all / 1024.0,
            nstage_report,
            nstage_report > 0 ? nstage_report - 1 : 0);
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
        pz::spsc_decode(st, buf.data(), words, ctx.nl, [&](uint32_t, uint32_t type, uint32_t hash, uint64_t, uint32_t) {
            if (hash == 0x7FFFu && type == PP_ZONE_START) {
                ss.stall++;
                w_stalls_++;
            }
            ss.emit++;
        });
        w_decode_ns_ +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_dec_all).count();
        return;
    }
    ss.batch.resize(read_chunk_recs_);  // upper bound on records from one read (words >= records)
    PerfDebugRec* bcur = ss.batch.data();
    PerfDebugRec* const bend = ss.batch.data() + ss.batch.size();
    const uint32_t dev_idx = static_cast<uint32_t>(&ctx - devices_.data());

    // Publish a span of records to the ring. Used both for the tail of a read and for a mid-decode flush
    // when the batch fills, so a full batch is never discarded.
    auto publish_recs = [&](const PerfDebugRec* p, size_t n) {
        if (n == 0 || !ring_) {
            return;
        }
        w_recs_ += n;
        // Stagger probe: remember each lane's FIRST marker timestamp (cheap: one compare per record).
        if (!first_ts_.empty()) {
            for (size_t k = 0; k < n; k++) {
                const uint32_t ln = p[k].lane & 0x00FFFFFFu;
                if (ln < first_ts_.size() && first_ts_[ln] == 0) {
                    first_ts_[ln] = p[k].ts;
                }
            }
        }
        ZoneScopedNC("publish", 0xE67E22);  // orange: publish records to the BroadcastRing
        const auto t_p0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> pg(publish_mu_);  // SP ring, N decoder threads -- see publish_mu_
            ring_->ring.writer().publish_batch(std::span<const PerfDebugRec>(p, n));
        }
        w_publish_ns_ +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_p0).count();
    };

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
            if (lane / kNRisc >= ctx.core_virt.size()) {
                w_drop_lane_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            if (bcur >= bend) {
                // FLUSH, never drop. Dropping deleted markers mid-stream, which unpairs ZONE_START/END and
                // leaves that lane's Tracy GPU stack permanently one level deeper (nesting is by push order),
                // so one lost END scrambles every zone after it on that lane for the rest of the run.
                publish_recs(ss.batch.data(), static_cast<size_t>(bcur - ss.batch.data()));
                bcur = ss.batch.data();
                w_batch_flush_.fetch_add(1, std::memory_order_relaxed);
            }
            {
                // Lanes never cross sockets (contiguous core split), so [sock][lane] has a single writer.
                const size_t li = static_cast<size_t>(sock_idx) * ctx.nl + lane;
                if (li < pub_last_ts_.size()) {
                    if (ts < pub_last_ts_[li]) {
                        w_pub_regress_.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        pub_last_ts_[li] = ts;
                        w_pub_ok_.fetch_add(1, std::memory_order_relaxed);
                    }
                }
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

    publish_recs(ss.batch.data(), static_cast<size_t>(bcur - ss.batch.data()));
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
        w_poll_ns_ +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
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
    // Drain-to-empty on stop: stop() writes the quiesce value first, so the drainer stops producing; keep reading until
    // every socket has been empty for a sustained window, else the tail of the run is lost. Deadline backstops it.
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
                    for (uint32_t dd = 0; dd < c.n_drisc; dd++) {
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
    w_wall_ns_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_wall0).count();
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
    std::vector<uint64_t> con_last_ts_(4096, 0);
    auto emit_batch = [&](std::span<PerfDebugRec> b) {
        for (const auto& r : b) {
            if (r.type != PP_ZONE_START && r.type != PP_ZONE_END) {
                continue;
            }
            const uint32_t ln = r.lane & 0xFFFFFFu;
            if (ln < con_last_ts_.size()) {
                w_con_seen_.fetch_add(1, std::memory_order_relaxed);
                if (r.ts < con_last_ts_[ln]) {
                    w_con_regress_.fetch_add(1, std::memory_order_relaxed);
                } else {
                    con_last_ts_[ln] = r.ts;
                }
            }
        }
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
            // DRISC self-profiling. These ids are FIXED rather than source-location hashes (see
            // DriscSelfZone), so generateZoneSourceLocationsHashes() can never supply their names --
            // registering them here is the only thing that stops them showing up as "Zone_32752".
            zone_names_[kernel_profiler::DRISC_ZONE_SWEEP] = "DRISC-SWEEP";
            zone_names_[kernel_profiler::DRISC_ZONE_READ] = "DRISC-READ";
            zone_names_[kernel_profiler::DRISC_ZONE_READ_WAIT] = "DRISC-READ-WAIT";
            zone_names_[kernel_profiler::DRISC_ZONE_PROC] = "DRISC-PROC";
            zone_names_[kernel_profiler::DRISC_ZONE_CREDIT_WAIT] = "DRISC-CREDIT-WAIT";
            zone_names_[kernel_profiler::DRISC_ZONE_WRITE] = "DRISC-WRITE";
            zone_names_[kernel_profiler::DRISC_ZONE_WR_BARRIER] = "DRISC-WR-BARRIER";
            zone_names_[kernel_profiler::DRISC_ZONE_PACE] = "DRISC-PACE";
            zone_names_[kernel_profiler::DRISC_ZONE_SYNC] = "DRISC-SYNC";
            // Explicit colours for the drainer zones. The pair that matters is SWEEP vs PACE: they alternate
            // continuously on a filler row, so "the drainer is working" vs "the controller is holding it off"
            // has to be readable without reading labels. SWEEP is a saturated blue; PACE is a recessive grey,
            // because it is deliberate idleness and should not compete for attention with real work.
            zone_colors_[kernel_profiler::DRISC_ZONE_SWEEP] = 0x2E86C1;        // blue: a sweep of the grid
            zone_colors_[kernel_profiler::DRISC_ZONE_PACE] = 0x707B7C;         // grey: paced idle, on purpose
            zone_colors_[kernel_profiler::DRISC_ZONE_READ] = 0x27AE60;         // green: NoC reads
            zone_colors_[kernel_profiler::DRISC_ZONE_READ_WAIT] = 0x196F3D;    // dark green: read barrier
            zone_colors_[kernel_profiler::DRISC_ZONE_PROC] = 0x8E44AD;         // purple: scan + head write-back
            zone_colors_[kernel_profiler::DRISC_ZONE_CREDIT_WAIT] = 0xC0392B;  // red: the phase that sets the knee
            zone_colors_[kernel_profiler::DRISC_ZONE_WRITE] = 0xD35400;        // orange: egress
            zone_colors_[kernel_profiler::DRISC_ZONE_WR_BARRIER] = 0xF1C40F;   // yellow: waiting for write acks
            // White, and the same on both roles: the sync marker is a fiducial, not a phase. It should be
            // findable at a glance on any row and should not read as belonging to the work palette.
            zone_colors_[kernel_profiler::DRISC_ZONE_SYNC] = 0xFFFFFF;
            zone_colors_mover_[kernel_profiler::DRISC_ZONE_SYNC] = 0xFFFFFF;
            // MOVER palette. Same zone ids, different hues, because the two roles are different machines: a
            // filler scans worker L1 and stages to DRAM, a mover reads DRAM and pushes PCIe. Reading a mover row
            // with a filler's colour scale in your head is the mistake this prevents.
            zone_colors_mover_[kernel_profiler::DRISC_ZONE_SWEEP] = 0x16A085;        // teal: a mover's visit
            zone_colors_mover_[kernel_profiler::DRISC_ZONE_READ] = 0x52BE80;         // light green: DRAM read
            zone_colors_mover_[kernel_profiler::DRISC_ZONE_CREDIT_WAIT] = 0xE74C3C;  // bright red: THE knee phase
            zone_colors_mover_[kernel_profiler::DRISC_ZONE_WRITE] = 0xE67E22;        // light orange: PCIe push
            zone_colors_mover_[kernel_profiler::DRISC_ZONE_WR_BARRIER] = 0xF7DC6F;   // light yellow: write acks
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
            // Colour by zone AND by role. Core indices at or past n_worker_cores are the drainers, appended in
            // DRISC order, so the role comes straight off ctx.role[] -- see the lane-block comment in
            // boot_device.
            {
                const bool is_mover = ctx.n_worker_cores != 0 && ci >= ctx.n_worker_cores &&
                                      (ci - ctx.n_worker_cores) < ctx.n_drisc &&
                                      ctx.role[ci - ctx.n_worker_cores] == kRoleMover;
                const auto& tbl = is_mover ? zone_colors_mover_ : zone_colors_;
                if (auto it = tbl.find(static_cast<uint16_t>(r.zone)); it != tbl.end()) {
                    pkt.color = it->second;
                } else if (auto it2 = zone_colors_.find(static_cast<uint16_t>(r.zone)); it2 != zone_colors_.end()) {
                    pkt.color = it2->second;  // mover table has no override for this zone
                }
            }
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
        case 16: phase_name = "RING-WAIT(DRAM ring full)"; break;
        default: break;
    }
    uint32_t np = 0, fifo_pages = 0;
    // A role-split FILLER owns no socket, so there is no FIFO to report -- its back-pressure is the DRAM
    // ring, whose head/tail live in the pad this function already reads.
    const uint32_t sk = ctx.sock_of[d];
    if (sk != kNoSocket && ctx.sockets[sk]) {
        np = ctx.sockets[sk]->pages_available();
        fifo_pages = ctx.sockets[sk]->get_fifo_curr_size() / ctx.sockets[sk]->get_page_size();
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
    // A FILLER's back-pressure is its DRAM ring, not a host FIFO, so "0 of 0 fifo pages" above says nothing
    // about it. Read the live handshake words instead: a filler frozen at phase 16 with head-tail pinned at
    // capacity is ring-blocked (its mover died), which is a completely different failure from a credit-wait.
    if (ctx.role[d] == kRoleFiller) {
        uint32_t hd = 0, tl = 0;
        const uint64_t hs = ctx.drisc_l1_noc[d] + (ctx.hs_addr[d] - ctx.drisc_l1_base[d]);
        cluster.read_core(&hd, sizeof(hd), drisc, hs + kHsHead);
        cluster.read_core(&tl, sizeof(tl), drisc, hs + kHsTail);
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler]   filler {}: DRAM ring head {} tail {} => {} frames in flight of {} "
            "capacity{}",
            d,
            hd,
            tl,
            hd - tl,
            ctx.dram_frames,
            (ctx.dram_frames != 0 && (hd - tl) >= ctx.dram_frames) ? "  <<< RING FULL: the mover is not consuming"
                                                                   : "");
    }
    if (b[1] == a[1] && !exited && b[2] == 3 && np == 0) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler]   => CONTRADICTION: drainer blocked on credits while host sees an EMPTY "
            "fifo. bytes_sent/bytes_acked have desynchronized; the sender is waiting for credit the host "
            "believes it already granted.");
    }
}

// COMMON-TRIGGER SYNC EVENT: release every drainer from a rendezvous barrier (req | ack | go in the pad behind
// `stop`) so all six mark the SAME instant, leaving anchor + render error as the only thing the spread contains.
// Waiting for every `ack` is load-bearing -- it guarantees no core is still mid-sweep (~157 us of phase error).
// The release is 6 sequential writes (scattered DRAM cores are not a multicast rectangle), so the ORDER IS
// ROTATED per generation: that turns the write skew from an unknown into a measurable, separable term (N+48).
void PerfDebugProfiler::fire_sync_events() {
    const uint32_t n = sync_event_count();
    if (n == 0 || devices_.empty()) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    for (auto& ctx : devices_) {
        // Collect the participants once: a drainer with no program never parks, and waiting on its ack would
        // time out every trigger.
        std::vector<uint32_t> live;
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            if (ctx.drain_program[d] != nullptr) {
                live.push_back(d);
            }
        }
        if (live.empty()) {
            continue;
        }
        auto word = [&](uint32_t d, uint32_t off) {
            return ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]) + off;
        };
        for (uint32_t gen = 1; gen <= n; gen++) {
            // 1. ask everyone to park.
            for (uint32_t d : live) {
                cluster.write_core(&gen, sizeof(gen), tt_cxy_pair(ctx.chip_id, ctx.drisc_virtual[d]), word(d, 4));
            }
            // 2. wait for every ack. A drainer notices `req` at the top of its next sweep, so the wait is one
            // sweep at worst (~157 us on a filler, ~1.3 us on a mover); 2 s is 4 orders of magnitude of slack
            // and only expires if a drainer is genuinely not looping.
            bool all_parked = true;
            const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            for (uint32_t d : live) {
                uint32_t ack = 0;
                while (std::chrono::steady_clock::now() < dl) {
                    cluster.read_core(&ack, sizeof(ack), tt_cxy_pair(ctx.chip_id, ctx.drisc_virtual[d]), word(d, 8));
                    if (ack == gen) {
                        break;
                    }
                }
                if (ack != gen) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] sync event gen {}: DRISC {} never parked (ack={}); this trigger is "
                        "INCOMPLETE and its spread must not be quoted -- a missing participant makes the spread "
                        "over the cores that did answer look artificially tight",
                        gen,
                        d,
                        ack);
                    all_parked = false;
                }
            }
            // 3. release. Rotated start, and the host clock is read either side so the skew is BOUNDED BY
            // MEASUREMENT rather than by the 170 ns estimate.
            const size_t rot = (gen - 1) % live.size();
            const int64_t h0 = tracy::Profiler::GetTime();
            for (size_t i = 0; i < live.size(); i++) {
                const uint32_t d = live[(rot + i) % live.size()];
                cluster.write_core(&gen, sizeof(gen), tt_cxy_pair(ctx.chip_id, ctx.drisc_virtual[d]), word(d, 12));
            }
            const int64_t h1 = tracy::Profiler::GetTime();
            std::string order;
            for (size_t i = 0; i < live.size(); i++) {
                order += fmt::format("{}{}", i ? "," : "", live[(rot + i) % live.size()]);
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] SYNC EVENT gen {} fired on device {}: release order [{}], host-measured "
                "write span {} ns over {} drainers ({:.0f} ns/write){}",
                gen,
                ctx.chip_id,
                order,
                h1 - h0,
                live.size(),
                live.empty() ? 0.0 : static_cast<double>(h1 - h0) / static_cast<double>(live.size()),
                all_parked ? "" : " -- INCOMPLETE, see warning above");
            if (gen < n) {
                std::this_thread::sleep_for(std::chrono::milliseconds(sync_event_gap_ms()));
            }
        }
    }
    // The frames carrying these zones are already in flight; give the reader/decoder/consumer chain a moment
    // to pull them through before stop() starts quiescing, so a sync marker cannot be lost to teardown.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

void PerfDebugProfiler::stop() {
    if (stopped_.exchange(true)) {
        return;
    }

    // BEFORE any quiesce: the drainers must still be looping to answer a rendezvous, and firing here rather
    // than at bring-up keeps the measurement out of the workload's way (a parked drainer is not draining) and
    // lets the zone-name harvest run against already-compiled kernels.
    fire_sync_events();

    // ---- ROLE SPLIT: quiesce in the right ORDER, before the per-DRISC teardown loop below ----
    //
    // The loop below tears each DRISC down completely -- stop=1, wait for done, read results, stop=2 -- one
    // at a time. stop=2 is what lets a kernel run its NIU-restore tail, and in NOC2AXI mode an inbound
    // DRAM-range address is forwarded to GDDR instead of terminating at L1. So releasing filler 0 while its
    // mover is still running makes the mover's `head` read return GDDR contents.
    //
    // THIS ACTUALLY HAPPENED and it is worth spelling out, because every counter looked fine: the mover read
    // 0xF5AE93CB as a head, head-tail underflowed to ~4.1e9, the "clamp to kNStage" step turned that into
    // "7 frames are ready", and the movers shipped ~1,800 frames of garbage -- reporting 5,957 frames moved
    // against 4,144 ever staged. The kernel now refuses an impossible head (out[57], hs_bad), and this
    // pre-pass removes the cause: get every kernel out of its loop first, and only then release any NIU.
    //
    // Three phases, because the ring has to be drained in between:
    //   1. quiesce the FILLERS  -- they stop producing but stay resident in stream mode, so head stays readable
    //   2. let each MOVER catch its filler's final head -- otherwise the ring's tail is lost every run
    //   3. quiesce the MOVERS
    // The default 2-drainer path shares nothing between drainers and skips all of this.
    for (auto& ctx : devices_) {
        if (ctx.n_drisc <= kNSockets) {
            continue;
        }
        auto& cluster = MetalContext::instance().get_cluster();
        auto stop_word = [&](uint32_t d) { return ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]); };
        auto done_word = [&](uint32_t d) { return ctx.drisc_l1_noc[d] + (ctx.done_addr[d] - ctx.drisc_l1_base[d]); };
        auto quiesce = [&](uint32_t d) {
            if (ctx.drain_program[d] == nullptr) {
                return;
            }
            const tt_cxy_pair drisc(ctx.chip_id, ctx.drisc_virtual[d]);
            uint32_t one = 1;
            cluster.write_core(&one, sizeof(uint32_t), drisc, stop_word(d));
            const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            uint32_t done = 0;
            while (std::chrono::steady_clock::now() < dl) {
                cluster.read_core(&done, sizeof(uint32_t), drisc, done_word(d));
                if ((done & 0xFFFF0000u) == 0xD09E0000u) {
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] Device {}: role-split DRISC {} ({}) did not acknowledge stop in the "
                "ordered quiesce",
                ctx.chip_id,
                d,
                ctx.role[d] == kRoleFiller ? "filler" : "mover");
            dump_drainer_state(ctx, d, "ordered-quiesce");
        };
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            if (ctx.role[d] == kRoleFiller) {
                quiesce(d);
            }
        }
        // Phase 2. The filler publishes its final head in its exit tail, so by now head is the true total.
        // ONE WAIT PER PEER RING, not per mover: a dual-ring mover can be caught up on ring 0 and still owe
        // hundreds of frames on ring 1, and the tail of a capture that goes missing this way is invisible in
        // every host-side counter (the records were simply never sent).
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            if (ctx.role[d] != kRoleMover || ctx.drain_program[d] == nullptr) {
                continue;
            }
            for (uint32_t pi = 0; pi < ctx.n_peer[d]; pi++) {
                const uint32_t p = ctx.peer_of[d][pi];
                if (ctx.drain_program[p] == nullptr) {
                    continue;
                }
                uint32_t head = 0;
                cluster.read_core(
                    &head,
                    sizeof(head),
                    tt_cxy_pair(ctx.chip_id, ctx.drisc_virtual[p]),
                    ctx.drisc_l1_noc[p] + (ctx.hs_addr[p] - ctx.drisc_l1_base[p]) + kHsHead);
                const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                uint32_t tail = 0;
                while (std::chrono::steady_clock::now() < dl) {
                    // The mover's LIVE tail for peer slot pi: four words per peer behind `done`, so slot 0's
                    // tail is at +32 and slot 1's at +48.
                    cluster.read_core(
                        &tail,
                        sizeof(tail),
                        tt_cxy_pair(ctx.chip_id, ctx.drisc_virtual[d]),
                        done_word(d) + 32 + 16 * pi);
                    if (tail >= head) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if (tail < head) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] role split: mover {} did not drain filler {}'s ring (peer slot {}) "
                        "before teardown ({} of {} frames); the tail of the capture is lost",
                        d,
                        p,
                        pi,
                        tail,
                        head);
                }
            }
        }
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            if (ctx.role[d] == kRoleMover) {
                quiesce(d);
            }
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
                // The writer thread is still draining, so the socket keeps emptying while we wait.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if ((done & 0xFFFF0000u) != 0xD09E0000u) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] Device {}: DRISC drainer did not acknowledge stop",
                    ctx.chip_id);
                // WHERE is it stuck? The phase word is live while the loop runs, so this says which part of
                // the kernel is blocking instead of leaving it to inference. POLL = sweep body (NoC reads or
                // the control-vector pass), RESERVE = credit wait (should be impossible now it is bounded),
                // WRITE = the PCIe write / push / notify / barrier, EXIT = the socket teardown tail.
                dump_drainer_state(ctx, d, "stop-not-acked");
            }
            // The drainer's own view of the run. Host-side page and marker counts cannot distinguish a
            // bandwidth wall from a latency one; sweeps/frames/cycles can.
            std::vector<uint32_t> res(kernel_profiler::SPSC_DRAIN_RESULT_WORDS, 0);
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
                // WORKER cores. With self-profiling on, core_virt also holds the drainers, and the head
                // write-back is issued to producers only -- reporting 126 issues/sweep on a 120-core grid.
                ctx.n_worker_cores != 0 ? ctx.n_worker_cores : ctx.core_virt.size());
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
            // ---- ROLE SPLIT counters (only printed when the knob is on) ----
            //
            // ring high-water is the number the whole change rests on: the point of staging in DRAM is that
            // the elastic buffer is no longer 21 busy sweeps. If high-water stays tiny, the mover is keeping
            // up and the ring is doing nothing; if it approaches capacity, the DRAM ring has become the new
            // wall and ring_blocked will be non-zero to prove it. Reported per DRISC, always, so a bad run
            // never has to be re-run to find out.
            if (res[48] != 0) {
                const bool is_mv = res[48] == 2;
                // ONE LINE PER RING, never a per-DRISC summary: a dual-ring mover with one healthy ring and
                // one short one has to be visible at a glance, and a summed "frames moved" would hide it.
                // Peer 1's counters mirror peer 0's at out[58..63] (see the kernel's results block).
                const uint32_t nring = is_mv ? std::max<uint32_t>(1u, ctx.n_peer[d]) : 1u;
                for (uint32_t pi = 0; pi < nring; pi++) {
                    const uint32_t moved = pi == 0 ? res[49] : res[58];
                    const uint32_t hi = pi == 0 ? res[50] : res[63];
                    const uint32_t tail = pi == 0 ? res[53] : res[59];
                    const uint32_t batch = pi == 0 ? res[54] : res[60];
                    log_info(
                        tt::LogMetal,
                        "[perf-debug profiler] role split DRISC {} ({}){}: {} frames {}, DRAM ring head-tail "
                        "high-water {}/{} frames ({:.1f}% of {:.1f} MiB) | ring-room waits {} [0 = the ring never "
                        "became the bottleneck] | tail {} | max batch {}{}",
                        d,
                        is_mv ? "MOVER" : "FILLER",
                        is_mv ? fmt::format(" peer slot {} = filler {}", pi, ctx.peer_of[d][pi]) : std::string(),
                        moved,
                        is_mv ? "moved out of the ring" : "staged into the ring",
                        hi,
                        res[52],
                        res[52] ? 100.0 * hi / static_cast<double>(res[52]) : 0.0,
                        (static_cast<double>(res[52]) * 10560.0) / (1024.0 * 1024.0),
                        res[51],
                        tail,
                        batch,
                        res[51] != 0 ? "  <<< the DRAM ring FILLED: raise TT_METAL_PERF_DEBUG_ROLE_RING_MB" : "");
                }
                if (is_mv) {
                    // MUST be 0. Non-zero means the mover was handed a value that cannot be a head, so it
                    // stopped shipping -- see the ordered-quiesce comment in stop() for the failure this
                    // catches.
                    if (res[57] != 0) {
                        log_warning(
                            tt::LogMetal,
                            "[perf-debug profiler] role split DRISC {}: {} IMPOSSIBLE head reads (head - tail "
                            "exceeded the ring capacity, summed over its rings). The mover declared egress dead "
                            "and stopped shipping. A filler's L1 became unreadable mid-run -- most likely an NIU "
                            "released early.",
                            d,
                            res[57]);
                    }
                    // The first frame word the mover ever read out of DRAM, PER RING. If this is not the frame
                    // magic the filler writes, the two sides disagree about the ring address and the capture is
                    // garbage regardless of how healthy every other counter looks. Checked per ring because the
                    // rings are in different DRAM banks and only one of them may be mis-addressed.
                    const uint32_t want = kernel_profiler::spsc_span_w0();
                    for (uint32_t pi = 0; pi < nring; pi++) {
                        const uint32_t got = pi == 0 ? res[55] : res[61];
                        const uint32_t moved = pi == 0 ? res[49] : res[58];
                        if (moved != 0 && got != want) {
                            log_warning(
                                tt::LogMetal,
                                "[perf-debug profiler] role split DRISC {} peer slot {} (filler {}): first DRAM "
                                "frame word was 0x{:08X}, expected the frame header 0x{:08X}. The filler's ring "
                                "address and the mover's do NOT agree -- treat this run's markers as invalid.",
                                d,
                                pi,
                                ctx.peer_of[d][pi],
                                got,
                                want);
                        }
                    }
                }
            }
            // ---- DRISC SELF-PROFILING counters (only printed when the knob is on) ----
            //
            // Everything needed to tell "captured the right 0.5% of the run" from "captured the idle loop and
            // looked healthy doing it": how many captured sweeps DID WORK (the ones that matter -- a mover's
            // credit wait only exists on those), how many were discarded, how many were refused for budget, and
            // the bytes. A summary frame count alone cannot distinguish those cases, which is the failure mode
            // this whole block exists to make impossible.
            if (res[64] != 0 || res[66] != 0 || res[69] != 0) {
                const uint64_t c_self_cyc = (static_cast<uint64_t>(res[72]) << 32) | res[71];
                const uint32_t self_bytes = res[64] * 10560u;
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] DRISC {} SELF-ZONES ({}, detail {}): TRACED {} of {} sweeps "
                    "({:.1f}% of ALL sweeps, {} of them did work) across {} work window(s) | {} frames of a {} "
                    "budget ({:.0f} KB, {:.2f}% of this drainer's {:.1f} MB egress) | {} markers, {} words, "
                    "{:.1f} markers/sweep | publish cost {:.2f} ms ({:.2f}% of the run){}{}",
                    d,
                    res[48] == kRoleFiller ? "FILLER" : (res[48] == kRoleMover ? "MOVER" : "FULL"),
                    res[86],
                    res[66],
                    res[4],
                    res[4] ? 100.0 * res[66] / static_cast<double>(res[4]) : 0.0,
                    res[67],
                    res[68],
                    res[64],
                    res[85],
                    self_bytes / 1024.0,
                    res[5] ? 100.0 * self_bytes / (static_cast<double>(res[5]) * 64.0) : 0.0,
                    (static_cast<double>(res[5]) * 64.0) / (1024.0 * 1024.0),
                    res[65],
                    res[73],
                    res[66] ? static_cast<double>(res[65]) / res[66] : 0.0,
                    (static_cast<double>(c_self_cyc) / kCycPerUs) / 1000.0,
                    cyc ? 100.0 * static_cast<double>(c_self_cyc) / static_cast<double>(cyc) : 0.0,
                    // The budget is a COVERAGE limit now, so say when it bound: the row just ends, and a short
                    // row is otherwise indistinguishable from a drainer that stopped having work to do.
                    res[69] != 0 ? fmt::format(
                                       "  <<< BUDGET EXHAUSTED after {} frames: {} later sweeps went untraced, "
                                       "so this row ENDS EARLY rather than the drainer going idle. Raise "
                                       "TT_METAL_PERF_DEBUG_DRISC_ZONE_FRAMES.",
                                       res[64],
                                       res[69])
                                 : std::string(),
                    res[70] != 0 ? fmt::format(" | {} markers LOST (a publish could not free the ring)", res[70])
                                 : std::string());
                // COMMON-TRIGGER SYNC EVENT. Reported per drainer because the measurement is only valid if
                // EVERY participant answered EVERY trigger: a drainer that timed out at the barrier contributed
                // no marker, and the spread over the cores that did answer would then read artificially tight.
                // So state the counts and warn on any timeout rather than leaving it to be inferred from a
                // missing row in the CSV.
                if (res[130] != 0 || res[131] != 0) {
                    log_info(
                        tt::LogMetal,
                        "[perf-debug profiler] DRISC {} SYNC EVENTS: {} marked, {} timed out | last barrier park "
                        "{:.1f} us",
                        d,
                        res[130],
                        res[131],
                        res[132] / kCycPerUs);
                }
                if (res[131] != 0) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] DRISC {} SYNC EVENTS: {} barrier(s) were NEVER RELEASED. Those "
                        "triggers are incomplete and their spread is not a valid alignment measurement.",
                        d,
                        res[131]);
                }
                // Trace COMPLETENESS: every word written into the self ring must have been shipped. A shortfall
                // is trace stranded in the ring at teardown, which no other counter reveals -- it just makes the
                // Tracy row end early, indistinguishable from the drainer going quiet.
                if (res[87] != res[73]) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] DRISC {} SELF-ZONES: {} of {} words shipped -- {} words ({} "
                        "markers) of this drainer's own trace were STRANDED IN THE RING at teardown, so its "
                        "Tracy row ends early.",
                        d,
                        res[87],
                        res[73],
                        res[73] - res[87],
                        (res[73] - res[87]) / 2);
                }
                // A capture whose sweeps all turned out idle is a FAILED capture even though every count above
                // looks healthy -- on a mover especially, since the credit wait that sets the knee only happens
                // on a busy visit. Say so rather than let a plausible frame count stand in for coverage.
                if (res[66] != 0 && res[67] == 0) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] DRISC {} SELF-ZONES: all {} traced sweeps were IDLE -- the window "
                        "opened but nothing of consequence happened inside it. Treat this row as cadence only.",
                        d,
                        res[66]);
                }
            }
            // ---- NoC FOOTPRINT (out[88..129], TT_METAL_PERF_DEBUG_NOC_FOOTPRINT=1) ----
            //
            // The hardware's own tally of what this drainer put on the mesh, replacing arithmetic over
            // frame counts. TWO BLOCKS, NEVER BLENDED: the workload window (first busy sweep -> last busy
            // sweep) and the resident lifetime (device open -> teardown). A resident drainer polls from
            // device open, so the lifetime figure is dominated by traffic no workload asked for, and one
            // number for both is the wrong-population trap this file has been burned by twice.
            if (res[128] != 0) {
                const uint32_t wbytes = res[129];  // NOC_WORD_BYTES, from the device header -- never hardcoded
                // The Tracy PLOT path cannot afford a per-sample lookup, so it carries kNocWordBytes = 64 as a
                // constant -- the one assumed input in an otherwise measured chain. Cross-check it against the
                // device's own value here and say so loudly on a mismatch, otherwise every NoC GB/s plot would
                // be silently mis-scaled while this block stayed correct.
                if (wbytes != 64u) {
                    log_error(
                        tt::LogMetal,
                        "[perf-debug profiler] NOC_WORD_BYTES is {} on device but the Tracy plot path assumes 64 "
                        "-- every NoC GB/s plot is scaled by {}x. Fix kNocWordBytes in "
                        "perf_debug_profiler_tracy_handler.cpp.",
                        wbytes,
                        wbytes / 64.0);
                }
                auto q = [&](uint32_t base, uint32_t noc, uint32_t k) {
                    const uint32_t o = base + (noc * 4u + k) * 2u;
                    return (static_cast<uint64_t>(res[o + 1]) << 32) | res[o];
                };
                // A posted write is acked without the data words being counted where we read them, so a
                // non-zero count means the byte totals are UNDER-reported. Every write on this path is
                // posted=false, so this must be 0; say so rather than print a plausible low number.
                const bool posted_ok = (res[125] == 0 && res[126] == 0);
                const bool win_ok = (res[127] != 0);
                for (uint32_t blk = 0; blk < 2; blk++) {
                    const uint32_t base = (blk == 0) ? 104u : 88u;
                    if (blk == 0 && !win_ok) {
                        log_warning(
                            tt::LogMetal,
                            "[perf-debug profiler] DRISC {} NoC FOOTPRINT -- WORKLOAD WINDOW: no sweep ever "
                            "did work, so there is no window to report. Lifetime block follows.",
                            d);
                        continue;
                    }
                    uint64_t rw = 0, rt = 0, ww = 0, wt = 0;
                    for (uint32_t n = 0; n < 2; n++) {
                        rw += q(base, n, 0);
                        rt += q(base, n, 1);
                        ww += q(base, n, 2);
                        wt += q(base, n, 3);
                    }
                    const uint64_t bytes = (rw + ww) * wbytes;
                    const uint64_t txns = rt + wt;
                    log_info(
                        tt::LogMetal,
                        "[perf-debug profiler] DRISC {} NoC FOOTPRINT -- {}: {:.2f} MB in {} txns "
                        "[NoC0 rd {:.2f} MB/{} txn, wr {:.2f} MB/{} txn | NoC1 rd {:.2f} MB/{} txn, wr "
                        "{:.2f} MB/{} txn] | mean {} B/txn{}",
                        d,
                        blk == 0 ? "WORKLOAD WINDOW" : "RESIDENT LIFETIME",
                        bytes / (1024.0 * 1024.0),
                        txns,
                        q(base, 0, 0) * wbytes / (1024.0 * 1024.0),
                        q(base, 0, 1),
                        q(base, 0, 2) * wbytes / (1024.0 * 1024.0),
                        q(base, 0, 3),
                        q(base, 1, 0) * wbytes / (1024.0 * 1024.0),
                        q(base, 1, 1),
                        q(base, 1, 2) * wbytes / (1024.0 * 1024.0),
                        q(base, 1, 3),
                        txns != 0 ? bytes / txns : 0,
                        posted_ok ? "" : "  <<< POSTED WRITES SEEN: byte totals are UNDER-reported");
                    if (blk == 0) {
                        const uint64_t wcyc = (static_cast<uint64_t>(res[122]) << 32) | res[121];
                        log_info(
                            tt::LogMetal,
                            "[perf-debug profiler] DRISC {}   window = {} sweeps / {} cycles. Amplification "
                            "is NOT printed here: the frame payload is on the role-split line above, and a "
                            "ratio belongs in FINDINGS with its derivation shown rather than asserted here",
                            d,
                            res[120],
                            wcyc);
                    }
                }
                // The instrument's own cost, as a share of the run. If this is not small the footprint
                // numbers describe a perturbed drainer, and N+41 measured that +4% on idle-sweep cost is
                // enough to cross the knee at SD delay 15.
                const uint64_t nfc = (static_cast<uint64_t>(res[124]) << 32) | res[123];
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] DRISC {}   instrument cost {} cycles over {} sweeps "
                    "({} cyc/sweep) -- compare against the sweep durations above before trusting the "
                    "footprint at the knee",
                    d,
                    nfc,
                    res[4],
                    res[4] != 0 ? nfc / res[4] : 0);
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] DRISC {}   NO workload-relative percentage is quoted: the "
                    "workload's own NoC traffic was not measured. That needs victim-side NIU_SLV deltas "
                    "against a drainers-off baseline.",
                    d);
            }
            // The cross-check side: phase totals over EXACTLY the sweeps the zones describe, so summing
            // zone durations per name out of the Tracy capture must reproduce these. See the kernel.
            auto u64r = [&res](size_t i) { return (static_cast<uint64_t>(res[i + 1]) << 32) | res[i]; };
            if (res[86] != 0) {
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] DRISC {} SELF-ZONES cross-check over {} fully-instrumented sweeps: "
                    "read {:.1f} us | proc {:.1f} us | credit-wait {:.1f} us | write {:.1f} us | wr-barrier "
                    "{:.1f} us -- sum(DRISC-READ)+sum(DRISC-READ-WAIT), sum(DRISC-PROC)-sum(DRISC-CREDIT-WAIT)"
                    "-sum(DRISC-WRITE), sum(DRISC-CREDIT-WAIT), sum(DRISC-WRITE), sum(DRISC-WR-BARRIER) from "
                    "the capture must match these",
                    d,
                    res[74],
                    u64r(75) / kCycPerUs,
                    u64r(77) / kCycPerUs,
                    u64r(79) / kCycPerUs,
                    u64r(81) / kCycPerUs,
                    u64r(83) / kCycPerUs);
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
            cluster.write_core(
                &two, sizeof(uint32_t), drisc, ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
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
        // WORKER cores only. With DRISC self-profiling on, core_virt also holds the drainer cores, and a DRAM
        // core has no producer and no stall counters -- reading the TENSIX profiler address on one returns
        // whatever is at that offset in DRISC L1. Measured before this bound existed: "80,475,310,058 producer
        // stalls across 73 of 126 cores", which reads as a catastrophically perturbed run and is pure garbage.
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
            n_stall_cores,
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
        log_info(
            tt::LogMetal,
            // "order-checked", not "records": these counts are the records the per-lane ORDER INVARIANT examined,
            // which is fewer than the records published once the NoC-footprint series is on. A PP_DATA sample
            // carries a counter value, not a per-lane monotonic timestamp, so there is no ordering to test and it
            // is correctly excluded. Labelling it "of N at publish" made a healthy run look like it had lost 6,000
            // records next to "consumer took" -- same defect as reporting `dropped N` without saying the survivors
            // are scrambled. The difference (took - order-checked) is a useful free cross-check: it measures series
            // records emitted, independently of what the device reports.
            "[perf-debug profiler] order/loss: per-lane ts regressions {} of {} order-checked at publish, {} of {} "
            "order-checked at consume [both MUST be 0; non-zero = records reordered => Tracy nesting corrupt] | "
            "lane-bound drops {} | batch flushes {}",
            w_pub_regress_.load(),
            w_pub_ok_.load() + w_pub_regress_.load(),
            w_con_regress_.load(),
            w_con_seen_.load(),
            w_drop_lane_.load(),
            w_batch_flush_.load());
    }
    tracy_.reset();
    devices_.clear();
}

}  // namespace tt::tt_metal
