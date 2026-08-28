// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <queue>
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
#include "tools/profiler/perf_debug_consumer.hpp"
#include "tools/profiler/perf_debug_env.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"
#include "tools/profiler/perf_debug_receiver.hpp"
#include "tools/profiler/sync/eth_wallclock_sync_host.hpp"
#include "tools/profiler/perf_debug_tracy_consumer.hpp"
#include "llrt/zone_meta.hpp"  // per-ELF (zone id -> source location), the streaming name source
#include "tools/profiler/spsc_packet.h"

namespace tt::tt_metal {

namespace pz = tt::tt_metal::profiler;

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
//
// DEFAULT 448 MiB, because runway is what keeps producers unstalled and it scales with VOLUME, not rate:
// roughly 19 MB per 1k iterations per filler at the knee, so 448 covers ~24k iterations' worth of backlog
// while 64 covers ~3.4k. Nothing is reserved unless the streaming profiler is enabled (see the gate above),
// so the cost is paid only by runs that profile.
uint32_t role_ring_mb() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ROLE_RING_MB");
        const uint32_t n = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 448u;
        return n == 0 ? 448u : n;
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
// TT_METAL_PERF_DEBUG_FILLERS: the role-split shape. 6 (default) = 6 fillers + 1 mover: every DRAM view
// sweeps workers (cores/filler 30 -> 20, ~1.5x faster revisit cadence = the onset lever) and the single
// mover owns all six rings, at the cost of HALF the sustained evacuation ceiling. 4 = 4 fillers + 2 movers,
// the sustained-optimized roster: the second mover is worth ~2.3x on the sustained knee, so switch to 4 for
// long max-rate captures. The 6-shape is the default because it keeps producers unstalled at the highest
// offered rates, which is what perturbs a workload's own timing.
uint32_t n_fillers() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FILLERS");
        const uint32_t x = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 6u;
        TT_FATAL(x == 4 || x == 6, "TT_METAL_PERF_DEBUG_FILLERS must be 4 or 6, got {}", x);
        return x;
    }();
    return v;
}
uint32_t n_sockets_split() { return n_fillers() == 6 ? 1u : 2u; }

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
            // 6-filler shape: bank 3 (the second y==0 safe bank, unused with 1 mover) and bank 2 join.
            out = n_fillers() == 6 ? std::vector<uint32_t>{5u, 6u, 4u, 1u, 2u, 3u}
                                   : std::vector<uint32_t>{5u, 6u, 4u, 1u};
        }
        return out;
    }();
    return v;
}

// ALLOCATOR bank ids the RINGS live in. Since stage_run pushes frames with the DRAM tile's own GDDR DMA
// engine (gddr_dma.h), a filler's ring MUST live in its OWN bank: the DMA reaches only the channel its tile
// fronts, and allocator bank ids are DRAM view ids one-for-one (l1_banking_allocator sizes the DRAM
// allocator off get_num_dram_views() with identity bank<->channel maps). So the default is rb[i] == fb[i],
// and boot_device FATALs on any override that breaks it -- a non-local ring does not fail loudly, it makes
// the DMA write a well-formed ring into the WRONG bank's HAL profiler region while the mover replays stale
// laps from the configured one (measured 2026-08-24: 6.4M resync words, 79,580 order regressions, a
// 45-minute "zone window" of mixed-boot timestamps).
//
// This replaces the NoC-era stagger (fb[i] != rb[i] pairwise, rings {1,2,3,4,5,6}), whose own justification
// had already been relaxed to "never on a MOVER bank" -- still satisfied: fillers never sit on mover banks.
// Ring traffic now terminates at the filler's own channel, which also takes the ~1.4 GB/s ring write off
// the NoC entirely. Overridable with TT_METAL_PERF_DEBUG_ROLE_RING_BANKS (validated, see above).
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
            out = role_filler_banks();  // rb[i] == fb[i]: the ring must be DMA-reachable from its filler
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
        uint32_t d = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 0u;
        // Detail-1 phases plus the read-split read path overflow the 11,264 B DRISC code region by
        // ~300 B (measured; every other knob combination fits). Degrade loudly rather than let the
        // filler fail to load and produce an empty capture.
        if (d != 0 && env_flag("TT_METAL_PERF_DEBUG_READ_SPLIT")) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] DRISC_ZONE_DETAIL={} with READ_SPLIT does not fit the DRISC code "
                "region; forcing detail 0 (SWEEP/PACE zones only)",
                d);
            d = 0;
        }
        return d;
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
    // Gated on the streaming profiler actually being ENABLED, not merely on a Tracy-enabled build. The HAL
    // reserves this region in every DRAM bank before any device is opened, and it comes straight out of
    // DRAM_UNRESERVED -- i.e. out of what models can allocate. At the 448 MiB default that is ~3 GiB across
    // a 7-bank part, which no build should pay for a profiler it never turns on.
    const char* on = std::getenv("TT_METAL_STREAMING_PROFILER");
    if (on == nullptr || *on == '\0' || *on == '0') {
        return 0;
    }
    if (!role_split()) {
        return 0;
    }
    const uint64_t want_bytes = static_cast<uint64_t>(role_ring_mb()) * 1024ull * 1024ull;
    return static_cast<uint32_t>((want_bytes + 99) / 100);
}

// TT_METAL_STREAMING_PROFILER_TRACY=1: attach the Tracy sink. OFF BY DEFAULT -- the streaming profiler's
// primary consumers are the registered ones (register_consumer / the ops CSV); Tracy is one more consumer
// and an expensive one, so it is opt-in rather than something every capture pays for. (This inverts the
// old TT_METAL_PERF_DEBUG_NO_TRACY.) Drain and decode run EXACTLY the same either way, which also makes
// the off state the sink-cost ablation: if the relay stops host-waiting with Tracy off, the Tracy push is
// provably the bottleneck.
// TT_METAL_PERF_DEBUG_FILL_PCT: target span fill for the drainer's pacing controller, as a percent of a
// span's live capacity (kNumRisc * ring words). 0 disables the loop and leaves the fixed
// TT_METAL_PERF_DEBUG_DRISC_GAP behaviour.
//
// Why it exists: the drainer ships the WHOLE span per core per frame regardless of how much is live, so
// host cost is frames x 10,560 B and the fill ratio decides bytes-per-marker. Sweeping continuously
// against slow producers returns ~37%-full spans, which is why producer stalls got WORSE as producers got
// SLOWER -- ~2x the host bytes for the same payload. Pacing holds the spans full instead.
// TT_METAL_PERF_DEBUG_STAGE_MIN_FILL_PCT: per-core staging fill gate, as a percent of a core's live
// span capacity (kNumRisc * ring words). A filler SKIPS shipping a core whose live words are below this
// (leaving them in the worker's ring to accumulate) unless (a) any of the core's RISCs is past the
// pace high-water valve -- occupancy safety always wins, (b) this is a force-ship sweep (every 8th, the
// latency bound and the sparse-traffic batcher), or (c) the stop-word drain is in progress (teardown
// completeness). 0 disables the gate. Why it exists: frames are fixed 10,560 B regardless of fill, and
// the SUSTAINED word ceiling is mover-frames/us x fill x span words -- skipping half-empty cores is the
// direct way to raise fill without touching the mover (FINDINGS N+65). MEASURED: full-pipeline
// sustained stalls -45% (delay 100: 193k -> 107k), frame bytes -63% (fill 33% -> ~79%); NO_DECODE
// sustained -12-17%. KNOWN TRADE: at high offered rates WITH large ring runway (which is now the
// default, ROLE_RING_MB=448) deferral erodes the worker-ring margin -- delay-25 went clean -> 38k
// stalls. That is why the gate DEFAULTS TO 0 (off): holding a core's words back to ship fuller frames
// costs producer headroom exactly where the default runway is spending it. Set 50 to trade that back
// for ~2x fewer frames per zone on long sustained captures, where host cost dominates.
// TT_METAL_PERF_DEBUG_MOVER_FRAME_GAP: deliberate mover traffic shaping, in DRISC cycles per frame
// just moved. After a PRODUCTIVE sweep that nonetheless KEPT UP (every peer's backlog fit in one batch),
// the mover pauses gap x frames_moved (capped at its 10 us pace ceiling) before the next sweep. A
// backlogged sweep never pauses, so the sustained evacuation ceiling is untouched by construction.
// Why (FINDINGS N+66): with instant acks (NO_DECODE) the mover compresses its sweeps ~2.4x, and that
// burst density inflates the landing tail of the fillers' posted head write-backs -- the write whose
// LANDING releases a blocked producer -- converting onset grazes into stalls (measured 9-16x). Decode-
// paced acks were shaping this traffic by accident; this knob does it on purpose. 0 disables.
uint32_t mover_frame_gap() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_MOVER_FRAME_GAP");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 1600u;
    }();
    return v;
}

uint32_t stage_min_fill_pct() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_STAGE_MIN_FILL_PCT");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 0u;
    }();
    return v;
}

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

uint32_t gap_max_cycles() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_GAP_MAX");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 200000u;
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

// TT_METAL_PERF_DEBUG_ETH_SYNC: measure the device-to-device clock offsets over ethernet at bring-up.
// On by default when there is more than one device; a single-device run has nothing to sync and skips it.
namespace {
bool eth_sync_enabled() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ETH_SYNC");
        return (s == nullptr || *s == '\0') ? true : (*s != '0');
    }();
    return v;
}
uint32_t eth_sync_samples() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ETH_SYNC_SAMPLES");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 256u;
    }();
    return v;
}
uint32_t eth_sync_gap_us() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ETH_SYNC_GAP_US");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 200u;
    }();
    return v;
}
}  // namespace

// Measure one link per device pair, over a SPANNING TREE rooted at the first device: n-1 measurements for
// n devices, not one per link. Every extra edge would cost another ~n_samples * gap of bring-up time and
// tell us something we can already derive, so the tree is what gets measured and any remaining edges stay
// available as a consistency check when we want one.
void PerfDebugProfiler::sync_devices_over_eth(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    if (!eth_sync_enabled()) {
        return;
    }
    std::vector<IDevice*> devices;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (mesh_device->is_local(coord)) {
            devices.push_back(mesh_device->get_device(coord));
        }
    }
    if (devices.size() < 2) {
        return;  // nothing to sync against
    }

    eth_sync::LinkSyncConfig cfg;
    cfg.n_samples = eth_sync_samples();
    cfg.gap_us = eth_sync_gap_us();

    // BFS from the root so every device is reached exactly once, over whichever link the cluster reports
    // first for that pair. Which physical link is picked matters at the nanosecond level (they are not
    // identical lengths), so it is logged.
    std::set<int> visited;
    std::queue<IDevice*> q;
    eth_sync_root_chip_ = static_cast<uint32_t>(devices.front()->id());
    visited.insert(devices.front()->id());
    q.push(devices.front());
    const auto t_start = std::chrono::steady_clock::now();
    while (!q.empty()) {
        IDevice* snd = q.front();
        q.pop();
        for (const CoreCoord& ec : snd->get_active_ethernet_cores(true)) {
            std::tuple<ChipId, CoreCoord> peer;
            try {
                peer = snd->get_connected_ethernet_core(ec);
            } catch (const std::exception&) {
                continue;
            }
            const int peer_id = static_cast<int>(std::get<0>(peer));
            if (visited.count(peer_id) != 0) {
                continue;
            }
            IDevice* rcv = nullptr;
            for (IDevice* d : devices) {
                if (d->id() == peer_id) {
                    rcv = d;
                    break;
                }
            }
            if (rcv == nullptr) {
                continue;  // link leaves this mesh
            }
            visited.insert(peer_id);
            q.push(rcv);

            const auto r = eth_sync::measure_link(snd, ec, rcv, std::get<1>(peer), cfg);
            LinkSync ls;
            ls.sender_chip = static_cast<uint32_t>(snd->id());
            ls.receiver_chip = static_cast<uint32_t>(rcv->id());
            if (r.solution.valid) {
                ls.offset = r.solution.offset;
                ls.ref_mid = r.solution.mid_ref;
                ls.rate = r.solution.rate;
                ls.residual_rms = r.solution.residual_rms;
                ls.rtt_min = r.solution.rtt_min;
                ls.valid = true;
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] eth sync {} -> {} via eth ({},{}): offset {} cycles, rate {:.9f} "
                    "({:+.2f} ppm), rtt_min {} cycles, residual {:.1f} cycles",
                    ls.sender_chip,
                    ls.receiver_chip,
                    ec.x,
                    ec.y,
                    ls.offset,
                    ls.rate,
                    (ls.rate - 1.0) * 1e6,
                    ls.rtt_min,
                    ls.residual_rms);
            } else {
                // Loud, and non-fatal: without this edge the receiver keeps its own host anchor, which is
                // what every device used before this existed. A worse alignment is not a reason to lose
                // the capture.
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] eth sync {} -> {} FAILED (sender {}, receiver {}); device {} "
                    "falls back to its own host anchor",
                    ls.sender_chip,
                    ls.receiver_chip,
                    eth_sync::status_name(r.sender_status),
                    eth_sync::status_name(r.receiver_status),
                    ls.receiver_chip);
            }
            if (ls.valid) {
                eth_sync_parent_edge_[ls.receiver_chip] = link_syncs_.size();
            }
            link_syncs_.push_back(ls);
        }
    }
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] eth sync: {} link(s) measured across {} devices in {} ms ({} samples {} us apart)",
        link_syncs_.size(),
        devices.size(),
        ms,
        cfg.n_samples,
        cfg.gap_us);
}

// Walk chip -> root collecting edges, then apply them forward from the root. Each edge says: at sender
// clock y the receiver reads (ref + offset) + rate * (y - ref); chaining those IS the composition.
bool PerfDebugProfiler::eth_sync_anchor_for(
    uint32_t chip, uint64_t root_clock, uint64_t& chip_clock, double& rate_vs_root) const {
    if (chip == eth_sync_root_chip_) {
        chip_clock = root_clock;
        rate_vs_root = 1.0;
        return true;
    }
    std::vector<size_t> path;
    uint32_t cur = chip;
    while (cur != eth_sync_root_chip_) {
        const auto it = eth_sync_parent_edge_.find(cur);
        if (it == eth_sync_parent_edge_.end()) {
            return false;  // no measured route to the root
        }
        const auto& e = link_syncs_[it->second];
        if (!e.valid) {
            return false;
        }
        path.push_back(it->second);
        cur = e.sender_chip;
        if (path.size() > link_syncs_.size()) {
            return false;  // cycle guard; a tree should make this unreachable
        }
    }
    double y = static_cast<double>(root_clock);
    double rate = 1.0;
    for (auto rit = path.rbegin(); rit != path.rend(); ++rit) {
        const auto& e = link_syncs_[*rit];
        const double ref = static_cast<double>(e.ref_mid);
        y = (ref + static_cast<double>(e.offset)) + e.rate * (y - ref);
        rate *= e.rate;
    }
    chip_clock = y <= 0.0 ? 0u : static_cast<uint64_t>(y);
    rate_vs_root = rate;
    return true;
}

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

    // Device-to-device clock sync FIRST, while the eth cores are still free and no drainer is resident.
    // Both conditions stop holding the moment the loop below starts booting devices.
    sync_devices_over_eth(mesh_device);

    // ---- PASS 2: bring up each device's drainers -------------------------------------------------------
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
        // ONE HOST ANCHOR FOR THE WHOLE MESH. Only the root fits the host; every other device is placed
        // on the root's timeline through the ethernet-measured transform. Before this each device fitted
        // the host independently, so any two devices were separated by the SUM of two independent fits --
        // the dominant term in cross-device alignment. The eth path replaces it with one link measurement
        // whose residual is a couple of cycles.
        uint64_t derived_clock = 0;
        double derived_rate = 1.0;
        bool derived = false;
        if (ctx.chip_id != eth_sync_root_chip_ && root_sync_valid_ &&
            eth_sync_anchor_for(ctx.chip_id, root_dev_at_anchor_, derived_clock, derived_rate)) {
            ctx.clock_synced = true;
            ctx.freq_ghz = root_freq_ghz_ * derived_rate;
            tracy_->AddDevice(
                ctx.chip_id, root_host_anchor_, static_cast<double>(derived_clock), ctx.freq_ghz);
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] Device {} anchored via eth to root {}: device clock {} at the root's "
                "host anchor, {:.6f} GHz ({:+.2f} ppm vs root)",
                ctx.chip_id,
                eth_sync_root_chip_,
                derived_clock,
                ctx.freq_ghz,
                (derived_rate - 1.0) * 1e6);
            derived = true;
        }

        PerfDebugSync sync;
        if (!derived && !ctx.core_virt.empty()) {
            const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
            // LONG BASELINE, deliberately: 100 samples x 500 us spans ~50 ms instead of ~360 us, cutting the
            // fitted-frequency error by the baseline ratio (~140x). This is the ONE frequency every context on
            // this chip will use (see below), so it is worth 50 ms of a 9-12 s device open to measure it well.
            sync = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/500);
        }
        if (derived) {
            // already anchored from the root above
        } else if (sync.valid) {
            ctx.clock_synced = true;
            ctx.freq_ghz = sync.frequency;
            if (ctx.chip_id == eth_sync_root_chip_) {
                root_sync_valid_ = true;
                root_host_anchor_ = sync.host_anchor;
                root_dev_at_anchor_ = sync.device_at_anchor;
                root_freq_ghz_ = sync.frequency;
            }
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
        } else if (!derived) {
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
        // The WORKER-domain anchor and slope this device is rendered with, whichever way it was obtained:
        // its own host fit, or the root's fit carried over the ethernet links.
        const double chip_freq_ghz = ctx.freq_ghz;
        const uint64_t chip_worker_anchor = derived ? derived_clock : sync.device_at_anchor;
        if ((sync.valid || derived) && tracy_ != nullptr) {
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
                    chip_freq_ghz);
                // Log the OFFSET, not just the anchor: it is the board-dependence tell. Microseconds means the
                // part shares an origin and this changed nothing; minutes means it just fixed the capture.
                // Divided by the SHARED slope -- the one this row is actually rendered with, so the reported
                // offset cannot disagree with the mapping in force.
                const double off_ms =
                    (static_cast<double>(ds.device_at_anchor) - static_cast<double>(chip_worker_anchor)) /
                    (chip_freq_ghz > 0.0 ? chip_freq_ghz : 1.0) / 1e6;
                // The core's own fit is REPORTED but never applied: its spread is the diagnostic that identified
                // this error term (measured -79.6 to +71.7 ppm across 9 runs on 2 parts).
                const double fit_ppm =
                    chip_freq_ghz > 0.0 ? (ds.frequency - chip_freq_ghz) / chip_freq_ghz * 1e6 : 0.0;
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] Device {} DRISC {} NOC0 ({},{}) clock sync: frequency={:.6f} GHz "
                    "(SHARED across all contexts); this core's own fit {:.6f} = {:+.1f} ppm, NOT APPLIED, "
                    "device_time_at_anchor={} cycles, offset vs worker anchor {:+.3f} ms",
                    ctx.chip_id,
                    d,
                    nit->second.first,
                    nit->second.second,
                    chip_freq_ghz,
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
            rd.lane_table.reserve(ctx.nl);
            for (uint32_t ci = 0; ci < rd.num_cores; ci++) {
                const auto [vx, vy] = ctx.core_virt[ci];
                uint32_t nx = vx, ny = vy;
                if (auto it = ctx.virt_to_noc0.find((static_cast<uint64_t>(vx) << 32) | vy);
                    it != ctx.virt_to_noc0.end()) {
                    nx = it->second.first;
                    ny = it->second.second;
                }
                auto role = perf_debug::PerfDebugLaneRole::Worker;
                if (ctx.n_worker_cores != 0 && ci >= ctx.n_worker_cores) {
                    switch (ctx.role[ci - ctx.n_worker_cores]) {
                        case kRoleFiller: role = perf_debug::PerfDebugLaneRole::Filler; break;
                        case kRoleMover: role = perf_debug::PerfDebugLaneRole::Mover; break;
                        default: role = perf_debug::PerfDebugLaneRole::Full; break;
                    }
                }
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
            for (uint32_t sk = 0; sk < n_sockets_split(); sk++) {
                if (ctx.sockets[sk] != nullptr) {
                    TT_FATAL(sk == rd.sockets.size(), "sockets must form a contiguous prefix");
                    rd.sockets.push_back(std::move(ctx.sockets[sk]));
                }
            }
        }
        perf_debug::ReceiverConfig rcfg;
        // No load_zone_names hook: names come per-ELF from llrt::ZoneMetaRegistry, which each consumer
        // mirrors lazily (the table GROWS as binaries JIT-load, so a one-shot snapshot would be taken when
        // it holds a fraction of its final size). PRODUCER-STALL and the DRISC self-zones are ordinary
        // zones with ordinary ELF records now -- nothing is registered by hand.
        rcfg.starvation_diagnostic = [this](uint32_t dev, uint32_t sock) {
            DeviceCtx& ctx = devices_[dev];
            for (uint32_t d = 0; d < ctx.n_drisc; d++) {
                if (ctx.sock_of[d] == sock || ctx.role[d] == kRoleFiller) {
                    dump_drainer_state(ctx, d, "receiver-starved");
                }
            }
        };
        receiver_ = std::make_unique<perf_debug::PerfDebugReceiver>(std::move(rcfg), std::move(rdevs));
        if (tracy_push_enabled()) {
            tracy_consumer_ = std::make_unique<perf_debug::PerfDebugTracyConsumer>(tracy_.get());
            // An ordinary paired-contract consumer, like the ops CSV. Zones arrive END-ordered, so
            // the consumer buffers them and reconstructs Tracy's begin/end push order in a teardown
            // flush -- see perf_debug_tracy_consumer.hpp.
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
            (static_cast<uint64_t>(kHRingWords) * 4) / (1024 * 1024),
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

bool PerfDebugProfiler::boot_device(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    DeviceCtx& ctx,
    const distributed::MeshCoordinate& coord) {
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
    // ---- ROLE SPLIT: decide the roster before anything is flipped or launched ----
    //
    // Roles, banks and socket ownership are all resolved up front because the NIU pre-pass below needs every
    // drainer's core in ONE launch (see its comment) and because a mover's compile args reference its
    // filler's L1, so the fillers must be set up first. The default path takes the `else` and is unchanged.
    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    // DEGRADE, don't fail, when the part cannot host the full roster. The 6-DRISC split needs n_fillers() filler
    // banks plus n_sockets_split() host-facing banks, and the fillers' RINGS need banks too; a harvested or smaller
    // part may not have them. Nothing downstream checked this -- the roster was fixed before
    // pick_unused_dram_logical_core() was ever called, so a short part would have indexed past the end of
    // kSafeBanks / the filler-bank list rather than reporting anything.
    //
    // Ladder: 6 (4 fillers + 2 movers) -> 2 full-role drainers -> 1. Each step is a configuration already
    // measured to work, just with a worse knee (§N+34: 1 drainer knee 100, 2 -> 20; §N+40: the split -> 15).
    // Losslessness never depends on the count: fewer drainers means producers stall sooner, not that markers
    // are dropped.
    const uint32_t need_split = n_fillers() + n_sockets_split();
    bool rsplit = role_split() && !tensix_drain;
    if (rsplit && nbanks < need_split) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] role split needs {} DRAM views (4 fillers + 2 movers) but this part has {} "
            "-- falling back to {} full-role drainer(s). Capture stays LOSSLESS; the knee moves in (§N+34).",
            need_split,
            nbanks,
            std::min<uint32_t>(n_sockets_split(), nbanks));
        rsplit = false;
    }
    // Second rung: even the 2-drainer path needs 2 host-facing banks. With one, run a single drainer over the
    // whole grid -- the original shape, knee ~100 (§N+34) but still complete.
    const uint32_t n_full = std::min<uint32_t>(n_sockets_split(), nbanks == 0 ? 1u : nbanks);
    if (!rsplit && n_full < n_sockets_split()) {
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] only {} DRAM view(s) available: running {} drainer(s) over the whole grid.",
            nbanks,
            n_full);
    }
    std::vector<uint32_t> banks;     // DRAM bank hosting DRISC d itself
    std::vector<uint32_t> ringbank;  // DRAM bank holding the ring DRISC d reads/writes (0 when unused)
    if (rsplit) {
        TT_FATAL(
            n_fillers() % n_sockets_split() == 0 && n_fillers() / n_sockets_split() <= kNPeerMax,
            "role-split shape invalid: {} fillers / {} movers",
            n_fillers(),
            n_sockets_split());
        ctx.n_drisc = n_fillers() + n_sockets_split();
        const auto& fb = role_filler_banks();
        const auto& rb = role_ring_banks();
        TT_FATAL(
            fb.size() >= n_fillers() && rb.size() >= n_fillers(),
            "perf-debug role split needs {} filler banks and {} ring banks (got {} and {})",
            n_fillers(),
            n_fillers(),
            fb.size(),
            rb.size());
        for (uint32_t f = 0; f < n_fillers(); f++) {
            ctx.role[f] = kRoleFiller;
            ctx.sock_of[f] = kNoSocket;
            ctx.n_peer[f] = 0;
            // A filler pushes frames with its tile's GDDR DMA engine, which reaches ONLY its own channel --
            // and allocator bank ids are view ids one-for-one. A non-local ring here is not slow, it is
            // silent stale-lap corruption (see role_ring_banks), so refuse it outright.
            TT_FATAL(
                rb[f] == fb[f],
                "perf-debug role split: filler {} on view {} but its ring is on bank {} -- the GDDR DMA push "
                "reaches only the filler's own channel. Fix TT_METAL_PERF_DEBUG_ROLE_RING_BANKS (or unset it).",
                f,
                fb[f],
                rb[f]);
            banks.push_back(fb[f]);
            ringbank.push_back(rb[f]);
        }
        // Mover m drains fillers m, m + n_sockets_split(), ... -- so at 4 fillers, mover 0 takes fillers 0 and 2 and
        // mover 1 takes 1 and 3. STRIDED rather than adjacent on purpose: the fillers own contiguous quarters
        // of the grid in index order, so striding gives each socket one low-half slice and one high-half
        // slice. Adjacent pairing would put both halves of the grid's busy end on one socket if the workload
        // is not uniform across the grid.
        for (uint32_t m = 0; m < n_sockets_split(); m++) {
            const uint32_t d = n_fillers() + m;
            ctx.role[d] = kRoleMover;
            ctx.sock_of[d] = m;
            ctx.n_peer[d] = n_fillers() / n_sockets_split();
            for (uint32_t p = 0; p < ctx.n_peer[d]; p++) {
                ctx.peer_of[d][p] = m + p * n_sockets_split();
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
    // them after the worker grid keeps every worker lane id exactly where it was.
    const uint32_t self_core0 = static_cast<uint32_t>(num_cores);
    const bool self_zones_on = drisc_zones() && drisc_zone_frames() != 0;
    if (self_zones_on) {
        ctx.n_worker_cores = static_cast<uint32_t>(num_cores);
        ctx.nl = (static_cast<uint32_t>(num_cores) + ctx.n_drisc) * kNRisc;
        ctx.core_virt.resize(static_cast<size_t>(num_cores) + ctx.n_drisc);
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
            for (uint32_t f = 0; f < n_fillers(); f++) {
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
            for (uint32_t f = 0; f < n_fillers(); f++) {
                const int32_t off = ctx.device->allocator()->get_bank_offset(BufferType::DRAM, ringbank[f]);
                ctx.dram_addr[f] = static_cast<uint32_t>(static_cast<int64_t>(region_addr) - off);
            }
            // Movers address their peers' rings, so copy the peer-0 pair over for the log line only.
            for (uint32_t d = n_fillers(); d < ctx.n_drisc; d++) {
                ctx.dram_addr[d] = ctx.dram_addr[ctx.peer_of[d][0]];
            }

            std::string ring_desc;
            for (uint32_t f = 0; f < n_fillers(); f++) {
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
                n_fillers(),
                ctx.dram_frames,
                ring_bytes / (1024.0 * 1024.0),
                region_addr,
                ring_desc,
                alloc_banks,
                region_bytes / (1024.0 * 1024.0),
                (static_cast<uint64_t>(region_bytes) * alloc_banks) / (1024 * 1024),
                (static_cast<uint64_t>(region_bytes) * n_fillers()) / (1024 * 1024),
                (static_cast<uint64_t>(region_bytes) * (alloc_banks - n_fillers())) / (1024 * 1024));
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
        // as n_sockets_split(). Off: 2 full-job drainers take halves. On: n_fillers() fillers take n_fillers()-ths,
        // which is the entire point of the change (the knee is the filler's scan over its slice, FINDINGS N+28), so
        // getting this denominator wrong would look like a working build that simply did not improve.
        const uint32_t n_slices = rsplit ? n_fillers() : n_sockets_split();
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
            // n_sockets_split() at 2. pick_unused_dram_logical_core() does NOT know any of this: it reserves
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
        // point of the split. NOTE these arrays are sized n_sockets_split(), not kMaxDrisc -- indexing them by the
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
            // address. Fillers occupy indices [0, n_fillers()) and are set up FIRST, which is why these are
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
                (sync_event_count() != 0 && self_frames_base != 0) ? 1u : 0u,
                // arg 39: the per-core staging fill gate (percent of live span capacity; 0 = off).
                stage_min_fill_pct(),
                // arg 40: mover traffic shaping, cycles per frame moved on a keeping-up sweep (0 = off).
                mover_frame_gap(),
                // args 41-56: mover peer slots 2..5 (xy, hs, ring bank, ring addr each) -- the 6-filler
                // shape's single mover. All zero on the default 4+2 shape and for fillers.
                peer_xy[2],
                peer_hs[2],
                peer_bank[2],
                peer_addr[2],
                peer_xy[3],
                peer_hs[3],
                peer_bank[3],
                peer_addr[3],
                peer_xy[4],
                peer_hs[4],
                peer_bank[4],
                peer_addr[4],
                peer_xy[5],
                peer_hs[5],
                peer_bank[5],
                peer_addr[5]};
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
                    // Peers 0/1 echo into the 64 B done pad (+20/+36); peers 2-5 (6-filler shape) echo
                    // into the results region's live block at out[92 + (pi-2)*4].
                    const uint64_t echo_off =
                        pi < 2 ? (ctx.done_addr[d] - ctx.drisc_l1_base[d]) + 20 + 16 * pi
                               : (ctx.results_addr[d] - ctx.drisc_l1_base[d]) + (92 + (pi - 2) * 4) * 4;
                    cluster.read_core(&echo, sizeof(echo), mv, ctx.drisc_l1_noc[d] + echo_off);
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
                              "TT_METAL_PERF_DEBUG_DRISC_ZONES=1 with DRISC_ZONE_DETAIL=1 plus "
                              "TT_METAL_PERF_DEBUG_NOC_FOOTPRINT is the largest config (~180 B of headroom); "
                              "a u64 division anywhere in the kernel costs a 956 B soft-div."
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

    // ---- DRISC SELF-PROFILING: register the drainer cores' identities ------------------------------------
    //
    // AFTER the bring-up loop, because a drainer's virtual coords only exist once it has been placed. A
    // frame whose SPSC_CORE_XY is missing from the map is skipped whole -- silently.
    if (self_zones_on) {
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            const uint32_t xy = (static_cast<uint32_t>(ctx.drisc_virtual[d].x) & 0xFFFFu) |
                                ((static_cast<uint32_t>(ctx.drisc_virtual[d].y) & 0xFFFFu) << 16);
            ctx.core_of_xy[xy] = self_core0 + d;
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
    // ring, whose head/tail live in the pad this function already reads. After start() the sockets belong
    // to the receiver (single-threaded per instance, so they must not be polled from here), leaving this
    // path only for bring-up-time dumps.
    const uint32_t sk = ctx.sock_of[d];
    const bool have_fifo = sk != kNoSocket && ctx.sockets[sk] != nullptr;
    if (have_fifo) {
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
    if (have_fifo && b[1] == a[1] && !exited && b[2] == 3 && np == 0) {
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
        if (ctx.n_drisc <= n_sockets_split()) {
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
                // The receiver's decode threads are still draining, so the socket keeps emptying while we wait.
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
            } else if (receiver_ != nullptr && ctx.sock_of[d] != kNoSocket) {
                // done follows the drainer's socket barrier, i.e. the host has already read and acked every
                // byte this socket will ever carry -- the stream can retire itself on one final empty check.
                receiver_->notify_producers_done(static_cast<uint32_t>(&ctx - devices_.data()), ctx.sock_of[d]);
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
            const double kCycPerUs = ctx.freq_ghz > 0.0 ? ctx.freq_ghz * 1000.0 : 1350.0;
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
                // Peers 0/1 report at out[49..63] as always; peers 2-5 (6-filler shape) at
                // out[108 + (pi-2)*5] = {moved, ring_hi, tail, max_batch, first-frame word}.
                auto ring_word = [&](uint32_t pi, uint32_t p0, uint32_t p1, uint32_t k) -> uint32_t {
                    return pi == 0 ? res[p0] : (pi == 1 ? res[p1] : res[108 + (pi - 2) * 5 + k]);
                };
                for (uint32_t pi = 0; pi < nring; pi++) {
                    const uint32_t moved = ring_word(pi, 49, 58, 0);
                    const uint32_t hi = ring_word(pi, 50, 63, 1);
                    const uint32_t tail = ring_word(pi, 53, 59, 2);
                    const uint32_t batch = ring_word(pi, 54, 60, 3);
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
                        const uint32_t got = pi == 0 ? res[55] : (pi == 1 ? res[61] : res[108 + (pi - 2) * 5 + 4]);
                        const uint32_t moved = pi == 0 ? res[49] : (pi == 1 ? res[58] : res[108 + (pi - 2) * 5 + 0]);
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
                // Packed wire cost: shipped words plus each frame's 320 B prefix+control (pads ignorable).
                const uint32_t self_bytes = res[87] * 4u + res[64] * 320u;
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
            if (res[133] != 0) {
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] DRISC {} stop-path drain: {} sweeps after stop, {} words recovered",
                    d,
                    res[133],
                    res[134]);
            }
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
        }
        total += core_total;
        worst = std::max(worst, core_total);
        cores_hit += (core_total != 0) ? 1 : 0;
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
        "(worst core {}) [0 = the capture did not perturb the workload]",
        ctx.chip_id,
        total,
        cores_hit,
        n_stall_cores,
        worst);
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
