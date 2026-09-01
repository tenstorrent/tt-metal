// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <deque>
#include <queue>
#include "tools/profiler/perf_debug_profiler.hpp"

#include <algorithm>
#include <numeric>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>  // std::memcpy for the fabric-sync config re-arm
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
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "llrt/metal_soc_descriptor.hpp"
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
#include "hostdevcommon/fabric_router_sync.h"
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
// TT_METAL_PERF_DEBUG_ETH_COVERAGE: sweep the ACTIVE eth cores alongside the workers. On by default --
// an eth core's markers are otherwise written into its L1 ring and never drained, so anything a fabric
// router or the device<->device clock sync records is simply lost. Set 0 to restore worker-only coverage,
// which is also the fallback when the part exposes no eth profiler region.
bool eth_coverage() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ETH_COVERAGE");
        return (s == nullptr || *s == '\0') ? true : (*s != '0');
    }();
    return v;
}
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
namespace {
// Fabric routers own their eth cores outright: on Blackhole an eth core has exactly ONE Metal-programmable
// RISC and the router kernel occupies it, and get_active_ethernet_cores(true) does NOT filter router cores
// on this arch (the FABRIC_ROUTER skip exists only in the non-BH branch). Routers are also already RUNNING
// by the time this profiler starts -- MeshDevice::create calls initialize_fabric_and_dispatch_fw() before
// init_perf_debug_profiler() -- so every eth-sync launch (init tree, closure, resident tracker, close-check)
// must keep off the claimed set or it writes a launch message into a live router. The claim set queried
// here is the post-trim ACTIVE ROUTER set: exactly the channels fabric_init programs routers on, which is
// ALL trained channels under the default configs the fabric unit tests and CCL/models use (planes=nullopt
// -> STRICT claims everything; free channels exist only when num_routing_planes is configured lower).
// Fabric DISABLED skips the query entirely, so the common profiling case pays nothing.
bool eth_core_is_fabric_claimed(uint32_t chip_id, const CoreCoord& logical_eth) {
    if (MetalContext::instance().get_fabric_config() == tt::tt_fabric::FabricConfig::DISABLED) {
        return false;
    }
    try {
        const auto& cp = MetalContext::instance().get_control_plane();
        const auto node = cp.get_fabric_node_id_from_physical_chip_id(chip_id);
        const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(chip_id);
        for (const auto& [chan, dir] : cp.get_active_fabric_eth_channels(node)) {
            const auto core = soc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
            if (CoreCoord(core.x, core.y) == logical_eth) {
                return true;
            }
        }
        return false;
    } catch (const std::exception&) {
        // Fabric is configured but its claims are unreadable. Treat the core as claimed: launching on a
        // live router can wedge the mesh, declining only costs eth sync (host anchors still stand).
        return true;
    }
}
}  // namespace

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
// Reads per sample in sync_device_clock(); the narrowest-bracket one wins. 1 = the old single-read
// behaviour, kept so the effect can be A/B'd on the same box.
uint32_t sync_reads_per_sample() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_SYNC_READS_PER_SAMPLE");
        const uint32_t n = (s != nullptr && *s != '\0') ? (uint32_t)std::strtoul(s, nullptr, 10) : 8u;
        return n == 0 ? 1u : n;
    }();
    return v;
}
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
    // MIN-OF-N PER SAMPLE. The bracket [t0,t1] contains the host->device read cost, which goes through a UMD
    // TLB window (Cluster::read_reg -> driver_->read_from_device_reg). A CONSTANT cost cancels at the midpoint;
    // a VARIABLE one does not -- it lands straight in the fit's jitter, and jitter is what bounds accuracy.
    // Taking the narrowest of N back-to-back attempts keeps the sample that was least disturbed, on the same
    // reasoning as the min-RTT filter in the eth solve: contention only ever ADDS delay, and asymmetrically.
    // The outer 100-sample / spacing_us structure is unchanged -- this only cleans each sample before it enters
    // the regression. N=1 reproduces the previous behaviour exactly, for A/B.
    const uint32_t reads_per_sample = sync_reads_per_sample();
    for (uint32_t i = 0; i < kSamples; i++) {
        int64_t best_rt = 0;
        int64_t best_mid = 0;
        uint64_t best_dev = 0;
        for (uint32_t r = 0; r < reads_per_sample; r++) {
            uint32_t lo = 0, hi = 0;
            const int64_t t0 = tracy::Profiler::GetTime();
            cluster.read_reg(&lo, target, kWallClockL);  // latches H
            cluster.read_reg(&hi, target, kWallClockH);
            const int64_t t1 = tracy::Profiler::GetTime();
            const int64_t rt = t1 - t0;
            if (r == 0 || rt < best_rt) {
                best_rt = rt;
                best_mid = (t0 + t1) / 2;
                best_dev = (static_cast<uint64_t>(hi) << 32) | lo;
            }
        }
        samples.push_back(S{best_mid, best_dev, best_rt});
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
// How many REDUNDANT (non-tree) links to measure for the closure self-check. The spanning tree needs only
// N-1 edges; any extra link between two already-synced devices is a free accuracy check, because its offset
// is predictable from the tree and the two numbers must agree. Costs one link measurement each (~65 ms), so
// it is capped rather than exhaustive. 0 disables.
// Re-run the sync at close and price the session's drift. On by default: it costs one link measurement per
// tree edge (~65 ms) against a device close already measured in seconds, and without it a capture cannot say
// how stale its own alignment became. 0 disables.
// Re-anchor after bring-up rather than living with init-time transforms. 0 disables, restoring the old
// behaviour for comparison.
// TT_METAL_PERF_DEBUG_FORCE_AICLK=<MHz>: pin the AI clock for the whole capture via ARC FORCE_AICLK
// (0x33, arg0 = MHz, 0 releases), released at stop(). MEASURED: while the clock is GOVERNED the inter-chip
// offset decays ~5.4 us/s even though every limit is idle -- aiclk reports a flat 1350, the arbiter permits
// the full clock, TDP sits at 61/150 W. Pinned at the SAME 1350 the decay is 0.017 us/s and saturates. So
// this makes the sync's staleness nearly free, which matters because the sync has to run before fabric
// init while the first zone is not timestamped until 20-30 s of bring-up later.
// CAUTION: pinning likely bypasses the arbiter, so thermal and power protection may not apply. Off by
// default, and released unconditionally at stop().
// DEFAULT ON, as of the measurement below. Unset = AUTO: pin each chip at the aiclk it is already running
// at (the cluster is open by now, so that is the boosted maximum), which adapts to any part instead of
// hardcoding this one's 1350. "0" turns it off; an explicit number pins at that many MHz.
//
// WHY IT IS ON BY DEFAULT. A capture's zone placement decays at decay_rate * (time since the anchor was
// fitted). Governed, that rate is 5.4-7.7 us/s; pinned it is ~0.05 us/s. Measured over a 20.7 s capture on
// bh-31, worst anchor staleness across 4 devices:
//
//   governed   -207 us          (device 0: -122 us)
//   pinned       -2.5 us        (device 0:   -2.2 us, and uniform across all four devices)
//
// ~180x, and it matches the ratio of the two decay rates, so the mechanism is understood rather than
// merely observed. Nothing else moved this number: re-anchoring after bring-up removes the bring-up gap
// but not the decay during the capture, and a faster fit does not help because the error is not fit noise.
//
// THE RISK, stated because it is not hypothetical: FORCE_AICLK (ARC 0x33) very likely bypasses the clock
// arbiter, so thermal and power governance may not apply while it is held. On the runs measured here TDP sat
// at 61-65 W of 150 and therm_trip stayed 0, i.e. nowhere near a limit -- but that is an idle-ish
// microbenchmark, not a hot multi-second model. The clock is released at stop() and the release is verified
// by reading aiclk back. Set TT_METAL_PERF_DEBUG_FORCE_AICLK=0 to disable.
constexpr uint32_t kForceAiclkAuto = 0xFFFFFFFFu;
uint32_t force_aiclk_mhz() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FORCE_AICLK");
        if (s == nullptr || *s == '\0') {
            return kForceAiclkAuto;
        }
        return (uint32_t)std::strtoul(s, nullptr, 10);
    }();
    return v;
}
// WHEN the sync is taken dominates HOW WELL it is taken, by orders of magnitude. Measured on bh-31
// (4x p150b, 2x2 mesh), worst-link error reported by the close-check:
//
//   governed clock, sync 26.5 s before the workload  ->  -144.301 us
//   pinned clock,   sync 32.9 s before the workload  ->    +1.618 us
//   pinned clock,   sync  ~0.5 s before the workload  ->   -0.077 us
//
// Two independent effects, and they multiply. (1) The offset decays at ~5.4 us/s while the clock is
// GOVERNED and ~0.05 us/s while it is PINNED (TT_METAL_PERF_DEBUG_FORCE_AICLK) -- see the FORCE_AICLK
// knob for the evidence that this is governance, not any power/thermal limit. (2) Whatever the decay
// rate, it is multiplied by the gap between the sync and the first zone, and on a 4-device mesh that
// gap is the ~26-33 s of drainer bring-up, against a workload of tens of ms.
//
// DEFAULT FALSE ON THIS BRANCH. Re-anchoring after bring-up is measurably better, but it is only
// reachable because fabric is disabled here: metal runs initialize_fabric_and_dispatch_fw() before
// init_perf_debug_profiler(), and with fabric up the routers own the eth cores, so a post-bring-up
// sync has no core to run on. The pre-fabric sync is therefore the only one that generalises, and
// PINNING THE CLOCK is what makes its staleness survivable (+1.6 us across 33 s instead of -144 us).
// Host-only late re-anchor: refit every device's host<->device anchor AFTER bring-up. Unlike the eth
// re-anchor this needs no eth core and launches nothing, so it is legal with fabric up -- which is the
// case that matters, since metal brings fabric up before this profiler starts.
bool host_reanchor() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_HOST_REANCHOR");
        return (s == nullptr || *s == '\0') ? true : (*s != '0');
    }();
    return v;
}
bool eth_sync_late() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ETH_SYNC_LATE");
        return (s == nullptr || *s == '\0') ? false : (*s != '0');
    }();
    return v;
}
bool eth_sync_close_check() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ETH_SYNC_CLOSE");
        return (s == nullptr || *s == '\0') ? true : (*s != '0');
    }();
    return v;
}
uint32_t eth_sync_closure_links() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ETH_SYNC_CLOSURE");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 4u;
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
// Draw the eth sync's RAW round trips onto the two eth cores' own Tracy lanes: the sender's [t0,t2] as a
// zone, the peer's t1 as a marker. This is the one alignment indicator in the trace that is NOT derived
// from the anchors. A marker derived FROM the fit lines up by construction and cannot contradict a wrong
// offset, which is why an earlier per-core SYNC_ANCHOR marker was dropped rather than kept as decoration.
// These can: each is a measurement on its own device's clock, rendered
// through that device's own anchor, so if the anchors are right the peer's t1 lands INSIDE the sender's
// zone. A t1 outside its zone is a causality violation, and therefore proof the alignment is wrong.
//
// Scale warning for whoever reads this in the GUI: the round trip is ~858 cycles (~0.64 us) against a
// multi-millisecond capture, so it is sub-pixel until you zoom to microseconds. It is a check you run, not
// one you notice.
// Re-measure every tree link at close and compare with what the init fit predicted for that instant.
//
// What this prices that nothing else does: the init sync reports how well the two clocks agreed AT THE
// MOMENT IT RAN. Every zone after that is placed by EXTRAPOLATING that fit, and the extrapolation is only
// as good as the fitted rate -- 1 ppm of rate error is 1 us per second of session, so a long capture can be
// far more misaligned at its end than any init-time statistic suggests. Re-running the same links at close
// turns that into a measurement: the gap between the predicted offset and the measured one IS the session's
// accumulated error, and it is the honest bound on a zone placed near the end of the capture.
//
// Deliberately re-measures the SAME eth cores in the SAME direction. Physical links differ at the
// nanosecond level, so a different link would show up as drift that never happened.
// WHY THIS EXISTS. The anchor a device's zones are rendered with is fitted at profiler init, but no zone is
// timestamped until bring-up finishes -- ~26-33 s later on a 4-device mesh, against workloads of tens of ms.
// The offset decays at ~5.4 us/s while the clock is governed, so the capture inherits ~140 us of error that
// has nothing to do with how well the sync was measured. Re-fitting each device here collapses the elapsed
// term to the ~0.1 s between this call and the first zone.
//
// HOST-ONLY, deliberately. The eth re-anchor (reanchor_after_boot) measures better but cannot run once fabric
// owns the eth cores, and fabric comes up before this profiler starts. A host fit needs no core and launches
// nothing, so it is the only re-anchor that generalises to a real multi-device run.
//
// Each device is refitted INDEPENDENTLY rather than derived from the root through the eth tree: those link
// transforms were measured at init and are exactly as stale as the anchors being replaced.
void PerfDebugProfiler::host_reanchor_after_boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    if (!host_reanchor() || devices_.empty() || tracy_ == nullptr) {
        return;
    }
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    uint32_t done = 0;
    for (auto& ctx : devices_) {
        if (ctx.core_virt.empty()) {
            continue;
        }
        // ROOT ONLY for the HOST FIT. Non-root devices are composed from it below, off measured
        // device<->device links -- see the loop after this one.
        //
        // This function used to refit EVERY device independently, and that is where the
        // four disagreeing anchors came from -- it runs after start(), so it overwrote whatever
        // start() had composed. Measured cost of those independent fits, as the gap between the two
        // ends of a link where the same instant is drawn twice: 3.5 us (1->2), 7.1 us (0->1),
        // 13.7 us (0->3), constant across the run, against a device<->device sync closing to
        // -37..-57 ns. One host fit at the root, everything else composed off it -- never a fit per
        // device.
        if (ctx.chip_id != eth_sync_root_chip_) {
            continue;
        }
        const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
        const PerfDebugSync s = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/500);
        if (!s.valid) {
            // Keep the init anchor: stale placement still renders, a missing one does not.
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] HOST RE-ANCHOR: device {} refit failed; keeping its init-time anchor",
                ctx.chip_id);
            continue;
        }
        ctx.clock_synced = true;
        ctx.freq_ghz = s.frequency;
        tracy_->AddDevice(ctx.chip_id, s.host_anchor, static_cast<double>(s.device_at_anchor), s.frequency);
        ctx.anchor_host = s.host_anchor;
        ctx.anchor_dev = s.device_at_anchor;
        ctx.anchor_valid = true;
        if (ctx.chip_id == eth_sync_root_chip_) {
            root_sync_valid_ = true;
            root_host_anchor_ = s.host_anchor;
            root_dev_at_anchor_ = s.device_at_anchor;
            root_freq_ghz_ = s.frequency;
        }
        ++done;
    }
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] HOST RE-ANCHOR: refitted {} device anchor(s) after bring-up (host MMIO only, "
        "no eth core, no program launch -- legal with fabric up)",
        done);
}

// Scores the anchors THEMSELVES, with no eth and no composition: refit each device now, and ask how far the
// anchor still being used to render it has drifted from that fresh fit. This is the quantity that misplaces a
// zone, and unlike the eth close-check it stays measurable when fabric owns the eth cores.
// ISOLATED TEST OF THE TWO PATHS AGAINST EACH OTHER.
//
// The eth link measures (eth_B - eth_A). Host fits of a DRISC core on each chip give (drisc_B - drisc_A).
// Those are different clock DOMAINS, so their difference carries a constant intra-chip gap G -- comparing
// them once tells you nothing (an earlier version of the cross-check did exactly that against worker cores
// and reported 75-238 ms of "error" that was really G).
//
// Measuring the pair REPEATEDLY cancels G: it is constant, so any TREND in the difference is one path
// drifting against the other, and the spread is the combined noise of both. That is the isolated question
// -- how well does a freshly-taken host anchor track the eth measurement -- with no staleness confound,
// because both sides are re-measured every round.
//
// Host fits use spacing 0 (back-to-back, ~360 us baseline) so they are taken as close in time to the eth
// measurement as possible, and at the highest rate the read path allows.
// PERIODIC RE-ANCHOR. A one-shot anchor is wrong by decay * (time since it was fitted): at the 5-8 us/s
// measured on this part that is tens of us across a 20 s capture, which is what the close-time ANCHOR
// STALENESS numbers were reporting. Re-fitting cannot move a Tracy context anchor (baked at creation), so
// the fresh fit is published as a per-device CORRECTION that the consumer adds to each zone timestamp.
//
// Rate is the whole point: error is bounded by decay / rate. At 7.7 us/s, 1 Hz bounds it to ~8 us, 10 Hz to
// ~0.8 us, 100 Hz to ~80 ns -- the floor being the ~50 ns two independent paths agree to (PATH TRACKING).
//
// Host MMIO only: no eth core, no program launch, so this is legal while fabric is up and while a capture is
// in flight. OFF by default -- it adds MMIO traffic concurrent with the drainers, which is exactly the
// contention the knee work is sensitive to, so it should be measured before it is trusted.
void PerfDebugProfiler::start_drift_corrector(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const char* s = std::getenv("TT_METAL_PERF_DEBUG_DRIFT_HZ");
    if (s == nullptr || *s == '\0' || *s == '0' || devices_.empty()) {
        return;
    }
    if (eth_track_ != nullptr) {
        // Two writers with different bases (host fit vs eth link) racing one ratcheted value would mix
        // reference frames; the eth tracker is the more direct measurement, so it wins.
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] DRIFT CORRECTOR disabled: the ETH TRACKER already owns the per-device "
            "corrections");
        return;
    }
    const double hz = std::strtod(s, nullptr);
    if (!(hz > 0.0)) {
        return;
    }
    const auto period = std::chrono::microseconds(static_cast<int64_t>(1e6 / hz));
    const auto context_id = mesh_device->impl().get_context_id();
    drift_stop_.store(false);
    drift_thread_ = std::thread([this, context_id, period]() {
        auto& cluster = MetalContext::instance(context_id).get_cluster();
#ifdef TRACY_ENABLE
        const double npt_raw = TracyGetTimerMul();
#else
        const double npt_raw = 1.0;
#endif
        const double npt = npt_raw > 0.0 ? npt_raw : 1.0;
        while (!drift_stop_.load(std::memory_order_relaxed)) {
            for (const auto& ctx : devices_) {
                if (drift_stop_.load(std::memory_order_relaxed)) {
                    break;
                }
                if (!ctx.anchor_valid || ctx.core_virt.empty() || ctx.freq_ghz <= 0.0) {
                    continue;
                }
                const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
                // spacing 0: back-to-back samples, the shortest fit the read path allows, so the correction
                // is as close in time to the zones it will be applied to as possible.
                const PerfDebugSync s2 = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/0);
                if (!s2.valid) {
                    continue;
                }
                const double dt_ns = static_cast<double>(s2.host_anchor - ctx.anchor_host) * npt;
                const double predicted = static_cast<double>(ctx.anchor_dev) + dt_ns * ctx.freq_ghz;
                // correction = predicted - actual: what the clock has LOST, added back onto timestamps.
                const double corr = predicted - static_cast<double>(s2.device_at_anchor);
                perf_debug::set_zone_ts_correction(ctx.chip_id, static_cast<int64_t>(corr));
            }
            std::this_thread::sleep_for(period);
        }
    });
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] DRIFT CORRECTOR: re-anchoring every {:.1f} ms (host MMIO only); zone "
        "placement error should be bounded by decay/rate rather than by capture length",
        1000.0 / hz);
}

void PerfDebugProfiler::stop_drift_corrector() {
    drift_stop_.store(true);
    if (drift_thread_.joinable()) {
        drift_thread_.join();
    }
}

void PerfDebugProfiler::cross_path_tracking_test() {
    const char* on = std::getenv("TT_METAL_PERF_DEBUG_TRACK_TEST");
    if (on == nullptr || *on == '\0' || *on == '0' || link_syncs_.empty()) {
        return;
    }
    const uint32_t rounds = static_cast<uint32_t>(std::strtoul(on, nullptr, 10));
    auto& cluster = MetalContext::instance().get_cluster();
    eth_sync::LinkSyncConfig cfg;
    cfg.n_samples = eth_sync_samples();
    cfg.gap_us = eth_sync_gap_us();
#ifdef TRACY_ENABLE
    const double npt_raw = TracyGetTimerMul();
#else
    const double npt_raw = 1.0;
#endif
    const double npt = npt_raw > 0.0 ? npt_raw : 1.0;

    for (const auto& e : link_syncs_) {
        if (!e.valid || e.snd_dev == nullptr || e.rcv_dev == nullptr) {
            continue;
        }
        if (eth_core_is_fabric_claimed(e.sender_chip, e.snd_eth) ||
            eth_core_is_fabric_claimed(e.receiver_chip, e.rcv_eth)) {
            continue;  // the link's channel is fabric-owned by now; see the close-check skip
        }
        // DRISC core on each end. Deliberately a DRAM core: it is where our drainers live, its coords are
        // virtual (so read_reg resolves them, unlike the LOGICAL eth coords that aborted an earlier attempt),
        // and it is the domain our drainer zones are timestamped in.
        const DeviceCtx* a = nullptr;
        const DeviceCtx* b = nullptr;
        for (const auto& d : devices_) {
            if (d.chip_id == e.sender_chip) { a = &d; }
            if (d.chip_id == e.receiver_chip) { b = &d; }
        }
        if (a == nullptr || b == nullptr || a->n_drisc == 0 || b->n_drisc == 0) {
            continue;
        }
        std::vector<double> deltas;
        std::vector<double> secs;
        int64_t t_first = 0;
        for (uint32_t r = 0; r < rounds; r++) {
            const auto lr = eth_sync::measure_link(e.snd_dev, e.snd_eth, e.rcv_dev, e.rcv_eth, cfg);
            if (!lr.solution.valid || lr.solution.residual_rms > 20.0) {
                continue;
            }
            const PerfDebugSync ha = sync_device_clock(cluster, a->chip_id, a->drisc_virtual[0], 0);
            const PerfDebugSync hb = sync_device_clock(cluster, b->chip_id, b->drisc_virtual[0], 0);
            if (!ha.valid || !hb.valid) {
                continue;
            }
            // Host-derived offset, evaluated at the midpoint of the two fits so neither is extrapolated far.
            const double t_mid = 0.5 * (static_cast<double>(ha.host_anchor) + static_cast<double>(hb.host_anchor));
            const double devA = static_cast<double>(ha.device_at_anchor) +
                                (t_mid - static_cast<double>(ha.host_anchor)) * npt * ha.frequency;
            const double devB = static_cast<double>(hb.device_at_anchor) +
                                (t_mid - static_cast<double>(hb.host_anchor)) * npt * hb.frequency;
            const double host_offset = devB - devA;
            const double delta = static_cast<double>(lr.solution.offset) - host_offset;
            if (t_first == 0) {
                t_first = ha.host_anchor;
            }
            deltas.push_back(delta);
            secs.push_back(static_cast<double>(ha.host_anchor - t_first) * npt / 1e9);
        }
        if (deltas.size() < 2) {
            continue;
        }
        // Spread and trend. The MEAN is meaningless (it is the domain gap G); the spread is the combined
        // noise of the two paths and the trend is their relative drift.
        double mn = deltas[0], mx = deltas[0], sum = 0.0;
        for (double d : deltas) {
            mn = std::min(mn, d);
            mx = std::max(mx, d);
            sum += d;
        }
        const double mean = sum / static_cast<double>(deltas.size());
        double num = 0.0, den = 0.0;
        const double smean = std::accumulate(secs.begin(), secs.end(), 0.0) / static_cast<double>(secs.size());
        for (size_t i = 0; i < deltas.size(); i++) {
            num += (secs[i] - smean) * (deltas[i] - mean);
            den += (secs[i] - smean) * (secs[i] - smean);
        }
        const double slope_cyc_per_s = den > 0.0 ? num / den : 0.0;
        const double f = root_freq_ghz_ > 0.0 ? root_freq_ghz_ : 1.35;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] PATH TRACKING {} -> {}: {} rounds over {:.1f} s | spread {:.3f} us "
            "(min-max, = combined noise of BOTH paths) | drift {:+.3f} us/s (= relative divergence) | "
            "mean {:+.0f} cycles is the constant eth-vs-DRISC domain gap and carries no information",
            e.sender_chip,
            e.receiver_chip,
            deltas.size(),
            secs.back(),
            (mx - mn) / f / 1000.0,
            slope_cyc_per_s / f / 1000.0,
            mean);
    }
}

void PerfDebugProfiler::check_anchor_staleness_at_close() {
    if (!eth_sync_close_check() || devices_.empty()) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    double worst_us = 0.0;
    uint32_t worst_chip = 0;
    uint32_t checked = 0;
    for (auto& ctx : devices_) {
        if (!ctx.anchor_valid || ctx.core_virt.empty() || ctx.freq_ghz <= 0.0) {
            continue;
        }
        const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
        const PerfDebugSync s = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/0);
        if (!s.valid) {
            continue;
        }
        // Where the rendering anchor says this device's clock should be at the instant of the fresh fit.
        // Composed as a difference so the precision stays in the answer rather than in the epoch.
        //
        // UNITS TRAP, and it cost a run: sync_device_clock() returns host_anchor in host TICKS (raw
        // Profiler::GetTime()) but frequency in cycles per NANOSECOND -- it divides the fitted slope by
        // TracyGetTimerMul() before storing it. Multiplying a tick delta by a per-ns rate overstates the
        // predicted advance by the TSC rate (~2.3x here), which showed up as a bogus 43% clock deficit that
        // scaled with elapsed time in both A/B arms identically -- the signature of a scale error, not drift.
#ifdef TRACY_ENABLE
        const double ns_per_tick = TracyGetTimerMul();
#else
        const double ns_per_tick = 1.0;
#endif
        const double dt_ns = static_cast<double>(s.host_anchor - ctx.anchor_host) *
                             (ns_per_tick > 0.0 ? ns_per_tick : 1.0);
        const double predicted = static_cast<double>(ctx.anchor_dev) + dt_ns * ctx.freq_ghz;
        const double err_cycles = static_cast<double>(s.device_at_anchor) - predicted;
        const double err_us = err_cycles / ctx.freq_ghz / 1000.0;
        // What Tracy actually renders is raw + correction, so the error that SURVIVES is err + correction.
        // With the corrector running, that is the drift accumulated since its last refresh -- i.e. bounded by
        // the refresh period rather than by the capture length. With it off the correction is 0 and the two
        // numbers agree, which is the check reducing to its old self.
        const int64_t applied = perf_debug::get_zone_ts_correction(ctx.chip_id);
        const double residual_us = (err_cycles + static_cast<double>(applied)) / ctx.freq_ghz / 1000.0;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] ANCHOR STALENESS device {}: anchor has drifted {:+.3f} us after {:.1f} s "
            "({:+.0f} cycles); live correction {:+.0f} cycles; RENDERED error {:+.3f} us <- this is what a "
            "zone's placement is actually off by",
            ctx.chip_id,
            err_us,
            dt_ns / 1e9,
            err_cycles,
            static_cast<double>(applied),
            residual_us);
        if (std::abs(residual_us) > std::abs(worst_us)) {
            worst_us = residual_us;
            worst_chip = ctx.chip_id;
        }
        ++checked;
    }
    if (checked != 0) {
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] ANCHOR STALENESS: worst RENDERED error {:+.3f} us on device {} over {} "
            "device(s) -- this bounds where a zone near the END of the capture is placed",
            worst_us,
            worst_chip,
            checked);
    }
}

namespace {
uint32_t eth_track_hz() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ETH_TRACK_HZ");
        return (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 0u;
    }();
    return v;
}
}  // namespace

// One resident pair per spanning-tree link, plus the tracking thread. Fit parameters are COPIES of the
// init link_syncs_ entries, not references: the container may be rebuilt by a late re-anchor, and the
// tracker's deviation basis must be the exact fits the close-check will later predict from.
struct PerfDebugProfiler::EthTrackState {
    struct Edge {
        int64_t off0 = 0;
        uint64_t ref0 = 0;
        double rate0 = 1.0;
        uint32_t snd_chip = 0;
        uint32_t rcv_chip = 0;
        eth_sync::ResidentLink rl;
        int64_t corr_edge = 0;  // (init-fit linear) - (measured), cycles: what the child LOST vs the model
        bool have = false;
        uint32_t fails = 0;
        // ---- Tracy ribbon: one record per successful round, drawn at teardown on the SECOND-channel
        // lanes (emission is deferred so nothing pushes into the handler while the live consumer owns it).
        // The three receiver markers make the story same-lane, i.e. independent of any anchor: MEASURED is
        // the raw echo, PRED_INIT_FIT walks away at the fixed-rate error, PRED_TRACKED uses the correction
        // published BEFORE its round -- causal, so the estimator never grades its own round.
        struct DrawRec {
            uint64_t t0 = 0, t1 = 0, t2 = 0;
            int64_t pred_init = 0, pred_tracked = 0;
            // The sender's round-trip window converted onto the receiver's clock by the CAUSAL tracked
            // prediction -- drawn on the receiver lane so the echo-inside-window check is anchor-free.
            int64_t conv_start = 0, conv_end = 0;
        };
        std::vector<DrawRec> draws;
        uint32_t snd_nx = 0, snd_ny = 0, rcv_nx = 0, rcv_ny = 0;  // second-channel NOC0 coords
        bool lanes_ok = false;  // both second-channel tiles anchored -> the ribbon can render
    };
    std::vector<Edge> edges;
    std::unordered_map<uint32_t, size_t> parent_edge;  // chip -> index into edges
    uint32_t root = 0;
    std::thread th;
    std::atomic<bool> stop{false};
    int64_t baseline = 0;  // common additive term keeping ratcheted values non-decreasing; cancels in diffs
};

void PerfDebugProfiler::start_eth_tracker(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const uint32_t hz = eth_track_hz();
    if (hz == 0 || eth_track_ != nullptr || link_syncs_.empty() || eth_sync_parent_edge_.empty()) {
        return;
    }
    if (eth_sync_late()) {
        // The late re-anchor DISCARDS and re-measures the tree the tracker would deviate against; running
        // both would score corrections built on fits the close-check no longer uses. Unsupported combo.
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] ETH TRACKER disabled: TT_METAL_PERF_DEBUG_ETH_SYNC_LATE rebuilds the "
            "fits the tracker deviates against");
        return;
    }
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    auto* st = new EthTrackState();
    st->root = eth_sync_root_chip_;
    for (const auto& [chip, idx] : eth_sync_parent_edge_) {
        const auto& e = link_syncs_[idx];
        if (!e.valid || e.snd_dev == nullptr || e.rcv_dev == nullptr) {
            continue;
        }
        // A SECOND channel on the same pair: the init sync used e.snd_eth and the close-check will use it
        // again, so the resident pair takes any other trained channel toward the same peer.
        CoreCoord snd2{};
        CoreCoord rcv2{};
        bool found = false;
        for (const auto& [peer, cores] : cluster.get_ethernet_cores_grouped_by_connected_chips(e.snd_dev->id())) {
            if (static_cast<uint32_t>(peer) != e.receiver_chip) {
                continue;
            }
            for (const auto& c : cores) {
                if (c == e.snd_eth) {
                    continue;
                }
                auto [pid, pcore] = e.snd_dev->get_connected_ethernet_core(c);
                if (static_cast<uint32_t>(pid) != e.receiver_chip) {
                    continue;
                }
                if (eth_core_is_fabric_claimed(e.sender_chip, c) ||
                    eth_core_is_fabric_claimed(e.receiver_chip, pcore)) {
                    continue;  // a fabric router owns this channel; keep looking for a free one
                }
                snd2 = c;
                rcv2 = pcore;
                found = true;
                break;
            }
            break;
        }
        if (!found) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] ETH TRACKER: no second eth channel on {} -> {}; this link is not "
                "tracked and its subtree keeps correction 0",
                e.sender_chip,
                e.receiver_chip);
            continue;
        }
        eth_sync::LinkSyncConfig rcfg;
        rcfg.n_samples = 16;  // measured: ~7 ns offset noise back-to-back; rate comes from ROUND deltas
        rcfg.gap_us = 0;
        rcfg.host_timeout_ms = 1000;
        EthTrackState::Edge ed;
        ed.off0 = e.offset;
        ed.ref0 = e.ref_mid;
        ed.rate0 = e.rate;
        ed.snd_chip = e.sender_chip;
        ed.rcv_chip = e.receiver_chip;
        ed.rl = eth_sync::start_resident_link(e.snd_dev, snd2, e.rcv_dev, rcv2, rcfg);
        // RIBBON LANES: eth tiles share a rate but not an origin (an eth tile on this box banked ~53 min
        // more than its chip's worker), so without their own per-core anchors the second-channel rows
        // would inherit the chip anchor and render minutes off the timeline. Anchored HERE, before the
        // first round, by the same bracket-read helper the init pass runs on the first-channel tiles.
        // devices_ is not populated yet at this point in start(), so the frequency comes from the helper's
        // own per-core fit, exactly where root_freq_ghz_ itself comes from.
        if (tracy_ != nullptr) {
            const CoreCoord sn2 = cluster.get_physical_coordinate_from_logical_coordinates(
                e.snd_dev->id(), snd2, CoreType::ETH, /*no_warn=*/true);
            const CoreCoord rn2 = cluster.get_physical_coordinate_from_logical_coordinates(
                e.rcv_dev->id(), rcv2, CoreType::ETH, /*no_warn=*/true);
            ed.snd_nx = static_cast<uint32_t>(sn2.x);
            ed.snd_ny = static_cast<uint32_t>(sn2.y);
            ed.rcv_nx = static_cast<uint32_t>(rn2.x);
            ed.rcv_ny = static_cast<uint32_t>(rn2.y);
            const PerfDebugSync sa2 = sync_device_clock(cluster, ed.snd_chip, ed.rl.snd_v);
            const PerfDebugSync ra2 = sync_device_clock(cluster, ed.rcv_chip, ed.rl.rcv_v);
            if (sa2.valid && ra2.valid && sa2.frequency > 0.0 && ra2.frequency > 0.0) {
                tracy_->AddCore(
                    ed.snd_chip,
                    ed.snd_nx,
                    ed.snd_ny,
                    sa2.host_anchor,
                    static_cast<double>(sa2.device_at_anchor),
                    sa2.frequency);
                tracy_->AddCore(
                    ed.rcv_chip,
                    ed.rcv_nx,
                    ed.rcv_ny,
                    ra2.host_anchor,
                    static_cast<double>(ra2.device_at_anchor),
                    ra2.frequency);
                tracy_->RegisterEthCore(ed.snd_chip, ed.snd_nx, ed.snd_ny);
                tracy_->RegisterEthCore(ed.rcv_chip, ed.rcv_nx, ed.rcv_ny);
                ed.lanes_ok = true;
            }
        }
        st->edges.push_back(std::move(ed));
        st->parent_edge[chip] = st->edges.size() - 1;
    }
    if (st->edges.empty()) {
        delete st;
        return;
    }
    eth_track_ = st;
    const auto period = std::chrono::microseconds(static_cast<int64_t>(1e6 / hz));
    st->th = std::thread([st, period]() {
        while (!st->stop.load(std::memory_order_relaxed)) {
            for (auto& ed : st->edges) {
                if (st->stop.load(std::memory_order_relaxed)) {
                    break;
                }
                auto r = eth_sync::resident_round(ed.rl);
                if (!r.solution.valid) {
                    if (++ed.fails == 1 || (ed.fails % 200) == 0) {
                        log_warning(
                            tt::LogMetal,
                            "[perf-debug profiler] ETH TRACKER round INVALID on {} -> {} (x{}): status {}/{}, "
                            "samples {}/{}",
                            ed.snd_chip,
                            ed.rcv_chip,
                            ed.fails,
                            eth_sync::status_name(r.sender_status),
                            eth_sync::status_name(r.receiver_status),
                            r.sender_samples,
                            r.receiver_samples);
                    }
                    continue;
                }
                const double dt = static_cast<double>(static_cast<int64_t>(r.solution.mid_ref - ed.ref0));
                const double linear = static_cast<double>(ed.off0) + dt * (ed.rate0 - 1.0);
                const int64_t corr_prev = ed.corr_edge;  // as published BEFORE this round: the causal estimate
                ed.corr_edge = static_cast<int64_t>(std::llround(linear - static_cast<double>(r.solution.offset)));
                ed.have = true;
                if (ed.lanes_ok && !r.trips.empty() && ed.draws.size() < 200000) {
                    // The min-RTT trip is the one the solve trusted; render that one, one per round.
                    size_t bi = 0;
                    uint64_t best = ~0ull;
                    for (size_t i = 0; i < r.trips.size(); i++) {
                        const uint64_t w = r.trips[i].t2 - r.trips[i].t0;
                        if (w < best) {
                            best = w;
                            bi = i;
                        }
                    }
                    const auto& tr = r.trips[bi];
                    const uint64_t mid = (tr.t0 + tr.t2) / 2;
                    const double dm = static_cast<double>(static_cast<int64_t>(mid - ed.ref0));
                    EthTrackState::Edge::DrawRec dr;
                    dr.t0 = tr.t0;
                    dr.t1 = tr.t1;
                    dr.t2 = tr.t2;
                    dr.pred_init = static_cast<int64_t>(mid) + ed.off0 +
                                   static_cast<int64_t>(std::llround(dm * (ed.rate0 - 1.0)));
                    dr.pred_tracked = dr.pred_init - corr_prev;
                    const double d0 = static_cast<double>(static_cast<int64_t>(tr.t0 - ed.ref0));
                    const double d2 = static_cast<double>(static_cast<int64_t>(tr.t2 - ed.ref0));
                    dr.conv_start = static_cast<int64_t>(tr.t0) + ed.off0 +
                                    static_cast<int64_t>(std::llround(d0 * (ed.rate0 - 1.0))) - corr_prev;
                    dr.conv_end = static_cast<int64_t>(tr.t2) + ed.off0 +
                                  static_cast<int64_t>(std::llround(d2 * (ed.rate0 - 1.0))) - corr_prev;
                    ed.draws.push_back(dr);
                }
            }
            // Compose per-device RELATIVE corrections down the tree (root = 0). Unlike the host corrector''s
            // device-vs-host values, RELATIVE deviations wander in BOTH directions, and the setter''s ratchet
            // (grow-only, required for Tracy''s per-lane ordering) would freeze them at their maximum. So a
            // COMMON baseline B is added to every device including the root: chosen each tick so every
            // published value is non-decreasing, it rides through the ratchet unharmed and cancels out of
            // every difference -- and differences are all that cross-device alignment and the close-check
            // prediction ever read.
            bool all_have = true;
            std::unordered_map<uint32_t, int64_t> rel;
            rel[st->root] = 0;
            for (const auto& [chip, _] : st->parent_edge) {
                int64_t corr = 0;
                uint32_t cur = chip;
                bool ok = true;
                int hops = 0;
                while (cur != st->root && hops++ < 8) {
                    auto it = st->parent_edge.find(cur);
                    if (it == st->parent_edge.end() || !st->edges[it->second].have) {
                        ok = false;
                        break;
                    }
                    corr += st->edges[it->second].corr_edge;
                    cur = st->edges[it->second].snd_chip;
                }
                if (ok) {
                    rel[chip] = corr;
                } else {
                    all_have = false;
                }
            }
            if (all_have) {
                for (const auto& [chip, r] : rel) {
                    const int64_t prev = perf_debug::get_zone_ts_correction(chip);
                    const int64_t need = prev - r;
                    if (need > st->baseline) {
                        st->baseline = need;
                    }
                }
                for (const auto& [chip, r] : rel) {
                    perf_debug::set_zone_ts_correction(chip, r + st->baseline);
                }
            }
            std::this_thread::sleep_for(period);
        }
    });
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] ETH TRACKER: {} resident link(s) at {} Hz on second channels -- per-device "
        "corrections now come from the links themselves, so the close-check scores the tracked offset "
        "instead of a one-shot extrapolation",
        st->edges.size(),
        hz);
}

void PerfDebugProfiler::stop_eth_tracker() {
    if (eth_track_ == nullptr) {
        return;
    }
    eth_track_->stop.store(true);
    if (eth_track_->th.joinable()) {
        eth_track_->th.join();
    }
    for (auto& ed : eth_track_->edges) {
        eth_sync::stop_resident_link(ed.rl);
    }
    // ---- DRAW THE TRACKING RIBBON ----
    // Deferred to here on purpose: the receiver and its consumers shut down before this call, so nothing
    // else is pushing into the handler. Timestamps are raw device cycles, exactly like the init/close eth
    // lanes; each lane's events are sorted before emission (the tracked marker can sit on either side of
    // the echo, and Tracy wants non-decreasing arrival per lane).
    if (tracy_ != nullptr) {
        const double freq = root_freq_ghz_ > 0.0 ? root_freq_ghz_ : 1.35;
        static constexpr std::string_view kTrkRtt = "ETH_TRACK_RTT";
        static constexpr std::string_view kTrkEcho = "ETH_TRACK_ECHO_MEASURED";
        static constexpr std::string_view kTrkInit = "ETH_TRACK_PRED_INIT_FIT";
        static constexpr std::string_view kTrkLive = "ETH_TRACK_PRED_TRACKED";
        static constexpr std::string_view kTrkWin = "ETH_TRACK_RTT_TRACKED";
        for (const auto& ed : eth_track_->edges) {
            if (!ed.lanes_ok || ed.draws.empty()) {
                continue;
            }
            // Rounds are sequential and the sender clock is monotonic, so zone order is already correct.
            for (const auto& dr : ed.draws) {
                perf_debug::WorkerZonePacket z;
                z.chip_id = ed.snd_chip;
                z.core_virtual_x = ed.snd_nx;
                z.core_virtual_y = ed.snd_ny;
                z.core_noc0_x = ed.snd_nx;
                z.core_noc0_y = ed.snd_ny;
                z.risc = 0;
                z.timer_id = 0;
                z.name = kTrkRtt;
                z.start = dr.t0;
                z.end = dr.t2;
                z.color = 0x1ABC9Cu;  // teal: the resident heartbeat, distinct from init green / close red
                tracy_->HandleWorkerZone(z);
            }
            // The receiver lane interleaves the converted windows (zones) with the three markers, in
            // timestamp order -- CROSS-lane echo-vs-zone is an anchor artifact (the per-core Tracy anchors
            // are one-shot host fits whose ppm-scale error integrates to tens of us by close), so the
            // window is carried onto THIS lane where the anchors cancel and echo-inside-window is exact.
            struct RibItem {
                uint64_t ts;
                uint64_t end;
                std::string_view name;
                bool is_zone;
            };
            std::vector<RibItem> items;
            items.reserve(ed.draws.size() * 4);
            size_t in_window = 0;
            for (const auto& dr : ed.draws) {
                const uint64_t w0 = static_cast<uint64_t>(dr.conv_start);
                const uint64_t w1 = static_cast<uint64_t>(dr.conv_end);
                in_window += (dr.t1 >= w0 && dr.t1 <= w1) ? 1 : 0;
                items.push_back({w0, w1, kTrkWin, true});
                items.push_back({dr.t1, 0, kTrkEcho, false});
                items.push_back({static_cast<uint64_t>(dr.pred_init), 0, kTrkInit, false});
                items.push_back({static_cast<uint64_t>(dr.pred_tracked), 0, kTrkLive, false});
            }
            std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.ts < b.ts; });
            for (const auto& it : items) {
                if (it.is_zone) {
                    perf_debug::WorkerZonePacket zw;
                    zw.chip_id = ed.rcv_chip;
                    zw.core_virtual_x = ed.rcv_nx;
                    zw.core_virtual_y = ed.rcv_ny;
                    zw.core_noc0_x = ed.rcv_nx;
                    zw.core_noc0_y = ed.rcv_ny;
                    zw.risc = 0;
                    zw.timer_id = 0;
                    zw.name = it.name;
                    zw.start = it.ts;
                    zw.end = it.end;
                    zw.color = 0x76D7C4u;  // light teal: the sender's window, tracked onto this lane
                    tracy_->HandleWorkerZone(zw);
                } else {
                    perf_debug::WorkerEventPacket pe;
                    pe.chip_id = ed.rcv_chip;
                    pe.core_virtual_x = ed.rcv_nx;
                    pe.core_virtual_y = ed.rcv_ny;
                    pe.core_noc0_x = ed.rcv_nx;
                    pe.core_noc0_y = ed.rcv_ny;
                    pe.risc = 0;
                    pe.id = 0;
                    pe.name = it.name;
                    pe.timestamp = it.ts;
                    pe.num_values = 0;
                    tracy_->HandleWorkerEvent(pe);
                }
            }
            const auto& last = ed.draws.back();
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] ETH TRACKER ribbon {} -> {}: {} round(s) drawn on second-channel eth "
                "NOC0 ({},{}) -> ({},{}); at the LAST round the init-fit marker sits {:+.3f} us from the "
                "measured echo where the causally-tracked marker sits {:+.1f} ns -- and {} of {} echoes fall "
                "INSIDE their causally-tracked ETH_TRACK_RTT_TRACKED window (a miss = a clock step in that "
                "round's gap, corrected one round later)",
                ed.snd_chip,
                ed.rcv_chip,
                ed.draws.size(),
                ed.snd_nx,
                ed.snd_ny,
                ed.rcv_nx,
                ed.rcv_ny,
                static_cast<double>(last.pred_init - static_cast<int64_t>(last.t1)) / freq / 1000.0,
                static_cast<double>(last.pred_tracked - static_cast<int64_t>(last.t1)) / freq,
                in_window,
                ed.draws.size());
        }
    }
    delete eth_track_;
    eth_track_ = nullptr;
    // Corrections stay FROZEN at the last round -- the close-check reads them a few lines later, so the
    // staleness it scores is one round period, not a session.
}

void PerfDebugProfiler::check_sync_drift_at_close() {
    if (!eth_sync_close_check() || link_syncs_.empty()) {
        return;
    }
    eth_sync::LinkSyncConfig cfg;
    cfg.n_samples = eth_sync_samples();
    cfg.gap_us = eth_sync_gap_us();
    const double freq = root_freq_ghz_ > 0.0 ? root_freq_ghz_ : 1.35;  // cycles/ns

    uint32_t checked = 0;
    double worst_ns = 0.0;
    uint32_t worst_a = 0, worst_b = 0;
    double worst_elapsed_s = 0.0;
    for (const auto& e : link_syncs_) {
        if (!e.valid || e.snd_dev == nullptr || e.rcv_dev == nullptr) {
            continue;
        }
        if (eth_core_is_fabric_claimed(e.sender_chip, e.snd_eth) ||
            eth_core_is_fabric_claimed(e.receiver_chip, e.rcv_eth)) {
            // Cannot happen when the init selection ran under the same fabric config (it avoids claimed
            // channels), but a config brought up between init and close would land here.
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] eth sync CLOSE-CHECK {} -> {} SKIPPED: a fabric router now owns the "
                "link's eth channel; the session's accumulated drift is UNKNOWN for this link",
                e.sender_chip,
                e.receiver_chip);
            continue;
        }
        const auto r = eth_sync::measure_link(e.snd_dev, e.snd_eth, e.rcv_dev, e.rcv_eth, cfg);
        // A solution can be `valid` and still be junk. Observed in a trajectory run: one measurement came
        // back with residual 861.8 cycles and a nonsense +16 ppm where every neighbour sat at 2.4 cycles and
        // -1.6 ppm. Without this gate that measurement becomes a "drift" number indistinguishable from a
        // real one. 20 cycles is ~8x the normal 2.1-2.7 and far below anything pathological.
        constexpr double kMaxResidualCycles = 20.0;
        if (r.solution.valid && r.solution.residual_rms > kMaxResidualCycles) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] eth sync CLOSE-CHECK {} -> {}: DISCARDED, residual {:.1f} cycles "
                "exceeds {:.0f} (a blown fit, not a drift measurement)",
                e.sender_chip,
                e.receiver_chip,
                r.solution.residual_rms,
                kMaxResidualCycles);
            continue;
        }
        if (!r.solution.valid) {
            // Not fatal and not silent. The most likely cause is that something else now owns these eth
            // cores (fabric FW, which is NOT up when the init sync runs but may be by close). The kernels
            // are deadline-bounded precisely so this reports instead of wedging the core.
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] eth sync CLOSE-CHECK {} -> {} could not be measured ({}, {}); the "
                "session's accumulated drift is UNKNOWN for this link",
                e.sender_chip,
                e.receiver_chip,
                eth_sync::status_name(r.sender_status),
                eth_sync::status_name(r.receiver_status));
            continue;
        }
        // What the init fit says the offset should be at the instant we just re-measured. Composed as a
        // difference from the init reference, like every other extrapolation here, to keep the precision in
        // the answer rather than in the epoch.
        const double dt_cycles = static_cast<double>(static_cast<int64_t>(r.solution.mid_ref - e.ref_mid));
        const int64_t predicted = e.offset + static_cast<int64_t>(std::llround(dt_cycles * (e.rate - 1.0)));
        const int64_t err_cycles = r.solution.offset - predicted;

        // ---- HOST-ASSISTED PREDICTION -------------------------------------------------------------------
        //
        // The prediction above extrapolates the init eth fit with a FIXED rate, so it cannot know about a
        // rate that moved mid-session -- and it does move (observed -1.65 -> -1.75 ppm across one capture),
        // which is most of why this check sits at microseconds even with the clock pinned.
        //
        // The host path already measures what each device's clock actually did, live: the drift corrector's
        // per-device correction IS that clock's deviation from its own linear model. Both predictions assume
        // linearity, so the DIFFERENCE of the two devices' corrections is exactly the non-linear part the eth
        // extrapolation is missing:
        //
        //   offset_actual = offset_linear + (corr_sender - corr_receiver)
        //
        // This costs nothing -- the corrections are already being computed for zone placement -- and it needs
        // no eth traffic, so it works on a fabric-enabled run where the links cannot be re-measured.
        const int64_t corr_s = perf_debug::get_zone_ts_correction(e.sender_chip);
        const int64_t corr_r = perf_debug::get_zone_ts_correction(e.receiver_chip);
        const int64_t predicted_hosted = predicted + (corr_s - corr_r);
        const int64_t err_hosted = r.solution.offset - predicted_hosted;
        if (corr_s != 0 || corr_r != 0) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] eth sync CLOSE-CHECK {} -> {}: host-assisted prediction is {:+.3f} us "
                "off vs {:+.3f} us for the fixed-rate extrapolation (corrections {:+} / {:+} cycles) -- the live "
                "corrections (eth tracker or host drift corrector) supply what the one-shot fit could not "
                "know about",
                e.sender_chip,
                e.receiver_chip,
                static_cast<double>(err_hosted) / freq / 1000.0,
                static_cast<double>(err_cycles) / freq / 1000.0,
                corr_s,
                corr_r);
        }

        // ---- INDEPENDENT CROSS-CHECK: host MMIO vs the eth link, on the SAME two cores ----------------
        //
        // Every other number here grades one estimator with itself, so a shared systematic error cancels.
        // This one does not: it fits each of the two ETH cores against the host over PCIe MMIO, composes
        // those two fits into a predicted inter-device offset, and compares against what the eth link just
        // measured directly. No shared transport (PCIe vs an eth link) and no shared estimator (a host
        // least-squares vs a min-RTT Cristian solve), so a disagreement is real information.
        //
        // MUST be the eth cores, not the chip/worker anchors. A first version of this compared the link
        // against the WORKER-domain chip anchors and reported 75-238 ms of disagreement -- which was not a
        // finding, it was the eth-vs-Tensix clock-origin gap (measured elsewhere at up to ~56 minutes within
        // one chip) leaking in as an apparent error. Same-core is what makes the comparison mean anything.
        {
            // Uses the eth cores' OWN host anchors, captured at sync time by the same pass that measured
            // this link. No new device reads: an earlier version fitted e.snd_eth directly and aborted with
            // "No core type found for TRANSLATED (0,11)" -- those are LOGICAL eth coords, not the virtual
            // ones read_reg resolves.
            const LinkSync* tr = e.host_anchors_valid ? &e : nullptr;
            if (tr != nullptr && root_freq_ghz_ > 0.0) {
#ifdef TRACY_ENABLE
                const double ns_per_tick = TracyGetTimerMul();
#else
                const double ns_per_tick = 1.0;
#endif
                const double npt = ns_per_tick > 0.0 ? ns_per_tick : 1.0;
                const double f = root_freq_ghz_;  // one shared slope; each core keeps its own ORIGIN
                // Sender clock instant -> host tick (sender eth core's own anchor) -> receiver clock at that
                // same host tick (receiver eth core's anchor). Differences only, so the epoch never enters.
                const double t_ticks =
                    static_cast<double>(tr->snd_host_anchor) +
                    (static_cast<double>(r.solution.mid_ref) - static_cast<double>(tr->snd_dev_at_anchor)) /
                        (npt * f);
                const double dev_rcv = static_cast<double>(tr->rcv_dev_at_anchor) +
                                       (t_ticks - static_cast<double>(tr->rcv_host_anchor)) * npt * f;
                const double predicted_offset = dev_rcv - static_cast<double>(r.solution.mid_ref);
                const double disagree_cycles = static_cast<double>(r.solution.offset) - predicted_offset;
                const double disagree_us = disagree_cycles / f / 1000.0;
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] eth sync CROSS-CHECK {} -> {}: the two eth cores' HOST anchors "
                    "predict this link's offset {:+.3f} us ({:+.0f} cycles) from what the link just measured "
                    "[INDEPENDENT: PCIe MMIO + host least-squares vs an eth link + min-RTT solve]",
                    e.sender_chip,
                    e.receiver_chip,
                    disagree_us,
                    disagree_cycles);
            }
        }
        const double err_ns = static_cast<double>(err_cycles) / freq;
        const double elapsed_s = dt_cycles / (freq * 1e9);
        const double rate_delta_ppm = (r.solution.rate - e.rate) * 1e6;
        // ---- DRAW IT: measured vs predicted, on the peer's own eth lane ----
        // Both timestamps are readings of the SAME clock (the receiver's), so rendering them through that
        // core's one context puts them exactly `err_cycles` apart on screen -- the session's accumulated
        // error, made visible instead of inferred from a log line. The context's anchor is the INIT one and
        // is now stale, so their absolute position drifts with it; the SEPARATION is the honest quantity,
        // and it is the one being read here.
        if (tracy_ != nullptr) {
            auto& cl = MetalContext::instance().get_cluster();
            const CoreCoord rn = cl.get_physical_coordinate_from_logical_coordinates(
                e.rcv_dev->id(), e.rcv_eth, CoreType::ETH, /*no_warn=*/true);
            const CoreCoord sn = cl.get_physical_coordinate_from_logical_coordinates(
                e.snd_dev->id(), e.snd_eth, CoreType::ETH, /*no_warn=*/true);
            static constexpr std::string_view kCloseRtt = "ETH_SYNC_CLOSE_RTT";
            static constexpr std::string_view kMeasured = "ETH_SYNC_ECHO_MEASURED";
            static constexpr std::string_view kPredicted = "ETH_SYNC_ECHO_PREDICTED";
            static constexpr std::string_view kTracked = "ETH_SYNC_ECHO_TRACKED";
            constexpr size_t kMaxDraw = 64;
            const size_t n = std::min(r.trips.size(), kMaxDraw);
            // Third marker: the init prediction MOVED BY THE FROZEN CORRECTIONS -- the same (corr_s - corr_r)
            // the hosted number above adds. A tracker-off run has zero corrections, so it lands exactly on
            // PREDICTED; a tracked run shows it back at the measured echo while PREDICTED stays away.
            const int64_t chost = corr_s - corr_r;

            // Tracy wants non-decreasing arrival per lane, and the predicted marker can fall on either side
            // of the measured one depending on the sign of the drift -- so order everything explicitly
            // rather than assuming. Interleaving unsorted is what would corrupt the lane.
            //
            // kWinTracked is the sender's round-trip window carried onto the receiver's OWN lane through
            // the tracked prediction. Same-lane on purpose: each eth row's Tracy anchor is a one-shot host
            // MMIO fit from init, and its ppm-scale frequency error integrates to tens of us by close (the
            // host-vs-eth CROSS-CHECK above measures exactly that disagreement), so CROSS-lane echo-vs-zone
            // is an anchor artifact, not a sync statement. On this lane the anchor cancels: the measured
            // echo falls inside the window iff the tracked prediction matches reality. Tracker off:
            // chost = 0 and the window sits the whole session error away -- honest in both directions.
            static constexpr std::string_view kWinTracked = "ETH_SYNC_CLOSE_RTT_TRACKED";
            struct CloseItem {
                uint64_t ts;
                uint64_t end;
                std::string_view name;
                bool is_zone;
            };
            std::vector<CloseItem> items;
            items.reserve(n * 4);
            size_t in_window = 0;
            const double drate = e.rate - 1.0;
            for (size_t i = 0; i < n; i++) {
                const uint64_t mid = (r.trips[i].t0 + r.trips[i].t2) / 2;
                const double d = static_cast<double>(static_cast<int64_t>(mid - e.ref_mid));
                // What the INIT fit says the peer's clock read at this instant.
                const uint64_t t1_pred = static_cast<uint64_t>(
                    static_cast<int64_t>(mid) + e.offset + static_cast<int64_t>(std::llround(d * drate)));
                const auto conv = [&](uint64_t s) {  // sender instant -> tracked receiver clock
                    const double ds = static_cast<double>(static_cast<int64_t>(s - e.ref_mid));
                    return static_cast<uint64_t>(
                        static_cast<int64_t>(s) + e.offset + static_cast<int64_t>(std::llround(ds * drate)) +
                        chost);
                };
                const uint64_t w0 = conv(r.trips[i].t0);
                const uint64_t w1 = conv(r.trips[i].t2);
                in_window += (r.trips[i].t1 >= w0 && r.trips[i].t1 <= w1) ? 1 : 0;
                items.push_back({w0, w1, kWinTracked, true});
                items.push_back({r.trips[i].t1, 0, kMeasured, false});
                items.push_back({t1_pred, 0, kPredicted, false});
                items.push_back({static_cast<uint64_t>(static_cast<int64_t>(t1_pred) + chost), 0, kTracked, false});
            }
            std::sort(items.begin(), items.end(), [](const auto& x, const auto& y) { return x.ts < y.ts; });
            for (const auto& it : items) {
                if (it.is_zone) {
                    perf_debug::WorkerZonePacket zw;
                    zw.chip_id = e.receiver_chip;
                    zw.core_virtual_x = static_cast<uint32_t>(rn.x);
                    zw.core_virtual_y = static_cast<uint32_t>(rn.y);
                    zw.core_noc0_x = static_cast<uint32_t>(rn.x);
                    zw.core_noc0_y = static_cast<uint32_t>(rn.y);
                    zw.risc = 0;
                    zw.timer_id = 0;
                    zw.name = it.name;
                    zw.start = it.ts;
                    zw.end = it.end;
                    zw.color = 0xF1948Au;  // light red: the close window, tracked onto the peer's lane
                    tracy_->HandleWorkerZone(zw);
                } else {
                    perf_debug::WorkerEventPacket ev;
                    ev.chip_id = e.receiver_chip;
                    ev.core_virtual_x = static_cast<uint32_t>(rn.x);
                    ev.core_virtual_y = static_cast<uint32_t>(rn.y);
                    ev.core_noc0_x = static_cast<uint32_t>(rn.x);
                    ev.core_noc0_y = static_cast<uint32_t>(rn.y);
                    ev.risc = 0;
                    ev.id = 0;
                    ev.name = it.name;
                    ev.timestamp = it.ts;
                    ev.num_values = 0;
                    tracy_->HandleWorkerEvent(ev);
                }
            }
            // The sender's round trips too, so the peer's pair has something to sit against.
            for (size_t i = 0; i < n; i++) {
                perf_debug::WorkerZonePacket z;
                z.chip_id = e.sender_chip;
                z.core_virtual_x = static_cast<uint32_t>(sn.x);
                z.core_virtual_y = static_cast<uint32_t>(sn.y);
                z.core_noc0_x = static_cast<uint32_t>(sn.x);
                z.core_noc0_y = static_cast<uint32_t>(sn.y);
                z.risc = 0;
                z.timer_id = 0;
                z.name = kCloseRtt;
                z.start = r.trips[i].t0;
                z.end = r.trips[i].t2;
                z.color = 0xC0392Bu;  // red: the close-time pass, distinct from the green init pass
                tracy_->HandleWorkerZone(z);
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] eth sync CLOSE-CHECK {} -> {}: drew {} measured/predicted echo pair(s) "
                "on chip {} eth NOC0 ({},{}) -- the gap between ETH_SYNC_ECHO_MEASURED and "
                "ETH_SYNC_ECHO_PREDICTED IS the session's accumulated error ({:+.3f} us); {} of {} echoes "
                "fall INSIDE their ETH_SYNC_CLOSE_RTT_TRACKED window (same-lane; cross-lane placement is "
                "host-anchor-limited and NOT a sync statement)",
                e.sender_chip,
                e.receiver_chip,
                n,
                e.receiver_chip,
                rn.x,
                rn.y,
                err_ns / 1000.0,
                in_window,
                n);
        }

        ++checked;
        if (std::fabs(err_ns) > std::fabs(worst_ns)) {
            worst_ns = err_ns;
            worst_a = e.sender_chip;
            worst_b = e.receiver_chip;
            worst_elapsed_s = elapsed_s;
        }
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] eth sync CLOSE-CHECK {} -> {}: after {:.1f} s the measured offset is "
            "{:+} cycles ({:+.3f} us) from what the init fit predicted; rate now {:+.2f} ppm vs {:+.2f} ppm "
            "at init (delta {:+.2f} ppm), residual {:.1f} cycles",
            e.sender_chip,
            e.receiver_chip,
            elapsed_s,
            err_cycles,
            err_ns / 1000.0,
            (r.solution.rate - 1.0) * 1e6,
            (e.rate - 1.0) * 1e6,
            rate_delta_ppm,
            r.solution.residual_rms);
    }
    if (checked != 0) {
        // The headline number for the capture: how stale its alignment had become by the end. An implied
        // rate error is more useful than the raw microseconds, because it is what scales to a longer run.
        const double equiv_ppm = worst_elapsed_s > 0.0 ? (worst_ns / 1e9) / worst_elapsed_s * 1e6 : 0.0;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] eth sync SESSION DRIFT: worst {:+.3f} us over {:.1f} s ({} -> {}) "
            "[{:+.2f} ppm equivalent, but NOT a rate error -- see below]",
            worst_ns / 1000.0,
            worst_elapsed_s,
            worst_a,
            worst_b,
            equiv_ppm);
        // Say what this is, because the obvious reading of the number above is wrong. MEASURED, two ways
        // (device-side eth sync, and host MMIO reads of both wall clocks with no kernels involved): the
        // relative rate is STABLE at -1.6 ppm and fits to ~0.01 ppm, but the offset takes sporadic DISCRETE
        // steps of 5-40 us, several per 10 s, which survive min-RTT and min-bracket filtering in their
        // respective paths. So the drift is accumulated phase steps, not a mis-fitted slope -- which is why
        // its sign and size scatter between runs (+225, -143, -65 us observed) while every fit stays clean.
        // Consequence: a better rate fit CANNOT fix this. Only re-syncing more often can, and the re-sync
        // interval sets the worst-case misalignment.
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] eth sync SESSION DRIFT: the clocks step, they do not merely drift -- "
            "sporadic {}discrete offset steps accumulate between syncs, so re-sync interval (not fit "
            "quality) bounds cross-device alignment over a long capture",
            "");
    }
}

// ---- FABRIC ROUTER SYNC: host side of the in-router device-to-device clock sync -----------------
//
// The device half lives in the fabric router itself (fabric_router_sync_hook.hpp): at a host-set
// cadence the two routers on a link run Cristian's algorithm over raw 16 B eth messages and emit
// each timestamp as a self-timestamping PP_SYNC packet into their own SPSC rings, which the DRISC
// fillers sweep like any other marker. This host half:
//
//   1. DISCOVERS the routers: one fabric-claimed channel per spanning-tree link (plus every extra
//      link that closes a cycle, kept as a CLOSURE edge -- measured, never composed), reads each
//      hook's L1 block address from the AERISC fabric scratch words, and writes both ends' configs
//      (the LOWER chip id initiates; a config of zeros means disabled, so a router that never gets
//      one stays inert).
//   2. AGGREGATES: the receiver's decode threads relay every PP_SYNC packet to one sink; this
//      thread joins T0/T2 (initiator's stream) with T1 (responder's stream) by (round, idx),
//      min-RTT solves each round with the same eth_sync::solve the resident-pair path uses, and
//      tracks a per-edge offset series (rate from ROUND deltas -- the within-round rate of
//      back-to-back samples is meaningless).
//   3. COMPOSES the tree into per-device corrections against the ROOT device's timeline. The
//      correction is expressed against what each device's HOST ANCHOR already encodes:
//
//        corr_D = f_D * (AH_root - AH_D) + (AD_D - AD_root) - measured(D minus root)
//
//      i.e. (the anchor-implied device-vs-root offset) minus (the eth-measured one) -- anchors
//      absorb the huge power-on offsets, the eth link supplies the ns-class truth, and corr_D is
//      the small anchor error the link can see. Root's correction is 0 by construction.
//      Published through the same grow-only ratchet + common-baseline mechanics as the eth
//      tracker (the baseline rides through the ratchet and cancels in every difference).
//
// MUTUAL EXCLUSION: when the resident-pair eth tracker owns the corrections
// (TT_METAL_PERF_DEBUG_ETH_TRACK_HZ set), this path still MEASURES -- that is the planes=2
// cross-check -- but does not publish; at teardown it prints the disagreement between the two
// instruments on each shared link.

namespace {
double fabric_sync_hz() {
    static const double v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_FABRIC_SYNC_HZ");
        return (s != nullptr && *s != '\0') ? std::strtod(s, nullptr) : 0.0;
    }();
    return v;
}
}  // namespace

struct PerfDebugProfiler::FabricSyncState {
    static constexpr uint32_t kNSamples = tt::tt_fabric::router_sync::kMaxSamples;
    // Cap on per-edge render data, so a long soak cannot grow this without bound (a 20 Hz link at
    // 16 samples/round fills this in ~10 min of continuous syncing; beyond that the drawing stops
    // but the MEASUREMENT is untouched).
    static constexpr size_t kMaxDraws = 200000;

    struct End {
        uint32_t chip = 0;
        CoreCoord logical;
        CoreCoord virt;
        uint32_t blk_addr = 0;
        // NOC0 coords + whether this lane got its OWN Tracy anchor. Contexts are created LAZILY on a
        // core's first zone, and the only AddCore calls upstream are for DRISC cores -- so without an
        // explicit anchor here an eth lane is minted against the WORKER anchor, and eth tiles do not
        // share the Tensix origin. Measured cost of getting this wrong: FSYNC zones landed 10.4 s away
        // from the workload they ran inside.
        uint32_t noc0_x = 0, noc0_y = 0;
        bool tracy_ok = false;
        // Saved Cfg body (flags..peer_blk), so the aggregator can RE-ARM this end. The hook zeroes
        // its own block at kernel start, so every fabric re-init silently unconfigures the hook.
        uint32_t cfg[7] = {};
        bool cfg_valid = false;
    };
    struct Round {
        uint64_t t0[kNSamples] = {}, t1[kNSamples] = {}, t2[kNSamples] = {};
        uint32_t got0 = 0, got1 = 0, got2 = 0;  // bitmasks by sample idx
        std::chrono::steady_clock::time_point first_seen;
    };
    struct Edge {
        End init, resp;  // init = lower chip id; measured offset is resp MINUS init
        bool in_tree = false;
        std::map<uint32_t, Round> pending;  // by 17-bit wire round (wraps; joins are per-round)
        // solved series
        bool have = false;
        int64_t off_last = 0;   // resp - init, cycles, at ref_last (init clock)
        uint64_t ref_last = 0;
        double rate = 1.0;  // d(resp)/d(init), from ROUND deltas
        std::deque<std::pair<uint64_t, int64_t>> series;  // (mid_ref, offset) for the rate fit
        // stats
        uint64_t rounds_solved = 0, rounds_partial = 0, rounds_dropped = 0;
        uint64_t trips_total = 0;
        uint64_t rtt_min = ~0ull;
        int64_t off_min = 0, off_max = 0;
        double residual_last = 0.0;
        uint64_t stray_samples = 0;  // which/side mismatches (should stay 0)
        // Render data for the Tracy consumer, filled at SOLVE time so only samples the solver
        // actually accepted are drawn: an FSYNC_RTT zone on the INITIATOR lane spanning t0 -> t2
        // (its width IS the measured round trip) and an FSYNC_ECHO marker on the RESPONDER lane
        // at t1 (the instant that ping was observed).
        struct Draw {
            uint64_t t0 = 0, t1 = 0, t2 = 0;
            // The round's solved offset (resp MINUS init). Lets t1 be carried onto the INITIATOR lane,
            // where both endpoints share one anchor so the anchor error cancels and t1-inside-[t0,t2]
            // is exact. Cross-lane it is not: per-core anchors are one-shot fits whose ppm error
            // integrates to tens of us, which is the same trap the eth tracker documents.
            int64_t off = 0;
            // The published cross-device corrections AS THEY STOOD WHEN THIS ROUND SOLVED.
            // publish_fabric_sync_corrections() reruns every 40 ms, so the correction is time-varying
            // and a record materialised mid-run gets the value current THEN. Reading it once at
            // teardown and applying it to draws spanning the whole run leaves a residual equal to how
            // far the offset drifted since -- measured: the residual tracked each link's offset SPAN
            // at a constant 1.19x (0->3 15534 cy / 13.7 us, 0->1 8035 cy / 7.1 us, 1->2 3985 cy /
            // 3.5 us). Sampling per draw is what makes the two ends coincide.
            int64_t corr_i = 0;
            int64_t corr_r = 0;
        };
        std::vector<Draw> draws;
    };
    std::vector<Edge> edges;
    // (chip << 32 | vx << 16 | vy) -> (edge index, 0 = initiator end, 1 = responder end)
    std::unordered_map<uint64_t, std::pair<size_t, int>> end_by_core;
    uint32_t root = 0;
    std::unordered_map<uint32_t, size_t> parent_edge;  // chip -> tree edge toward root
    // host anchors snapshot (what the Tracy transforms encode), taken at start
    struct Anchor {
        int64_t host_ns = 0;
        uint64_t dev = 0;
        double f = 0.0;  // cycles per ns
        bool valid = false;
    };
    std::unordered_map<uint32_t, Anchor> anchors;
    // sink queue: decode threads push, the aggregator swaps out
    std::mutex mu;
    std::vector<perf_debug::PerfDebugSyncSample> q;
    uint64_t sink_samples = 0;
    std::thread th;
    std::atomic<bool> stop{false};
    bool publish = false;
    int64_t baseline = 0;
    bool disabled_devices = false;
    uint64_t rearms = 0;  // times a router relaunch wiped a config and the aggregator rewrote it

    static uint64_t core_key(uint32_t chip, uint32_t vx, uint32_t vy) {
        return (static_cast<uint64_t>(chip) << 32) | (static_cast<uint64_t>(vx) << 16) | vy;
    }
    // Composition uses each edge's LATEST solved offset, never a cross-clock extrapolation: the
    // reference instants live in different chips' clock domains (absolute offsets are minutes), so
    // an out-of-domain (t - ref) times (rate - 1) would inject microseconds of false correction.
    // Cross-edge measurement skew is bounded by one round period; at ~ppm relative rates that is
    // nanoseconds, and the ring closure MEASURES what it actually costs. `rate` is a statistic.
};

// Install the receiver sink. Must run BEFORE receiver_->start(); the sink only enqueues (decode
// threads must never block on the aggregator).
void PerfDebugProfiler::install_fabric_sync_sink() {
    if (fabric_sync_hz() <= 0.0 || receiver_ == nullptr) {
        return;
    }
    if (MetalContext::instance().get_fabric_config() == tt::tt_fabric::FabricConfig::DISABLED) {
        return;
    }
    auto* st = new FabricSyncState();
    fabric_sync_ = st;
    receiver_->set_sync_sink([st](const perf_debug::PerfDebugSyncSample& s) {
        std::lock_guard<std::mutex> lk(st->mu);
        st->q.push_back(s);
        st->sink_samples++;
    });
}

void PerfDebugProfiler::start_fabric_sync(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    if (fabric_sync_ == nullptr) {
        return;
    }
    auto* st = fabric_sync_;
    const double hz = fabric_sync_hz();
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const auto devices = mesh_device->get_devices();

    uint64_t scratch_addr = 0;
    try {
        scratch_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_ROUTER_SYNC_SCRATCH);
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[perf-debug profiler] FABRIC SYNC: no router scratch on this arch ({}); off", e.what());
        return;
    }

    // ---- discover one claimed channel per link: BFS tree edges + cycle-closing CLOSURE edges ----
    auto pair_key = [](uint32_t a, uint32_t b) {
        return (static_cast<uint64_t>(std::min(a, b)) << 32) | std::max(a, b);
    };
    std::set<uint64_t> seen_pairs;
    std::set<int> visited;
    std::queue<IDevice*> bfs;
    st->root = static_cast<uint32_t>(devices.front()->id());
    visited.insert(devices.front()->id());
    bfs.push(devices.front());
    auto find_dev = [&](int id) -> IDevice* {
        for (IDevice* d : devices) {
            if (d->id() == id) {
                return d;
            }
        }
        return nullptr;
    };
    while (!bfs.empty()) {
        IDevice* a = bfs.front();
        bfs.pop();
        for (const CoreCoord& ec : a->get_active_ethernet_cores(true)) {
            std::tuple<ChipId, CoreCoord> peer;
            try {
                peer = a->get_connected_ethernet_core(ec);
            } catch (const std::exception&) {
                continue;
            }
            const int pid = static_cast<int>(std::get<0>(peer));
            IDevice* b = find_dev(pid);
            if (b == nullptr) {
                continue;  // leaves the mesh
            }
            const uint64_t pk = pair_key(a->id(), pid);
            if (seen_pairs.count(pk) != 0) {
                continue;  // one channel per chip pair is enough for a clock
            }
            // The hook lives in ROUTERS: only a channel CLAIMED on both ends carries it.
            if (!eth_core_is_fabric_claimed(static_cast<uint32_t>(a->id()), ec) ||
                !eth_core_is_fabric_claimed(static_cast<uint32_t>(pid), std::get<1>(peer))) {
                continue;
            }
            seen_pairs.insert(pk);
            FabricSyncState::Edge e;
            const bool a_initiates = a->id() < pid;
            FabricSyncState::End ea{static_cast<uint32_t>(a->id()), ec, CoreCoord{}, 0};
            FabricSyncState::End eb{static_cast<uint32_t>(pid), std::get<1>(peer), CoreCoord{}, 0};
            e.init = a_initiates ? ea : eb;
            e.resp = a_initiates ? eb : ea;
            e.in_tree = visited.count(pid) == 0;
            if (e.in_tree) {
                visited.insert(pid);
                bfs.push(b);
                // parent hop for the BFS CHILD (pid). The edge's init/resp orientation is chosen by
                // chip id, not by tree direction -- the composition walk checks which end the child is.
                st->parent_edge[static_cast<uint32_t>(pid)] = st->edges.size();
            }
            st->edges.push_back(std::move(e));
        }
    }
    if (st->edges.empty()) {
        log_warning(tt::LogMetal, "[perf-debug profiler] FABRIC SYNC: no fabric-claimed links found; off");
        return;
    }

    // ---- discovery handshake + config write on every end ----
    auto virt_of = [&](const FabricSyncState::End& en) {
        return cluster.get_virtual_coordinate_from_logical_coordinates(en.chip, en.logical, CoreType::ETH);
    };
    auto read_blk = [&](FabricSyncState::End& en) -> bool {
        en.virt = virt_of(en);
        uint32_t w[3] = {0, 0, 0};
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3)) {
            cluster.read_core(w, sizeof(w), tt_cxy_pair(en.chip, en.virt), scratch_addr);
            if (w[0] == tt::tt_fabric::router_sync::kDiscMagic && w[1] != 0) {
                en.blk_addr = w[1];
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    };
    size_t kept = 0;
    for (auto& e : st->edges) {
        if (!read_blk(e.init) || !read_blk(e.resp)) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] FABRIC SYNC: router hook not discovered on {} ({},{}) <-> {} ({},{}) "
                "(no scratch magic; is the hook compiled in?); link dropped",
                e.init.chip, e.init.logical.x, e.init.logical.y,
                e.resp.chip, e.resp.logical.x, e.resp.logical.y);
            e.init.blk_addr = 0;
            continue;
        }
        kept++;
    }
    if (kept == 0) {
        log_warning(tt::LogMetal, "[perf-debug profiler] FABRIC SYNC: no link discovered its hooks; off");
        return;
    }

    // Anchors snapshot: what the per-device Tracy transforms encode; the corrections below are the
    // measured deviation FROM these.
    for (const auto& ctx : devices_) {
        FabricSyncState::Anchor an;
        an.valid = ctx.anchor_valid && ctx.freq_ghz > 0.0;
        an.host_ns = ctx.anchor_host;
        an.dev = ctx.anchor_dev;
        an.f = ctx.freq_ghz;
        st->anchors[ctx.chip_id] = an;
    }

    // Publication policy: the resident-pair tracker wins when both are up (planes=2 cross-check runs
    // both on purpose -- this path then only measures and reports).
    st->publish = (eth_track_hz() == 0);

    // ---- write configs: responder first, magic last ----
    using namespace tt::tt_fabric::router_sync;
    auto write_cfg = [&](FabricSyncState::End& en, bool initiator, uint64_t interval, uint32_t peer_blk) {
        uint32_t body[7];  // flags .. peer_blk (Cfg minus magic)
        body[0] = kFlagEnabled | (initiator ? kFlagInitiator : 0);
        body[1] = static_cast<uint32_t>(interval & 0xFFFFFFFFu);
        body[2] = static_cast<uint32_t>(interval >> 32);
        body[3] = FabricSyncState::kNSamples;
        body[4] = 1u << 21;  // first_wait ~1.55 ms: responder poll cadence + a context-switch stretch
        body[5] = 1u << 19;  // next_wait ~0.39 ms: mid-round tolerance for the peer's context switch
        body[6] = peer_blk;
        cluster.write_core(body, sizeof(body), tt_cxy_pair(en.chip, en.virt), en.blk_addr + 4);
        const uint32_t magic = kCfgMagic;
        cluster.write_core(&magic, sizeof(magic), tt_cxy_pair(en.chip, en.virt), en.blk_addr);
        std::memcpy(en.cfg, body, sizeof(body));
        en.cfg_valid = true;
    };
    for (auto& e : st->edges) {
        if (e.init.blk_addr == 0) {
            continue;
        }
        const int aiclk_mhz = cluster.get_device_aiclk(e.init.chip);
        const double cyc_per_s = (aiclk_mhz > 0 ? aiclk_mhz : 1350) * 1e6;
        const uint64_t init_interval = static_cast<uint64_t>(cyc_per_s / hz);
        const uint64_t resp_interval = std::min<uint64_t>(init_interval / 4, 1u << 17);  // <= ~97 us poll
        write_cfg(e.resp, false, resp_interval, e.init.blk_addr);
        write_cfg(e.init, true, init_interval, e.resp.blk_addr);
        // Readback: proves the config bytes are IN L1 at the discovered address (a wrong address or
        // a swallowed PCIe write shows up here, not as a silent dead link).
        uint32_t rb[8] = {0};
        cluster.read_core(rb, sizeof(rb), tt_cxy_pair(e.init.chip, e.init.virt), e.init.blk_addr);
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] FSYNC cfg readback init chip {} @0x{:x}: magic 0x{:x} flags 0x{:x} "
            "interval {} n {} waits {}/{} peer 0x{:x}",
            e.init.chip,
            e.init.blk_addr,
            rb[0],
            rb[1],
            (static_cast<uint64_t>(rb[3]) << 32) | rb[2],
            rb[4],
            rb[5],
            rb[6],
            rb[7]);
        st->end_by_core[FabricSyncState::core_key(e.init.chip, e.init.virt.x, e.init.virt.y)] = {
            static_cast<size_t>(&e - st->edges.data()), 0};
        st->end_by_core[FabricSyncState::core_key(e.resp.chip, e.resp.virt.x, e.resp.virt.y)] = {
            static_cast<size_t>(&e - st->edges.data()), 1};
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] FABRIC SYNC link {} ({},{}) -> {} ({},{}): {} rounds/s x {} samples "
            "in-router on the CLAIMED channel{}",
            e.init.chip,
            e.init.logical.x,
            e.init.logical.y,
            e.resp.chip,
            e.resp.logical.x,
            e.resp.logical.y,
            hz,
            FabricSyncState::kNSamples,
            e.in_tree ? "" : " [closure edge]");
    }

    // NOTE ON ORDER: this runs AFTER the config write, and that is deliberate. The sync window
    // (start_fabric_sync -> the stop() that disables the devices) can be as short as ~100 ms on a
    // small workload, which at 20 Hz is only 1-2 rounds. The reference host fit below costs ~50 ms
    // of MMIO (spacing_us=500 x ~100 samples), so doing it BEFORE the config write ate half the
    // window and regressed the continuous-sync test to a single round. Anchoring is safe to defer:
    // a Tracy context bakes its anchor in at CREATION, contexts are created lazily on a core's
    // first zone, and the only zones these eth lanes ever carry are the FSYNC ones drawn at
    // teardown (fabric routers carry no DeviceZoneScopedN of their own).
    // ---- the aggregator thread ----
    const auto& ctx = receiver_->capture_context();
    st->th = std::thread([this, st, &ctx, &cluster, scratch_addr]() {
        std::vector<perf_debug::PerfDebugSyncSample> batch;
        uint64_t last_log_solved = 0;
        auto last_pub = std::chrono::steady_clock::now();
        auto last_dbg = std::chrono::steady_clock::now();
        auto last_rearm = std::chrono::steady_clock::now();
        while (!st->stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            // ---- RE-ARM: the config does NOT survive a fabric re-init ----
            // The hook zeroes its own block at kernel start (init()), and the host wrote the config
            // exactly once at start_fabric_sync. So the first time fabric tears down and relaunches
            // its routers, magic goes to 0 and the hook parks in state 1 (unconfigured) for the rest
            // of the run -- it keeps ticking, it is just never asked for another round. Symptom is a
            // run that solves ONE round and then reports `device init ok/fail 1/0` forever.
            // Keyed on the magic only: fabric_sync_disable_devices() clears FLAGS and deliberately
            // leaves magic valid, so teardown is never fought by this.
            if (!st->disabled_devices &&
                std::chrono::steady_clock::now() - last_rearm >= std::chrono::milliseconds(100)) {
                last_rearm = std::chrono::steady_clock::now();
                const uint32_t magic_v = kCfgMagic;
                for (auto& e : st->edges) {
                    // responder first, then initiator -- same order as the initial write, so the
                    // initiator can never start a round against an unconfigured peer
                    for (auto* en : {&e.resp, &e.init}) {
                        if (!en->cfg_valid || en->blk_addr == 0) {
                            continue;
                        }
                        uint32_t magic = 0;
                        cluster.read_core(&magic, sizeof(magic), tt_cxy_pair(en->chip, en->virt), en->blk_addr);
                        if (magic == magic_v) {
                            continue;
                        }
                        cluster.write_core(en->cfg, sizeof(en->cfg), tt_cxy_pair(en->chip, en->virt), en->blk_addr + 4);
                        cluster.write_core(&magic_v, sizeof(magic_v), tt_cxy_pair(en->chip, en->virt), en->blk_addr);
                        st->rearms++;
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lk(st->mu);
                batch.swap(st->q);
            }
            // route samples into per-edge pending rounds
            for (const auto& s : batch) {
                if (s.dev >= ctx.devices.size() || s.lane >= ctx.devices[s.dev].lanes.size() ||
                    s.idx >= FabricSyncState::kNSamples) {
                    continue;
                }
                const auto& li = ctx.devices[s.dev].lanes[s.lane];
                const auto it = st->end_by_core.find(FabricSyncState::core_key(li.chip_id, li.virtual_x, li.virtual_y));
                if (it == st->end_by_core.end()) {
                    continue;  // an eth core we did not configure (another plane's router)
                }
                auto& e = st->edges[it->second.first];
                const bool from_init = it->second.second == 0;
                const bool ok = from_init ? (s.which == 0 || s.which == 2) : (s.which == 1);
                if (!ok) {
                    e.stray_samples++;
                    continue;
                }
                auto& r = e.pending[s.round];
                if (r.got0 == 0 && r.got1 == 0 && r.got2 == 0) {
                    r.first_seen = std::chrono::steady_clock::now();
                }
                if (s.which == 0) {
                    r.t0[s.idx] = s.ts;
                    r.got0 |= 1u << s.idx;
                } else if (s.which == 1) {
                    r.t1[s.idx] = s.ts;
                    r.got1 |= 1u << s.idx;
                } else {
                    r.t2[s.idx] = s.ts;
                    r.got2 |= 1u << s.idx;
                }
            }
            batch.clear();
            // solve rounds that are complete, or stale enough that the missing packets are lost
            const auto now_tp = std::chrono::steady_clock::now();
            for (auto& e : st->edges) {
                for (auto it = e.pending.begin(); it != e.pending.end();) {
                    auto& r = it->second;
                    const uint32_t full = (1u << FabricSyncState::kNSamples) - 1;
                    const bool complete = (r.got0 & r.got1 & r.got2) == full;
                    const bool stale = now_tp - r.first_seen > std::chrono::milliseconds(complete ? 0 : 700);
                    if (!complete && !stale) {
                        ++it;
                        continue;
                    }
                    std::vector<eth_sync::Trip> trips;
                    const size_t draws_before = e.draws.size();
                    const uint32_t have = r.got0 & r.got1 & r.got2;
                    for (uint32_t i = 0; i < FabricSyncState::kNSamples; i++) {
                        if ((have & (1u << i)) == 0 || r.t2[i] < r.t0[i]) {
                            continue;
                        }
                        const uint64_t mid = r.t0[i] + (r.t2[i] - r.t0[i]) / 2;
                        trips.push_back(eth_sync::Trip{
                            r.t0[i], r.t1[i], r.t2[i],
                            static_cast<int64_t>(r.t1[i]) - static_cast<int64_t>(mid),
                            r.t2[i] - r.t0[i], mid});
                        if (e.draws.size() < FabricSyncState::kMaxDraws) {
                            e.draws.push_back({r.t0[i], r.t1[i], r.t2[i], 0, 0, 0});
                        }
                    }
                    if (trips.size() < 4) {
                        e.rounds_dropped++;
                        it = e.pending.erase(it);
                        continue;
                    }
                    auto sol = eth_sync::solve(std::move(trips), 0.25);
                    if (!sol.valid) {
                        e.rounds_dropped++;
                        it = e.pending.erase(it);
                        continue;
                    }
                    complete ? e.rounds_solved++ : e.rounds_partial++;
                    e.trips_total += sol.n_total;
                    e.rtt_min = std::min(e.rtt_min, sol.rtt_min);
                    e.residual_last = sol.residual_rms;
                    if (!e.have) {
                        e.off_min = e.off_max = sol.offset;
                    }
                    e.off_min = std::min(e.off_min, sol.offset);
                    e.off_max = std::max(e.off_max, sol.offset);
                    e.off_last = sol.offset;
                    e.ref_last = sol.mid_ref;
                    e.have = true;
                    // Backfill this round's draws with the offset the round solved to, so the render can
                    // put the echo on the initiator's clock without reaching for a global average.
                    const int64_t corr_i_now = perf_debug::get_zone_ts_correction(e.init.chip);
                    const int64_t corr_r_now = perf_debug::get_zone_ts_correction(e.resp.chip);
                    for (size_t k = draws_before; k < e.draws.size(); k++) {
                        e.draws[k].off = sol.offset;
                        e.draws[k].corr_i = corr_i_now;
                        e.draws[k].corr_r = corr_r_now;
                    }
                    e.series.emplace_back(sol.mid_ref, sol.offset);
                    if (e.series.size() > 256) {
                        e.series.pop_front();
                    }
                    // rate from ROUND deltas, least squares, once the baseline is long enough
                    if (e.series.size() >= 8 && (e.series.back().first - e.series.front().first) > (1ull << 30)) {
                        double sx = 0, sy = 0, sxx = 0, sxy = 0;
                        const uint64_t x0 = e.series.front().first;
                        const double n = static_cast<double>(e.series.size());
                        for (const auto& [m, o] : e.series) {
                            const double x = static_cast<double>(m - x0);
                            const double y = static_cast<double>(o - e.series.front().second);
                            sx += x; sy += y; sxx += x * x; sxy += x * y;
                        }
                        const double den = n * sxx - sx * sx;
                        if (den > 0) {
                            e.rate = 1.0 + (n * sxy - sx * sy) / den;
                        }
                    }
                    it = e.pending.erase(it);
                }
                // hard cap on pending rounds (drain hiccup): drop oldest
                while (e.pending.size() > 64) {
                    e.pending.erase(e.pending.begin());
                    e.rounds_dropped++;
                }
            }
            // compose + publish corrections
            if (now_tp - last_pub > std::chrono::milliseconds(40)) {
                last_pub = now_tp;
                publish_fabric_sync_corrections();
            }
            // live device-side breadcrumbs: scratch stat + poll-state words on both ends
            if (now_tp - last_dbg > std::chrono::seconds(2)) {
                last_dbg = now_tp;
                for (const auto& e : st->edges) {
                    if (e.init.blk_addr == 0) {
                        continue;
                    }
                    uint32_t si[4] = {0}, sr[4] = {0};
                    cluster.read_core(si, sizeof(si), tt_cxy_pair(e.init.chip, e.init.virt), scratch_addr);
                    cluster.read_core(sr, sizeof(sr), tt_cxy_pair(e.resp.chip, e.resp.virt), scratch_addr);
                    // SPSC control vector of the initiator eth core: lane heads (words 0..1) and
                    // tails (words 24..25) -- says whether the producer writes where the drain reads.
                    uint32_t cv[26] = {0};
                    uint64_t eth_prof = 0;
                    try {
                        eth_prof = MetalContext::instance().hal().get_dev_addr(
                            HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::PROFILER);
                        cluster.read_core(cv, sizeof(cv), tt_cxy_pair(e.init.chip, e.init.virt), eth_prof);
                    } catch (const std::exception&) {
                    }
                    // both ends message-slot view: ping.tag / echo.tag
                    uint32_t bi[20] = {0}, br[20] = {0};
                    cluster.read_core(bi, sizeof(bi), tt_cxy_pair(e.init.chip, e.init.virt), e.init.blk_addr);
                    cluster.read_core(br, sizeof(br), tt_cxy_pair(e.resp.chip, e.resp.virt), e.resp.blk_addr);
                    log_info(
                        tt::LogMetal,
                        "[perf-debug profiler] FSYNC DBG {}->{}: init stat 0x{:x} poll 0x{:x} | resp stat "
                        "0x{:x} poll 0x{:x} | sink {} | init cv h {}/{} t {}/{} | init ping/echo 0x{:x}/0x{:x} "
                        "resp 0x{:x}/0x{:x} resp_magic 0x{:x}",
                        e.init.chip, e.resp.chip, si[2], si[3], sr[2], sr[3], st->sink_samples,
                        cv[0], cv[1], cv[24], cv[25], bi[11], bi[15], br[11], br[15], br[0]);
                }
            }
            // periodic per-link line
            uint64_t solved = 0;
            for (const auto& e : st->edges) {
                solved += e.rounds_solved + e.rounds_partial;
            }
            if (solved - last_log_solved >= 200) {
                last_log_solved = solved;
                for (const auto& e : st->edges) {
                    if (!e.have) {
                        continue;
                    }
                    log_info(
                        tt::LogMetal,
                        "[perf-debug profiler] FABRIC SYNC {} -> {}: {} rounds ({} partial, {} dropped), "
                        "offset {} cy (span {}), rtt_min {} cy, rate {:+.3f} ppm, residual {:.1f} cy{}",
                        e.init.chip, e.resp.chip, e.rounds_solved, e.rounds_partial, e.rounds_dropped,
                        e.off_last, e.off_max - e.off_min, e.rtt_min, (e.rate - 1.0) * 1e6, e.residual_last,
                        e.in_tree ? "" : " [closure]");
                }
                log_fabric_sync_closure(false);
            }
        }
    });
    // ORDER MATTERS AND IS EASY TO GET WRONG (three attempts got it wrong before this one):
    //   configs written -> AGGREGATOR RUNNING -> compose device anchors -> anchor the eth lanes.
    // Composition waits for rounds to solve, and only the aggregator thread above solves them, so it
    // must run after the thread exists -- an earlier version sat before it and could never succeed.
    // The eth-lane anchoring then follows, because it reads ctx.anchor_* for every chip.
    // All of this still completes long before any zone reaches Tracy: worker contexts are created
    // lazily on their first zone during the workload, and the FSYNC rows are drawn at teardown.
    // Compose the DEVICE anchors first: the eth-lane anchoring below reads ctx.anchor_*, so every
    // non-root device must already be expressed against the root before we get there.
    compose_device_anchors_from_root();

    // ---- anchor the sync eth lanes: ONE host fit, everything else via device<->device ----
    // Contexts are minted lazily on a core's first zone and bake their anchor in at creation, so this
    // has to happen now (start), never at the teardown render.
    //
    // NOT a host fit per lane. Two eth cores on the SAME die, in the same reset domain -- whose true
    // origins differ by ~nothing -- disagreed by 4-16 ms when each got its own fit (bh-31 chips 0..3:
    // 7.263 / 11.365 / 4.029 / 15.791 ms). That spread IS the host<->device sync error, and it is
    // ~5 orders of magnitude worse than the device<->device sync it would be competing with (ring
    // closure -43.70 ns). Host fits are simply the wrong instrument for anything but the root.
    //
    // So: ONE host fit establishes the ETH-CLASS origin, and every other lane is that origin carried
    // across by the tree -- the same rule the per-DEVICE anchors already follow (root gets the careful
    // fit; everyone else is eth_sync_anchor_for + root_host_anchor_). The class origin does have to be
    // measured once, because eth cores do NOT share the worker origin: using the worker anchor put
    // these rows 10.4 s away from the workload they ran inside.
    if (tracy_ != nullptr && root_sync_valid_ && root_freq_ghz_ > 0.0) {
        std::vector<FabricSyncState::End*> ends;
        for (auto& e : st->edges) {
            ends.push_back(&e.init);
            ends.push_back(&e.resp);
        }
        // Prefer a reference lane on the ROOT chip: then the class origin needs no carrying at all and
        // the one remaining hop of composition disappears.
        FabricSyncState::End* ref = nullptr;
        for (auto* en : ends) {
            if (en->chip == eth_sync_root_chip_) {
                ref = en;
                break;
            }
        }
        if (ref == nullptr && !ends.empty()) {
            ref = ends.front();
        }

        // Per-chip clock at the ROOT's host instant, from the anchors already composed off the root.
        const int64_t h0 = root_host_anchor_;
        auto chip_clock_at_h0 = [&](uint32_t chip, double& freq_out) -> std::optional<double> {
            for (const auto& ctx : devices_) {
                if (ctx.chip_id != chip) {
                    continue;
                }
                if (!ctx.anchor_valid || ctx.freq_ghz <= 0.0) {
                    return std::nullopt;
                }
                freq_out = ctx.freq_ghz;
                return static_cast<double>(ctx.anchor_dev) + static_cast<double>(h0 - ctx.anchor_host) * ctx.freq_ghz;
            }
            return std::nullopt;
        };

        // THE one host fit. spacing_us=500 matches the root's own: the slope is baseline-limited, and
        // at back-to-back spacing the fitted frequency carries ~1e-4 of error.
        double f_ref = 0.0;
        std::optional<double> ref_eth_at_h0, ref_chip_at_h0;
        if (ref != nullptr) {
            const PerfDebugSync rs = sync_device_clock(cluster, ref->chip, ref->virt, /*spacing_us=*/500);
            double f_tmp = 0.0;
            ref_chip_at_h0 = chip_clock_at_h0(ref->chip, f_tmp);
            if (rs.valid && ref_chip_at_h0.has_value()) {
                f_ref = f_tmp;
                ref_eth_at_h0 =
                    static_cast<double>(rs.device_at_anchor) + static_cast<double>(h0 - rs.host_anchor) * f_ref;
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] FABRIC SYNC: eth-class origin measured ONCE on chip {} "
                    "(the only host<->device fit for these lanes); every other lane is carried across "
                    "by the device<->device tree",
                    ref->chip);
            }
        }

        for (auto* en : ends) {
            CoreCoord n0;
            try {
                n0 = cluster.get_physical_coordinate_from_logical_coordinates(
                    en->chip, en->logical, CoreType::ETH, /*no_warn=*/true);
            } catch (const std::exception&) {
                continue;
            }
            double f_chip = 0.0;
            const std::optional<double> chip_at_h0 = chip_clock_at_h0(en->chip, f_chip);
            if (!ref_eth_at_h0.has_value() || !ref_chip_at_h0.has_value() || !chip_at_h0.has_value() || f_chip <= 0.0) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] FABRIC SYNC: eth lane chip {} NOC0 ({},{}) has no route to the "
                    "root anchor; its FSYNC rows would fall back to the worker anchor, so they are NOT drawn",
                    en->chip,
                    n0.x,
                    n0.y);
                continue;
            }
            // The eth-class origin, carried from the reference chip to this one by the chip-to-chip
            // delta the device<->device sync measured. No second host fit anywhere.
            const double dev = *ref_eth_at_h0 + (*chip_at_h0 - *ref_chip_at_h0);
            tracy_->AddCore(en->chip, n0.x, n0.y, h0, dev, f_chip);
            tracy_->RegisterEthCore(en->chip, n0.x, n0.y);
            en->noc0_x = static_cast<uint32_t>(n0.x);
            en->noc0_y = static_cast<uint32_t>(n0.y);
            en->tracy_ok = true;
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] FABRIC SYNC: eth lane chip {} NOC0 ({},{}) VIRTUAL ({},{}) {} "
                "anchored (host_anchor {}, device_at_anchor {:.0f} cy, {:.6f} GHz){}",
                en->chip,
                n0.x,
                n0.y,
                // The eth-wedge crash names a VIRTUAL core; whether that core was SENDING or only
                // RECEIVING points at completely different causes, so log both the coord and the role.
                en->virt.x,
                en->virt.y,
                // Role comes from the config body we just wrote (anchoring runs after write_cfg), so
                // no need to reach back into the edge.
                (en->cfg_valid && (en->cfg[0] & kFlagInitiator)) != 0 ? "INITIATOR(sends)" : "RESPONDER(recv)",
                h0,
                dev,
                f_chip,
                en == ref ? " [reference: the one host fit]" : " [carried by device<->device]");
        }
    }

    log_info(
        tt::LogMetal,
        "[perf-debug profiler] FABRIC SYNC: {} in-router link(s) at {} Hz -- t0/t1/t2 ride the profiler "
        "drain as PP_SYNC packets; corrections {} (root chip {})",
        kept, hz, st->publish ? "PUBLISHED centrally" : "measure-only (eth tracker owns corrections)", st->root);
}

// Compose tree edges into per-chip (chip minus root) offsets at a common instant, convert to
// anchor-relative corrections, ratchet, publish.
// Compose one chip's clock offset against the ROOT by walking the fabric-sync tree: every LINKED ETH
// PAIR is measured directly (~50 ns closure), and each chip reaches the root through its parent edge.
// This is the same traversal publish_fabric_sync_corrections() already does for corrections -- lifted
// out so ANCHORING can use it too, which is the whole point: one host fit at the root, every other
// device reached through measured links, never its own host fit.
//
// Returns false if any edge on this chip's path has not solved a round yet, so the caller can wait
// rather than anchor off a half-built tree.
bool PerfDebugProfiler::fabric_sync_delta_for(uint32_t chip, int64_t& delta_out, double& rate_out) const {
    auto* st = fabric_sync_;
    if (st == nullptr) {
        return false;
    }
    if (chip == st->root) {
        delta_out = 0;
        rate_out = 1.0;
        return true;
    }
    int64_t d = 0;
    double rate = 1.0;
    uint32_t cur = chip;
    int hops = 0;
    while (cur != st->root && hops++ < 8) {
        auto it = st->parent_edge.find(cur);
        if (it == st->parent_edge.end() || !st->edges[it->second].have) {
            return false;
        }
        const auto& e = st->edges[it->second];
        // off_last is resp MINUS init, from the latest solved round on that link.
        if (e.resp.chip == cur) {
            d += e.off_last;
            rate *= e.rate;
            cur = e.init.chip;
        } else {
            d -= e.off_last;
            rate /= (e.rate != 0.0 ? e.rate : 1.0);
            cur = e.resp.chip;
        }
    }
    if (cur != st->root) {
        return false;
    }
    delta_out = d;
    rate_out = rate;
    return true;
}

// Express every non-root device's clock against the ROOT by walking measured eth links, and anchor
// it there. The root is the only device that ever touches the host clock.
//
// MUST run AFTER start_fabric_sync() has written the hook config (so rounds can solve) and BEFORE the
// eth lanes are anchored (they compose off ctx.anchor_*). Getting that order wrong is easy: an earlier
// attempt put this in host_reanchor_after_boot(), which the log proved runs 1.5 s BEFORE the fabric
// sync is even configured -- so the wait expired before there was anything to wait for.
void PerfDebugProfiler::compose_device_anchors_from_root() {
    //
    // This is the whole design: sync every LINKED ETH PAIR (the in-router hook does, to ~50 ns), walk
    // each device's path to the root through those links, and express its clock against the root's.
    // The root is the ONLY device that ever touches the host clock. No device fits the host itself --
    // independent fits disagree, measured here as 3.5 / 7.1 / 13.7 us between the two ends of a link
    // where the same instant is drawn twice, constant across the run, against a link closure of
    // -37..-57 ns.
    //
    // Sequencing: this runs right after start_fabric_sync(), so the hook is configured but has not
    // solved a round yet -- hence the bounded wait. Anchors are still registered long before any zone
    // reaches Tracy (worker contexts are created lazily on their first zone, during the workload), so
    // waiting here is free in capture terms.
    if (!root_sync_valid_ || fabric_sync_ == nullptr) {
        return;
    }
    std::vector<uint32_t> pending;
    for (const auto& ctx : devices_) {
        if (ctx.chip_id != eth_sync_root_chip_) {
            pending.push_back(ctx.chip_id);
        }
    }
    if (pending.empty()) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(3000);
    std::unordered_map<uint32_t, std::pair<int64_t, double>> composed;
    while (composed.size() < pending.size() && std::chrono::steady_clock::now() < deadline) {
        for (uint32_t chip : pending) {
            if (composed.count(chip) != 0) {
                continue;
            }
            int64_t delta = 0;
            double rate = 1.0;
            if (fabric_sync_delta_for(chip, delta, rate)) {
                composed[chip] = {delta, rate};
            }
        }
        if (composed.size() < pending.size()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    uint32_t n_comp = 0;
    for (auto& ctx : devices_) {
        auto it = composed.find(ctx.chip_id);
        if (it == composed.end()) {
            continue;
        }
        const auto [delta, rate] = it->second;
        // delta is chip-clock MINUS root-clock at a common instant, so the chip's clock reading at the
        // ROOT's host anchor is the root's reading plus delta. Same host anchor for every device --
        // that is what makes them comparable.
        const double dev_at_root_anchor = static_cast<double>(static_cast<int64_t>(root_dev_at_anchor_) + delta);
        // SHARED SLOPE, own ORIGIN -- the same rule the DRISC contexts follow. Alignment is RELATIVE,
        // so a common rate error is common-mode and invisible on a timeline, whereas a PER-CHIP rate
        // makes the devices diverge linearly. Measured with per-chip rate applied: the pair started at
        // 0 ns (composition nails the origin) and drifted to -6065 ns over 84 s, ~72 ppb of differential
        // slope. `rate` stays a reported statistic only.
        (void)rate;
        ctx.freq_ghz = root_freq_ghz_;
        ctx.clock_synced = true;
        tracy_->AddDevice(ctx.chip_id, root_host_anchor_, dev_at_root_anchor, ctx.freq_ghz);
        ctx.anchor_host = root_host_anchor_;
        ctx.anchor_dev = static_cast<uint64_t>(dev_at_root_anchor);
        ctx.anchor_valid = true;
        // REFRESH THE SNAPSHOT. st->anchors is captured early in start_fabric_sync -- before this
        // composition runs -- so it records these devices while they are still unanchored. Left stale,
        // publish_fabric_sync_corrections() hits its "a device without an anchor cannot take a
        // correction" guard and returns forever, and EVERY zone_ts_correction stays 0. That is exactly
        // what was measured (CORRDBG early-return: chip 1 has no valid ANCHOR SNAPSHOT), and it is why
        // the cross-device residual accumulated into a smooth 6 us ramp instead of being re-zeroed at
        // each 40 ms publish. The guard is right; the snapshot was stale.
        if (fabric_sync_ != nullptr) {
            FabricSyncState::Anchor an;
            an.valid = true;
            an.host_ns = ctx.anchor_host;
            an.dev = ctx.anchor_dev;
            an.f = ctx.freq_ghz;
            fabric_sync_->anchors[ctx.chip_id] = an;
        }
        n_comp++;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] Device {} COMPOSED off root {} through measured eth links: "
            "delta {:+} cy, {:.6f} GHz ({:+.2f} ppm vs root) -- no host fit on this device",
            ctx.chip_id,
            eth_sync_root_chip_,
            delta,
            ctx.freq_ghz,
            (rate - 1.0) * 1e6);
    }
    if (n_comp != pending.size()) {
        log_error(
            tt::LogMetal,
            "[perf-debug profiler] only {} of {} non-root device(s) could be composed off root {} within "
            "1.5 s; the rest have NO anchor and will have no Tracy rows. A device is composable once every "
            "link on its path to the root has solved a round.",
            n_comp,
            pending.size(),
            eth_sync_root_chip_);
    }
}

void PerfDebugProfiler::publish_fabric_sync_corrections() {
    // CORRDBG: say which branch is taken, ONCE each. The corrections were measured to be 0 on every
    // chip, and "it never reaches the write" has several possible causes -- let the log distinguish
    // them instead of deriving which one it must be.
    static bool dbg_any = false, dbg_all = false, dbg_pub = false, dbg_root = false, dbg_chip = false,
                dbg_write = false;
    auto* st = fabric_sync_;
    if (st == nullptr) {
        return;
    }
    bool any = false;
    for (const auto& e : st->edges) {
        any = any || (e.in_tree && e.have);
    }
    if (!any) {
        if (!dbg_any) {
            dbg_any = true;
            log_info(tt::LogMetal, "[perf-debug profiler] CORRDBG early-return: no in-tree edge has solved yet");
        }
        return;
    }
    // delta[chip] = chip clock MINUS root clock, composed from each edge's latest solved offset
    std::unordered_map<uint32_t, int64_t> delta;
    delta[st->root] = 0;
    bool all = true;
    for (const auto& [chip, _] : st->parent_edge) {
        int64_t d = 0;
        uint32_t cur = chip;
        bool ok = true;
        int hops = 0;
        while (cur != st->root && hops++ < 8) {
            auto it = st->parent_edge.find(cur);
            if (it == st->parent_edge.end() || !st->edges[it->second].have) {
                ok = false;
                break;
            }
            const auto& e = st->edges[it->second];
            const int64_t off = e.off_last;  // resp - init, latest solved round
            if (e.resp.chip == cur) {
                d += off;  // cur is the resp end; parent is init
                cur = e.init.chip;
            } else {
                d -= off;
                cur = e.resp.chip;
            }
        }
        if (ok && cur == st->root) {
            delta[chip] = d;
        } else {
            all = false;
        }
    }
    if (!all || !st->publish) {
        if (!all && !dbg_all) {
            dbg_all = true;
            log_info(tt::LogMetal, "[perf-debug profiler] CORRDBG early-return: !all (a chip has no composed path)");
        }
        if (!st->publish && !dbg_pub) {
            dbg_pub = true;
            log_info(tt::LogMetal, "[perf-debug profiler] CORRDBG early-return: !publish (measure-only policy)");
        }
        return;
    }
    // corr = (anchor-implied delta) - (measured delta); root = 0 by construction
    const auto ra = st->anchors.find(st->root);
    if (ra == st->anchors.end() || !ra->second.valid) {
        if (!dbg_root) {
            dbg_root = true;
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] CORRDBG early-return: ROOT chip {} has no valid anchor snapshot",
                st->root);
        }
        return;
    }
    std::unordered_map<uint32_t, int64_t> corr;
    for (const auto& [chip, d] : delta) {
        const auto a = st->anchors.find(chip);
        if (a == st->anchors.end() || !a->second.valid) {
            if (!dbg_chip) {
                dbg_chip = true;
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] CORRDBG early-return: chip {} has no valid ANCHOR SNAPSHOT (the "
                    "snapshot is taken early in start_fabric_sync, before compose_device_anchors_from_root)",
                    chip);
            }
            return;  // a device without an anchor cannot take a correction against one
        }
        const double anchor_implied =
            a->second.f * static_cast<double>(ra->second.host_ns - a->second.host_ns) +
            static_cast<double>(static_cast<int64_t>(a->second.dev - ra->second.dev));
        corr[chip] = static_cast<int64_t>(std::llround(anchor_implied)) - d;
    }
    // the tracker's ratchet mechanics, verbatim: common baseline keeps every published value
    // non-decreasing and cancels out of every cross-device difference
    for (const auto& [chip, c] : corr) {
        const int64_t prev = perf_debug::get_zone_ts_correction(chip);
        const int64_t need = prev - c;
        if (need > st->baseline) {
            st->baseline = need;
        }
    }
    for (const auto& [chip, c] : corr) {
        perf_debug::set_zone_ts_correction(chip, c + st->baseline);
    }
    if (!dbg_write) {
        dbg_write = true;
        for (const auto& [chip, c] : corr) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] CORRDBG WRITE chip {}: corr {} + baseline {} = {} cy",
                chip,
                c,
                st->baseline,
                c + st->baseline);
        }
    }
}

// Ring closure: every non-tree edge's measured offset vs the tree-composed prediction. THE accuracy
// bound for the full-planes configuration -- it shares no link with the prediction.
void PerfDebugProfiler::log_fabric_sync_closure(bool final_report) {
    auto* st = fabric_sync_;
    if (st == nullptr) {
        return;
    }
    for (const auto& e : st->edges) {
        if (e.in_tree || !e.have) {
            continue;
        }
        // compose init->resp through the tree from each edge's latest solved offset
        auto delta_of = [&](uint32_t chip, bool& ok) -> int64_t {
            int64_t d = 0;
            uint32_t cur = chip;
            int hops = 0;
            ok = true;
            while (cur != st->root && hops++ < 8) {
                auto it = st->parent_edge.find(cur);
                if (it == st->parent_edge.end() || !st->edges[it->second].have) {
                    ok = false;
                    return 0;
                }
                const auto& pe = st->edges[it->second];
                const int64_t off = pe.off_last;
                if (pe.resp.chip == cur) {
                    d += off;
                    cur = pe.init.chip;
                } else {
                    d -= off;
                    cur = pe.resp.chip;
                }
            }
            ok = ok && cur == st->root;
            return d;
        };
        bool ok1 = false, ok2 = false;
        const int64_t di = delta_of(e.init.chip, ok1);
        const int64_t dr = delta_of(e.resp.chip, ok2);
        if (!ok1 || !ok2) {
            continue;
        }
        const int64_t predicted = dr - di;  // resp - init via the tree
        const int64_t closure = e.off_last - predicted;
        const double freq = st->anchors.count(e.init.chip) != 0 && st->anchors[e.init.chip].valid
                                ? st->anchors[e.init.chip].f
                                : 1.35;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] FABRIC SYNC RING CLOSURE {} -> {}: measured {} cy vs tree {} cy -> "
            "closure {:+} cy ({:+.2f} ns){}",
            e.init.chip, e.resp.chip, e.off_last, predicted, closure,
            static_cast<double>(closure) / freq, final_report ? " [final]" : "");
    }
}

// Early half of teardown: stop the DEVICES (no new rounds) while the drain is still sweeping, so
// the last in-flight round's packets still reach the aggregator.
void PerfDebugProfiler::fabric_sync_disable_devices() {
    auto* st = fabric_sync_;
    if (st == nullptr || st->disabled_devices) {
        return;
    }
    st->disabled_devices = true;
    auto& cluster = MetalContext::instance().get_cluster();
    const uint32_t zero = 0;

    // PASS 1 -- stop every INITIATOR first, everywhere, before touching a single responder. The
    // initiator is the only side that STARTS a round, so until all of them are quiet a responder
    // cleared here can just be re-entered by its peer on the next tick.
    for (const auto& e : st->edges) {
        if (e.init.blk_addr == 0) {
            continue;
        }
        cluster.write_core(&zero, sizeof(zero), tt_cxy_pair(e.init.chip, e.init.virt), e.init.blk_addr + 4);
    }
    // PASS 2 -- then the responders. flags = 0 disables at the next deadline; the config magic stays
    // valid on purpose, so the aggregator's re-arm (which keys on magic) never fights this.
    for (const auto& e : st->edges) {
        if (e.init.blk_addr == 0) {
            continue;
        }
        cluster.write_core(&zero, sizeof(zero), tt_cxy_pair(e.resp.chip, e.resp.virt), e.resp.blk_addr + 4);
    }

    // PASS 3 -- DRAIN, and this is the point of the whole function.
    //
    // Clearing flags only stops the NEXT round; it does not wait for the one already in flight. The
    // hook sends on the RAW sender TXQ, so a round caught mid-send leaves that TXQ busy, the eth core
    // never goes active again, and the NEXT profiler session dies in llrt.cpp with "Timed out while
    // waiting for active ethernet core ... to become active again" -> terminate -> core dump.
    // A/B-confirmed as ours on the same 8-group 2D Mesh config: hook at 20 Hz crashed at session 2
    // (twice); hook not compiled ran 6 sessions / 196 tests with zero eth timeouts.
    //
    // Quiesced == the core has run poll() SINCE the flags were cleared and taken the disabled branch,
    // i.e. published state 1 (unconfigured/disabled) or 2 (interval 0). States 4/5 mean it is INSIDE a
    // round right now; 6..9 are failure sites published from inside one, so they are not idle either.
    // Worst case is one round (~7.4 ms = first_wait 1.55 ms + 15 x next_wait 0.39 ms) plus one whole
    // interval (50 ms at 20 Hz) before it polls again -- so ~60 ms. The bound below is ~8x that.
    //
    // ON EXPIRY WE PROCEED ANYWAY, loudly. A hook stuck in-round must never hang teardown: the card
    // may then need a warm_reset, but a warning the user can act on beats a hang they cannot. We do
    // NOT try to force the TXQ quiescent from the host -- the only lever is writing eth TXQ registers
    // out from under a live router, which risks corrupting fabric state on cores that are still fine.
    uint64_t scratch_addr = 0;
    try {
        const auto& hal = MetalContext::instance().hal();
        scratch_addr =
            hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_ROUTER_SYNC_SCRATCH);
    } catch (const std::exception&) {
        return;  // no scratch address means no breadcrumb to wait on; nothing safe to do here
    }

    struct Pending {
        uint32_t chip;
        CoreCoord virt;
        uint32_t last_state = 0;
    };
    std::vector<Pending> pending;
    for (const auto& e : st->edges) {
        if (e.init.blk_addr == 0) {
            continue;
        }
        pending.push_back({e.init.chip, e.init.virt, 0});
        pending.push_back({e.resp.chip, e.resp.virt, 0});
    }
    if (pending.empty()) {
        return;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    size_t still_busy = pending.size();
    while (still_busy != 0 && std::chrono::steady_clock::now() < deadline) {
        still_busy = 0;
        for (auto& pd : pending) {
            uint32_t dbg = 0;
            try {
                cluster.read_core(&dbg, sizeof(dbg), tt_cxy_pair(pd.chip, pd.virt), scratch_addr + 12);
            } catch (const std::exception&) {
                continue;  // unreadable core: not something this drain can fix
            }
            pd.last_state = dbg >> 28;
            if (pd.last_state != 1 && pd.last_state != 2) {
                still_busy++;
            }
        }
        if (still_busy != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (still_busy != 0) {
        std::string who;
        for (const auto& pd : pending) {
            if (pd.last_state != 1 && pd.last_state != 2) {
                who += fmt::format(" chip{} ({},{}) state={}", pd.chip, pd.virt.x, pd.virt.y, pd.last_state);
            }
        }
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] FABRIC SYNC: {} of {} eth lane(s) did NOT leave their round within 500 ms "
            "of being disabled -- proceeding with teardown anyway (a hang here would be worse). Their sender "
            "TXQ may still be busy, which can leave the eth core unable to go active again and take the NEXT "
            "profiler session down; a warm_reset clears it. Still in-round:{}",
            still_busy,
            pending.size(),
            who);
    } else {
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] FABRIC SYNC: all {} eth lane(s) left their round before teardown",
            pending.size());
    }
}

// Draw the in-router fabric sync onto the two eth lanes that actually carried it: one FSYNC_RTT zone
// per accepted sample on the INITIATOR lane (t0 -> t2, so the box width is the measured round trip)
// and one FSYNC_ECHO marker on the RESPONDER lane at t1. This is what makes the sync VISIBLE -- the
// PP_SYNC packets are routed OUT of the record stream at decode and never reach a consumer, so
// without this the exchanges exist only in the log.
//
// Same shape and the same deferral as the eth-tracker ribbon: called at teardown, when the receiver
// consumers are already down and nothing else is pushing into the handler. Tracy requires
// non-decreasing zone END per (context, thread), so each lane is sorted on the field that lane is
// keyed by -- t2 for the initiator zones, t1 for the responder markers.
void PerfDebugProfiler::render_fabric_sync_into_tracy() {
    auto* st = fabric_sync_;
    if (st == nullptr || tracy_ == nullptr) {
        return;
    }
    static constexpr std::string_view kRtt = "FSYNC_RTT";
    static constexpr std::string_view kEchoOn = "FSYNC_ECHO";       // carried onto the initiator lane
    static constexpr std::string_view kEchoRaw = "FSYNC_ECHO_RAW";  // responder's own clock, own lane
    uint64_t zones = 0, marks = 0, links = 0, inside = 0, total = 0;
    bool capped = false;
    for (const auto& e : st->edges) {
        // No own anchor means the lane would render against the worker anchor, seconds away from the
        // workload. Drawing that is worse than drawing nothing, so it is skipped (and warned at start).
        if (e.draws.empty() || !e.init.tracy_ok || !e.resp.tracy_ok) {
            continue;
        }
        capped = capped || e.draws.size() >= FabricSyncState::kMaxDraws;

        // APPLY THE PUBLISHED CROSS-DEVICE CORRECTION -- this render used to skip it, and that is the
        // whole reason FSYNC_ECHO and FSYNC_ECHO_RAW (the SAME instant, drawn on the two ends) sat
        // 73.4 ms apart on link 2->3 instead of on top of each other: the gap was device 3's raw
        // +73.5 ms timeline offset from device 2, not sync residual. Worst cross-device start skew in
        // that capture was 189 ms, against a sync that measures alignment to ~50 ns.
        //
        // The record path already does exactly this at materialization (perf_debug_receiver.cpp:
        // `r.ts - r.dur + corr`), but this render draws straight from st->edges at teardown and never
        // passes through it. Same store, same convention: corrected = raw + get_zone_ts_correction(chip).
        // Per-draw corrections are used below; these remain only as the fallback for a draw that
        // solved before any correction had been published.
        const int64_t corr_init_fallback = perf_debug::get_zone_ts_correction(e.init.chip);
        const int64_t corr_resp_fallback = perf_debug::get_zone_ts_correction(e.resp.chip);

        // ---- initiator lane: the RTT box + the echo CARRIED ONTO THIS CLOCK ----
        // t1 lives on the responder's clock; t1 - offset puts it on the initiator's. Both endpoints
        // then share ONE anchor, so the anchor error cancels and "echo inside the box" is a genuine
        // causality check instead of an artifact of two independent per-core fits.
        struct Item {
            uint64_t ts;
            uint64_t end;
            std::string_view name;
            bool is_zone;
        };
        std::vector<Item> items;
        items.reserve(e.draws.size() * 2);
        for (const auto& dr : e.draws) {
            if (dr.t2 < dr.t0) {
                continue;
            }
            // The inside-box test stays in RAW initiator-clock space: it is a same-lane causality
            // check, and a correction common to both endpoints cannot change it. Only the DRAWN
            // timestamps are corrected.
            const int64_t t1_on_init = static_cast<int64_t>(dr.t1) - dr.off;
            const int64_t ci = dr.corr_i != 0 ? dr.corr_i : corr_init_fallback;
            items.push_back(
                {static_cast<uint64_t>(static_cast<int64_t>(dr.t0) + ci),
                 static_cast<uint64_t>(static_cast<int64_t>(dr.t2) + ci),
                 kRtt,
                 true});
            if (t1_on_init > 0) {
                items.push_back({static_cast<uint64_t>(t1_on_init + ci), 0, kEchoOn, false});
                total++;
                inside +=
                    (static_cast<uint64_t>(t1_on_init) >= dr.t0 && static_cast<uint64_t>(t1_on_init) <= dr.t2) ? 1 : 0;
            }
        }
        // Tracy wants non-decreasing arrival per (context, thread); zones and markers share the lane.
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.ts < b.ts; });
        for (const auto& it : items) {
            if (it.is_zone) {
                perf_debug::WorkerZonePacket z;
                z.chip_id = e.init.chip;
                z.core_virtual_x = e.init.noc0_x;
                z.core_virtual_y = e.init.noc0_y;
                z.core_noc0_x = e.init.noc0_x;
                z.core_noc0_y = e.init.noc0_y;
                z.risc = 0;
                z.timer_id = 0;
                z.name = kRtt;
                z.start = it.ts;
                z.end = it.end;
                z.color = 0x9B59B6u;  // purple: in-router fabric sync, vs the eth tracker teal
                tracy_->HandleWorkerZone(z);
                zones++;
            } else {
                perf_debug::WorkerEventPacket pe;
                pe.chip_id = e.init.chip;
                pe.core_virtual_x = e.init.noc0_x;
                pe.core_virtual_y = e.init.noc0_y;
                pe.core_noc0_x = e.init.noc0_x;
                pe.core_noc0_y = e.init.noc0_y;
                pe.risc = 0;
                pe.id = 0;
                pe.name = it.name;
                pe.timestamp = it.ts;
                pe.num_values = 0;
                tracy_->HandleWorkerEvent(pe);
                marks++;
            }
        }

        // ---- responder lane: the raw stamp on its OWN clock ----
        std::vector<uint64_t> t1s;
        t1s.reserve(e.draws.size());
        for (const auto& dr : e.draws) {
            const int64_t cr = dr.corr_r != 0 ? dr.corr_r : corr_resp_fallback;
            t1s.push_back(static_cast<uint64_t>(static_cast<int64_t>(dr.t1) + cr));
        }
        std::sort(t1s.begin(), t1s.end());
        for (uint64_t t1 : t1s) {
            perf_debug::WorkerEventPacket pe;
            pe.chip_id = e.resp.chip;
            pe.core_virtual_x = e.resp.noc0_x;
            pe.core_virtual_y = e.resp.noc0_y;
            pe.core_noc0_x = e.resp.noc0_x;
            pe.core_noc0_y = e.resp.noc0_y;
            pe.risc = 0;
            pe.id = 0;
            pe.name = kEchoRaw;
            pe.timestamp = t1;  // already corrected per-draw when t1s was built
            pe.num_values = 0;
            tracy_->HandleWorkerEvent(pe);
            marks++;
        }
        links++;
    }
    if (links != 0) {
        // The inside-fraction is the self-check: an echo that does NOT sit inside its own round trip on
        // the initiator clock means the offset for that round is wrong. It should be ~100%.
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] FABRIC SYNC drawn: {} zone(s) + {} marker(s) across {} eth link(s); "
            "{}/{} echoes fall INSIDE their own FSYNC_RTT box on the initiator clock ({:.1f}%){}",
            zones,
            marks,
            links,
            inside,
            total,
            total != 0 ? 100.0 * static_cast<double>(inside) / static_cast<double>(total) : 0.0,
            capped ? " -- DRAW CAP HIT, the tail is not drawn" : "");
    }
}

void PerfDebugProfiler::stop_fabric_sync() {
    auto* st = fabric_sync_;
    if (st == nullptr) {
        return;
    }
    fabric_sync_disable_devices();
    st->stop.store(true);
    if (st->th.joinable()) {
        st->th.join();
    }
    auto& cluster = MetalContext::instance().get_cluster();
    for (const auto& e : st->edges) {
        if (e.init.blk_addr == 0) {
            continue;
        }
        uint32_t stat_i = 0, stat_r = 0;
        try {
            const auto& hal = MetalContext::instance().hal();
            const uint64_t scratch =
                hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_ROUTER_SYNC_SCRATCH);
            cluster.read_core(&stat_i, sizeof(stat_i), tt_cxy_pair(e.init.chip, e.init.virt), scratch + 8);
            cluster.read_core(&stat_r, sizeof(stat_r), tt_cxy_pair(e.resp.chip, e.resp.virt), scratch + 8);
        } catch (const std::exception&) {
        }
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] FABRIC SYNC {} -> {} FINAL: {} rounds solved ({} partial, {} dropped, "
            "{} stray), offset last {} cy span {} cy, rtt_min {} cy, rate {:+.3f} ppm | device init "
            "ok/fail {}/{} resp {}/{}{}",
            e.init.chip, e.resp.chip, e.rounds_solved, e.rounds_partial, e.rounds_dropped, e.stray_samples,
            e.off_last, e.off_max - e.off_min, e.rtt_min == ~0ull ? 0ull : e.rtt_min, (e.rate - 1.0) * 1e6,
            stat_i >> 8, stat_i & 0xFF, stat_r >> 8, stat_r & 0xFF, e.in_tree ? "" : " [closure]");
    }
    log_fabric_sync_closure(true);
    render_fabric_sync_into_tracy();
    // planes=2 cross-check: when the resident-pair tracker measured the same links on FREE channels,
    // print the two instruments' disagreement per shared chip pair.
    if (eth_track_ != nullptr) {
        for (const auto& e : st->edges) {
            if (!e.have) {
                continue;
            }
            for (const auto& ed : eth_track_->edges) {
                const bool same = (ed.snd_chip == e.init.chip && ed.rcv_chip == e.resp.chip) ||
                                  (ed.snd_chip == e.resp.chip && ed.rcv_chip == e.init.chip);
                if (!same || !ed.have) {
                    continue;
                }
                // tracker: measured = init-fit linear - corr_edge. The fit's reference lives in
                // the tracker SENDER's clock; convert the hook's instant into that domain first
                // (same chip -> same clock; otherwise shift by the measured link offset).
                const uint64_t t_snd = (ed.snd_chip == e.init.chip)
                                           ? e.ref_last
                                           : static_cast<uint64_t>(static_cast<int64_t>(e.ref_last) + e.off_last);
                const double dtf = static_cast<double>(static_cast<int64_t>(t_snd - ed.ref0));
                const int64_t tracker_meas =
                    ed.off0 + static_cast<int64_t>(std::llround(dtf * (ed.rate0 - 1.0))) - ed.corr_edge;
                const int64_t hook_meas = (ed.snd_chip == e.init.chip) ? e.off_last : -e.off_last;
                const double freq = st->anchors.count(e.init.chip) != 0 && st->anchors[e.init.chip].valid
                                        ? st->anchors[e.init.chip].f
                                        : 1.35;
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] FABRIC SYNC vs ETH TRACKER on {} <-> {}: hook {} cy, tracker {} cy "
                    "-> disagreement {:+} cy ({:+.2f} ns) [two instruments, two channels, same link]",
                    e.init.chip, e.resp.chip, hook_meas, tracker_meas, hook_meas - tracker_meas,
                    static_cast<double>(hook_meas - tracker_meas) / freq);
            }
        }
    }
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] FABRIC SYNC: {} samples reached the sink end to end; {} config re-arm(s) "
        "after router relaunch",
        st->sink_samples,
        st->rearms);
    delete st;
    fabric_sync_ = nullptr;
}

void PerfDebugProfiler::reanchor_after_boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    if (!eth_sync_late() || !eth_sync_enabled() || devices_.empty() || tracy_ == nullptr) {
        return;
    }
    const auto context_id = mesh_device->impl().get_context_id();
    auto& cluster = MetalContext::instance(context_id).get_cluster();

    // Discard the init-time transforms wholesale rather than blending them: they are the stale thing being
    // replaced, and a half-updated tree would relate devices through a mix of old and new offsets.
    link_syncs_.clear();
    eth_sync_parent_edge_.clear();
    eth_sync_traces_.clear();
    eth_sync_closure_valid_ = false;
    const bool had_root = root_sync_valid_;
    root_sync_valid_ = false;

    sync_devices_over_eth(mesh_device);

    // Root first: every other device hangs off its fit.
    for (auto& ctx : devices_) {
        if (ctx.chip_id != eth_sync_root_chip_ || ctx.core_virt.empty()) {
            continue;
        }
        const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
        const PerfDebugSync s = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/500);
        if (!s.valid) {
            break;
        }
        root_sync_valid_ = true;
        root_host_anchor_ = s.host_anchor;
        root_dev_at_anchor_ = s.device_at_anchor;
        root_freq_ghz_ = s.frequency;
        ctx.freq_ghz = s.frequency;
        tracy_->AddDevice(ctx.chip_id, s.host_anchor, static_cast<double>(s.device_at_anchor), s.frequency);
        ctx.anchor_host = s.host_anchor;
        ctx.anchor_dev = s.device_at_anchor;
        ctx.anchor_valid = true;
    }
    if (!root_sync_valid_) {
        // Keep whatever init produced. A stale anchor still renders; a missing one does not.
        root_sync_valid_ = had_root;
        log_warning(
            tt::LogMetal,
            "[perf-debug profiler] eth sync LATE re-anchor: root host fit failed; keeping the init-time "
            "anchors (zones stay {} s stale)",
            "~bring-up");
        return;
    }

    uint32_t rechipped = 0, recored = 0;
    for (auto& ctx : devices_) {
        double chip_freq = ctx.freq_ghz;
        if (ctx.chip_id != eth_sync_root_chip_) {
            uint64_t derived_clock = 0;
            double derived_rate = 1.0;
            if (!eth_sync_anchor_for(ctx.chip_id, root_dev_at_anchor_, derived_clock, derived_rate)) {
                continue;  // no route to the root; this device keeps its init anchor
            }
            chip_freq = root_freq_ghz_ * derived_rate;
            ctx.freq_ghz = chip_freq;
            tracy_->AddDevice(
                ctx.chip_id, root_host_anchor_, static_cast<double>(derived_clock), chip_freq);
            ctx.anchor_host = root_host_anchor_;
            ctx.anchor_dev = derived_clock;
            ctx.anchor_valid = true;
        }
        ++rechipped;
        // The DRAM-core origins go stale exactly like the chip ones, and for the same reason.
        for (uint32_t d = 0; d < ctx.n_drisc; d++) {
            const auto nit = ctx.virt_to_noc0.find(
                (static_cast<uint64_t>(ctx.drisc_virtual[d].x) << 32) |
                static_cast<uint64_t>(ctx.drisc_virtual[d].y));
            if (nit == ctx.virt_to_noc0.end()) {
                continue;
            }
            const PerfDebugSync ds = sync_device_clock(cluster, ctx.chip_id, ctx.drisc_virtual[d]);
            if (!ds.valid) {
                continue;
            }
            tracy_->AddCore(
                ctx.chip_id,
                nit->second.first,
                nit->second.second,
                ds.host_anchor,
                static_cast<double>(ds.device_at_anchor),
                chip_freq);
            ++recored;
        }
    }
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] eth sync LATE re-anchor: {} device(s) and {} DRISC core(s) re-fitted AFTER "
        "bring-up, so zones are placed by transforms measured ~now instead of before the boot [the init "
        "fit is stale by the whole bring-up, which measured 26.3 s]",
        rechipped,
        recored);
}

void PerfDebugProfiler::emit_eth_sync_lanes() {
    if (tracy_ == nullptr || eth_sync_traces_.empty()) {
        return;
    }
    // Both eth cores carry real samples that PREDATE the anchor instant, and Tracy wants per-lane arrival
    // in non-decreasing time -- so the anchor marker must not be minted first on these cores.
    for (const auto& t : eth_sync_traces_) {
        tracy_->RegisterEthCore(t.sender_chip, t.snd_x, t.snd_y);
        tracy_->RegisterEthCore(t.receiver_chip, t.rcv_x, t.rcv_y);
    }

    // ---- PER-ETH-CORE ANCHOR (the ORIGIN) ----
    // An eth tile runs at the SAME RATE as a Tensix worker but does NOT share its origin: the two domains
    // bank different counter totals (the same duty-cycle effect that forced the per-DRAM-core anchor).
    // Measured on bh-31: an eth tile had banked ~53 MINUTES more than its chip's worker, so on the chip
    // anchor these rows rendered ~25 minutes right of the workload with their durations still exactly
    // right -- the identical signature to the DRAM case. The anchors below were captured AT SYNC TIME
    // (see EthSyncTrace); re-measuring them here instead cost 36 us of extrapolation error.
    std::set<uint64_t> anchored;
    auto anchor_eth = [&](uint32_t chip, uint32_t nx, uint32_t ny, bool valid, int64_t host_anchor,
                          uint64_t dev_at_anchor) {
        const uint64_t key = (static_cast<uint64_t>(chip) << 40) | (static_cast<uint64_t>(nx) << 20) | ny;
        if (!anchored.insert(key).second) {
            return;
        }
        double freq = 0.0;
        for (const auto& d : devices_) {
            if (d.chip_id == chip) {
                freq = d.freq_ghz;
                break;
            }
        }
        if (!valid || freq <= 0.0) {
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] eth sync lanes: no per-core anchor for chip {} eth NOC0 ({},{}); its "
                "rows will inherit the WORKER anchor and sit far from the workload",
                chip,
                nx,
                ny);
            return;
        }
        tracy_->AddCore(chip, nx, ny, host_anchor, static_cast<double>(dev_at_anchor), freq);
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] eth core anchor: chip {} NOC0 ({},{}) device_time_at_anchor={} cycles on "
            "the chip's shared {:.6f} GHz, offset vs the chip's WORKER anchor {:+.3f} ms [eth and Tensix "
            "share a rate, not an origin]",
            chip,
            nx,
            ny,
            dev_at_anchor,
            freq,
            (static_cast<double>(dev_at_anchor) - static_cast<double>(root_dev_at_anchor_)) / freq / 1e6);
    };
    for (const auto& t : eth_sync_traces_) {
        anchor_eth(t.sender_chip, t.snd_x, t.snd_y, t.snd_anchor_valid, t.snd_host_anchor, t.snd_dev_at_anchor);
        anchor_eth(t.receiver_chip, t.rcv_x, t.rcv_y, t.rcv_anchor_valid, t.rcv_host_anchor, t.rcv_dev_at_anchor);
    }

    static constexpr std::string_view kRttName = "ETH_SYNC_RTT";
    static constexpr std::string_view kEchoName = "ETH_SYNC_ECHO";
    size_t n_zones = 0, n_marks = 0;
    // Traces are stored in measurement order and each trace's trips are chronological, so pushing in this
    // order is already non-decreasing per lane.
    for (const auto& t : eth_sync_traces_) {
        for (const auto& tr : t.trips) {
            perf_debug::WorkerZonePacket z;
            z.chip_id = t.sender_chip;
            z.core_virtual_x = t.snd_x;
            z.core_virtual_y = t.snd_y;
            z.core_noc0_x = t.snd_x;
            z.core_noc0_y = t.snd_y;
            z.risc = 0;
            z.timer_id = 0;
            z.name = kRttName;
            z.start = tr[0];
            z.end = tr[2];
            z.color = 0x27AE60;
            tracy_->HandleWorkerZone(z);
            ++n_zones;
        }
        for (const auto& tr : t.trips) {
            perf_debug::WorkerEventPacket e;
            e.chip_id = t.receiver_chip;
            e.core_virtual_x = t.rcv_x;
            e.core_virtual_y = t.rcv_y;
            e.core_noc0_x = t.rcv_x;
            e.core_noc0_y = t.rcv_y;
            e.risc = 0;
            e.id = 0;
            e.name = kEchoName;
            e.timestamp = tr[1];
            e.num_values = 0;
            tracy_->HandleWorkerEvent(e);
            ++n_marks;
        }
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] eth sync lanes: {} -> {} drawn on eth NOC0 ({},{}) and ({},{}), {} round "
            "trip(s); the peer's ETH_SYNC_ECHO must fall INSIDE the sender's ETH_SYNC_RTT zone",
            t.sender_chip,
            t.receiver_chip,
            t.snd_x,
            t.snd_y,
            t.rcv_x,
            t.rcv_y,
            t.trips.size());
    }
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] eth sync lanes: {} zone(s) + {} marker(s) across {} link(s) -- RAW samples, "
        "not fitted values, so they can contradict the anchors rather than restate them",
        n_zones,
        n_marks,
        eth_sync_traces_.size());
    // Carry the eth cores' host anchors over to the LinkSync entries, which outlive the traces. This is the
    // only place both are in scope, and the close-check's independent cross-check needs them.
    for (const auto& t : eth_sync_traces_) {
        if (!t.snd_anchor_valid || !t.rcv_anchor_valid) {
            continue;
        }
        for (auto& ls : link_syncs_) {
            if (ls.sender_chip == t.sender_chip && ls.receiver_chip == t.receiver_chip) {
                ls.snd_host_anchor = t.snd_host_anchor;
                ls.snd_dev_at_anchor = t.snd_dev_at_anchor;
                ls.rcv_host_anchor = t.rcv_host_anchor;
                ls.rcv_dev_at_anchor = t.rcv_dev_at_anchor;
                ls.host_anchors_valid = true;
                break;
            }
        }
    }
    eth_sync_traces_.clear();
}

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

    // For translating eth logical coords -> NOC0, the coordinate space every Tracy row is keyed by.
    const auto eth_context_id = mesh_device->impl().get_context_id();
    auto& eth_cluster = MetalContext::instance(eth_context_id).get_cluster();
    // Bounded on purpose: 256 samples x 2 rows per link is plenty to see the pattern, and the point of
    // these rows is the causality invariant, not volume.
    constexpr size_t kMaxEthTrips = 256;
    auto stash_trace = [&](IDevice* sd, const CoreCoord& s_eth, IDevice* rd, const CoreCoord& r_eth,
                           const eth_sync::LinkSyncResult& res) {
        if (res.trips.empty()) {
            return;
        }
        const CoreCoord sn = eth_cluster.get_physical_coordinate_from_logical_coordinates(
            sd->id(), s_eth, CoreType::ETH, /*no_warn=*/true);
        const CoreCoord rn = eth_cluster.get_physical_coordinate_from_logical_coordinates(
            rd->id(), r_eth, CoreType::ETH, /*no_warn=*/true);
        EthSyncTrace t;
        t.sender_chip = static_cast<uint32_t>(sd->id());
        t.receiver_chip = static_cast<uint32_t>(rd->id());
        t.snd_x = static_cast<uint32_t>(sn.x);
        t.snd_y = static_cast<uint32_t>(sn.y);
        t.rcv_x = static_cast<uint32_t>(rn.x);
        t.rcv_y = static_cast<uint32_t>(rn.y);
        const CoreCoord sv = eth_cluster.get_virtual_coordinate_from_logical_coordinates(
            sd->id(), s_eth, CoreType::ETH);
        const CoreCoord rv = eth_cluster.get_virtual_coordinate_from_logical_coordinates(
            rd->id(), r_eth, CoreType::ETH);
        t.snd_vx = static_cast<uint32_t>(sv.x);
        t.snd_vy = static_cast<uint32_t>(sv.y);
        t.rcv_vx = static_cast<uint32_t>(rv.x);
        t.rcv_vy = static_cast<uint32_t>(rv.y);
        // Anchor both eth tiles NOW, microseconds-to-milliseconds from their own samples, not seconds.
        const PerfDebugSync sa = sync_device_clock(eth_cluster, static_cast<uint32_t>(sd->id()), sv);
        const PerfDebugSync ra = sync_device_clock(eth_cluster, static_cast<uint32_t>(rd->id()), rv);
        t.snd_anchor_valid = sa.valid;
        t.rcv_anchor_valid = ra.valid;
        t.snd_host_anchor = sa.host_anchor;
        t.rcv_host_anchor = ra.host_anchor;
        t.snd_dev_at_anchor = sa.device_at_anchor;
        t.rcv_dev_at_anchor = ra.device_at_anchor;
        const size_t n = std::min(res.trips.size(), kMaxEthTrips);
        t.trips.reserve(n);
        for (size_t i = 0; i < n; i++) {
            t.trips.push_back({res.trips[i].t0, res.trips[i].t1, res.trips[i].t2});
        }
        eth_sync_traces_.push_back(std::move(t));
    };

    eth_sync::LinkSyncConfig cfg;
    cfg.n_samples = eth_sync_samples();
    cfg.gap_us = eth_sync_gap_us();

    // BFS from the root so every device is reached exactly once, over whichever link the cluster reports
    // first for that pair. Which physical link is picked matters at the nanosecond level (they are not
    // identical lengths), so it is logged.
    // Unordered (min,max) chip pairs already measured, so the closure pass below only measures links
    // that close a CYCLE rather than re-measuring a tree edge from the other side.
    auto pair_key = [](uint32_t x, uint32_t y) {
        return (static_cast<uint64_t>(std::min(x, y)) << 32) | static_cast<uint64_t>(std::max(x, y));
    };
    std::set<uint64_t> measured_pairs;

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
            if (eth_core_is_fabric_claimed(static_cast<uint32_t>(snd->id()), ec) ||
                eth_core_is_fabric_claimed(static_cast<uint32_t>(peer_id), std::get<1>(peer))) {
                continue;  // a fabric router owns one end; a later channel to this peer may still be free
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
            ls.snd_dev = snd;
            ls.rcv_dev = rcv;
            ls.snd_eth = ec;
            ls.rcv_eth = std::get<1>(peer);
            if (ls.valid) {
                eth_sync_parent_edge_[ls.receiver_chip] = link_syncs_.size();
                stash_trace(snd, ec, rcv, std::get<1>(peer), r);
            }
            measured_pairs.insert(pair_key(ls.sender_chip, ls.receiver_chip));
            link_syncs_.push_back(ls);
        }
    }
    // ---- CLOSURE SELF-CHECK ----
    // Every link measured above is a TREE edge, so nothing so far cross-checks anything: a wrong edge just
    // silently moves its whole subtree. A link between two devices the tree already relates is different --
    // its offset is PREDICTED by composing the tree, so measuring it tests the composition end to end over
    // a path that shares no link with the prediction. The disagreement ("closure") is the one number here
    // that is an accuracy bound rather than a fit statistic: residual_rms says the line fits its own
    // samples, closure says two independent routes to the same clock agree.
    //
    // Both directions are composed in DIFFERENCES from each edge's reference instant, for the same reason
    // eth_sync_anchor_for is: the clocks sit near 1e13 cycles, where a double holds only ~0.001 cycle.
    auto edge_to_parent = [](const LinkSync& e, uint64_t t_child) {
        // Inverse of the parent->child map. Forward is t_c = ref + u*rate + offset with u = t_p - ref.
        const double u = (static_cast<double>(static_cast<int64_t>(t_child - e.ref_mid)) -
                          static_cast<double>(e.offset)) / e.rate;
        return static_cast<uint64_t>(static_cast<int64_t>(e.ref_mid) + std::llround(u));
    };
    auto to_root = [&](uint32_t chip, uint64_t t_chip, uint64_t& t_root) {
        uint32_t cur = chip;
        uint64_t t = t_chip;
        for (size_t guard = 0; cur != eth_sync_root_chip_; ++guard) {
            const auto it = eth_sync_parent_edge_.find(cur);
            if (it == eth_sync_parent_edge_.end() || guard > link_syncs_.size()) {
                return false;
            }
            const auto& e = link_syncs_[it->second];
            t = edge_to_parent(e, t);
            cur = e.sender_chip;
        }
        t_root = t;
        return true;
    };

    uint32_t closure_done = 0;
    int64_t worst_closure = 0;
    uint32_t worst_a = 0, worst_b = 0;
    for (IDevice* snd : devices) {
        if (closure_done >= eth_sync_closure_links()) {
            break;
        }
        for (const CoreCoord& ec : snd->get_active_ethernet_cores(true)) {
            if (closure_done >= eth_sync_closure_links()) {
                break;
            }
            std::tuple<ChipId, CoreCoord> peer;
            try {
                peer = snd->get_connected_ethernet_core(ec);
            } catch (const std::exception&) {
                continue;
            }
            const auto peer_id = static_cast<uint32_t>(std::get<0>(peer));
            const auto snd_id = static_cast<uint32_t>(snd->id());
            if (measured_pairs.count(pair_key(snd_id, peer_id)) != 0) {
                continue;  // a tree edge, or a pair already closed
            }
            if (eth_core_is_fabric_claimed(snd_id, ec) ||
                eth_core_is_fabric_claimed(peer_id, std::get<1>(peer))) {
                continue;
            }
            IDevice* rcv = nullptr;
            for (IDevice* d : devices) {
                if (static_cast<uint32_t>(d->id()) == peer_id) {
                    rcv = d;
                    break;
                }
            }
            // Both ends must already hang off the tree, or there is nothing to predict against.
            if (rcv == nullptr || eth_sync_parent_edge_.count(peer_id) == 0) {
                continue;
            }
            measured_pairs.insert(pair_key(snd_id, peer_id));

            const auto r = eth_sync::measure_link(snd, ec, rcv, std::get<1>(peer), cfg);
            if (!r.solution.valid) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] eth sync CLOSURE {} -> {} could not be measured ({}, {}); "
                    "skipping this check",
                    snd_id,
                    peer_id,
                    eth_sync::status_name(r.sender_status),
                    eth_sync::status_name(r.receiver_status));
                continue;
            }

            uint64_t t_root = 0;
            uint64_t pred_b = 0, pred_a = 0;
            double rate_b = 1.0, rate_a = 1.0;
            if (!to_root(snd_id, r.solution.mid_ref, t_root) ||
                !eth_sync_anchor_for(peer_id, t_root, pred_b, rate_b) ||
                !eth_sync_anchor_for(snd_id, t_root, pred_a, rate_a)) {
                continue;
            }
            // Measured: where the redundant link says the peer's clock stood at this same instant.
            const int64_t meas_b = static_cast<int64_t>(r.solution.mid_ref) + r.solution.offset;
            stash_trace(snd, ec, rcv, std::get<1>(peer), r);
            const int64_t closure = meas_b - static_cast<int64_t>(pred_b);
            const double rate_pred = rate_b / rate_a;
            ++closure_done;
            // `closure_done == 1` seeds it: a FIRST result of exactly 0 would fail the > test and leave
            // the reported pair as 0/0, which reads like a real pair rather than "unset".
            if (closure_done == 1 || std::llabs(closure) > std::llabs(worst_closure)) {
                worst_closure = closure;
                worst_a = snd_id;
                worst_b = peer_id;
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] eth sync CLOSURE {} -> {} via eth ({},{}): {:+} cycles vs the "
                "tree-composed prediction, rate {:+.2f} ppm vs {:+.2f} ppm predicted (residual {:.1f} "
                "cycles) [independent route; small = the composition is right]",
                snd_id,
                peer_id,
                ec.x,
                ec.y,
                closure,
                (r.solution.rate - 1.0) * 1e6,
                (rate_pred - 1.0) * 1e6,
                r.solution.residual_rms);
        }
    }
    if (closure_done != 0) {
        eth_sync_worst_closure_ = worst_closure;
        eth_sync_closure_valid_ = true;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] eth sync ACCURACY: worst closure {:+} cycles over {} independent "
            "route(s) ({} vs {}) -- this bounds cross-device alignment error; residual_rms only bounds "
            "each link's own fit",
            worst_closure,
            closure_done,
            worst_a,
            worst_b);
    } else if (eth_sync_closure_links() != 0) {
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] eth sync: no redundant link available, so cross-device alignment is "
            "UNCHECKED (a tree alone cannot detect a bad edge); topology has no cycle among synced devices");
    }

    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start).count();
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] eth sync: {} tree link(s) + {} closure link(s) measured across {} devices "
        "in {} ms ({} samples {} us apart)",
        link_syncs_.size(),
        closure_done,
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
    if (force_aiclk_mhz() != 0) {  // 0 = explicitly disabled
        auto& cl = MetalContext::instance(mesh_device->impl().get_context_id()).get_cluster();
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
            if (!mesh_device->is_local(coord)) {
                continue;
            }
            const int cid = mesh_device->get_device(coord)->id();
            // AUTO: pin at whatever this chip is running at now. The cluster is open, so that is its boosted
            // maximum -- pinning at the current value cannot ask for a frequency the part does not support.
            uint32_t mhz = force_aiclk_mhz();
            if (mhz == kForceAiclkAuto) {
                mhz = static_cast<uint32_t>(cl.get_device_aiclk(cid));
                if (mhz == 0) {
                    log_warning(
                        tt::LogMetal,
                        "[perf-debug profiler] FORCE_AICLK auto: chip {} reports aiclk 0; leaving the clock "
                        "governed (zone placement will decay ~5-8 us/s instead of ~0.05)",
                        cid);
                    continue;
                }
            }
            try {
                cl.get_driver()->get_chip(cid)->arc_msg(0x33, true, {mhz});
                forced_aiclk_chips_.push_back(cid);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal, "[perf-debug profiler] FORCE_AICLK on chip {} failed: {}", cid, e.what());
            }
        }
        if (!forced_aiclk_chips_.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] AICLK FORCED to {} MHz on {} chip(s) (chip 0 now reports {} MHz) -- "
                "the clock is pinned for this capture, so the sync's staleness costs ~0.017 us/s instead of "
                "~5.4 us/s [released at close]",
                force_aiclk_mhz(),
                forced_aiclk_chips_.size(),
                (unsigned)MetalContext::instance(mesh_device->impl().get_context_id())
                    .get_cluster()
                    .get_device_aiclk(forced_aiclk_chips_.front()));
        }
    }
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
    // And the RESIDENT tracker immediately after, in the same free window: its pairs launch once here and
    // then only ever answer mailbox commands, so nothing is launched mid-capture and bring-up drift is
    // already inside the corrections by the time the first zone is stamped.
    start_eth_tracker(mesh_device);

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
            ctx.anchor_host = root_host_anchor_;
            ctx.anchor_dev = derived_clock;
            ctx.anchor_valid = true;
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
        // ONLY THE ROOT MAY HOST-FIT. Every other device composes off it (see the branch below), so
        // there is no reason to spend ~50 ms of MMIO measuring a fit we must not use.
        if (!derived && ctx.chip_id == eth_sync_root_chip_ && !ctx.core_virt.empty()) {
            const CoreCoord w{ctx.core_virt[0].first, ctx.core_virt[0].second};
            // LONG BASELINE, deliberately: 100 samples x 500 us spans ~50 ms instead of ~360 us, cutting the
            // fitted-frequency error by the baseline ratio (~140x). This is the ONE frequency every context on
            // this chip will use (see below), so it is worth 50 ms of a 9-12 s device open to measure it well.
            sync = sync_device_clock(cluster, ctx.chip_id, w, /*spacing_us=*/500);
        }
        if (derived) {
            // already anchored from the root above
        } else if (ctx.chip_id != eth_sync_root_chip_) {
            // DELIBERATELY NO FALLBACK. A non-root device must NEVER take its own independent host
            // fit: the whole design is one host sync at the root plus measured device<->device
            // offsets, because independent per-device fits DISAGREE. Measured on bh-31, as the gap
            // between the two ends of a link where the same instant is drawn twice:
            //     link 1->2  3.5 us     link 0->1  7.1 us     link 0->3  13.7 us
            // constant across the run (7131 ns at the first pair, 7101 ns at the last), against a
            // device<->device sync that closes to -37..-57 ns. Falling back therefore does not
            // "degrade gracefully" -- it silently injects microseconds of cross-device skew that
            // looks like real data, and it did exactly that for the whole of this investigation.
            //
            // With no anchor, GetOrCreateContext returns nullptr and this device simply has no rows.
            // That is the point: absent rows are debuggable, wrong rows are not.
            log_error(
                tt::LogMetal,
                "[perf-debug profiler] Device {} could NOT be anchored from root {} (no measured "
                "device<->device route). Refusing to fall back to an independent host fit -- that "
                "injects us-scale cross-device skew. This device will have NO Tracy rows. Fix the "
                "sync path (see: 'eth sync: 0 tree link(s)' means the init eth sync found no free "
                "channel, because fabric claimed them all).",
                ctx.chip_id,
                eth_sync_root_chip_);
            ctx.anchor_valid = false;
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
            ctx.anchor_host = sync.host_anchor;
            ctx.anchor_dev = sync.device_at_anchor;
            ctx.anchor_valid = true;
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] Device {} clock sync: frequency={:.6f} GHz (aiclk reports {:.6f}), "
                "device_time_at_anchor={} cycles",
                ctx.chip_id,
                sync.frequency,
                freq,
                sync.device_at_anchor);
        } else if (!derived) {
            // Root only -- non-root took the refusal branch above. A root with no host fit anchors
            // nothing at all, so first-marker guessing is all that is left; it is loud on purpose.
            log_error(
                tt::LogMetal,
                "[perf-debug profiler] ROOT device {} clock sync FAILED; falling back to first-marker "
                "anchoring. EVERY device composes off the root, so the whole capture's cross-device "
                "alignment is now guesswork.",
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
        install_fabric_sync_sink();
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
    // Bring-up is over, so the init-time transforms are as stale as they will ever be and no Tracy context
    // has been created yet. This is the last moment a fresh fit can still reach every zone.
    reanchor_after_boot(mesh_device);
    // Runs after the eth re-anchor and is skipped by it when that one succeeded (it sets the same anchors).
    // Independent of it: this one is the path that survives fabric.
    host_reanchor_after_boot(mesh_device);
    start_drift_corrector(mesh_device);
    // After every device has its anchor: the eth samples are rendered through those anchors, so drawing
    // them any earlier would map them with an anchor that does not exist yet.
    emit_eth_sync_lanes();
    start_fabric_sync(mesh_device);
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
// DEFENCE IN DEPTH for the resident-eth ring accounting. The rebase at session start should make this
// unreachable, but an unbounded spin whose only escape is broadcast to the WRONG CORE TYPE is the actual
// reason a ring-accounting slip escalated into a wedged card and a dead session: disarm_producers() walks
// compute_with_storage_grid_size() with CoreType::WORKER and the TENSIX profiler address, so it has never
// reached an eth core, and it only runs in the no-drainer fallback anyway.
//
// So on EVERY normal stop, set PROFILER_TERMINATE on the eth lanes too. While set, a producer that finds
// a full ring proceeds instead of blocking -- markers are dropped, the erisc keeps servicing its router
// and keeps yielding to base FW, and the next session can still bring the link up. The next session's
// control-vector rebase clears the flag again, so this costs nothing while things are healthy.
void PerfDebugProfiler::terminate_eth_producers() {
    const uint32_t one = 1;
    for (const auto& ctx : devices_) {
        if (ctx.eth_prof_l1 == 0 || ctx.n_eth_cores == 0) {
            continue;
        }
        auto& cluster = MetalContext::instance().get_cluster();
        for (uint32_t k = 0; k < ctx.n_eth_cores; k++) {
            const uint32_t idx = ctx.eth_start + k;
            if (idx >= ctx.core_virt.size()) {
                break;
            }
            const CoreCoord v{ctx.core_virt[idx].first, ctx.core_virt[idx].second};
            try {
                cluster.write_core(
                    &one,
                    sizeof(one),
                    tt_cxy_pair(ctx.chip_id, v),
                    ctx.eth_prof_l1 + kernel_profiler::PROFILER_TERMINATE * sizeof(uint32_t));
            } catch (const std::exception&) {
                // A core we cannot reach is not one this can rescue; the rebase is the primary fix.
            }
        }
    }
}

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

    // ---- ETH COVERAGE ------------------------------------------------------------------------------
    //
    // Eth cores run kernels too (fabric routers, and the device<->device clock sync), and their markers go
    // into the same per-RISC SPSC rings a worker uses -- but nothing drained them, so anything an eth core
    // recorded was stranded in its L1. Append them to the poll list so the fillers sweep them like any
    // other core.
    //
    // Their profiler_msg_t sits at a DIFFERENT L1 address (an eth core is its own programmable core type),
    // which is why the drain kernel takes a second base and a split index rather than one address for the
    // whole list. Eth cores go at the END so that split is a single compare.
    //
    // Lane numbering stays uniform (5 slots per core) even though a Blackhole eth core has 2 processors,
    // not 5: profiler_msg_t is control_vector[64] then buffer[PROCESSOR_COUNT], so the control vector --
    // where every SPSC head and tail lives -- is laid out identically, and slots 2-4 simply stay at
    // head == tail == 0 forever because nothing produces into them. Keeping kNRisc uniform is what lets
    // core_of_xy, ctx.nl and the decoder's lane math stay untouched.
    const uint32_t n_worker_only = static_cast<uint32_t>(num_cores);
    uint64_t eth_prof_l1 = 0;
    std::vector<CoreCoord> eth_virt;
    if (eth_coverage()) {
        try {
            eth_prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::PROFILER);
        } catch (const std::exception& e) {
            log_warning(tt::LogMetal, "[perf-debug profiler] no eth profiler region ({}); eth coverage off", e.what());
        }
    }
    if (eth_prof_l1 != 0) {
        IDevice* dev0 = mesh_device->get_device(coord);
        // Active cores only: an inactive eth core has no firmware answering, and the sync work this exists
        // for runs on links that are up.
        for (const CoreCoord& lg : dev0->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true)) {
            const CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(device_id, lg, CoreType::ETH);
            eth_virt.push_back(v);
        }
    }
    if (!eth_virt.empty()) {
        const uint32_t base = static_cast<uint32_t>(coords.size());
        coords.resize(base + eth_virt.size());
        ctx.core_virt.resize(base + eth_virt.size());
        for (size_t k = 0; k < eth_virt.size(); k++) {
            const uint32_t idx = base + static_cast<uint32_t>(k);
            const uint32_t vx = static_cast<uint32_t>(eth_virt[k].x), vy = static_cast<uint32_t>(eth_virt[k].y);
            coords[idx] = (vx & 0xFFFFu) | ((vy & 0xFFFFu) << 16);
            ctx.core_of_xy[coords[idx]] = idx;
            // REBASE, NOT A ZERO -- this line used to write zero_ctrl and that is what wedged cards.
            //
            // A WORKER's kernel relaunches per program, so its producer-local `wIndex` resets alongside a
            // zeroed HEAD and the two stay in step. An eth core running a RESIDENT router never relaunches:
            // `wIndex` persists across profiler sessions. Zeroing HEAD here therefore makes
            // (wIndex - HEAD) == wIndex, and the moment that exceeds RING_USABLE the ring LOOKS permanently
            // full -- the next checked emit parks in ring_ensure_room_slow's unbounded spin, the erisc never
            // returns to the router loop or calls risc_context_switch(), base FW never ticks its heartbeat,
            // and the NEXT session's device open dies at llrt.cpp:603 with "Timed out while waiting for
            // active ethernet core ... to become active again".
            //
            // Measured on bh-31 (ring 506 usable words, 64 words/round): 6 rounds = 384 w survived,
            // 11 rounds = 704 w wedged at session 2. So carry HEAD forward to the producer's own published
            // TAIL rather than resetting it under a producer that never restarted.
            std::vector<uint32_t> eth_ctrl(zero_ctrl.size() / sizeof(uint32_t), 0);
            {
                std::vector<uint32_t> live(zero_ctrl.size() / sizeof(uint32_t), 0);
                cluster.read_core(
                    live.data(), (uint32_t)zero_ctrl.size(), tt_cxy_pair(device_id, eth_virt[k]), eth_prof_l1);
                for (uint32_t r = 0; r < kernel_profiler::PROFILER_SPSC_MAX_RISC; r++) {
                    // HEAD := TAIL leaves (wIndex - HEAD) at the handful of words the producer wrote but
                    // has not published yet, instead of its whole lifetime count.
                    const uint32_t tail = live[kernel_profiler::SPSC_RING_TAIL_0 + r];
                    eth_ctrl[kernel_profiler::SPSC_RING_HEAD_0 + r] = tail;
                    eth_ctrl[kernel_profiler::SPSC_RING_TAIL_0 + r] = tail;
                }
            }
            cluster.write_core(
                eth_ctrl.data(), (uint32_t)zero_ctrl.size(), tt_cxy_pair(device_id, eth_virt[k]), eth_prof_l1);
            // SPSC_CORE_XY, which on a WORKER the BRISC FW rewrites at every launch (after this
            // zeroing). An eth core gets no launches while a resident router runs, so the identity
            // must come from here -- without it every eth span frame decodes as unknown-core and is
            // discarded wholesale.
            const uint32_t xy = (vy << 16) | vx;
            cluster.write_core(
                &xy,
                sizeof(xy),
                tt_cxy_pair(device_id, eth_virt[k]),
                eth_prof_l1 + kernel_profiler::SPSC_CORE_XY * 4u);
            ctx.core_virt[idx] = {vx, vy};
        }
        ctx.nl = static_cast<uint32_t>(coords.size()) * kNRisc;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] Device {}: eth coverage ON -- {} active eth cores appended after {} workers "
            "(eth profiler L1 0x{:x})",
            device_id,
            eth_virt.size(),
            n_worker_only,
            eth_prof_l1);
    }
    ctx.n_eth_cores = static_cast<uint32_t>(eth_virt.size());
    ctx.eth_start = n_worker_only;
    ctx.eth_prof_l1 = static_cast<uint32_t>(eth_prof_l1);

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
        // Slice the WHOLE poll list -- workers PLUS the appended eth cores. num_cores counts workers only,
        // so slicing on it would enumerate the eth cores and then sweep none of them.
        const uint64_t n_poll = static_cast<uint64_t>(coords.size());
        const uint32_t lo = is_mover ? 0u : static_cast<uint32_t>((n_poll * slice) / n_slices);
        const uint32_t hi = is_mover ? 0u : static_cast<uint32_t>((n_poll * (slice + 1)) / n_slices);
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
            // Eth cores sit at the END of the global poll list, so a filler's slice [lo,hi) contains them
            // only if it reaches past eth_start; translate the global split into this slice's local one.
            const uint32_t local_eth_start =
                (ctx.eth_start > lo) ? std::min<uint32_t>(ctx.eth_start - lo, my_cores) : 0u;
            std::vector<uint32_t> rt = {
                my_cores, static_cast<uint32_t>(prof_l1), ctx.eth_prof_l1, local_eth_start};
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

    // Stop the in-router sync DEVICES first (no new rounds), while the fillers are still sweeping,
    // so the final in-flight round's packets still reach the aggregator. The host half joins later,
    // next to the other trackers.
    fabric_sync_disable_devices();

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
    // Last, with the drainers quiesced so the links are quiet and nothing else is competing for the NoC.
    stop_drift_corrector();
    stop_fabric_sync();  // before stop_eth_tracker: the teardown cross-check reads the tracker's edges
    stop_eth_tracker();
    terminate_eth_producers();
    cross_path_tracking_test();
    check_sync_drift_at_close();
    check_anchor_staleness_at_close();
    if (!forced_aiclk_chips_.empty()) {
        auto& cl = MetalContext::instance().get_cluster();
        for (const int cid : forced_aiclk_chips_) {
            try {
                cl.get_driver()->get_chip(cid)->arc_msg(0x33, true, {0});
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogMetal, "[perf-debug profiler] FORCE_AICLK release on chip {} failed: {}", cid,
                    e.what());
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] AICLK force RELEASED on {} chip(s); chip {} back to {} MHz",
            forced_aiclk_chips_.size(),
            forced_aiclk_chips_.front(),
            (unsigned)cl.get_device_aiclk(forced_aiclk_chips_.front()));
        forced_aiclk_chips_.clear();
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
        // ADDRESS PER CORE CLASS. Eth cores are appended after the workers in this poll list and their
        // profiler_msg_t lives at a DIFFERENT L1 base (ctx.eth_prof_l1); reading them at the TENSIX
        // address lands in an unrelated region and returns whatever happens to be there.
        //
        // That is not hypothetical: with eth coverage on, this function reported
        //   "30183953311 producer stalls across 2 of 118 cores"  and
        //   "580/590 lanes fully drained; 10 lanes stranded 19118519543 words (worst lane 552)"
        // on a capture that was independently EXACTLY lossless (samples == rounds x links x 16 x 3).
        // 118 = 110 workers + 8 eth, and lane 552 is in the eth range (110 workers x 5 riscs = 550),
        // so every bogus lane was an eth lane read at the wrong base. A completeness checker that
        // calls a perfect capture broken is worse than no checker: it trains people to ignore it.
        const bool is_eth =
            ctx.n_eth_cores != 0 && ci >= ctx.eth_start && ci < static_cast<size_t>(ctx.eth_start) + ctx.n_eth_cores;
        if (is_eth && ctx.eth_prof_l1 == 0) {
            continue;  // eth coverage without a resolved base: nothing safe to read
        }
        const uint64_t core_prof_l1 = is_eth ? static_cast<uint64_t>(ctx.eth_prof_l1) : prof_l1;
        cluster.read_core(
            cv.data(),
            kernel_profiler::SPSC_CONTROL_END * sizeof(uint32_t),
            tt_cxy_pair(ctx.chip_id, CoreCoord{vx, vy}),
            core_prof_l1);
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
