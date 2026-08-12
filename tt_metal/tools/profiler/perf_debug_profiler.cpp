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
#include <cstring>
#include <thread>
#include <x86intrin.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

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

// Largest payload the 7-bit wire size field can express, in uint64s -- bounds the consumer's scratch.
constexpr uint32_t kMaxEventValues = 64;

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <vector>
// Physical-core pin plan (single-NUMA 6c/12t here). Producers on phys 0–1, decode on next N phys,
// publisher on SMT sibling of last decode -- never on an SMT sibling of a producer.
enum class PinRole : int { Producer0 = 0, Producer1 = 1, Decode = 2, Publisher = 3 };
struct PinPlan {
    bool on = false;
    int prod[2] = {-1, -1};
    std::vector<int> decode;  // physical CPUs for decode workers
    int publish = -1;
};
static const PinPlan& pin_plan() {
    static const PinPlan p = [] {
        PinPlan out;
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_PIN_BASE");
        if (s != nullptr && *s != '\0' && std::strtol(s, nullptr, 10) < 0) {
            return out;  // disabled
        }
        const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        if (ncpu <= 0) {
            return out;
        }
        std::vector<bool> seen(static_cast<size_t>(ncpu), false);
        std::vector<int> phys;
        for (long c = 0; c < ncpu; ++c) {
            if (seen[static_cast<size_t>(c)]) {
                continue;
            }
            phys.push_back(static_cast<int>(c));
            std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            std::string line;
            if (std::getline(in, line)) {
                for (char& ch : line) {
                    if (ch == ',' || ch == '-') {
                        ch = ' ';
                    }
                }
                std::stringstream ss(line);
                int id;
                while (ss >> id) {
                    if (id >= 0 && id < ncpu) {
                        seen[static_cast<size_t>(id)] = true;
                    }
                }
            } else {
                seen[static_cast<size_t>(c)] = true;
            }
        }
        if (phys.size() < 4) {
            return out;
        }
        out.on = true;
        // Two producers (one per socket) on phys 0–1.
        out.prod[0] = phys[0];
        out.prod[1] = phys[1];
        constexpr size_t prod_cores = 2u;
        // Default is 3 decode workers. Pin them on dedicated phys cores after the producers.
        // Publisher must NOT share an SMT sibling with a live decode worker (measured: that alone
        // drops marker-wire ~20 → ~15 GB/s). Prefer the next free phys core's SMT; fall back to the
        // last decode core's sibling only when the machine is too small.
        const size_t n_decode = std::min<size_t>(3, phys.size() > prod_cores ? phys.size() - prod_cores : 0);
        for (size_t i = 0; i < n_decode; ++i) {
            out.decode.push_back(phys[prod_cores + i]);
        }
        if (out.decode.empty()) {
            out.decode.push_back(phys.back());
        }
        {
            const size_t pub_phys_i = prod_cores + n_decode;
            const int pub_phys = (pub_phys_i < phys.size()) ? phys[pub_phys_i] : out.decode.back();
            std::ifstream in(
                "/sys/devices/system/cpu/cpu" + std::to_string(pub_phys) + "/topology/thread_siblings_list");
            std::string line;
            out.publish = pub_phys;
            if (std::getline(in, line)) {
                for (char& ch : line) {
                    if (ch == ',' || ch == '-') {
                        ch = ' ';
                    }
                }
                std::stringstream ss(line);
                int id;
                while (ss >> id) {
                    if (id != pub_phys) {
                        out.publish = id;
                        break;
                    }
                }
            }
        }
        return out;
    }();
    return p;
}
static int pin_cpu_for(PinRole role, uint32_t index = 0) {
    const PinPlan& p = pin_plan();
    if (!p.on) {
        return -1;
    }
    switch (role) {
        case PinRole::Producer0: return p.prod[0];
        case PinRole::Producer1: return p.prod[1];
        case PinRole::Decode: return p.decode[index % static_cast<uint32_t>(p.decode.size())];
        case PinRole::Publisher: return p.publish;
    }
    return -1;
}
static void pin_self_to_cpu(int cpu) {
    if (cpu < 0) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(cpu), &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
#else
enum class PinRole : int { Producer0 = 0, Producer1 = 1, Decode = 2, Publisher = 3 };
static int pin_cpu_for(PinRole, uint32_t = 0) { return -1; }
static void pin_self_to_cpu(int) {}
#endif

#if defined(__x86_64__)
// Zone-block emit into 12 B {ts,meta} records.
static uint32_t emit_zone_block_avx2(
    const uint32_t* lin, uint32_t nwords, uint32_t base_meta, uint32_t hi, uint32_t /*prog*/, PerfDebugRec* out) {
    uint32_t k = 0;
    for (uint32_t i = 0; i + 1 < nwords; i += 2) {
        _mm_prefetch(reinterpret_cast<const char*>(lin + i + 16), _MM_HINT_T0);
        out[k++] = PerfDebugRec{
            pp_full_ts(hi, lin[i + 1]), ((lin[i] >> 27) << kRecTypeShift) | base_meta | (lin[i] & 0xFFFFu)};
    }
    return k;
}
#endif

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
bool role_split() {
    static const bool v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_ROLE_SPLIT");
        return s != nullptr && *s != '\0' && *s != '0';
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

// Hold off on a sock read until at least this many pages are pending, so FIFO→staging memcpy sees
// FINDINGS-sized chunks (~64 KB+). 0 = disabled (default). Override: TT_METAL_PERF_DEBUG_MIN_PAGES.
// When set, drain_pass still takes sub-min tails once stop_ is set so teardown cannot strand pages.
uint32_t min_pages_per_read() {
    static const uint32_t v = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_MIN_PAGES");
        if (s == nullptr || *s == '\0') {
            return 0u;
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

// TT_METAL_PERF_DEBUG_D2H_DISCARD=1: wait for pages, pop+ack, NEVER memcpy out of the FIFO.
// Isolates the PCIe/flow-control ceiling (FINDINGS: ~57 GB/s with discard vs ~25 GB/s with memcpy).
bool d2h_discard_only() {
    static const bool on = [] {
        const char* s = std::getenv("TT_METAL_PERF_DEBUG_D2H_DISCARD");
        return s != nullptr && *s != '\0' && *s != '0';
    }();
    return on;
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
PerfDebugProfiler::DeviceCtx::DeviceCtx(DeviceCtx&& o) noexcept :
    chip_id(o.chip_id),
    params_addr(o.params_addr),
    nl(o.nl),
    device(o.device),
    n_drisc(o.n_drisc),
    dram_frames(o.dram_frames),
    core_virt(std::move(o.core_virt)),
    virt_to_noc0(std::move(o.virt_to_noc0)),
    active(o.active),
    hz_raw(std::move(o.hz_raw)),
    nharts(o.nharts),
    marker_ts_base(o.marker_ts_base),
    synced(o.synced),
    freq_ghz(o.freq_ghz) {
    for (uint32_t i = 0; i < kNSockets; i++) {
        sockets[i] = std::move(o.sockets[i]);
        decode[i] = std::move(o.decode[i]);
    }
    for (uint32_t i = 0; i < kMaxDecodeWorkers * kNSockets; i++) {
        wdecode[i] = std::move(o.wdecode[i]);
    }
    for (uint32_t d = 0; d < kMaxDrisc; d++) {
        drain_program[d] = std::move(o.drain_program[d]);
        drisc_logical[d] = o.drisc_logical[d];
        drisc_virtual[d] = o.drisc_virtual[d];
        drisc_l1_noc[d] = o.drisc_l1_noc[d];
        drisc_l1_base[d] = o.drisc_l1_base[d];
        stop_addr[d] = o.stop_addr[d];
        done_addr[d] = o.done_addr[d];
        results_addr[d] = o.results_addr[d];
        role[d] = o.role[d];
        sock_of[d] = o.sock_of[d];
        hs_addr[d] = o.hs_addr[d];
        n_peer[d] = o.n_peer[d];
        dram_bank[d] = o.dram_bank[d];
        dram_addr[d] = o.dram_addr[d];
        for (uint32_t p = 0; p < kNPeerMax; p++) {
            peer_of[d][p] = o.peer_of[d][p];
        }
    }
    for (uint32_t s = 0; s < kNSockets; s++) {
        auto& dst = sock_state[s];
        auto& src = o.sock_state[s];
        dst.buf = std::move(src.buf);
        dst.words = src.words;
        dst.ranges = std::move(src.ranges);
        dst.carry = std::move(src.carry);
        dst.carry_pend = std::move(src.carry_pend);
        dst.fill_seq.store(src.fill_seq.load(std::memory_order_relaxed), std::memory_order_relaxed);
        for (uint32_t w = 0; w < kMaxDecodeWorkers; w++) {
            dst.wdone[w].store(src.wdone[w].load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        dst.iters = src.iters;
        dst.pages = src.pages;
        dst.stall = src.stall;
        dst.read_ns = src.read_ns;
        dst.wait_ns = src.wait_ns;
        dst.copy_ns = src.copy_ns;
        dst.ack_ns = src.ack_ns;
        dst.poll_ns = src.poll_ns;
        dst.polls = src.polls;
        dst.reads = src.reads;
        dst.bytes = src.bytes;
        dst.wall_ns = src.wall_ns;
        dst.quiesce = src.quiesce;
        dst.done = src.done;
        dst.overflow_reported = src.overflow_reported;
    }
}

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
            ctx.freq_ghz = freq;
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

    // Spawn AFTER devices_ is stable (the threads index into it): 2 producers (one per socket),
    // DECODE_WORKERS decode threads, one publisher, one consumer (ring -> Tracy). The slow Tracy
    // sink stays off the drain path.
    if (!devices_.empty()) {
        const uint32_t cap = max_pages_per_read(kMaxPagesPerRead);
        // Records per read: a page holds at most page_words/2 two-word markers.
        const size_t recs_per_page = (kPageSize / sizeof(uint32_t)) / 2;
        read_chunk_recs_ = cap ? static_cast<size_t>(cap) * recs_per_page : static_cast<size_t>(kHRingWords);
        pub_last_ts_.assign(static_cast<size_t>(kNSockets) * devices_.front().nl, 0);
        // Pre-size every socket's ping-pong staging buffers to the max read. Hot-path drain_pass must
        // NEVER allocate or value-init.
        const size_t max_read_words = static_cast<size_t>(cap ? cap : kHRingWords) * (kPageSize / sizeof(uint32_t));
        for (auto& ctx : devices_) {
            for (uint32_t s = 0; s < kNSockets; s++) {
                auto& ss = ctx.sock_state[s];
                for (auto& b : ss.buf) {
                    b.assign(max_read_words, 0u);
#if defined(__linux__)
                    {
                        void* p = b.data();
                        const size_t bytes = b.size() * sizeof(uint32_t);
                        if (bytes >= (size_t{1} << 20)) {
                            madvise(p, bytes, MADV_HUGEPAGE);
                            madvise(p, bytes, MADV_WILLNEED);
                        }
                    }
#endif
                }
                ss.fill_seq.store(0, std::memory_order_relaxed);
                for (auto& wd : ss.wdone) {
                    wd.store(0, std::memory_order_relaxed);
                }
            }
        }
        // Pre-size ping-pong publish batches to the worst-case record yield of ONE decode call:
        // max staged words + the decode-state residual carry, at the worst records-per-word ratio
        // (a bare 2-word PP_DATA yields head + EXT = 2 records, 1/word). decode_into_ring relies on
        // this to append with NO per-marker capacity check.
        const size_t words_max =
            (cap ? static_cast<size_t>(cap) * (kPageSize / sizeof(uint32_t)) : static_cast<size_t>(kHRingWords)) +
            kDecodeCarryWords;
        const size_t batch_cap = words_max + 16;
        // 3 decode workers (default) → L3 scratch batches → publisher NT-publish_batch into BroadcastRing.
        // Matches 4-worker marker-wire; 2 workers are decode-bound (~3 ms vs publish ~2.3 ms).
        n_workers_ = [] {
            const char* s = std::getenv("TT_METAL_PERF_DEBUG_DECODE_WORKERS");
            uint32_t n = (s != nullptr && *s != '\0') ? static_cast<uint32_t>(std::strtoul(s, nullptr, 10)) : 3u;
            return std::clamp<uint32_t>(n, 1u, kMaxDecodeWorkers);
        }();
        for (uint32_t w = 0; w < n_workers_; w++) {
            for (auto& b : wpub_[w].batches) {
                b.recs.assign(batch_cap, PerfDebugRec{});
                b.n = 0;
#if defined(__linux__)
                // Ask THP for the scratch so emit stays in huge pages / fewer TLB misses vs 4K.
                {
                    void* p = b.recs.data();
                    const size_t bytes = b.recs.size() * sizeof(PerfDebugRec);
                    if (bytes >= (size_t{1} << 20)) {
                        madvise(p, bytes, MADV_HUGEPAGE);
                        madvise(p, bytes, MADV_WILLNEED);
                        std::memset(p, 0, bytes);  // fault + warm on bring-up thread
                    }
                }
#endif
            }
            wpub_[w].prod.store(0, std::memory_order_relaxed);
            wpub_[w].cons.store(0, std::memory_order_relaxed);
        }
        TT_FATAL(
            devices_.size() <= kRecDevMax,
            "[perf-debug profiler] {} devices exceed the record meta's {}-device field",
            devices_.size(),
            kRecDevMax);
        for (const auto& d : devices_) {
            TT_FATAL(
                d.nl <= kRecLaneMax,
                "[perf-debug profiler] {} lanes exceed the record meta's {}-lane field",
                d.nl,
                kRecLaneMax);
        }
        publisher_stop_.store(false, std::memory_order_relaxed);
        decoder_stop_.store(false, std::memory_order_relaxed);
        // One producer per socket (poll/copy/ack/split). Scratch decode + dedicated publisher
        // publish_batch (NT) into a single BroadcastRing.
        // TT_METAL_PERF_DEBUG_NO_CONSUMER=1 skips only the Tracy consumer -- the ring is still fed.
        static const bool no_consumer = [] {
            const char* s = std::getenv("TT_METAL_PERF_DEBUG_NO_CONSUMER");
            return s != nullptr && *s != '\0' && *s != '0';
        }();
        ring_ = std::make_unique<RecRingHolder>(ring_capacity_recs());
        ring_->ring.warm_pages(/*lock=*/true);
        publisher_ = std::thread(&PerfDebugProfiler::publisher_thread, this);
        for (uint32_t w = 0; w < n_workers_; w++) {
            workers_.emplace_back(&PerfDebugProfiler::decode_worker, this, w);
        }
        for (uint32_t sk = 0; sk < kNSockets; sk++) {
            producers_[sk] = std::thread(&PerfDebugProfiler::producer_thread, this, sk);
        }
        const uint32_t n_con = no_consumer ? 0u : 1u;
        if (!no_consumer) {
            consumer_ = std::thread(&PerfDebugProfiler::consumer_thread, this);
        }
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] host threads: {} producer + {} decode + 1 publisher + {} consumer "
            "(total {}) | scratch+publisher",
            kNSockets,
            n_workers_,
            n_con,
            kNSockets + n_workers_ + 1u + n_con);
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
    const bool rsplit = role_split() && !tensix_drain;
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
        ctx.n_drisc = kNSockets;
        const int bank_ov_pre = drisc_bank_override();
        for (uint32_t d = 0; d < kNSockets; d++) {
            ctx.role[d] = kRoleFull;
            ctx.sock_of[d] = d;
            banks.push_back(
                (bank_ov_pre >= 0) ? static_cast<uint32_t>((static_cast<uint32_t>(bank_ov_pre) + d) % nbanks)
                                   : kSafeBanks[d]);
            ringbank.push_back(0);
        }
    }
    for (uint32_t d = 0; d < ctx.n_drisc; d++) {
        ctx.dram_bank[d] = ringbank[d];
    }

    const uint32_t span_bytes_all =
        (kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE + kNRisc * kernel_profiler::PROFILER_L1_VECTOR_SIZE) *
        sizeof(uint32_t);
    const uint32_t slot_bytes_all = kernel_profiler::SPSC_SPAN_PREFIX_WORDS * sizeof(uint32_t) + span_bytes_all;

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
    // Frames are still whole: capacity truncates to a multiple of the slot frame, so a FRAME never
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
        const uint32_t stage_base = ctx.drisc_l1_base[d];
        const uint32_t head_scratch = stage_base + nstage * slot_bytes;
        ctx.done_addr[d] = head_scratch + kScratchBytes;
        ctx.stop_addr[d] = ctx.done_addr[d] + 64;
        ctx.results_addr[d] = ctx.stop_addr[d] + 64;
        // The role-split handshake block. Allocated for every role so the L1 layout (and hence every other
        // address) is identical whether the knob is on or off -- a mover reads its FILLER's block, and that
        // only works because both cores lay their L1 out the same way.
        ctx.hs_addr[d] = ctx.results_addr[d] + 256;
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
                // sender_is_l2cpu switches the socket between "physical NoC coord + full L1 address" (DRISC,
                // drainer) and the normal worker path (logical coord, worker-L1 semantics). The socket picks the
                // static-vs-dynamic write path by ASKING UMD whether this core has a window (see init_sender_tlb),
                // so the window configured just above is what puts the DRISC on the static path.
                g_bringup_step = fmt::format("drainer {}: D2HSocket construct (writes config into DRISC L1)", d);
                ctx.sockets[sk] = std::make_unique<distributed::D2HSocket>(
                    mesh_device,
                    distributed::MeshCoreCoord{
                        scoord, tensix_drain ? ctx.drisc_logical[d] : CoreCoord(drisc_phys.x, drisc_phys.y)},
                    static_cast<uint32_t>((static_cast<uint64_t>(kHRingWords) * 4 / kPageSize) * kPageSize),
                    distributed::D2HSocket::ExternalConfigBuffer{.address = cfg_l1, .sender_is_l2cpu = !tensix_drain});
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
                // no stream to decode: its frames reach the host through its mover's socket. ctx.decode[sk]
                // stays the producer's splitter lookup (xy_core); the workers decode with their own
                // per-(worker, socket) states below.
                ctx.decode[sk] = std::make_unique<pz::SpscDecodeState>();
                ctx.decode[sk]->reset(ctx.nl);
                for (uint32_t c = 0; c < num_cores; c++) {
                    TT_FATAL(
                        ctx.decode[sk]->set_core_xy(coords[c], c),
                        "[perf-debug profiler] virtual coord 0x{:x} exceeds SpscDecodeState::kXyGrid={} -- raise "
                        "the dense xy_core table",
                        coords[c],
                        pz::SpscDecodeState::kXyGrid);
                }
                for (uint32_t w = 0; w < kMaxDecodeWorkers; w++) {
                    auto& wd = ctx.wdecode[w * kNSockets + sk];
                    wd = std::make_unique<pz::SpscDecodeState>();
                    wd->reset(ctx.nl);
                    std::memcpy(wd->xy_core, ctx.decode[sk]->xy_core, sizeof(wd->xy_core));
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
            uint32_t zero1 = 0;
            cluster.write_core(
                &zero1,
                sizeof(zero1),
                tt_cxy_pair(device_id, ctx.drisc_virtual[d]),
                ctx.drisc_l1_noc[d] + (ctx.stop_addr[d] - ctx.drisc_l1_base[d]));
            // Same reason, for the results block: it is published only on kernel exit, so a drainer that is
            // still running at teardown leaves the PREVIOUS run's numbers there and they read as this run's.
            // That is how a 42 s run reported "495.7 ms, credit-wait 0.1%" and hid its own credit timeouts.
            const std::vector<uint32_t> zero_res(64, 0);
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
                nstage,
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
                peer_addr[1]};
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
            log_warning(
                tt::LogMetal,
                "[perf-debug profiler] Device {}: DRISC {} failed to start ({}); continuing without capture",
                device_id,
                d,
                e.what());
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

    return true;
}

// Decode one slot's whole-frame ranges into an L3 scratch PublishBatch, then hand off to the
// dedicated publisher via the per-worker SPSC (publish_submit_batch). Early ACK already happened
// in drain_pass -- this never touches the FIFO.
void PerfDebugProfiler::decode_ranges(DeviceCtx& ctx, uint32_t sock_idx, uint32_t worker, size_t slot) {
    DeviceCtx::SockState& ss = ctx.sock_state[sock_idx];
    auto& ranges = ss.ranges[slot][worker];
    if (ranges.empty()) {
        return;
    }
    pz::SpscDecodeState& st = *ctx.wdecode[worker * kNSockets + sock_idx];
    WorkerPub& wp = wpub_[worker];
    size_t total_words = 0;
    for (const auto& r : ranges) {
        total_words += r.second;
    }
    const auto t_dec_all = std::chrono::steady_clock::now();
    if (stall_only()) {
        for (const auto& r : ranges) {
            pz::spsc_decode(
                st, r.first, r.second, ctx.nl, [&](uint32_t, uint32_t type, uint32_t hash, uint64_t, uint32_t) {
                    if (hash == 0x7FFFu && type == PP_ZONE_START) {
                        ss.stall++;
                        wp.stall++;
                    }
                    wp.emit[sock_idx]++;
                });
        }
        wp.decode_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_dec_all).count();
        return;
    }
    if (!ring_) {
        return;
    }

    const uint32_t dev_idx = static_cast<uint32_t>(&ctx - devices_.data());
    const uint32_t dev_bits = dev_idx << kRecDevShift;
    const size_t bound = total_words + st.resid.size() + 16;

    auto emit_into = [&](PerfDebugRec* out, size_t& k) {
        for (const auto& r : ranges) {
            pz::spsc_decode(
                st,
                r.first,
                r.second,
                ctx.nl,
                [&](uint32_t lane, uint32_t type, uint32_t hash, uint64_t ts, uint32_t /*prog*/) {
                    if (type > PP_ZONE_END) {
                        return;
                    }
                    out[k++] = PerfDebugRec{ts, (type << kRecTypeShift) | dev_bits | (lane << kRecLaneShift) | hash};
                },
                [&](uint32_t lane,
                    uint32_t type,
                    uint32_t id,
                    uint64_t ts,
                    uint32_t /*prog*/,
                    const uint32_t* payload,
                    uint32_t n) {
                    const uint32_t ml = dev_bits | (lane << kRecLaneShift);
                    const uint32_t rt = (type == PP_EVENT) ? kRecEvent : kRecData;
                    out[k++] = PerfDebugRec{ts, (rt << kRecTypeShift) | ml | (id & 0xFFFFu)};
                    out[k++] = PerfDebugRec{(static_cast<uint64_t>(id) << 32) | n, (kRecExt << kRecTypeShift) | ml};
                    for (uint32_t c = 0; c < (n + 1u) / 2u; c++) {
                        const uint64_t hi = payload[2 * c];
                        const uint64_t lo = (2 * c + 1 < n) ? payload[2 * c + 1] : 0u;
                        out[k++] = PerfDebugRec{(hi << 32) | lo, (kRecCont << kRecTypeShift) | ml};
                    }
                },
                [&](const uint32_t* p, uint32_t nw, uint32_t lane, uint32_t bhi, uint32_t bprog) {
#if defined(__x86_64__)
                    k += emit_zone_block_avx2(p, nw, dev_bits | (lane << kRecLaneShift), bhi, bprog, out + k);
#else
                    const uint32_t ml = dev_bits | (lane << kRecLaneShift);
                    for (uint32_t j = 0; j < nw; j += 2) {
                        out[k++] = PerfDebugRec{
                            pp_full_ts(bhi, p[j + 1]), ((p[j] >> 27) << kRecTypeShift) | ml | (p[j] & 0xFFFFu)};
                    }
#endif
                });
        }
    };

    const auto t_dec0 = std::chrono::steady_clock::now();
    size_t k = 0;

    PublishBatch* cur = publish_acquire_batch(worker);
    TT_FATAL(
        bound <= cur->recs.size(),
        "[perf-debug profiler] publish batch under-sized: {} bound vs {} rec slots",
        bound,
        cur->recs.size());
    PerfDebugRec* const out = cur->recs.data();
    emit_into(out, k);
    wp.decode_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_dec0).count();
    cur->n = k;
    if (k != 0) {
        publish_submit_batch(worker);
    }

    if (k != 0) {
        if (ctx.marker_ts_base == 0) {
            // First record ts; benign if two sockets race the latch.
            ctx.marker_ts_base = 1;
        }
        wp.emit[sock_idx] += k;
        wp.zone_recs += k;
        wp.recs += k;
        wp.last_rec_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }
}

// Producer-side splitter. Walks the staged slot packet-by-packet (a BULK_SPAN hop reads 2 header words
// per ~10 KB frame) and routes each WHOLE frame to worker = core * n_workers / num_cores; anything that
// is not an identifiable frame goes to worker 0. A packet cut at the slot boundary is copied out into
// carry_pend (the slot's staging gets recycled) and completed from the next slot into carry[slot], which
// that slot's ranges then reference -- so workers only ever see whole packets and their decode states
// never carry a residual.
void PerfDebugProfiler::split_slot(
    DeviceCtx& ctx, uint32_t sock_idx, size_t slot, const uint32_t* stage, size_t words) {
    DeviceCtx::SockState& ss = ctx.sock_state[sock_idx];
    for (uint32_t w = 0; w < n_workers_; w++) {
        ss.ranges[slot][w].clear();
    }
    const auto& xymap = *ctx.decode[sock_idx];
    const uint32_t nl = ctx.nl;
    const uint32_t ncores = nl / kNRisc;
    const uint32_t nworkers = n_workers_;
    auto worker_of_frame = [&](const uint32_t* frame) -> uint32_t {
        if (nworkers == 1 || !pp_is_bulkspan(frame[0])) {
            return 0;
        }
        const uint32_t xy = frame[kernel_profiler::SPSC_SPAN_PREFIX_WORDS + kernel_profiler::SPSC_CORE_XY];
        const uint32_t core = xymap.lookup_core(xy);
        return core == 0xFFFFFFFFu ? 0u : (core * nworkers) / ncores;
    };
    auto emit_range = [&](uint32_t w, const uint32_t* ptr, uint32_t len) {
        auto& rl = ss.ranges[slot][w];
        // The mover ships same-filler batches, so adjacent frames usually share a worker: coalesce.
        if (!rl.empty() && rl.back().first + rl.back().second == ptr) {
            rl.back().second += len;
        } else {
            rl.emplace_back(ptr, len);
        }
    };

    size_t p = 0;
    // Complete the packet carried from the previous slot, if any.
    if (!ss.carry_pend.empty()) {
        auto& pend = ss.carry_pend;
        for (;;) {
            if (pend.size() >= 2) {
                const size_t need = pz::spsc_top_packet_words(pend[0], pend[1]);
                if (pend.size() >= need) {
                    break;
                }
                const size_t take = std::min(need - pend.size(), words - p);
                pend.insert(pend.end(), stage + p, stage + p + take);
                p += take;
                if (pend.size() < need) {
                    // The whole slot went into the still-incomplete packet (possible only for tiny reads).
                    return;
                }
                break;
            }
            if (p >= words) {
                return;
            }
            pend.push_back(stage[p++]);
        }
        ss.carry[slot].swap(pend);
        pend.clear();
        emit_range(
            worker_of_frame(ss.carry[slot].data()),
            ss.carry[slot].data(),
            static_cast<uint32_t>(ss.carry[slot].size()));
    }
    while (p < words) {
        if (p + 1 >= words) {
            if (pp_is_src(stage[p]) || pp_is_timer(stage[p])) {
                emit_range(0, stage + p, 1);  // a complete 1-word packet needs no carry
                return;
            }
            ss.carry_pend.assign(stage + p, stage + words);
            return;
        }
        const size_t need = pz::spsc_top_packet_words(stage[p], stage[p + 1]);
        if (p + need > words) {
            ss.carry_pend.assign(stage + p, stage + words);
            return;
        }
        emit_range(worker_of_frame(stage + p), stage + p, static_cast<uint32_t>(need));
        p += need;
    }
}

bool PerfDebugProfiler::drain_pass(DeviceCtx& ctx, uint32_t sock_idx) {
    distributed::D2HSocket* sock = ctx.sockets[sock_idx].get();
    if (sock == nullptr) {
        return false;
    }
    DeviceCtx::SockState& ss = ctx.sock_state[sock_idx];
    const uint32_t page_words = kPageSize / sizeof(uint32_t);
    static const bool ddbg = (std::getenv("TT_PERF_DEBUG_ZONE_DUMP") != nullptr);

    uint32_t fifo_pages;
    uint32_t np;
    {
        const auto t0 = std::chrono::steady_clock::now();
        fifo_pages = sock->get_fifo_curr_size() / sock->get_page_size();
        np = sock->pages_available();
        ss.poll_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        ss.polls++;
    }
    if (np == 0) {
        return false;
    }
    if (np >= fifo_pages) {
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
    // Coalesce small arrivals into a wider memcpy (FINDINGS: ~25 GB/s needs ~64–640 KB/read).
    // Do not wait if the FIFO is already half full -- drain to protect credits. Always take a
    // sub-min chunk once stop_ is set so quiesce cannot mark the socket done with pages still pending.
    const uint32_t min_pg = min_pages_per_read();
    if (min_pg != 0 && np < min_pg && np < (fifo_pages / 2) && !stop_.load(std::memory_order_relaxed)) {
        return false;
    }
    // Brief spin-coalesce removed: holding ACK for fatter memcpy extended HOST wall and starved movers.
    if (ddbg && ss.iters < 40) {
        log_info(tt::LogMetal, "[drain sock={}] iter={} np={} fifo_pages={}", sock_idx, ss.iters, np, fifo_pages);
    }
    ss.iters++;
    ss.pages += np;

    if (d2h_discard_only()) {
        // Ceiling probe: retire pages without touching FIFO payload (no staging memcpy).
        const uint64_t t0 = tsc_now();
        (void)sock->peek(np);  // wait_for_bytes only
        sock->pop(np, /*notify_sender=*/false);
        const uint64_t t1 = tsc_now();
        sock->probe_ack_write();
        const uint64_t t2 = tsc_now();
        ss.read_ns += t1 - t0;
        ss.ack_ns += t2 - t1;
        ss.reads++;
        ss.bytes += static_cast<uint64_t>(np) * kPageSize;
        return true;
    }

    // Wait for a free staging slot: every worker must be past it (its ranges, and any carry it
    // references, are about to be overwritten).
    const uint64_t fill = ss.fill_seq.load(std::memory_order_relaxed);
    for (;;) {
        uint64_t wmin = UINT64_MAX;
        for (uint32_t w = 0; w < n_workers_; w++) {
            wmin = std::min(wmin, ss.wdone[w].load(std::memory_order_acquire));
        }
        if (fill - wmin < DeviceCtx::kStageSlots) {
            break;
        }
        _mm_pause();
    }
    const size_t slot = static_cast<size_t>(fill & (DeviceCtx::kStageSlots - 1));
    auto& stage = ss.buf[slot];
    TT_FATAL(
        stage.size() >= static_cast<size_t>(np) * page_words,
        "[perf-debug profiler] staging buffer too small ({} words < {} needed) -- start() must pre-size "
        "to max read; hot-path allocation is forbidden",
        stage.size(),
        static_cast<size_t>(np) * page_words);
    {
        // Split wait vs FIFO→staging copy (read() lumps them; FINDINGS cares about both).
        const uint64_t t0 = tsc_now();
        const auto view = sock->peek(np);
        const uint64_t t1 = tsc_now();
        auto* dst = reinterpret_cast<char*>(stage.data());
        // Wide copy: FIFO is cold (PCIe); staging is the decode working set. Plain memcpy; the prior
        // MOVNTDQA path regressed. Clang turns this into ERMSB/AVX for large n.
        std::memcpy(dst, view.first, view.first_bytes);
        if (view.second_bytes != 0) {
            std::memcpy(dst + view.first_bytes, view.second, view.second_bytes);
        }
        const uint64_t t2 = tsc_now();
        sock->pop(np, /*notify_sender=*/false);
        sock->probe_ack_write();  // batched notify: one PCIe write of cumulative bytes_acked
        const uint64_t t3 = tsc_now();
        ss.wait_ns += t1 - t0;
        ss.copy_ns += t2 - t1;
        ss.read_ns += t2 - t0;  // wait+copy (legacy "sock-read" numerator)
        ss.ack_ns += t3 - t2;
        ss.reads++;
        ss.bytes += static_cast<uint64_t>(np) * kPageSize;
    }
    if (decode_disabled()) {
        // No decode consumers -- retire the slot immediately so fill never stalls.
        ss.words[slot] = 0;
        ss.fill_seq.store(fill + 1, std::memory_order_relaxed);
        for (uint32_t w = 0; w < n_workers_; w++) {
            ss.wdone[w].store(fill + 1, std::memory_order_release);
        }
        return true;
    }
    // ACK already issued -- split into per-worker whole-frame ranges and publish the slot.
    ss.words[slot] = static_cast<size_t>(np) * page_words;
    split_slot(ctx, sock_idx, slot, stage.data(), ss.words[slot]);
    ss.fill_seq.store(fill + 1, std::memory_order_release);
    return true;
}

bool PerfDebugProfiler::decoder_work_pending() const {
    for (const auto& ctx : devices_) {
        for (uint32_t s = 0; s < kNSockets; s++) {
            const auto& ss = ctx.sock_state[s];
            const uint64_t fill = ss.fill_seq.load(std::memory_order_acquire);
            for (uint32_t w = 0; w < n_workers_; w++) {
                if (ss.wdone[w].load(std::memory_order_acquire) != fill) {
                    return true;
                }
            }
        }
    }
    return false;
}

void PerfDebugProfiler::decode_worker(uint32_t worker) {
    tracy::SetThreadName(fmt::format("perf-debug-decode{}", worker).c_str());
    pin_self_to_cpu(pin_cpu_for(PinRole::Decode, worker));
    // Prefault/TLB-warm on the core that will emit into the ring. Constructor memset ran on the
    // bring-up thread; soft faults + remote-NUMA first-touch would otherwise land in decode_ns.
    if (worker == 0 && ring_) {
        ring_->ring.warm_pages(/*lock=*/true);
    }
    for (;;) {
        bool any = false;
        for (auto& ctx : devices_) {
            for (uint32_t s = 0; s < kNSockets; s++) {
                DeviceCtx::SockState& ss = ctx.sock_state[s];
                const uint64_t mine = ss.wdone[worker].load(std::memory_order_relaxed);
                const uint64_t fill = ss.fill_seq.load(std::memory_order_acquire);
                if (mine >= fill) {
                    continue;
                }
                const size_t slot = static_cast<size_t>(mine & (DeviceCtx::kStageSlots - 1));
                decode_ranges(ctx, s, worker, slot);
                ss.wdone[worker].store(mine + 1, std::memory_order_release);
                any = true;
            }
        }
        if (any) {
            continue;
        }
        if (decoder_stop_.load(std::memory_order_acquire) && !decoder_work_pending()) {
            break;
        }
        _mm_pause();
    }
}

// Double-buffer SPSC: decoder fills slots[prod%N], bumps prod; publisher drains slots[cons%N], bumps cons.
// Slack = kPublishBatchSlots. No mutex.
PerfDebugProfiler::PublishBatch* PerfDebugProfiler::publish_acquire_batch(uint32_t worker) {
    for (;;) {
        WorkerPub& wp = wpub_[worker];
        const uint64_t prod = wp.prod.load(std::memory_order_relaxed);
        const uint64_t cons = wp.cons.load(std::memory_order_acquire);
        if (prod - cons < kPublishBatchSlots) {
            PublishBatch& b = wp.batches[prod & (kPublishBatchSlots - 1)];
            b.n = 0;
            return &b;
        }
        _mm_pause();
    }
}

void PerfDebugProfiler::publish_submit_batch(uint32_t worker) {
    wpub_[worker].prod.fetch_add(1, std::memory_order_release);
}

void PerfDebugProfiler::publish_wait_idle() {
    for (uint32_t w = 0; w < n_workers_; w++) {
        while (wpub_[w].cons.load(std::memory_order_acquire) != wpub_[w].prod.load(std::memory_order_relaxed)) {
            _mm_pause();
        }
    }
}

void PerfDebugProfiler::publish_stop() { publisher_stop_.store(true, std::memory_order_release); }

void PerfDebugProfiler::publisher_thread() {
    tracy::SetThreadName("perf-debug-pub");
    pin_self_to_cpu(pin_cpu_for(PinRole::Publisher));
    auto& writer = ring_->ring.writer();
    uint64_t my_pub_ns = 0;
    for (;;) {
        bool any = false;
        for (uint32_t w = 0; w < n_workers_; w++) {
            WorkerPub& wp = wpub_[w];
            const uint64_t cons = wp.cons.load(std::memory_order_relaxed);
            const uint64_t prod = wp.prod.load(std::memory_order_acquire);
            if (cons >= prod) {
                continue;
            }
            PublishBatch& b = wp.batches[cons & (kPublishBatchSlots - 1)];
            if (b.n != 0) {
                const auto t0 = std::chrono::steady_clock::now();
                writer.publish_batch(std::span<const PerfDebugRec>(b.recs.data(), b.n));
                my_pub_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            }
            wp.cons.store(cons + 1, std::memory_order_release);
            any = true;
        }
        if (any) {
            writer.wake_readers();
            continue;
        }
        if (publisher_stop_.load(std::memory_order_acquire)) {
            break;
        }
        _mm_pause();
    }
    w_publish_ns_ = my_pub_ns;
}

// Producer for ONE socket: poll -> read -> ack -> split.
void PerfDebugProfiler::producer_thread(uint32_t sock_idx) {
    tracy::SetThreadName(fmt::format("perf-debug-prod{}", sock_idx).c_str());
    pin_self_to_cpu(pin_cpu_for(sock_idx == 0 ? PinRole::Producer0 : PinRole::Producer1));
    const auto t_writer_entry = std::chrono::steady_clock::now();
    const auto t_wall0 = t_writer_entry;
    auto watchdog = std::chrono::steady_clock::now();
    auto backoff = std::chrono::microseconds(writer_backoff_us());
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
            DeviceCtx::SockState& ss = ctx.sock_state[sock_idx];
            if (ss.done) {
                continue;
            }
            all_done = false;
            if (drain_pass(ctx, sock_idx)) {
                any = true;
                ss.quiesce = 0;
            } else if (
                stopping && (++ss.quiesce >= kQuiesceEmpties || std::chrono::steady_clock::now() >= drain_deadline)) {
                ss.done = true;
            }
        }
        if (any) {
            std::call_once(first_data_once_, [&]() {
                host_first_data_ns_ = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                std::chrono::steady_clock::now().time_since_epoch())
                                                                .count());
                const double ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_writer_entry)
                        .count();
                log_info(
                    tt::LogMetal,
                    "[perf-debug profiler] producer: first data {:.2f} ms after thread start [large => the FIFO sat "
                    "unserviced and producers will have stalled once]",
                    ms);
            });
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
            if (decoder_work_pending()) {
                continue;
            }
            if (std::chrono::steady_clock::now() - watchdog > writer_timeout()) {
                log_warning(
                    tt::LogMetal,
                    "[perf-debug profiler] producer WALL TIMEOUT ({} s no progress)",
                    std::chrono::duration_cast<std::chrono::seconds>(writer_timeout()).count());
                for (auto& c : devices_) {
                    for (uint32_t dd = 0; dd < c.n_drisc; dd++) {
                        dump_drainer_state(c, dd, "producer-wall-timeout");
                    }
                    for (uint32_t s = 0; s < kNSockets; s++) {
                        c.sock_state[s].done = true;
                    }
                }
                break;
            }
            std::this_thread::sleep_for(backoff);
        }
    }
    const uint64_t wall = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_wall0).count());
    for (auto& ctx : devices_) {
        ctx.sock_state[sock_idx].wall_ns = wall;
    }
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
        uint32_t meta = 0;
        uint64_t ts = 0;
        uint32_t id = 0;
        uint32_t type = 0;
        uint32_t prog = 0;
        uint32_t want = 0;  // uint64s expected; UINT32_MAX until the EXT record supplies it
        uint32_t got = 0;
        uint64_t vals[kMaxEventValues] = {};
    } pend;
    std::vector<uint64_t> con_last_ts_(4096, 0);
    auto emit_batch = [&](std::span<PerfDebugRec> b) {
        for (const auto& r : b) {
            const uint32_t rt = r.meta >> kRecTypeShift;
            if (rt != kRecZoneStart && rt != kRecZoneEnd) {
                continue;
            }
            const uint32_t ln = (r.meta >> kRecLaneShift) & (kRecLaneMax - 1u);
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
        });
        // Resolve a lane to its coords + name and push the reassembled event. Shares the coord/name
        // resolution shape with the zone path below.
        auto flush_event = [&]() {
            if (!pend.active) {
                return;
            }
            pend.active = false;
            const uint32_t dev_idx = (pend.meta >> kRecDevShift) & (kRecDevMax - 1u);
            const uint32_t lane = (pend.meta >> kRecLaneShift) & (kRecLaneMax - 1u);
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
            pkt.runtime_id = (pend.type == kRecEvent);
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
            const uint32_t rt = r.meta >> kRecTypeShift;
            if (rt == kRecCont) {
                if (pend.active && pend.got < kMaxEventValues) {
                    pend.vals[pend.got++] = r.ts;
                }
                if (pend.active && pend.got >= pend.want) {
                    flush_event();
                }
                continue;
            }
            if (rt == kRecExt) {
                if (pend.active) {
                    pend.id = static_cast<uint32_t>(r.ts >> 32);
                    pend.want = ((static_cast<uint32_t>(r.ts) & 0x7Fu) + 1u) / 2u;  // payload words -> uint64s
                    if (pend.want == 0) {
                        flush_event();  // a bare event (DeviceRecordEvent) has no continuations
                    }
                }
                continue;
            }
            if (rt == kRecData || rt == kRecEvent) {
                flush_event();  // defensive: a truncated predecessor must not absorb this event's payload
                pend = PendingEvent{};
                pend.active = true;
                pend.meta = r.meta;
                pend.ts = r.ts;
                pend.id = r.meta & 0xFFFFu;  // provisional; the EXT record supplies the full 20-bit id
                pend.type = rt;
                pend.prog = 0;
                pend.want = UINT32_MAX;  // unknown until the EXT record lands
                continue;
            }
            if (rt != kRecZoneStart && rt != kRecZoneEnd) {
                continue;
            }
            const uint32_t dev_idx = (r.meta >> kRecDevShift) & (kRecDevMax - 1u);
            const uint32_t lane = (r.meta >> kRecLaneShift) & (kRecLaneMax - 1u);
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
            if (auto it = zone_names_.find(static_cast<uint16_t>(r.meta)); it != zone_names_.end()) {
                name = it->second;
            }
            perf_debug::WorkerZonePacket pkt;
            pkt.chip_id = ctx.chip_id;
            pkt.core_virtual_x = vx;
            pkt.core_virtual_y = vy;
            pkt.core_noc0_x = nx;
            pkt.core_noc0_y = ny;
            pkt.risc = risc;
            pkt.timer_id = r.meta & 0xFFFFu;
            pkt.name = name;
            // Synced: push the RAW device timestamp -- the context was anchored with a real (host, device)
            // pair, so Tracy places it exactly. Unsynced: fall back to rebasing on the first marker seen.
            const uint64_t base = ctx.synced ? 0 : ctx.marker_ts_base;
            pkt.timestamp = (r.ts >= base) ? (r.ts - base) : 0;
            pkt.is_start = (rt == kRecZoneStart);
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

// Host sustained throughput from copy+ack+decode+publish busy time (and optional wall). Device first→last
// zone rate is NOT derived here -- that wants a device-side window, not a host scan of every record.
void PerfDebugProfiler::report_sustained_throughput() {
    if (w_zone_recs_ == 0 && w_bytes_ == 0) {
        log_info(tt::LogMetal, "[perf-debug profiler] sustained throughput: nothing drained");
        return;
    }
    const uint64_t marker_wire_bytes = w_zone_recs_ * 8ull;
    const uint64_t zones = w_zone_recs_ / 2ull;

    auto gbps = [](uint64_t bytes, double span_s) {
        return span_s > 0.0 ? (static_cast<double>(bytes) / 1e9) / span_s : 0.0;
    };
    auto mzps = [](uint64_t z, double span_s) { return span_s > 0.0 ? (static_cast<double>(z) / 1e6) / span_s : 0.0; };

    // Pipeline busy: copy+ack, decode, and ring publish overlap on different threads. Report the max
    // (true pipeline occupancy). Publish is in the max -- BroadcastRing ingest is the product.
    const double copy_ack_s = (w_read_ns_ * tsc_ns_per_tick() + w_ack_ns_ * tsc_ns_per_tick()) / 1e9;
    const double decode_s = w_decode_ns_ / 1e9;
    const double publish_s = w_publish_ns_ / 1e9;
    const double host_busy_s = std::max(copy_ack_s, std::max(decode_s, publish_s));
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] SUSTAINED THROUGHPUT -- {} zone-markers ({} zones) | marker-wire {:.1f} MB | "
        "D2H {:.1f} MB",
        w_zone_recs_,
        zones,
        marker_wire_bytes / (1024.0 * 1024.0),
        w_bytes_ / (1024.0 * 1024.0));
    log_info(
        tt::LogMetal,
        "[perf-debug profiler]   HOST busy (pipeline max(copy+ack {:.3f}, decode {:.3f}, publish {:.3f}) ms): "
        "marker-wire {:.2f} GB/s | D2H {:.2f} GB/s | {:.2f} Mzones/s",
        copy_ack_s * 1e3,
        decode_s * 1e3,
        publish_s * 1e3,
        gbps(marker_wire_bytes, host_busy_s),
        gbps(w_bytes_, host_busy_s),
        mzps(zones, host_busy_s));

    if (host_first_data_ns_ != 0 && host_last_rec_ns_ > host_first_data_ns_) {
        const double host_wall_s = static_cast<double>(host_last_rec_ns_ - host_first_data_ns_) / 1e9;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler]   HOST wall (first data→last decode submit {:.3f} ms): "
            "marker-wire {:.2f} GB/s | D2H {:.2f} GB/s | {:.2f} Mzones/s",
            host_wall_s * 1e3,
            gbps(marker_wire_bytes, host_wall_s),
            gbps(w_bytes_, host_wall_s),
            mzps(zones, host_wall_s));
    }
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

void PerfDebugProfiler::stop() {
    if (stopped_.exchange(true)) {
        return;
    }

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
            std::vector<uint32_t> res(64, 0);
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
    for (auto& t : producers_) {
        if (t.joinable()) {
            t.join();  // drains its socket to quiescence (staging fills included)
        }
    }
    // Finish any in-flight slot decodes, then stop the workers before draining publish batches.
    decoder_stop_.store(true, std::memory_order_release);
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    publish_wait_idle();
    publish_stop();
    if (publisher_.joinable()) {
        publisher_.join();
    }
    // Aggregate: volumes sum; critical-path times take the MAX across parallel sockets (matching
    // decode workers). Summing two parallel drain copy times double-counted the copy stage.
    uint64_t max_copy_ack_ns = 0;
    for (auto& ctx : devices_) {
        for (uint32_t sk = 0; sk < kNSockets; sk++) {
            const auto& ss = ctx.sock_state[sk];
            w_read_ns_ = std::max(w_read_ns_, ss.read_ns);
            w_wait_ns_ = std::max(w_wait_ns_, ss.wait_ns);
            w_copy_ns_ = std::max(w_copy_ns_, ss.copy_ns);
            w_ack_ns_ = std::max(w_ack_ns_, ss.ack_ns);
            w_poll_ns_ = std::max(w_poll_ns_, ss.poll_ns);
            max_copy_ack_ns = std::max(max_copy_ack_ns, ss.read_ns + ss.ack_ns);
            w_polls_ += ss.polls;
            w_reads_ += ss.reads;
            w_bytes_ += ss.bytes;
            w_wall_ns_ = std::max(w_wall_ns_, ss.wall_ns);
            uint64_t emit = 0;
            for (uint32_t w = 0; w < n_workers_; w++) {
                emit += wpub_[w].emit[sk];
            }
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] socket {} drained: {} pages, {} markers ({} reads); producer stall "
                "zones: {} | copy {:.2f} ms ({:.2f} GB/s) [0 stalls = drainer kept up]",
                sk,
                ss.pages,
                emit,
                ss.iters,
                ss.stall,
                ss.copy_ns * tsc_ns_per_tick() / 1e6,
                ss.copy_ns ? (static_cast<double>(ss.bytes) / (ss.copy_ns * tsc_ns_per_tick())) : 0.0);
        }
    }
    // Prefer per-socket (copy+ack) max so HOST busy doesn't mix sock0 copy with sock1 ack.
    if (max_copy_ack_ns != 0) {
        w_read_ns_ = max_copy_ack_ns;
        w_ack_ns_ = 0;
    }
    for (uint32_t w = 0; w < n_workers_; w++) {
        const auto& wp = wpub_[w];
        w_decode_ns_ = std::max(w_decode_ns_, wp.decode_ns);
        w_recs_ += wp.recs;
        w_zone_recs_ += wp.zone_recs;
        w_stalls_ += wp.stall;
        host_last_rec_ns_ = std::max(host_last_rec_ns_, wp.last_rec_ns);
        if (wp.recs != 0) {
            log_info(
                tt::LogMetal,
                "[perf-debug profiler] decode worker {}: {} records in {:.1f} ms",
                w,
                wp.recs,
                wp.decode_ns / 1e6);
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

    report_lane_spread();
    report_sustained_throughput();
    writer_done_.store(true, std::memory_order_release);
    if (w_reads_ != 0) {
        const double read_ms = w_read_ns_ * tsc_ns_per_tick() / 1e6;
        const double dec_ms = w_decode_ns_ / 1e6, pub_ms = w_publish_ns_ / 1e6;
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] host producer: {} reads, {:.1f} MB, {} records | sock-read {:.1f} ms "
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
            "[perf-debug profiler] host producer sock-read split: wait {:.1f} ms | copy {:.1f} ms ({:.2f} GB/s, "
            "{:.1f} ns/KB) | ack {:.1f} ms ({:.0f} ns/read) | predrain {:.1f} ms ({:.0f} ns/read)",
            w_wait_ns_ * tsc_ns_per_tick() / 1e6,
            w_copy_ns_ * tsc_ns_per_tick() / 1e6,
            w_copy_ns_ ? (static_cast<double>(w_bytes_) / (w_copy_ns_ * tsc_ns_per_tick())) : 0.0,
            w_bytes_ ? (static_cast<double>(w_copy_ns_) * tsc_ns_per_tick() * 1024.0 / static_cast<double>(w_bytes_))
                     : 0.0,
            w_ack_ns_ * tsc_ns_per_tick() / 1e6,
            w_reads_ ? (static_cast<double>(w_ack_ns_) * tsc_ns_per_tick() / w_reads_) : 0.0,
            w_predrain_ns_ * tsc_ns_per_tick() / 1e6,
            w_reads_ ? (static_cast<double>(w_predrain_ns_) * tsc_ns_per_tick() / w_reads_) : 0.0);
        const double wall_ms = w_wall_ns_ / 1e6;
        const double tick_ms = tsc_ns_per_tick() / 1e6;
        const double sock_ms = (w_read_ns_ + w_ack_ns_) * tick_ms;
        const double dec_only_ms = w_decode_ns_ / 1e6;
        const double pub_only_ms = w_publish_ns_ / 1e6;
        // decode/publish run on sibling threads (overlapped with sock-read).
        const double work_ms = sock_ms + (w_poll_ns_ / 1e6);
        log_info(
            tt::LogMetal,
            "[perf-debug profiler] PRODUCER thread wall {:.1f} ms: poll {:.1f}% ({} polls) | sock-read {:.1f}% "
            "| idle {:.1f}% -- {:.0f}% busy (decode {:.1f} ms on decoder, publish {:.1f} ms on publisher; both "
            "overlapped)",
            wall_ms,
            wall_ms > 0 ? 100.0 * (w_poll_ns_ / 1e6) / wall_ms : 0.0,
            w_polls_,
            wall_ms > 0 ? 100.0 * sock_ms / wall_ms : 0.0,
            wall_ms > 0 ? 100.0 * (wall_ms - work_ms) / wall_ms : 0.0,
            wall_ms > 0 ? 100.0 * work_ms / wall_ms : 0.0,
            dec_only_ms,
            pub_only_ms);
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
        ring_->ring.writer().wake_readers();
    }
    if (consumer_.joinable()) {
        consumer_.join();
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
            "[perf-debug profiler] order/loss: per-lane ts regressions {} of {} at publish, {} of {} at consume "
            "[both MUST be 0; non-zero = records reordered => Tracy nesting corrupt] | lane-bound drops {} | "
            "batch flushes {}",
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
