// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop-collector: headless sampler that publishes per-core utilization
// state to /dev/shm/tt_device_<asic>_util for any number of viewer processes.
//
// This is Phase 1 of the plan in tt_metal/tools/ttnvtop/PLAN.md. In Phase 1
// the only signal populated is dispatch-occupancy (go_msg.signal sampled
// over PCIe). Phase 2 will add per-pipeline perf-counter samples from an
// on-chip sampler; at that point only this file changes — the SHM schema,
// publisher, and viewer are designed to stay stable.
//
// Coexistence model:
//   We open chips via umd::TopologyDiscovery, which does not construct a
//   umd::Cluster and therefore does not call LocalChip::start_device(). That
//   call site is where UMD takes the CHIP_IN_USE robust mutex, so by
//   bypassing it we can coexist with any running tt-metal workload. Reads
//   are plain PCIe TLB reads via TTDevice::read_from_device — non-destructive.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umd/device/firmware/firmware_info_provider.hpp"  // get_dram_speed(), for the DRAM peak
#include "umd/device/soc_arch_descriptor.hpp"
#include "umd/device/soc_descriptor.hpp"
#include "umd/device/topology/topology_discovery.hpp"
#include "umd/device/topology/topology_discovery_options.hpp"
#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"
#include "umd/device/types/xy_pair.hpp"

// dev_msgs.h refuses direct host inclusion; HAL_BUILD wraps it in a unique
// namespace (same trick wh_hal_tensix.cpp uses). Phase 1 supports Wormhole
// only — BH is a Phase 3 task per the plan.
#define HAL_BUILD ttnvtop_wh_tensix
#include "dev_mem_map.h"

// Phase 2.2 M1. Unlike util_sampler.h -- which is firmware-only and therefore
// hand-mirrored below as ttnvtop_ring -- util_aggregator.h is deliberately
// dependency-free and can be included directly on the host, so the journal layout
// has exactly one definition shared with the kernel.
#include "util_aggregator.h"
#include "tools/ttnvtop/host/agg_layout.hpp"
#include "tools/ttnvtop/host/agg_core_select.hpp"
#include "hostdev/dev_msgs.h"
using namespace ttnvtop_wh_tensix;  // NOLINT(google-build-using-namespace)

#include "shm_publisher.hpp"
#include "common/program_registry.hpp"

// util_sampler.h is normally a firmware-only header (it dereferences a fixed
// L1 address). We only need its layout types and constants, not its
// host-unsafe inline functions. Pull just the layout in by skipping the
// inline ring()/init()/maybe_tick() helpers — easiest path is a parallel
// declaration block, since the header guards are unconditional.
namespace ttnvtop_ring {
constexpr uint32_t kMagic = 0x53555454u;  // 'TTUS' little-endian — must match util_sampler.h UTIL_SAMPLER_MAGIC.
constexpr uint32_t kVersion = 2u;
// Phase 2.1.c: header grew 16 -> 32 B (added current_kernel_id + reserved
// pad), ring shrank 63 -> 62 entries. Mirrors util_sampler.h.
constexpr uint32_t kRingSize = 62u;
struct Entry {
    uint32_t wall_clock_l;
    uint32_t kernel_id;
    uint32_t fpu_count;
    uint8_t math_fidelity;
    uint8_t counter_sel;
    uint8_t producer_riscv;
    uint8_t flags;
};
static_assert(sizeof(Entry) == 16, "Entry must mirror util_sampler.h util_sampler_entry_t");
struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t head;
    uint32_t period_cycles;
    uint32_t current_kernel_id;  // stashed by trisc1 firmware on kernel start (Phase 2.1.c)
    uint32_t next_due_wall_l;    // persistent sample deadline, written by trisc1 (Phase 2.1.c.i)
    uint32_t reserved[2];        // pad to 32 B; future per-thread metadata
};
static_assert(sizeof(Header) == 32);
}  // namespace ttnvtop_ring

using tt::CoordSystem;
using tt::CoreType;
using tt::umd::CoreCoord;
using tt::umd::SocArchDescriptor;
using tt::umd::SocDescriptor;
using tt::umd::TopologyDiscovery;
using tt::umd::TopologyDiscoveryOptions;
using tt::umd::TTDevice;

namespace {

constexpr int kDefaultSampleHz = 100;
constexpr int kWindowSamples = 100;  // rolling window for dispatch occupancy (~1s at 100Hz)
constexpr int kDefaultPublishHz = 10;

// Tensix RISC-V debug register layout on Wormhole. Same addresses on host-NOC:
// per UMD/exalens, RISCV_DEBUG_REGS_START_ADDR is both the private and NOC
// address for these regs. We read a 220-byte window covering
// RISCV_DEBUG_REG_PERF_CNT_OUT_{L,H}_FPU (@0x120, 0x124) and
// RISCV_DEBUG_REG_WALL_CLOCK_{L,H} (@0x1F0, 0x1F8) in one transaction.
//
// WH and BH share these offsets; the value lives in tensix.h. Hardcoded here
// so the collector is self-contained (the firmware header cannot be included
// in a host-only build without the HAL_BUILD dance).
constexpr uint64_t kRiscvDebugBase = 0xFFB12000ull;

// Perf-counter control registers (per tt_metal/tools/profiler/perf_counters.hpp
// and tt_llk/docs/performance_counters/performance_counters.md):
//   FPU0 @ 0x018: reference period (unused in continuous mode)
//   FPU1 @ 0x01C: [7:0] mode (0 = continuous), [16:8] counter_sel (MUX for OUT_H)
//   FPU2 @ 0x020: [0] start, [1] stop. Rising edge on [0] clears + starts the counter.
// counter_sel muxes which underlying counter is read out at OUT_H: 0 =
// FPU_INSTRUCTION, 1 = SFPU_INSTRUCTION, 257 = combined. Both underlying
// counters accumulate in parallel in hardware; counter_sel only changes
// which one is visible at OUT_H. The collector alternates counter_sel per
// tick to sample FPU and SFPU busy% from the single output port.
constexpr uint64_t kRegFpu0 = kRiscvDebugBase + 0x018;
constexpr uint64_t kRegFpu1 = kRiscvDebugBase + 0x01C;
constexpr uint64_t kRegFpu2 = kRiscvDebugBase + 0x020;

// counter_sel field in PERF_CNT_FPU1 (bits 16:8). Used to mux which FPU-bank
// counter reads out at OUT_H.
constexpr uint32_t kFpuCounterSelShift = 8;
constexpr uint32_t kFpuCounterSelFpu = 0;   // FPU_INSTRUCTION
constexpr uint32_t kFpuCounterSelSfpu = 1;  // SFPU_INSTRUCTION
// Mode 0 = continuous (0..7 bits), 0 in bits 31:17.
constexpr uint32_t kFpuModeContinuous = 0;

// Data window we read every tick: FPU_OUT_L/H (0x120, 0x124) and
// WALL_CLOCK_L/H (0x1F0, 0x1F8). One NOC read covers 220 bytes including gaps.
//
// FPU_OUT_L and FPU_OUT_H are TWO INDEPENDENT 32-bit counters in the FPU
// counter group — not the low/high halves of one 64-bit value:
//   OUT_L = ref_cnt (independent of counter_sel — reference cycles)
//   OUT_H = counter muxed by PERF_CNT_FPU1[16:8] (FPU or SFPU)
// We alternate counter_sel per tick and route the OUT_H read into the
// matching (FPU or SFPU) delta/EWMA branch. WALL_CLOCK is the denominator.
//
// WALL_CLOCK_L / WALL_CLOCK_H ARE paired halves of one 64-bit free-running
// cycle counter.
// Counter addresses. Remote chips cannot be read with a single large block —
// UMD's ETH tunnel aliases/zeros past the first 4 bytes of a debug-reg
// block. We use one 8-byte read for the adjacent OUT_L/OUT_H pair, then
// two separate 4-byte reads for WALL_CLOCK_L and WALL_CLOCK_H (there's a
// 4-byte gap at 0x1F4 between them).
constexpr uint64_t kAddrFpuOutL = kRiscvDebugBase + 0x120;  // 4 B
constexpr uint64_t kAddrFpuOutH = kRiscvDebugBase + 0x124;  // 4 B, adjacent to OUT_L
constexpr uint64_t kAddrWallL = kRiscvDebugBase + 0x1F0;    // 4 B
constexpr uint64_t kAddrWallH = kRiscvDebugBase + 0x1F8;    // 4 B, +8 from WALL_L

// EWMA smoothing for compute_busy. alpha = 2/(N+1) for N-sample horizon.
// N=10 gives a ~1s smoothing window at 10 Hz publish rate.
constexpr double kComputeEwmaAlpha = 2.0 / (10.0 + 1.0);

struct CliOptions {
    int sample_hz = kDefaultSampleHz;
    int publish_hz = kDefaultPublishHz;
    std::set<int> device_filter;  // empty = all chips
    std::string log_file;         // non-empty = redirect stderr there
    bool show_help = false;
    bool journal_probe = false;  // Phase 2.2 M1: scan for aggregator journals and exit
    std::string launch_go;       // Phase 2.2 7.4: descriptor of a staged aggregator to start
    bool journal_transport = false;  // Phase 2.2 5k: emulate the host-pull journal transport
    bool read_latency_probe = false;  // Phase 2.2: remote-read latency vs transfer size
    std::string launch_artifact;      // Phase 2.2: dir holding aggregator.image/.desc
    int fidelity_probe_s = 0;         // Phase 2.2: drain the aggregator journal for N s and report loss
    bool stop_aggregator = false;     // Phase 2.2: ask every running aggregator to return
    bool pin_tunnel = false;          // Narrow each remote chip's tunnel to its real link channels
    bool local_only = false;          // Skip remote discovery entirely -- the rescue path for a wedged tunnel
    std::string peek;                 // "chip,x,y,addr[,len]" -- read raw L1 twice and print it
    int watchdog_s = 0;               // Guard against a dead aggregator hard-blocking the next device open

    // ---- BEGIN ported from tt_coremon (CliOptions::remote_budget, source line 262)
    // Tunnelled NOC transactions per second per REMOTE chip that the per-core ring
    // drain is allowed to spend. 0 = uncapped.
    //
    // This is a FALLBACK knob, not the primary strategy. In the tt_coremon lineage the
    // per-core sweep over the ETH tunnel was the only path to a remote chip, so it had
    // to be rate-limited to stay polite. Here it is not: a chip with `journal_active`
    // publishes from its on-chip aggregator and its per-core drain is switched off
    // entirely (see ChipState::journal_active and the early return in `ring_drain`),
    // which removes the tunnel traffic rather than metering it.
    //
    // What is left to budget is the UN-AGGREGATED remote chip -- no aggregator was
    // launched, or its launch failed. There the unthrottled per-core drain measured
    // 0.1-0.2 Hz with 99.4-99.9% sample loss on a busy remote chip: it was not
    // producing usable data, and it was taking UMD's NON_MMIO mutex thousands of times
    // a second to fail at it. So this bounds THAT drain, so an un-aggregated remote
    // chip degrades gracefully instead of hammering the tunnel.
    //
    // The currency is transaction COUNT, not bytes or time: UMD takes the NON_MMIO
    // mutex per read_from_non_mmio_device() call and documents the acquisition as
    // non-trivial, so what a workload's request loses the race against is the
    // back-to-back acquire/release cycles, not the payload size.
    //
    // Cores are visited round-robin within the budget rather than freezing the whole
    // chip, so coverage stays continuous. That costs no accuracy in kind: the ring
    // already reports what it lost (ChipState::drain_lost_samples), and a
    // less-frequently-visited core simply loses more of it -- on a path that was
    // already losing 99%+. Pass 0 to uncap (an idle box being debugged by one user).
    double remote_budget = 1500.0;
    // ---- END ported from tt_coremon
};

void print_help(const char* argv0) {
    std::cout << "ttnvtop-collector — live per-core utilization sampler\n"
                 "\n"
                 "Publishes /dev/shm/tt_device_<asic>_util files that ttnvtop (and other\n"
                 "viewers) read. Coexists with running tt-metal workloads — does not\n"
                 "take the UMD CHIP_IN_USE lock.\n"
                 "\n"
                 "Usage: "
              << argv0
              << " [options]\n"
                 "\n"
                 "Options:\n"
                 "  -h, --help              Show this help and exit.\n"
                 "  --sample-hz N           PCIe sampling rate per core (default "
              << kDefaultSampleHz
              << ").\n"
                 "  --publish-hz N          SHM write rate (default "
              << kDefaultPublishHz
              << ").\n"
                 "  --device N              Only monitor chip N. Repeat to select several.\n"
                 "  --journal-probe         Scan every chip's ethernet cores for a Phase 2.2\n"
                 "                          aggregator journal, print what was found, and exit.\n"
                 "  --remote-budget N       Tunnelled transactions/s the per-core ring drain may\n"
                 "                          spend on a remote chip that has NO on-chip aggregator\n"
                 "                          feeding it (default 1500, 0 = uncapped). A chip with an\n"
                 "                          aggregator does not drain per-core at all, so this\n"
                 "                          never applies to it.\n"
                 "  --local-only            Monitor only the MMIO chips, skipping remote discovery.\n"
                 "                          An explicit choice, never a fallback: the remote chips ARE\n"
                 "                          the hard case this tool exists for, so dropping them is\n"
                 "                          never the automatic answer to a wedged tunnel.\n"
                 "  --log-file PATH         Redirect collector stderr (incl. UMD logs) to PATH.\n"
                 "\n"
                 "Examples:\n"
                 "  "
              << argv0
              << "                        # monitor every chip in the system\n"
                 "  "
              << argv0
              << " --device 0             # just the local mmio chip\n"
                 "  "
              << argv0 << " --log-file /tmp/ttnvtop.log   # silence UMD eth warnings\n";
}

bool parse_int(const char* s, int& out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || (end != nullptr && *end != '\0')) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

bool parse_cli(int argc, char* argv[], CliOptions& out) {
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto need_arg = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "ttnvtop-collector: " << flag << " requires a value\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            out.show_help = true;
            return true;
        } else if (a == "--sample-hz") {
            const char* v = need_arg("--sample-hz");
            if (v == nullptr || !parse_int(v, out.sample_hz) || out.sample_hz <= 0) {
                return false;
            }
        } else if (a == "--publish-hz") {
            const char* v = need_arg("--publish-hz");
            if (v == nullptr || !parse_int(v, out.publish_hz) || out.publish_hz <= 0) {
                return false;
            }
        } else if (a == "--device") {
            const char* v = need_arg("--device");
            int d = -1;
            if (v == nullptr || !parse_int(v, d) || d < 0) {
                std::cerr << "ttnvtop-collector: --device expects a non-negative int\n";
                return false;
            }
            out.device_filter.insert(d);
        } else if (a == "--journal-probe") {
            out.journal_probe = true;
        } else if (a == "--launch-aggregator") {
            const char* v = need_arg("--launch-aggregator");
            if (v == nullptr) {
                return false;
            }
            out.launch_artifact = v;
        } else if (a == "--watchdog") {
            const char* v = need_arg("--watchdog");
            if (v == nullptr || !parse_int(v, out.watchdog_s) || out.watchdog_s <= 0) {
                std::cerr << "ttnvtop-collector: --watchdog expects a positive number of seconds\n";
                return false;
            }
        } else if (a == "--peek") {
            const char* v = need_arg("--peek");
            if (v == nullptr) {
                return false;
            }
            out.peek = v;
        } else if (a == "--pin-tunnel") {
            out.pin_tunnel = true;
        } else if (a == "--stop-aggregator") {
            out.stop_aggregator = true;
        } else if (a == "--local-only") {
            out.local_only = true;
        } else if (a == "--fidelity-probe") {
            const char* v = need_arg("--fidelity-probe");
            if (v == nullptr) {
                return false;
            }
            out.fidelity_probe_s = std::atoi(v);
            if (out.fidelity_probe_s <= 0) {
                std::cerr << "ttnvtop-collector: --fidelity-probe expects a positive number of seconds\n";
                return false;
            }
        } else if (a == "--read-latency-probe") {
            out.read_latency_probe = true;
        } else if (a == "--journal-transport") {
            out.journal_transport = true;
        } else if (a == "--launch-go") {
            const char* v = need_arg("--launch-go");
            if (v == nullptr) {
                return false;
            }
            out.launch_go = v;
        } else if (a == "--remote-budget") {
            // ---- BEGIN ported from tt_coremon (parse, source lines 387-396)
            const char* v = need_arg("--remote-budget");
            if (v == nullptr) {
                return false;
            }
            out.remote_budget = std::strtod(v, nullptr);
            if (!(out.remote_budget >= 0.0)) {  // also rejects NaN
                std::cerr << "ttnvtop-collector: --remote-budget must be >= 0.\n";
                return false;
            }
            // ---- END ported from tt_coremon
        } else if (a == "--log-file") {
            const char* v = need_arg("--log-file");
            if (v == nullptr) {
                return false;
            }
            out.log_file = v;
        } else {
            std::cerr << "ttnvtop-collector: unknown argument: " << a << "\n";
            return false;
        }
    }
    return true;
}

struct CoreState {
    uint32_t noc_x = 0;
    uint32_t noc_y = 0;
    tt_xy_pair translated{0, 0};
    TTDevice* device = nullptr;
    bool is_remote = false;

    // Rolling dispatch-occupancy window.
    std::array<uint8_t, kWindowSamples> samples{};
    size_t head = 0;
    uint32_t busy_count = 0;
    uint64_t samples_seen = 0;
    uint8_t last_dispatched = 0;

    // Phase 2.0 perf-counter state. Delta math against wall clock.
    //   busy_fraction_of_wall_time = delta(OUT_H) / delta(WALL_CLOCK)
    // OUT_H only accumulates when the counter is armed, WALL_CLOCK always
    // ticks — so idle gaps are correctly attributed as 0% busy.
    // If OUT_H went backwards between ticks (kernel-side StartPerfCounters
    // reset), use OUT_H as a conservative post-reset delta estimate rather
    // than dropping the tick; over many ticks this converges to truth.
    bool counter_armed = false;
    // Which counter_sel the *next* tick will read (hardware's currently-programmed selector).
    uint32_t next_counter_sel = kFpuCounterSelFpu;
    // Primed separately per counter because each is sampled every other tick.
    bool perf_primed_fpu = false;
    bool perf_primed_sfpu = false;
    uint64_t last_wall_fpu = 0;
    uint64_t last_wall_sfpu = 0;
    uint32_t last_fpu_out_h = 0;
    uint32_t last_sfpu_out_h = 0;
    double fpu_busy_ewma = 0.0;   // [0..1]
    double sfpu_busy_ewma = 0.0;  // [0..1]

    // Phase 2.1.e kernel attribution: raw host_assigned_id from the running
    // launch slot. Full u32; viewer decodes bits 30:10 as the program id.
    uint32_t last_kernel_id = 0;

    // Phase 2.1.c ring drain state. The 50 Hz drain thread reads the
    // per-core L1 sampler ring (1 KiB at MEM_UTIL_SAMPLER_BASE), parses
    // entries newer than `last_ring_head`, and applies wrap-aware
    // wall_clock_l deltas against `last_ring_wall_l`.
    uint32_t last_ring_head = 0;       // last `head` value we saw — entries [last_ring_head, head) are new.
    uint32_t last_ring_wall_l = 0;     // last entry's wall_clock_l, for delta math against the next batch.
    uint32_t last_ring_kernel_id = 0;  // kernel_id of the most recent ring entry we drained.
    bool ring_primed = false;          // first drain just records state, second drain begins delta accumulation.
};

// Phase 2.1.c: per-kernel rolling-window cycle accumulator. We track total
// wall-clock cycles attributed to each `kernel_id` over the last 1 second
// across all worker cores on the chip. Old samples decay out the back of
// the deque; the current sum is published into the program registry's
// `cycles_in_window` field for the viewer's TIME% column.
struct KernelTimeAccumulator {
    // Each sample is (recv_steady_us, cycles_added_at_this_tick).
    std::deque<std::pair<uint64_t, uint64_t>> samples;
    uint64_t total = 0;  // sum of cycles in samples — kept incrementally to avoid full re-summation.

    void add(uint64_t now_us, uint64_t cycles) {
        if (cycles == 0) {
            return;
        }
        samples.emplace_back(now_us, cycles);
        total += cycles;
    }

    // Drop entries older than `now_us - window_us` from the front.
    void decay(uint64_t now_us, uint64_t window_us) {
        while (!samples.empty() && samples.front().first + window_us <= now_us) {
            total -= samples.front().second;
            samples.pop_front();
        }
    }
};

// Phase 2.2 M1 journal support.
//
// An aggregator kernel on a remote chip's idle eth core pushes its journal over
// fabric into an idle-eth L1 slot on that chip's MMIO chip. The collector reads the
// LANDING copy over plain PCIe and never touches the ethernet tunnel -- which is the
// entire point: a host read of a remote chip takes UMD's NON_MMIO mutex and blocks for
// tens of seconds under load (PLAN_ETH_AGGREGATOR.md 5c).
namespace ttnvtop_agg {

// Idle-eth UNRESERVED base -- the well-known address the journal lives at (5k).
//
// PER ARCH, and NOT one formula. This was a single mirrored expression, and it was
// silently wrong on Blackhole: the two HALs do not agree on how to derive this.
//
//   wh_hal_idle_eth.cpp:  ((MEM_IERISC_MAP_END + L1_KERNEL_CONFIG_SIZE - 1) | (align-1)) + 1
//   bh_hal_idle_eth.cpp:  tt::align(MEM_AERISC_MAP_END + MEM_ERISC_KERNEL_CONFIG_SIZE, align)
//
// Different base symbol (AERISC vs IERISC) and a different rounding. With the WH
// expression the collector looked for Blackhole journals at 0xe200 while they were at
// 0x15740, found nothing, and reported "no aggregator is running" about an aggregator it
// had just verified as RUNNING two seconds earlier.
//
// The collector deliberately does not link tt-metal (PLAN 7.2), so it cannot call the
// Hal, and CMake puts only the WORMHOLE hw/inc on its include path -- so every mirrored
// constant here is a Wormhole constant whether or not it is labelled one. These two
// numbers are therefore spelled out, with their derivation above, and
// TestEthAggregatorLandingBaseMatchesHal asserts the value for the running arch against
// `hal.get_dev_addr(IDLE_ETH, UNRESERVED)` -- in the test binary, which does have the
// Hal. That is where drift gets caught; it cannot be caught here.
//
//   Wormhole   MEM_IERISC_MAP_END 0x7df0 + 25 KiB, rounded to 32  -> 0xe200  (57856)
//   Blackhole  MEM_AERISC_MAP_END 0xf330 + 25 KiB, aligned to 32  -> 0x15740 (87872)
// routing_info_t.routing_enabled, first word of ERISC_APP_ROUTING_INFO_BASE
// (dev_mem_map.h:258 MEM_ERISC_MAX_L1_LOADING_ADDR; dev_msgs.h:459 puts routing_enabled
// at offset 0). Set to 1 for exactly the control-plane ACTIVE eth set, forced channel 15
// included (llrt/tt_cluster.cpp:1499-1522).
constexpr uint64_t kEthRoutingInfoAddr = 0x3DC00ull;

constexpr uint64_t kLandingBaseWormhole = 0xe200ull;
constexpr uint64_t kLandingBaseBlackhole = 0x15740ull;

inline uint64_t landing_base(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: return kLandingBaseWormhole;
        case tt::ARCH::BLACKHOLE: return kLandingBaseBlackhole;
        default: return 0;  // unknown arch -- probe finds nothing rather than reading garbage
    }
}

// A journal found on an MMIO chip, describing one remote chip.
struct Landing {
    tt::umd::CoreCoord core{};  // idle eth core on the MMIO chip holding the journal
    uint32_t src_chip = 0;      // fabric node id the aggregator stamped in
    uint32_t capacity = 0;
    uint32_t num_cores = 0;
    uint32_t last_head = 0;
    uint32_t last_sweep_count = 0;
    uint64_t last_advance_us = 0;  // for staleness: when sweep_count last moved
    bool primed = false;
};

// Scan one chip's ethernet cores for a landed journal.
//
// All 16 channels are probed, not just the ones we believe are idle: the collector has
// no control plane and therefore no get_inactive_ethernet_cores(), and 16 reads of 64 B
// over PCIe is nothing. A core either has 'TTAG' at the arch's landing base or it does not.
//
// The header is self-describing -- it carries src_chip, capacity and num_cores -- so
// discovery needs no coordination with whoever launched the aggregator. That matters
// because in M1 the launcher is the workload process and the collector is a separate
// one, with no IPC between them.
template <typename Dev>
inline std::vector<Landing> probe_landings(
    Dev* dev, const tt::umd::SocDescriptor& soc, uint64_t now_us, tt::ARCH arch) {
    std::vector<Landing> found;
    const uint64_t base = landing_base(arch);
    if (base == 0) {
        return found;
    }
    std::vector<uint8_t> buf(sizeof(util_agg_msg_t));
    for (const auto& eth : soc.get_cores(CoreType::ETH, CoordSystem::TRANSLATED)) {
        try {
            dev->read_from_device(buf.data(), eth, base, buf.size());
        } catch (...) {
            continue;  // core not readable (harvested, in reset) -- not an error here
        }
        util_agg_hdr_view_t hdr{};
        std::memcpy(&hdr, buf.data(), sizeof(hdr));
        if (hdr.magic != UTIL_AGG_MAGIC) {
            continue;
        }
        // Chunk 0 is self-validating (head/head_xor); the static checksum guards the
        // layout. A remote read is served in 16 B chunks from different moments, so
        // this pair is the only way to know the head is trustworthy.
        if (!util_agg_hdr_ok(hdr) ||
            util_agg_hdr_checksum(hdr.magic, hdr.version, hdr.capacity, hdr.num_cores, hdr.src_chip) !=
                hdr.hdr_checksum) {
            continue;  // torn or not ours; the next probe will catch it
        }
        Landing l;
        l.core = eth;
        l.src_chip = hdr.src_chip;
        l.capacity = hdr.capacity;
        l.num_cores = hdr.num_cores;
        l.last_head = hdr.head;
        l.last_sweep_count = hdr.sweep_count;
        l.last_advance_us = now_us;
        found.push_back(l);
    }
    return found;
}

}  // namespace ttnvtop_agg

struct ChipState {
    uint64_t asic_id = 0;
    tt::ChipId chip_id = 0;
    tt::ARCH arch = tt::ARCH::Invalid;
    bool is_remote = false;
    std::vector<CoreState> cores;
    ttnvtop::ShmPublisher publisher;

    // ---- BEGIN ported from tt_coremon (ChipState, source lines 546-566)
    // DRAM bandwidth axis. Endpoints in TRANSLATED coords; noc0 Y is kept
    // alongside because WH selects one of three NIU bases from it.
    std::vector<tt_xy_pair> dram_at;
    std::vector<uint32_t> dram_noc0_y;
    std::vector<uint32_t> dram_prev_rd, dram_prev_wr;
    bool dram_primed = false;
    // AICLK cached by this chip's OWN telemetry thread. Read on the publish path, which
    // must never touch the device: get_clock() is a legacy ARC message on Wormhole and we
    // have measured it timing out ("Timed out after waiting 1000 ms for ARC to respond"),
    // so calling it inline made one sick chip stall the publish of every chip behind it in
    // the same pass -- the whole SHM set went stale and the viewer showed a frozen grid.
    // Plain word, like the dram_* fields beside it: one aligned u32 written by this
    // chip's telemetry thread and read by the publisher. The race is benign and
    // advisory -- a reader either sees the old clock or the new one -- and ChipState
    // has to stay movable because `chips` is a vector that grows during discovery.
    uint32_t aiclk_mhz = 0;
    uint32_t dram_rd_mbps = 0;
    uint32_t dram_wr_mbps = 0;
    uint32_t dram_peak_mbps = 0;

    // Remote per-core-drain budget (see CliOptions::remote_budget). Cores are visited
    // round-robin from `drain_rr_cursor`, spending from a token bucket, so a constrained
    // budget stretches the revisit interval instead of freezing whole chips. Touched
    // only by this chip's own drain thread (the drain is one thread per chip), so these
    // need no synchronization of their own.
    size_t drain_rr_cursor = 0;
    double drain_tokens = 0.0;
    std::chrono::steady_clock::time_point drain_last_refill{};
    uint64_t drain_cores_sampled = 0;  // since the last drain debug line
    // ---- END ported from tt_coremon

    // Phase 2.1.c: per-kernel cycle attribution. Updated by the drain
    // thread; protected by `ring_mx`. Wrapped in unique_ptr to keep
    // ChipState movable (std::mutex is neither copyable nor movable, and
    // ChipState is moved into the `chips` vector at startup).
    std::unique_ptr<std::mutex> ring_mx = std::make_unique<std::mutex>();
    std::unordered_map<uint32_t, KernelTimeAccumulator> kernel_cycles;
    // Phase 2.1.c.i: per-kernel monotonic cycle total since collector start.
    // Updated alongside kernel_cycles[kid].add() — but never decays. Lives
    // here (not in KernelTimeAccumulator) so that kernels which decay-out of
    // kernel_cycles still retain their cumulative total for compare.py.
    std::unordered_map<uint32_t, uint64_t> kernel_cycles_total;
    uint64_t drain_ticks = 0;         // total ring-drain ticks completed (debug)
    // PER-CHIP debug-log throttle. These were `static thread_local` locals inside the
    // per-chip loop, which made the 5 s gate GLOBAL TO THE DRAIN THREAD: whichever chip
    // the loop reached first after the interval elapsed logged and reset the timer, and
    // the other seven were silently skipped. On a T3K that printed 15 lines for chip 0
    // and one each for chips 4, 5 and 7 over 75 s -- which reads exactly like the drain
    // starving the other chips, and is nothing of the kind. It also made drain_hz
    // meaningless, since the tick baseline came from whichever chip logged last.
    uint64_t last_debug_us = 0;
    uint64_t last_debug_ticks = 0;
    // Also per-chip, and for the same reason: the drain runs ONE THREAD PER CHIP, so a
    // `static` local in that lambda is shared by every thread and the first chip to pass
    // the interval gate silently suppresses the rest. As `static` locals these two meant
    // only one chip of eight ever had its sampler period reasserted or probed.
    uint64_t last_period_assert_us = 0;
    uint64_t last_period_probe_us = 0;

    // JOURNAL FEED (the on-chip aggregator as the data source for this chip).
    //
    // When an aggregator is running on this chip we publish from ITS accumulators instead
    // of draining 64 per-core rings ourselves, and we stop the ring drain for this chip
    // entirely. That is not merely an optimisation: on a remote chip the per-core drain
    // measured 0.1-0.2 Hz and 99.4-99.9% sample loss under a CCL workload, while the
    // aggregator measured lost=0 on every remote chip across four runs. Dropping the
    // 64 KiB-per-pass tunnel traffic also removes the contention that caused it.
    bool journal_active = false;
    tt::umd::CoreCoord journal_core{};
    uint32_t journal_num_cores = 0;
    std::vector<int> journal_to_core;  // journal state index -> index into `cores`
    std::vector<uint32_t> prev_busy, prev_wall, prev_seq;
    uint64_t drain_lost_samples = 0;  // count of entries dropped because head moved more than ring capacity (debug)
    uint64_t drain_entries_seen = 0;  // total ring entries ingested across all cores (debug)

    // Phase 2.2 M1. Populated on an MMIO chip for each remote chip whose aggregator
    // lands its journal here. Empty when no aggregator is running, in which case the
    // per-core tunnel drain above remains the only path -- the fallback is not a
    // degraded mode, it is what the collector has always done.
    std::vector<ttnvtop_agg::Landing> landings;
    uint64_t journal_entries_seen = 0;
    uint64_t journal_lost_reported = 0;  // `lost` as counted BY THE AGGREGATOR, on-chip
    uint64_t journal_torn_headers = 0;   // checksum mismatches: read again, never skip forward
    uint64_t journal_stale_ticks = 0;
};

// Phase 2.1.c registry-SHM writer. Opens `/dev/shm/tt_program_registry`
// read-write for `cycles_in_window` updates only. The workload-side
// registrar library still owns runtime_id+pid+epoch_us+name (slot claim);
// we just patch the disjoint cycle field. Magic mismatch -> disabled, no
// writes happen until the workload (re)initializes the file.
struct RegistryWriter {
    int fd = -1;
    void* map = nullptr;
    size_t map_size = 0;
    ttnvtop::RegistryHeader* header = nullptr;
    ttnvtop::RegistryEntry* entries = nullptr;
    bool enabled = false;

    bool open_or_attach() {
        const size_t want = ttnvtop::registry_file_size();
        // O_RDWR — workload may not have started yet; if the file does not
        // exist we skip and re-probe periodically. We do NOT create the
        // file: that's the workload's job (it sets the header magic).
        fd = ::open(ttnvtop::kRegistryShmPath, O_RDWR);
        if (fd < 0) {
            return false;
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < want) {
            ::close(fd);
            fd = -1;
            return false;
        }
        map = ::mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            ::close(fd);
            fd = -1;
            map = nullptr;
            return false;
        }
        map_size = want;
        header = static_cast<ttnvtop::RegistryHeader*>(map);
        entries = reinterpret_cast<ttnvtop::RegistryEntry*>(static_cast<char*>(map) + sizeof(ttnvtop::RegistryHeader));
        if (std::memcmp(header->magic, ttnvtop::kRegistryMagic, sizeof(ttnvtop::kRegistryMagic)) != 0 ||
            header->version != ttnvtop::kRegistryVersion ||
            header->entry_size != static_cast<uint16_t>(sizeof(ttnvtop::RegistryEntry))) {
            ::munmap(map, map_size);
            ::close(fd);
            map = nullptr;
            fd = -1;
            return false;
        }
        enabled = true;
        return true;
    }

    // Find the slot whose runtime_id matches the program identified by `kid`
    // and update both `cycles_in_window` (rolling, viewer TIME%) and
    // `cycles_total` (monotonic, compare.py).
    //
    // ID encoding: every dispatch path observed in tt-metal writes
    // `host_assigned_id = EncodePerDeviceProgramID(runtime_id, dev_id)
    //                  = (runtime_id << 10) | dev_id` into launch_msg.
    // The LLK Hook B propagates this encoded value into the L1 ring
    // entries verbatim. The registrar (`register_program`) on the host
    // side writes the RAW `program.get_runtime_id()` — also at every call
    // site we've audited. So `kid` is always encoded and `entries[i].runtime_id`
    // is always raw. We decode `kid` to its raw form before lookup.
    //
    // Empirical confirmation (run 20260427-160642, Llama-3.2-1B 1-layer):
    //   profiler host_assigned_id range: 1024..974079 (all >= 1024 ⇒ all encoded)
    //   registry rid range:              0..2079 (raw runtime_ids)
    // Direct-match-only would catch only the two registry slots whose raw
    // rid coincidentally equals an encoded kid (rid=1024 ⇒ kid=encoded(1,0),
    // rid=2048 ⇒ kid=encoded(2,0)) — i.e., 2 out of ~460 attributable
    // programs. Decoded match catches all of them.
    //
    // An earlier version tried "direct OR decoded" (dual lookup). That was
    // wrong: raw rids routinely exceed 1024, so for any kid with the same
    // numeric value as an unrelated raw rid, BOTH entries would be
    // updated, causing 1024× cross-contamination. Decoded-only avoids it.
    //
    // Linear scan over the live region — capacity is 16384 so worst case
    // is a few thousand reads, well below the drain budget. Writers can
    // race with us on a slot's name field, but we touch only the disjoint
    // cycle fields so no torn-read window.
    // The caller passes a RAW runtime_id (already decoded from any
    // host_assigned_id encoding by the ring drain). We match directly
    // against `entries[i].runtime_id`, which is also raw.
    void update_kernel_cycles(uint32_t runtime_id, uint64_t cycles_in_window, uint64_t cycles_total) {
        if (!enabled) {
            return;
        }
        const uint32_t total = header->write_cursor.load(std::memory_order_relaxed);
        const uint32_t scan = std::min<uint32_t>(total, ttnvtop::kRegistryCapacity);
        for (uint32_t i = 0; i < scan; ++i) {
            if (entries[i].runtime_id == runtime_id) {
                entries[i].cycles_in_window = cycles_in_window;
                entries[i].cycles_total = cycles_total;
            }
        }
    }
};

std::atomic<bool> g_stop{false};

// Read L1 in <=1 KiB pieces, checking g_stop between them.
//
// UMD splits a remote transfer into MAX_BLOCK_SIZE tunnel transactions -- 1024 B when the
// process has no SysmemManager, which a UMD-only monitor does not -- and holds the
// INTERPROCESS mutex `NON_MMIO_<n>_PCIe` for the whole call, not per block. So the 2112 B
// journal read (64 B header + 64 cores x 32 B) is three blocks of held lock with no point
// at which a stop can take effect. One block per call gives back both: the lock is
// released between blocks so another process can interleave, and a stop lands within one
// block instead of never. See 5u -- two feed threads spinning inside this call held two of
// those locks for nine minutes and starved a Llama teardown in an unrelated process.
//
// On an MMIO chip this is a plain BAR access and the chunking costs nothing measurable.
inline bool read_chunked_stoppable(TTDevice* dev, const tt_xy_pair& core, uint64_t addr, uint8_t* dst, uint32_t bytes) {
    constexpr uint32_t kChunk = 1024;
    for (uint32_t off = 0; off < bytes; off += kChunk) {
        if (g_stop.load(std::memory_order_relaxed)) {
            return false;  // partial buffer; caller must not publish it
        }
        dev->read_from_device(dst + off, core, addr + off, std::min(kChunk, bytes - off));
    }
    return true;
}

// Set once discovery has returned. Until then the process is inside a blocking UMD
// call that no signal can interrupt.
std::atomic<bool> g_discovery_done{false};

void handle_sigint(int) {
    g_stop.store(true);
    // During topology discovery there is no loop checking g_stop -- we are inside UMD,
    // which can block for up to its ETH/ARC startup timeouts (ARC_STARTUP_TIMEOUT is
    // 300 s, compile-time). A SIGTERM there does nothing, which is how an ordinary slow
    // start became an unkillable process that needed a board reset.
    //
    // Exiting here is safe SPECIFICALLY because discovery is read-only: it probes
    // heartbeats, versions and mailboxes. There is no half-finished write to strand, so
    // unlike a kill during the sampling loop this cannot leave the tunnel inconsistent.
    if (!g_discovery_done.load(std::memory_order_relaxed)) {
        const char msg[] = "ttnvtop-collector: signal during topology discovery — exiting.\n";
        ssize_t ignored = ::write(STDERR_FILENO, msg, sizeof(msg) - 1);
        (void)ignored;
        ::_exit(130);
    }
}

const char* arch_name(tt::ARCH a) {
    switch (a) {
        case tt::ARCH::WORMHOLE_B0: return "Wormhole";
        case tt::ARCH::BLACKHOLE: return "Blackhole";
        case tt::ARCH::QUASAR: return "Quasar";
        default: return "Unknown";
    }
}

// Arm the FPU counter bank on a single Tensix for free-running continuous
// mode. Rising edge on FPU2[0] both clears and starts; this single rising
// edge starts BOTH underlying counters (FPU_INSTRUCTION and SFPU_INSTRUCTION)
// — they tick in parallel in hardware. The collector later reads them in
// turn by muxing counter_sel (PERF_CNT_FPU1[16:8]) per tick. We must NOT
// pulse start again after this, or the accumulators would reset.
// Returns true on success; false if any of the four writes threw. A user
// kernel that calls StartPerfCounters later will fight with this (transient
// reset + new start), which our sampler's negative-delta guard tolerates.
bool arm_fpu_counter(TTDevice* device, const tt_xy_pair& core) {
    const auto write32 = [&](uint64_t addr, uint32_t value) {
        device->write_to_device(&value, core, addr, sizeof(value));
    };
    try {
        write32(kRegFpu0, 0u);  // reference period (unused in continuous)
        // mode=0 continuous, counter_sel=0 (FPU). SFPU will be sampled by
        // muxing counter_sel=1 on later ticks; the underlying SFPU counter
        // is already accumulating regardless of the mux setting.
        write32(kRegFpu1, (kFpuCounterSelFpu << kFpuCounterSelShift) | kFpuModeContinuous);
        write32(kRegFpu2, 0u);  // ensure start bit is low
        write32(kRegFpu2, 1u);  // rising edge on start clears + starts
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

// ---- BEGIN ported from tt_coremon (source lines 793-870, 905-947)
// ---------------------------------------------------------------------------
// DRAM bandwidth axis: NIU counters at the DRAM NOC endpoints.
//
// These are FREE-RUNNING -- no arming, so unlike the Tensix perf counters they
// carry no two-owner hazard and cannot be reset out from under us.
//
// Read at the DRAM endpoints, not summed from the workers: a Tensix master-side
// NIU counts words regardless of target, so it cannot separate DRAM traffic from
// L1<->L1 mcast. Every Tensix->DRAM transfer traverses exactly one DRAM endpoint,
// so summing the slave side counts each transfer exactly once.
//
// Validated (see alex-notes/claude-memory/tt-core-utilization-metrics.md):
//   512 MiB host->DRAM, exact byte count : +0.17%  (BH)
//   4096^3 matmul, BH                    : +2.7%, R:W exactly 2.000:1
//   4096^3 matmul, WH                    : +1.5%, R:W 1.998:1
// Accuracy ceiling: these count NOC words, not GDDR bursts, so they miss
// read-modify-write amplification and refresh overhead.
constexpr uint64_t kNocRegsBase = 0xFFB20000ull;
constexpr uint64_t kNocInstanceOff = 0x10000ull;
constexpr uint64_t kNocStatusOff = 0x200ull;
constexpr uint32_t kNumNocs = 2;
// Slave-side: what the endpoint SERVED, i.e. the GDDR bytes.
constexpr uint32_t kNiuSlvRdDataWordSent = 0x33;
constexpr uint32_t kNiuSlvNonpostedWrDataWordRecv = 0x38;
constexpr uint32_t kNiuSlvPostedWrDataWordRecv = 0x39;

// WH DRAM endpoints do NOT share the Tensix NIU addressing. Blackhole does reuse
// it (0xFFB20000, stride 0x10000), but on WH each DRAM block carries THREE NIU
// instances at separate bases, selected by the noc0 Y coordinate mod 3.
//
// Getting this wrong does not fault -- it reads a dead region and returns a
// confident 0.00 GB/s, indistinguishable from an idle chip. Mapping mirrors
// ttexalens/hardware/wormhole/dram_block.py.
constexpr uint64_t kWhDramNiuBase[3][2] = {
    {0x100080000ull, 0x100088000ull},  // y%3 == 1
    {0x100090000ull, 0x100098000ull},  // y%3 == 2
    {0x1000A0000ull, 0x1000A8000ull},  // y%3 == 0
};
constexpr uint64_t wh_dram_niu_base(uint32_t noc0_y, uint32_t noc) {
    const uint32_t m = noc0_y % 3;
    const uint32_t loc = (m == 1) ? 0u : (m == 2) ? 1u : 2u;
    return kWhDramNiuBase[loc][noc];
}
constexpr uint64_t dram_noc_status(bool blackhole, uint32_t noc0_y, uint32_t noc, uint32_t idx) {
    return blackhole ? (kNocRegsBase + noc * kNocInstanceOff + kNocStatusOff + idx * 4)
                     : (wh_dram_niu_base(noc0_y, noc) + kNocStatusOff + idx * 4);
}
// NOC payload width, words -> bytes. Arch-specific (unlike the perf-counter
// block): NOC_PAYLOAD_WIDTH is 256 bits on WH, 512 on BH.
constexpr uint32_t noc_word_bytes(bool blackhole) { return blackhole ? 64u : 32u; }
// Peak GDDR bandwidth in MB/s, as channels x 32 bit x rate / 8:
//   WH n300  6 ch x 12 Gbps = 288 GB/s
//   BH p150a 8 ch x 16 Gbps = 512 GB/s
// Fallback only -- prefer the board's reported dram_speed. tt-metal hardcodes
// WH=384 (test_dram_read.cpp), which is a 16 Gbps part; 12G boards peak at 288,
// so 384 understates utilization by 33%.
constexpr uint32_t dram_peak_mbps_default(bool blackhole) { return blackhole ? 512000u : 288000u; }
// The NIU counters are 32-bit and WRAP -- at full bandwidth roughly every 8 s per
// endpoint, so the sample interval must stay well under that.
constexpr uint32_t wrap_delta(uint32_t a, uint32_t b) { return b - a; }
constexpr auto kDramSampleInterval = std::chrono::milliseconds(200);  // 5 Hz, vs ~8 s wrap

// Approximate tunnelled transactions the RING DRAIN spends per core per pass on a
// remote chip -- this constant only prices the remote budget, since local chips are
// never budgeted. The drain's per-core cost is one 1 KiB read of the L1 sampler ring;
// the period reassert (1 Hz) and the read-back probe (0.2 Hz) amortize to nothing
// against a 200 Hz pass. It is 1.0 rather than tt_coremon's kOpsPerCore = 6.0 (source
// line 955) because that priced a different access pattern: the perf-counter sweep's
// six small register reads per core, not one bulk ring read.
//
// Sanity on the 1500 tx/s default: 64 cores / 1500 = a 43 ms rotation, against a
// 62-slot ring that fills in ~155 ms at the 5 ms firmware period x 2 producers. So
// the default budget still revisits every core before its ring can wrap. tt_coremon's
// kMaxRotationSeconds hard-fail (source line 969) is deliberately NOT ported: it
// guarded the 32-bit FPU counter's ~4.3 s wrap, where a wrapped delta is silently
// DISCARDED and the chip reports nothing at all. The ring has no such failure mode --
// it counts what it dropped in ChipState::drain_lost_samples -- so a too-small budget
// here degrades visibly rather than lying, which is the whole point of the fallback.
constexpr double kDrainOpsPerCore = 1.0;

// Roll one chip's DRAM rates forward over `dt` seconds. The first call only
// primes the anchor. Read failures leave the previous rate standing rather than
// publishing a false 0.
void dram_update(ChipState& chip, double dt) {
    if (chip.dram_at.empty() || chip.cores.empty()) {
        return;
    }
    const bool bh = chip.arch == tt::ARCH::BLACKHOLE;
    TTDevice* dev = chip.cores.front().device;
    std::vector<uint32_t> rd, wr;
    rd.reserve(chip.dram_at.size() * kNumNocs);
    wr.reserve(chip.dram_at.size() * kNumNocs);
    const auto r32 = [&](const tt_xy_pair& c, uint64_t addr) -> uint32_t {
        uint32_t v = 0;
        dev->read_from_device(&v, c, addr, sizeof(v));
        return v;
    };
    try {
        for (size_t i = 0; i < chip.dram_at.size(); ++i) {
            const auto& d = chip.dram_at[i];
            const uint32_t y = chip.dram_noc0_y[i];
            for (uint32_t n = 0; n < kNumNocs; ++n) {
                rd.push_back(r32(d, dram_noc_status(bh, y, n, kNiuSlvRdDataWordSent)));
                wr.push_back(static_cast<uint32_t>(
                    r32(d, dram_noc_status(bh, y, n, kNiuSlvNonpostedWrDataWordRecv)) +
                    r32(d, dram_noc_status(bh, y, n, kNiuSlvPostedWrDataWordRecv))));
            }
        }
    } catch (const std::exception&) {
        return;  // keep the last good rate
    }
    if (chip.dram_primed && dt > 0.0 && rd.size() == chip.dram_prev_rd.size()) {
        const double wb = static_cast<double>(noc_word_bytes(bh));
        uint64_t r = 0, w = 0;
        for (size_t i = 0; i < rd.size(); ++i) {
            r += wrap_delta(chip.dram_prev_rd[i], rd[i]);
            w += wrap_delta(chip.dram_prev_wr[i], wr[i]);
        }
        chip.dram_rd_mbps = static_cast<uint32_t>(static_cast<double>(r) * wb / dt / 1e6);
        chip.dram_wr_mbps = static_cast<uint32_t>(static_cast<double>(w) * wb / dt / 1e6);
    }
    chip.dram_prev_rd = std::move(rd);
    chip.dram_prev_wr = std::move(wr);
    chip.dram_primed = true;
}
// ---- END ported from tt_coremon

}  // namespace

int main(int argc, char* argv[]) {
    CliOptions cli;
    if (!parse_cli(argc, argv, cli)) {
        print_help(argv[0]);
        return 2;
    }
    if (cli.show_help) {
        print_help(argv[0]);
        return 0;
    }
    if (!cli.log_file.empty()) {
        // Redirect fd 1 and fd 2 at the kernel level so UMD/spdlog — which
        // caches its own FILE* at library load — also routes into the file.
        int lfd = ::open(cli.log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (lfd < 0) {
            std::cerr << "ttnvtop-collector: cannot open " << cli.log_file << " for writing\n";
            return 1;
        }
        ::dup2(lfd, STDOUT_FILENO);
        ::dup2(lfd, STDERR_FILENO);
        ::close(lfd);
        std::fprintf(stderr, "--- ttnvtop-collector log start ---\n");
        std::fflush(stderr);
    }

    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    // Mailbox field layout — resolved at compile time from the WH struct.
    constexpr uint64_t kMailboxBase = static_cast<uint64_t>(MEM_MAILBOX_BASE);
    constexpr size_t kGoMessagesOff = offsetof(mailboxes_t, go_messages);
    constexpr size_t kGoIndexOff = offsetof(mailboxes_t, go_message_index);
    static_assert(kGoIndexOff > kGoMessagesOff, "unexpected mailboxes_t layout");
    constexpr size_t kReadSize = (kGoIndexOff + sizeof(uint32_t)) - kGoMessagesOff;
    constexpr size_t kIdxOffInBuf = kGoIndexOff - kGoMessagesOff;
    static_assert(sizeof(go_msg_t) == sizeof(uint32_t), "go_msg_t must be 4 bytes");
    const uint64_t read_addr = kMailboxBase + static_cast<uint64_t>(kGoMessagesOff);

    // Level 1 kernel attribution (Phase 2.1.e).
    //   host_assigned_id layout per dev_msgs.h line 161-165:
    //     [9:0]   physical device id
    //     [30:10] program id
    //     [31]    0 (program-running-on-device marker)
    //   launch[] is an 8-entry ring; launch_msg_rd_ptr indexes the currently-
    //   executing slot. At slot boundaries the value is transiently stale —
    //   acceptable for a 4 Hz display. Tie to the D bar for "is this value
    //   meaningful right now": if D ≈ 0, kernel_id is just whatever ran last.
    constexpr size_t kLaunchRdPtrOff = offsetof(mailboxes_t, launch_msg_rd_ptr);
    constexpr size_t kLaunchArrOff = offsetof(mailboxes_t, launch);
    constexpr size_t kLaunchEntrySize = sizeof(launch_msg_t);
    // launch_msg_t is effectively kernel_config_msg_t (no other fields). Be
    // explicit about the offset chain so this stays correct if that changes.
    constexpr size_t kHostIdOffInLaunch =
        offsetof(launch_msg_t, kernel_config) + offsetof(kernel_config_msg_t, host_assigned_id);
    constexpr uint32_t kLaunchBufEntries = launch_msg_buffer_num_entries;

    TopologyDiscoveryOptions opts;
    // --local-only is the RESCUE path. Remote discovery is what hangs when an ethernet
    // core's firmware heartbeat has stopped: UMD walks the links to build the tunnel and
    // sits in ETH_STARTUP_TIMEOUT (10 s/core, compile-time) on the dead one. Skipping it
    // reaches the MMIO chips over their own BARs, which cannot hang on a remote link, so
    // `--stop-aggregator --local-only` can still clear the local ethernet cores. Freeing
    // those is often what lets a subsequent full discovery -- and therefore a full
    // `--stop-aggregator` -- succeed without a board reset. See 5u.
    opts.discover_remote_devices = !cli.local_only;
    opts.wait_on_ethernet_link_training = false;
    opts.low_power = true;
    opts.cmfw_mismatch_action = TopologyDiscoveryOptions::Action::IGNORE;
    opts.cmfw_unsupported_action = TopologyDiscoveryOptions::Action::IGNORE;
    opts.eth_fw_mismatch_action = TopologyDiscoveryOptions::Action::IGNORE;
    opts.eth_fw_heartbeat_failure = TopologyDiscoveryOptions::Action::IGNORE;
    opts.unexpected_routing_firmware_config = TopologyDiscoveryOptions::Action::IGNORE;

    // Bound the startup. UMD's discovery has its own timeouts -- ETH_STARTUP_TIMEOUT
    // 10 s per core, ARC_STARTUP_TIMEOUT 300 s -- all compile-time, so a degraded chip
    // can hold the process for minutes with no output. For a monitoring tool that is
    // indistinguishable from a hang, and it cost several board resets to diagnose.
    // A watchdog turns it into a bounded, diagnosable failure.
    {
        const char* env = std::getenv("TTNVTOP_DISCOVERY_TIMEOUT_S");
        const int limit_s = env ? std::atoi(env) : 60;
        if (limit_s > 0) {
            std::thread([limit_s]() {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(limit_s);
                while (!g_discovery_done.load(std::memory_order_relaxed)) {
                    if (std::chrono::steady_clock::now() > deadline) {
                        std::fprintf(
                            stderr,
                            "ttnvtop-collector: topology discovery did not finish within %d s.\n"
                            "  UMD is still inside its own timeouts (ETH_STARTUP 10 s/core, ARC_STARTUP 300 s).\n"
                            "  Usual cause: an ethernet core whose firmware heartbeat has stopped - for example a\n"
                            "  persistent kernel occupying ERISC0, or a process killed mid-transaction.\n"
                            "  A collector exiting here leaves nothing behind: it has written no bytes yet.\n"
                            "  If this persists, an ethernet core is still holding a stopped heartbeat from\n"
                            "  an EARLIER process that did not stop its aggregator; that needs a board reset.\n"
                            "  Raise the bound with TTNVTOP_DISCOVERY_TIMEOUT_S if this machine is simply slow.\n",
                            limit_s);
                        std::fflush(stderr);
                        ::_exit(3);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
            }).detach();
        }
    }
    auto [cluster_desc, devices] = TopologyDiscovery::discover(opts);
    g_discovery_done.store(true, std::memory_order_relaxed);

    // SHUTDOWN WATCHDOG -- the guarantee that a stop is a stop.
    //
    // g_stop is cooperative, and a thread inside UMD's remote-tunnel poll does not
    // cooperate: it spins on the remote ERISC command queue with no timeout, holding the
    // interprocess `NON_MMIO_<n>_PCIe` mutex the whole time. No signal reaches it. In 5u
    // two feed threads sat like that at 100% CPU for nine minutes after a SIGINT, holding
    // two of those locks, and a Llama-70B teardown in a completely separate process
    // blocked behind them until this collector was SIGKILLed from outside.
    //
    // A monitor that can wedge the machine it monitors is worse than no monitor, so the
    // last resort is unconditional: once a stop has been requested, leave -- joined or
    // not. Leaving is what releases the locks. They are robust mutexes, so the kernel
    // marks them owner-died and the next acquirer recovers them; that is measured, not
    // assumed -- the blocked process resumed 20 s after the kill that proved this path.
    {
        const char* env = std::getenv("TTNVTOP_SHUTDOWN_GRACE_S");
        const int grace_s = env ? std::atoi(env) : 5;
        if (grace_s > 0) {
            std::thread([grace_s]() {
                while (!g_stop.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                // A clean exit returns from main well inside this window and takes the
                // whole process, this thread included, with it. Reaching the end of the
                // sleep means at least one thread is stuck where g_stop cannot reach.
                std::this_thread::sleep_for(std::chrono::seconds(grace_s));
                const char msg[] =
                    "ttnvtop-collector: a thread did not stop within the grace period -- almost certainly\n"
                    "  inside a UMD remote-tunnel poll, which cannot be interrupted and holds an\n"
                    "  interprocess lock. Exiting hard to release it. Any journal still resident on an\n"
                    "  ethernet core is untouched: clear it with --stop-aggregator.\n";
                ssize_t ignored = ::write(STDERR_FILENO, msg, sizeof(msg) - 1);
                (void)ignored;
                ::_exit(130);
            }).detach();
        }
    }
    if (devices.empty()) {
        std::cerr << "ttnvtop-collector: no Tenstorrent devices discovered.\n";
        return 1;
    }

    std::vector<ChipState> chips;
    chips.reserve(devices.size());
    for (auto& [chip_id, dev_up] : devices) {
        if (!cli.device_filter.empty() &&
            cli.device_filter.find(static_cast<int>(chip_id)) == cli.device_filter.end()) {
            continue;
        }
        TTDevice* dev = dev_up.get();
        const tt::ARCH arch = dev->get_arch();
        // MONITORING is Wormhole-only for Phase 1 -- the per-core sampler drain and the
        // shm publisher are built against WH's grid and heartbeat. The LAUNCHER and the
        // journal-side DIAGNOSTICS are not: the launch contract is arch-independent L1
        // writes, the journal header is self-describing, and the kernel is known to run
        // on Blackhole (120 cores). Admitting BH for those is what makes this work
        // debuggable on a board where a wedged eth core costs nothing, instead of only on
        // the T3K where it costs a board reset (tt-eth-idle-core-firmware-contracts).
        const bool arch_agnostic_mode = !cli.launch_artifact.empty() || cli.journal_probe || cli.fidelity_probe_s > 0 ||
                                        cli.read_latency_probe || cli.stop_aggregator || !cli.peek.empty() ||
                                        cli.watchdog_s > 0;
        if (arch != tt::ARCH::WORMHOLE_B0 && !(arch_agnostic_mode && arch == tt::ARCH::BLACKHOLE)) {
            std::cerr << "ttnvtop-collector: skipping chip " << chip_id << " arch " << arch_name(arch)
                      << " (Wormhole-only for Phase 1).\n";
            continue;
        }
        ChipState chip;
        chip.chip_id = chip_id;
        chip.asic_id = static_cast<uint64_t>(chip_id);
        chip.arch = arch;
        chip.is_remote = dev->is_remote();

        // UMD moved SocDescriptor's first parameter from tt::ARCH to a
        // shared_ptr<const SocArchDescriptor>. This mirrors exactly what
        // TTDevice::construct_soc_descriptor(nullptr) does internally, so the
        // descriptor is identical to before — and unlike dev->get_soc_descriptor()
        // it does not require init_tt_device() to have run (which probes ARC and
        // would undercut this tool's non-perturbing coexistence model).
        SocDescriptor soc(std::make_shared<SocArchDescriptor>(arch), dev->get_chip_info());
        const auto worker_cores_noc0 = soc.get_cores(CoreType::TENSIX, CoordSystem::NOC0);
        chip.cores.reserve(worker_cores_noc0.size());
        for (const auto& cc_noc0 : worker_cores_noc0) {
            const auto cc_trans = soc.translate_coord_to(cc_noc0, CoordSystem::TRANSLATED);
            CoreState c;
            c.noc_x = cc_noc0.x;
            c.noc_y = cc_noc0.y;
            c.translated = tt_xy_pair(cc_trans.x, cc_trans.y);
            c.device = dev;
            c.is_remote = chip.is_remote;
            chip.cores.push_back(std::move(c));
        }
        std::sort(chip.cores.begin(), chip.cores.end(), [](const CoreState& a, const CoreState& b) {
            if (a.noc_y != b.noc_y) {
                return a.noc_y < b.noc_y;
            }
            return a.noc_x < b.noc_x;
        });

        if (!chip.publisher.open(
                chip.asic_id,
                static_cast<uint32_t>(arch),
                ttnvtop::SIGNAL_SRC_DISPATCH | ttnvtop::SIGNAL_SRC_COMPUTE,
                static_cast<uint32_t>(chip.cores.size()))) {
            std::cerr << "ttnvtop-collector: failed to open /dev/shm for asic " << chip.asic_id << " (errno " << errno
                      << ").\n";
            continue;
        }
        // Populate static per-core fields once, and arm the FPU perf counter
        // so it runs continuously. Kernels that call StartPerfCounters will
        // transiently reset it; the sampler's delta guard drops those ticks.
        //
        // --journal-probe advertises itself as read-only, so it must be: arming the
        // perf counters, overriding period_cycles and the FPU_OUT_L liveness probe are
        // all device WRITES, and on a remote chip they cross the NON_MMIO tunnel.
        // Publish the static per-core fields, then skip every write below.
        if (cli.journal_probe || cli.read_latency_probe || cli.fidelity_probe_s > 0 || !cli.peek.empty() ||
            cli.watchdog_s > 0 || !cli.launch_artifact.empty()) {
            for (size_t i = 0; i < chip.cores.size(); ++i) {
                auto& v = chip.publisher.cores()[i];
                v.noc_x = static_cast<uint8_t>(chip.cores[i].noc_x);
                v.noc_y = static_cast<uint8_t>(chip.cores[i].noc_y);
                v.is_remote = chip.is_remote ? 1u : 0u;
            }
            chips.push_back(std::move(chip));
            continue;
        }

        // ---- BEGIN ported from tt_coremon (source lines 1333-1367)
        // DRAM endpoints. WH exposes 6 channels x 3 endpoints = 18, BH 8 x 3 = 24
        // (BH really is 8 channels, even though NUM_DRAM_INSTANCE reads 6 in both
        // syseng bringup trees).
        //
        // Deliberately placed AFTER the read-only-probe early-continue above, not
        // before it as in the source: get_dram_speed() below is a firmware/telemetry
        // read, and --journal-probe advertises itself as touching nothing. Probe-mode
        // chips therefore leave dram_peak_mbps at 0, which the schema already defines
        // as "unknown, so hide %-of-peak".
        for (const auto& cc : soc.get_cores(CoreType::DRAM, CoordSystem::NOC0)) {
            const auto t = soc.translate_coord_to(cc, CoordSystem::TRANSLATED);
            chip.dram_at.push_back(tt_xy_pair(t.x, t.y));
            chip.dram_noc0_y.push_back(static_cast<uint32_t>(cc.y));
        }
        // Peak = channels x 32 bit x rate / 8, with the rate read from the board
        // rather than assumed. This matters: tt-metal hardcodes WH=384 GB/s,
        // which is a 16 Gbps part -- on a 12 Gbps n300 (peak 288) that overstates
        // the denominator and understates utilization by 33%.
        chip.dram_peak_mbps = 0;
        {
            const bool bh = arch == tt::ARCH::BLACKHOLE;
            std::optional<uint16_t> gbps;  // per-pin rate, Mbps
            try {
                if (auto* fw = dev->get_firmware_info_provider(); fw != nullptr) {
                    gbps = fw->get_dram_speed();
                }
            } catch (const std::exception&) {
            }
            if (gbps.has_value() && *gbps > 0) {
                // UMD reports the per-pin rate in Mbps (16000 for a "16G" part),
                // not Gbps. Accept either so a units change upstream cannot
                // silently inflate the denominator by 1000x.
                const uint32_t rate_mbps =
                    (*gbps >= 1000u) ? static_cast<uint32_t>(*gbps) : static_cast<uint32_t>(*gbps) * 1000u;
                // channels x 32-bit bus x rate / 8 bits-per-byte  ->  MB/s
                const uint32_t channels = bh ? 8u : 6u;
                chip.dram_peak_mbps = channels * 4u * rate_mbps;
            } else {
                chip.dram_peak_mbps = dram_peak_mbps_default(bh);
                std::cerr << "ttnvtop-collector: chip " << chip_id << " dram_speed unavailable; assuming "
                          << chip.dram_peak_mbps / 1000 << " GB/s peak.\n";
            }
        }
        // ---- END ported from tt_coremon

        int armed = 0;
        for (size_t i = 0; i < chip.cores.size(); ++i) {
            auto& v = chip.publisher.cores()[i];
            v.noc_x = static_cast<uint8_t>(chip.cores[i].noc_x);
            v.noc_y = static_cast<uint8_t>(chip.cores[i].noc_y);
            v.logical_x = 0;  // Phase 1: logical coords not yet wired
            v.logical_y = 0;
            v.is_remote = chip.is_remote ? 1u : 0u;
            chip.cores[i].counter_armed = arm_fpu_counter(chip.cores[i].device, chip.cores[i].translated);
            if (chip.cores[i].counter_armed) {
                ++armed;
            }
        }
        std::cerr << "ttnvtop-collector: chip " << chip.chip_id << " armed FPU counter on " << armed << "/"
                  << chip.cores.size() << " cores.\n";

        // Phase 2.1.c.i: override `period_cycles` on every core's L1 sampler
        // ring header. The firmware default is set at brisc init() time from
        // util_sampler.h's UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES, but that value
        // is baked into the precompiled-firmware ELF — TT_METAL_DISABLE_PRECOMPILED_FW=1
        // alone isn't sufficient if the kernel JIT cache holds a stale firmware
        // hash. Writing the override here from the host bypasses all caching
        // and gives the operator a runtime knob without rebuilds. 1 ms at
        // 1 GHz keeps total sample rate ~128k/sec/chip — well below the
        // 200 Hz × 62 × 64 = ~793k/sec drain budget.
        constexpr uint32_t kPeriodOverrideCycles = 1'000'000u;  // 1 ms @ 1 GHz
        constexpr uint64_t kPeriodAddr =
            static_cast<uint64_t>(MEM_UTIL_SAMPLER_BASE) + 12;  // offset of period_cycles field
        int period_set = 0;
        for (auto& c : chip.cores) {
            try {
                c.device->write_to_device(&kPeriodOverrideCycles, c.translated, kPeriodAddr, sizeof(uint32_t));
                ++period_set;
            } catch (...) {
                // remote-chip ETH-tunnel writes can fail silently — leave
                // those cores at the firmware default; the L1 read on TRISC1
                // still works, just with whatever brisc init() seeded.
            }
        }
        std::cerr << "ttnvtop-collector: chip " << chip.chip_id << " set period_cycles=" << kPeriodOverrideCycles
                  << " on " << period_set << "/" << chip.cores.size() << " cores.\n";

        // Verify the arm actually took effect: read FPU_OUT_L twice on the
        // first worker core. If it advances, the counter is genuinely running;
        // if it stays at zero, the arm write hit a dead end (likely: remote
        // chips routed through ETH tunnel, which may not honor writes into
        // RISCV_DEBUG_REG MMIO space).
        if (!chip.cores.empty()) {
            auto* dev = chip.cores[0].device;
            auto coord = chip.cores[0].translated;
            uint32_t l0 = 0, l1 = 0;
            try {
                dev->read_from_device(&l0, coord, kAddrFpuOutL, sizeof(l0));
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                dev->read_from_device(&l1, coord, kAddrFpuOutL, sizeof(l1));
            } catch (const std::exception&) {
            }
            const bool ticking = (l1 != l0);
            std::cerr << "ttnvtop-collector: chip " << chip.chip_id << " FPU_OUT_L probe @ core(" << chip.cores[0].noc_x
                      << "," << chip.cores[0].noc_y << "): " << l0 << " -> " << l1 << " ("
                      << (ticking ? "counter RUNNING" : "counter STUCK — arm likely did not land") << ")\n";
        }

        chips.push_back(std::move(chip));
    }

    // PIN THE TUNNEL to the channels that actually reach each remote chip.
    //
    // UMD defaults remote transfers to EVERY active eth channel on the MMIO chip -- six
    // on a T3K, of which only the internal pair reaches that chip's own remote ASIC; the
    // rest go to a QSFP cage or the Warp bridge, i.e. other boards. It round-robins
    // across the whole set and `wait_for_non_mmio_flush` waits for all of them (5h).
    //
    // It was recorded as impossible for this process to fix, because
    // `Cluster::configure_active_ethernet_cores_for_mmio_device` is a `umd::Cluster`
    // method and a monitoring process never builds a Cluster (that is how it avoids
    // CHIP_IN_USE). That was wrong: the Cluster method is only a wrapper, and everything
    // it wraps is public and reachable from a bare TTDevice --
    // `TTDevice::get_remote_communication()` plus
    // `RemoteCommunication::set_remote_transfer_ethernet_cores()`.
    //
    // Derived, never hardcoded: the channel pair comes from the cluster descriptor.
    if (cli.pin_tunnel) {
        int pinned_chips = 0;
        for (auto& chip : chips) {
            if (!chip.is_remote || chip.cores.empty()) {
                continue;
            }
            try {
                const tt::ChipId mmio_id = cluster_desc->get_closest_mmio_capable_chip(chip.chip_id);
                std::set<uint32_t> local_channels;
                for (const auto& [local_ch, remote_ch] :
                     cluster_desc->get_directly_connected_ethernet_channels_between_chips(mmio_id, chip.chip_id)) {
                    local_channels.insert(static_cast<uint32_t>(local_ch));
                }
                if (local_channels.empty()) {
                    continue;
                }
                // The xy pairs are on the MMIO chip -- that is where the transfer is
                // issued from -- so they must come from the MMIO chip's descriptor.
                TTDevice* mmio_dev = nullptr;
                for (auto& c : chips) {
                    if (c.chip_id == mmio_id && !c.cores.empty()) {
                        mmio_dev = c.cores.front().device;
                    }
                }
                if (mmio_dev == nullptr) {
                    continue;
                }
                SocDescriptor mmio_soc(
                    std::make_shared<SocArchDescriptor>(tt::ARCH::WORMHOLE_B0), mmio_dev->get_chip_info());
                const auto xy = mmio_soc.get_eth_xy_pairs_for_channels(local_channels, CoordSystem::TRANSLATED);
                auto* rc = chip.cores.front().device->get_remote_communication();
                if (rc == nullptr) {
                    continue;
                }
                rc->set_remote_transfer_ethernet_cores(xy);

                // ALSO pin the MMIO chip's OWN RemoteCommunication.
                //
                // `Cluster::configure_active_ethernet_cores_for_mmio_device` sets TWO
                // objects, and the first attempt here set only the first: each remote
                // chip's RemoteCommunication, AND
                // `remote_communications_.at(mmio_chip)` -- the MMIO chip's own, which
                // the Cluster comment says "local chips hold communication primitives
                // for broadcasting". Pinning one side only left the two disagreeing
                // about which channels are in play, and the measured result was
                // incoherent: one remote chip's drain returned zero entries and another
                // lost its aggregator launch, with overall loss slightly WORSE than
                // unpinned. Set both.
                if (auto* mmio_rc = mmio_dev->get_remote_communication(); mmio_rc != nullptr) {
                    mmio_rc->set_remote_transfer_ethernet_cores(xy);
                }
                ++pinned_chips;
                std::cout << "pinned chip " << chip.chip_id << " (via mmio " << mmio_id << ") to channels";
                for (uint32_t ch : local_channels) {
                    std::cout << " " << ch;
                }
                std::cout << "\n";
            } catch (const std::exception& e) {
                std::cerr << "  chip " << chip.chip_id << ": pin failed: " << e.what() << "\n";
            }
        }
        std::cout << "tunnel pinned on " << pinned_chips << " remote chip(s).\n";
    }

    // Bind each chip to a running aggregator, if there is one.
    //
    // The journal's states[] are indexed exactly as the kernel addresses the grid:
    //   i -> translated (xs[i % nx], ys[i / nx])
    // (eth_aggregator.cpp ring_base_of). Reproducing that mapping is mandatory -- getting
    // it wrong would publish real numbers against the wrong cores, which is worse than
    // publishing nothing.
    if (!cli.journal_probe && cli.fidelity_probe_s == 0 && cli.launch_artifact.empty() && !cli.stop_aggregator &&
        cli.peek.empty() && cli.watchdog_s == 0 && !cli.read_latency_probe) {
        const uint64_t now_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
        for (auto& chip : chips) {
            if (chip.cores.empty()) {
                continue;
            }
            auto* dev = chip.cores.front().device;
            SocDescriptor soc(std::make_shared<SocArchDescriptor>(chip.arch), dev->get_chip_info());
            const auto found = ttnvtop_agg::probe_landings(dev, soc, now_us, chip.arch);
            if (found.empty()) {
                continue;
            }
            const auto& l = found.front();
            std::set<uint32_t> xs, ys;
            for (const auto& c : soc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED)) {
                xs.insert(static_cast<uint32_t>(c.x));
                ys.insert(static_cast<uint32_t>(c.y));
            }
            const std::vector<uint32_t> xv(xs.begin(), xs.end()), yv(ys.begin(), ys.end());
            if (xv.empty() || yv.empty() || l.num_cores != xv.size() * yv.size()) {
                std::cerr << "  chip " << chip.chip_id << ": journal reports " << l.num_cores
                          << " cores but the grid is " << xv.size() << "x" << yv.size() << " — refusing to map\n";
                continue;
            }
            std::unordered_map<uint64_t, int> by_translated;
            for (size_t i = 0; i < chip.cores.size(); ++i) {
                by_translated
                    [(static_cast<uint64_t>(chip.cores[i].translated.x) << 32) |
                     static_cast<uint32_t>(chip.cores[i].translated.y)] = static_cast<int>(i);
            }
            chip.journal_to_core.assign(l.num_cores, -1);
            size_t mapped = 0;
            for (uint32_t i = 0; i < l.num_cores; ++i) {
                const uint64_t key = (static_cast<uint64_t>(xv[i % xv.size()]) << 32) | yv[i / xv.size()];
                auto it = by_translated.find(key);
                if (it != by_translated.end()) {
                    chip.journal_to_core[i] = it->second;
                    ++mapped;
                }
            }
            if (mapped != l.num_cores) {
                std::cerr << "  chip " << chip.chip_id << ": mapped only " << mapped << "/" << l.num_cores
                          << " journal cores — refusing to publish from it\n";
                chip.journal_to_core.clear();
                continue;
            }
            chip.journal_active = true;
            chip.journal_core = l.core;
            chip.journal_num_cores = l.num_cores;
            chip.prev_busy.assign(l.num_cores, 0);
            chip.prev_wall.assign(l.num_cores, 0);
            chip.prev_seq.assign(l.num_cores, 0);
            std::cout << "chip " << chip.chip_id << (chip.is_remote ? " (remote)" : " (mmio)")
                      << ": publishing from the ON-CHIP AGGREGATOR at eth (" << l.core.x << "," << l.core.y << "), "
                      << l.num_cores << " cores — per-core ring drain disabled for this chip\n";
        }
    }

    if (chips.empty()) {
        std::cerr << "ttnvtop-collector: no supported chips to monitor.\n";
        return 1;
    }

    // Launch the aggregator from an emitted artifact. UMD ONLY -- no tt-metal.
    //
    // This is the whole point of the artifact: a monitoring process cannot take
    // CHIP_IN_USE from the workload it is monitoring, so it must not link tt-metal. The
    // artifact was produced once, offline, by a process that could. Here we replay it:
    //
    //   1. write the configured kernel-config image to kernel_config_base
    //   2. overwrite the runtime args in place, computed for THIS chip
    //   3. write the launch message
    //   4. write RUN_MSG_GO
    //
    // Four plain L1 writes. The idle-erisc firmware is already polling
    // go_messages[0].signal, and idle eth is always DISPATCH_MODE_HOST, so no dispatch
    // is involved (3.5). 5j proved step 4 works cross-process; steps 1-3 are the same
    // kind of write.
    if (!cli.launch_artifact.empty()) {
        const std::string dir = cli.launch_artifact;
        uint32_t cfg_base = 0, image_bytes = 0, rta_off = 0, launch_addr = 0, go_addr = 0, eth_l1 = 0;
        uint32_t go_index_addr = 0;
        uint32_t rd_ptr_addr = 0;
        std::vector<uint8_t> launch_bytes, go_bytes, reset_go_bytes;
        {
            std::ifstream f(dir + "/aggregator.desc");
            if (!f.good()) {
                std::cerr << "ttnvtop-collector: cannot read " << dir << "/aggregator.desc\n";
                return 1;
            }
            std::string k;
            while (f >> k) {
                if (k == "kernel_config_base") {
                    f >> cfg_base;
                } else if (k == "image_bytes") {
                    f >> image_bytes;
                } else if (k == "rta_offset") {
                    f >> rta_off;
                } else if (k == "launch_addr") {
                    f >> launch_addr;
                } else if (k == "go_addr") {
                    f >> go_addr;
                } else if (k == "go_msg_index_addr") {
                    f >> go_index_addr;
                } else if (k == "launch_msg_rd_ptr_addr") {
                    f >> rd_ptr_addr;
                } else if (k == "eth_l1_unreserved") {
                    f >> eth_l1;
                } else if (k == "launch_bytes" || k == "go_bytes" || k == "reset_go_bytes") {
                    size_t n = 0;
                    f >> n;
                    std::vector<uint8_t> v(n);
                    for (size_t i = 0; i < n; i++) {
                        unsigned b = 0;
                        f >> b;
                        v[i] = static_cast<uint8_t>(b);
                    }
                    if (k == "launch_bytes") {
                        launch_bytes = std::move(v);
                    } else if (k == "go_bytes") {
                        go_bytes = std::move(v);
                    } else {
                        reset_go_bytes = std::move(v);
                    }
                } else {
                    std::string skip;
                    std::getline(f, skip);
                }
            }
        }
        std::vector<uint8_t> image(image_bytes);
        {
            std::ifstream f(dir + "/aggregator.image", std::ios::binary);
            if (!f.good() || !f.read(reinterpret_cast<char*>(image.data()), image_bytes)) {
                std::cerr << "ttnvtop-collector: cannot read " << dir << "/aggregator.image\n";
                return 1;
            }
        }
        if (cfg_base == 0 || image_bytes == 0 || launch_bytes.empty() || go_bytes.empty()) {
            std::cerr << "ttnvtop-collector: incomplete artifact in " << dir << "\n";
            return 1;
        }

        // NOTE: remote-chip launches are UNRELIABLE here, and knowingly so.
        //
        // The launch writes ~65 KB to the target chip. On a remote chip that crosses the
        // NON_MMIO tunnel, which needs its channels pinned to the ones that actually
        // reach that chip or it times out (5h: 1/6 -> 14/14 with pinning). But
        // configure_active_ethernet_cores_for_mmio_device is a umd::Cluster method, and
        // this process deliberately never constructs a Cluster -- that is how it avoids
        // taking CHIP_IN_USE from the workload it is monitoring.
        //
        // So remote launches are attempted and may fail with an eth-service timeout.
        // Resolving it needs either a UMD API to pin without a Cluster, or launching
        // remote aggregators from the process that owns the device.
        int launched = 0;
        for (auto& chip : chips) {
            if (chip.cores.empty()) {
                continue;
            }
            auto* dev = chip.cores.front().device;
            SocDescriptor soc(std::make_shared<SocArchDescriptor>(chip.arch), dev->get_chip_info());

            // Same cross-product derivation the kernel's addressing assumes.
            ttnvtop::AggGrid grid;
            {
                std::set<uint32_t> xs, ys;
                for (const auto& c : soc.get_cores(CoreType::TENSIX, CoordSystem::TRANSLATED)) {
                    xs.insert(static_cast<uint32_t>(c.x));
                    ys.insert(static_cast<uint32_t>(c.y));
                }
                grid.xs.assign(xs.begin(), xs.end());
                grid.ys.assign(ys.begin(), ys.end());
            }
            if (grid.num_cores() == 0) {
                continue;
            }

            // Pick an INACTIVE eth core. This must never be an active one.
            //
            // An earlier revision took "the first core that reads successfully", which is
            // every core — so it landed on an ACTIVE link-carrying core on all four MMIO
            // chips, overwrote ERISC0, killed the eth firmware that services the NON_MMIO
            // tunnel, and wedged topology discovery machine-wide. Readable is not the
            // same predicate as unused, and the difference costs a board reset.
            //
            // The cluster descriptor knows which channels are active. Exclude them, and
            // prefer the channels no shipped WH descriptor ever routes (2), so a
            // recabling cannot turn our core into a live link underneath us.
            // ALL usable candidates, in preference order -- not just the best one.
            //
            // "No live link" is NOT the same as "nobody is using this core". The cluster
            // descriptor's active-channel set is about LINKS, and tt-metal's FABRIC can
            // place an EDM on a link-free eth core; which cores it takes varies per
            // device-open epoch. Measured on a T3K: within one epoch every repetition
            // gives an identical result, but across epochs remote successes were 4/4,
            // 3/4, 2/4 and 1/4. On a chip where fabric owns our pick, the 25 KB image and
            // the launch message land byte-identical to a working chip and only the go
            // word fails to take -- because the core is not running idle_erisc and nothing
            // is polling go_messages[0].
            //
            // agg_core_select.hpp documents exactly this hazard for the ETH DISPATCH pool.
            // Fabric was not considered. Rather than try to query a control plane this
            // process deliberately does not have, try candidates in order and keep the
            // first that VERIFIES as running -- the verification already exists and is the
            // only ground truth available.
            const auto active = cluster_desc->get_active_eth_channels(chip.chip_id);
            std::vector<std::pair<int, tt::umd::CoreCoord>> candidates;
            for (const auto& eth : soc.get_cores(CoreType::ETH, CoordSystem::TRANSLATED)) {
                const auto logical = soc.translate_coord_to(eth, CoordSystem::LOGICAL);
                const uint32_t channel = static_cast<uint32_t>(logical.y);
                if (active.count(channel) != 0) {
                    continue;  // carries a live link — never ours
                }
                // Channel 15 is forced ACTIVE by the control plane on any WH MMIO chip
                // with tunnels, link or no link (fabric/control_plane.cpp:2255-2272), and
                // carries UMD's base routing for the NON_MMIO tunnel. Never take it.
                if (ttnvtop::is_umd_base_routing_channel(channel)) {
                    continue;
                }
                candidates.emplace_back(ttnvtop::aggregator_channel_rank(channel), eth);
            }
            std::stable_sort(
                candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            const bool found = !candidates.empty();
            tt::umd::CoreCoord target = found ? candidates.front().second : tt::umd::CoreCoord{};
            if (!found) {
                std::cerr << "  chip " << chip.chip_id
                          << ": every ethernet core is active — refusing to displace a live link\n";
                continue;
            }

            // Try candidates in preference order; keep the first that VERIFIES.
            bool chip_ok = false;
            for (size_t cand = 0; cand < candidates.size() && !chip_ok; ++cand) {
                target = candidates[cand].second;

                // PRE-FLIGHT: prove the core is free BEFORE writing a single byte.
                //
                // The previous version wrote its 25 KB image into candidate after
                // candidate and inferred ownership from whether the kernel started. That
                // is structurally unsound -- the image overwrites whatever kernel is
                // resident, and on chip 4 it walked EIGHT cores, clobbering fabric and
                // dispatch kernels and killing the workload with a corrupted completion
                // queue. Verification cannot come after the destructive act.
                //
                // Three conditions, ALL required. Together they are race-free against
                // anything already resident:
                //
                //  a) heartbeat at 0x1C has signature 0xAABB and ADVANCES across two
                //     samples. This is the strong one, because an advancing counter can
                //     only come from live code -- no staleness is possible. 0xAABB is
                //     FABRIC_HEARTBEAT_SIGNATURE, which is what tt-metal's idle_erisc
                //     posts from its wait-for-GO loop (idle_erisc.cc:68-76); 0xABCD is
                //     BASE_FW_HEARTBEAT_SIGNATURE, meaning UMD/syseng eth base FW owns the
                //     core and it was never handed to tt-metal. Frozen either way means
                //     wedged. Only "0xAABB advancing" means tt-metal firmware is sitting
                //     there ready to accept a go signal -- which is also exactly the
                //     residency precondition a launch needs.
                //  b) routing_enabled == 0 at 0x3DC00. Written to 1 for precisely the
                //     control-plane ACTIVE set, INCLUDING forced channel 15
                //     (llrt/tt_cluster.cpp:1499-1522).
                //  c) go_messages[0].signal != RUN_MSG_GO. A persistent kernel leaves GO
                //     standing for its whole life, since idle_erisc.cc only writes
                //     RUN_MSG_DONE after the kernel returns.
                //
                // (c) alone is NOT sufficient and was the bug in the first attempt at this
                // check: tt-metal never writes the IDLE_ETH go word on a core it classified
                // ACTIVE_ETH (risc_firmware_initializer.cpp:340-372), so on those cores it
                // reads stale L1. (a) is what closes that hole.
                try {
                    uint32_t routing_enabled = 0;
                    dev->read_from_device(
                        &routing_enabled, target, ttnvtop_agg::kEthRoutingInfoAddr, sizeof(routing_enabled));
                    if (routing_enabled != 0) {
                        std::cout << "  chip " << chip.chip_id << " eth (" << target.x << "," << target.y
                                  << "): routing_enabled=1 — control-plane ACTIVE core, skipping\n";
                        continue;
                    }
                    uint32_t hb0 = 0, hb1 = 0;
                    dev->read_from_device(&hb0, target, ttnvtop::kWormholeEthHeartbeatAddr, sizeof(hb0));
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    dev->read_from_device(&hb1, target, ttnvtop::kWormholeEthHeartbeatAddr, sizeof(hb1));
                    const uint32_t sig = hb1 >> 16;
                    if (sig != 0xAABBu || hb0 == hb1) {
                        std::cout << "  chip " << chip.chip_id << " eth (" << target.x << "," << target.y
                                  << "): heartbeat 0x" << std::hex << hb0 << " -> 0x" << hb1 << std::dec
                                  << (sig == 0xABCDu ? " (base FW owns it)"
                                                     : (hb0 == hb1 ? " (frozen)" : " (unknown signature)"))
                                  << " — skipping\n";
                        continue;
                    }
                    uint32_t go_now = 0;
                    dev->read_from_device(&go_now, target, go_addr, sizeof(go_now));
                    if (((go_now >> 24) & 0xFFu) == tt::tt_metal::dev_msgs::RUN_MSG_GO) {
                        std::cout << "  chip " << chip.chip_id << " eth (" << target.x << "," << target.y
                                  << "): RUN_MSG_GO — a kernel is running here, skipping\n";
                        continue;
                    }
                } catch (const std::exception&) {
                    continue;  // cannot read it, so cannot establish it is free
                }
                const auto l1 = ttnvtop::agg_layout(eth_l1, grid.num_cores());
                const auto rt = ttnvtop::agg_rt_args(
                    grid,
                    l1,
                    static_cast<uint32_t>(chip.chip_id),
                    1000000u,
                    16u,
                    chip.arch == tt::ARCH::WORMHOLE_B0 ? ttnvtop::kWormholeEthHeartbeatAddr : 0u);
                try {
                    // Reset the go signal FIRST, exactly as LaunchProgram does via
                    // send_reset_go_signal(). Omitting this was why the replay never started
                    // the kernel: the firmware reads mailboxes->launch[launch_msg_rd_ptr],
                    // and without RUN_MSG_RESET_READ_PTR_FROM_HOST plus a zeroed
                    // GO_MSG_INDEX it can keep a stale read pointer and consume a different
                    // launch slot than the one we write.
                    if (!reset_go_bytes.empty()) {
                        dev->write_to_device(reset_go_bytes.data(), target, go_addr, reset_go_bytes.size());
                    }
                    if (go_index_addr != 0) {
                        const uint32_t zero = 0;
                        dev->write_to_device(&zero, target, go_index_addr, sizeof(zero));
                    }

                    // ZERO launch_msg_rd_ptr, exactly as device init does
                    // (risc_firmware_initializer.cpp writes 0 here).
                    //
                    // The launcher always writes launch slot 0, so the firmware must be reading
                    // slot 0. It is not guaranteed to be: idle_erisc.cc advances this pointer
                    // whenever a kernel returns under DISPATCH_MODE_DEV, and
                    // RUN_MSG_RESET_READ_PTR_FROM_HOST does NOT reset it -- that firmware has no
                    // handler for it at all, unlike brisc/active_erisc. So the go-signal reset
                    // cannot be relied on and the pointer is zeroed explicitly. Without this,
                    // the first launch works and every one after it silently starts nothing.
                    if (rd_ptr_addr != 0) {
                        const uint32_t zero = 0;
                        dev->write_to_device(&zero, target, rd_ptr_addr, sizeof(zero));
                    }
                    dev->write_to_device(image.data(), target, cfg_base, image_bytes);
                    dev->write_to_device(rt.data(), target, cfg_base + rta_off, rt.size() * 4);

                    // Zero the journal header before the go word.
                    //
                    // A journal left by an earlier aggregator keeps a valid magic and a valid
                    // header checksum indefinitely, and its sweep_count simply stops. Verify
                    // against that and a dead core reads as a healthy one -- it has already
                    // produced two near-false-positives. Worse, a RESTARTED aggregator
                    // counts from zero, so "did the count go up" is wrong in both
                    // directions unless the baseline is known. Zeroing it makes the baseline
                    // zero, which is the only baseline that cannot lie.
                    const std::vector<uint8_t> hdr_zero(sizeof(util_agg_msg_t), 0);
                    dev->write_to_device(hdr_zero.data(), target, l1.journal, hdr_zero.size());

                    // READ IT BACK before releasing the go word.
                    //
                    // On a REMOTE chip these two writes are not ordered against each other:
                    // the zeroing goes over the NON_MMIO tunnel and the go word follows, and
                    // measured on T3K chip 7 the zeroing landed AFTER the kernel had already
                    // stamped its magic -- leaving a live aggregator publishing an advancing
                    // sweep_count under `magic 0x0`. `probe_landings` gates on the magic, so
                    // the launcher called a running kernel NOT RUNNING and --stop-aggregator
                    // could not reach it either. The tell was sweeps ADVANCING while the magic
                    // read zero.
                    //
                    // A read of the same address forces the write to have landed, which closes
                    // the window instead of relying on the kernel to republish over it.
                    for (int rb = 0; rb < 20; ++rb) {
                        std::vector<uint8_t> chk(sizeof(util_agg_msg_t), 0xFF);
                        dev->read_from_device(chk.data(), target, l1.journal, chk.size());
                        if (std::all_of(chk.begin(), chk.end(), [](uint8_t b) { return b == 0; })) {
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    }

                    dev->write_to_device(launch_bytes.data(), target, launch_addr, launch_bytes.size());

                    // WRITE THE GO WORD, THEN VERIFY IT, AND RETRY.
                    //
                    // On a remote chip this last 4-byte write is the one that gets lost.
                    // Measured on T3K chip 6, deterministically, every attempt: the 25 KB
                    // kernel image and the 144 B launch message both land byte-identical to a
                    // chip that works (same 0x00007df0 config base, same 0xf4010113 first
                    // instruction) -- and the go word reads back 0x00000000 instead of
                    // 0x80000000. Everything arrives except the one write that starts the
                    // kernel, so the launcher reported a correct-looking failure.
                    //
                    // Read-back-and-retry rather than a blind second write: a blind retry
                    // cannot tell "landed" from "lost", and writing GO twice to a core that
                    // already started is not free.
                    bool go_landed = false;
                    for (int attempt = 0; attempt < 8 && !go_landed; ++attempt) {
                        dev->write_to_device(go_bytes.data(), target, go_addr, go_bytes.size());
                        std::vector<uint8_t> back(go_bytes.size(), 0);
                        dev->read_from_device(back.data(), target, go_addr, back.size());
                        go_landed = std::equal(go_bytes.begin(), go_bytes.end(), back.begin());
                        if (!go_landed) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }
                    }
                    if (!go_landed) {
                        std::cout << " — go word never landed after 8 attempts";
                    }
                } catch (const std::exception& e) {
                    std::cerr << "  chip " << chip.chip_id << ": launch failed: " << e.what() << "\n";
                    continue;
                }
                std::cout << "  chip " << chip.chip_id << (chip.is_remote ? " (remote)" : " (mmio)") << " eth ("
                          << target.x << "," << target.y << ") " << grid.xs.size() << "x" << grid.ys.size() << " = "
                          << grid.num_cores() << " cores, journal 0x" << std::hex << l1.journal << std::dec;

                // VERIFY, do not assume. The four writes succeeding says only that L1
                // accepted them; it says nothing about whether anything on the core is
                // polling go_messages[0]. This launcher spent a day reporting success on all
                // eight chips while starting nothing, because "the writes returned" was
                // treated as "the kernel is running".
                //
                // Liveness is an ADVANCING sweep_count from a ZEROED baseline, never a valid
                // magic. The header was zeroed above, so a live kernel republishes the magic
                // and counts up from zero, and anything that does not is not running.
                bool running = false;
                uint32_t s0 = 0, s1 = 0;
                uint32_t magic = 0;
                try {
                    std::vector<uint8_t> buf(sizeof(util_agg_msg_t), 0);
                    util_agg_hdr_view_t h{};
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    dev->read_from_device(buf.data(), target, l1.journal, sizeof(util_agg_hdr_view_t));
                    std::memcpy(&h, buf.data(), sizeof(h));
                    s0 = h.sweep_count;
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    dev->read_from_device(buf.data(), target, l1.journal, sizeof(util_agg_hdr_view_t));
                    std::memcpy(&h, buf.data(), sizeof(h));
                    s1 = h.sweep_count;
                    magic = h.magic;
                    running = magic == UTIL_AGG_MAGIC && s1 > s0;
                } catch (const std::exception& e) {
                    std::cout << " — verify read failed: " << e.what() << "\n";
                    continue;
                }
                if (running) {
                    ++launched;
                    chip_ok = true;
                    std::cout << " — RUNNING (sweeps " << s0 << " -> " << s1 << ")\n";
                } else {
                    std::cout << " — NOT RUNNING (magic 0x" << std::hex << magic << std::dec << ", sweeps " << s0
                              << " -> " << s1 << ")";

                    // NEUTRALISE AN ABANDONED CANDIDATE BEFORE MOVING ON.
                    //
                    // By this point the 25 KB image, the runtime args, the launch message
                    // AND the go word have all been written to this core. If the kernel
                    // merely started LATE -- past the verification window -- walking away
                    // leaves a live aggregator on a core nothing tracks, which no
                    // --stop-aggregator will ever find and whose 0xABCD heartbeat then
                    // freezes when it dies. That is the likely source of the stray
                    // "Stuck at 0xabcd9a81" seen on core e4-0, which is NOT the core the
                    // launcher settles on: an abandoned fallback candidate.
                    //
                    // Disarm it: take the go word back and clear the journal magic so no
                    // later probe reads a corpse as live.
                    try {
                        const std::vector<uint8_t> go_off(go_bytes.size(), 0);
                        dev->write_to_device(go_off.data(), target, go_addr, go_off.size());
                        const uint32_t jz = 0;
                        dev->write_to_device(&jz, target, l1.journal, sizeof(jz));
                        std::cout << ", disarmed";
                    } catch (const std::exception&) {
                        std::cout << ", DISARM FAILED";
                    }
                    std::cout << ((cand + 1 < candidates.size()) ? " — trying next core\n" : "\n");
                }
            }
            if (!chip_ok) {
                std::cerr << "  chip " << chip.chip_id << ": no idle ethernet core accepted the launch ("
                          << candidates.size() << " tried). Fabric may own them all this epoch.\n";
            }
        }
        std::cout << "launched " << launched << " aggregator(s) — no tt-metal, no dispatch, no fabric.\n";
        return launched > 0 ? 0 : 1;
    }

    // Remote-read latency vs transfer size.
    //
    // The host-pull design's drain rate is bounded by how long a remote read takes, not
    // by how many transactions it issues (5l: ~250 ms per read while Llama runs, giving
    // 0.5 Hz). Whether aggregating on-chip helps depends entirely on whether that
    // latency is FIXED OVERHEAD -- in which case shrinking the payload buys nothing --
    // or SIZE-PROPORTIONAL, in which case a 2 KB state table drains comfortably.
    if (cli.read_latency_probe) {
        static const size_t kSizes[] = {64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        std::vector<uint8_t> buf(131072);
        std::cout << "remote-read latency probe (read-only), base 0x" << std::hex
                  << ttnvtop_agg::landing_base(chips.front().arch) << std::dec << "\n";
        int probed = 0;
        for (auto& chip : chips) {
            if (!chip.is_remote || chip.cores.empty() || probed >= 1) {
                continue;  // one remote chip is enough to characterise the tunnel
            }
            ++probed;
            auto* dev = chip.cores.front().device;
            const auto core = chip.cores.front().translated;
            std::cout << "  chip " << chip.chip_id << " (remote):" << std::flush;
            for (size_t sz : kSizes) {
                // Three reads, report the median, so one outlier does not set the number.
                double ms[3] = {0, 0, 0};
                bool ok = true;
                for (int r = 0; r < 3 && ok; r++) {
                    const auto t0 = std::chrono::steady_clock::now();
                    try {
                        dev->read_from_device(buf.data(), core, ttnvtop_agg::landing_base(chip.arch), sz);
                    } catch (...) {
                        ok = false;
                        break;
                    }
                    ms[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                }
                if (!ok) {
                    std::cout << "  " << sz << "B=ERR";
                    continue;
                }
                std::sort(ms, ms + 3);
                std::cout << "  " << sz << "B=" << std::fixed << std::setprecision(1) << ms[1] << "ms" << std::flush;
            }
            std::cout << "\n";

            // THE DRAIN'S ACTUAL ACCESS PATTERN, timed.
            //
            // The size sweep above reads one core repeatedly, which is not what the drain
            // does. The drain reads 1 KiB from EACH of 64 different Tensix cores per pass
            // -- 64 separate `read_non_mmio` calls, each acquiring the interprocess
            // NON_MMIO mutex and each issuing its own eth command. This times that exact
            // shape so the per-pass cost is measured rather than inferred from a
            // per-transfer number (a mistake made twice already in this work: never
            // divide a loop time by a transaction count, and never multiply a
            // single-transfer time up to a loop).
            //
            // Compared against the same 64 KiB pulled from ONE core in a single call,
            // which UMD chunks into 64 x MAX_BLOCK_SIZE(1024 B) blocks under ONE mutex
            // acquisition. Same bytes, same block count; the delta is per-call overhead.
            {
                const uint32_t kNCores = 64, kChunk = 1024;
                // chip.cores already holds every Tensix core in translated coords -- the
                // same list and the same order the drain walks.
                std::vector<tt_xy_pair> tensix;
                for (const auto& c : chip.cores) {
                    tensix.push_back(c.translated);
                    if (tensix.size() >= kNCores) {
                        break;
                    }
                }
                double spread = 1e9, same = 1e9, single = 1e9;
                bool ok = !tensix.empty();
                for (int r = 0; r < 3 && ok; r++) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (const auto& tc : tensix) {
                        try {
                            dev->read_from_device(buf.data(), tc, static_cast<uint64_t>(MEM_UTIL_SAMPLER_BASE), kChunk);
                        } catch (...) {
                            ok = false;
                            break;
                        }
                    }
                    spread = std::min(
                        spread,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                for (int r = 0; r < 3 && ok; r++) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (uint32_t i = 0; i < kNCores; i++) {
                        try {
                            dev->read_from_device(buf.data(), core, ttnvtop_agg::landing_base(chip.arch), kChunk);
                        } catch (...) {
                            ok = false;
                            break;
                        }
                    }
                    same = std::min(
                        same, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                for (int r = 0; r < 3 && ok; r++) {
                    auto t0 = std::chrono::steady_clock::now();
                    try {
                        dev->read_from_device(buf.data(), core, ttnvtop_agg::landing_base(chip.arch), kChunk * kNCores);
                    } catch (...) {
                        ok = false;
                        break;
                    }
                    single = std::min(
                        single,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                if (ok) {
                    std::cout << std::setprecision(2) << "  drain pattern (" << tensix.size()
                              << " cores x 1KiB): " << spread << " ms/pass -> " << (spread > 0 ? 1000.0 / spread : 0.0)
                              << " passes/s   |  same-core x64: " << same << " ms  |  one 64KiB call: " << single
                              << " ms  |  per-call overhead: " << ((same - single) / (kNCores - 1) * 1000.0) << " us\n";
                }
            }
        }
        return 0;
    }

    // Phase 2.2, open question 7.4. Start an aggregator that ANOTHER PROCESS staged.
    //
    // That process wrote the kernel binary, its runtime args (including the fabric
    // connection spec) and the launch message, but deliberately withheld the go word.
    // All that is left is one L1 write, done here over raw UMD with no tt-metal in the
    // picture -- which is the point: the device-side launch contract is plain L1 writes
    // (3.5), and the only reason the aggregator currently needs tt-metal is JIT-building
    // the ELF and computing the fabric connection args, neither of which is a device
    // requirement.
    if (!cli.launch_go.empty()) {
        std::ifstream f(cli.launch_go);
        if (!f.good()) {
            std::cerr << "ttnvtop-collector: cannot read " << cli.launch_go << "\n";
            return 1;
        }
        int chip = -1, ex = -1, ey = -1;
        uint64_t go_addr = 0;
        std::vector<uint8_t> go_bytes;
        std::string key;
        while (f >> key) {
            if (key == "chip") {
                f >> chip;
            } else if (key == "eth_translated") {
                f >> ex >> ey;
            } else if (key == "go_addr") {
                f >> go_addr;
            } else if (key == "go_bytes") {
                std::string rest;
                std::getline(f, rest);
                std::istringstream is(rest);
                unsigned b = 0;
                while (is >> b) {
                    go_bytes.push_back(static_cast<uint8_t>(b));
                }
            }
        }
        if (chip < 0 || ex < 0 || ey < 0 || go_addr == 0 || go_bytes.empty()) {
            std::cerr << "ttnvtop-collector: malformed descriptor " << cli.launch_go << "\n";
            return 1;
        }
        ChipState* target = nullptr;
        for (auto& c : chips) {
            if (static_cast<int>(c.chip_id) == chip) {
                target = &c;
            }
        }
        if (target == nullptr || target->cores.empty()) {
            std::cerr << "ttnvtop-collector: chip " << chip << " not present\n";
            return 1;
        }
        const tt::umd::CoreCoord eth(
            static_cast<size_t>(ex), static_cast<size_t>(ey), CoreType::ETH, CoordSystem::TRANSLATED);
        std::cout << "launch-go: chip " << chip << " eth (" << ex << "," << ey << ") addr 0x" << std::hex << go_addr
                  << std::dec << " " << go_bytes.size() << " bytes\n";
        try {
            target->cores.front().device->write_to_device(go_bytes.data(), eth, go_addr, go_bytes.size());
        } catch (const std::exception& e) {
            std::cerr << "launch-go: write failed: " << e.what() << "\n";
            return 1;
        }
        std::cout << "launch-go: go word written — the staged aggregator should now be running.\n";
        return 0;
    }

    // WATCHDOG: stop a dead aggregator from hard-blocking the next device open.
    //
    // THE HAZARD. Our kernel maintains the eth firmware heartbeat it displaced, writing
    // (0xABCD << 16) | counter. A CLEAN cooperative stop is already safe: the kernel
    // returns, idle_erisc.cc regains its wait loop and resumes posting its own heartbeat.
    // But if the kernel dies WITHOUT returning -- it hangs, or the owning process closes
    // the device under it -- the word is left FROZEN with a VALID signature, and UMD's
    // eth_heartbeat_running THROWS on exactly that. The next tt-metal device open then
    // fails outright:
    //
    //     RuntimeError: Timed out waiting for ETH heartbeat ... Stuck at 0xabcd9a81
    //
    // Observed for real. A monitor that can prevent the next job from starting is not
    // shippable, and no in-kernel fix can cover it -- a dead kernel cannot write.
    //
    // THE FIX, from reading UMD's predicate rather than guessing. Two loops:
    //   loop 1 waits for the word to be NON-ZERO; still zero at timeout -> error. So
    //          "clearing" the heartbeat to 0 makes things WORSE, not better.
    //   loop 2 if (value >> 16) is neither 0xABCD nor 0xAABB it logs "FW possibly
    //          corrupted" and returns false IMMEDIATELY, with no throw.
    // So an INVALID signature is a warning; a frozen VALID one is fatal. Writing a
    // non-zero word with a deliberately invalid signature converts a hard block into a
    // skipped core, which for an INACTIVE eth core costs nothing.
    if (cli.watchdog_s > 0) {
        constexpr uint32_t kDeadSignature = 0xDEADu;  // deliberately NOT 0xABCD or 0xAABB
        constexpr int kStallPolls = 5;                // ~2.5 s at 2 Hz before declaring death
        struct Watched {
            tt::ChipId chip_id = 0;
            tt::umd::CoreCoord core{};
            TTDevice* dev = nullptr;
            uint32_t hb_addr = 0;
            uint32_t last_sweeps = 0;
            int stalled = 0;
            bool neutralised = false;
        };
        const uint64_t t_now_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
        std::vector<Watched> watched;
        for (auto& chip : chips) {
            if (chip.cores.empty() || chip.arch != tt::ARCH::WORMHOLE_B0) {
                continue;  // Blackhole's idle_erisc posts no heartbeat and its discovery does not check one
            }
            auto* dev = chip.cores.front().device;
            SocDescriptor soc(std::make_shared<SocArchDescriptor>(chip.arch), dev->get_chip_info());
            for (const auto& l : ttnvtop_agg::probe_landings(dev, soc, t_now_us, chip.arch)) {
                Watched w;
                w.chip_id = chip.chip_id;
                w.core = l.core;
                w.dev = dev;
                w.hb_addr = ttnvtop::kWormholeEthHeartbeatAddr;
                w.last_sweeps = l.last_sweep_count;
                watched.push_back(w);
            }
        }
        std::cout << "watchdog: " << watched.size() << " aggregator(s), " << cli.watchdog_s << " s at 2 Hz\n";
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(cli.watchdog_s);
        int neutralised = 0;
        while (std::chrono::steady_clock::now() < deadline && !g_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            for (auto& w : watched) {
                if (w.neutralised) {
                    continue;
                }
                uint32_t sweeps = 0;
                bool ok = false;
                try {
                    std::vector<uint8_t> buf(sizeof(util_agg_msg_t), 0);
                    w.dev->read_from_device(
                        buf.data(), w.core, ttnvtop_agg::landing_base(tt::ARCH::WORMHOLE_B0), sizeof(buf[0]) * 64);
                    util_agg_hdr_view_t h{};
                    std::memcpy(&h, buf.data(), sizeof(h));
                    ok = h.magic == UTIL_AGG_MAGIC;
                    sweeps = h.sweep_count;
                } catch (const std::exception&) {
                    ok = false;
                }
                if (!ok) {
                    w.stalled = 0;  // no journal -> stopped cleanly, nothing to neutralise
                    continue;
                }
                w.stalled = (sweeps == w.last_sweeps) ? w.stalled + 1 : 0;
                w.last_sweeps = sweeps;
                if (w.stalled < kStallPolls) {
                    continue;
                }
                // Dead with a live-looking journal. Neutralise the heartbeat and clear the
                // magic so no later reader mistakes the corpse for a running aggregator.
                try {
                    const uint32_t dead = (kDeadSignature << 16) | (sweeps & 0xFFFFu);
                    w.dev->write_to_device(&dead, w.core, w.hb_addr, sizeof(dead));
                    const uint32_t zero = 0;
                    w.dev->write_to_device(
                        &zero, w.core, ttnvtop_agg::landing_base(tt::ARCH::WORMHOLE_B0), sizeof(zero));
                    w.neutralised = true;
                    ++neutralised;
                    std::cout << "  chip " << w.chip_id << " eth (" << w.core.x << "," << w.core.y
                              << "): aggregator DEAD at sweeps=" << sweeps << " — heartbeat neutralised to 0x"
                              << std::hex << dead << std::dec << " (invalid signature: UMD warns, does not throw)\n";
                } catch (const std::exception& e) {
                    std::cout << "  chip " << w.chip_id << ": neutralise failed: " << e.what() << "\n";
                }
            }
        }
        std::cout << "watchdog: neutralised " << neutralised << " dead aggregator(s).\n";
        return 0;
    }

    // Raw L1 peek: --peek chip,x,y,addr[,len]   (TRANSLATED eth coords; addr hex or decimal)
    //
    // Most dead ends in this work were "what does the device actually hold there", and each
    // needed a throwaway probe. Reads TWICE 300 ms apart and flags changed words, because
    // for a heartbeat the only question that matters is whether it is ADVANCING -- a static
    // word with a valid signature is exactly the failure mode that costs a board reset.
    if (!cli.peek.empty()) {
        long long pc = -1, px = -1, py = -1, plen = 32;
        unsigned long long paddr = 0;
        {
            std::string t = cli.peek;
            std::replace(t.begin(), t.end(), ',', ' ');
            std::istringstream is(t);
            std::string astr;
            is >> pc >> px >> py >> astr;
            if (!(is >> plen)) {
                plen = 32;
            }
            try {
                paddr = astr.rfind("0x", 0) == 0 ? std::stoull(astr, nullptr, 16) : std::stoull(astr);
            } catch (...) {
                std::cerr << "ttnvtop-collector: --peek could not parse address '" << astr << "'\n";
                return 1;
            }
        }
        if (pc < 0 || px < 0 || py < 0) {
            std::cerr << "ttnvtop-collector: --peek expects chip,x,y,addr[,len]\n";
            return 1;
        }
        ChipState* ptarget = nullptr;
        for (auto& c : chips) {
            if (static_cast<long long>(c.chip_id) == pc) {
                ptarget = &c;
            }
        }
        if (ptarget == nullptr || ptarget->cores.empty()) {
            std::cerr << "ttnvtop-collector: chip " << pc << " not present\n";
            return 1;
        }
        plen = std::min<long long>(std::max<long long>(plen, 4), 512);
        plen = (plen + 3) & ~3LL;
        const tt::umd::CoreCoord pcore(
            static_cast<size_t>(px), static_cast<size_t>(py), CoreType::ETH, CoordSystem::TRANSLATED);
        std::vector<uint8_t> pa(plen, 0), pb(plen, 0);
        try {
            ptarget->cores.front().device->read_from_device(pa.data(), pcore, paddr, plen);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            ptarget->cores.front().device->read_from_device(pb.data(), pcore, paddr, plen);
        } catch (const std::exception& e) {
            std::cerr << "peek: read failed: " << e.what() << "\n";
            return 1;
        }
        std::cout << "peek chip " << pc << " eth (" << px << "," << py << ") 0x" << std::hex << paddr << std::dec
                  << " len " << plen << "   (two reads, 300 ms apart)\n";
        for (long long o = 0; o < plen; o += 4) {
            uint32_t wa = 0, wb = 0;
            std::memcpy(&wa, pa.data() + o, 4);
            std::memcpy(&wb, pb.data() + o, 4);
            std::cout << "  +0x" << std::hex << std::setw(3) << std::setfill('0') << o << "  0x" << std::setw(8) << wa
                      << "  0x" << std::setw(8) << wb << std::setfill(' ') << std::dec
                      << (wa != wb ? "   <- ADVANCING" : "") << "\n";
        }
        return 0;
    }

    // Ask every running aggregator to RETURN, and confirm it did.
    //
    // This exists so an experiment on a Wormhole T3K is reversible. The aggregator's
    // other exit -- the RISC reset `stop_aggregator()` asserts -- leaves the core with no
    // firmware AND leaves our 0xABCD heartbeat word frozen with a valid signature, which
    // tt-metal turns into a hard error on the next device open ("Stuck at 0xabcd....")
    // and a board reset to clear. Returning instead hands the core back to
    // idle_erisc.cc, which resumes its own heartbeat while it waits for the next
    // RUN_MSG_GO, so the core stays discoverable and relaunchable.
    //
    // MUST be called while the tt-metal process that started the aggregator still holds
    // the device. Once it closes, the aggregator is already gone and the word is already
    // frozen.
    if (cli.stop_aggregator) {
        const uint64_t now_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
        int asked = 0, confirmed = 0;
        for (auto& chip : chips) {
            if (chip.cores.empty()) {
                continue;
            }
            auto* dev = chip.cores.front().device;
            SocDescriptor soc(std::make_shared<SocArchDescriptor>(chip.arch), dev->get_chip_info());
            for (const auto& l : ttnvtop_agg::probe_landings(dev, soc, now_us, chip.arch)) {
                const auto l1 =
                    ttnvtop::agg_layout(static_cast<uint32_t>(ttnvtop_agg::landing_base(chip.arch)), l.num_cores);
                std::cout << "  chip " << chip.chip_id << " eth (" << l.core.x << "," << l.core.y
                          << ") sweeps=" << l.last_sweep_count << std::flush;
                try {
                    // dbg[3]. See kStopRequest in eth_aggregator.cpp for why the request
                    // rides here and not in a runtime argument.
                    const uint32_t req = 0x504F5453u;  // 'STOP'
                    dev->write_to_device(&req, l.core, l1.dbg + 12u, sizeof(req));
                } catch (const std::exception& e) {
                    std::cout << " — write failed: " << e.what() << "\n";
                    continue;
                }
                ++asked;
                // The kernel clears the journal magic on its way out, so "did it stop" is
                // answered by the magic going away -- not by the sweep count freezing,
                // which is exactly what a wedged kernel looks like too.
                bool gone = false;
                for (int attempt = 0; attempt < 20 && !gone; ++attempt) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    try {
                        std::vector<uint8_t> buf(sizeof(util_agg_msg_t), 0);
                        dev->read_from_device(buf.data(), l.core, ttnvtop_agg::landing_base(chip.arch), buf.size());
                        util_agg_hdr_view_t h{};
                        std::memcpy(&h, buf.data(), sizeof(h));
                        gone = h.magic != UTIL_AGG_MAGIC;
                    } catch (const std::exception&) {
                        break;
                    }
                }
                if (gone) {
                    ++confirmed;
                    std::cout << " — STOPPED (journal magic cleared)\n";
                } else {
                    std::cout << " — DID NOT STOP; the core will need a reset\n";
                }
            }
        }
        if (asked == 0) {
            std::cout << "no running aggregator found — nothing to stop.\n";
            return 0;
        }
        std::cout << "stopped " << confirmed << " of " << asked << " aggregator(s).\n";
        return confirmed == asked ? 0 : 1;
    }

    // FIDELITY PROBE -- the aggregator arm of the comparison this whole feature rests on.
    //
    // The original justification for on-chip aggregation was that host monitoring stalls
    // workloads. It did not reproduce: three Llama-70B arms spanned 0.5% with the
    // ordering backwards (5l/5q). The ONLY surviving justification is FIDELITY -- that a
    // host draining 62-entry per-core rings over PCIe and, worse, over the NON_MMIO
    // tunnel cannot keep up with the producers, while a sweep running on-chip can.
    //
    // Both consumers read the same rings and neither consumes, so they can run at the
    // same time against ONE workload. That matters: it removes "were the two arms even
    // producing the same samples" as a variable, which separate runs cannot.
    //
    // This is the aggregator arm: drain the journal at 10 Hz for N seconds and report
    // what the ON-CHIP sweep folded and what it missed. The host arm is the normal
    // collector's own `[ring-drain] entries=/lost=` line over the same window.
    //
    // Read-only. Two small reads per chip per tick -- 0.2 ms for a 2 KB table even over
    // the tunnel (5m).
    if (cli.fidelity_probe_s > 0) {
        struct Tracked {
            tt::ChipId chip_id = 0;
            bool is_remote = false;
            tt::umd::CoreCoord core{};
            uint32_t num_cores = 0;
            uint32_t src_chip = 0;
            TTDevice* dev = nullptr;
            uint32_t journal = 0;
            // First and last observation, so every number below is a DELTA over the
            // window and nothing carries in from before the probe started.
            bool primed = false;
            // `head` is the RAW count of ring entries folded (`head += behind` in the
            // kernel), which is the ONLY quantity comparable to the host drain's
            // `drain_entries_seen`. The per-core `samples` field is NOT: it counts
            // ACCEPTED deltas, so every entry whose FPU counter went backwards lands in
            // `resets` instead, and an implausible wall delta is dropped silently.
            // Comparing `samples` against the host's raw entry count made the aggregator
            // look like it was folding 1.7x fewer samples at lost=0 -- an apples-to-
            // oranges comparison that invalidated the 5s cross-arm agreement claim.
            uint64_t first_head = 0, last_head = 0;
            uint64_t first_samples = 0, last_samples = 0;
            uint64_t first_resets = 0, last_resets = 0;
            uint32_t first_lost = 0, last_lost = 0;
            uint32_t first_sweeps = 0, last_sweeps = 0;
            uint32_t cores_advancing = 0;
            uint32_t torn_reads = 0, failed_reads = 0, ticks = 0;
        };

        const uint64_t t_now_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
        std::vector<Tracked> tracked;
        for (auto& chip : chips) {
            if (chip.cores.empty()) {
                continue;
            }
            auto* dev = chip.cores.front().device;
            SocDescriptor soc(std::make_shared<SocArchDescriptor>(chip.arch), dev->get_chip_info());
            for (const auto& l : ttnvtop_agg::probe_landings(dev, soc, t_now_us, chip.arch)) {
                Tracked t;
                t.chip_id = chip.chip_id;
                t.is_remote = chip.is_remote;
                t.core = l.core;
                t.num_cores = l.num_cores;
                t.src_chip = l.src_chip;
                t.dev = dev;
                // The journal IS at the landing base by construction (5k): its size is
                // not known until its own header is read, so it has to sit at a
                // well-known address.
                t.journal = static_cast<uint32_t>(ttnvtop_agg::landing_base(chip.arch));
                tracked.push_back(t);
            }
        }
        if (tracked.empty()) {
            std::cerr << "ttnvtop-collector: no aggregator journal found — nothing to measure.\n"
                      << "  An aggregator can only be STARTED while a tt-metal process holds the device:\n"
                      << "  idle_erisc.cc's wait loop is the only thing that polls go_messages[0], and only\n"
                      << "  a tt-metal device init puts that firmware on an inactive eth core (5r).\n";
            return 1;
        }
        std::cout << "fidelity probe: " << tracked.size() << " journal(s), " << cli.fidelity_probe_s
                  << " s at 10 Hz, read-only\n";

        const auto t0 = std::chrono::steady_clock::now();
        const auto t_end = t0 + std::chrono::seconds(cli.fidelity_probe_s);
        std::vector<uint8_t> buf;
        std::vector<uint32_t> prev_seq;
        while (std::chrono::steady_clock::now() < t_end && !g_stop.load(std::memory_order_relaxed)) {
            for (auto& t : tracked) {
                const uint32_t table_bytes = util_agg_bytes_for(t.num_cores);
                buf.assign(table_bytes, 0);
                try {
                    t.dev->read_from_device(buf.data(), t.core, t.journal, table_bytes);
                } catch (const std::exception&) {
                    ++t.failed_reads;
                    continue;
                }
                util_agg_hdr_view_t hdr{};
                std::memcpy(&hdr, buf.data(), sizeof(hdr));
                // A remote read arrives in 16 B chunks from different moments, so the
                // header's self-check is not optional -- 5n measured head and the
                // checksum coming from different publishes EVERY time.
                if (hdr.magic != UTIL_AGG_MAGIC || !util_agg_hdr_ok(hdr)) {
                    ++t.torn_reads;
                    continue;
                }
                uint64_t samples = 0, resets = 0;
                uint32_t advancing = 0;
                for (uint32_t i = 0; i < t.num_cores; ++i) {
                    util_agg_core_state_t st{};
                    std::memcpy(
                        &st, buf.data() + sizeof(util_agg_msg_t) + i * sizeof(util_agg_core_state_t), sizeof(st));
                    samples += st.samples;
                    resets += st.resets;
                    if (st.seq != 0) {
                        ++advancing;
                    }
                }
                if (!t.primed) {
                    t.first_head = hdr.head;
                    t.first_samples = samples;
                    t.first_resets = resets;
                    t.first_lost = hdr.lost;
                    t.first_sweeps = hdr.sweep_count;
                    t.primed = true;
                }
                t.last_head = hdr.head;
                t.last_samples = samples;
                t.last_resets = resets;
                t.last_lost = hdr.lost;
                t.last_sweeps = hdr.sweep_count;
                t.cores_advancing = advancing;
                ++t.ticks;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::cout << "\n=== AGGREGATOR ARM — " << std::fixed << std::setprecision(1) << secs << " s ===\n";
        std::cout << "chip  loc     cores  sweeps     entries      lost   loss%    accepted     resets  entries/s\n";
        uint64_t tot_folded = 0, tot_lost = 0;
        for (const auto& t : tracked) {
            if (!t.primed) {
                std::cout << "  " << t.chip_id << "  NEVER READ CLEANLY (torn " << t.torn_reads << ", fail "
                          << t.failed_reads << ")\n";
                continue;
            }
            // ENTRIES, not accepted samples -- see the note on `first_head` above.
            const uint64_t folded = t.last_head - t.first_head;
            const uint64_t accepted = t.last_samples - t.first_samples;
            const uint64_t resets = t.last_resets - t.first_resets;
            const uint32_t lost = t.last_lost - t.first_lost;
            const uint64_t produced = folded + lost;
            tot_folded += folded;
            tot_lost += lost;
            std::cout << std::setw(4) << t.chip_id << "  " << (t.is_remote ? "remote" : "mmio  ") << std::setw(7)
                      << t.cores_advancing << std::setw(9) << (t.last_sweeps - t.first_sweeps) << std::setw(12)
                      << folded << std::setw(10) << lost << std::setw(8) << std::setprecision(2)
                      << (produced ? 100.0 * static_cast<double>(lost) / static_cast<double>(produced) : 0.0)
                      << std::setw(12) << accepted << std::setw(11) << resets << std::setw(11) << std::setprecision(0)
                      << (secs > 0 ? folded / secs : 0.0) << "\n";
        }
        const uint64_t tot_produced = tot_folded + tot_lost;
        std::cout << std::setprecision(2) << "TOTAL entries=" << tot_folded << " lost=" << tot_lost << " loss="
                  << (tot_produced ? 100.0 * static_cast<double>(tot_lost) / static_cast<double>(tot_produced) : 0.0)
                  << "%  aggregate " << std::setprecision(0) << (secs > 0 ? tot_folded / secs : 0.0) << " samples/s\n";
        std::cout << "\nCompare against the host arm's `[ring-drain] entries=/lost=` over the same window.\n";
        return 0;
    }

    // Phase 2.2 M1 diagnostic. Scans every chip's ethernet cores for a landed
    // aggregator journal and reports what is there, then exits. Read-only, and it
    // never touches a remote chip -- journals land on MMIO chips by design, so this
    // is plain PCIe and cannot take the NON_MMIO mutex.
    if (cli.journal_probe) {
        const uint64_t now_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
        int total = 0;
        std::cout << "journal probe: landing base 0x" << std::hex << ttnvtop_agg::landing_base(chips.front().arch)
                  << std::dec << " (" << arch_name(chips.front().arch) << ")\n";
        for (auto& chip : chips) {
            // ALL chips, including remote ones.
            //
            // An earlier revision skipped remote chips, because the push design landed
            // journals on the MMIO side. v2 removed the push (5k): the journal now lives
            // in the aggregator's own eth L1, which for a remote chip's aggregator is on
            // the remote chip. Reading it over the tunnel is the design, and it is two
            // small reads -- measured at 0.1-1.1 ms even under full Llama load (5m).
            tt::umd::SocDescriptor soc(
                std::make_shared<tt::umd::SocArchDescriptor>(chip.arch), chip.cores.front().device->get_chip_info());
            const auto found = ttnvtop_agg::probe_landings(chip.cores.front().device, soc, now_us, chip.arch);
            for (const auto& l : found) {
                ++total;
                std::cout << "  chip " << chip.chip_id << " eth (" << l.core.x << "," << l.core.y << ")"
                          << "  src_chip=" << l.src_chip << " cores=" << l.num_cores << " capacity=" << l.capacity
                          << " head=" << l.last_head << " sweeps=" << l.last_sweep_count << "\n";
            }
        }
        if (total == 0) {
            std::cout << "  none found — no aggregator is running, or it landed elsewhere.\n";
        }
        return 0;
    }

    std::cout << "ttnvtop-collector: " << chips.size() << " chip(s), "
              << "publishing /dev/shm/tt_device_<asic>_util. Ctrl-C to exit.\n";

    std::mutex state_mx;  // protects CoreState samples across sampler/publisher

    // Sampling thread: two PCIe block reads per core per tick. One pulls the
    // launch mailbox (dispatch signal), one pulls the Tensix perf-counter
    // window (wall clock + FPU busy). Kept as a single thread for simplicity;
    // per-chip threads are a Phase 3 optimization.
    auto sampler = [&]() {
        std::vector<uint8_t> buf(kReadSize);
        const auto period = std::chrono::microseconds(1'000'000 / cli.sample_hz);
        auto next = std::chrono::steady_clock::now();
        // ---- BEGIN ported from tt_coremon (source lines 1535-1541)
        // DRAM NIU counters are free-running -- no arming, no owner conflict --
        // so they ride along on this thread at a fixed 5 Hz. That is far below
        // the sweep rate (the counters are 32-bit and wrap in ~8 s at full
        // bandwidth, so 5 Hz leaves ample margin) and keeps the extra NOC reads
        // off the hot per-core path.
        // ---- END ported from tt_coremon
        while (!g_stop.load(std::memory_order_relaxed)) {
            for (auto& chip : chips) {
                // 5k transport emulation: the host-pull design does NO per-core remote
                // traffic at all — the on-chip aggregator sweeps locally and the host
                // reads one journal. Skipping this loop for remote chips reproduces
                // that load profile without needing an aggregator running.
                if (cli.journal_transport && chip.is_remote) {
                    continue;
                }
                for (auto& c : chip.cores) {
                    uint8_t now_bit = 0;
                    bool have_dispatch = false;
                    try {
                        c.device->read_from_device(buf.data(), c.translated, read_addr, kReadSize);
                        have_dispatch = true;
                    } catch (const std::exception&) {
                        // continue — still try perf read below
                    }
                    if (have_dispatch) {
                        uint32_t idx = 0;
                        std::memcpy(&idx, buf.data() + kIdxOffInBuf, sizeof(idx));
                        if (idx >= go_message_num_entries) {
                            idx = 0;
                        }
                        uint32_t go_word = 0;
                        std::memcpy(&go_word, buf.data() + idx * sizeof(uint32_t), sizeof(go_word));
                        const uint8_t signal = static_cast<uint8_t>((go_word >> 24) & 0xFFu);
                        now_bit = (signal == static_cast<uint8_t>(RUN_MSG_GO)) ? 1 : 0;
                    }

                    // Phase 2.1.e kernel attribution: read launch_msg_rd_ptr to
                    // find the active slot, then pull host_assigned_id from it.
                    // +2 small PCIe reads per core per tick (vs the existing 4
                    // for perf + 1 block for dispatch). Failure keeps the prior
                    // kernel_id, which is harmless — viewer disambiguates via
                    // the D bar.
                    uint32_t kernel_id_now = 0;
                    bool have_kernel_id = false;
                    try {
                        uint32_t rd_ptr = 0;
                        c.device->read_from_device(
                            &rd_ptr, c.translated, kMailboxBase + kLaunchRdPtrOff, sizeof(rd_ptr));
                        const uint32_t slot = rd_ptr % kLaunchBufEntries;
                        const uint64_t host_id_addr =
                            kMailboxBase + kLaunchArrOff + slot * kLaunchEntrySize + kHostIdOffInLaunch;
                        c.device->read_from_device(&kernel_id_now, c.translated, host_id_addr, sizeof(kernel_id_now));
                        have_kernel_id = true;
                    } catch (const std::exception&) {
                    }

                    // Perf counters (Phase 2.0). busy% = delta(OUT_H) / delta(WALL_CLOCK).
                    //
                    // We use four separate 4-byte reads per core per tick
                    // (OUT_L, OUT_H, WALL_L, WALL_H). The 220-byte block-read
                    // approach did not survive UMD's ETH tunnel on remote
                    // chips: past the first 4 bytes the returned buffer
                    // either aliased (OUT_H == OUT_L) or read all-ones
                    // (WALL_CLOCK). Four small reads land correctly on
                    // both local and remote.
                    bool have_perf = false;
                    uint32_t fpu_out_l_now = 0;
                    uint32_t fpu_out_h_now = 0;
                    uint32_t wall_l_now = 0;
                    uint32_t wall_h_now = 0;
                    uint64_t wall_now = 0;
                    try {
                        c.device->read_from_device(&fpu_out_l_now, c.translated, kAddrFpuOutL, sizeof(fpu_out_l_now));
                        c.device->read_from_device(&fpu_out_h_now, c.translated, kAddrFpuOutH, sizeof(fpu_out_h_now));
                        c.device->read_from_device(&wall_l_now, c.translated, kAddrWallL, sizeof(wall_l_now));
                        c.device->read_from_device(&wall_h_now, c.translated, kAddrWallH, sizeof(wall_h_now));
                        have_perf = true;
                    } catch (const std::exception&) {
                    }
                    if (have_perf) {
                        wall_now = (static_cast<uint64_t>(wall_h_now) << 32) | wall_l_now;
                        // Periodic live probe: dump raw counter values for
                        // core (1,1) on each chip once per second so we can
                        // tell "counter stopped" from "counter ticking but
                        // OUT_H reads 0" from "everything working, workload
                        // really has no FPU activity here". Also prints the
                        // current counter_sel and the matching last-seen
                        // OUT_H so the log captures the alternation.
                        if (c.noc_x == 1 && c.noc_y == 1) {
                            static std::atomic<uint64_t> last_probe_us{0};
                            const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                                    std::chrono::steady_clock::now().time_since_epoch())
                                                    .count();
                            const uint64_t lp = last_probe_us.load(std::memory_order_relaxed);
                            if (static_cast<uint64_t>(now_us) - lp > 1'000'000) {
                                last_probe_us.store(now_us, std::memory_order_relaxed);
                                const uint32_t sel = c.next_counter_sel;
                                const uint32_t last_matching =
                                    (sel == kFpuCounterSelSfpu) ? c.last_sfpu_out_h : c.last_fpu_out_h;
                                std::fprintf(
                                    stderr,
                                    "[live-probe] core(1,1) is_remote=%d sel=%u "
                                    "wall=0x%016lx out_l=0x%08x out_h=0x%08x last_%s=0x%08x\n",
                                    c.is_remote ? 1 : 0,
                                    sel,
                                    wall_now,
                                    fpu_out_l_now,
                                    fpu_out_h_now,
                                    (sel == kFpuCounterSelSfpu) ? "sfpu_out_h" : "fpu_out_h",
                                    last_matching);
                                std::fflush(stderr);
                            }
                        }
                    }

                    {
                        std::lock_guard<std::mutex> lk(state_mx);
                        if (have_dispatch) {
                            const uint8_t was = c.samples[c.head];
                            c.samples[c.head] = now_bit;
                            c.head = (c.head + 1) % kWindowSamples;
                            c.busy_count = c.busy_count + now_bit - was;
                            c.samples_seen += 1;
                            c.last_dispatched = now_bit;
                        }
                        if (have_kernel_id) {
                            c.last_kernel_id = kernel_id_now;
                        }
                        if (have_perf) {
                            // Route fpu_out_h_now to the FPU or SFPU branch
                            // based on the currently-programmed counter_sel
                            // (captured in c.next_counter_sel for "what the
                            // next read will yield" — which is *this* read).
                            if (c.next_counter_sel == kFpuCounterSelSfpu) {
                                if (c.perf_primed_sfpu) {
                                    const uint64_t wall_d = wall_now - c.last_wall_sfpu;
                                    // Negative-delta guard: a backwards count
                                    // means kernel-side StartPerfCounters
                                    // reset the bank mid-interval; treat the
                                    // current value as a conservative
                                    // post-reset delta.
                                    const uint64_t sfpu_d =
                                        (fpu_out_h_now >= c.last_sfpu_out_h)
                                            ? static_cast<uint64_t>(fpu_out_h_now - c.last_sfpu_out_h)
                                            : static_cast<uint64_t>(fpu_out_h_now);
                                    if (wall_d > 0 && wall_d < (1ull << 40) && sfpu_d <= wall_d) {
                                        const double inst = static_cast<double>(sfpu_d) / static_cast<double>(wall_d);
                                        c.sfpu_busy_ewma =
                                            kComputeEwmaAlpha * inst + (1.0 - kComputeEwmaAlpha) * c.sfpu_busy_ewma;
                                    }
                                }
                                c.last_wall_sfpu = wall_now;
                                c.last_sfpu_out_h = fpu_out_h_now;
                                c.perf_primed_sfpu = true;
                            } else {
                                if (c.perf_primed_fpu) {
                                    const uint64_t wall_d = wall_now - c.last_wall_fpu;
                                    const uint64_t fpu_d = (fpu_out_h_now >= c.last_fpu_out_h)
                                                               ? static_cast<uint64_t>(fpu_out_h_now - c.last_fpu_out_h)
                                                               : static_cast<uint64_t>(fpu_out_h_now);
                                    if (wall_d > 0 && wall_d < (1ull << 40) && fpu_d <= wall_d) {
                                        const double inst = static_cast<double>(fpu_d) / static_cast<double>(wall_d);
                                        c.fpu_busy_ewma =
                                            kComputeEwmaAlpha * inst + (1.0 - kComputeEwmaAlpha) * c.fpu_busy_ewma;
                                    }
                                }
                                c.last_wall_fpu = wall_now;
                                c.last_fpu_out_h = fpu_out_h_now;
                                c.perf_primed_fpu = true;
                            }
                        }
                    }

                    // Flip counter_sel for the NEXT tick. Keep mode=0
                    // continuous in bits [7:0]. If the write fails, leave
                    // next_counter_sel unchanged so the next tick re-reads
                    // the same counter rather than misrouting data.
                    if (have_perf) {
                        const uint32_t new_sel =
                            (c.next_counter_sel == kFpuCounterSelFpu) ? kFpuCounterSelSfpu : kFpuCounterSelFpu;
                        const uint32_t new_fpu1 = (new_sel << kFpuCounterSelShift) | kFpuModeContinuous;
                        bool wrote = false;
                        try {
                            c.device->write_to_device(&new_fpu1, c.translated, kRegFpu1, sizeof(new_fpu1));
                            wrote = true;
                        } catch (const std::exception&) {
                        }
                        if (wrote) {
                            std::lock_guard<std::mutex> lk(state_mx);
                            c.next_counter_sel = new_sel;
                        }
                    }
                }
            }
            // DRAM sampling used to live here, iterating every chip from this one thread.
            // That is the same defect as the single-threaded ring drain and the
            // single-threaded journal feed: its reads cross the NON_MMIO tunnel on a remote
            // chip, so one slow link stalled the dispatch sampling of all eight. It is now
            // per-chip, in `chip_telemetry`.
            next += period;
            const auto now_t = std::chrono::steady_clock::now();
            if (next < now_t) {
                next = now_t;
            } else {
                std::this_thread::sleep_until(next);
            }
        }
    };
    std::thread sampler_thread(sampler);

    // Phase 2.1.c ring-drain thread. 50 Hz per-chip, per-core bulk read of
    // the L1 sampler ring (1 KiB at MEM_UTIL_SAMPLER_BASE), parses entries
    // newer than the per-core `last_ring_head`, applies wrap-aware
    // `wall_clock_l` deltas, and accumulates cycles per `kernel_id` with a
    // 1-second rolling window. Result is published into the program
    // registry's `cycles_in_window` field for the viewer's TIME% column.
    //
    // Until the LLK PR (Phase 2.1.c firmware) lands, the only ring producer
    // is the brisc idle-loop sampler which always writes kernel_id=0 — so
    // expect all attribution to land under runtime_id 0 (i.e. nothing
    // visible in the registry until LLK ships). That's the documented
    // intermediate state.
    // Phase 2.1.c.i: 200 Hz drain (5 ms interval). At 50 Hz with 1 ms TRISC1
    // sampling on 64 cores × 2 producer threads (BRISC + TRISC1) = ~128k
    // samples/sec/chip vs 198k drainable, observed 46% loss empirically (the
    // average rate fits but the per-tick burstiness over 64 cores doesn't —
    // each core's 62-slot ring fills in 31 ms at 2 samples/ms, so 50 Hz =
    // 20 ms interval grants no headroom). 200 Hz quadruples drain capacity to
    // ~793k/sec/chip, leaves ~6× headroom for bursts and the eventual
    // sub-ms period override.
    constexpr uint64_t kRingDrainHz = 200;
    constexpr uint64_t kRingWindowUs = 1'000'000;  // 1 s rolling window for cycle attribution.
    constexpr uint64_t kSamplerBase = static_cast<uint64_t>(MEM_UTIL_SAMPLER_BASE);
    constexpr size_t kSamplerSize = static_cast<size_t>(MEM_UTIL_SAMPLER_SIZE);
    static_assert(kSamplerSize == 1024, "ring drain assumes a 1 KiB per-core ring");
    static_assert(
        sizeof(ttnvtop_ring::Header) + ttnvtop_ring::kRingSize * sizeof(ttnvtop_ring::Entry) == kSamplerSize,
        "ring layout mismatch with util_sampler.h");

    RegistryWriter registry_writer;
    registry_writer.open_or_attach();  // best-effort; re-tried inside the loop if the file isn't there yet.
    // Shared across the per-chip drain threads below.
    std::mutex registry_mx;

    // ONE DRAIN THREAD PER CHIP.
    //
    // This was a single thread doing `for (auto& chip : chips)` with 64 serial 1 KiB
    // reads inside each -- 512 sequential reads per pass on a T3K. It asked for
    // kRingDrainHz (200) and achieved 17.1, missing its own target by 12x, and the
    // shortfall was read by me as a structural ceiling of the host-pull design. It is
    // not. Restricting the SAME single-threaded drain to one chip took sample loss from
    // 50.8% to 0.00% on an unchanged workload (measured 2026-08-30, 5t) -- the ring, the
    // tunnel and the sample rate were never the binding constraint; serialising eight
    // chips onto one thread was.
    //
    // Per-chip is the natural split: `chip.kernel_cycles` is already guarded by the
    // per-chip `chip.ring_mx`, and the only genuinely shared sink is the registry writer,
    // which takes `registry_mx` below. Remote chips may still serialise against each
    // other inside UMD's NON_MMIO mutex -- that is what the measurement is for.
    auto ring_drain = [&](size_t chip_idx) {
        const auto period = std::chrono::microseconds(1'000'000ull / kRingDrainHz);
        auto next = std::chrono::steady_clock::now();
        std::vector<uint8_t> buf(kSamplerSize);
        uint64_t reattach_throttle_us = 0;
        auto& chip = chips[chip_idx];
        if (chip.journal_active) {
            return;  // the aggregator feeds this chip; do not add tunnel traffic
        }
        while (!g_stop.load(std::memory_order_relaxed)) {
            const auto loop_t0 = std::chrono::steady_clock::now();
            const uint64_t now_us =
                std::chrono::duration_cast<std::chrono::microseconds>(loop_t0.time_since_epoch()).count();

            // Lazy re-attach to the registry if the workload hadn't started
            // yet at collector launch. Probe at most once per 2 s to avoid
            // syscall churn.
            if (!registry_writer.enabled && now_us > reattach_throttle_us) {
                std::lock_guard<std::mutex> rg(registry_mx);
                registry_writer.open_or_attach();
                reattach_throttle_us = now_us + 2'000'000;
            }

            // Phase 2.1.c.i: re-assert period_cycles override every drain
            // tick (200 Hz, 5 ms). brisc init() re-runs at every device
            // open and races our override; at 1 Hz reassert, brisc can
            // sample at the firmware's baked-in default for up to ~1 s
            // between corrections — enough to overflow the ring on any
            // sub-ms baked period. Per-tick reassert closes that window
            // to the drain interval. 64 cores × 1 u32 × 200 Hz =
            // ~50 KB/sec/chip — negligible host PCIe usage.
            // 5 ms at 1 GHz. Drain runs at ~10 Hz observed (UMD read latency
            // for 128 cores × 1 KiB/tick saturates), so drainable per chip is
            // ~40k/sec. At 5 ms period: 200 samples/sec/core × 2 producers ×
            // 64 cores = 25.6k/sec/chip. Leaves ~36% headroom for transient
            // bursts. Sub-10 ms is still finer than host poll's effective
            // ~10 ms cadence, so per-program TIME% is meaningfully better
            // than the pre-Phase-2.1.c baseline.
            constexpr uint32_t kPeriodOverrideCycles = 5'000'000u;
            constexpr uint64_t kPeriodAddr = static_cast<uint64_t>(MEM_UTIL_SAMPLER_BASE) + 12;
            // Reassert at 1 Hz only. Per-tick reassert at 200 Hz costs 128
            // UMD writes/tick — pushing drain rate below 50 Hz observed.
            // brisc init() runs only on device open, so 1 Hz is enough to
            // re-cover that race window.
            const bool reassert_period = (now_us - chip.last_period_assert_us) > 1'000'000;
            if (reassert_period) {
                chip.last_period_assert_us = now_us;
            }
            // Phase 2.1.c.i diagnostic: every 5 s, read back period_cycles
            // from one core after writing it, to verify the override is
            // sticking. If the read-back value != override, something else
            // (firmware re-init, kernel-side write) is racing us.

            const bool probe_period = (now_us - chip.last_period_probe_us) > 5'000'000;
            if (probe_period) {
                chip.last_period_probe_us = now_us;
            }

            {
                std::lock_guard<std::mutex> lk(*chip.ring_mx);
                ++chip.drain_ticks;

                // 5k transport emulation. Two remote reads per tick — a 64 B journal
                // header and one bulk entry read — instead of num_cores KiB-sized
                // per-core reads. That is the transport profile the aggregator design
                // imposes: ~20 remote transactions/s against ~12,800 today.
                if (cli.journal_transport && chip.is_remote && !chip.cores.empty()) {
                    auto* dev = chip.cores.front().device;
                    const auto core = chip.cores.front().translated;
                    std::vector<uint8_t> jbuf(8192);
                    try {
                        dev->read_from_device(jbuf.data(), core, ttnvtop_agg::landing_base(chip.arch), 64);
                        dev->read_from_device(jbuf.data(), core, ttnvtop_agg::landing_base(chip.arch) + 64, 8192);
                        chip.journal_entries_seen += 2;
                    } catch (...) {
                        ++chip.journal_stale_ticks;
                    }
                    continue;
                }

                int probe_good = 0, probe_bad = 0;
                // ---- BEGIN ported from tt_coremon (source lines 1556-1583)
                // REMOTE BUDGET. Reached only on a remote chip with NO aggregator --
                // an aggregated chip returned from this lambda before the loop ever
                // started. So this is the fallback path's back-pressure, not the
                // primary strategy: see CliOptions::remote_budget for why the drain
                // being throttled here was measuring 0.1-0.2 Hz at 99.4-99.9% sample
                // loss anyway, and why bounding it costs nothing real.
                //
                // Local chips are never budgeted -- they take no NON_MMIO lock, so
                // there is nothing to be polite about.
                size_t budget_cores = chip.cores.size();
                if (chip.is_remote && cli.remote_budget > 0.0 && !chip.cores.empty()) {
                    if (chip.drain_last_refill.time_since_epoch().count() == 0) {
                        chip.drain_last_refill = loop_t0;
                    }
                    const double refill_dt = std::chrono::duration<double>(loop_t0 - chip.drain_last_refill).count();
                    chip.drain_last_refill = loop_t0;
                    chip.drain_tokens += cli.remote_budget * refill_dt;
                    // Cap the bucket at one full rotation: an idle stretch must
                    // not bank credit and then spend it as one long burst,
                    // which is the starvation we are trying to avoid.
                    const double cap = kDrainOpsPerCore * static_cast<double>(chip.cores.size());
                    if (chip.drain_tokens > cap) {
                        chip.drain_tokens = cap;
                    }
                    budget_cores = static_cast<size_t>(chip.drain_tokens / kDrainOpsPerCore);
                    if (budget_cores > chip.cores.size()) {
                        budget_cores = chip.cores.size();
                    }
                    chip.drain_tokens -= static_cast<double>(budget_cores) * kDrainOpsPerCore;
                }
                for (size_t bk = 0; bk < budget_cores; ++bk) {
                    // Resume from drain_rr_cursor so a constrained budget rotates
                    // coverage rather than re-reading the same first N cores forever.
                    auto& c = chip.cores[chip.drain_rr_cursor];
                    chip.drain_rr_cursor = (chip.drain_rr_cursor + 1) % chip.cores.size();
                    ++chip.drain_cores_sampled;
                    // ---- END ported from tt_coremon
                    if (reassert_period) {
                        try {
                            c.device->write_to_device(
                                &kPeriodOverrideCycles, c.translated, kPeriodAddr, sizeof(uint32_t));
                        } catch (...) { /* same silent-fail as initial write */
                        }
                    }
                    if (probe_period) {
                        try {
                            uint32_t period_rb = 0;
                            c.device->read_from_device(&period_rb, c.translated, kPeriodAddr, 4);
                            if (period_rb == kPeriodOverrideCycles) {
                                ++probe_good;
                            } else {
                                ++probe_bad;
                                if (probe_bad <= 3) {
                                    std::fprintf(
                                        stderr,
                                        "[probe-bad] chip=%u core=(%d,%d) period=%u\n",
                                        static_cast<unsigned>(chip.chip_id),
                                        c.noc_x,
                                        c.noc_y,
                                        period_rb);
                                }
                            }
                        } catch (...) {
                        }
                    }
                    bool have_buf = false;
                    try {
                        c.device->read_from_device(buf.data(), c.translated, kSamplerBase, kSamplerSize);
                        have_buf = true;
                    } catch (const std::exception&) {
                        // Skip this core for this tick; remote chips may
                        // intermittently fail under ETH tunnel pressure.
                    }
                    if (!have_buf) {
                        continue;
                    }

                    ttnvtop_ring::Header hdr;
                    std::memcpy(&hdr, buf.data(), sizeof(hdr));
                    if (hdr.magic != ttnvtop_ring::kMagic) {
                        // Firmware hasn't initialized the sampler on this
                        // core yet. Common at collector startup. Silent
                        // skip — re-probed every tick.
                        continue;
                    }
                    if (hdr.version != ttnvtop_ring::kVersion) {
                        // Schema mismatch: log once per tick at most for
                        // the *first* core in the chip, never spam.
                        static thread_local uint64_t last_warn_us = 0;
                        if (now_us - last_warn_us > 5'000'000) {
                            std::fprintf(
                                stderr,
                                "[ring-drain] chip %u: util_sampler version mismatch (got %u, want %u) — "
                                "skipping drain\n",
                                static_cast<unsigned>(chip.chip_id),
                                hdr.version,
                                ttnvtop_ring::kVersion);
                            std::fflush(stderr);
                            last_warn_us = now_us;
                        }
                        continue;
                    }

                    const uint32_t cur_head = hdr.head;
                    if (!c.ring_primed) {
                        // First sight of this ring: just record state.
                        // Subsequent ticks compute deltas.
                        c.last_ring_head = cur_head;
                        // Seed wall-clock from the most recent slot if any.
                        if (cur_head > 0) {
                            const uint32_t last_slot = (cur_head - 1) % ttnvtop_ring::kRingSize;
                            ttnvtop_ring::Entry e;
                            std::memcpy(
                                &e,
                                buf.data() + sizeof(ttnvtop_ring::Header) + last_slot * sizeof(ttnvtop_ring::Entry),
                                sizeof(e));
                            c.last_ring_wall_l = e.wall_clock_l;
                            c.last_ring_kernel_id = e.kernel_id;
                        }
                        c.ring_primed = true;
                        continue;
                    }

                    if (cur_head == c.last_ring_head) {
                        continue;  // no new entries this tick
                    }

                    // Detect ring re-initialization: brisc's `init()` resets
                    // `head` to 0 every time the firmware boots (e.g., across
                    // pytest test boundaries that close+reopen the device).
                    // Without this check, the unsigned subtraction below
                    // underflows and ~2^32 gets added per core to
                    // drain_lost_samples (visible as ~275B jumps per chip in
                    // the drain log, since 64 cores × 2^32 ≈ 275 GB).
                    if (cur_head < c.last_ring_head) {
                        c.last_ring_head = cur_head;
                        continue;  // treat as fresh start; pick up fresh entries next tick
                    }

                    // How many entries to ingest. If host stalled and the
                    // producer wrote more than the ring capacity, we lost
                    // the oldest ones — clamp and report. The most recent
                    // `kRingSize` slots are always intact.
                    uint32_t new_count = cur_head - c.last_ring_head;
                    if (new_count > ttnvtop_ring::kRingSize) {
                        chip.drain_lost_samples += (new_count - ttnvtop_ring::kRingSize);
                        new_count = ttnvtop_ring::kRingSize;
                    }
                    const uint32_t start_head = cur_head - new_count;

                    for (uint32_t i = 0; i < new_count; ++i) {
                        const uint32_t slot = (start_head + i) % ttnvtop_ring::kRingSize;
                        ttnvtop_ring::Entry e;
                        std::memcpy(
                            &e,
                            buf.data() + sizeof(ttnvtop_ring::Header) + slot * sizeof(ttnvtop_ring::Entry),
                            sizeof(e));
                        ++chip.drain_entries_seen;

                        // Phase 2.1.c.i attribution model: each Hook B fire
                        // contributes `period` cycles to its own kid. This is
                        // the unbiased "fires × period" estimator: the
                        // expected fires for kernel K is total_TRISC1_time_K
                        // / period, so fires × period is a one-shot estimate
                        // of K's TRISC1 cycle time.
                        //
                        // Trade-offs accepted:
                        //   • COVERAGE is high — every program that ever ran
                        //     TRISC1 work and was sampled at least once gets
                        //     cycles_total > 0. compare.py uses cycles_total
                        //     > 0 as the "Hook B saw it" predicate.
                        //   • PER-OP CYCLE PRECISION is workload-dependent —
                        //     a kernel running for time T < period gets
                        //     attributed period (over-count by period/T per
                        //     fire, symmetric across kernels). For Llama
                        //     decode where many kernels are ~100µs at
                        //     period=5ms, R² between profiler and registry
                        //     is structurally low. R² is reported as
                        //     informational only — not gated.
                        //
                        // Same-kid wall_d attribution was tested earlier and
                        // gave clean R² (0.5+) but only attributed kernels
                        // long enough to span two same-kid samples — 1.5%
                        // coverage on Llama. For "is ttnvtop seeing the op"
                        // questions, fires-based is the right semantic.
                        const uint32_t cur_raw = (e.kernel_id >> 10) & 0x1FFFFFu;
                        const uint64_t period =
                            hdr.period_cycles ? static_cast<uint64_t>(hdr.period_cycles) : 5'000'000ull;
                        if (cur_raw != 0) {
                            chip.kernel_cycles[cur_raw].add(now_us, period);
                            chip.kernel_cycles_total[cur_raw] += period;
                        }

                        c.last_ring_wall_l = e.wall_clock_l;
                        c.last_ring_kernel_id = e.kernel_id;
                    }
                    c.last_ring_head = cur_head;
                }

                // Decay the rolling window. Walk the map, drop stale
                // entries, erase empty kernels (keeps the map bounded
                // even if many transient kernel_ids show up).
                uint64_t total_chip_cycles = 0;
                for (auto it = chip.kernel_cycles.begin(); it != chip.kernel_cycles.end();) {
                    it->second.decay(now_us, kRingWindowUs);
                    if (it->second.samples.empty()) {
                        it = chip.kernel_cycles.erase(it);
                    } else {
                        total_chip_cycles += it->second.total;
                        ++it;
                    }
                }

                // Publish per-kernel totals into the registry SHM. The
                // registrar publishes runtime_id under host_assigned_id
                // bits [9:0]/[30:10] — but the registrar uses the lower
                // tt-metal `runtime_id` (program.get_runtime_id()) which
                // matches the launch_msg field directly, so we look up
                // by kernel_id as-is. Kernel_id 0 = "no kernel" -> skip.
                //
                // We iterate over the SUPERSET of (kernel_cycles ∪
                // kernel_cycles_total) so that kernels which decayed out of
                // the rolling-window map still get their monotonic total
                // refreshed in the registry. Without this union, a kernel
                // that finished >1 s before the audit point would have its
                // last cycles_total snapshot frozen at decay time — fine for
                // most cases, but we want every drain tick to reflect the
                // current monotonic total in case downstream tooling polls
                // the registry between bursts.
                // Iterate over the UNION of (kernel_cycles, kernel_cycles_total)
                // so that:
                //   - Programs with live samples but no same-kid pair yet
                //     (cycles_total still 0) still get cycles_in_window
                //     refreshed in the registry → viewer TIME% works.
                //   - Programs that ended before the current window (decayed
                //     out of kernel_cycles) still get cycles_total kept current
                //     in the registry → compare.py audit works.
                std::unordered_set<uint32_t> all_rids;
                for (auto& [k, _] : chip.kernel_cycles) {
                    all_rids.insert(k);
                }
                for (auto& [k, _] : chip.kernel_cycles_total) {
                    all_rids.insert(k);
                }
                for (uint32_t rid : all_rids) {
                    if (rid == 0) {
                        continue;
                    }
                    auto wit = chip.kernel_cycles.find(rid);
                    auto tit = chip.kernel_cycles_total.find(rid);
                    const uint64_t cyc_window = (wit != chip.kernel_cycles.end()) ? wit->second.total : 0ull;
                    const uint64_t cyc_total = (tit != chip.kernel_cycles_total.end()) ? tit->second : 0ull;
                    {
                        std::lock_guard<std::mutex> rg(registry_mx);
                        registry_writer.update_kernel_cycles(rid, cyc_window, cyc_total);
                    }
                }

                // Periodic debug log: once per 5 s, dump tick + lost +
                // entries + map size per chip. Cheap, useful when triaging
                // "TIME% is stuck at 0".
                if (now_us - chip.last_debug_us > 5'000'000) {
                    const uint64_t dt_us = (chip.last_debug_us == 0) ? 1 : (now_us - chip.last_debug_us);
                    const double drain_hz =
                        static_cast<double>(chip.drain_ticks - chip.last_debug_ticks) * 1'000'000.0 / dt_us;
                    chip.last_debug_us = now_us;
                    chip.last_debug_ticks = chip.drain_ticks;
                    std::fprintf(
                        stderr,
                        "[ring-drain] chip=%u ticks=%lu drain_hz=%.1f entries=%lu lost=%lu kernels=%zu chip_cycles=%lu "
                        "period_ok=%d/%d\n",
                        static_cast<unsigned>(chip.chip_id),
                        static_cast<unsigned long>(chip.drain_ticks),
                        drain_hz,
                        static_cast<unsigned long>(chip.drain_entries_seen),
                        static_cast<unsigned long>(chip.drain_lost_samples),
                        chip.kernel_cycles.size(),
                        static_cast<unsigned long>(total_chip_cycles),
                        probe_good,
                        probe_good + probe_bad);
                    std::fflush(stderr);
                    // ---- BEGIN ported from tt_coremon (report_health, source lines 1055-1061)
                    // "Quiet" and "budgeted down to nothing" must not look the same from
                    // the outside; without this line a throttled remote chip is
                    // indistinguishable from an idle one -- the same ambiguity that let a
                    // frozen counter masquerade as an idle workload.
                    if (chip.is_remote && cli.remote_budget > 0.0 && !chip.cores.empty()) {
                        const double per_core_hz = static_cast<double>(chip.drain_cores_sampled) /
                                                   static_cast<double>(chip.cores.size()) * 1'000'000.0 /
                                                   static_cast<double>(dt_us);
                        std::fprintf(
                            stderr,
                            "[remote-budget] chip=%u no aggregator: per-core drain %.2f Hz at %.0f tx/s budget\n",
                            static_cast<unsigned>(chip.chip_id),
                            per_core_hz,
                            cli.remote_budget);
                        std::fflush(stderr);
                    }
                    chip.drain_cores_sampled = 0;
                    // ---- END ported from tt_coremon
                }
            }

            next += period;
            const auto now_t = std::chrono::steady_clock::now();
            if (next < now_t) {
                next = now_t;
            } else {
                std::this_thread::sleep_until(next);
            }
        }
    };
    // JOURNAL FEED THREAD: the on-chip aggregator -> shm, so the viewers see the lossless
    // source instead of a per-core drain that reads 0.1 Hz on a busy remote chip.
    //
    // Two small reads per chip per tick (header + state table, ~2 KB) against the 64 KiB
    // the ring drain moved, and the accumulators are MONOTONIC -- a late or missed read
    // costs staleness, never data, which is the property the whole design rests on. So
    // this loop cannot lose samples no matter how contended the tunnel gets; that is why
    // it is the answer for remote chips.
    // ONE FEED THREAD PER CHIP.
    //
    // This was a single thread iterating all chips, and it reproduced exactly the defect the
    // ring drain had: a remote-chip read crosses the NON_MMIO tunnel and contends with the
    // workload's own traffic, so one slow read starves every chip behind it. Measured before
    // this fix: 8 trace lines in 25 s where ~250 ticks were due, and chips 4-7 never reached
    // the loop body at all. Per chip, a stalled remote read costs only that chip's
    // freshness -- which the monotonic accumulators already tolerate by design.
    auto journal_feed = [&](size_t chip_idx) {
        auto& chip = chips[chip_idx];
        if (!chip.journal_active) {
            return;
        }
        std::vector<uint8_t> buf;
        std::map<uint32_t, uint64_t> jf_ticks, jf_pub, jf_skip_seq, jf_skip_dwall;
        while (!g_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            {
                const uint32_t nc = chip.journal_num_cores;
                const uint32_t bytes = util_agg_bytes_for(nc);
                buf.assign(bytes, 0);
                auto* dev = chip.cores.front().device;
                // Unconditional trace on the first ticks, BEFORE any bail-out. Without it a
                // silent `continue` here is indistinguishable from the thread never running,
                // which is exactly the ambiguity that cost the last three runs.
                const bool trace = jf_ticks[chip.chip_id] < 3;
                ++jf_ticks[chip.chip_id];
                try {
                    if (!read_chunked_stoppable(
                            dev, chip.journal_core, ttnvtop_agg::landing_base(chip.arch), buf.data(), bytes)) {
                        break;  // stopping: the buffer is partial, do not fold it in
                    }
                } catch (const std::exception& e) {
                    if (trace) {
                        std::fprintf(
                            stderr,
                            "[journal-feed] chip=%u READ THREW at 0x%llx len %u: %s\n",
                            static_cast<unsigned>(chip.chip_id),
                            static_cast<unsigned long long>(ttnvtop_agg::landing_base(chip.arch)),
                            bytes,
                            e.what());
                        std::fflush(stderr);
                    }
                    continue;
                }
                util_agg_hdr_view_t hdr{};
                std::memcpy(&hdr, buf.data(), sizeof(hdr));
                if (trace) {
                    std::fprintf(
                        stderr,
                        "[journal-feed] chip=%u read ok: magic=0x%08x (want 0x%08x) head=%u head_xor=0x%08x "
                        "hdr_ok=%d sweeps=%u nc=%u bytes=%u\n",
                        static_cast<unsigned>(chip.chip_id),
                        hdr.magic,
                        UTIL_AGG_MAGIC,
                        hdr.head,
                        hdr.head_xor,
                        util_agg_hdr_ok(hdr) ? 1 : 0,
                        hdr.sweep_count,
                        nc,
                        bytes);
                    std::fflush(stderr);
                }
                // A remote read arrives in 16 B chunks from different moments, so the
                // header's self-check is not optional.
                if (hdr.magic != UTIL_AGG_MAGIC || !util_agg_hdr_ok(hdr)) {
                    continue;
                }
                uint32_t dbg_samples = 0, dbg_busy = 0, dbg_wall = 0;
                for (uint32_t i = 0; i < nc; ++i) {
                    const int ci = chip.journal_to_core[i];
                    if (ci < 0) {
                        continue;
                    }
                    util_agg_core_state_t st{};
                    std::memcpy(
                        &st, buf.data() + sizeof(util_agg_msg_t) + i * sizeof(util_agg_core_state_t), sizeof(st));
                    if (i == 0) {
                        dbg_samples = st.samples;
                        dbg_busy = st.busy_cycles;
                        dbg_wall = st.wall_cycles;
                    }
                    if (st.seq == chip.prev_seq[i]) {
                        ++jf_skip_seq[chip.chip_id];
                        continue;  // this core's block has not been rewritten since last read
                    }
                    // Unsigned subtraction is wrap-correct; both counters are u32 and
                    // wall_cycles wraps every ~4.3 s at 1 GHz, which is why the drain must
                    // read faster than that (10 Hz here, three orders of margin).
                    const uint32_t dbusy = st.busy_cycles - chip.prev_busy[i];
                    const uint32_t dwall = st.wall_cycles - chip.prev_wall[i];
                    chip.prev_busy[i] = st.busy_cycles;
                    chip.prev_wall[i] = st.wall_cycles;
                    chip.prev_seq[i] = st.seq;
                    if (dwall == 0) {
                        ++jf_skip_dwall[chip.chip_id];
                        continue;
                    }
                    ++jf_pub[chip.chip_id];
                    double util = static_cast<double>(dbusy) / static_cast<double>(dwall);
                    util = std::clamp(util, 0.0, 1.0);
                    auto& v = chip.publisher.cores()[static_cast<size_t>(ci)];
                    // counter_sel says WHICH pipe the accumulated deltas came from, so the
                    // number lands in the matching field rather than being asserted as FPU.
                    if (st.counter_sel == 1u) {
                        v.sfpu_busy_p1000 = static_cast<uint16_t>(util * 1000.0);
                    } else {
                        v.compute_busy_p1000 = static_cast<uint16_t>(util * 1000.0);
                    }
                    v.samples_seen = st.samples;
                    if (st.kernel_id != 0) {
                        v.last_kernel_id = (st.kernel_id >> 10) & 0x1FFFFFu;
                    }
                }
                chip.publisher.mark_updated();
                // Periodic diagnostic: without it, "the viewer shows zeros" is
                // indistinguishable from "the feed never ran", "seq never advanced" and
                // "the deltas are genuinely zero".
                if (jf_ticks[chip.chip_id] % 100 == 0) {
                    std::fprintf(
                        stderr,
                        "[journal-feed] chip=%u ticks=%lu published=%lu skipped_seq=%lu skipped_dwall=%lu "
                        "hdr_head=%u last_samples=%u last_busy=%u last_wall=%u\n",
                        static_cast<unsigned>(chip.chip_id),
                        static_cast<unsigned long>(jf_ticks[chip.chip_id]),
                        static_cast<unsigned long>(jf_pub[chip.chip_id]),
                        static_cast<unsigned long>(jf_skip_seq[chip.chip_id]),
                        static_cast<unsigned long>(jf_skip_dwall[chip.chip_id]),
                        hdr.head,
                        dbg_samples,
                        dbg_busy,
                        dbg_wall);
                    std::fflush(stderr);
                }
            }
        }
    };
    std::vector<std::thread> journal_feed_threads;
    {
        for (size_t i = 0; i < chips.size(); ++i) {
            if (chips[i].journal_active) {
                journal_feed_threads.emplace_back(journal_feed, i);
            }
        }
    }

    std::vector<std::thread> ring_drain_threads;
    ring_drain_threads.reserve(chips.size());
    for (size_t i = 0; i < chips.size(); ++i) {
        ring_drain_threads.emplace_back(ring_drain, i);
    }

    // ONE TELEMETRY THREAD PER CHIP -- AICLK and DRAM counters.
    //
    // Both are per-chip device reads, and on a remote chip both cross the tunnel. Held on
    // a shared loop they starved everything behind them: get_clock() on the publish path
    // froze the whole SHM set, and dram_update() on the sampler path froze dispatch
    // sampling for all eight chips. Per chip, a bad link costs only that chip's telemetry,
    // which the reader already renders correctly -- 0 means "not sampled" for the DRAM
    // fields and "unknown" for AICLK, both of which the viewer hides rather than faking.
    //
    // DRAM keeps the 200 ms cadence its counters need (32-bit, ~8 s wrap at full
    // bandwidth). AICLK is the expensive one -- a legacy ARC message on Wormhole -- so it
    // goes at 1 Hz, which is plenty for a clock, and its cost lands on this chip alone.
    std::vector<std::thread> telemetry_threads;
    for (size_t ci = 0; ci < chips.size(); ++ci) {
        telemetry_threads.emplace_back([&chips, ci]() {
            auto& chip = chips[ci];
            auto last_t = std::chrono::steady_clock::now();
            int tick = 0;
            while (!g_stop.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(kDramSampleInterval);
                if (g_stop.load(std::memory_order_relaxed)) {
                    break;
                }
                const auto now_t = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(now_t - last_t).count();
                last_t = now_t;
                dram_update(chip, dt);
                if (++tick % 5 == 0 && !chip.cores.empty()) {
                    try {
                        chip.aiclk_mhz = chip.cores.front().device->get_clock();
                    } catch (const std::exception&) {
                        // KEEP THE LAST GOOD VALUE. get_clock() is a legacy ARC message on
                        // Wormhole and it times out intermittently -- observed rotating
                        // between chips run to run (5, 7, then 4, 6, 7), which is a flaky
                        // mailbox, not a chip whose clock is genuinely 0. Zeroing on every
                        // transient failure made the viewer flip to "@ 0 MHz" and back,
                        // which reads as broken telemetry. 0 now means only "never read
                        // successfully", which is what the viewer's "unknown" branch means.
                    }
                }
            }
        });
    }

    // Publisher: copy rolling stats into SHM at the configured rate.
    const auto publish_period = std::chrono::milliseconds(1000 / cli.publish_hz);
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(publish_period);
        for (auto& chip : chips) {
            auto* views = chip.publisher.cores();
            if (views == nullptr) {
                continue;
            }
            // AICLK and the DRAM rates are whatever this chip's telemetry thread last
            // cached. No device access on this path: publishing is the one thing that must
            // stay fast for every chip regardless of what any single chip's link is doing.
            if (auto* header = chip.publisher.header(); header != nullptr) {
                header->aiclk_mhz = chip.aiclk_mhz;
                // ---- BEGIN ported from tt_coremon (source lines 2343-2345)
                header->dram_rd_mbps = chip.dram_rd_mbps;
                header->dram_wr_mbps = chip.dram_wr_mbps;
                header->dram_peak_mbps = chip.dram_peak_mbps;
                // ---- END ported from tt_coremon
            }
            std::lock_guard<std::mutex> lk(state_mx);
            for (size_t i = 0; i < chip.cores.size(); ++i) {
                const auto& c = chip.cores[i];
                auto& v = views[i];
                v.dispatched = c.last_dispatched;
                // busy_count is the count over kWindowSamples samples. Convert to per-mille.
                v.dispatch_busy_p1000 = static_cast<uint16_t>((c.busy_count * 1000u) / kWindowSamples);
                v.samples_seen = static_cast<uint32_t>(c.samples_seen);
                // Compute busy: two EWMAs in [0,1] — FPU (MATH/matmul pipe)
                // and SFPU (vector pipe). Clamp each and convert to per-mille.
                // compute_busy_p1000 continues to mean FPU-only; SFPU goes in
                // its own field.
                double fb = c.fpu_busy_ewma;
                if (fb < 0.0) {
                    fb = 0.0;
                } else if (fb > 1.0) {
                    fb = 1.0;
                }
                double sb = c.sfpu_busy_ewma;
                if (sb < 0.0) {
                    sb = 0.0;
                } else if (sb > 1.0) {
                    sb = 1.0;
                }
                // ONE WRITER PER FIELD. On a chip fed by an on-chip aggregator the journal
                // feed owns these two fields, and it must not be overwritten here.
                //
                // Both paths were writing them every tick from different sources -- the feed
                // from the aggregator's busy/wall deltas, this loop from its own EWMAs -- so
                // whichever ran last won. The visible symptom was a reading that jumped
                // between chips and between the FPU and SFPU fields tick to tick, reported
                // as "intermittent, and sometimes telemetry just hangs". It also explains
                // pipes reading a flat 0 while the other showed a plausible number: the feed
                // deliberately writes only the pipe that counter_sel says was measured, and
                // this loop then clobbered the other one with a stale EWMA.
                if (!chip.journal_active) {
                    v.compute_busy_p1000 = static_cast<uint16_t>(fb * 1000.0);
                    v.sfpu_busy_p1000 = static_cast<uint16_t>(sb * 1000.0);
                }
                v.last_kernel_id = c.last_kernel_id;
                // Phase 2+ fields stay at their zero-initialized values.
            }
            chip.publisher.mark_updated();
        }
    }

    // RETIRE THE AGGREGATORS ON THE WAY OUT -- first shutdown action, before any join.
    //
    // The kernel keeps the ethernet firmware's 0xABCD heartbeat advancing while it runs,
    // so a LIVE aggregator is not what wedges topology discovery. An abandoned one is:
    // once this process is gone nothing pets that word again, and UMD's next
    // `eth_heartbeat_running` sits in its timeouts for every process on the machine.
    // Leaving them resident is what made a stuck collector cost a board reset in 5u.
    //
    // Four bytes per chip, so it fits inside the shutdown grace even on a contended
    // tunnel, and it is idempotent -- `--stop-aggregator` does exactly this. If a write
    // does block behind a spinning tunnel read, the grace watchdog still takes the
    // process down; this only ever improves on that outcome, never delays it.
    {
        int asked = 0;
        for (auto& chip : chips) {
            if (!chip.journal_active || chip.cores.empty()) {
                continue;
            }
            const auto l1 = ttnvtop::agg_layout(
                static_cast<uint32_t>(ttnvtop_agg::landing_base(chip.arch)), chip.journal_num_cores);
            const uint32_t req = 0x504F5453u;  // 'STOP' -- see kStopRequest in eth_aggregator.cpp
            try {
                chip.cores.front().device->write_to_device(&req, chip.journal_core, l1.dbg + 12u, sizeof(req));
                ++asked;
            } catch (const std::exception&) {
                // Best effort. A chip we cannot reach now is one --stop-aggregator must
                // clear later, which is why that command still exists.
            }
        }
        if (asked > 0) {
            std::cout << "\nttnvtop-collector: asked " << asked << " aggregator(s) to retire.\n";
        }
    }

    sampler_thread.join();
    for (auto& t : ring_drain_threads) {
        t.join();
    }
    for (auto& t : journal_feed_threads) {
        t.join();
    }
    for (auto& t : telemetry_threads) {
        t.join();
    }
    std::cout << "\nttnvtop-collector: exiting.\n";
    return 0;
}
