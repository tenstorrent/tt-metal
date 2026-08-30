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
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
                 "  --journal-probe        scan every chip's ethernet cores for an\n"
                 "                         aggregator journal, print what was found, exit\n"
                 "  --journal-probe         Scan every chip's ethernet cores for a Phase 2.2\n"
                 "                          aggregator journal, print what was found, and exit.\n"
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

// Idle-eth UNRESERVED base, mirroring wh_hal_idle_eth.cpp / bh_hal_idle_eth.cpp:
//     ((MEM_IERISC_MAP_END + L1_KERNEL_CONFIG_SIZE - 1) | (max_alignment - 1)) + 1
// The collector deliberately does not link tt-metal (see PLAN 7.2), so the Hal is not
// available and this is mirrored, like the ring layout above. L1_KERNEL_CONFIG_SIZE is
// 25 KiB and max_alignment is max(DRAM_ALIGNMENT, L1_ALIGNMENT) = 32 on both arches.
constexpr uint32_t kL1KernelConfigSize = 25u * 1024u;
constexpr uint32_t kMaxAlignment = 32u;
constexpr uint64_t kLandingBase =
    ((static_cast<uint64_t>(MEM_IERISC_MAP_END) + kL1KernelConfigSize - 1u) | (kMaxAlignment - 1u)) + 1u;

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
// over PCIe is nothing. A core either has 'TTAG' at kLandingBase or it does not.
//
// The header is self-describing -- it carries src_chip, capacity and num_cores -- so
// discovery needs no coordination with whoever launched the aggregator. That matters
// because in M1 the launcher is the workload process and the collector is a separate
// one, with no IPC between them.
template <typename Dev>
inline std::vector<Landing> probe_landings(Dev* dev, const tt::umd::SocDescriptor& soc, uint64_t now_us) {
    std::vector<Landing> found;
    std::vector<uint8_t> buf(sizeof(util_agg_msg_t));
    for (const auto& eth : soc.get_cores(CoreType::ETH, CoordSystem::TRANSLATED)) {
        try {
            dev->read_from_device(buf.data(), eth, kLandingBase, buf.size());
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

void handle_sigint(int) { g_stop.store(true); }

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
    opts.discover_remote_devices = true;
    opts.wait_on_ethernet_link_training = false;
    opts.low_power = true;
    opts.cmfw_mismatch_action = TopologyDiscoveryOptions::Action::IGNORE;
    opts.cmfw_unsupported_action = TopologyDiscoveryOptions::Action::IGNORE;
    opts.eth_fw_mismatch_action = TopologyDiscoveryOptions::Action::IGNORE;
    opts.eth_fw_heartbeat_failure = TopologyDiscoveryOptions::Action::IGNORE;
    opts.unexpected_routing_firmware_config = TopologyDiscoveryOptions::Action::IGNORE;

    auto [cluster_desc, devices] = TopologyDiscovery::discover(opts);
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
        if (arch != tt::ARCH::WORMHOLE_B0) {
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
        if (cli.journal_probe || cli.read_latency_probe) {
            for (size_t i = 0; i < chip.cores.size(); ++i) {
                auto& v = chip.publisher.cores()[i];
                v.noc_x = static_cast<uint8_t>(chip.cores[i].noc_x);
                v.noc_y = static_cast<uint8_t>(chip.cores[i].noc_y);
                v.is_remote = chip.is_remote ? 1u : 0u;
            }
            chips.push_back(std::move(chip));
            continue;
        }

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

    if (chips.empty()) {
        std::cerr << "ttnvtop-collector: no supported chips to monitor.\n";
        return 1;
    }

    // Remote-read latency vs transfer size.
    //
    // The host-pull design's drain rate is bounded by how long a remote read takes, not
    // by how many transactions it issues (5l: ~250 ms per read while Llama runs, giving
    // 0.5 Hz). Whether aggregating on-chip helps depends entirely on whether that
    // latency is FIXED OVERHEAD -- in which case shrinking the payload buys nothing --
    // or SIZE-PROPORTIONAL, in which case a 2 KB state table drains comfortably.
    if (cli.read_latency_probe) {
        static const size_t kSizes[] = {64, 256, 1024, 2048, 4096, 8192};
        std::vector<uint8_t> buf(131072);
        std::cout << "remote-read latency probe (read-only), base 0x" << std::hex << ttnvtop_agg::kLandingBase
                  << std::dec << "\n";
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
                        dev->read_from_device(buf.data(), core, ttnvtop_agg::kLandingBase, sz);
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

    // Phase 2.2 M1 diagnostic. Scans every chip's ethernet cores for a landed
    // aggregator journal and reports what is there, then exits. Read-only, and it
    // never touches a remote chip -- journals land on MMIO chips by design, so this
    // is plain PCIe and cannot take the NON_MMIO mutex.
    if (cli.journal_probe) {
        const uint64_t now_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count());
        int total = 0;
        std::cout << "journal probe: landing base 0x" << std::hex << ttnvtop_agg::kLandingBase << std::dec << "\n";
        for (auto& chip : chips) {
            // MMIO chips only. Journals land on the MMIO side by design, and probing a
            // remote chip would read over the NON_MMIO tunnel -- the exact thing this
            // whole feature exists to keep the collector off.
            if (chip.is_remote) {
                continue;
            }
            tt::umd::SocDescriptor soc(
                std::make_shared<tt::umd::SocArchDescriptor>(chip.arch), chip.cores.front().device->get_chip_info());
            const auto found = ttnvtop_agg::probe_landings(chip.cores.front().device, soc, now_us);
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
    static_assert(kSamplerSize == 1024, "ring drain assumes 1 KiB per-core ring");
    static_assert(
        sizeof(ttnvtop_ring::Header) + ttnvtop_ring::kRingSize * sizeof(ttnvtop_ring::Entry) == kSamplerSize,
        "ring layout mismatch with util_sampler.h");

    RegistryWriter registry_writer;
    registry_writer.open_or_attach();  // best-effort; re-tried inside the loop if the file isn't there yet.

    auto ring_drain = [&]() {
        const auto period = std::chrono::microseconds(1'000'000ull / kRingDrainHz);
        auto next = std::chrono::steady_clock::now();
        std::vector<uint8_t> buf(kSamplerSize);
        uint64_t reattach_throttle_us = 0;
        while (!g_stop.load(std::memory_order_relaxed)) {
            const auto loop_t0 = std::chrono::steady_clock::now();
            const uint64_t now_us =
                std::chrono::duration_cast<std::chrono::microseconds>(loop_t0.time_since_epoch()).count();

            // Lazy re-attach to the registry if the workload hadn't started
            // yet at collector launch. Probe at most once per 2 s to avoid
            // syscall churn.
            if (!registry_writer.enabled && now_us > reattach_throttle_us) {
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
            static uint64_t last_period_assert_us = 0;
            const bool reassert_period = (now_us - last_period_assert_us) > 1'000'000;
            if (reassert_period) {
                last_period_assert_us = now_us;
            }
            // Phase 2.1.c.i diagnostic: every 5 s, read back period_cycles
            // from one core after writing it, to verify the override is
            // sticking. If the read-back value != override, something else
            // (firmware re-init, kernel-side write) is racing us.
            static uint64_t last_period_probe_us = 0;
            const bool probe_period = (now_us - last_period_probe_us) > 5'000'000;
            if (probe_period) {
                last_period_probe_us = now_us;
            }

            for (auto& chip : chips) {
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
                        dev->read_from_device(jbuf.data(), core, ttnvtop_agg::kLandingBase, 64);
                        dev->read_from_device(jbuf.data(), core, ttnvtop_agg::kLandingBase + 64, 8192);
                        chip.journal_entries_seen += 2;
                    } catch (...) {
                        ++chip.journal_stale_ticks;
                    }
                    continue;
                }

                int probe_good = 0, probe_bad = 0;
                for (auto& c : chip.cores) {
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
                    registry_writer.update_kernel_cycles(rid, cyc_window, cyc_total);
                }

                // Periodic debug log: once per 5 s, dump tick + lost +
                // entries + map size per chip. Cheap, useful when triaging
                // "TIME% is stuck at 0".
                static thread_local uint64_t last_debug_us = 0;
                static thread_local uint64_t last_debug_ticks = 0;
                if (now_us - last_debug_us > 5'000'000) {
                    const uint64_t dt_us = (last_debug_us == 0) ? 1 : (now_us - last_debug_us);
                    const double drain_hz =
                        static_cast<double>(chip.drain_ticks - last_debug_ticks) * 1'000'000.0 / dt_us;
                    last_debug_us = now_us;
                    last_debug_ticks = chip.drain_ticks;
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
    std::thread ring_drain_thread(ring_drain);

    // Publisher: copy rolling stats into SHM at the configured rate.
    const auto publish_period = std::chrono::milliseconds(1000 / cli.publish_hz);
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(publish_period);
        for (auto& chip : chips) {
            auto* views = chip.publisher.cores();
            if (views == nullptr) {
                continue;
            }
            // Refresh AICLK from ARC telemetry. get_clock() is non-blocking
            // (telemetry read on BH, legacy ARC message on WH). Failure -> 0.
            if (auto* header = chip.publisher.header(); header != nullptr) {
                uint32_t aiclk_mhz = 0;
                if (chip.cores.empty() == false) {
                    try {
                        aiclk_mhz = chip.cores.front().device->get_clock();
                    } catch (const std::exception&) {
                    }
                }
                header->aiclk_mhz = aiclk_mhz;
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
                v.compute_busy_p1000 = static_cast<uint16_t>(fb * 1000.0);
                v.sfpu_busy_p1000 = static_cast<uint16_t>(sb * 1000.0);
                v.last_kernel_id = c.last_kernel_id;
                // Phase 2+ fields stay at their zero-initialized values.
            }
            chip.publisher.mark_updated();
        }
    }

    sampler_thread.join();
    ring_drain_thread.join();
    std::cout << "\nttnvtop-collector: exiting.\n";
    return 0;
}
