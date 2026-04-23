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
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

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
#include "hostdev/dev_msgs.h"
using namespace ttnvtop_wh_tensix;  // NOLINT(google-build-using-namespace)

#include "shm_publisher.hpp"

using tt::CoordSystem;
using tt::CoreType;
using tt::umd::CoreCoord;
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

// Perf-counter control registers (per tt_metal/tools/profiler/perf_counters.hpp):
//   FPU0 @ 0x018: reference period (unused in continuous mode)
//   FPU1 @ 0x01C: [7:0] mode (0 = continuous), [12:8] bank select (0 = FPU_COUNTER),
//                 [16] output H selector (0 = req_cnt, 1 = grant_cnt)
//   FPU2 @ 0x020: [0] start, [1] stop. Rising edge on [0] clears + starts the counter.
constexpr uint64_t kRegFpu0 = kRiscvDebugBase + 0x018;
constexpr uint64_t kRegFpu1 = kRiscvDebugBase + 0x01C;
constexpr uint64_t kRegFpu2 = kRiscvDebugBase + 0x020;

// Data window we read every tick: FPU_OUT_L/H (0x120, 0x124) and
// WALL_CLOCK_L/H (0x1F0, 0x1F8). One NOC read covers 220 bytes including gaps.
//
// FPU_OUT_L and FPU_OUT_H are TWO INDEPENDENT 32-bit counters in the FPU
// counter group — not the low/high halves of one 64-bit value:
//   OUT_L = ref_cnt (cycles counter was armed)
//   OUT_H = req_cnt (FPU request cycles) when FPU1[16]=0
// We read OUT_H as "FPU busy" and use WALL_CLOCK as the denominator.
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

// Phase 2.1 on-chip sampler. When brisc is running the Phase 2.1 firmware,
// it writes a ring of perf-counter snapshots into mailboxes->util_sampler
// every ~100 us. We detect presence by reading the magic field at startup;
// if present, we pull the ring instead of polling the debug registers — one
// PCIe read per tick instead of four, and the samples were taken on-chip at
// a rate independent of our host poll rate.
constexpr uint32_t kUtilSamplerMagic = 0x53555454u;  // 'TTUS' little-endian
constexpr uint32_t kUtilSamplerRingSize = 64;
struct UtilSamplerEntry {
    uint32_t wall_clock_l;
    uint32_t wall_clock_h;
    uint32_t fpu_out_l;
    uint32_t fpu_out_h;
};
static_assert(sizeof(UtilSamplerEntry) == 16);
struct UtilSamplerHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t head;
    uint32_t period_cycles;
};
static_assert(sizeof(UtilSamplerHeader) == 16);
constexpr size_t kUtilSamplerTotal = sizeof(UtilSamplerHeader) + kUtilSamplerRingSize * sizeof(UtilSamplerEntry);

// EWMA smoothing for compute_busy. alpha = 2/(N+1) for N-sample horizon.
// N=10 gives a ~1s smoothing window at 10 Hz publish rate.
constexpr double kComputeEwmaAlpha = 2.0 / (10.0 + 1.0);

struct CliOptions {
    int sample_hz = kDefaultSampleHz;
    int publish_hz = kDefaultPublishHz;
    std::set<int> device_filter;  // empty = all chips
    std::string log_file;         // non-empty = redirect stderr there
    bool show_help = false;
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
    bool perf_primed = false;
    uint64_t last_wall_clock = 0;
    uint32_t last_fpu_out_h = 0;
    double compute_busy_ewma = 0.0;  // [0..1]

    // Phase 2.1 on-chip sampler state.
    bool ring_available = false;     // firmware has Phase 2.1 hook active
    uint64_t util_sampler_addr = 0;  // L1 address of mailboxes->util_sampler
    uint32_t last_ring_head = 0;     // last head index we read from the ring
};

struct ChipState {
    uint64_t asic_id = 0;
    tt::ChipId chip_id = 0;
    tt::ARCH arch = tt::ARCH::Invalid;
    bool is_remote = false;
    std::vector<CoreState> cores;
    ttnvtop::ShmPublisher publisher;
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

// Arm the FPU counter on a single Tensix for free-running continuous mode
// with bank=FPU_COUNTER. Rising edge on FPU2[0] both clears and starts.
// Returns true on success; false if any of the four writes threw. A user
// kernel that calls StartPerfCounters later will fight with this (transient
// reset + new start), which our sampler's negative-delta guard tolerates.
bool arm_fpu_counter(TTDevice* device, const tt_xy_pair& core) {
    const auto write32 = [&](uint64_t addr, uint32_t value) {
        device->write_to_device(&value, core, addr, sizeof(value));
    };
    try {
        write32(kRegFpu0, 0u);  // reference period (unused in continuous)
        write32(kRegFpu1, 0u);  // mode=0 continuous, bank=FPU_COUNTER, H=req_cnt
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

        SocDescriptor soc(arch, dev->get_chip_info());
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
        // Also probe for Phase 2.1 firmware: if mailboxes->util_sampler.magic
        // is 'TTUS' we switch to ring-based sampling for that core.
        constexpr uint64_t kMailboxBaseAddr = static_cast<uint64_t>(MEM_MAILBOX_BASE);
        constexpr size_t kUtilSamplerOffset = offsetof(mailboxes_t, util_sampler);
        int armed = 0;
        int rings_live = 0;
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
            // Probe for Phase 2.1 sampler magic. If firmware isn't the
            // Phase 2.1 build, magic reads as stale L1 garbage / 0 and we
            // fall back to direct-register polling on this core.
            chip.cores[i].util_sampler_addr = kMailboxBaseAddr + kUtilSamplerOffset;
            uint32_t probe_magic = 0;
            try {
                chip.cores[i].device->read_from_device(
                    &probe_magic, chip.cores[i].translated, chip.cores[i].util_sampler_addr, sizeof(probe_magic));
            } catch (const std::exception&) {
            }
            if (probe_magic == kUtilSamplerMagic) {
                chip.cores[i].ring_available = true;
                ++rings_live;
            }
        }
        std::cerr << "ttnvtop-collector: chip " << chip.chip_id << " armed FPU counter on " << armed << "/"
                  << chip.cores.size() << " cores; Phase 2.1 ring detected on " << rings_live << "/"
                  << chip.cores.size() << " cores.\n";

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

                    // Perf counters. Two acquisition paths:
                    //   (A) Phase 2.1 ring: brisc firmware has been writing
                    //       samples to mailboxes->util_sampler.ring. We read
                    //       the whole region (1056 B) in one L1 block read,
                    //       extract the most recent entry, and use that as
                    //       "now". One PCIe read per core instead of four.
                    //   (B) Phase 2.0 fallback: old firmware / no ring.
                    //       Direct register reads of OUT_L, OUT_H, WALL_L,
                    //       WALL_H. Four small reads per core.
                    bool have_perf = false;
                    uint32_t fpu_out_l_now = 0;
                    uint32_t fpu_out_h_now = 0;
                    uint64_t wall_now = 0;
                    // Probe magic each tick so the collector self-heals: if a
                    // workload loads Phase 2.1 firmware after the collector
                    // starts, the ring pops into existence and we switch
                    // seamlessly.
                    uint32_t probe_magic = 0;
                    try {
                        c.device->read_from_device(
                            &probe_magic, c.translated, c.util_sampler_addr, sizeof(probe_magic));
                    } catch (const std::exception&) {
                    }
                    const bool ring_live = (probe_magic == kUtilSamplerMagic);
                    c.ring_available = ring_live;
                    if (ring_live) {
                        uint8_t ring_buf[kUtilSamplerTotal];
                        try {
                            c.device->read_from_device(ring_buf, c.translated, c.util_sampler_addr, kUtilSamplerTotal);
                            UtilSamplerHeader hdr;
                            std::memcpy(&hdr, ring_buf, sizeof(hdr));
                            if (hdr.magic == kUtilSamplerMagic && hdr.head > 0) {
                                const uint32_t slot = (hdr.head - 1) & (kUtilSamplerRingSize - 1);
                                UtilSamplerEntry e;
                                std::memcpy(
                                    &e,
                                    ring_buf + sizeof(UtilSamplerHeader) + slot * sizeof(UtilSamplerEntry),
                                    sizeof(e));
                                fpu_out_l_now = e.fpu_out_l;
                                fpu_out_h_now = e.fpu_out_h;
                                wall_now = (static_cast<uint64_t>(e.wall_clock_h) << 32) | e.wall_clock_l;
                                have_perf = true;
                            }
                        } catch (const std::exception&) {
                        }
                    }
                    if (!have_perf) {
                        uint32_t wl = 0, wh = 0;
                        try {
                            c.device->read_from_device(
                                &fpu_out_l_now, c.translated, kAddrFpuOutL, sizeof(fpu_out_l_now));
                            c.device->read_from_device(
                                &fpu_out_h_now, c.translated, kAddrFpuOutH, sizeof(fpu_out_h_now));
                            c.device->read_from_device(&wl, c.translated, kAddrWallL, sizeof(wl));
                            c.device->read_from_device(&wh, c.translated, kAddrWallH, sizeof(wh));
                            wall_now = (static_cast<uint64_t>(wh) << 32) | wl;
                            have_perf = true;
                        } catch (const std::exception&) {
                        }
                    }
                    if (have_perf && c.noc_x == 1 && c.noc_y == 1) {
                        // Periodic live probe: dump raw counter values for
                        // core (1,1) on each chip once per second. Also
                        // reports which acquisition path was used so we can
                        // verify the Phase 2.1 ring is wired up when firmware
                        // supports it.
                        static std::atomic<uint64_t> last_probe_us{0};
                        const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now().time_since_epoch())
                                                .count();
                        const uint64_t lp = last_probe_us.load(std::memory_order_relaxed);
                        if (static_cast<uint64_t>(now_us) - lp > 1'000'000) {
                            last_probe_us.store(now_us, std::memory_order_relaxed);
                            std::fprintf(
                                stderr,
                                "[live-probe] core(1,1) is_remote=%d src=%s "
                                "wall=0x%016lx out_l=0x%08x out_h=0x%08x\n",
                                c.is_remote ? 1 : 0,
                                c.ring_available ? "ring" : "regs",
                                wall_now,
                                fpu_out_l_now,
                                fpu_out_h_now);
                            std::fflush(stderr);
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
                        if (have_perf) {
                            if (c.perf_primed) {
                                const uint64_t wall_d = wall_now - c.last_wall_clock;
                                // Estimate FPU-active cycles during this
                                // interval. If OUT_H went backwards, a kernel
                                // called StartPerfCounters somewhere in the
                                // interval; conservatively take OUT_H as the
                                // post-reset portion and discard the pre-reset
                                // portion (unknown).
                                const uint64_t fpu_d = (fpu_out_h_now >= c.last_fpu_out_h)
                                                           ? static_cast<uint64_t>(fpu_out_h_now - c.last_fpu_out_h)
                                                           : static_cast<uint64_t>(fpu_out_h_now);
                                if (wall_d > 0 && wall_d < (1ull << 40) && fpu_d <= wall_d) {
                                    const double inst = static_cast<double>(fpu_d) / static_cast<double>(wall_d);
                                    c.compute_busy_ewma =
                                        kComputeEwmaAlpha * inst + (1.0 - kComputeEwmaAlpha) * c.compute_busy_ewma;
                                }
                            }
                            c.last_wall_clock = wall_now;
                            c.last_fpu_out_h = fpu_out_h_now;
                            c.perf_primed = true;
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
                // Compute busy: EWMA is in [0,1]; clamp and convert to per-mille.
                double cb = c.compute_busy_ewma;
                if (cb < 0.0) {
                    cb = 0.0;
                } else if (cb > 1.0) {
                    cb = 1.0;
                }
                v.compute_busy_p1000 = static_cast<uint16_t>(cb * 1000.0);
                // Phase 2+ fields stay at their zero-initialized values.
            }
            chip.publisher.mark_updated();
        }
    }

    sampler_thread.join();
    std::cout << "\nttnvtop-collector: exiting.\n";
    return 0;
}
