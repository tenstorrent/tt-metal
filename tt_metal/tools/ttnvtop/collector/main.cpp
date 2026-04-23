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
constexpr uint64_t kPerfWindowStart = kRiscvDebugBase + 0x120;
constexpr size_t kPerfWindowSize = 0x1F8 + 4 - 0x120;
constexpr size_t kOffFpuOutH = 0x124 - 0x120;  // 0x04 — "FPU busy" cycles (req_cnt)
constexpr size_t kOffWallL = 0x1F0 - 0x120;    // 0xD0 — low half of 64-bit wall clock
constexpr size_t kOffWallH = 0x1F8 - 0x120;    // 0xD8 — high half

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

    // Phase 2.0 perf-counter state. We arm the FPU counter from host at
    // startup (continuous mode, bank=FPU_COUNTER, rising edge on start) so it
    // free-runs until someone stops it. Tracks a 64-bit wall-clock and the
    // 32-bit FPU_OUT_H counter; we extend that 32-bit value across wraps by
    // tracking previous and adding 2^32 on decrement.
    bool perf_primed = false;
    bool counter_armed = false;
    uint64_t last_wall_clock = 0;
    uint32_t last_fpu_out_h = 0;
    uint64_t fpu_out_h_extended = 0;  // tracks wraps
    double compute_busy_ewma = 0.0;   // [0..1]
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
        std::vector<uint8_t> perf_buf(kPerfWindowSize);
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

                    // Perf counters (Phase 2.0). Free-running; deltas give
                    // busy%. Negative deltas mean the kernel called
                    // StartPerfCounters and reset — skip those ticks.
                    bool have_perf = false;
                    uint64_t wall_now = 0;
                    uint32_t fpu_out_h_now = 0;  // 32-bit FPU busy counter snapshot
                    try {
                        c.device->read_from_device(perf_buf.data(), c.translated, kPerfWindowStart, kPerfWindowSize);
                        have_perf = true;
                    } catch (const std::exception&) {
                    }
                    if (have_perf) {
                        uint32_t wl = 0, wh = 0;
                        std::memcpy(&wl, perf_buf.data() + kOffWallL, sizeof(wl));
                        std::memcpy(&wh, perf_buf.data() + kOffWallH, sizeof(wh));
                        wall_now = (static_cast<uint64_t>(wh) << 32) | wl;
                        std::memcpy(&fpu_out_h_now, perf_buf.data() + kOffFpuOutH, sizeof(fpu_out_h_now));
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
                                // FPU_OUT_H is a 32-bit counter. Compute the
                                // delta via 32-bit arithmetic (handles a single
                                // wrap naturally) then sanity-check against wall
                                // delta. If fpu_d > wall_d we must be seeing a
                                // kernel-side reset (StartPerfCounters) or a
                                // multi-wrap skip; drop that tick and keep old
                                // EWMA (which will decay as we collect valid
                                // samples).
                                const uint64_t fpu_d = static_cast<uint64_t>(fpu_out_h_now - c.last_fpu_out_h);
                                const bool wall_ok = wall_d > 0 && wall_d < (1ull << 40);
                                if (wall_ok && fpu_d <= wall_d) {
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
