// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop: live, ASCII per-Tensix utilization monitor that coexists with a
// running tt-metal workload.
//
// Why this is safe to run alongside a workload:
//   tt-metal's Cluster takes a per-chip robust mutex (CHIP_IN_USE) inside
//   LocalChip::start_device(). Opening devices via umd::TopologyDiscovery
//   builds raw umd::TTDevice handles WITHOUT ever calling start_device(),
//   so this binary never contends for CHIP_IN_USE. Same mechanism tt-mgmt
//   uses for read-only telemetry. Reads go through plain PCIe TLB windows
//   (TTDevice::read_from_device), which are non-destructive to the running
//   workload.
//
// Signal:
//   mailboxes_t.go_messages[go_message_index].signal == RUN_MSG_GO means a
//   kernel is currently dispatched to that worker. We sample that one byte
//   per core at kSampleHz and render a 1s rolling busy%.
//
// Arch scope:
//   PoC compiles against Wormhole Tensix mailbox layout. At runtime we
//   reject chips of any other arch with a clear error. BH/Quasar support is
//   a matter of wiring in the matching dev_mem_map.h + core_config.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// UMD: direct chip access (no tt-metal Cluster, no CHIP_IN_USE lock).
#include "umd/device/soc_descriptor.hpp"
#include "umd/device/topology/topology_discovery.hpp"
#include "umd/device/topology/topology_discovery_options.hpp"
#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"
#include "umd/device/types/xy_pair.hpp"

// Device-shared mailbox layout — Wormhole.
// dev_msgs.h refuses direct host inclusion; HAL_BUILD wraps it in a unique
// namespace (same trick wh_hal_tensix.cpp uses).
#define HAL_BUILD ttnvtop_wh_tensix
#include "hostdev/dev_msgs.h"
#include "dev_mem_map.h"
using namespace ttnvtop_wh_tensix;

using tt::CoordSystem;
using tt::CoreType;
using tt::umd::CoreCoord;
using tt::umd::SocDescriptor;
using tt::umd::TopologyDiscovery;
using tt::umd::TopologyDiscoveryOptions;
using tt::umd::TTDevice;

namespace {

constexpr int kSampleHz = 100;
constexpr int kWindowSamples = 100;
constexpr int kRenderHz = 4;

struct CoreStats {
    uint64_t asic_id;  // chip identity (from TTDevice)
    int chip_ordinal;  // stable index for display
    bool is_remote;
    TTDevice* device;       // borrowed, owned by the devices map
    tt_xy_pair translated;  // coord in TRANSLATED space for read_from_device
    uint32_t noc_x;         // original NOC0 for display
    uint32_t noc_y;
    std::array<uint8_t, kWindowSamples> samples{};
    size_t head = 0;
    uint32_t busy_count = 0;
};

std::atomic<bool> g_stop{false};

void handle_sigint(int) { g_stop.store(true); }

std::string make_bar(uint32_t pct, int width) {
    uint32_t filled = (pct * width) / 100;
    if (filled > static_cast<uint32_t>(width)) {
        filled = static_cast<uint32_t>(width);
    }
    std::string s;
    s.reserve(static_cast<size_t>(width) + 2);
    s.push_back('[');
    for (int i = 0; i < width; ++i) {
        s.push_back(i < static_cast<int>(filled) ? '#' : ' ');
    }
    s.push_back(']');
    return s;
}

const char* arch_name(tt::ARCH a) {
    switch (a) {
        case tt::ARCH::WORMHOLE_B0: return "Wormhole";
        case tt::ARCH::BLACKHOLE: return "Blackhole";
        case tt::ARCH::QUASAR: return "Quasar";
        default: return "Unknown";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    std::signal(SIGINT, handle_sigint);

    // Mailbox layout — all compile-time against WH Tensix layout.
    // MEM_MAILBOX_BASE = 16 on WH Tensix L1 (from dev_mem_map.h).
    constexpr uint64_t kMailboxBase = static_cast<uint64_t>(MEM_MAILBOX_BASE);
    constexpr size_t kGoMessagesOff = offsetof(mailboxes_t, go_messages);
    constexpr size_t kGoIndexOff = offsetof(mailboxes_t, go_message_index);
    static_assert(kGoIndexOff > kGoMessagesOff, "unexpected mailboxes_t layout");
    constexpr size_t kReadSize = (kGoIndexOff + sizeof(uint32_t)) - kGoMessagesOff;
    constexpr size_t kIdxOffInBuf = kGoIndexOff - kGoMessagesOff;
    static_assert(sizeof(go_msg_t) == sizeof(uint32_t), "go_msg_t must be 4 bytes");
    const uint64_t read_addr = kMailboxBase + static_cast<uint64_t>(kGoMessagesOff);

    // Discover every chip in the system (local + remote) WITHOUT opening
    // umd::Cluster — so we never take the CHIP_IN_USE robust mutex.
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
        std::cerr << "ttnvtop: no Tenstorrent devices discovered.\n";
        return 1;
    }

    // Build per-core sampling plan.
    std::vector<CoreStats> stats;
    std::vector<std::pair<int, tt::ChipId>> chip_summary;
    int ordinal = 0;
    for (auto& [chip_id, dev_up] : devices) {
        TTDevice* dev = dev_up.get();
        const tt::ARCH arch = dev->get_arch();
        if (arch != tt::ARCH::WORMHOLE_B0) {
            std::cerr << "ttnvtop: chip " << ordinal << " arch " << arch_name(arch)
                      << " not supported by this PoC (Wormhole-only). Skipping.\n";
            ++ordinal;
            continue;
        }
        const bool remote = dev->is_remote();
        const uint64_t asic_id = static_cast<uint64_t>(chip_id);
        chip_summary.emplace_back(ordinal, chip_id);

        SocDescriptor soc(arch, dev->get_chip_info());
        auto worker_cores_noc0 = soc.get_cores(CoreType::TENSIX, CoordSystem::NOC0);

        for (const auto& cc_noc0 : worker_cores_noc0) {
            auto cc_trans = soc.translate_coord_to(cc_noc0, CoordSystem::TRANSLATED);
            CoreStats s;
            s.asic_id = asic_id;
            s.chip_ordinal = ordinal;
            s.is_remote = remote;
            s.device = dev;
            s.translated = tt_xy_pair(cc_trans.x, cc_trans.y);
            s.noc_x = cc_noc0.x;
            s.noc_y = cc_noc0.y;
            stats.push_back(std::move(s));
        }
        ++ordinal;
    }

    if (stats.empty()) {
        std::cerr << "ttnvtop: no supported (Wormhole) worker cores to monitor.\n";
        return 1;
    }

    // Stable display order: chip ordinal, then (y,x).
    std::sort(stats.begin(), stats.end(), [](const CoreStats& a, const CoreStats& b) {
        if (a.chip_ordinal != b.chip_ordinal) {
            return a.chip_ordinal < b.chip_ordinal;
        }
        if (a.noc_y != b.noc_y) {
            return a.noc_y < b.noc_y;
        }
        return a.noc_x < b.noc_x;
    });

    std::cout << "ttnvtop: " << chip_summary.size() << " chip(s), " << stats.size() << " worker cores. Sampling "
              << kSampleHz << " Hz, window " << kWindowSamples << " samples. Ctrl-C to exit.\n";

    std::mutex stats_mx;

    // Sampling thread — one TTDevice::read_from_device per core per tick.
    auto sampler = [&]() {
        std::vector<uint8_t> buf(kReadSize);
        const auto period = std::chrono::microseconds(1'000'000 / kSampleHz);
        auto next = std::chrono::steady_clock::now();
        while (!g_stop.load(std::memory_order_relaxed)) {
            for (auto& s : stats) {
                try {
                    s.device->read_from_device(buf.data(), s.translated, read_addr, kReadSize);
                } catch (const std::exception&) {
                    // Transient error (e.g., chip momentarily unreachable). Skip this tick.
                    continue;
                }
                uint32_t idx = 0;
                std::memcpy(&idx, buf.data() + kIdxOffInBuf, sizeof(idx));
                if (idx >= go_message_num_entries) {
                    idx = 0;
                }
                uint32_t go_word = 0;
                std::memcpy(&go_word, buf.data() + idx * sizeof(uint32_t), sizeof(go_word));
                // go_msg_t::signal is the top byte of the uint32_t union.
                const uint8_t signal = static_cast<uint8_t>((go_word >> 24) & 0xFFu);
                const uint8_t now = (signal == static_cast<uint8_t>(RUN_MSG_GO)) ? 1 : 0;
                {
                    std::lock_guard<std::mutex> lk(stats_mx);
                    const uint8_t was = s.samples[s.head];
                    s.samples[s.head] = now;
                    s.head = (s.head + 1) % kWindowSamples;
                    s.busy_count = s.busy_count + now - was;
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

    // Group stats pointers per chip_ordinal for side-by-side rendering.
    std::vector<std::vector<const CoreStats*>> per_chip(chip_summary.size());
    {
        std::unordered_map<int, size_t> ord_to_idx;
        for (size_t i = 0; i < chip_summary.size(); ++i) {
            ord_to_idx[chip_summary[i].first] = i;
        }
        for (const auto& s : stats) {
            auto it = ord_to_idx.find(s.chip_ordinal);
            if (it != ord_to_idx.end()) {
                per_chip[it->second].push_back(&s);
            }
        }
    }

    // Renderer: two-column layout when we have >= 2 chips.
    const int kCorePanelWidth = 40;  // "( x, y)  [bar]  pct%"
    const auto render_period = std::chrono::milliseconds(1000 / kRenderHz);
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(render_period);

        // Snapshot busy counts once per render to keep column alignment consistent.
        std::vector<std::vector<uint32_t>> snapshot(per_chip.size());
        {
            std::lock_guard<std::mutex> lk(stats_mx);
            for (size_t c = 0; c < per_chip.size(); ++c) {
                snapshot[c].reserve(per_chip[c].size());
                for (const auto* s : per_chip[c]) {
                    snapshot[c].push_back(s->busy_count);
                }
            }
        }

        std::ostringstream out;
        out << "\x1b[H\x1b[2J";  // ANSI home + clear
        out << "ttnvtop  |  chips=" << chip_summary.size() << "  cores=" << stats.size() << "  sample=" << kSampleHz
            << "Hz  window=" << kWindowSamples << " samples (" << (kWindowSamples / kSampleHz) << "s)"
            << "  (coexists with running workloads)\n";
        out << "\n";
        out << "What this measures:\n";
        out << "  Per Tensix worker core, each tick we read 1 byte from L1 mailbox:\n";
        out << "    mailboxes_t.go_messages[go_message_index].signal\n";
        out << "  The dispatcher writes RUN_MSG_GO (0x80) to this byte when it\n";
        out << "  dispatches a kernel to the core, RUN_MSG_DONE (0x00) when the\n";
        out << "  kernel finishes. We sample at " << kSampleHz << " Hz per core and show the\n";
        out << "  fraction of the last " << kWindowSamples << " samples (~1s) observed as GO.\n";
        out << "\n";
        out << "  100% = core had a kernel dispatched for the full window.\n";
        out << "    0% = core was idle (RUN_MSG_DONE or still in RUN_MSG_INIT).\n";
        out << "  Caveat: this is dispatch-occupancy, not compute busy%. A core\n";
        out << "  that has a kernel dispatched but is stalled on NOC/CB will still\n";
        out << "  read 100%. True FPU/PACK/UNPACK busy% would need perf counters.\n";
        out << "\n";

        // Header: two chip titles side-by-side.
        for (size_t c = 0; c < per_chip.size(); ++c) {
            const bool remote = !per_chip[c].empty() && per_chip[c].front()->is_remote;
            std::ostringstream title;
            title << "chip " << chip_summary[c].first << "  asic 0x" << std::hex
                  << (per_chip[c].empty() ? 0 : per_chip[c].front()->asic_id) << std::dec << "  ["
                  << (remote ? "remote" : " mmio ") << "]";
            std::string t = title.str();
            if (static_cast<int>(t.size()) < kCorePanelWidth) {
                t.append(kCorePanelWidth - t.size(), ' ');
            }
            out << t;
            if (c + 1 < per_chip.size()) {
                out << " | ";
            }
        }
        out << "\n";
        for (size_t c = 0; c < per_chip.size(); ++c) {
            out << std::string(kCorePanelWidth, '-');
            if (c + 1 < per_chip.size()) {
                out << "-+-";
            }
        }
        out << "\n";

        // Row by row: one line per core index across all chips.
        size_t max_rows = 0;
        for (const auto& v : per_chip) {
            max_rows = std::max(max_rows, v.size());
        }
        for (size_t row = 0; row < max_rows; ++row) {
            for (size_t c = 0; c < per_chip.size(); ++c) {
                std::string cell;
                if (row < per_chip[c].size()) {
                    const CoreStats* s = per_chip[c][row];
                    const uint32_t busy = snapshot[c][row];
                    const uint32_t pct = (busy * 100u) / kWindowSamples;
                    std::ostringstream line;
                    line << "(" << std::setw(2) << s->noc_x << "," << std::setw(2) << s->noc_y << ")  "
                         << make_bar(pct, 22) << " " << std::setw(3) << pct << "%";
                    cell = line.str();
                } else {
                    cell = "";
                }
                if (static_cast<int>(cell.size()) < kCorePanelWidth) {
                    cell.append(kCorePanelWidth - cell.size(), ' ');
                }
                out << cell;
                if (c + 1 < per_chip.size()) {
                    out << " | ";
                }
            }
            out << "\n";
        }

        // Per-chip average footer.
        for (size_t c = 0; c < per_chip.size(); ++c) {
            out << std::string(kCorePanelWidth, '-');
            if (c + 1 < per_chip.size()) {
                out << "-+-";
            }
        }
        out << "\n";
        for (size_t c = 0; c < per_chip.size(); ++c) {
            uint32_t sum = 0;
            for (uint32_t b : snapshot[c]) {
                sum += b;
            }
            const uint32_t avg =
                snapshot[c].empty() ? 0u : (sum * 100u) / (static_cast<uint32_t>(snapshot[c].size()) * kWindowSamples);
            std::ostringstream f;
            f << "avg " << std::setw(3) << avg << "%   (" << snapshot[c].size() << " cores)";
            std::string t = f.str();
            if (static_cast<int>(t.size()) < kCorePanelWidth) {
                t.append(kCorePanelWidth - t.size(), ' ');
            }
            out << t;
            if (c + 1 < per_chip.size()) {
                out << " | ";
            }
        }
        out << "\n";

        out << "\n[Ctrl-C to exit]\n";
        std::cout << out.str();
        std::cout.flush();
    }

    sampler_thread.join();
    std::cout << "\nttnvtop: exiting.\n";
    return 0;
}
