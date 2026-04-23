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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
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

constexpr int kSampleHz = 100;
constexpr int kWindowSamples = 100;  // rolling window for dispatch occupancy
constexpr int kPublishHz = 10;       // how often we write SHM

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

}  // namespace

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
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
                ttnvtop::SIGNAL_SRC_DISPATCH,
                static_cast<uint32_t>(chip.cores.size()))) {
            std::cerr << "ttnvtop-collector: failed to open /dev/shm for asic " << chip.asic_id << " (errno " << errno
                      << ").\n";
            continue;
        }
        // Populate static per-core fields once.
        for (size_t i = 0; i < chip.cores.size(); ++i) {
            auto& v = chip.publisher.cores()[i];
            v.noc_x = static_cast<uint8_t>(chip.cores[i].noc_x);
            v.noc_y = static_cast<uint8_t>(chip.cores[i].noc_y);
            v.logical_x = 0;  // Phase 1: logical coords not yet wired
            v.logical_y = 0;
            v.is_remote = chip.is_remote ? 1u : 0u;
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

    // Sampling thread: one PCIe block read per core per tick. Kept as a single
    // thread for simplicity; per-chip threads are a Phase 3 optimization.
    auto sampler = [&]() {
        std::vector<uint8_t> buf(kReadSize);
        const auto period = std::chrono::microseconds(1'000'000 / kSampleHz);
        auto next = std::chrono::steady_clock::now();
        while (!g_stop.load(std::memory_order_relaxed)) {
            for (auto& chip : chips) {
                for (auto& c : chip.cores) {
                    try {
                        c.device->read_from_device(buf.data(), c.translated, read_addr, kReadSize);
                    } catch (const std::exception&) {
                        continue;
                    }
                    uint32_t idx = 0;
                    std::memcpy(&idx, buf.data() + kIdxOffInBuf, sizeof(idx));
                    if (idx >= go_message_num_entries) {
                        idx = 0;
                    }
                    uint32_t go_word = 0;
                    std::memcpy(&go_word, buf.data() + idx * sizeof(uint32_t), sizeof(go_word));
                    const uint8_t signal = static_cast<uint8_t>((go_word >> 24) & 0xFFu);
                    const uint8_t now_bit = (signal == static_cast<uint8_t>(RUN_MSG_GO)) ? 1 : 0;
                    {
                        std::lock_guard<std::mutex> lk(state_mx);
                        const uint8_t was = c.samples[c.head];
                        c.samples[c.head] = now_bit;
                        c.head = (c.head + 1) % kWindowSamples;
                        c.busy_count = c.busy_count + now_bit - was;
                        c.samples_seen += 1;
                        c.last_dispatched = now_bit;
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

    // Publisher: copy rolling stats into SHM at kPublishHz.
    const auto publish_period = std::chrono::milliseconds(1000 / kPublishHz);
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
                // Phase 2+ fields stay at their zero-initialized values.
            }
            chip.publisher.mark_updated();
        }
    }

    sampler_thread.join();
    std::cout << "\nttnvtop-collector: exiting.\n";
    return 0;
}
