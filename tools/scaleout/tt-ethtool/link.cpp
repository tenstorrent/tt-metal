// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/tt-ethtool/commands.hpp"
#include "tools/scaleout/tt-ethtool/eth_fw.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include <fmt/core.h>

#include "umd/device/arch/architecture_implementation.hpp"
#include "umd/device/cluster.hpp"
#include "umd/device/cluster_descriptor.hpp"
#include "umd/device/soc_descriptor.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt_ethtool {

namespace {

constexpr auto kMailboxPollTimeout = std::chrono::seconds(10);
constexpr auto kRetrainPollTimeout = std::chrono::seconds(10);
constexpr auto kPollInterval = std::chrono::milliseconds(1);

int parse_int(std::string_view s) {
    int value = 0;
    auto* begin = s.data();
    auto* end = s.data() + s.size();
    auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(fmt::format("'{}' is not a valid integer", s));
    }
    return value;
}

tt::umd::CoreCoord get_eth_core(const tt::umd::Cluster& cluster, tt::ChipId chip_id, int channel) {
    const auto& soc_desc = cluster.get_soc_descriptor(chip_id);
    return soc_desc.get_eth_core_for_channel(static_cast<std::uint32_t>(channel), tt::CoordSystem::NOC0);
}

std::uint32_t read_u32(tt::umd::Cluster& cluster, tt::ChipId chip_id, tt::umd::CoreCoord core, std::uint64_t addr) {
    std::uint32_t value = 0;
    cluster.read_from_device(&value, chip_id, core, addr, sizeof(value));
    return value;
}

void write_u32(
    tt::umd::Cluster& cluster, tt::ChipId chip_id, tt::umd::CoreCoord core, std::uint64_t addr, std::uint32_t value) {
    cluster.write_to_device(&value, sizeof(value), chip_id, core, addr);
}

// ---------------- Wormhole retrain flow ----------------

void wormhole_trigger_retrain(tt::umd::Cluster& cluster, tt::ChipId chip_id, int channel, tt::umd::CoreCoord core) {
    namespace wh = eth_fw::wormhole;

    write_u32(cluster, chip_id, core, wh::ETH_RETRAIN_ADDR, wh::ETH_TRIGGER_RETRAIN_VAL);

    const auto deadline = std::chrono::steady_clock::now() + kRetrainPollTimeout;
    while (read_u32(cluster, chip_id, core, wh::ETH_RETRAIN_ADDR) != 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error(fmt::format(
                "Timed out waiting for ETH FW to acknowledge retrain on chip {} channel {}", chip_id, channel));
        }
        std::this_thread::sleep_for(kPollInterval);
    }
}

// ---------------- Blackhole mailbox flow ----------------

// Wait for the host mailbox slot to be empty (value 0) or signal DONE.
void bh_mailbox_wait_ready(tt::umd::Cluster& cluster, tt::ChipId chip_id, int channel, tt::umd::CoreCoord core) {
    namespace bh = eth_fw::blackhole;

    const auto deadline = std::chrono::steady_clock::now() + kMailboxPollTimeout;
    while (true) {
        const std::uint32_t msg = read_u32(cluster, chip_id, core, bh::ETH_MAILBOX_HOST_MSG_ADDR);
        const std::uint32_t status = msg & bh::ETH_MSG_STATUS_MASK;
        if (status == 0 || status == bh::ETH_MSG_DONE) {
            return;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error(fmt::format(
                "Timed out waiting for ETH FW mailbox (status=0x{:08x}) on chip {} channel {}", msg, chip_id, channel));
        }
        std::this_thread::sleep_for(kPollInterval);
    }
}

void bh_send_mailbox_msg(
    tt::umd::Cluster& cluster,
    tt::ChipId chip_id,
    int channel,
    tt::umd::CoreCoord core,
    std::uint32_t msg_type,
    std::array<std::uint32_t, eth_fw::blackhole::ETH_MAILBOX_NUM_ARGS> args,
    bool wait_for_done) {
    namespace bh = eth_fw::blackhole;

    bh_mailbox_wait_ready(cluster, chip_id, channel, core);

    // Write args first, then the CALL word last to avoid the FW observing a
    // partially populated command.
    for (std::uint32_t i = 0; i < bh::ETH_MAILBOX_NUM_ARGS; ++i) {
        const std::uint64_t addr = bh::ETH_MAILBOX_HOST_ARG0_ADDR + i * sizeof(std::uint32_t);
        write_u32(cluster, chip_id, core, addr, args[i]);
    }

    const std::uint32_t cmd = bh::ETH_MSG_CALL | msg_type;
    write_u32(cluster, chip_id, core, bh::ETH_MAILBOX_HOST_MSG_ADDR, cmd);

    if (wait_for_done) {
        bh_mailbox_wait_ready(cluster, chip_id, channel, core);
    }
}

// ---------------- Dispatch ----------------

enum class LinkAction {
    UP,
    DOWN,
};

constexpr std::string_view to_str(LinkAction action) { return action == LinkAction::UP ? "up" : "down"; }

int run_link_action(LinkRef link, LinkAction action) {
    // Validate link ref against the cluster descriptor before constructing the
    // heavier Cluster object (which performs full device init).
    {
        auto desc = tt::umd::Cluster::create_cluster_descriptor();
        const auto& all_chips = desc->get_all_chips();
        if (!all_chips.contains(link.chip_id)) {
            std::cerr << fmt::format("Chip {} not found on this host.\n", link.chip_id);
            return EXIT_FAILURE;
        }
        const tt::ARCH arch = desc->get_arch(link.chip_id);
        const std::uint32_t num_eth = tt::umd::architecture_implementation::create(arch)->get_num_eth_channels();
        if (link.channel < 0 || static_cast<std::uint32_t>(link.channel) >= num_eth) {
            std::cerr << fmt::format(
                "Channel {} out of range for chip {} ({}; has {} channels).\n",
                link.channel,
                link.chip_id,
                tt::arch_to_str(arch),
                num_eth);
            return EXIT_FAILURE;
        }
    }

    tt::umd::Cluster cluster;
    const tt::ARCH arch = cluster.get_cluster_description()->get_arch(link.chip_id);
    const tt::umd::CoreCoord core = get_eth_core(cluster, link.chip_id, link.channel);

    std::cout << fmt::format(
        "Bringing chip {} channel {} {} (arch={}, NOC0=({},{}))\n",
        link.chip_id,
        link.channel,
        to_str(action),
        tt::arch_to_str(arch),
        core.x,
        core.y);

    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: {
            if (action == LinkAction::DOWN) {
                std::cerr << "Wormhole ETH FW does not expose a runtime link-down; "
                             "only 'link up' (retrain) is supported.\n";
                return EXIT_FAILURE;
            }
            wormhole_trigger_retrain(cluster, link.chip_id, link.channel, core);
            break;
        }
        case tt::ARCH::BLACKHOLE: {
            namespace bh = eth_fw::blackhole;
            const std::uint32_t arg0 =
                (action == LinkAction::UP) ? bh::ETH_PORT_ACTION_LINK_UP : bh::ETH_PORT_ACTION_LINK_DOWN;
            bh_send_mailbox_msg(
                cluster,
                link.chip_id,
                link.channel,
                core,
                bh::ETH_MSG_TYPE_PORT_ACTION,
                {arg0, 0, 0},
                /*wait_for_done=*/true);
            break;
        }
        default:
            std::cerr << fmt::format("Unsupported architecture for link control: {}\n", tt::arch_to_str(arch));
            return EXIT_FAILURE;
    }

    std::cout << fmt::format(
        "Chip {} channel {}: link {} request acknowledged by ETH FW.\n", link.chip_id, link.channel, to_str(action));
    return EXIT_SUCCESS;
}

}  // namespace

LinkRef parse_link_ref(std::string_view input) {
    const auto colon = input.find(':');
    if (colon == std::string_view::npos) {
        throw std::invalid_argument(fmt::format("link spec '{}' must be in the form <chip>:<channel>", input));
    }
    LinkRef ref{
        .chip_id = parse_int(input.substr(0, colon)),
        .channel = parse_int(input.substr(colon + 1)),
    };
    return ref;
}

int run_link_up(LinkRef link) { return run_link_action(link, LinkAction::UP); }

int run_link_down(LinkRef link) { return run_link_action(link, LinkAction::DOWN); }

}  // namespace tt_ethtool
