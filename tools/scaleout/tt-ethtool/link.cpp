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
    //return soc_desc.get_eth_core_for_channel(static_cast<std::uint32_t>(channel), tt::CoordSystem::NOC0);
    return soc_desc.get_eth_core_for_channel(static_cast<std::uint32_t>(channel), tt::CoordSystem::LOGICAL);
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
        if (status == 0) {
            std::cout << fmt::format(
                "ETH FW mailbox (status=0x{:08x}) on chip {} channel {} is empty\n", msg, chip_id, channel);
            return;
        }
        if (status == bh::ETH_MSG_DONE) {
            std::cout << fmt::format(
                "ETH FW mailbox (status=0x{:08x}) on chip {} channel {} is done\n", msg, chip_id, channel);
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

std::array<std::uint32_t, eth_fw::blackhole::ETH_MAILBOX_NUM_ARGS> bh_read_mailbox_args(
    tt::umd::Cluster& cluster, tt::ChipId chip_id, tt::umd::CoreCoord core) {
    namespace bh = eth_fw::blackhole;
    std::array<std::uint32_t, bh::ETH_MAILBOX_NUM_ARGS> args{};
    for (std::uint32_t i = 0; i < bh::ETH_MAILBOX_NUM_ARGS; ++i) {
        const std::uint64_t addr = bh::ETH_MAILBOX_HOST_ARG0_ADDR + i * sizeof(std::uint32_t);
        args[i] = read_u32(cluster, chip_id, core, addr);
    }
    return args;
}

// ---------------- Dispatch ----------------

enum class LinkAction {
    UP,
    DOWN,
};

constexpr std::string_view to_str(LinkAction action) { return action == LinkAction::UP ? "up" : "down"; }

constexpr std::string_view bh_port_status_to_str(std::uint32_t status) {
    namespace bh = eth_fw::blackhole;
    switch (status) {
        case bh::PORT_STATUS_UNKNOWN: return "PORT_UNKNOWN";
        case bh::PORT_STATUS_UP: return "PORT_UP";
        case bh::PORT_STATUS_DOWN: return "PORT_DOWN";
        case bh::PORT_STATUS_UNUSED: return "PORT_UNUSED";
        default: return "?";
    }
}

constexpr std::string_view bh_reinit_option_to_str(std::uint32_t option) {
    namespace bh = eth_fw::blackhole;
    switch (option) {
        case bh::ETH_PORT_REINIT_OPT_MAC_ONLY: return "mac-only";
        case bh::ETH_PORT_REINIT_OPT_MAC_SERDES_RETRAIN: return "mac+serdes-retrain";
        case bh::ETH_PORT_REINIT_OPT_MAC_SERDES: return "mac+serdes-reset";
        case bh::ETH_PORT_REINIT_OPT_MAC_SERDES_TX_BARRIER: return "mac+serdes-reset-tx-barrier";
        default: return "?";
    }
}

constexpr std::string_view wh_train_status_to_str(std::uint32_t status) {
    namespace wh = eth_fw::wormhole;
    switch (status) {
        case wh::TRAIN_STATUS_IN_PROGRESS: return "IN_PROGRESS";
        case wh::TRAIN_STATUS_SUCCESS: return "SUCCESS";
        case wh::TRAIN_STATUS_FAIL: return "FAIL";
        case wh::TRAIN_STATUS_NOT_CONNECTED: return "NOT_CONNECTED";
        default: return "?";
    }
}

// Validate link ref against the cluster descriptor before constructing the
// heavier Cluster object (which performs full device init). Returns true on
// success; prints an error and returns false otherwise.
bool validate_link_ref(LinkRef link) {
    auto desc = tt::umd::Cluster::create_cluster_descriptor();
    const auto& all_chips = desc->get_all_chips();
    if (!all_chips.contains(link.chip_id)) {
        std::cerr << fmt::format("Chip {} not found on this host.\n", link.chip_id);
        return false;
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
        return false;
    }
    return true;
}

int run_link_action(LinkRef link, LinkAction action) {
    if (!validate_link_ref(link)) {
        return EXIT_FAILURE;
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

int run_link_status_action(LinkRef link) {
    if (!validate_link_ref(link)) {
        return EXIT_FAILURE;
    }

    tt::umd::Cluster cluster;
    const tt::ARCH arch = cluster.get_cluster_description()->get_arch(link.chip_id);
    const tt::umd::CoreCoord core = get_eth_core(cluster, link.chip_id, link.channel);

    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: {
            namespace wh = eth_fw::wormhole;
            const std::uint32_t retrain = read_u32(cluster, link.chip_id, core, wh::ETH_RETRAIN_ADDR);
            const bool retrain_in_progress = (retrain == wh::ETH_TRIGGER_RETRAIN_VAL);
            const std::uint32_t train_status = read_u32(cluster, link.chip_id, core, wh::ETH_TRAIN_STATUS_ADDR);
            const bool link_up = !retrain_in_progress && train_status == wh::TRAIN_STATUS_SUCCESS;

            std::cout << fmt::format(
                "Chip {} channel {} (arch={}, NOC0=({},{})):\n"
                "  link          : {}\n"
                "  train_status  : {} (0x{:x})\n"
                "  retrain_state : {}\n",
                link.chip_id,
                link.channel,
                tt::arch_to_str(arch),
                core.x,
                core.y,
                link_up ? "up" : "down",
                wh_train_status_to_str(train_status),
                train_status,
                retrain_in_progress ? "in_progress" : "idle");
            return link_up ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        case tt::ARCH::BLACKHOLE: {
            namespace bh = eth_fw::blackhole;

            // ETH_MSG_PORT_UP_CHECK: no input arguments. FW performs a fast
            // link-up check (result written to eth_live_status.rx_link_up) and
            // a true/full link-up check (returned via arg0).
            bh_send_mailbox_msg(
                cluster,
                link.chip_id,
                link.channel,
                core,
                bh::ETH_MSG_TYPE_PORT_UP_CHECK,
                {0, 0, 0},
                /*wait_for_done=*/true);

            const auto args = bh_read_mailbox_args(cluster, link.chip_id, core);
            const std::uint32_t true_link_up = args[0];
            const std::uint32_t rx_link_up = read_u32(cluster, link.chip_id, core, bh::ETH_RX_LINK_UP_ADDR);
            const std::uint32_t port_status = read_u32(cluster, link.chip_id, core, bh::ETH_PORT_STATUS_ADDR);
            const std::uint32_t retrain_count = read_u32(cluster, link.chip_id, core, bh::ETH_RETRAIN_COUNT_ADDR);
            const bool link_up = (true_link_up == 1);

            std::cout << fmt::format(
                "Chip {} channel {} (arch={}, NOC0=({},{})):\n"
                "  link          : {}\n"
                "  true_link_up  : {} (mailbox arg0)\n"
                "  rx_link_up    : {} (eth_live_status.rx_link_up)\n"
                "  port_status   : {} ({})\n"
                "  retrain_count : {}\n",
                link.chip_id,
                link.channel,
                tt::arch_to_str(arch),
                core.x,
                core.y,
                link_up ? "up" : "down",
                true_link_up,
                rx_link_up,
                bh_port_status_to_str(port_status),
                port_status,
                retrain_count);
            return link_up ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        default:
            std::cerr << fmt::format("Unsupported architecture for link status: {}\n", tt::arch_to_str(arch));
            return EXIT_FAILURE;
    }
}

int run_link_reinit_action(LinkRef link, unsigned int reinit_option, unsigned int retries) {
    namespace bh = eth_fw::blackhole;
    if (reinit_option > bh::ETH_PORT_REINIT_OPT_MAC_SERDES_TX_BARRIER) {
        std::cerr << fmt::format(
            "Invalid reinit option {}; expected 0..{}.\n", reinit_option, bh::ETH_PORT_REINIT_OPT_MAC_SERDES_TX_BARRIER);
        return EXIT_FAILURE;
    }
    if (!validate_link_ref(link)) {
        return EXIT_FAILURE;
    }

    tt::umd::Cluster cluster;
    const tt::ARCH arch = cluster.get_cluster_description()->get_arch(link.chip_id);
    const tt::umd::CoreCoord core = get_eth_core(cluster, link.chip_id, link.channel);

    std::cout << fmt::format(
        "Reinitializing chip {} channel {} (arch={}, NOC0=({},{}), reinit_option={} ({}), retries={})\n",
        link.chip_id,
        link.channel,
        tt::arch_to_str(arch),
        core.x,
        core.y,
        reinit_option,
        bh_reinit_option_to_str(reinit_option),
        retries);

    switch (arch) {
        case tt::ARCH::BLACKHOLE: {
            bh_send_mailbox_msg(
                cluster,
                link.chip_id,
                link.channel,
                core,
                bh::ETH_MSG_TYPE_PORT_REINIT_MACPCS,
                {retries, reinit_option, 0},
                /*wait_for_done=*/true);

            const auto args = bh_read_mailbox_args(cluster, link.chip_id, core);
            const std::uint32_t result = args[0];
            std::cout << fmt::format(
                "Chip {} channel {}: ETH FW acknowledged reinit; result=0x{:08x} (mailbox arg0).\n",
                link.chip_id,
                link.channel,
                result);
            return EXIT_SUCCESS;
        }
        default:
            std::cerr << fmt::format(
                "Unsupported architecture for link reinit: {}. Only Blackhole exposes "
                "ETH_MSG_PORT_REINIT_MACPCS; use 'link up' on Wormhole to trigger a retrain.\n",
                tt::arch_to_str(arch));
            return EXIT_FAILURE;
    }
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

int run_link_status(LinkRef link) { return run_link_status_action(link); }

int run_link_reinit(LinkRef link, unsigned int reinit_option, unsigned int retries) {
    return run_link_reinit_action(link, reinit_option, retries);
}

}  // namespace tt_ethtool
