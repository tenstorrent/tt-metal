// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/tt-ethtool/cli/format.hpp"

#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>

#include <fmt/core.h>

#include "tools/scaleout/tt-ethtool/lib/eth_fw.hpp"
#include "umd/device/types/arch.hpp"

namespace tt_ethtool::cli {

namespace {

// Helpers to flatten `EthPeer` into the existing peer column text.
struct PeerFormatter {
    std::string operator()(const LocalPeer& p) const { return fmt::format("chip {} channel {}", p.chip_id, p.channel); }
    std::string operator()(const RemotePeer& p) const {
        return fmt::format("remote uid 0x{:x} channel {}", p.remote_unique_id, p.channel);
    }
};

}  // namespace

std::string_view to_str(EthChannelState state) {
    switch (state) {
        case EthChannelState::ACTIVE: return "ACTIVE";
        case EthChannelState::IDLE: return "IDLE";
        case EthChannelState::HARVESTED: return "HARVESTED";
    }
    return "UNKNOWN";
}

std::string_view bh_port_status_to_str(std::uint32_t status) {
    namespace bh = eth_fw::blackhole;
    switch (status) {
        case bh::PORT_STATUS_UNKNOWN: return "PORT_UNKNOWN";
        case bh::PORT_STATUS_UP: return "PORT_UP";
        case bh::PORT_STATUS_DOWN: return "PORT_DOWN";
        case bh::PORT_STATUS_UNUSED: return "PORT_UNUSED";
        default: return "?";
    }
}

std::string_view bh_reinit_option_to_str(std::uint32_t option) {
    namespace bh = eth_fw::blackhole;
    switch (option) {
        case bh::ETH_PORT_REINIT_OPT_MAC_ONLY: return "mac-only";
        case bh::ETH_PORT_REINIT_OPT_MAC_SERDES_RETRAIN: return "mac+serdes-retrain";
        case bh::ETH_PORT_REINIT_OPT_MAC_SERDES: return "mac+serdes-reset";
        case bh::ETH_PORT_REINIT_OPT_MAC_SERDES_TX_BARRIER: return "mac+serdes-reset-tx-barrier";
        default: return "?";
    }
}

std::string_view wh_train_status_to_str(std::uint32_t status) {
    namespace wh = eth_fw::wormhole;
    switch (status) {
        case wh::TRAIN_STATUS_IN_PROGRESS: return "IN_PROGRESS";
        case wh::TRAIN_STATUS_SUCCESS: return "SUCCESS";
        case wh::TRAIN_STATUS_FAIL: return "FAIL";
        case wh::TRAIN_STATUS_NOT_CONNECTED: return "NOT_CONNECTED";
        default: return "?";
    }
}

void print_list(std::ostream& os, const std::vector<ChipEthInfo>& chips) {
    if (chips.empty()) {
        os << "No Tenstorrent devices detected on this host.\n";
        return;
    }

    for (const auto& chip : chips) {
        os << fmt::format(
            "Chip {} [{}, {}]{}{}:\n",
            chip.chip_id,
            tt::board_type_to_string(chip.board_type),
            tt::arch_to_str(chip.arch),
            chip.is_mmio_capable ? "" : " (remote)",
            chip.pci_bdf.empty() ? "" : fmt::format(" PCI {}", chip.pci_bdf));

        if (chip.channels.empty()) {
            os << "  (no ethernet channels)\n\n";
            continue;
        }

        os << fmt::format("  {:<8}{:<14}{:<12}{}\n", "Channel", "NOC0 (x,y)", "State", "Peer");

        for (const auto& info : chip.channels) {
            std::string coord_str = "-";
            if (info.noc0_coord) {
                coord_str = fmt::format("({},{})", info.noc0_coord->x, info.noc0_coord->y);
            }
            std::string peer_str = "-";
            if (info.peer) {
                peer_str = std::visit(PeerFormatter{}, *info.peer);
            }
            os << fmt::format("  {:<8}{:<14}{:<12}{}\n", info.channel, coord_str, to_str(info.state), peer_str);
        }
        os << "\n";
    }
}

void print_link_action_header(
    std::ostream& os, std::string_view action_name, LinkRef link, const LinkContext& context) {
    os << fmt::format(
        "Bringing chip {} channel {} {} (arch={}, NOC0=({},{}))\n",
        link.chip_id,
        link.channel,
        action_name,
        tt::arch_to_str(context.arch),
        context.noc0_coord.x,
        context.noc0_coord.y);
}

void print_link_action_footer(std::ostream& os, std::string_view action_name, LinkRef link) {
    os << fmt::format(
        "Chip {} channel {}: link {} request acknowledged by ETH FW.\n", link.chip_id, link.channel, action_name);
}

void print_link_status(std::ostream& os, LinkRef link, const LinkStatus& status) {
    std::visit(
        [&](const auto& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, WormholeLinkStatus>) {
                os << fmt::format(
                    "Chip {} channel {} (arch={}, NOC0=({},{})):\n"
                    "  link          : {}\n"
                    "  train_status  : {} (0x{:x})\n"
                    "  retrain_state : {}\n",
                    link.chip_id,
                    link.channel,
                    tt::arch_to_str(s.context.arch),
                    s.context.noc0_coord.x,
                    s.context.noc0_coord.y,
                    s.link_up ? "up" : "down",
                    wh_train_status_to_str(s.train_status),
                    s.train_status,
                    s.retrain_in_progress ? "in_progress" : "idle");
            } else if constexpr (std::is_same_v<T, BlackholeLinkStatus>) {
                os << fmt::format(
                    "Chip {} channel {} (arch={}, NOC0=({},{})):\n"
                    "  link          : {}\n"
                    "  true_link_up  : {} (mailbox arg0)\n"
                    "  rx_link_up    : {} (eth_live_status.rx_link_up)\n"
                    "  port_status   : {} ({})\n"
                    "  retrain_count : {}\n",
                    link.chip_id,
                    link.channel,
                    tt::arch_to_str(s.context.arch),
                    s.context.noc0_coord.x,
                    s.context.noc0_coord.y,
                    s.link_up ? "up" : "down",
                    s.true_link_up,
                    s.rx_link_up,
                    s.port_status,
                    bh_port_status_to_str(s.port_status),
                    s.retrain_count);
            }
        },
        status);
}

void print_link_reinit_header(
    std::ostream& os, LinkRef link, const LinkContext& context, std::uint32_t reinit_option, std::uint32_t retries) {
    os << fmt::format(
        "Reinitializing chip {} channel {} (arch={}, NOC0=({},{}), reinit_option={} ({}), retries={})\n",
        link.chip_id,
        link.channel,
        tt::arch_to_str(context.arch),
        context.noc0_coord.x,
        context.noc0_coord.y,
        reinit_option,
        bh_reinit_option_to_str(reinit_option),
        retries);
}

void print_link_reinit_footer(std::ostream& os, LinkRef link, std::uint32_t fw_result) {
    os << fmt::format(
        "Chip {} channel {}: ETH FW acknowledged reinit; result=0x{:08x} (mailbox arg0).\n",
        link.chip_id,
        link.channel,
        fw_result);
}

void print_link_stats_header(std::ostream& os, LinkRef link, const LinkContext& context, std::uint32_t copy_addr) {
    namespace bh = eth_fw::blackhole;
    os << fmt::format(
        "Refreshing chip {} channel {} live status (arch={}, NOC0=({},{}), copy_addr=0x{:08x}{})\n",
        link.chip_id,
        link.channel,
        tt::arch_to_str(context.arch),
        context.noc0_coord.x,
        context.noc0_coord.y,
        copy_addr,
        copy_addr == bh::ETH_LIVE_STATUS_NO_COPY ? " [no copy]" : "");
}

void print_link_stats(std::ostream& os, LinkRef link, const EthLiveStats& stats) {
    namespace bh = eth_fw::blackhole;

    struct CounterRow {
        std::string_view label;
        std::uint64_t value;
    };

    const std::array<CounterRow, 16> tx_rx_rows = {{
        {"frames_txd", stats.frames_txd},
        {"frames_txd_ok", stats.frames_txd_ok},
        {"frames_txd_badfcs", stats.frames_txd_badfcs},
        {"bytes_txd", stats.bytes_txd},
        {"bytes_txd_ok", stats.bytes_txd_ok},
        {"bytes_txd_badfcs", stats.bytes_txd_badfcs},
        {"frames_rxd", stats.frames_rxd},
        {"frames_rxd_ok", stats.frames_rxd_ok},
        {"frames_rxd_badfcs", stats.frames_rxd_badfcs},
        {"frames_rxd_dropped", stats.frames_rxd_dropped},
        {"bytes_rxd", stats.bytes_rxd},
        {"bytes_rxd_ok", stats.bytes_rxd_ok},
        {"bytes_rxd_badfcs", stats.bytes_rxd_badfcs},
        {"bytes_rxd_dropped", stats.bytes_rxd_dropped},
        {"corr_cw", stats.corr_cw},
        {"uncorr_cw", stats.uncorr_cw},
    }};

    os << fmt::format(
        "Chip {} channel {} eth_live_status @ 0x{:08x}:\n"
        "  retrain_count      : {}\n"
        "  rx_link_up         : {}\n",
        link.chip_id,
        link.channel,
        bh::ETH_LIVE_STATUS_BASE_ADDR,
        stats.retrain_count,
        stats.rx_link_up);
    for (const auto& row : tx_rx_rows) {
        os << fmt::format("  {:<19}: {}\n", row.label, row.value);
    }
}

}  // namespace tt_ethtool::cli
