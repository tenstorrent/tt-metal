// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/tt-ethtool/commands.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <string_view>

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

enum class EthChannelState : std::uint8_t {
    ACTIVE,
    IDLE,
    HARVESTED,
};

constexpr std::string_view to_str(EthChannelState state) {
    switch (state) {
        case EthChannelState::ACTIVE: return "ACTIVE";
        case EthChannelState::IDLE: return "IDLE";
        case EthChannelState::HARVESTED: return "HARVESTED";
    }
    return "UNKNOWN";
}

// `get_chip_pci_bdfs()` only contains entries for MMIO-capable chips; return empty string otherwise.
std::string pci_bdf_for_chip(const tt::umd::ClusterDescriptor& desc, tt::ChipId chip_id) {
    const auto& bdfs = desc.get_chip_pci_bdfs();
    auto it = bdfs.find(chip_id);
    return it == bdfs.end() ? std::string{} : it->second;
}

}  // namespace

int run_list() {
    std::unique_ptr<tt::umd::ClusterDescriptor> cluster_desc = tt::umd::Cluster::create_cluster_descriptor();

    const auto& all_chips = cluster_desc->get_all_chips();
    if (all_chips.empty()) {
        std::cout << "No Tenstorrent devices detected on this host.\n";
        return EXIT_SUCCESS;
    }

    std::set<tt::ChipId> chip_ids_sorted(all_chips.begin(), all_chips.end());

    for (tt::ChipId chip_id : chip_ids_sorted) {
        const tt::ARCH arch = cluster_desc->get_arch(chip_id);
        const tt::BoardType board = cluster_desc->get_board_type(chip_id);
        const tt::HarvestingMasks harvesting = cluster_desc->get_harvesting_masks(chip_id);
        const bool is_local = cluster_desc->is_chip_mmio_capable(chip_id);
        const std::string bdf = pci_bdf_for_chip(*cluster_desc, chip_id);

        tt::ChipInfo chip_info{};
        chip_info.harvesting_masks = harvesting;
        chip_info.board_type = board;
        chip_info.asic_location = cluster_desc->get_asic_location(chip_id);
        tt::umd::SocDescriptor soc_desc(arch, chip_info);

        const std::uint32_t num_eth_channels =
            tt::umd::architecture_implementation::create(arch)->get_num_eth_channels();

        std::cout << fmt::format(
            "Chip {} [{}, {}]{}{}:\n",
            chip_id,
            tt::board_type_to_string(board),
            tt::arch_to_str(arch),
            is_local ? "" : " (remote)",
            bdf.empty() ? "" : fmt::format(" PCI {}", bdf));

        if (num_eth_channels == 0) {
            std::cout << "  (no ethernet channels)\n\n";
            continue;
        }

        const std::set<std::uint32_t> active = cluster_desc->get_active_eth_channels(chip_id);
        const std::set<std::uint32_t> idle = cluster_desc->get_idle_eth_channels(chip_id);

        std::cout << fmt::format("  {:<8}{:<14}{:<12}{}\n", "Channel", "NOC0 (x,y)", "State", "Peer");

        for (std::uint32_t channel = 0; channel < num_eth_channels; ++channel) {
            EthChannelState state;
            if (active.contains(channel)) {
                state = EthChannelState::ACTIVE;
            } else if (idle.contains(channel)) {
                state = EthChannelState::IDLE;
            } else {
                state = EthChannelState::HARVESTED;
            }

            std::string coord_str = "-";
            if (state != EthChannelState::HARVESTED) {
                try {
                    const auto core = soc_desc.get_eth_core_for_channel(channel, tt::CoordSystem::NOC0);
                    coord_str = fmt::format("({},{})", core.x, core.y);
                } catch (const std::exception&) {
                    coord_str = "-";
                }
            }

            std::string peer_str = "-";
            if (state == EthChannelState::ACTIVE) {
                const auto [peer_chip, peer_channel] =
                    cluster_desc->get_chip_and_channel_of_remote_ethernet_core(chip_id, channel);
                peer_str = fmt::format("chip {} channel {}", peer_chip, peer_channel);
            } else {
                // If not active locally, the link may still cross into another cluster.
                const auto& remote_conns = cluster_desc->get_ethernet_connections_to_remote_devices();
                auto chip_it = remote_conns.find(chip_id);
                if (chip_it != remote_conns.end()) {
                    auto chan_it = chip_it->second.find(channel);
                    if (chan_it != chip_it->second.end()) {
                        const auto& [remote_unique_id, remote_channel] = chan_it->second;
                        state = EthChannelState::ACTIVE;
                        peer_str = fmt::format("remote uid 0x{:x} channel {}", remote_unique_id, remote_channel);
                    }
                }
            }

            std::cout << fmt::format("  {:<8}{:<14}{:<12}{}\n", channel, coord_str, to_str(state), peer_str);
        }
        std::cout << "\n";
    }

    return EXIT_SUCCESS;
}

}  // namespace tt_ethtool
