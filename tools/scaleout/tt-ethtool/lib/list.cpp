// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/tt-ethtool/lib/operations.hpp"

#include <cstdint>
#include <exception>
#include <memory>
#include <set>
#include <string>

#include "umd/device/arch/architecture_implementation.hpp"
#include "umd/device/cluster.hpp"
#include "umd/device/cluster_descriptor.hpp"
#include "umd/device/soc_descriptor.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt_ethtool {

namespace {

// `get_chip_pci_bdfs()` only contains entries for MMIO-capable chips; return empty string otherwise.
std::string pci_bdf_for_chip(const tt::umd::ClusterDescriptor& desc, tt::ChipId chip_id) {
    const auto& bdfs = desc.get_chip_pci_bdfs();
    auto it = bdfs.find(chip_id);
    return it == bdfs.end() ? std::string{} : it->second;
}

ChipEthInfo collect_chip_info(tt::umd::ClusterDescriptor& desc, tt::ChipId chip_id) {
    ChipEthInfo chip{};
    chip.chip_id = chip_id;
    chip.arch = desc.get_arch(chip_id);
    chip.board_type = desc.get_board_type(chip_id);
    chip.is_mmio_capable = desc.is_chip_mmio_capable(chip_id);
    chip.pci_bdf = pci_bdf_for_chip(desc, chip_id);

    const tt::HarvestingMasks harvesting = desc.get_harvesting_masks(chip_id);
    tt::ChipInfo chip_info{};
    chip_info.harvesting_masks = harvesting;
    chip_info.board_type = chip.board_type;
    chip_info.asic_location = desc.get_asic_location(chip_id);
    tt::umd::SocDescriptor soc_desc(chip.arch, chip_info);

    const std::uint32_t num_eth_channels =
        tt::umd::architecture_implementation::create(chip.arch)->get_num_eth_channels();
    if (num_eth_channels == 0) {
        return chip;
    }

    const std::set<std::uint32_t> active = desc.get_active_eth_channels(chip_id);
    const std::set<std::uint32_t> idle = desc.get_idle_eth_channels(chip_id);
    const auto& remote_conns = desc.get_ethernet_connections_to_remote_devices();

    chip.channels.reserve(num_eth_channels);
    for (std::uint32_t channel = 0; channel < num_eth_channels; ++channel) {
        EthChannelInfo info{};
        info.channel = channel;
        if (active.contains(channel)) {
            info.state = EthChannelState::ACTIVE;
        } else if (idle.contains(channel)) {
            info.state = EthChannelState::IDLE;
        } else {
            info.state = EthChannelState::HARVESTED;
        }

        if (info.state != EthChannelState::HARVESTED) {
            try {
                info.noc0_coord = soc_desc.get_eth_core_for_channel(channel, tt::CoordSystem::NOC0);
            } catch (const std::exception&) {
                info.noc0_coord.reset();
            }
        }

        if (info.state == EthChannelState::ACTIVE) {
            const auto [peer_chip, peer_channel] = desc.get_chip_and_channel_of_remote_ethernet_core(chip_id, channel);
            info.peer = LocalPeer{.chip_id = peer_chip, .channel = static_cast<std::uint32_t>(peer_channel)};
        } else {
            // If not active locally, the link may still cross into another cluster.
            auto chip_it = remote_conns.find(chip_id);
            if (chip_it != remote_conns.end()) {
                auto chan_it = chip_it->second.find(channel);
                if (chan_it != chip_it->second.end()) {
                    const auto& [remote_unique_id, remote_channel] = chan_it->second;
                    info.state = EthChannelState::ACTIVE;
                    info.peer = RemotePeer{
                        .remote_unique_id = remote_unique_id,
                        .channel = static_cast<std::uint32_t>(remote_channel),
                    };
                }
            }
        }

        chip.channels.push_back(info);
    }
    return chip;
}

}  // namespace

std::vector<ChipEthInfo> list_eth_ports(tt::umd::ClusterDescriptor& desc) {
    const auto& all_chips = desc.get_all_chips();
    std::set<tt::ChipId> chip_ids_sorted(all_chips.begin(), all_chips.end());

    std::vector<ChipEthInfo> result;
    result.reserve(chip_ids_sorted.size());
    for (tt::ChipId chip_id : chip_ids_sorted) {
        result.push_back(collect_chip_info(desc, chip_id));
    }
    return result;
}

std::vector<ChipEthInfo> list_eth_ports() {
    std::unique_ptr<tt::umd::ClusterDescriptor> desc = tt::umd::Cluster::create_cluster_descriptor();
    return list_eth_ports(*desc);
}

}  // namespace tt_ethtool
