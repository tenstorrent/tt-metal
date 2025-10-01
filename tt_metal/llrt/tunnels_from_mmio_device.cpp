// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tunnels_from_mmio_device.hpp"

#include <tt_stl/assert.hpp>

namespace tt::llrt {

// TODO: Stop using these functions here and in PhysicalSystemDescriptor once UMD provides support for ASIC index/offset
// in WH systems
const std::unordered_set<chip_id_t>& get_devices_controlled_by_mmio_device(
    const std::unique_ptr<tt::umd::Cluster>& cluster, chip_id_t mmio_device_id) {
    const auto& cluster_descriptor = cluster->get_cluster_description();
    TT_ASSERT(
        cluster_descriptor->get_chips_grouped_by_closest_mmio().count(mmio_device_id),
        "Expected device {} to be an MMIO device!",
        mmio_device_id);
    return cluster_descriptor->get_chips_grouped_by_closest_mmio().at(mmio_device_id);
}

#define MAX_TUNNEL_DEPTH 4
std::map<chip_id_t, std::vector<std::vector<chip_id_t>>> discover_tunnels_from_mmio_device(
    const std::unique_ptr<tt::umd::Cluster>& cluster) {
    std::map<chip_id_t, std::vector<std::vector<tt::umd::chip_id_t>>> tunnels_from_mmio_device = {};

    for (const auto& mmio_chip_id : cluster->get_target_mmio_device_ids()) {
        std::vector<std::vector<chip_id_t>> tunnels_from_mmio = {};
        const auto& all_eth_connections = cluster->get_cluster_description()->get_ethernet_connections();
        TT_ASSERT(cluster->get_cluster_description()->is_chip_mmio_capable(mmio_chip_id));

        if (all_eth_connections.find(mmio_chip_id) == all_eth_connections.end()) {
            tunnels_from_mmio_device.insert({mmio_chip_id, {}});
            continue;
        }

        auto device_ids = get_devices_controlled_by_mmio_device(cluster, mmio_chip_id);
        device_ids.erase(mmio_chip_id);

        if (device_ids.empty()) {
            tunnels_from_mmio_device.insert({mmio_chip_id, {}});
            continue;
        }

        for (const auto& [eth_chan, connected_chip_chan] : all_eth_connections.at(mmio_chip_id)) {
            const auto& other_chip_id = std::get<0>(connected_chip_chan);
            if (device_ids.find(other_chip_id) != device_ids.end()) {
                // mmio chip is connected to a remote chip in its mmio group.
                // erase from the pool so multiple ethernet connections to same remote device do not
                // pollute the counts.
                device_ids.erase(other_chip_id);
                std::vector<chip_id_t> first_stop = {other_chip_id};
                auto it = std::find(tunnels_from_mmio.begin(), tunnels_from_mmio.end(), first_stop);
                TT_ASSERT(
                    it == tunnels_from_mmio.end(),
                    "Duplicate first tunnel stop found when finding FD2 Tunnel devices.");
                tunnels_from_mmio.push_back(first_stop);
            }
        }

        log_debug(
            tt::LogMetal,
            "Found {} FD Tunnels originating from MMIO Device {}",
            tunnels_from_mmio.size(),
            mmio_chip_id);

        device_ids = get_devices_controlled_by_mmio_device(cluster, mmio_chip_id);
        device_ids.erase(mmio_chip_id);

        for (auto& tunnel : tunnels_from_mmio) {
            TT_ASSERT(tunnel.size() == 1, "Tunnel depth must be 1 when it has only 1 stop in it.");
            device_ids.erase(tunnel[0]);
        }

        bool tunneled_device_hit;
        while (!device_ids.empty()) {
            tunneled_device_hit = false;
            for (auto& dev_vec : tunnels_from_mmio) {
                for (const auto& [eth_chan, connected_chip_chan] : all_eth_connections.at(dev_vec.back())) {
                    const auto& other_chip_id = std::get<0>(connected_chip_chan);
                    auto id_iter = device_ids.find(other_chip_id);
                    if (id_iter != device_ids.end()) {
                        device_ids.erase(id_iter);
                        dev_vec.push_back(other_chip_id);
                        tunneled_device_hit = true;
                        break;
                    }
                }
            }
            TT_FATAL(
                tunneled_device_hit || device_ids.empty(),
                "Detected ethernet connections did not match expected device connectivity, try re-running "
                "tt-topology.");
        }

        TT_ASSERT(!tunnels_from_mmio.empty(), "Must have at least 1 tunnel from MMIO Device.");
        uint32_t tunnel_depth = tunnels_from_mmio[0].size();
        log_debug(tt::LogMetal, "Each FD Tunnel is {} deep.", tunnel_depth);

        for (auto& dev_vec : tunnels_from_mmio) {
            TT_ASSERT(
                dev_vec.size() == tunnel_depth,
                "All tunnels from mmio device must have same depth. Found {}. Expected {}.",
                dev_vec.size(),
                tunnel_depth);
            // Now that all remote chips have been added to respective tunnels,
            // add mmio device at start of each of the tunnels.
            if (dev_vec.size() > MAX_TUNNEL_DEPTH) {
                dev_vec.resize(dev_vec.size() - (dev_vec.size() - MAX_TUNNEL_DEPTH));
            }
            dev_vec.insert(dev_vec.begin(), mmio_chip_id);
        }
        tunnels_from_mmio_device.insert({mmio_chip_id, tunnels_from_mmio});
    }

    return tunnels_from_mmio_device;
}

}  // namespace tt::llrt
