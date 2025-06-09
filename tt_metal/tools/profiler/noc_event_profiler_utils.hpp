// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Utility functions and data structures related to tt-metal kernel profiler's noc tracing feature

#include <tuple>
#include <optional>
#include <map>
#include <vector>
#include <utility>
#include <nlohmann/json.hpp>

#include "tt_cluster.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "tt_metal.hpp"

namespace tt {

namespace tt_metal {

// precomputes the mapping between EDM router physical coordinate locations and their associated fabric channel IDs
class FabricRoutingLookup {
public:
    // both of these are keyed by physical chip id!
    using EthCoreToChannelMap = std::map<std::tuple<chip_id_t, CoreCoord>, tt::tt_fabric::chan_id_t>;

    // Default constructor for cases where lookup is not built (e.g., non-1D fabric)
    FabricRoutingLookup() = default;

    FabricRoutingLookup(const IDevice* device) {
        using namespace tt::tt_fabric;

        Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

        // get sorted list of all physical chip ids
        auto physical_chip_id_set = cluster.user_exposed_chip_ids();
        std::vector<chip_id_t> physical_chip_ids(physical_chip_id_set.begin(), physical_chip_id_set.end());
        std::sort(physical_chip_ids.begin(), physical_chip_ids.end());

        for (chip_id_t chip_id_src : physical_chip_ids) {
            if (device->is_mmio_capable() && (cluster.get_cluster_type() == tt::ClusterType::TG)) {
                // skip lauching on gateways for TG
                continue;
            }

            // NOTE: soc desc is for chip_id_src, not device->id()
            const auto& soc_desc = cluster.get_soc_desc(chip_id_src);
            // Build a mapping of (eth_core --> eth_chan)
            for (auto eth_chan = 0; eth_chan < soc_desc.get_num_eth_channels(); eth_chan++) {
                auto eth_physical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::PHYSICAL);
                eth_core_to_channel_lookup_.emplace(std::make_tuple(chip_id_src, eth_physical_core), eth_chan);
            }
        }
    }

    // lookup Eth Channel ID given a physical chip id and physical EDM router core coordinate
    std::optional<tt::tt_fabric::chan_id_t> getRouterEthCoreToChannelLookup(
        chip_id_t phys_chip_id, CoreCoord eth_router_phys_core_coord) const {
        auto it = eth_core_to_channel_lookup_.find(std::make_tuple(phys_chip_id, eth_router_phys_core_coord));
        if (it != eth_core_to_channel_lookup_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    EthCoreToChannelMap eth_core_to_channel_lookup_;
};

inline void dumpClusterCoordinatesAsJson(const std::filesystem::path& filepath) {
    Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    nlohmann::ordered_json cluster_json;
    cluster_json["physical_chip_to_eth_coord"] = nlohmann::ordered_json();
    for (auto& [chip_id, eth_core] : cluster.get_user_chip_ethernet_coordinates()) {
        eth_coord_t eth_coord = eth_core;
        auto& entry = cluster_json["physical_chip_to_eth_coord"][std::to_string(chip_id)];
        entry["rack"] = eth_coord.rack;
        entry["shelf"] = eth_coord.shelf;
        entry["x"] = eth_coord.x;
        entry["y"] = eth_coord.y;
    }

    std::ofstream cluster_json_ofs(filepath);
    if (cluster_json_ofs.is_open()) {
        cluster_json_ofs << cluster_json.dump(2);
    } else {
        log_error(tt::LogMetal, "Failed to open file '{}' for dumping cluster coordinate map", filepath.string());
    }
}

// determines the implied unicast hop count in tt_fabric::LowLatencyRoutingFields
inline int get_low_latency_routing_hops(uint32_t llrf_value) {
    uint32_t value = llrf_value;
    uint32_t hops = 0;
    while (value) {
        value >>= tt::tt_fabric::LowLatencyRoutingFields::FIELD_WIDTH;
        hops++;
    }
    return hops;
}

// determines the implied unicast hop count in tt_fabric::RoutingFields
inline int get_routing_hops(uint8_t routing_fields_value) {
    return tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK & routing_fields_value;
}

}  // namespace tt_metal
}  // namespace tt
