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
#include "fabric/fabric_context.hpp"
#include "fabric.hpp"
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

inline void dumpRoutingInfo(const std::filesystem::path& filepath) {
    nlohmann::ordered_json topology_json;

    const Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    topology_json["mesh_shapes"] = nlohmann::ordered_json::array();
    for (auto [mesh_id, mesh_shape] : tt::tt_fabric::get_physical_mesh_shapes()) {
        topology_json["mesh_shapes"].push_back({
            {"mesh_id", mesh_id.get()},
            {"shape", std::vector(mesh_shape.cbegin(), mesh_shape.cend())},
        });
    }

    topology_json["fabric_config"] = magic_enum::enum_name(tt::tt_metal::MetalContext::instance().get_fabric_config());
    if (tt::tt_metal::MetalContext::instance().get_fabric_config() != FabricConfig::DISABLED) {
        topology_json["routing_planes"] = nlohmann::ordered_json::array();
        topology_json["device_id_to_fabric_node_id"] = nlohmann::ordered_json::object();
        for (auto physical_chip_id : cluster.get_cluster_desc()->get_all_chips()) {
            auto fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(physical_chip_id);
            auto device_routing_planes = nlohmann::ordered_json::array();
            topology_json["device_id_to_fabric_node_id"][std::to_string(physical_chip_id)] = {
                fabric_node_id.mesh_id.get(), fabric_node_id.chip_id};

            for (const auto& direction : tt::tt_fabric::FabricContext::routing_directions) {
                auto eth_routing_planes_in_dir =
                    tt::tt_fabric::get_active_fabric_eth_routing_planes_in_direction(fabric_node_id, direction);

                while (device_routing_planes.size() < eth_routing_planes_in_dir.size()) {
                    device_routing_planes.push_back(
                        {{"routing_plane_id", device_routing_planes.size()},
                         {"ethernet_channels", nlohmann::ordered_json::object()}});
                }

                for (int j = 0; j < eth_routing_planes_in_dir.size(); j++) {
                    chip_id_t eth_channel = eth_routing_planes_in_dir[j];
                    device_routing_planes[j]["ethernet_channels"][magic_enum::enum_name(direction)] = eth_channel;
                }
            }

            topology_json["routing_planes"].push_back(
                {{"device_id", physical_chip_id}, {"device_routing_planes", device_routing_planes}});
        }
    }

    topology_json["eth_chan_to_coord"] = nlohmann::ordered_json::object();
    auto physical_chip_id = *(cluster.get_cluster_desc()->get_all_chips().begin());
    for (int j = 0; j < cluster.get_soc_desc(physical_chip_id).get_num_eth_channels(); j++) {
        tt::umd::CoreCoord edm_eth_core =
            cluster.get_soc_desc(physical_chip_id).get_eth_core_for_channel(j, CoordSystem::PHYSICAL);
        topology_json["eth_chan_to_coord"][std::to_string(j)] = {edm_eth_core.x, edm_eth_core.y};
    }

    std::ofstream topology_json_ofs(filepath);
    TT_FATAL(topology_json_ofs.is_open(), "Failed to open file '{}' for dumping topology", filepath.string());
    topology_json_ofs << topology_json.dump(2);
}

// determines the implied unicast/multicast start distance and range in tt_fabric::LowLatencyRoutingFields
inline std::tuple<int, int> get_low_latency_routing_start_distance_and_range(uint32_t llrf_value) {
    using LLRF = tt::tt_fabric::LowLatencyRoutingFields;

    uint32_t value = llrf_value;
    int start_distance = 1;
    int range = 0;
    while ((value & LLRF::FIELD_MASK) == LLRF::FORWARD_ONLY) {
        value >>= LLRF::FIELD_WIDTH;
        start_distance++;
    }
    // checks if it is either write+forward or just write only
    while (value & LLRF::WRITE_ONLY) {
        value >>= LLRF::FIELD_WIDTH;
        range++;
    }

    return {start_distance, range};
}

// determines the implied unicast/multicast start distance and range in tt_fabric::RoutingFields
inline std::tuple<int, int> get_routing_start_distance_and_range(uint8_t routing_fields_value) {
    int start_distance = tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK & routing_fields_value;
    int range = routing_fields_value >> tt::tt_fabric::RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH;
    return {start_distance, range};
}

}  // namespace tt_metal
}  // namespace tt
