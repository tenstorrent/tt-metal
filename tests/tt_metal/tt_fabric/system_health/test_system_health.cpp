// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <magic_enum/magic_enum.hpp>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <tt-metalium/logger.hpp>
#include <tt-metalium/system_memory_manager.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace system_health_tests {

TEST(Cluster, ReportSystemHealth) {
    // Despite potential error messages, this test will not fail
    // It is a report of system health
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const auto& unique_chip_ids = tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids();
    std::stringstream ss;
    ss << "Found " << unique_chip_ids.size() << " chips in cluster:" << std::endl;
    std::vector<std::uint32_t> read_vec;
    auto retrain_count_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);

    std::vector<std::string> unexpected_system_states;
    for (const auto& [chip_id, unique_chip_id] : unique_chip_ids) {
        const auto& soc_desc = cluster.get_soc_desc(chip_id);
        std::stringstream chip_id_ss;
        chip_id_ss << std::dec << "Chip: " << chip_id << " Unique ID: " << std::hex << unique_chip_id;
        ss << chip_id_ss.str() << std::endl;
        for (const auto& [eth_core, chan] : soc_desc.logical_eth_core_to_chan_map) {
            tt_cxy_pair virtual_eth_core(
                chip_id, cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
            std::stringstream eth_ss;
            cluster.read_core(read_vec, sizeof(uint32_t), virtual_eth_core, retrain_count_addr);
            eth_ss << " eth channel " << std::dec << (uint32_t)chan << " " << eth_core.str();
            if (cluster.is_ethernet_link_up(chip_id, eth_core)) {
                const auto& [connected_chip_id, connected_eth_core] =
                    cluster.get_connected_ethernet_core(std::make_tuple(chip_id, eth_core));
                eth_ss << " link UP, retrain: " << read_vec[0] << ", connected to chip " << connected_chip_id << " "
                       << connected_eth_core.str();
                if (read_vec[0] > 0) {
                    unexpected_system_states.push_back(chip_id_ss.str() + eth_ss.str());
                }
            } else {
                eth_ss << " link DOWN";
                unexpected_system_states.push_back(chip_id_ss.str() + eth_ss.str());
            }
            ss << eth_ss.str() << std::endl;
        }
        ss << std::endl;
    }
    log_info(tt::LogTest, "{}", ss.str());

    // Print a summary of unexpected system states
    for (const auto& err_str : unexpected_system_states) {
        log_error(tt::LogTest, "{}", err_str);
    }
}

TEST(Cluster, TestMeshFullConnectivity) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& eth_connections = cluster.get_ethernet_connections();
    std::uint32_t num_expected_chips = 0;
    std::uint32_t num_connections_per_side = 0;
    auto cluster_type = cluster.get_cluster_type();
    if (cluster_type == tt::ClusterType::T3K) {
        num_expected_chips = 8;
        num_connections_per_side = 2;
    } else if (cluster_type == tt::ClusterType::GALAXY) {
        num_expected_chips = 32;
        num_connections_per_side = 4;
    } else {
        GTEST_SKIP() << "Mesh check not supported for system type " << magic_enum::enum_name(cluster_type);
    }
    EXPECT_EQ(eth_connections.size(), num_expected_chips)
        << " Expected " << num_expected_chips << " in " << magic_enum::enum_name(cluster_type) << " cluster";
    for (const auto& [chip, connections] : eth_connections) {
        std::map<chip_id_t, int> num_connections_to_chip;
        for (const auto& [channel, remote_chip_and_channel] : connections) {
            num_connections_to_chip[std::get<0>(remote_chip_and_channel)]++;
        }
        for (const auto& [other_chip, count] : num_connections_to_chip) {
            EXPECT_EQ(count, num_connections_per_side) << "Chip " << chip << " has " << count << " connections to Chip "
                                                       << other_chip << ", expected " << num_connections_per_side;
        }
    }
}

}  // namespace system_health_tests
}  // namespace tt::tt_fabric
