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
#include <tt-metalium/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"

namespace tt::tt_fabric {
namespace system_health_tests {

const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
    {tt::ARCH::WORMHOLE_B0, {0x00, 0x40, 0xC0, 0x80}},
    {tt::ARCH::BLACKHOLE, {0xC0, 0x80, 0x00, 0x40}},
};

std::pair<std::uint32_t, std::uint32_t> get_ubb_ids(chip_id_t chip_id) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tray_bus_ids = ubb_bus_ids.at(cluster.arch());
    auto tray_bus_id_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), cluster.get_bus_id(chip_id) & 0xF0);
    if (tray_bus_id_it != tray_bus_ids.end()) {
        auto ubb_asic_id = cluster.get_ubb_asic_id(chip_id);
        return std::make_pair(tray_bus_id_it - tray_bus_ids.begin() + 1, ubb_asic_id);
    }
    return std::make_pair(0, 0);
}

TEST(Cluster, ReportSystemHealth) {
    // Despite potential error messages, this test will not fail
    // It is a report of system health
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& eth_connections = cluster.get_ethernet_connections();
    const auto& eth_connections_to_remote_mmio_devices = cluster.get_ethernet_connections_to_remote_mmio_devices();

    auto unique_chip_ids = tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids();
    std::stringstream ss;
    std::vector<std::uint32_t> read_vec;
    auto retrain_count_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
    if (unique_chip_ids.empty()) {
        // Temporary patch to workaround unique chip ids not being set for non-6U systems
        for (const auto& chip_id : cluster.user_exposed_chip_ids()) {
            unique_chip_ids[chip_id] = chip_id;
        }
    }
    ss << "Found " << unique_chip_ids.size() << " chips in cluster:" << std::endl;

    auto cluster_type = cluster.get_cluster_type();

    std::vector<std::string> unexpected_system_states;
    for (const auto& [chip_id, unique_chip_id] : unique_chip_ids) {
        const auto& soc_desc = cluster.get_soc_desc(chip_id);
        std::stringstream chip_id_ss;
        chip_id_ss << std::dec << "Chip: " << chip_id << " Unique ID: " << std::hex << unique_chip_id;
        if (cluster_type == tt::ClusterType::GALAXY) {
            auto [tray_id, ubb_asic_id] = get_ubb_ids(chip_id);
            chip_id_ss << " Tray: " << tray_id << " N" << ubb_asic_id;
        }
        ss << chip_id_ss.str() << std::endl;
        for (const auto& [eth_core, chan] : soc_desc.logical_eth_core_to_chan_map) {
            tt_cxy_pair virtual_eth_core(
                chip_id, cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
            std::stringstream eth_ss;
            cluster.read_core(read_vec, sizeof(uint32_t), virtual_eth_core, retrain_count_addr);
            eth_ss << " eth channel " << std::dec << (uint32_t)chan << " " << eth_core.str();
            std::string connection_type =
                cluster.is_external_cable(chip_id, eth_core) ? "(external connector)" : "(internal trace)";
            if (cluster.is_ethernet_link_up(chip_id, eth_core)) {
                if (eth_connections.at(chip_id).find(chan) != eth_connections.at(chip_id).end()) {
                    const auto& [connected_chip_id, connected_eth_core] =
                        cluster.get_connected_ethernet_core(std::make_tuple(chip_id, eth_core));
                    std::cout << "Connected chip: " << connected_chip_id
                              << " connected eth core: " << connected_eth_core.str() << std::endl;
                    eth_ss << " link UP " << connection_type << ", retrain: " << read_vec[0] << ", connected to chip "
                           << connected_chip_id << " " << connected_eth_core.str();
                } else {
                    const auto& [connected_chip_unique_id, connected_eth_core] =
                        cluster.get_connected_ethernet_core_to_remote_mmio_device(std::make_tuple(chip_id, eth_core));
                    std::cout << "Connected unique chip: " << connected_chip_unique_id
                              << " connected eth core: " << connected_eth_core.str() << std::endl;
                    eth_ss << " link UP " << connection_type << ", retrain: " << read_vec[0] << ", connected to chip "
                           << connected_chip_unique_id << " " << connected_eth_core.str();
                }
                if (read_vec[0] > 0) {
                    unexpected_system_states.push_back(chip_id_ss.str() + eth_ss.str());
                }
            } else {
                eth_ss << " link DOWN/unconnected " << connection_type;
                unexpected_system_states.push_back(chip_id_ss.str() + eth_ss.str());
            }
            ss << eth_ss.str() << std::endl;
        }
        ss << std::endl;
    }
    log_info(tt::LogTest, "{}", ss.str());

    // Print a summary of unexpected system states
    for (const auto& err_str : unexpected_system_states) {
        log_warning(tt::LogTest, "{}", err_str);
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

    auto input_args = ::testing::internal::GetArgvs();

    if (test_args::has_command_option(input_args, "-h") || test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(
            LogTest,
            "  --min-connections: target minimum number of connections between chips (default depends on system "
            "type) ");
        log_info(
            LogTest,
            "  --system-topology: system topology to check (defaults to no topology check) Valid values: {}",
            magic_enum::enum_names<FabricType>());
        return;
    }

    // Parse command line arguments
    std::uint32_t num_target_connections = 0;
    std::tie(num_target_connections, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--min-connections", 0);
    if (num_target_connections > num_connections_per_side) {
        log_warning(
            tt::LogTest,
            "Min connections specified is greater than expected num connections per side for {}, overriding to {}.",
            magic_enum::enum_name(cluster_type),
            num_connections_per_side);
        num_target_connections = num_connections_per_side;
    }

    std::optional<FabricType> target_system_topology = std::nullopt;
    std::string target_system_topology_str = "";
    std::tie(target_system_topology_str, input_args) =
        test_args::get_command_option_and_remaining_args(input_args, "--system-topology", "");
    if (not target_system_topology_str.empty()) {
        target_system_topology =
            magic_enum::enum_cast<FabricType>(target_system_topology_str, magic_enum::case_insensitive);
        if (*target_system_topology != FabricType::TORUS_2D) {
            log_warning(
                tt::LogTest,
                "System topology {} not supported for mesh check, skipping topology verification",
                target_system_topology_str);
            target_system_topology = std::nullopt;
        }
    }
    for (const auto& [chip, connections] : eth_connections) {
        std::stringstream chip_ss;
        chip_ss << "Chip " << chip;
        if (cluster_type == tt::ClusterType::GALAXY) {
            auto [tray_id, ubb_asic_id] = get_ubb_ids(chip);
            chip_ss << " Tray: " << tray_id << " N" << ubb_asic_id;
        }
        const auto& soc_desc = cluster.get_soc_desc(chip);
        std::map<chip_id_t, int> num_connections_to_chip;
        for (const auto& [channel, remote_chip_and_channel] : connections) {
            tt::umd::CoreCoord logical_active_eth = soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
            if (cluster.is_ethernet_link_up(chip, logical_active_eth)) {
                num_connections_to_chip[std::get<0>(remote_chip_and_channel)]++;
            }
        }
        if (target_system_topology.has_value()) {
            if (*target_system_topology == FabricType::TORUS_2D) {
                static constexpr std::uint32_t num_expected_chip_connections = 4;
                EXPECT_EQ(num_connections_to_chip.size(), num_expected_chip_connections)
                    << chip_ss.str() << " has " << num_connections_to_chip.size()
                    << " connections to other chips, expected " << num_expected_chip_connections << " for "
                    << magic_enum::enum_name(*target_system_topology) << " topology";
            }
        }
        for (const auto& [other_chip, count] : num_connections_to_chip) {
            std::stringstream other_chip_ss;
            other_chip_ss << "Chip " << other_chip;
            if (cluster_type == tt::ClusterType::GALAXY) {
                auto [tray_id, ubb_asic_id] = get_ubb_ids(other_chip);
                other_chip_ss << " Tray: " << tray_id << " N" << ubb_asic_id;
            }
            if (num_target_connections > 0) {
                EXPECT_GE(count, num_target_connections)
                    << chip_ss.str() << " has " << count << " connections to " << other_chip_ss.str()
                    << ", expected at least " << num_target_connections;
            } else {
                EXPECT_EQ(count, num_connections_per_side)
                    << chip_ss.str() << " has " << count << " connections to " << other_chip_ss.str() << ", expected "
                    << num_connections_per_side;
            }
        }
    }
}

}  // namespace system_health_tests
}  // namespace tt::tt_fabric
