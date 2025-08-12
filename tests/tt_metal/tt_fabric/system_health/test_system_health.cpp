// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <iomanip>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include <tt_stl/caseless_comparison.hpp>

namespace tt::tt_fabric {
namespace system_health_tests {

struct UbbId {
    std::uint32_t tray_id;
    std::uint32_t asic_id;
};

enum class ConnectorType { UNUSED, QSFP, TFLY, TRACE, LK1, LK2, LK3 };

enum class LinkingBoardType {
    A,
    B,
};

const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
    {tt::ARCH::WORMHOLE_B0, {0xC0, 0x80, 0x00, 0x40}},
    {tt::ARCH::BLACKHOLE, {0x00, 0x40, 0xC0, 0x80}},
};

const std::unordered_map<ConnectorType, LinkingBoardType> linking_board_types = {
    {ConnectorType::LK1, LinkingBoardType::A},
    {ConnectorType::LK2, LinkingBoardType::A},
    {ConnectorType::LK3, LinkingBoardType::B},
};

std::uint64_t cw_pair_to_full(uint32_t hi, uint32_t lo) {
    return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

UbbId get_ubb_id(chip_id_t chip_id) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tray_bus_ids = ubb_bus_ids.at(cluster.arch());
    const auto bus_id = cluster.get_bus_id(chip_id);
    auto tray_bus_id_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_id & 0xF0);
    if (tray_bus_id_it != tray_bus_ids.end()) {
        auto ubb_asic_id = bus_id & 0x0F;
        return UbbId{tray_bus_id_it - tray_bus_ids.begin() + 1, ubb_asic_id};
    }
    return UbbId{0, 0};  // Invalid UBB ID if not found
}

ConnectorType get_connector_type(chip_id_t chip_id, CoreCoord eth_core, uint32_t chan, ClusterType cluster_type) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (cluster_type == ClusterType::GALAXY) {
        if (cluster.is_external_cable(chip_id, eth_core)) {
            return ConnectorType::QSFP;
        }
        auto ubb_id = get_ubb_id(chip_id);
        if ((ubb_id.asic_id == 5 || ubb_id.asic_id == 6) && (12 <= chan && chan <= 15)) {
            return ConnectorType::LK1;
        } else if ((ubb_id.asic_id == 7 || ubb_id.asic_id == 8) && (12 <= chan && chan <= 15)) {
            return ConnectorType::LK2;
        } else if ((ubb_id.asic_id == 4 || ubb_id.asic_id == 8) && (8 <= chan && chan <= 11)) {
            return ConnectorType::LK3;
        } else {
            return ConnectorType::TRACE;
        }
    } else {
        if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
            auto mmio_device_id = cluster.get_associated_mmio_device(chip_id);
            if (mmio_device_id == chip_id) {
                if (chan == 14 || chan == 15) {
                    return ConnectorType::TFLY;
                } else if (chan == 0 || chan == 1 || chan == 6 || chan == 7) {
                    return ConnectorType::QSFP;
                } else if ((chan == 8 || chan == 9) && cluster.get_board_type(chip_id) == tt::umd::BoardType::N300) {
                    return ConnectorType::TRACE;
                }
                return ConnectorType::UNUSED;
            } else {
                if (chan == 6 || chan == 7) {
                    return ConnectorType::TFLY;
                } else if (chan == 0 || chan == 1) {
                    return ConnectorType::TRACE;
                }
                return ConnectorType::UNUSED;
            }
            // TODO: Need to add proper support for other architectures
        } else {
            if (cluster.is_external_cable(chip_id, eth_core)) {
                return ConnectorType::QSFP;
            }
            return ConnectorType::TRACE;
        }
    }
}

bool is_chip_on_edge_of_mesh(chip_id_t physical_chip_id, tt::tt_metal::ClusterType cluster_type) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (cluster_type == tt::tt_metal::ClusterType::GALAXY) {
        auto ubb_asic_id = cluster.get_ubb_asic_id(physical_chip_id);
        return (ubb_asic_id >= 2) and (ubb_asic_id <= 5);
    } else if (cluster_type == tt::tt_metal::ClusterType::T3K) {
        // MMIO chips are on the edge of the mesh
        return cluster.get_associated_mmio_device(physical_chip_id) == physical_chip_id;
    } else {
        log_warning(
            tt::LogTest,
            "is_chip_on_edge_of_mesh not implemented for {} cluster type",
            enchantum::to_string(cluster_type));
        return false;
    }
}

bool is_chip_on_corner_of_mesh(chip_id_t physical_chip_id, tt::tt_metal::ClusterType cluster_type) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    if (cluster_type == tt::tt_metal::ClusterType::GALAXY) {
        auto ubb_asic_id = cluster.get_ubb_asic_id(physical_chip_id);
        return (ubb_asic_id == 1);
    } else if (cluster_type == tt::tt_metal::ClusterType::T3K) {
        // Remote chips are on the corner of the mesh
        return cluster.get_associated_mmio_device(physical_chip_id) != physical_chip_id;
    } else {
        log_warning(
            tt::LogTest,
            "is_chip_on_corner_of_mesh not implemented for {} cluster type",
            enchantum::to_string(cluster_type));
        return false;
    }
}

std::string get_ubb_id_str(chip_id_t chip_id) {
    auto ubb_id = get_ubb_id(chip_id);
    return "Tray: " + std::to_string(ubb_id.tray_id) + " N" + std::to_string(ubb_id.asic_id);
}

std::string get_physical_slot_str(chip_id_t chip_id) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto physical_slot = cluster.get_physical_slot(chip_id);
    if (physical_slot.has_value()) {
        return "Physical Slot: " + std::to_string(*physical_slot);
    }
    return "";
}

std::string get_physical_loc_str(chip_id_t chip_id, ClusterType cluster_type) {
    if (cluster_type == tt::tt_metal::ClusterType::GALAXY) {
        return get_ubb_id_str(chip_id);
    } else {
        return get_physical_slot_str(chip_id);
    }
}

std::string get_connector_str(chip_id_t chip_id, CoreCoord eth_core, uint32_t channel, ClusterType cluster_type) {
    auto connector = get_connector_type(chip_id, eth_core, channel, cluster_type);
    std::stringstream str;
    str << "(";
    switch (connector) {
        case ConnectorType::UNUSED: str << "unused"; break;
        case ConnectorType::QSFP: str << "QSFP"; break;
        case ConnectorType::TFLY: str << "TFLY"; break;
        case ConnectorType::TRACE: str << "internal trace"; break;
        case ConnectorType::LK1:
        case ConnectorType::LK2:
        case ConnectorType::LK3:
            str << "linking board " << enchantum::to_string(connector).back() << " type "
                << enchantum::to_string(linking_board_types.at(connector));
            break;
    }
    str << ")";
    return str.str();
}

TEST(Cluster, ReportIntermeshLinks) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Check if cluster supports intermesh links
    if (!control_plane.system_has_intermesh_links()) {
        log_info(tt::LogTest, "Cluster does not support intermesh links");
        return;
    }

    log_info(tt::LogTest, "Intermesh Link Configuration Report");
    log_info(tt::LogTest, "===================================");

    // Get all intermesh links in the system
    auto all_intermesh_links = control_plane.get_all_intermesh_eth_links();

    // Summary
    size_t total_chips = 0;
    size_t total_links = 0;
    for (const auto& [chip_id, links] : all_intermesh_links) {
        if (links.size() > 0) {
            total_chips++;
            total_links += links.size();
        }
    }

    log_info(tt::LogTest, "Total chips with intermesh links: {}", total_chips);
    log_info(tt::LogTest, "Total intermesh links: {}", total_links);
    log_info(tt::LogTest, "");

    // Detailed information per chip
    for (const auto& chip_id : cluster.user_exposed_chip_ids()) {
        if (control_plane.has_intermesh_links(chip_id)) {
            auto links = control_plane.get_intermesh_eth_links(chip_id);
            log_info(tt::LogTest, "Chip {}: {} inter-mesh ethernet links", chip_id, links.size());

            for (const auto& [eth_core, channel] : links) {
                log_info(tt::LogTest, "  Channel {} at {}", channel, eth_core.str());
            }
        }
    }
}

TEST(Cluster, ReportSystemHealth) {
    // Despite potential error messages, this test will not fail
    // It is a report of system health
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& eth_connections = cluster.get_ethernet_connections();

    auto unique_chip_ids = tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids();
    std::stringstream ss;
    std::vector<std::uint32_t> read_vec;
    auto retrain_count_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
    auto crc_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CRC_ERR);
    auto corr_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::CORR_CW);
    auto uncorr_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNCORR_CW);
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
        auto physical_loc = get_physical_loc_str(chip_id, cluster_type);
        if (not physical_loc.empty()) {
            chip_id_ss << " " << physical_loc;
        }
        ss << chip_id_ss.str() << std::endl;
        for (const auto& [eth_core, chan] : soc_desc.logical_eth_core_to_chan_map) {
            if (get_connector_type(chip_id, eth_core, chan, cluster_type) == ConnectorType::UNUSED) {
                continue;
            }
            tt_cxy_pair virtual_eth_core(
                chip_id, cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
            std::stringstream eth_ss;
            uint32_t crc_error_val = 0;
            uint32_t corr_val_lo = 0, corr_val_hi = 0, uncorr_val_lo = 0, uncorr_val_hi = 0;
            cluster.read_core(read_vec, sizeof(uint32_t), virtual_eth_core, retrain_count_addr);
            // TODO: remove WORMHOLE checks once register access available for all platform
            if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
                cluster.read_core(&crc_error_val, sizeof(uint32_t), virtual_eth_core, crc_addr);
                cluster.read_core(&corr_val_hi, sizeof(uint32_t), virtual_eth_core, corr_addr);
                cluster.read_core(&corr_val_lo, sizeof(uint32_t), virtual_eth_core, corr_addr + 4);
                cluster.read_core(&uncorr_val_hi, sizeof(uint32_t), virtual_eth_core, uncorr_addr);
                cluster.read_core(&uncorr_val_lo, sizeof(uint32_t), virtual_eth_core, uncorr_addr + 4);
            }
            eth_ss << " eth channel " << std::dec << (uint32_t)chan << " core " << eth_core.str();
            std::string connection_type = get_connector_str(chip_id, eth_core, chan, cluster_type);
            if (cluster.is_ethernet_link_up(chip_id, eth_core)) {
                eth_ss << " link UP " << connection_type;
                CoreCoord connected_eth_core = CoreCoord{0, 0};
                if (eth_connections.at(chip_id).find(chan) != eth_connections.at(chip_id).end()) {
                    chip_id_t connected_chip_id = 0;
                    std::tie(connected_chip_id, connected_eth_core) =
                        cluster.get_connected_ethernet_core(std::make_tuple(chip_id, eth_core));
                    eth_ss << ", connected to Chip " << connected_chip_id;
                    auto connected_physical_loc = get_physical_loc_str(connected_chip_id, cluster_type);
                    if (not connected_physical_loc.empty()) {
                        eth_ss << " " << connected_physical_loc;
                    }
                } else {
                    uint64_t connected_chip_unique_id = 0;
                    std::tie(connected_chip_unique_id, connected_eth_core) =
                        cluster.get_connected_ethernet_core_to_remote_mmio_device(std::make_tuple(chip_id, eth_core));
                    eth_ss << ", connected to Unique ID: " << std::hex << connected_chip_unique_id;
                    // Cannot use get_physical_loc_str here as connected_chip_unique_id is on other host
                }
                eth_ss << " core " << connected_eth_core.str();
                eth_ss << "\n\tRetrain count: " << read_vec[0];
                if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
                    eth_ss << " CRC Errors: 0x" << std::hex << crc_error_val;
                    eth_ss << " Corrected Codewords: 0x" << std::hex << cw_pair_to_full(corr_val_hi, corr_val_lo)
                           << " Uncorrected Codewords: 0x" << std::hex << cw_pair_to_full(uncorr_val_hi, uncorr_val_lo);
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
    std::uint32_t num_expected_mmio_chips = 0;

    auto input_args = ::testing::internal::GetArgvs();

    if (test_args::has_command_option(input_args, "-h") || test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(
            LogTest,
            "  --cluster-type: cluster type to check (defaults to inferred from system) Valid values: {}",
            enchantum::names<tt::tt_metal::ClusterType>);
        log_info(
            LogTest,
            "  --min-connections: target minimum number of connections between connected chips (default depends on "
            "system type).");
        log_info(
            LogTest,
            "  --system-topology: system topology to check (defaults to no topology check) Valid values: {}",
            enchantum::names<FabricType>);
        return;
    }

    // Parse command line arguments
    // Cluster type override is mainly needed for detecting T3K clusters
    // T3K cluster type is inferred based on number of chips and number of connections for MMIO and Remote chips
    // If it is missing all connections between chips, it will be set to N300
    // Allow forcing cluster type to enforce error checking if system is expected to be T3K
    std::string cluster_type_str = "";
    std::tie(cluster_type_str, input_args) =
        test_args::get_command_option_and_remaining_args(input_args, "--cluster-type", "");
    tt::tt_metal::ClusterType cluster_type = cluster.get_cluster_type();
    if (not cluster_type_str.empty()) {
        cluster_type = enchantum::cast<tt::tt_metal::ClusterType>(cluster_type_str, ttsl::ascii_caseless_comp).value();
    }

    if (cluster_type == tt::tt_metal::ClusterType::T3K) {
        num_expected_chips = 8;
        num_expected_mmio_chips = 4;
        num_connections_per_side = 2;
    } else if (cluster_type == tt::tt_metal::ClusterType::GALAXY) {
        num_expected_chips = 32;
        num_expected_mmio_chips = 32;
        num_connections_per_side = 4;
    } else if (cluster_type == tt::tt_metal::ClusterType::P150_X2) {
        num_expected_chips = 2;
        num_expected_mmio_chips = 2;
        num_connections_per_side = 4;
    } else if (cluster_type == tt::tt_metal::ClusterType::P150_X4) {
        num_expected_chips = 4;
        num_expected_mmio_chips = 4;
        num_connections_per_side = 4;
    } else if (cluster_type == tt::tt_metal::ClusterType::P150_X8) {
        num_expected_chips = 8;
        num_expected_mmio_chips = 8;
        num_connections_per_side = 2;
    } else {
        GTEST_SKIP() << "Mesh check not supported for system type " << enchantum::to_string(cluster_type);
    }

    EXPECT_EQ(eth_connections.size(), num_expected_chips)
        << " Expected " << num_expected_chips << " chips in " << enchantum::to_string(cluster_type)
        << " cluster but found " << eth_connections.size();
    std::uint32_t num_mmio_chips = cluster.number_of_pci_devices();
    EXPECT_EQ(num_mmio_chips, num_expected_mmio_chips)
        << " Expected " << num_expected_mmio_chips << " MMIO chips in " << enchantum::to_string(cluster_type)
        << " cluster but found " << num_mmio_chips;

    std::uint32_t num_target_connections = 0;
    std::tie(num_target_connections, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--min-connections", 0);
    if (num_target_connections > num_connections_per_side) {
        log_warning(
            tt::LogTest,
            "Min connections specified is greater than expected num connections per side for {}, overriding to {}.",
            enchantum::to_string(cluster_type),
            num_connections_per_side);
        num_target_connections = num_connections_per_side;
    }

    std::optional<FabricType> target_system_topology = std::nullopt;
    std::string target_system_topology_str = "";
    std::tie(target_system_topology_str, input_args) =
        test_args::get_command_option_and_remaining_args(input_args, "--system-topology", "");
    if (not target_system_topology_str.empty()) {
        target_system_topology =
            enchantum::cast<FabricType>(target_system_topology_str, ttsl::ascii_caseless_comp);
        // TORUS_XY is the only topology that is supported for all cluster types
        if (target_system_topology.has_value() && *target_system_topology != FabricType::TORUS_XY) {
            bool supported_topology = false;
            switch (cluster_type) {
                case tt::tt_metal::ClusterType::GALAXY:
                case tt::tt_metal::ClusterType::T3K:
                    supported_topology = *target_system_topology == FabricType::MESH;
                    break;
                default: supported_topology = false; break;
            };
            if (not supported_topology) {
                log_warning(
                    tt::LogTest,
                    "System topology {} not supported for topology validation on {} cluster, skipping topology "
                    "verification",
                    enchantum::to_string(*target_system_topology),
                    enchantum::to_string(cluster_type));
                target_system_topology = std::nullopt;
            }
        }
    }
    for (const auto& [chip, connections] : eth_connections) {
        std::stringstream chip_ss;
        chip_ss << "Chip " << chip;
        auto physical_loc = get_physical_loc_str(chip, cluster_type);
        if (not physical_loc.empty()) {
            chip_ss << " " << physical_loc;
        }
        const auto& soc_desc = cluster.get_soc_desc(chip);
        std::map<chip_id_t, int> num_connections_to_chip;
        std::map<chip_id_t, uint32_t> num_internal_connections_to_chip;
        std::map<chip_id_t, uint32_t> num_external_connections_to_chip;
        for (const auto& [channel, remote_chip_and_channel] : connections) {
            tt::umd::CoreCoord logical_active_eth_umd_coord =
                soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
            CoreCoord logical_active_eth_coord =
                CoreCoord{logical_active_eth_umd_coord.x, logical_active_eth_umd_coord.y};
            if (cluster.is_ethernet_link_up(chip, logical_active_eth_coord)) {
                auto remote_chip = std::get<0>(remote_chip_and_channel);
                num_connections_to_chip[remote_chip]++;
                if (cluster.is_external_cable(chip, logical_active_eth_coord)) {
                    num_external_connections_to_chip[remote_chip]++;
                } else {
                    num_internal_connections_to_chip[remote_chip]++;
                }
            }
        }
        if (target_system_topology.has_value()) {
            auto validate_num_connections = [&](uint32_t num_connections, uint32_t num_expected_chip_connections) {
                EXPECT_EQ(num_connections, num_expected_chip_connections)
                    << chip_ss.str() << " is connected to " << num_connections << " other chips, expected "
                    << num_expected_chip_connections << " chips for " << enchantum::to_string(*target_system_topology)
                    << " topology";
            };
            if (*target_system_topology == FabricType::TORUS_XY) {
                static constexpr std::uint32_t num_expected_chip_connections = 4;
                validate_num_connections(num_connections_to_chip.size(), num_expected_chip_connections);
            } else {
                uint32_t num_chip_connections = 0;
                if (cluster_type == tt::tt_metal::ClusterType::GALAXY) {
                    num_chip_connections = num_internal_connections_to_chip.size();
                } else {
                    num_chip_connections = num_connections_to_chip.size();
                }
                // TODO: This is UBB specific where we only consider internal connections when determining MESH topology
                if (is_chip_on_corner_of_mesh(chip, cluster_type)) {
                    static constexpr std::uint32_t num_expected_chip_connections = 2;
                    validate_num_connections(num_chip_connections, num_expected_chip_connections);
                } else if (is_chip_on_edge_of_mesh(chip, cluster_type)) {
                    static constexpr std::uint32_t num_expected_chip_connections = 3;
                    validate_num_connections(num_chip_connections, num_expected_chip_connections);
                } else {
                    static constexpr std::uint32_t num_expected_chip_connections = 4;
                    validate_num_connections(num_chip_connections, num_expected_chip_connections);
                }
            }
        }
        for (const auto& [other_chip, count] : num_connections_to_chip) {
            std::stringstream other_chip_ss;
            other_chip_ss << "Chip " << other_chip;
            auto other_physical_loc = get_physical_loc_str(other_chip, cluster_type);
            if (not other_physical_loc.empty()) {
                other_chip_ss << " " << other_physical_loc;
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
