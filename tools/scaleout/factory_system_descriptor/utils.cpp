// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include <fstream>
#include <set>
#include <tuple>
#include <sstream>

#include <fmt/base.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <yaml-cpp/yaml.h>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt_stl/reflection.hpp>
#include <board/board.hpp>
#include <cabling_generator/cabling_generator.hpp>

// Add protobuf includes
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

void validate_fsd_against_gsd(
    const std::string& fsd_filename, const std::string& gsd_filename, bool strict_validation) {
    // Read the generated FSD using protobuf
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor generated_fsd;
    std::ifstream fsd_file(fsd_filename);
    if (!fsd_file.is_open()) {
        throw std::runtime_error("Failed to open FSD file: " + fsd_filename);
    }

    std::string fsd_content((std::istreambuf_iterator<char>(fsd_file)), std::istreambuf_iterator<char>());
    fsd_file.close();

    if (!google::protobuf::TextFormat::ParseFromString(fsd_content, &generated_fsd)) {
        throw std::runtime_error("Failed to parse FSD protobuf from file: " + fsd_filename);
    }

    const auto& hosts = generated_fsd.hosts();

    // Read the discovered GSD (Global System Descriptor) - still using YAML
    YAML::Node discovered_gsd = YAML::LoadFile(gsd_filename);

    // Compare the FSD with the discovered GSD
    // First, compare hostnames from the hosts field
    if (generated_fsd.hosts().empty()) {
        throw std::runtime_error("FSD missing hosts");
    }

    // Handle the new GSD structure with compute_node_specs
    if (!discovered_gsd["compute_node_specs"]) {
        throw std::runtime_error("GSD missing compute_node_specs");
    }
    YAML::Node asic_info_node = discovered_gsd["compute_node_specs"];

    // Check that all discovered hostnames are present in the generated FSD hosts
    std::set<std::string> generated_hostnames;
    std::unordered_map<std::string, std::string> generated_motherboards;
    for (const auto& host : generated_fsd.hosts()) {
        generated_hostnames.insert(host.hostname());
        generated_motherboards[host.hostname()] = host.motherboard();
    }

    std::set<std::string> discovered_hostnames;
    for (const auto& hostname_entry : asic_info_node) {
        discovered_hostnames.insert(hostname_entry.first.as<std::string>());
    }

    if (strict_validation) {
        if (generated_hostnames != discovered_hostnames) {
            throw std::runtime_error("Hostnames mismatch");
        }
    } else {
        for (const auto& hostname : discovered_hostnames) {
            if (generated_hostnames.find(hostname) == generated_hostnames.end()) {
                throw std::runtime_error("Hostname not found in FSD: " + hostname);
            }
        }
    }

    // Compare motherboards
    for (const auto& host_entry : asic_info_node) {
        const std::string& hostname = host_entry.first.as<std::string>();
        const YAML::Node& host_node = host_entry.second;
        if (!host_node["motherboard"]) {
            throw std::runtime_error("Host " + hostname + " missing motherboard in GSD");
        }
        if (generated_motherboards[hostname] != host_node["motherboard"].as<std::string>()) {
            throw std::runtime_error(
                "Motherboard mismatch between FSD and GSD for host " + hostname +
                ": FSD=" + generated_motherboards[hostname] + ", GSD=" + host_node["motherboard"].as<std::string>());
        }
    }

    // Compare board types
    if (!generated_fsd.has_board_types()) {
        throw std::runtime_error("FSD missing board_types");
    }
    std::set<std::tuple<uint32_t, uint32_t, std::string>> generated_board_types;
    for (const auto& board_location : generated_fsd.board_types().board_locations()) {
        uint32_t host_id = board_location.host_id();
        uint32_t tray_id = board_location.tray_id();
        const std::string& board_type_name = board_location.board_type();
        generated_board_types.insert(std::make_tuple(host_id, tray_id, board_type_name));
    }

    // Strict validation: Each host, tray combination should have the same board type between FSD and GSD
    std::map<std::pair<std::string, uint32_t>, std::string> fsd_board_types;

    // Extract board types from FSD
    for (const auto& board_location : generated_fsd.board_types().board_locations()) {
        uint32_t host_id = board_location.host_id();
        uint32_t tray_id = board_location.tray_id();
        const std::string& board_type_name = board_location.board_type();
        fsd_board_types[std::make_pair(hosts[host_id].hostname(), tray_id)] = board_type_name;
    }

    // Compare board types between GSD and FSD
    for (const auto& host_entry : asic_info_node) {
        const std::string& hostname = host_entry.first.as<std::string>();
        const YAML::Node& host_node = host_entry.second;
        if (!host_node["asic_info"]) {
            throw std::runtime_error("Host " + hostname + " missing asic_info");
        }

        for (const auto& asic_info : host_node["asic_info"]) {
            uint32_t tray_id = asic_info["tray_id"].as<uint32_t>();
            std::string gsd_board_type = asic_info["board_type"].as<std::string>();

            auto fsd_key = std::make_pair(hostname, tray_id);
            if (strict_validation) {
                auto fsd_board_type = fsd_board_types.extract(fsd_key);

                if (fsd_board_type.empty()) {
                    throw std::runtime_error(
                        "Board type not found in FSD for host " + hostname + ", tray " + std::to_string(tray_id));
                }

                if (fsd_board_type.mapped() != gsd_board_type) {
                    throw std::runtime_error(
                        "Board type mismatch for host " + hostname + ", tray " + std::to_string(tray_id) +
                        ": FSD=" + fsd_board_type.mapped() + ", GSD=" + gsd_board_type);
                }
            } else {
                auto fsd_board_type = fsd_board_types.find(fsd_key);
                if (fsd_board_type != fsd_board_types.end()) {
                    if (fsd_board_type->second != gsd_board_type) {
                        throw std::runtime_error(
                            "Board type mismatch for host " + hostname + ", tray " + std::to_string(tray_id) +
                            ": FSD=" + fsd_board_type->second + ", GSD=" + gsd_board_type);
                    }
                } else {
                    throw std::runtime_error(
                        "Board type not found in FSD for host " + hostname + ", tray " + std::to_string(tray_id));
                }
            }
        }
    }
    if (strict_validation) {
        if (fsd_board_types.size() != 0) {
            throw std::runtime_error("Expected all board types to be found in FSD");
        }
    }

    // Compare chip connections
    if (!generated_fsd.has_eth_connections()) {
        throw std::runtime_error("FSD missing eth_connections");
    }

    // Determine which connection types exist in the discovered GSD
    bool has_local_eth_connections =
        discovered_gsd["local_eth_connections"] && !discovered_gsd["local_eth_connections"].IsNull();
    bool has_global_eth_connections =
        discovered_gsd["global_eth_connections"] && !discovered_gsd["global_eth_connections"].IsNull();

    // At least one connection type should exist
    if (!has_local_eth_connections && !has_global_eth_connections) {
        throw std::runtime_error("No connection types found in discovered GSD");
    }

    // Convert generated connections to a comparable format
    std::set<std::pair<PhysicalChannelEndpoint, PhysicalChannelEndpoint>> generated_connections;
    std::set<std::pair<PhysicalChannelEndpoint, PhysicalChannelEndpoint>> duplicate_generated_connections;

    for (const auto& connection : generated_fsd.eth_connections().connection()) {
        const auto& endpoint_a = connection.endpoint_a();
        const auto& endpoint_b = connection.endpoint_b();

        uint32_t host_id_1 = endpoint_a.host_id();
        uint32_t tray_id_1 = endpoint_a.tray_id();
        uint32_t asic_location_1 = endpoint_a.asic_location();
        uint32_t chan_id_1 = endpoint_a.chan_id();

        uint32_t host_id_2 = endpoint_b.host_id();
        uint32_t tray_id_2 = endpoint_b.tray_id();
        uint32_t asic_location_2 = endpoint_b.asic_location();
        uint32_t chan_id_2 = endpoint_b.chan_id();

        const std::string& hostname_1 = hosts[host_id_1].hostname();
        const std::string& hostname_2 = hosts[host_id_2].hostname();

        PhysicalChannelEndpoint conn_1{hostname_1, TrayId(tray_id_1), AsicChannel{asic_location_1, ChanId(chan_id_1)}};
        PhysicalChannelEndpoint conn_2{hostname_2, TrayId(tray_id_2), AsicChannel{asic_location_2, ChanId(chan_id_2)}};

        // Sort to ensure consistent ordering
        std::pair<PhysicalChannelEndpoint, PhysicalChannelEndpoint> connection_pair_sorted;
        if (conn_1 < conn_2) {
            connection_pair_sorted = std::make_pair(conn_1, conn_2);
        } else {
            connection_pair_sorted = std::make_pair(conn_2, conn_1);
        }

        // Check for duplicates before inserting
        if (generated_connections.find(connection_pair_sorted) != generated_connections.end()) {
            duplicate_generated_connections.insert(connection_pair_sorted);
        } else {
            generated_connections.insert(connection_pair_sorted);
        }
    }

    // Report any duplicates found in generated connections
    if (!duplicate_generated_connections.empty()) {
        std::string error_msg = "Duplicate connections found in generated FSD:\n";
        for (const auto& dup : duplicate_generated_connections) {
            std::ostringstream oss;
            oss << "  - " << dup.first << " <-> " << dup.second;
            error_msg += oss.str() + "\n";
        }
        throw std::runtime_error(error_msg);
    }

    // Convert discovered GSD connections to the same format
    std::set<PhysicalChannelConnection> discovered_connections;
    std::set<PhysicalChannelConnection> duplicate_discovered_connections;

    // Process local connections if they exist
    if (has_local_eth_connections) {
        for (const auto& connection_pair : discovered_gsd["local_eth_connections"]) {
            if (connection_pair.size() != 2) {
                throw std::runtime_error("Each connection should have exactly 2 endpoints");
            }

            const auto& first_conn = connection_pair[0];
            const auto& second_conn = connection_pair[1];

            std::string hostname_1 = first_conn["host_name"].as<std::string>();
            uint32_t chan_id_1 = first_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_1 = first_conn["tray_id"].as<uint32_t>();
            uint32_t asic_location_1 = first_conn["asic_location"].as<uint32_t>();

            std::string hostname_2 = second_conn["host_name"].as<std::string>();
            uint32_t chan_id_2 = second_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_2 = second_conn["tray_id"].as<uint32_t>();
            uint32_t asic_location_2 = second_conn["asic_location"].as<uint32_t>();

            PhysicalChannelEndpoint conn_1{
                hostname_1, TrayId(tray_id_1), AsicChannel{asic_location_1, ChanId(chan_id_1)}};
            PhysicalChannelEndpoint conn_2{
                hostname_2, TrayId(tray_id_2), AsicChannel{asic_location_2, ChanId(chan_id_2)}};

            // Sort to ensure consistent ordering
            PhysicalChannelConnection connection_pair_sorted;
            if (conn_1 < conn_2) {
                connection_pair_sorted = std::make_pair(conn_1, conn_2);
            } else {
                connection_pair_sorted = std::make_pair(conn_2, conn_1);
            }

            // Check for duplicates before inserting
            if (discovered_connections.find(connection_pair_sorted) != discovered_connections.end()) {
                duplicate_discovered_connections.insert(connection_pair_sorted);
            } else {
                discovered_connections.insert(connection_pair_sorted);
            }
        }
    }

    // Process global_eth_connections if they exist (for 5WHGalaxyYTorusSuperpod)
    if (has_global_eth_connections) {
        for (const auto& connection_pair : discovered_gsd["global_eth_connections"]) {
            if (connection_pair.size() != 2) {
                throw std::runtime_error("Each connection should have exactly 2 endpoints");
            }

            auto first_conn = connection_pair[0];
            auto second_conn = connection_pair[1];

            std::string hostname_1 = first_conn["host_name"].as<std::string>();
            uint32_t chan_id_1 = first_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_1 = first_conn["tray_id"].as<uint32_t>();
            uint32_t asic_location_1 = first_conn["asic_location"].as<uint32_t>();

            std::string hostname_2 = second_conn["host_name"].as<std::string>();
            uint32_t chan_id_2 = second_conn["chan_id"].as<uint32_t>();
            uint32_t tray_id_2 = second_conn["tray_id"].as<uint32_t>();
            uint32_t asic_location_2 = second_conn["asic_location"].as<uint32_t>();

            PhysicalChannelEndpoint conn_1{
                hostname_1, TrayId(tray_id_1), AsicChannel{asic_location_1, ChanId(chan_id_1)}};
            PhysicalChannelEndpoint conn_2{
                hostname_2, TrayId(tray_id_2), AsicChannel{asic_location_2, ChanId(chan_id_2)}};

            // Sort to ensure consistent ordering
            PhysicalChannelConnection connection_pair_sorted;
            if (conn_1 < conn_2) {
                connection_pair_sorted = std::make_pair(conn_1, conn_2);
            } else {
                connection_pair_sorted = std::make_pair(conn_2, conn_1);
            }

            // Check for duplicates before inserting
            if (discovered_connections.find(connection_pair_sorted) != discovered_connections.end()) {
                duplicate_discovered_connections.insert(connection_pair_sorted);
            } else {
                discovered_connections.insert(connection_pair_sorted);
            }
        }
    }

    // Report any duplicates found in discovered GSD connections
    if (!duplicate_discovered_connections.empty()) {
        std::string error_msg = "Duplicate connections found in discovered GSD:\n";
        for (const auto& dup : duplicate_discovered_connections) {
            std::ostringstream oss;
            oss << "  - " << dup.first << " <-> " << dup.second;
            error_msg += oss.str() + "\n";
        }
        throw std::runtime_error(error_msg);
    }

    // Lambda to extract port information from channel connections (shared by both validation modes)
    auto extract_port_info = [&](const std::set<PhysicalChannelConnection>& connections) {
        std::set<std::pair<PhysicalPortEndpoint, PhysicalPortEndpoint>> port_info;

        for (const auto& conn : connections) {
            // Get board types from FSD for both connections independently
            tt::umd::BoardType board_type_a = tt::umd::BoardType::UNKNOWN;
            tt::umd::BoardType board_type_b = tt::umd::BoardType::UNKNOWN;

            // Find host_id for each connection by matching hostname
            uint32_t host_id_a = 0;
            uint32_t host_id_b = 0;
            for (uint32_t i = 0; i < hosts.size(); ++i) {
                if (hosts[i].hostname() == conn.first.hostname) {
                    host_id_a = i;
                }
                if (hosts[i].hostname() == conn.second.hostname) {
                    host_id_b = i;
                }
            }

            // Look up board types for each connection
            for (const auto& board_location : generated_fsd.board_types().board_locations()) {
                if (board_location.host_id() == host_id_a && board_location.tray_id() == *conn.first.tray_id) {
                    board_type_a = get_board_type_from_string(board_location.board_type());
                }
                if (board_location.host_id() == host_id_b && board_location.tray_id() == *conn.second.tray_id) {
                    board_type_b = get_board_type_from_string(board_location.board_type());
                }
            }

            Board board_a = create_board(board_type_a);
            Board board_b = create_board(board_type_b);
            auto port_a = board_a.get_port_for_asic_channel(conn.first.asic_channel);
            auto port_b = board_b.get_port_for_asic_channel(conn.second.asic_channel);

            PhysicalPortEndpoint port_a_conn;
            PhysicalPortEndpoint port_b_conn;

            // Add deployment info for first connection if available
            for (const auto& host : hosts) {
                if (host.hostname() == conn.first.hostname) {
                    port_a_conn = PhysicalPortEndpoint{
                        conn.first.hostname,
                        host.aisle(),
                        host.rack(),
                        host.shelf_u(),
                        conn.first.tray_id,
                        port_a.port_type,
                        port_a.port_id};
                    break;
                }
            }

            // Add deployment info for second connection if available
            for (const auto& host : hosts) {
                if (host.hostname() == conn.second.hostname) {
                    port_b_conn = PhysicalPortEndpoint{
                        conn.second.hostname,
                        host.aisle(),
                        host.rack(),
                        host.shelf_u(),
                        conn.second.tray_id,
                        port_b.port_type,
                        port_b.port_id};
                    break;
                }
            }

            // Always insert in sorted order
            if (port_a_conn < port_b_conn) {
                port_info.insert(std::make_pair(port_a_conn, port_b_conn));
            } else {
                port_info.insert(std::make_pair(port_b_conn, port_a_conn));
            }
        }

        return port_info;
    };

    // Find connections that are mismatched between FSD and GSD
    std::set<PhysicalChannelConnection> missing_in_gsd;
    std::set<PhysicalChannelConnection> extra_in_gsd;

    // Always find connections in GSD but not in FSD (both validation modes check this)
    for (const auto& conn : discovered_connections) {
        if (generated_connections.find(conn) == generated_connections.end()) {
            extra_in_gsd.insert(conn);
        }
    }

    // Only in strict validation: also find connections in FSD but not in GSD
    if (strict_validation) {
        for (const auto& conn : generated_connections) {
            if (discovered_connections.find(conn) == discovered_connections.end()) {
                missing_in_gsd.insert(conn);
            }
        }
    }

    // Report missing connections (in FSD but not in GSD) - only in strict mode
    if (!missing_in_gsd.empty()) {
        auto missing_port_info = extract_port_info(missing_in_gsd);
        std::ostringstream oss;
        oss << "Channel Connections found in FSD but missing in GSD (" << std::to_string(missing_in_gsd.size())
            << " connections):\n";
        for (const auto& conn : missing_in_gsd) {
            oss << "  - " << conn.first << " <-> " << conn.second << "\n";
        }
        oss << "\n";

        oss << "Port Connections found in FSD but missing in GSD ("
            << std::to_string(missing_port_info.size()) + " connections):\n";
        for (const auto& conn : missing_port_info) {
            oss << "  - " << conn.first << " <-> " << conn.second << "\n";
        }
        std::cout << oss.str() << std::endl;
    }

    // Report extra connections (in GSD but not in FSD) - both modes check this
    if (!extra_in_gsd.empty()) {
        auto extra_port_info = extract_port_info(extra_in_gsd);

        std::ostringstream oss;
        oss << "Channel Connections found in GSD but missing in FSD (" << std::to_string(extra_in_gsd.size())
            << " connections):\n";
        for (const auto& conn : extra_in_gsd) {
            oss << "  - " << conn.first << " <-> " << conn.second << "\n";
        }
        oss << "\n";

        oss << "Port Connections found in GSD but missing in FSD ("
            << std::to_string(extra_port_info.size()) + " connections):\n";
        for (const auto& conn : extra_port_info) {
            oss << "  - " << conn.first << " <-> " << conn.second << "\n";
        }
        std::cout << oss.str() << std::endl;
    }

    // Handle validation results
    if (!missing_in_gsd.empty() || !extra_in_gsd.empty()) {
        std::string mode_text = strict_validation ? "" : " in non-strict validation";
        throw std::runtime_error("Connection mismatch detected" + mode_text + ". Check console output for details.");
    } else {
        // Success message differs based on validation mode
        if (strict_validation) {
            std::cout << "All connections match between FSD and GSD (" << generated_connections.size()
                      << " connections)" << std::endl;
        } else {
            std::cout << "All GSD connections found in FSD (" << discovered_connections.size()
                      << " connections checked)" << std::endl;
        }
    }
}

}  // namespace tt::scaleout_tools
