// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include <enchantum/enchantum.hpp>
#include <fmt/base.h>
#include <google/protobuf/text_format.h>
#include <yaml-cpp/yaml.h>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/arch.hpp>
#include <board/board.hpp>
#include <cabling_generator/cabling_generator.hpp>

// Add protobuf includes
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

bool check_min_connection_count_satisfied(
    const std::set<PhysicalChannelConnection>& discovered_connections,
    const std::set<std::pair<PhysicalChannelEndpoint, PhysicalChannelEndpoint>>& generated_connections,
    uint32_t min_connections,
    std::vector<std::pair<std::pair<AsicId, AsicId>, uint32_t>>& insufficient_connections) {
    // Helper to extract AsicId from PhysicalChannelEndpoint
    auto get_asic_id = [](const PhysicalChannelEndpoint& endpoint) -> AsicId {
        return std::make_tuple(endpoint.hostname, *endpoint.tray_id, endpoint.asic_channel.asic_location);
    };

    // Helper to create ordered pair of AsicIds (for consistent map keys)
    auto make_asic_pair = [](const AsicId& a, const AsicId& b) {
        return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
    };

    // Build map of expected ASIC pairs from generated_connections (FSD)
    // Initialize all expected pairs with count = 0
    std::map<std::pair<AsicId, AsicId>, uint32_t> asic_pair_connection_counts;

    for (const auto& conn : generated_connections) {
        auto asic_a = get_asic_id(conn.first);
        auto asic_b = get_asic_id(conn.second);
        auto asic_pair = make_asic_pair(asic_a, asic_b);
        // Initialize to 0 if not present (we count from discovered connections)
        if (!asic_pair_connection_counts.contains(asic_pair)) {
            asic_pair_connection_counts[asic_pair] = 0;
        }
    }

    // Count actual discovered connections for each ASIC pair
    for (const auto& conn : discovered_connections) {
        auto asic_a = get_asic_id(conn.first);
        auto asic_b = get_asic_id(conn.second);
        auto asic_pair = make_asic_pair(asic_a, asic_b);
        // Only count if this pair is expected (exists in FSD)
        if (asic_pair_connection_counts.contains(asic_pair)) {
            asic_pair_connection_counts[asic_pair]++;
        }
    }

    // Check if all ASIC pairs have at least min_connections
    bool all_satisfied = true;
    for (const auto& [asic_pair, count] : asic_pair_connection_counts) {
        if (count < min_connections) {
            all_satisfied = false;
            insufficient_connections.push_back({asic_pair, count});
        }
    }

    return all_satisfied;
}

std::set<PhysicalChannelConnection> validate_fsd_against_gsd_impl(
    const tt::scaleout_tools::fsd::proto::FactorySystemDescriptor& generated_fsd,
    const YAML::Node& discovered_gsd,
    bool strict_validation,
    bool assert_on_connection_mismatch,
    bool log_output,
    std::optional<uint32_t> min_connections = std::nullopt) {

    const auto& hosts = generated_fsd.hosts();

    // Compare the FSD with the discovered GSD
    // First, compare hostnames from the hosts field
    if (generated_fsd.hosts().empty()) {
        throw std::runtime_error("FSD missing hosts");
    }

    // Handle the new GSD structure with compute_node_specs
    if (!discovered_gsd["compute_node_specs"] || discovered_gsd["compute_node_specs"].size() == 0) {
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

    for (const auto& hostname : discovered_hostnames) {
        if (not generated_hostnames.contains(hostname)) {
            throw std::runtime_error("Hostname not found in FSD: " + hostname);
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
        for (const auto& [fsd_key, fsd_board_type] : fsd_board_types) {
            const auto& [hostname, tray_id] = fsd_key;
            if (discovered_hostnames.contains(hostname)) {
                throw std::runtime_error(
                    "Board type not found in GSD for discovered host " + hostname + ", tray " +
                    std::to_string(tray_id));
            }
        }
    }

    std::map<std::pair<HostId, TrayId>, BoardType> host_tray_to_board_type;
    for (const auto& board_location : generated_fsd.board_types().board_locations()) {
        HostId host_id{board_location.host_id()};
        TrayId tray_id{board_location.tray_id()};
        BoardType board_type_enum = get_board_type_from_string(board_location.board_type());
        host_tray_to_board_type[std::make_pair(host_id, tray_id)] = board_type_enum;
    }

    // Compare chip connections
    if (!generated_fsd.has_eth_connections()) {
        throw std::runtime_error("FSD missing eth_connections");
    }

    // Determine which connection types exist in the discovered GSD
    bool has_local_eth_connections = discovered_gsd["local_eth_connections"] &&
                                     !discovered_gsd["local_eth_connections"].IsNull() &&
                                     discovered_gsd["local_eth_connections"].size() > 0;
    bool has_global_eth_connections = discovered_gsd["global_eth_connections"] &&
                                      !discovered_gsd["global_eth_connections"].IsNull() &&
                                      discovered_gsd["global_eth_connections"].size() > 0;

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
        if (generated_connections.contains(connection_pair_sorted)) {
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
            if (discovered_connections.contains(connection_pair_sorted)) {
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
            if (discovered_connections.contains(connection_pair_sorted)) {
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

            // Look up board types using the prebuilt map
            auto key_a = std::make_pair(HostId(host_id_a), conn.first.tray_id);
            auto key_b = std::make_pair(HostId(host_id_b), conn.second.tray_id);
            BoardType board_type_a = host_tray_to_board_type.at(key_a);
            BoardType board_type_b = host_tray_to_board_type.at(key_b);

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

    // Always find connections in FSD but not in GSD (both validation modes check this)
    for (const auto& conn : generated_connections) {
        if (discovered_hostnames.contains(conn.first.hostname) && discovered_hostnames.contains(conn.second.hostname)) {
            if (not discovered_connections.contains(conn)) {
                missing_in_gsd.insert(conn);
            }
        }
    }

    // Only in strict validation: also find connections in GSD but not in FSD
    if (strict_validation) {
        for (const auto& conn : discovered_connections) {
            if (!generated_connections.contains(conn)) {
                extra_in_gsd.insert(conn);
            }
        }
    }

    // Report missing connections (in FSD but not in GSD) - only in strict mode
    if (!missing_in_gsd.empty()) {
        auto missing_port_info = extract_port_info(missing_in_gsd);
        std::ostringstream oss;
        oss << "Physical Discovery found " << missing_in_gsd.size()
            << " missing channel connections:\n";
        for (const auto& conn : missing_in_gsd) {
            oss << "  - " << conn.first << " <-> " << conn.second << "\n";
        }
        oss << "\n";

        oss << "Physical Discovery found " << missing_port_info.size()
            << " missing port/cable connections:\n";
        for (const auto& conn : missing_port_info) {
            oss << "  - " << conn.first << " <-> " << conn.second << "\n";
        }
        if (log_output) {
            std::cout << oss.str() << std::endl;
        }
    }

    // Report extra connections (in GSD but not in FSD) - both modes check this
    if (!extra_in_gsd.empty()) {
        auto extra_port_info = extract_port_info(extra_in_gsd);

        std::ostringstream oss;
        oss << "Physical Discovery found " << extra_in_gsd.size()
            << " extra channel connections:\n";
        for (const auto& conn : extra_in_gsd) {
            oss << "  - " << conn.first << " <-> " << conn.second << "\n";
        }
        oss << "\n";

        oss << "Physical Discovery found " << extra_port_info.size()
            << " extra port/cable connections:\n";
        for (const auto& conn : extra_port_info) {
            oss << "  - " << conn.first << " <-> " << conn.second << "\n";
        }
        if (log_output) {
            std::cout << oss.str() << std::endl;
        }
    }

    // Handle validation results - check relaxed mode at per-ASIC-pair granularity
    bool relaxed_mode_satisfied = false;
    std::vector<std::pair<std::pair<AsicId, AsicId>, uint32_t>> insufficient_connections;

    if (min_connections.has_value()) {
        relaxed_mode_satisfied = check_min_connection_count_satisfied(
            discovered_connections, generated_connections, min_connections.value(), insufficient_connections);
    }

    if (!missing_in_gsd.empty() || !extra_in_gsd.empty()) {
        std::string mode_text = strict_validation ? "" : " in non-strict validation";

        // In relaxed mode, skip throwing errors if all ASIC pairs have enough connections
        if (relaxed_mode_satisfied) {
            if (log_output) {
                std::cout << "Relaxed validation mode: All ASIC pairs have at least " << min_connections.value()
                          << " connections." << std::endl;
                if (!missing_in_gsd.empty()) {
                    std::cout << "Note: " << missing_in_gsd.size()
                              << " missing channel connections detected but ignored due to relaxed mode." << std::endl;
                }
                if (!extra_in_gsd.empty()) {
                    std::cout << "Note: " << extra_in_gsd.size()
                              << " extra channel connections detected but ignored due to relaxed mode." << std::endl;
                }
            }
            // Return empty set since we're treating this as success in relaxed mode
            return {};
        }

        // If min_connections was specified but not satisfied, report which ASIC pairs are insufficient
        if (min_connections.has_value() && !insufficient_connections.empty() && log_output) {
            std::cout << "Relaxed validation mode FAILED: The following ASIC pairs have fewer than "
                      << min_connections.value() << " connections:" << std::endl;
            for (const auto& [asic_pair, count] : insufficient_connections) {
                const auto& [asic_a, asic_b] = asic_pair;
                std::cout << "  - (" << std::get<0>(asic_a) << ", tray " << std::get<1>(asic_a) << ", asic "
                          << std::get<2>(asic_a) << ") <-> (" << std::get<0>(asic_b) << ", tray " << std::get<1>(asic_b)
                          << ", asic " << std::get<2>(asic_b) << "): " << count << " connections" << std::endl;
            }
        }

        if (assert_on_connection_mismatch) {
            throw std::runtime_error(
                "Connection mismatch detected" + mode_text + ". Check console output for details.");
        }
    } else {
        // Success message differs based on validation mode
        if (log_output) {
            if (strict_validation) {
                std::cout << "All connections match between FSD and GSD (" << generated_connections.size()
                          << " connections)" << std::endl;
            } else {
                std::cout << "All GSD connections found in FSD (" << discovered_connections.size()
                          << " connections checked)" << std::endl;
            }
        }
    }
    return missing_in_gsd;
}

std::set<PhysicalChannelConnection> validate_fsd_against_gsd(
    const std::string& fsd_filename,
    const std::string& gsd_filename,
    bool strict_validation,
    bool assert_on_connection_mismatch,
    bool log_output,
    std::optional<uint32_t> min_connections) {
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

    // Read the discovered GSD (Global System Descriptor) - still using YAML
    YAML::Node discovered_gsd = YAML::LoadFile(gsd_filename);

    // Call a shared validation function that does the actual work
    return validate_fsd_against_gsd_impl(generated_fsd, discovered_gsd, strict_validation, assert_on_connection_mismatch, log_output, min_connections);
}

std::set<PhysicalChannelConnection> validate_fsd_against_gsd(
    const fsd::proto::FactorySystemDescriptor& fsd_proto,
    const YAML::Node& gsd_yaml_node,
    bool strict_validation,
    bool assert_on_connection_mismatch,
    bool log_output,
    std::optional<uint32_t> min_connections) {
    return validate_fsd_against_gsd_impl(
        fsd_proto, gsd_yaml_node, strict_validation, assert_on_connection_mismatch, log_output, min_connections);
}

std::set<PhysicalChannelConnection> validate_cabling_descriptor_against_gsd(
    const std::string& cabling_descriptor_path,
    const std::vector<std::string>& hostnames,
    const YAML::Node& gsd_yaml_node,
    bool strict_validation,
    bool assert_on_connection_mismatch,
    bool log_output) {
    // Generate FSD from the cabling descriptor using CablingGenerator
    CablingGenerator cabling_generator(cabling_descriptor_path, hostnames);

    // Generate the FSD protobuf object in memory
    auto generated_fsd = cabling_generator.generate_factory_system_descriptor();

    // Call a shared validation function that does the actual work
    return validate_fsd_against_gsd_impl(generated_fsd, gsd_yaml_node, strict_validation, assert_on_connection_mismatch, log_output);
}

std::string generate_cluster_descriptor_from_fsd(
    const std::string& fsd_filename, const std::string& output_dir, const std::string& base_filename) {
    // Read the FSD using protobuf
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd;
    std::ifstream fsd_file(fsd_filename);
    if (!fsd_file.is_open()) {
        throw std::runtime_error("Failed to open FSD file: " + fsd_filename);
    }

    std::string fsd_content((std::istreambuf_iterator<char>(fsd_file)), std::istreambuf_iterator<char>());
    fsd_file.close();

    if (!google::protobuf::TextFormat::ParseFromString(fsd_content, &fsd)) {
        throw std::runtime_error("Failed to parse FSD protobuf from file: " + fsd_filename);
    }

    if (fsd.board_types().board_locations_size() == 0) {
        throw std::runtime_error("FSD must have at least one board location");
    }

    // Current generation only supports a unified board type per FSD
    const auto& first_board_location = fsd.board_types().board_locations(0);
    BoardType board_type = get_board_type_from_string(first_board_location.board_type());
    for (const auto& board_location : fsd.board_types().board_locations()) {
        BoardType curr_board_type = get_board_type_from_string(board_location.board_type());
        if (curr_board_type != board_type) {
            throw std::runtime_error(
                "All board types must be the same. Found " + board_type_to_string(curr_board_type) + " and " +
                board_type_to_string(board_type));
        }
    }

    // Assert for UBB board types (due to hardcoded bus ID lookup)
    // TODO: We should commonize the bus ID lookups with tt_metal/fabric/physical_system_descriptor.cpp
    if (board_type != BoardType::UBB_BLACKHOLE && board_type != BoardType::UBB_WORMHOLE) {
        throw std::runtime_error("Board type " + board_type_to_string(board_type) + " is not supported");
    }
    Board board = create_board(board_type);
    tt::ARCH arch = board.get_arch();

    std::string arch_str = tt::arch_to_str(arch);
    if (arch_str == "Invalid") {
        throw std::runtime_error("Unsupported architecture");
    }

    // Get the bus ID map for the architecture
    const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
        {tt::ARCH::WORMHOLE_B0, {0xC0, 0x80, 0x00, 0x40}},
        {tt::ARCH::BLACKHOLE, {0x00, 0x40, 0xC0, 0x80}},
    };

    if (!ubb_bus_ids.contains(arch)) {
        throw std::runtime_error("No bus ID mapping for architecture: " + arch_str);
    }
    const auto& tray_bus_ids = ubb_bus_ids.at(arch);

    std::map<std::pair<HostId, TrayId>, BoardType> host_tray_to_board_type;
    for (const auto& board_location : fsd.board_types().board_locations()) {
        HostId host_id{board_location.host_id()};
        TrayId tray_id{board_location.tray_id()};
        BoardType board_type_enum = get_board_type_from_string(board_location.board_type());
        host_tray_to_board_type[std::make_pair(host_id, tray_id)] = board_type_enum;
    }

    // Assign a unique_chip_id to each ASIC
    const auto& asic_locations_on_board = board.get_asic_locations();
    std::map<std::tuple<HostId, TrayId, uint32_t>, uint64_t> asic_to_unique_chip_id;
    std::unordered_map<uint64_t, std::tuple<HostId, TrayId, uint32_t>> unique_chip_id_to_asic;
    uint64_t next_unique_chip_id = 0;
    for (const auto& board_location : fsd.board_types().board_locations()) {
        HostId host_id{board_location.host_id()};
        TrayId tray_id{board_location.tray_id()};
        for (uint32_t asic_location : asic_locations_on_board) {
            auto key = std::make_tuple(host_id, tray_id, asic_location);
            if (!asic_to_unique_chip_id.contains(key)) {
                asic_to_unique_chip_id[key] = next_unique_chip_id;
                unique_chip_id_to_asic[next_unique_chip_id] = key;
                next_unique_chip_id++;
            }
        }
    }

    // Generate bus IDs and ASIC location mappings for all chips
    std::unordered_map<uint64_t, uint16_t> unique_chip_to_bus_id;
    std::unordered_map<uint64_t, uint32_t> unique_chip_to_asic_location;

    for (uint64_t unique_chip_id = 0; unique_chip_id < next_unique_chip_id; ++unique_chip_id) {
        auto [host_id, tray_id, asic_location] = unique_chip_id_to_asic[unique_chip_id];
        if (tray_id.get() < 1 || tray_id.get() > tray_bus_ids.size()) {
            throw std::runtime_error(
                "Invalid tray_id: " + std::to_string(tray_id.get()) + " (must be between 1 and " +
                std::to_string(tray_bus_ids.size()) + ")");
        }
        uint16_t tray_bus_id = tray_bus_ids[tray_id.get() - 1];
        uint16_t bus_id = tray_bus_id | (asic_location & 0x0F);
        unique_chip_to_bus_id[unique_chip_id] = bus_id;
        unique_chip_to_asic_location[unique_chip_id] = asic_location;
    }

    // Build ethernet connections
    std::vector<std::tuple<uint64_t, ChanId, uint64_t, ChanId>> ethernet_connections;
    for (const auto& connection : fsd.eth_connections().connection()) {
        const auto& endpoint_a = connection.endpoint_a();
        const auto& endpoint_b = connection.endpoint_b();
        auto key_a =
            std::make_tuple(HostId{endpoint_a.host_id()}, TrayId{endpoint_a.tray_id()}, endpoint_a.asic_location());
        auto key_b =
            std::make_tuple(HostId{endpoint_b.host_id()}, TrayId{endpoint_b.tray_id()}, endpoint_b.asic_location());
        uint64_t unique_chip_a = asic_to_unique_chip_id[key_a];
        uint64_t unique_chip_b = asic_to_unique_chip_id[key_b];
        ChanId chan_a{endpoint_a.chan_id()};
        ChanId chan_b{endpoint_b.chan_id()};
        ethernet_connections.push_back({unique_chip_a, chan_a, unique_chip_b, chan_b});
    }
    // Generate mapping YAML (only for multi-host)
    uint32_t num_hosts = fsd.hosts_size();
    bool is_single_host = (num_hosts == 1);
    YAML::Node mapping;
    if (!is_single_host) {
        mapping["rank_to_cluster_mock_cluster_desc"] = YAML::Node(YAML::NodeType::Map);
    }

    std::filesystem::create_directories(output_dir);
    std::string final_output_file;
    std::filesystem::path output_path = std::filesystem::path(output_dir);

    // Generate cluster descriptor for each host
    uint64_t unique_board_id = 0;
    for (uint32_t host_idx = 0; host_idx < num_hosts; ++host_idx) {
        // Map unique chip IDs to local chip IDs for this host
        HostId host_id{host_idx};
        std::unordered_map<uint64_t, ChipId> unique_to_local_chip_id;
        std::unordered_map<ChipId, uint64_t> local_to_unique_chip_id;
        ChipId next_local_chip_id = 0;
        for (const auto& [key, unique_chip_id] : asic_to_unique_chip_id) {
            auto [h_id, t_id, asic_loc] = key;
            if (h_id == host_id) {
                unique_to_local_chip_id[unique_chip_id] = next_local_chip_id;
                local_to_unique_chip_id[next_local_chip_id] = unique_chip_id;
                next_local_chip_id++;
            }
        }
        ChipId total_local_chips = next_local_chip_id;

        // Build cluster descriptor for this host
        YAML::Node cluster_desc;
        cluster_desc["arch"] = YAML::Node(YAML::NodeType::Map);
        for (ChipId local_chip_id = 0; local_chip_id < total_local_chips; ++local_chip_id) {
            cluster_desc["arch"][local_chip_id] = arch_str;
        }
        cluster_desc["chips"] = YAML::Node(YAML::NodeType::Map);
        cluster_desc["chip_unique_ids"] = YAML::Node(YAML::NodeType::Map);
        for (ChipId local_chip_id = 0; local_chip_id < total_local_chips; ++local_chip_id) {
            cluster_desc["chip_unique_ids"][local_chip_id] = local_to_unique_chip_id[local_chip_id];
        }

        cluster_desc["ethernet_connections"] = YAML::Node(YAML::NodeType::Sequence);
        cluster_desc["ethernet_connections_to_remote_devices"] = YAML::Node(YAML::NodeType::Sequence);
        for (const auto& [unique_chip_a, chan_a, unique_chip_b, chan_b] : ethernet_connections) {
            auto it_a = unique_to_local_chip_id.find(unique_chip_a);
            auto it_b = unique_to_local_chip_id.find(unique_chip_b);

            bool chip_a_local = (it_a != unique_to_local_chip_id.end());
            bool chip_b_local = (it_b != unique_to_local_chip_id.end());
            // If both chips are on this host, add a local connection, otherwise add a remote connection
            if (chip_a_local && chip_b_local) {
                // Both chips on this host - local connection
                YAML::Node endpoint_a;
                endpoint_a["chip"] = it_a->second;
                endpoint_a["chan"] = chan_a.get();
                YAML::Node endpoint_b;
                endpoint_b["chip"] = it_b->second;
                endpoint_b["chan"] = chan_b.get();
                YAML::Node connection_pair;
                connection_pair.push_back(endpoint_a);
                connection_pair.push_back(endpoint_b);
                cluster_desc["ethernet_connections"].push_back(connection_pair);
            } else if (chip_a_local) {
                YAML::Node endpoint_a;
                endpoint_a["chip"] = it_a->second;
                endpoint_a["chan"] = chan_a.get();
                YAML::Node endpoint_b;
                endpoint_b["remote_chip_id"] = unique_chip_b;
                endpoint_b["chan"] = chan_b.get();
                YAML::Node connection_pair;
                connection_pair.push_back(endpoint_a);
                connection_pair.push_back(endpoint_b);
                cluster_desc["ethernet_connections_to_remote_devices"].push_back(connection_pair);
            } else if (chip_b_local) {
                YAML::Node endpoint_a;
                endpoint_a["chip"] = it_b->second;
                endpoint_a["chan"] = chan_b.get();
                YAML::Node endpoint_b;
                endpoint_b["remote_chip_id"] = unique_chip_a;
                endpoint_b["chan"] = chan_a.get();
                YAML::Node connection_pair;
                connection_pair.push_back(endpoint_a);
                connection_pair.push_back(endpoint_b);
                cluster_desc["ethernet_connections_to_remote_devices"].push_back(connection_pair);
            }
        }

        cluster_desc["chips_with_mmio"] = YAML::Node(YAML::NodeType::Sequence);
        for (ChipId local_chip_id = 0; local_chip_id < total_local_chips; ++local_chip_id) {
            YAML::Node mmio_entry(YAML::NodeType::Map);
            mmio_entry[local_chip_id] = local_chip_id;
            cluster_desc["chips_with_mmio"].push_back(mmio_entry);
        }
        cluster_desc["io_device_type"] = "PCIe";

        cluster_desc["harvesting"] = YAML::Node(YAML::NodeType::Map);
        for (ChipId local_chip_id = 0; local_chip_id < total_local_chips; ++local_chip_id) {
            uint64_t unique_chip_id = local_to_unique_chip_id[local_chip_id];
            auto [h_id, t_id, asic_loc] = unique_chip_id_to_asic[unique_chip_id];

            // Look up board type using the prebuilt map
            BoardType board_type_enum = host_tray_to_board_type.at(std::make_pair(h_id, t_id));

            // Calculate harvest masks
            uint32_t tensix_harvested_units = expected_tensix_harvested_units_map.at(board_type_enum);
            uint32_t dram_harvested_units = expected_dram_harvested_units_map.at(board_type_enum);
            uint32_t eth_harvested_units = expected_eth_harvested_units_map.at(board_type_enum);

            uint32_t harvest_mask = tensix_harvested_units > 0 ? (1u << tensix_harvested_units) - 1 : 0;
            uint32_t dram_harvesting_mask = 0;
            if (board_type_enum == BoardType::P100) {
                dram_harvesting_mask = 8;
            } else {
                dram_harvesting_mask = dram_harvested_units > 0 ? (1u << dram_harvested_units) - 1 : 0;
            }
            uint32_t eth_harvesting_mask = 0;
            if (board_type_enum == BoardType::UBB_BLACKHOLE || board_type_enum == BoardType::P150 ||
                board_type_enum == BoardType::P300) {
                eth_harvesting_mask = 288;
            } else {
                eth_harvesting_mask = eth_harvested_units > 0 ? (1u << eth_harvested_units) - 1 : 0;
            }
            uint32_t pcie_harvesting_mask = 0;
            if (board_type_enum == BoardType::UBB_BLACKHOLE) {
                pcie_harvesting_mask = 1;
            } else if (board_type_enum == BoardType::P150 || board_type_enum == BoardType::P100) {
                pcie_harvesting_mask = 2;
            } else if (board_type_enum == BoardType::P300) {
                if (asic_loc == 0) {
                    pcie_harvesting_mask = 2;
                } else {
                    pcie_harvesting_mask = 1;
                }
            }
            uint32_t l2cpu_harvesting_mask = 0;
            cluster_desc["harvesting"][local_chip_id]["noc_translation"] = true;
            cluster_desc["harvesting"][local_chip_id]["harvest_mask"] = harvest_mask;
            cluster_desc["harvesting"][local_chip_id]["dram_harvesting_mask"] = dram_harvesting_mask;
            cluster_desc["harvesting"][local_chip_id]["eth_harvesting_mask"] = eth_harvesting_mask;
            cluster_desc["harvesting"][local_chip_id]["pcie_harvesting_mask"] = pcie_harvesting_mask;
            cluster_desc["harvesting"][local_chip_id]["l2cpu_harvesting_mask"] = l2cpu_harvesting_mask;
        }

        cluster_desc["chip_to_boardtype"] = YAML::Node(YAML::NodeType::Map);
        for (ChipId local_chip_id = 0; local_chip_id < total_local_chips; ++local_chip_id) {
            uint64_t unique_chip_id = local_to_unique_chip_id[local_chip_id];
            auto [h_id, t_id, asic_loc] = unique_chip_id_to_asic[unique_chip_id];

            // Look up board type using the prebuilt map
            BoardType board_type_enum = host_tray_to_board_type.at(std::make_pair(h_id, t_id));
            cluster_desc["chip_to_boardtype"][local_chip_id] = board_type_to_string(board_type_enum);
        }

        cluster_desc["chip_to_bus_id"] = YAML::Node(YAML::NodeType::Map);
        for (ChipId local_chip_id = 0; local_chip_id < total_local_chips; ++local_chip_id) {
            uint64_t unique_chip_id = local_to_unique_chip_id[local_chip_id];
            std::stringstream bus_id_hex;
            bus_id_hex << "0x" << std::hex << std::setfill('0') << std::setw(4)
                       << unique_chip_to_bus_id[unique_chip_id];
            cluster_desc["chip_to_bus_id"][local_chip_id] = bus_id_hex.str();
        }

        cluster_desc["boards"] = YAML::Node(YAML::NodeType::Sequence);
        std::map<uint64_t, std::vector<ChipId>> board_to_local_chips;
        for (const auto& board_location : fsd.board_types().board_locations()) {
            if (HostId{board_location.host_id()} != host_id) {
                continue;
            }
            TrayId tray_id{board_location.tray_id()};
            BoardType board_type_enum = get_board_type_from_string(board_location.board_type());

            // Find UPI for this board type
            uint64_t upi = 0;
            for (const auto& [map_upi, map_board_type] : board_upi_map) {
                if (map_board_type == board_type_enum) {
                    upi = map_upi;
                    break;
                }
            }
            if (upi == 0) {
                throw std::runtime_error("No UPI found for board type: " + board_location.board_type());
            }

            uint64_t encoded_board_id = (upi << 36) | unique_board_id;
            unique_board_id++;

            std::vector<ChipId> chips_on_board;
            for (ChipId local_chip_id = 0; local_chip_id < total_local_chips; ++local_chip_id) {
                uint64_t unique_chip_id = local_to_unique_chip_id[local_chip_id];
                auto [h_id, t_id, asic_loc] = unique_chip_id_to_asic[unique_chip_id];
                if (h_id == host_id && t_id == tray_id) {
                    chips_on_board.push_back(local_chip_id);
                }
            }
            std::sort(chips_on_board.begin(), chips_on_board.end());
            board_to_local_chips[encoded_board_id] = chips_on_board;
        }

        for (const auto& [encoded_board_id, chips_on_board] : board_to_local_chips) {
            std::string board_type_str = board_type_to_string(get_board_type_from_board_id(encoded_board_id));
            YAML::Node board_entry;
            YAML::Node board_id_node;
            board_id_node["board_id"] = encoded_board_id;
            YAML::Node board_type_node;
            board_type_node["board_type"] = board_type_str;
            YAML::Node chips_node;
            chips_node["chips"] = YAML::Node(YAML::NodeType::Sequence);
            for (ChipId chip_id : chips_on_board) {
                chips_node["chips"].push_back(chip_id);
            }
            board_entry.push_back(board_id_node);
            board_entry.push_back(board_type_node);
            board_entry.push_back(chips_node);
            cluster_desc["boards"].push_back(board_entry);
        }

        cluster_desc["asic_locations"] = YAML::Node(YAML::NodeType::Map);
        for (ChipId local_chip_id = 0; local_chip_id < total_local_chips; ++local_chip_id) {
            uint64_t unique_chip_id = local_to_unique_chip_id[local_chip_id];
            cluster_desc["asic_locations"][local_chip_id] = unique_chip_to_asic_location[unique_chip_id];
        }

        std::string output_file;
        if (is_single_host) {
            output_file = (output_path / fmt::format("{}.yaml", base_filename)).string();
        } else {
            output_file = (output_path / fmt::format("{}_rank_{}.yaml", base_filename, host_idx)).string();
            mapping["rank_to_cluster_mock_cluster_desc"][host_idx] = output_file;
        }
        std::ofstream out_file(output_file);
        if (!out_file.is_open()) {
            throw std::runtime_error("Failed to open output file: " + output_file);
        }
        YAML::Emitter emitter;
        emitter << cluster_desc;
        out_file << emitter.c_str();
        out_file.close();
        if (is_single_host) {
            final_output_file = output_file;
        }
    }

    // Write mapping file for multi-host
    if (!is_single_host) {
        final_output_file = (output_path / fmt::format("{}_mapping.yaml", base_filename)).string();
        std::ofstream mapping_file(final_output_file);
        if (!mapping_file.is_open()) {
            throw std::runtime_error("Failed to open mapping file: " + final_output_file);
        }
        YAML::Emitter mapping_emitter;
        mapping_emitter << mapping;
        mapping_file << mapping_emitter.c_str();
        mapping_file.close();
    }

    return final_output_file;
}

}  // namespace tt::scaleout_tools
